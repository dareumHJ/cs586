"""
Last modified date: 2024.12.16
Author: Bimanual Extension
Description: Bimanual initialization functions based on BimanGrasp paper
"""

import torch
import numpy as np
import math
import transforms3d


def matrix_to_rot6d(rotation_matrix):
    """
    Convert rotation matrix to 6D rotation representation
    Takes first two columns of rotation matrix
    
    Parameters
    ----------
    rotation_matrix: (B, 3, 3) torch.FloatTensor
        rotation matrices
        
    Returns
    -------
    rot6d: (B, 6) torch.FloatTensor
        6D rotation representation
    """
    return rotation_matrix[:, :, :2].reshape(-1, 6)


def initialize_bimanual_convex_hull(bimanual_hand_model, object_model, args, 
                                   left_target_direction=None, right_target_direction=None, 
                                   init_radius=None):
    """
    Initialize bimanual hand poses using target direction approach
    Object center is at (0,0,0), hands are positioned at specified directions and radius
    
    Parameters
    ----------
    bimanual_hand_model: BimanualHandModel
        bimanual hand model to initialize
    object_model: ObjectModel
        object model containing target objects
    args: argparse.Namespace
        configuration arguments
    left_target_direction: (3,) array-like, optional
        target direction for left hand from object center, default (0,-0.2,0)
    right_target_direction: (3,) array-like, optional  
        target direction for right hand from object center, default (0,0.2,0)
    init_radius: float, optional
        radius distance from object center, default 0.2
    """
    
    device = bimanual_hand_model.device
    n_objects = len(object_model.object_code_list) 
    batch_size_each = object_model.batch_size_each
    total_batch_size = n_objects * batch_size_each
    
    # 기본값 설정 - 왼손은 Y축 음방향, 오른손은 Y축 양방향으로 배치
    if left_target_direction is None:
        left_target_direction = [0, -0.2, 0]  # 왼손: Y축 음방향
    if right_target_direction is None:
        right_target_direction = [0, 0.2, 0]   # 오른손: Y축 양방향
    if init_radius is None:
        init_radius = 0.2  # 물체 중심에서 0.2m 거리
        
    # 텐서로 변환 및 GPU 메모리로 이동
    left_target_dir = torch.tensor(left_target_direction, device=device, dtype=torch.float32)
    right_target_dir = torch.tensor(right_target_direction, device=device, dtype=torch.float32)
    
    # 목표 방향 벡터를 정규화하고 지정된 반지름으로 스케일링
    # 이렇게 하면 방향은 유지하되 거리는 init_radius로 통일됨
    left_target_dir = left_target_dir / torch.norm(left_target_dir) * init_radius
    right_target_dir = right_target_dir / torch.norm(right_target_dir) * init_radius
    
    # 양손 포즈 초기화 (B, 62) - 왼손 31차원 + 오른손 31차원
    bimanual_poses = torch.zeros(total_batch_size, 62, device=device)
    
    # 접촉점 인덱스 무작위 초기화 (B, 8) - 왼손 4개 + 오른손 4개
    left_contact_indices = torch.randint(0, bimanual_hand_model.left_hand.n_contact_candidates, 
                                        (total_batch_size, 4), device=device)
    # 오른손 접촉점 인덱스는 왼손 접촉점 개수만큼 오프셋 추가
    right_contact_indices = torch.randint(0, bimanual_hand_model.right_hand.n_contact_candidates,
                                         (total_batch_size, 4), device=device) + bimanual_hand_model.left_hand.n_contact_candidates
    contact_indices = torch.cat([left_contact_indices, right_contact_indices], dim=1)
    
    for i in range(n_objects):
        # 물체 중심을 원점(0,0,0)으로 고정 - 새로운 메커니즘의 핵심
        object_center = torch.tensor([0.0, 0.0, 0.0], device=device)
        
        for j in range(batch_size_each):
            idx = i * batch_size_each + j
            
            # 물체 중심에서 목표 방향으로 손 위치 계산
            # 예: 왼손은 (0, -0.2, 0), 오른손은 (0, 0.2, 0)에 배치
            left_pos = object_center + left_target_dir
            right_pos = object_center + right_target_dir
            
            # 각 손의 방향(회전) 계산 - 손바닥이 물체를 향하도록
            left_pose = calculate_hand_orientation(left_pos, right_pos, object_center, device, hand_side='left')
            right_pose = calculate_hand_orientation(right_pos, left_pos, object_center, device, hand_side='right')
            
            # 관절 각도 초기화 (각 손마다 22개 관절)
            left_joints = torch.zeros(22, device=device)
            right_joints = torch.zeros(22, device=device)
            
            # 지터(jitter) 추가 - 초기 자세에 무작위 변동 추가하여 다양성 확보
            if args.jitter_strength > 0:
                joint_jitter_left = (torch.rand(22, device=device) - 0.5) * 2 * args.jitter_strength
                joint_jitter_right = (torch.rand(22, device=device) - 0.5) * 2 * args.jitter_strength
                left_joints += joint_jitter_left
                right_joints += joint_jitter_right
            
            # 관절 각도를 각 손의 물리적 한계 내로 제한
            left_joints = torch.clamp(left_joints, 
                                     bimanual_hand_model.left_hand.joints_lower,
                                     bimanual_hand_model.left_hand.joints_upper)
            right_joints = torch.clamp(right_joints,
                                      bimanual_hand_model.right_hand.joints_lower, 
                                      bimanual_hand_model.right_hand.joints_upper)
            
            # 최종 포즈 결합: [위치(3) + 회전(6) + 관절각도(22)] = 31차원
            left_full_pose = torch.cat([left_pose, left_joints])
            right_full_pose = torch.cat([right_pose, right_joints])
            
            # 양손 포즈를 하나의 벡터로 결합 (62차원)
            bimanual_poses[idx, :31] = left_full_pose    # 왼손 포즈 (0~30)
            bimanual_poses[idx, 31:] = right_full_pose   # 오른손 포즈 (31~61)
    
    # 계산된 포즈를 양손 모델에 설정
    bimanual_hand_model.set_parameters(bimanual_poses, contact_indices)


def calculate_hand_orientation(hand_pos, other_hand_pos, object_center, device, hand_side='left'):
    """
    Calculate hand orientation for 6D rotation representation
    Combines the good axis calculation logic with proper 6D transmission
    
    Parameters
    ----------
    hand_pos: (3,) torch.FloatTensor
        position of this hand
    other_hand_pos: (3,) torch.FloatTensor  
        position of the other hand
    object_center: (3,) torch.FloatTensor
        center of object (0,0,0)
    device: torch.Device
        computation device
    hand_side: str
        'left' or 'right'
        
    Returns
    -------
    pose: (9,) torch.FloatTensor
        hand pose [translation(3), rotation(6)]
    """
    
    # 1. 손바닥 방향 계산: 물체 중심을 향하도록
    palm_direction = object_center - hand_pos
    palm_direction = palm_direction / torch.norm(palm_direction)
    
    # 2. 손가락 방향을 견고하게 계산하는 방법 (개선된 로직)
    toward_other = other_hand_pos - hand_pos
    toward_other = toward_other / torch.norm(toward_other)
    
    # 다른 손 방향을 손바닥 방향에 수직인 평면에 투영
    toward_other_projected = toward_other - torch.dot(toward_other, palm_direction) * palm_direction
    
    if torch.norm(toward_other_projected) > 1e-6:
        toward_other_projected = toward_other_projected / torch.norm(toward_other_projected)
        finger_direction = -toward_other_projected
    else:
        # 대칭 배치에서 toward_other_projected가 영벡터가 되는 경우
        # 손에 따라 다른 기본 방향 설정
        if hand_side == 'right':
            finger_direction = torch.tensor([1.0, 0.0, 0.0], device=device)  # 오른손: X 양방향
        else:  # left hand
            finger_direction = torch.tensor([-1.0, 0.0, 0.0], device=device)  # 왼손: X 음방향
    
    # 3. 엄지 방향 계산: 손바닥과 손가락 방향 모두에 수직
    if hand_side == 'right':
        thumb_direction = -torch.cross(palm_direction, finger_direction)
    else:  # left hand
        thumb_direction = torch.cross(palm_direction, finger_direction)
    
    thumb_direction = thumb_direction / torch.norm(thumb_direction)
    
    # 축 정의
    x_axis = thumb_direction    # 엄지 방향
    y_axis = palm_direction     # 손바닥 방향 (물체 향함)
    z_axis = finger_direction   # 손가락 방향
    
    print(f"[{hand_side}] BEFORE matrix creation:")
    print(f"  x_axis (thumb): {x_axis}")
    print(f"  y_axis (palm): {y_axis}")  
    print(f"  z_axis (finger): {z_axis}")
    
    # Rotation matrix 생성 - 축들을 열(column)으로 배치
    stacked = torch.stack([x_axis, y_axis, z_axis], dim=0)
    print(f"[{hand_side}] stacked shape: {stacked.shape}")
    print(f"[{hand_side}] stacked:\n{stacked}")
    
    rotation_matrix_2d = stacked.T
    print(f"[{hand_side}] after transpose:\n{rotation_matrix_2d}")
    
    rotation_matrix = rotation_matrix_2d.unsqueeze(0)  # (1, 3, 3)
    print(f"[{hand_side}] final rotation_matrix shape: {rotation_matrix.shape}")
    print(f"[{hand_side}] final rotation_matrix:\n{rotation_matrix[0]}")
    
    # matrix_to_rot6d 계산 단계별 확인
    first_two_cols = rotation_matrix[:, :, :2]
    print(f"[{hand_side}] first_two_cols shape: {first_two_cols.shape}")
    print(f"[{hand_side}] first_two_cols:\n{first_two_cols[0]}")
    
    # transpose로 열 순서로 펼쳐지도록 수정
    first_two_cols_transposed = first_two_cols.transpose(-1, -2)  # (B, 2, 3)
    print(f"[{hand_side}] after transpose: shape {first_two_cols_transposed.shape}")
    print(f"[{hand_side}] after transpose:\n{first_two_cols_transposed[0]}")
    
    # matrix_to_rot6d로 올바른 6D representation 생성
    rot6d = first_two_cols_transposed.reshape(-1, 6).squeeze(0)  # (6,)
    print(f"[{hand_side}] final rot6d: {rot6d}")
    
    # 최종 결과: [위치(3) + 회전(6)] = 9차원 벡터
    return torch.cat([hand_pos, rot6d])


def initialize_bimanual_convex_hull_progressive(bimanual_hand_model, object_model, args):
    """
    Progressive initialization similar to original DexGraspNet but for bimanual setup
    Places hands on inflated convex hull and progressively moves them closer
    
    Parameters
    ----------
    bimanual_hand_model: BimanualHandModel
        bimanual hand model to initialize
    object_model: ObjectModel
        object model containing target objects  
    args: argparse.Namespace
        configuration arguments
    """
    
    device = bimanual_hand_model.device
    n_objects = len(object_model.object_code_list)
    batch_size_each = object_model.batch_size_each
    total_batch_size = n_objects * batch_size_each
    
    # Start with basic symmetric initialization
    initialize_bimanual_convex_hull(bimanual_hand_model, object_model, args)
    
    # Progressive refinement - move hands closer to object surface
    n_steps = 10
    for step in range(n_steps):
        # Calculate current distances to object
        if bimanual_hand_model.contact_points is not None:
            distances, _ = object_model.cal_distance(bimanual_hand_model.contact_points)
            
            # Move hands closer if they are too far
            target_distance = 0.01  # 1cm target distance
            movement_factor = 0.1   # Move 10% closer each step
            
            for i in range(total_batch_size):
                avg_distance = distances[i].mean().item()
                if avg_distance > target_distance:
                    # Move both hands closer to object center
                    object_center = torch.tensor([0.0, 0.0, 0.0], device=device)  # Object center is at origin
                    
                    # Adjust left hand position
                    left_pos = bimanual_hand_model.bimanual_pose[i, :3]
                    left_direction = (object_center - left_pos)
                    left_direction = left_direction / torch.norm(left_direction)
                    bimanual_hand_model.bimanual_pose[i, :3] += left_direction * movement_factor * (avg_distance - target_distance)
                    
                    # Adjust right hand position
                    right_pos = bimanual_hand_model.bimanual_pose[i, 31:34]
                    right_direction = (object_center - right_pos)
                    right_direction = right_direction / torch.norm(right_direction) 
                    bimanual_hand_model.bimanual_pose[i, 31:34] += right_direction * movement_factor * (avg_distance - target_distance)
        
        # Update hand model with new poses
        bimanual_hand_model.set_parameters(bimanual_hand_model.bimanual_pose, 
                                          bimanual_hand_model.contact_point_indices) 