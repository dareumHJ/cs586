"""
Last modified date: 2024.12.16
Author: Bimanual Extension
Description: Bimanual initialization functions based on BimanGrasp paper
"""

import torch
import numpy as np
import math
import transforms3d
import pytorch3d.structures
import pytorch3d.ops
import trimesh as tm
import torch.nn.functional


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
    Initialize bimanual hand poses using sampled positions and orientations.
    Positions are sampled in a spherical shell around the object center.
    Orientations are sampled to make the palm face the object center with added jitter.
    
    Parameters
    ----------
    bimanual_hand_model: BimanualHandModel
        bimanual hand model to initialize
    object_model: ObjectModel
        object model containing target objects
    args: argparse.Namespace
        configuration arguments (needs distance_lower, distance_upper, theta_lower, theta_upper, jitter_strength)
    # Note: left_target_direction, right_target_direction, init_radius are ignored in this implementation.
    """
    
    device = bimanual_hand_model.device
    n_objects = len(object_model.object_code_list) 
    batch_size_each = object_model.batch_size_each
    total_batch_size = n_objects * batch_size_each
    
    # Object center is fixed at (0,0,0)
    object_center = torch.zeros(3, device=device)
    
    # Initialize bimanual poses (B, 62) - left hand 31 + right hand 31
    bimanual_poses = torch.zeros(total_batch_size, 62, device=device)
    
    # Initialize contact point indices (B, 8) - 4 from each hand
    left_contact_indices = torch.randint(0, bimanual_hand_model.left_hand.n_contact_candidates, 
                                        (total_batch_size, 4), device=device)
    # Right hand contact point indices offset by left hand contact candidates count
    right_contact_indices = torch.randint(0, bimanual_hand_model.right_hand.n_contact_candidates, 
                                         (total_batch_size, 4), device=device) + bimanual_hand_model.left_hand.n_contact_candidates
    contact_indices = torch.cat([left_contact_indices, right_contact_indices], dim=1)
    
    # Sample initial pose parameters for each batch sample
    # Sample ONE random direction for the pair
    # pair_initial_dir_spherical = torch.rand(total_batch_size, 2, device=device) * torch.tensor([math.pi, 2*math.pi], device=device)
    # pair_initial_dir_cartesian = torch.stack([
    #     torch.sin(pair_initial_dir_spherical[:, 0]) * torch.cos(pair_initial_dir_spherical[:, 1]),
    #     torch.sin(pair_initial_dir_spherical[:, 0]) * torch.sin(pair_initial_dir_spherical[:, 1]),
    #     torch.cos(pair_initial_dir_spherical[:, 0])
    # ], dim=1) # (B, 3)
    # pair_initial_dir_cartesian = pair_initial_dir_cartesian / torch.norm(pair_initial_dir_cartesian, dim=1, keepdim=True) # Ensure unit vectors
    
    # Sample distances for left and right hands (can be different for each sample)
    dist_left = args.distance_lower + (args.distance_upper - args.distance_lower) * torch.rand([total_batch_size], dtype=torch.float, device=device) # (B,)
    dist_right = args.distance_lower + (args.distance_upper - args.distance_lower) * torch.rand([total_batch_size], dtype=torch.float, device=device) # (B,)
    
    # Use base left/right directions with jitter for position sampling
    base_left_dir = torch.tensor([0., -1., 0.], device=device).unsqueeze(0).repeat(total_batch_size, 1) # (B, 3)
    base_right_dir = torch.tensor([0., 1., 0.], device=device).unsqueeze(0).repeat(total_batch_size, 1)  # (B, 3)

    # Sample deviation angles around X and Z for left/right hands (controls deviation from Y axis)
    # Use args.theta_upper to limit deviation range
    dev_angles_xz_left = (torch.rand(total_batch_size, 2, device=device) - 0.5) * 2 * args.theta_upper # (B, 2)
    dev_angles_xz_right = (torch.rand(total_batch_size, 2, device=device) - 0.5) * 2 * args.theta_upper # (B, 2)

    # Sample rotation angle around the base Y axis (full 2*pi rotation)
    rot_angle_y_left = 2 * math.pi * torch.rand([total_batch_size], dtype=torch.float, device=device) # (B,)
    rot_angle_y_right = 2 * math.pi * torch.rand([total_batch_size], dtype=torch.float, device=device) # (B,)

    # Generate rotation matrices from deviation and rotation angles (batch-wise)
    # This rotation matrix should rotate the base direction [0, +/-1, 0]
    # A rotation around X by dev_angles_xz[:, 0] and around Z by dev_angles_xz[:, 1],
    # followed by a rotation around Y by rot_angle_y. Order: R_y @ R_z @ R_x ?
    # Or, simpler: rotate [0, +/-1, 0] by a matrix formed by Euler angles [dev_x, rot_y, dev_z]

    # Let's use a simpler approach: sample a random vector on a sphere with limited angle from base Y axis.
    # Convert spherical to Cartesian. Base direction is [0, 1, 0]. Spherical coordinates (r=1): x = sin(theta)cos(phi), y = cos(theta), z = sin(theta)sin(phi)
    # We want deviation from Y, so theta is angle from Y. Phi is angle around Y in XZ plane.
    # For left hand, base is [0, -1, 0]. In spherical, this is theta=pi, phi=any.
    # We want to deviate from [0, -1, 0]. Rotate by `deviation_theta_left` from -Y, and `rotation_phi_left` around -Y.

    # A simpler approach: start with base vector [0, +/-1, 0].
    # Apply a rotation matrix R to this vector, where R is formed by small random rotations around X and Z,
    # and a full random rotation around Y.

    # Generate rotation matrices from Euler angles (ZYX order: Rz @ Ry @ Rx)
    # Rx rotates around X axis, Rz rotates around Z axis. Use these for deviation.
    # Ry rotates around Y axis. Use this for rotation around the primary direction.

    # Sample deviation angles around X and Z (limited by theta_upper)
    dev_x_left = (torch.rand(total_batch_size, device=device) - 0.5) * 2 * args.theta_upper
    dev_z_left = (torch.rand(total_batch_size, device=device) - 0.5) * 2 * args.theta_upper
    rot_y_left = 2 * math.pi * torch.rand(total_batch_size, device=device) # Full rotation around Y

    dev_x_right = (torch.rand(total_batch_size, device=device) - 0.5) * 2 * args.theta_upper
    dev_z_right = (torch.rand(total_batch_size, device=device) - 0.5) * 2 * args.theta_upper
    rot_y_right = 2 * math.pi * torch.rand(total_batch_size, device=device) # Full rotation around Y

    # Calculate rotation matrices batch-wise (ZYX order for deviation and rotation)
    # R = Rz(dev_z) @ Ry(rot_y) @ Rx(dev_x)
    cos_x_l, sin_x_l = torch.cos(dev_x_left), torch.sin(dev_x_left)
    cos_z_l, sin_z_l = torch.cos(dev_z_left), torch.sin(dev_z_left)
    cos_y_l, sin_y_l = torch.cos(rot_y_left), torch.sin(rot_y_left)

    cos_x_r, sin_x_r = torch.cos(dev_x_right), torch.sin(dev_x_right)
    cos_z_r, sin_z_r = torch.cos(dev_z_right), torch.sin(dev_z_right)
    cos_y_r, sin_y_r = torch.cos(rot_y_right), torch.sin(rot_y_right)

    R_left_batch = torch.zeros(total_batch_size, 3, 3, device=device)
    R_left_batch[:, 0, 0] = cos_z_l * cos_y_l - sin_z_l * sin_y_l * cos_x_l
    R_left_batch[:, 0, 1] = -cos_z_l * sin_y_l - sin_z_l * cos_y_l * cos_x_l
    R_left_batch[:, 0, 2] = sin_z_l * sin_x_l
    R_left_batch[:, 1, 0] = sin_z_l * cos_y_l + cos_z_l * sin_y_l * cos_x_l
    R_left_batch[:, 1, 1] = -sin_z_l * sin_y_l + cos_z_l * cos_y_l * cos_x_l
    R_left_batch[:, 1, 2] = -cos_z_l * sin_x_l
    R_left_batch[:, 2, 0] = sin_y_l * sin_x_l
    R_left_batch[:, 2, 1] = cos_y_l * sin_x_l
    R_left_batch[:, 2, 2] = cos_x_l

    R_right_batch = torch.zeros(total_batch_size, 3, 3, device=device)
    R_right_batch[:, 0, 0] = cos_z_r * cos_y_r - sin_z_r * sin_y_r * cos_x_r
    R_right_batch[:, 0, 1] = -cos_z_r * sin_y_r - sin_z_r * cos_y_r * cos_x_r
    R_right_batch[:, 0, 2] = sin_z_r * sin_x_r
    R_right_batch[:, 1, 0] = sin_z_r * cos_y_r + cos_z_r * sin_y_r * cos_x_r
    R_right_batch[:, 1, 1] = -sin_z_r * sin_y_r + cos_z_r * cos_y_r * cos_x_r
    R_right_batch[:, 1, 2] = -cos_z_r * sin_x_r
    R_right_batch[:, 2, 0] = sin_y_r * sin_x_r
    R_right_batch[:, 2, 1] = cos_y_r * sin_x_r
    R_right_batch[:, 2, 2] = cos_x_r

    # Apply rotations to base directions
    left_initial_dir_cartesian = torch.bmm(R_left_batch, base_left_dir.unsqueeze(2)).squeeze(2) # (B, 3)
    right_initial_dir_cartesian = torch.bmm(R_right_batch, base_right_dir.unsqueeze(2)).squeeze(2) # (B, 3)

    # Initial positions
    left_initial_pos = object_center.unsqueeze(0) + left_initial_dir_cartesian * dist_left.unsqueeze(1) # (B, 3)
    right_initial_pos = object_center.unsqueeze(0) + right_initial_dir_cartesian * dist_right.unsqueeze(1) # (B, 3)

    # Calculate orientation for each hand (palm faces object center with jitter)
    # We can reuse calculate_hand_pose_from_sampled_params_batch, but need to pass the correct jitter angles.
    # The parameters `deviate_theta`, `process_theta`, `rotate_theta` in calculate_hand_pose_from_sampled_params_batch
    # control the *local* jitter of the hand's orientation *after* its base orientation (palm facing object) is set.
    # These should be sampled independently for orientation jitter, using args.theta_lower/upper.

    # Sample jitter angles for hand orientation (local rotation after palm faces object)
    jitter_dev_theta_left = args.theta_lower + (args.theta_upper - args.theta_lower) * torch.rand([total_batch_size], dtype=torch.float, device=device)
    jitter_proc_theta_left = 2 * math.pi * torch.rand([total_batch_size], dtype=torch.float, device=device)
    jitter_rot_theta_left = 2 * math.pi * torch.rand([total_batch_size], dtype=torch.float, device=device)

    jitter_dev_theta_right = args.theta_lower + (args.theta_upper - args.theta_lower) * torch.rand([total_batch_size], dtype=torch.float, device=device)
    jitter_proc_theta_right = 2 * math.pi * torch.rand([total_batch_size], dtype=torch.float, device=device)
    jitter_rot_theta_right = 2 * math.pi * torch.rand([total_batch_size], dtype=torch.float, device=device)

    # Calculate Left Hand pose (pos + rot6d)
    left_pose_pr = calculate_hand_pose_from_sampled_params_batch(left_initial_pos, object_center, 
                                                                 jitter_dev_theta_left, jitter_proc_theta_left, jitter_rot_theta_left, device)

    # Calculate Right Hand pose (pos + rot6d)
    right_pose_pr = calculate_hand_pose_from_sampled_params_batch(right_initial_pos, object_center, 
                                                                  jitter_dev_theta_right, jitter_proc_theta_right, jitter_rot_theta_right, device)


    # Combine position and rotation (9D pose) into bimanual pose
    bimanual_poses[:, :9] = left_pose_pr
    bimanual_poses[:, 31:40] = right_pose_pr

    # Initialize joint angles with jitter for both hands (batch-wise)
    joint_angles_mu = torch.tensor([0.1, 0, 0.6, 0, 0, 0, 0.6, 0, -0.1, 0, 0.6, 0, 0, -0.2, 0, 0.6, 0, 0, 1.2, 0, -0.2, 0], dtype=torch.float, device=device)
    joint_angles_sigma = args.jitter_strength * (bimanual_hand_model.left_hand.joints_upper - bimanual_hand_model.left_hand.joints_lower) # Assuming left and right hand have same joint limits for sigma

    # Sample jitter from truncated normal distribution (batch-wise)
    # Needs torch.nn.init.trunc_normal_ which works in-place or requires looping
    # Let's use a direct sampling approximation for batching, or loop if necessary

    # Simple uniform jitter for initial exploration
    left_joints_jitter = (torch.rand(total_batch_size, 22, device=device) - 0.5) * 2 * args.jitter_strength
    right_joints_jitter = (torch.rand(total_batch_size, 22, device=device) - 0.5) * 2 * args.jitter_strength

    left_joints = joint_angles_mu.unsqueeze(0) + left_joints_jitter # (B, 22)
    right_joints = joint_angles_mu.unsqueeze(0) + right_joints_jitter # (B, 22)

    # Clamp joint angles to limits (batch-wise)
    left_joints = torch.clamp(left_joints, 
                                 bimanual_hand_model.left_hand.joints_lower.unsqueeze(0), 
                                 bimanual_hand_model.left_hand.joints_upper.unsqueeze(0))
    right_joints = torch.clamp(right_joints, 
                                  bimanual_hand_model.right_hand.joints_lower.unsqueeze(0), 
                                  bimanual_hand_model.right_hand.joints_upper.unsqueeze(0))

    # Add joint angles to bimanual pose
    bimanual_poses[:, 9:31] = left_joints # Left hand joints (9-30)
    bimanual_poses[:, 40:62] = right_joints # Right hand joints (40-61)

    # Set the calculated bimanual poses and contact indices
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
    
    # 오른손의 경우 손바닥 방향을 왼손과 같은 방향으로 설정
    if hand_side == 'right':
        palm_direction = -palm_direction
    
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
        thumb_direction = torch.cross(finger_direction, palm_direction)  # 순서 바꿈 (finger × palm)
    else:  # left hand
        thumb_direction = torch.cross(palm_direction, finger_direction)  # 기존 (palm × finger)
    
    thumb_direction = thumb_direction / torch.norm(thumb_direction)
    
    # 축 정의
    x_axis = thumb_direction    # 엄지 방향
    y_axis = palm_direction     # 손바닥 방향 (물체 향함)
    z_axis = finger_direction   # 손가락 방향
    
    # Rotation matrix 생성 - 축들을 열(column)으로 배치
    rotation_matrix = torch.stack([x_axis, y_axis, z_axis], dim=0).T.unsqueeze(0)  # (1, 3, 3)
    
    # 6D representation 생성
    first_two_cols = rotation_matrix[:, :, :2]
    first_two_cols_transposed = first_two_cols.transpose(-1, -2)  # (B, 2, 3)
    rot6d = first_two_cols_transposed.reshape(-1, 6).squeeze(0)  # (6,)
    
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

# Helper function to calculate hand pose (pos + rot6d) from sampled parameters (Batch-wise)
def calculate_hand_pose_from_sampled_params_batch(initial_pos, object_center, deviate_theta, process_theta, rotate_theta, device):
    # initial_pos: (B, 3)
    # object_center: (3,) or (B, 3) - assuming (3,) and will be unsqueezed
    # deviate_theta, process_theta, rotate_theta: (B,)
    
    batch_size = initial_pos.shape[0]
    
    # Ensure object_center is (B, 3)
    if object_center.dim() == 1:
        object_center = object_center.unsqueeze(0).repeat(batch_size, 1)
    
    # Calculate rotation matrix to make palm (-Y) face object center (Batch-wise)
    # Target palm normal direction: -(object_center - initial_pos).normalize()
    to_object = object_center - initial_pos # (B, 3)
    target_palm_normal = -to_object / torch.norm(to_object, dim=1, keepdim=True) # (B, 3)
    
    # Sample a random 'up' direction (avoiding alignment with target_palm_normal)
    up_vector = torch.randn(batch_size, 3, device=device)
    # Project out the target_palm_normal component
    dot_product = torch.sum(up_vector * target_palm_normal, dim=1, keepdim=True)
    up_vector = up_vector - dot_product * target_palm_normal
    up_vector = up_vector / torch.norm(up_vector, dim=1, keepdim=True) # Ensure unit vectors
    
    # Calculate hand frame axes (Batch-wise)
    y_axis = -target_palm_normal # Hand's -Y (palm normal) should point towards object (B, 3)
    z_axis = torch.cross(y_axis, up_vector, dim=1) # Hand's +Z (grasping) orthogonal to -Y and 'up' (B, 3)
    z_axis = z_axis / torch.norm(z_axis, dim=1, keepdim=True) # Ensure unit vectors
    x_axis = torch.cross(y_axis, z_axis, dim=1)  # Hand's +X (thumb) orthogonal to -Y and +Z (B, 3)
    x_axis = x_axis / torch.norm(x_axis, dim=1, keepdim=True) # Ensure unit vectors
    
    rotation_matrix_global = torch.stack([x_axis, y_axis, z_axis], dim=1) # (B, 3, 3) - This is R_world_hand
    
    # Apply local jitter rotation (around hand's own axes after global alignment)
    # R_final = R_world_hand @ R_local
    # Convert Euler angles to rotation matrices (Batch-wise)
    # Using transforms3d requires looping or batch implementation
    # Let's use a batch-wise implementation
    
    # Simple batch-wise rotation matrix creation from Euler angles (axes='rzxz')
    # R = Rz(process_theta) @ Rx(deviate_theta) @ Rz(rotate_theta)
    cos_p = torch.cos(process_theta)
    sin_p = torch.sin(process_theta)
    cos_d = torch.cos(deviate_theta)
    sin_d = torch.sin(deviate_theta)
    cos_r = torch.cos(rotate_theta)
    sin_r = torch.sin(rotate_theta)
    
    R_z_p = torch.stack([
        torch.stack([cos_p, -sin_p, torch.zeros_like(cos_p)], dim=1),
        torch.stack([sin_p, cos_p, torch.zeros_like(cos_p)], dim=1),
        torch.stack([torch.zeros_like(cos_p), torch.zeros_like(cos_p), torch.ones_like(cos_p)], dim=1)
    ], dim=1) # (B, 3, 3)
    
    R_x_d = torch.stack([
        torch.stack([torch.ones_like(cos_d), torch.zeros_like(cos_d), torch.zeros_like(cos_d)], dim=1),
        torch.stack([torch.zeros_like(cos_d), cos_d, -sin_d], dim=1),
        torch.stack([torch.zeros_like(cos_d), sin_d, cos_d], dim=1)
    ], dim=1) # (B, 3, 3)
    
    R_z_r = torch.stack([
        torch.stack([cos_r, -sin_r, torch.zeros_like(cos_r)], dim=1),
        torch.stack([sin_r, cos_r, torch.zeros_like(cos_r)], dim=1),
        torch.stack([torch.zeros_like(cos_r), torch.zeros_like(cos_r), torch.ones_like(cos_r)], dim=1)
    ], dim=1) # (B, 3, 3)
    
    local_jitter_matrix = torch.bmm(torch.bmm(R_z_p, R_x_d), R_z_r) # (B, 3, 3)
    
    final_rotation_matrix = torch.bmm(rotation_matrix_global, local_jitter_matrix) # (B, 3, 3)
    
    # Convert final rotation matrix to 6D representation (Batch-wise)
    final_rot6d = final_rotation_matrix[:, :, :2].transpose(1, 2).reshape(batch_size, 6) # (B, 6)
    
    # Combine position and rotation
    hand_pose_pr = torch.cat([initial_pos, final_rot6d], dim=1) # (B, 9)
    
    return hand_pose_pr 