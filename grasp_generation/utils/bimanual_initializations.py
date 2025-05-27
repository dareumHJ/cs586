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


def initialize_bimanual_convex_hull(bimanual_hand_model, object_model, args):
    """
    Initialize bimanual hand poses around objects using symmetric convex hull approach
    Based on BimanGrasp paper initialization strategy
    
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
    
    # Initialize bimanual poses (B, 62)
    bimanual_poses = torch.zeros(total_batch_size, 62, device=device)
    
    # Initialize contact point indices (B, 8)
    left_contact_indices = torch.randint(0, bimanual_hand_model.left_hand.n_contact_candidates, 
                                        (total_batch_size, 4), device=device)
    right_contact_indices = torch.randint(0, bimanual_hand_model.right_hand.n_contact_candidates,
                                         (total_batch_size, 4), device=device) + bimanual_hand_model.left_hand.n_contact_candidates
    contact_indices = torch.cat([left_contact_indices, right_contact_indices], dim=1)
    
    for i in range(n_objects):
        # Get object properties
        object_mesh = object_model.object_mesh_list[i]
        object_scale = object_model.object_scale_tensor[i][0].item()  # Use first scale from tensor
        
        # Calculate object center and bounding box
        vertices = torch.tensor(object_mesh.vertices, dtype=torch.float, device=device) * object_scale
        object_center = vertices.mean(dim=0)  # (3,)
        bbox_min = vertices.min(dim=0)[0]
        bbox_max = vertices.max(dim=0)[0]
        object_size = (bbox_max - bbox_min).max().item()
        
        for j in range(batch_size_each):
            idx = i * batch_size_each + j
            
            # Initialize left and right hand poses symmetrically
            left_pose, right_pose = initialize_symmetric_poses(
                object_center, object_size, args, device, bimanual_hand_model
            )
            
            # Combine into bimanual pose
            bimanual_poses[idx, :31] = left_pose    # Left hand pose
            bimanual_poses[idx, 31:] = right_pose   # Right hand pose
    
    # Set parameters to the bimanual hand model
    bimanual_hand_model.set_parameters(bimanual_poses, contact_indices)


def initialize_symmetric_poses(object_center, object_size, args, device, bimanual_hand_model=None):
    """
    Initialize left and right hand poses symmetrically around object
    
    Parameters
    ----------
    object_center: (3,) torch.FloatTensor
        center of the target object
    object_size: float
        maximum dimension of the object
    args: argparse.Namespace
        configuration arguments
    device: torch.Device
        computation device
    bimanual_hand_model: BimanualHandModel, optional
        bimanual hand model for joint limits
        
    Returns
    -------
    left_pose: (31,) torch.FloatTensor
        left hand pose [translation(3), rotation(6), joints(22)]
    right_pose: (31,) torch.FloatTensor
        right hand pose [translation(3), rotation(6), joints(22)]
    """
    
    # Calculate approach distance based on object size
    approach_distance = max(args.distance_lower, min(args.distance_upper, object_size * 0.8))
    
    # Random approach angle variations
    theta_variation = torch.rand(1, device=device) * (args.theta_upper - args.theta_lower) + args.theta_lower
    phi_variation = torch.rand(1, device=device) * 2 * math.pi  # Random azimuthal angle
    
    # Base approach directions (symmetric)
    # Left hand approaches from left side (-x direction)
    left_approach_dir = torch.tensor([-1.0, 0.0, 0.0], device=device)
    # Right hand approaches from right side (+x direction)  
    right_approach_dir = torch.tensor([1.0, 0.0, 0.0], device=device)
    
    # Apply random rotations to approach directions
    rotation_z = torch.tensor([
        [torch.cos(theta_variation), -torch.sin(theta_variation), 0],
        [torch.sin(theta_variation), torch.cos(theta_variation), 0],
        [0, 0, 1]
    ], device=device).squeeze()
    
    rotation_y = torch.tensor([
        [torch.cos(phi_variation), 0, torch.sin(phi_variation)],
        [0, 1, 0],
        [-torch.sin(phi_variation), 0, torch.cos(phi_variation)]
    ], device=device).squeeze()
    
    total_rotation = rotation_y @ rotation_z
    
    left_approach_dir = total_rotation @ left_approach_dir
    right_approach_dir = total_rotation @ right_approach_dir
    
    # Calculate hand positions
    left_translation = object_center + left_approach_dir * approach_distance
    right_translation = object_center + right_approach_dir * approach_distance
    
    # Calculate hand orientations (palms facing object)
    left_forward = (object_center - left_translation)
    left_forward = left_forward / torch.norm(left_forward)
    
    right_forward = (object_center - right_translation)  
    right_forward = right_forward / torch.norm(right_forward)
    
    # Create rotation matrices (simplified - palms facing object)
    up = torch.tensor([0.0, 0.0, 1.0], device=device)
    
    # Left hand orientation
    left_right = torch.cross(left_forward, up)
    left_right = left_right / torch.norm(left_right)
    left_up = torch.cross(left_right, left_forward)
    left_rotation_matrix = torch.stack([left_right, left_up, left_forward], dim=1)
    
    # Right hand orientation  
    right_right = torch.cross(right_forward, up)
    right_right = right_right / torch.norm(right_right)
    right_up = torch.cross(right_right, right_forward)
    right_rotation_matrix = torch.stack([right_right, right_up, right_forward], dim=1)
    
    # Convert rotation matrices to 6D representation
    left_rot6d = matrix_to_rot6d(left_rotation_matrix.unsqueeze(0)).squeeze()
    right_rot6d = matrix_to_rot6d(right_rotation_matrix.unsqueeze(0)).squeeze()
    
    # Initialize joint angles with jitter
    left_joints = torch.zeros(22, device=device)
    right_joints = torch.zeros(22, device=device)
    
    # Add random jitter to joint angles
    if args.jitter_strength > 0:
        joint_jitter_left = (torch.rand(22, device=device) - 0.5) * 2 * args.jitter_strength
        joint_jitter_right = (torch.rand(22, device=device) - 0.5) * 2 * args.jitter_strength
        left_joints += joint_jitter_left
        right_joints += joint_jitter_right
    
    # Clamp joint angles to limits if bimanual_hand_model is provided
    if bimanual_hand_model is not None:
        left_joints = torch.clamp(left_joints, 
                                 bimanual_hand_model.left_hand.joints_lower,
                                 bimanual_hand_model.left_hand.joints_upper)
        right_joints = torch.clamp(right_joints,
                                  bimanual_hand_model.right_hand.joints_lower, 
                                  bimanual_hand_model.right_hand.joints_upper)
    
    # Combine pose components
    left_pose = torch.cat([left_translation, left_rot6d, left_joints])
    right_pose = torch.cat([right_translation, right_rot6d, right_joints])
    
    return left_pose, right_pose


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
                    object_idx = i // batch_size_each
                    object_mesh = object_model.object_mesh_list[object_idx]
                    object_scale = object_model.object_scale_tensor[object_idx][0].item()  # Use first scale from tensor
                    
                    vertices = torch.tensor(object_mesh.vertices, dtype=torch.float, device=device) * object_scale
                    object_center = vertices.mean(dim=0)
                    
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