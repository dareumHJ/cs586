"""
Last modified date: 2024.12.16
Author: Bimanual Extension
Description: Energy functions for bimanual grasping based on BimanGrasp paper
"""

import torch
import numpy as np


def cal_bimanual_energy(bimanual_hand_model, object_model, w_dis=100.0, w_pen=100.0, w_spen=10.0, 
                       w_joints=1.0, w_bimpen=50.0, w_vew=1.0, verbose=False):
    """
    Calculate bimanual energy function based on BimanGrasp paper
    
    Parameters
    ----------
    bimanual_hand_model: BimanualHandModel
        bimanual hand model with both left and right hands
    object_model: ObjectModel
        object model
    w_dis: float
        weight for hand-object distance term
    w_pen: float  
        weight for hand-object penetration term
    w_spen: float
        weight for hand self-penetration term
    w_joints: float
        weight for joint limit violation term
    w_bimpen: float
        weight for inter-hand penetration term
    w_vew: float
        weight for wrench ellipse volume term
    verbose: bool
        whether to return individual energy terms
        
    Returns
    -------
    energy: torch.FloatTensor
        total energy (if verbose=False) or tuple of energy terms (if verbose=True)
    """
    
    # E_dis: Hand-object distance (sum of both hands)
    batch_size, n_contact, _ = bimanual_hand_model.contact_points.shape  # (B, 8, 3)
    device = object_model.device
    distance, contact_normal = object_model.cal_distance(bimanual_hand_model.contact_points)  # (B, 8), (B, 8, 3)
    E_dis = torch.sum(distance.abs(), dim=-1, dtype=torch.float).to(device)  # (B,)

    # E_fc: Bimanual Force Closure (8 contact points)
    E_fc = cal_bimanual_force_closure(bimanual_hand_model, contact_normal, device)

    # E_vew: Wrench Ellipse Volume  
    E_vew = cal_wrench_ellipse_volume(bimanual_hand_model, contact_normal, device)
    
    # E_joints: Joint limit violations for both hands
    E_joints = cal_joint_violations(bimanual_hand_model)

    # E_pen: Hand-object penetration for both hands
    E_pen = cal_hand_object_penetration(bimanual_hand_model, object_model)

    # E_spen: Self-penetration for both hands
    E_spen = bimanual_hand_model.self_penetration()

    # E_bimpen: Inter-hand penetration (NEW)
    E_bimpen = bimanual_hand_model.inter_hand_penetration()

    # Check for NaN or infinite values for debugging
    energy_components = [E_fc, E_dis, E_pen, E_spen, E_joints, E_bimpen, E_vew]
    component_names = ['E_fc', 'E_dis', 'E_pen', 'E_spen', 'E_joints', 'E_bimpen', 'E_vew']
    
    for i, (component, name) in enumerate(zip(energy_components, component_names)):
        if torch.isnan(component).any() or torch.isinf(component).any():
            print(f"Warning: {name} contains NaN or Inf values!")
            # Replace NaN/Inf with large finite values to continue optimization
            energy_components[i] = torch.where(torch.isfinite(component), component, 
                                             torch.full_like(component, 1000.0))

    # Unpack cleaned components
    E_fc, E_dis, E_pen, E_spen, E_joints, E_bimpen, E_vew = energy_components

    if verbose:
        total_energy = E_fc + w_dis * E_dis + w_pen * E_pen + w_spen * E_spen + w_joints * E_joints + w_bimpen * E_bimpen + w_vew * E_vew
        return total_energy, E_fc, E_dis, E_pen, E_spen, E_joints, E_bimpen, E_vew
    else:
        return E_fc + w_dis * E_dis + w_pen * E_pen + w_spen * E_spen + w_joints * E_joints + w_bimpen * E_bimpen + w_vew * E_vew


def cal_bimanual_force_closure(bimanual_hand_model, contact_normal, device):
    """
    Calculate bimanual force closure with 8 contact points (4 from each hand)
    Based on BimanGrasp paper Eq. (2) and (3)
    
    Parameters
    ----------
    bimanual_hand_model: BimanualHandModel
        bimanual hand model
    contact_normal: (B, 8, 3) torch.FloatTensor
        contact normals at 8 contact points
    device: torch.Device
        computation device
        
    Returns
    -------
    E_fc: (B,) torch.FloatTensor
        force closure energy
    """
    batch_size, n_contact, _ = bimanual_hand_model.contact_points.shape  # (B, 8, 3)
    
    # Reshape contact normal for matrix operations
    contact_normal = contact_normal.reshape(batch_size, 1, 3 * n_contact)  # (B, 1, 24)
    
    # Transformation matrix for cross product operations
    transformation_matrix = torch.tensor([[0, 0, 0, 0, 0, -1, 0, 1, 0],
                                          [0, 0, 1, 0, 0, 0, -1, 0, 0],
                                          [0, -1, 0, 1, 0, 0, 0, 0, 0]],
                                         dtype=torch.float, device=device)
    
    # Build grasp matrix G according to Eq. (2)
    # G = [I I I I I I I I]  (3x24)
    #     [R1 R2 R3 R4 R5 R6 R7 R8]  (3x24)
    # where Ri is the skew-symmetric matrix of contact point i
    
    identity_part = torch.eye(3, dtype=torch.float, device=device).expand(batch_size, n_contact, 3, 3).reshape(batch_size, 3 * n_contact, 3)
    moment_part = (bimanual_hand_model.contact_points @ transformation_matrix).view(batch_size, 3 * n_contact, 3)
    
    g = torch.cat([identity_part, moment_part], dim=2).float().to(device)  # (B, 24, 6)
    
    # Calculate ||Gc||^2 where c is contact normal
    gc = contact_normal @ g  # (B, 1, 6)
    norm = torch.norm(gc, dim=[1, 2])  # (B,)
    E_fc = norm * norm
    
    return E_fc


def cal_wrench_ellipse_volume(bimanual_hand_model, contact_normal, device):
    """
    Calculate wrench ellipse volume term to prevent ill-conditioned grasp matrix
    Based on BimanGrasp paper Table I
    
    Parameters
    ----------
    bimanual_hand_model: BimanualHandModel
        bimanual hand model
    contact_normal: (B, 8, 3) torch.FloatTensor
        contact normals at 8 contact points
    device: torch.Device
        computation device
        
    Returns
    -------
    E_vew: (B,) torch.FloatTensor
        wrench ellipse volume energy
    """
    batch_size, n_contact, _ = bimanual_hand_model.contact_points.shape
    
    # Reshape contact normal
    contact_normal = contact_normal.reshape(batch_size, 1, 3 * n_contact)
    
    # Build grasp matrix G (same as in force closure)
    transformation_matrix = torch.tensor([[0, 0, 0, 0, 0, -1, 0, 1, 0],
                                          [0, 0, 1, 0, 0, 0, -1, 0, 0],
                                          [0, -1, 0, 1, 0, 0, 0, 0, 0]],
                                         dtype=torch.float, device=device)
    
    identity_part = torch.eye(3, dtype=torch.float, device=device).expand(batch_size, n_contact, 3, 3).reshape(batch_size, 3 * n_contact, 3)
    moment_part = (bimanual_hand_model.contact_points @ transformation_matrix).view(batch_size, 3 * n_contact, 3)
    
    G = torch.cat([identity_part, moment_part], dim=2).float().to(device)  # (B, 24, 6)
    
    # Calculate G^T G (6x6) instead of GG^T (24x24) to avoid singular matrix
    GTG = torch.bmm(G.transpose(1, 2), G)  # (B, 6, 6)
    
    # Calculate determinant^(-1/2) as in the paper
    # Add small regularization to prevent numerical issues
    epsilon = 1e-8
    det_GTG = torch.det(GTG + epsilon * torch.eye(GTG.size(-1), device=device).unsqueeze(0))
    
    # Avoid negative determinants and taking sqrt of negative numbers
    det_GTG = torch.clamp(det_GTG, min=epsilon)
    E_vew = 1.0 / torch.sqrt(det_GTG)
    
    return E_vew


def cal_joint_violations(bimanual_hand_model):
    """
    Calculate joint limit violations for both hands
    
    Parameters
    ----------
    bimanual_hand_model: BimanualHandModel
        bimanual hand model
        
    Returns
    -------
    E_joints: (B,) torch.FloatTensor
        joint limit violation energy
    """
    # Extract joint angles for both hands: [9:31] + [40:62] (skip trans + rot6d)
    left_joints = bimanual_hand_model.bimanual_pose[:, 9:31]  # (B, 22) - left hand joints
    right_joints = bimanual_hand_model.bimanual_pose[:, 40:62]  # (B, 22) - right hand joints
    
    # Combine all joint angles for both hands
    all_joints = torch.cat([left_joints, right_joints], dim=1)  # (B, 44)
    
    # Use bimanual joint limits (which combines both hands' limits)
    upper_violations = torch.sum((all_joints > bimanual_hand_model.joints_upper) * 
                                (all_joints - bimanual_hand_model.joints_upper), dim=-1)
    lower_violations = torch.sum((all_joints < bimanual_hand_model.joints_lower) * 
                                (bimanual_hand_model.joints_lower - all_joints), dim=-1)
    
    return upper_violations + lower_violations


def cal_hand_object_penetration(bimanual_hand_model, object_model):
    """
    Calculate hand-object penetration for both hands
    
    Parameters
    ----------
    bimanual_hand_model: BimanualHandModel
        bimanual hand model
    object_model: ObjectModel
        object model
        
    Returns
    -------
    E_pen: (B,) torch.FloatTensor
        hand-object penetration energy
    """
    # Get object surface points
    object_scale = object_model.object_scale_tensor.flatten().unsqueeze(1).unsqueeze(2)
    object_surface_points = object_model.surface_points_tensor * object_scale
    
    # Calculate penetration for left hand
    left_distances = bimanual_hand_model.left_hand.cal_distance(object_surface_points)
    # Use ReLU instead of direct indexing to maintain gradients
    left_penetration = torch.sum(torch.relu(-left_distances), dim=-1)
    
    # Calculate penetration for right hand
    right_distances = bimanual_hand_model.right_hand.cal_distance(object_surface_points) 
    # Use ReLU instead of direct indexing to maintain gradients
    right_penetration = torch.sum(torch.relu(-right_distances), dim=-1)
    
    return left_penetration + right_penetration 