"""
Last modified date: 2024.12.16
Author: Bimanual Extension
Description: BimanualHandModel for bimanual grasping based on two HandModel instances
"""

import torch
import numpy as np
from .hand_model import HandModel
from .rot6d import robust_compute_rotation_matrix_from_ortho6d


class BimanualHandModel:
    # Shadow Hand pose size: 3 (trans) + 6 (rot6d) + 22 (joints) = 31
    HAND_POSE_SIZE = 31
    
    def __init__(self, model_path, mesh_path, contact_points_path, penetration_points_path, n_surface_points=0, device='cpu'):
        """
        Create a Bimanual Hand Model using two HandModel instances
        
        Parameters
        ----------
        mjcf_path: str
            path to mjcf file (will be used for both hands)
        mesh_path: str
            path to mesh directory
        contact_points_path: str
            path to hand-selected contact candidates
        penetration_points_path: str
            path to hand-selected penetration keypoints
        n_surface_points: int
            number of points to sample from surface of each hand
        device: str | torch.Device
            device for torch tensors
        """
        
        self.device = device
        
        # Create two hand models
        # Reduce surface points for memory efficiency in bimanual setting
        reduced_surface_points = max(1, n_surface_points // 4)  # Reduce by 75%
        
        self.left_hand = HandModel(
            mjcf_path=model_path+'/left_shadow_hand_wrist_free.xml',
            mesh_path=mesh_path, 
            contact_points_path=contact_points_path+'/left_hand_contact_points.json',
            penetration_points_path=penetration_points_path,
            n_surface_points=reduced_surface_points,
            device=device
        )
        
        self.right_hand = HandModel(
            mjcf_path=model_path+'/right_shadow_hand_wrist_free.xml',
            mesh_path=mesh_path,
            contact_points_path=contact_points_path+'/right_hand_contact_points.json', 
            penetration_points_path=penetration_points_path,
            n_surface_points=reduced_surface_points,
            device=device
        )
        
        # Bimanual properties
        self.n_dofs = self.left_hand.n_dofs + self.right_hand.n_dofs  # 22 * 2 = 44
        self.n_contact_candidates = self.left_hand.n_contact_candidates + self.right_hand.n_contact_candidates
        self.n_keypoints = self.left_hand.n_keypoints + self.right_hand.n_keypoints
        
        # Joint limits for both hands
        self.joints_lower = torch.cat([self.left_hand.joints_lower, self.right_hand.joints_lower])
        self.joints_upper = torch.cat([self.left_hand.joints_upper, self.right_hand.joints_upper])
        
        # Bimanual parameters  
        self.bimanual_pose = None  # (B, 62) = (B, 31*2)
        self.contact_point_indices = None  # (B, 8) = (B, 4*2)
        self.contact_points = None  # (B, 8, 3)
        
    def set_parameters(self, bimanual_pose, contact_point_indices=None):
        """
        Set translation, rotation, joint angles, and contact points for both hands
        
        Parameters
        ----------
        bimanual_pose: (B, 62) torch.FloatTensor
            [left_trans(3), left_rot6d(6), left_joints(22), right_trans(3), right_rot6d(6), right_joints(22)]
        contact_point_indices: (B, 8) [Optional] torch.LongTensor
            indices of contact candidates (4 for left hand, 4 for right hand)
        """
        self.bimanual_pose = bimanual_pose
        batch_size = bimanual_pose.shape[0]
        
        # Split bimanual pose into left and right
        left_pose = bimanual_pose[:, :self.HAND_POSE_SIZE]   # (B, 31)
        right_pose = bimanual_pose[:, self.HAND_POSE_SIZE:]  # (B, 31)
        
        # Set parameters for each hand
        if contact_point_indices is not None:
            self.contact_point_indices = contact_point_indices
            left_contact_indices = contact_point_indices[:, :4]  # First 4 contacts for left hand
            right_contact_indices = contact_point_indices[:, 4:] # Last 4 contacts for right hand
            
            # Adjust right hand contact indices to account for left hand's contact candidates
            right_contact_indices_adjusted = right_contact_indices - self.left_hand.n_contact_candidates
            
            self.left_hand.set_parameters(left_pose, left_contact_indices)
            self.right_hand.set_parameters(right_pose, right_contact_indices_adjusted)
            
            # Combine contact points from both hands
            self.contact_points = torch.cat([
                self.left_hand.contact_points,   # (B, 4, 3)
                self.right_hand.contact_points   # (B, 4, 3)
            ], dim=1)  # (B, 8, 3)
        else:
            self.left_hand.set_parameters(left_pose)
            self.right_hand.set_parameters(right_pose)
            
    def get_combined_contact_candidates(self):
        """Get contact candidates from both hands"""
        left_candidates = self.left_hand.contact_candidates
        right_candidates = self.right_hand.contact_candidates
        return torch.cat([left_candidates, right_candidates], dim=0)
    
    def get_combined_penetration_keypoints(self):
        """Get penetration keypoints from both hands"""
        left_keypoints = self.left_hand.penetration_keypoints
        right_keypoints = self.right_hand.penetration_keypoints
        return torch.cat([left_keypoints, right_keypoints], dim=0)
        
    def cal_distance(self, x):
        """
        Calculate signed distances from object point clouds to both hand surface meshes
        
        Parameters
        ----------
        x: (B, N, 3) torch.FloatTensor
            object point clouds
            
        Returns
        -------
        distances: (B, N) torch.FloatTensor
            minimum distances to either hand surface
        """
        left_distances = self.left_hand.cal_distance(x)
        right_distances = self.right_hand.cal_distance(x)
        
        # Return minimum distance to either hand
        return torch.maximum(left_distances, right_distances)
    
    def inter_hand_penetration(self, threshold=0.002):
        """
        Calculate penetration between left and right hands
        
        Parameters
        ----------
        threshold: float
            penetration threshold in meters
            
        Returns
        -------
        penetration: (B,) torch.FloatTensor
            inter-hand penetration energy
        """
        if self.bimanual_pose is None:
            raise ValueError("Must call set_parameters first")
            
        batch_size = self.bimanual_pose.shape[0]
        
        # Get surface points from left hand
        left_surface_points = self.left_hand.get_surface_points()  # (B, N_left, 3)
        
        # Calculate distances from left hand surface points to right hand mesh
        right_distances = self.right_hand.cal_distance(left_surface_points)  # (B, N_left)
        
        # Get surface points from right hand  
        right_surface_points = self.right_hand.get_surface_points()  # (B, N_right, 3)
        
        # Calculate distances from right hand surface points to left hand mesh
        left_distances = self.left_hand.cal_distance(right_surface_points)  # (B, N_right)
        
        # Calculate penetration using ReLU to maintain gradients
        # Penetration occurs when distance < threshold
        left_to_right_penetration = torch.relu(threshold - torch.abs(right_distances))
        right_to_left_penetration = torch.relu(threshold - torch.abs(left_distances))
        
        # Sum penetrations
        total_penetration = left_to_right_penetration.sum(dim=-1) + right_to_left_penetration.sum(dim=-1)
        
        return total_penetration
    
    def self_penetration(self):
        """
        Calculate self-penetration for both hands
        
        Returns
        -------
        penetration: (B,) torch.FloatTensor
            combined self-penetration energy from both hands
        """
        left_spen = self.left_hand.self_penetration()
        right_spen = self.right_hand.self_penetration()
        return left_spen + right_spen
    
    def get_surface_points(self):
        """
        Get surface points from both hands
        
        Returns
        -------
        surface_points: (B, N_total, 3) torch.FloatTensor
            combined surface points from both hands
        """
        left_points = self.left_hand.get_surface_points()
        right_points = self.right_hand.get_surface_points()
        return torch.cat([left_points, right_points], dim=1)
    
    def get_plotly_data(self, i, opacity=0.5, left_color='lightblue', right_color='lightcoral', 
                       with_contact_points=False, pose=None):
        """
        Get plotly visualization data for both hands
        
        Parameters
        ----------
        i: int
            batch index
        opacity: float
            mesh opacity
        left_color: str
            color for left hand
        right_color: str  
            color for right hand
        with_contact_points: bool
            whether to include contact points
        pose: optional
            hand pose override
            
        Returns
        -------
        data: list
            plotly data objects for both hands
        """
        left_data = self.left_hand.get_plotly_data(i, opacity, left_color, with_contact_points, 
                                                   pose[:self.HAND_POSE_SIZE] if pose is not None else None)
        right_data = self.right_hand.get_plotly_data(i, opacity, right_color, with_contact_points,
                                                     pose[self.HAND_POSE_SIZE:] if pose is not None else None)
        return left_data + right_data 