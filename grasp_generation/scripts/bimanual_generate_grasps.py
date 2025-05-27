"""
Last modified date: 2024.12.16
Author: Bimanual Extension
Description: generate bimanual grasps in large-scale, use multiple graphics cards, no logging
"""

import os
import sys

sys.path.append(os.path.realpath('.'))

import argparse
import multiprocessing
import numpy as np
import torch
from tqdm import tqdm
import math
import random
import transforms3d

from utils.bimanual_hand_model import BimanualHandModel
from utils.object_model import ObjectModel
from utils.bimanual_initializations import initialize_bimanual_convex_hull
from utils.bimanual_energy import cal_bimanual_energy
from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d

from torch.multiprocessing import set_start_method

try:
    set_start_method('spawn')
except RuntimeError:
    pass


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
np.seterr(all='raise')


class BimanualAnnealing:
    """Custom annealing optimizer for bimanual grasping"""
    def __init__(self, bimanual_hand_model, **config):
        self.bimanual_hand_model = bimanual_hand_model
        self.config = config
        self.device = config['device']
        self.temperature = config['starting_temperature']
        self.step_count = 0
        self.prev_pose = None
        
    def try_step(self):
        # Save current pose for potential rollback (detach to avoid gradient issues)
        self.prev_pose = self.bimanual_hand_model.bimanual_pose.detach().clone()
        
        # Generate random step for bimanual pose
        batch_size = self.bimanual_hand_model.bimanual_pose.shape[0]
        step = torch.randn_like(self.bimanual_hand_model.bimanual_pose) * self.config['step_size']
        
        # Apply step with gradient consideration
        if self.bimanual_hand_model.bimanual_pose.grad is not None:
            step -= self.config['mu'] * self.bimanual_hand_model.bimanual_pose.grad * self.config['step_size']
        
        # Update pose in-place to maintain gradient tracking
        with torch.no_grad():
            self.bimanual_hand_model.bimanual_pose.add_(step)
        
        # Update hand model with new pose
        self.bimanual_hand_model.set_parameters(self.bimanual_hand_model.bimanual_pose, 
                                               self.bimanual_hand_model.contact_point_indices)
        
        return step
    
    def accept_step(self, old_energy, new_energy):
        batch_size = old_energy.shape[0]
        
        # Metropolis criterion
        energy_diff = new_energy - old_energy
        accept_prob = torch.exp(-energy_diff / self.temperature)
        random_vals = torch.rand(batch_size, device=self.device)
        accept = (random_vals < accept_prob) | (energy_diff <= 0)
        
        # Rollback rejected steps only (accepted steps are already applied)
        if (~accept).any():
            with torch.no_grad():
                self.bimanual_hand_model.bimanual_pose[~accept] = self.prev_pose[~accept]
            
            # Re-set parameters to ensure consistency
            self.bimanual_hand_model.set_parameters(self.bimanual_hand_model.bimanual_pose, 
                                                   self.bimanual_hand_model.contact_point_indices)
        
        # Update temperature
        self.step_count += 1
        if self.step_count % self.config['annealing_period'] == 0:
            self.temperature *= self.config['temperature_decay']
            
        return accept, self.temperature
    
    def zero_grad(self):
        if self.bimanual_hand_model.bimanual_pose.grad is not None:
            self.bimanual_hand_model.bimanual_pose.grad.zero_()


def generate(args_list):
    args, object_code_list, id, gpu_list = args_list

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # prepare models

    n_objects = len(object_code_list)

    worker = multiprocessing.current_process()._identity[0]
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list[worker - 1]
    device = torch.device('cuda')

    # Create bimanual hand model with minimal surface points for memory efficiency
    bimanual_hand_model = BimanualHandModel(
        mjcf_path='mjcf/shadow_hand_wrist_free.xml',
        mesh_path='mjcf/meshes',
        contact_points_path='mjcf/contact_points.json',
        penetration_points_path='mjcf/penetration_points.json',
        n_surface_points=100,  # Minimal surface points to reduce memory
        device=device
    )

    object_model = ObjectModel(
        data_root_path=args.data_root_path,
        batch_size_each=args.batch_size_each,
        num_samples=1000,  # Reduced from 2000 for bimanual memory efficiency
        device=device
    )
    object_model.initialize(object_code_list)

    initialize_bimanual_convex_hull(bimanual_hand_model, object_model, args)
    
    # Enable gradient tracking BEFORE saving initial pose
    bimanual_hand_model.bimanual_pose.requires_grad_(True)
    bimanual_pose_st = bimanual_hand_model.bimanual_pose.detach().clone()

    optim_config = {
        'switch_possibility': args.switch_possibility,
        'starting_temperature': args.starting_temperature,
        'temperature_decay': args.temperature_decay,
        'annealing_period': args.annealing_period,
        'step_size': args.step_size,
        'stepsize_period': args.stepsize_period,
        'mu': args.mu,
        'device': device
    }
    optimizer = BimanualAnnealing(bimanual_hand_model, **optim_config)

    # optimize
    
    weight_dict = dict(
        w_dis=args.w_dis,
        w_pen=args.w_pen,
        w_spen=args.w_spen,
        w_joints=args.w_joints,
        w_bimpen=args.w_bimpen,  # NEW: Inter-hand penetration
        w_vew=args.w_vew,        # NEW: Wrench ellipse volume
    )
    
    # Calculate initial energy
    energy, E_fc, E_dis, E_pen, E_spen, E_joints, E_bimpen, E_vew = cal_bimanual_energy(
        bimanual_hand_model, object_model, verbose=True, **weight_dict)

    energy.sum().backward(retain_graph=True)

    # Optimization loop with progress bar
    for step in tqdm(range(1, args.n_iter + 1), desc=f'Optimizing grasps (Worker {id})'):
        s = optimizer.try_step()

        optimizer.zero_grad()
        new_energy, new_E_fc, new_E_dis, new_E_pen, new_E_spen, new_E_joints, new_E_bimpen, new_E_vew = cal_bimanual_energy(
            bimanual_hand_model, object_model, verbose=True, **weight_dict)

        new_energy.sum().backward(retain_graph=True)

        with torch.no_grad():
            accept, t = optimizer.accept_step(energy, new_energy)

            energy[accept] = new_energy[accept]
            E_dis[accept] = new_E_dis[accept]
            E_fc[accept] = new_E_fc[accept]
            E_pen[accept] = new_E_pen[accept]
            E_spen[accept] = new_E_spen[accept]
            E_joints[accept] = new_E_joints[accept]
            E_bimpen[accept] = new_E_bimpen[accept]
            E_vew[accept] = new_E_vew[accept]


    # save results
    translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
    rot_names = ['WRJRx', 'WRJRy', 'WRJRz']
    joint_names = [
        'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
        'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
        'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
        'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
        'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
    ]
    
    # Bimanual naming convention: left hand gets 'L_', right hand gets 'R_'
    left_translation_names = ['L_WRJTx', 'L_WRJTy', 'L_WRJTz']
    left_rot_names = ['L_WRJRx', 'L_WRJRy', 'L_WRJRz']
    left_joint_names = ['L_' + name for name in joint_names]

    right_translation_names = ['R_WRJTx', 'R_WRJTy', 'R_WRJTz']
    right_rot_names = ['R_WRJRx', 'R_WRJRy', 'R_WRJRz']
    right_joint_names = ['R_' + name for name in joint_names]
    
    for i, object_code in enumerate(object_code_list):
        data_list = []
        for j in range(args.batch_size_each):
            idx = i * args.batch_size_each + j
            scale = object_model.object_scale_tensor[i][j].item()
            
            # Current bimanual pose
            bimanual_pose = bimanual_hand_model.bimanual_pose[idx].detach().cpu()
            
            # Split into left and right poses (using 31-dimensional poses)
            left_pose = bimanual_pose[:31]   # Left hand
            right_pose = bimanual_pose[31:]  # Right hand
            
            # Process left hand pose
            left_qpos = dict(zip(left_joint_names, left_pose[9:].tolist()))
            left_rot = robust_compute_rotation_matrix_from_ortho6d(left_pose[3:9].unsqueeze(0))[0]
            left_euler = transforms3d.euler.mat2euler(left_rot, axes='sxyz')
            left_qpos.update(dict(zip(left_rot_names, left_euler)))
            left_qpos.update(dict(zip(left_translation_names, left_pose[:3].tolist())))
            
            # Process right hand pose
            right_qpos = dict(zip(right_joint_names, right_pose[9:].tolist()))
            right_rot = robust_compute_rotation_matrix_from_ortho6d(right_pose[3:9].unsqueeze(0))[0]
            right_euler = transforms3d.euler.mat2euler(right_rot, axes='sxyz')
            right_qpos.update(dict(zip(right_rot_names, right_euler)))
            right_qpos.update(dict(zip(right_translation_names, right_pose[:3].tolist())))
            
            # Combine poses
            qpos = {**left_qpos, **right_qpos}
            
            # Process initial poses
            initial_bimanual_pose = bimanual_pose_st[idx].detach().cpu()
            left_pose_st = initial_bimanual_pose[:31]
            right_pose_st = initial_bimanual_pose[31:]
            
            left_qpos_st = dict(zip(left_joint_names, left_pose_st[9:].tolist()))
            left_rot_st = robust_compute_rotation_matrix_from_ortho6d(left_pose_st[3:9].unsqueeze(0))[0]
            left_euler_st = transforms3d.euler.mat2euler(left_rot_st, axes='sxyz')
            left_qpos_st.update(dict(zip(left_rot_names, left_euler_st)))
            left_qpos_st.update(dict(zip(left_translation_names, left_pose_st[:3].tolist())))
            
            right_qpos_st = dict(zip(right_joint_names, right_pose_st[9:].tolist()))
            right_rot_st = robust_compute_rotation_matrix_from_ortho6d(right_pose_st[3:9].unsqueeze(0))[0]
            right_euler_st = transforms3d.euler.mat2euler(right_rot_st, axes='sxyz')
            right_qpos_st.update(dict(zip(right_rot_names, right_euler_st)))
            right_qpos_st.update(dict(zip(right_translation_names, right_pose_st[:3].tolist())))
            
            qpos_st = {**left_qpos_st, **right_qpos_st}
            
            data_list.append(dict(
                scale=scale,
                qpos=qpos,
                qpos_st=qpos_st,
                energy=energy[idx].item(),
                E_fc=E_fc[idx].item(),
                E_dis=E_dis[idx].item(),
                E_pen=E_pen[idx].item(),
                E_spen=E_spen[idx].item(),
                E_joints=E_joints[idx].item(),
                E_bimpen=E_bimpen[idx].item(),  # NEW
                E_vew=E_vew[idx].item(),        # NEW
            ))
        np.save(os.path.join(args.result_path, 'bimanual_' + object_code + '.npy'), data_list, allow_pickle=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # experiment settings
    parser.add_argument('--result_path', default="../data/bimanual_graspdata", type=str)
    parser.add_argument('--data_root_path', default="../data/meshdata", type=str)
    parser.add_argument('--object_code_list', nargs='*', type=str)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--todo', action='store_true')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--n_contact', default=8, type=int)  # Changed from 4 to 8 for bimanual
    parser.add_argument('--batch_size_each', default=5, type=int)  # Extremely reduced for bimanual memory usage
    parser.add_argument('--max_total_batch_size', default=10, type=int)  # Extremely reduced for bimanual
    parser.add_argument('--n_iter', default=100, type=int)  # Reduced for testing bimanual
    # hyper parameters
    parser.add_argument('--switch_possibility', default=0.5, type=float)
    parser.add_argument('--mu', default=0.98, type=float)
    parser.add_argument('--step_size', default=0.003, type=float)  # Slightly reduced for stability
    parser.add_argument('--stepsize_period', default=50, type=int)
    parser.add_argument('--starting_temperature', default=20, type=float)  # Increased for bimanual
    parser.add_argument('--annealing_period', default=40, type=int)
    parser.add_argument('--temperature_decay', default=0.95, type=float)
    # Energy weights
    parser.add_argument('--w_dis', default=100.0, type=float)
    parser.add_argument('--w_pen', default=100.0, type=float)
    parser.add_argument('--w_spen', default=10.0, type=float)
    parser.add_argument('--w_joints', default=1.0, type=float)
    parser.add_argument('--w_bimpen', default=50.0, type=float)  # NEW: Inter-hand penetration weight
    parser.add_argument('--w_vew', default=1.0, type=float)     # NEW: Wrench ellipse volume weight
    # initialization settings
    parser.add_argument('--jitter_strength', default=0.1, type=float)
    parser.add_argument('--distance_lower', default=0.25, type=float)  # Increased for bimanual
    parser.add_argument('--distance_upper', default=0.35, type=float)  # Increased for bimanual
    parser.add_argument('--theta_lower', default=-math.pi / 6, type=float)
    parser.add_argument('--theta_upper', default=math.pi / 6, type=float)
    # energy thresholds
    parser.add_argument('--thres_fc', default=0.5, type=float)      # Relaxed for bimanual
    parser.add_argument('--thres_dis', default=0.008, type=float)   # Relaxed for bimanual
    parser.add_argument('--thres_pen', default=0.002, type=float)   # Relaxed for bimanual

    args = parser.parse_args()

    gpu_list = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    print(f'gpu_list: {gpu_list}')

    # check whether arguments are valid and process arguments

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    
    if not os.path.exists(args.data_root_path):
        raise ValueError(f'data_root_path {args.data_root_path} doesn\'t exist')
    
    if (args.object_code_list is not None) + args.all != 1:
        raise ValueError('exactly one among \'object_code_list\' \'all\' should be specified')
    
    if args.todo:
        with open("todo.txt", "r") as f:
            lines = f.readlines()
            object_code_list_all = [line[:-1] for line in lines]
    else:
        object_code_list_all = os.listdir(args.data_root_path)
    
    if args.object_code_list is not None:
        object_code_list = args.object_code_list
        if not set(object_code_list).issubset(set(object_code_list_all)):
            raise ValueError('object_code_list isn\'t a subset of dirs in data_root_path')
    else:
        object_code_list = object_code_list_all
    
    if not args.overwrite:
        for object_code in object_code_list.copy():
            if os.path.exists(os.path.join(args.result_path, 'bimanual_' + object_code + '.npy')):
                object_code_list.remove(object_code)

    if args.batch_size_each > args.max_total_batch_size:
        raise ValueError(f'batch_size_each {args.batch_size_each} should be smaller than max_total_batch_size {args.max_total_batch_size}')
    
    print(f'n_objects: {len(object_code_list)}')
    print(f'Bimanual grasp generation will start with {len(object_code_list)} objects')
    
    # generate

    random.seed(args.seed)
    random.shuffle(object_code_list)
    objects_each = args.max_total_batch_size // args.batch_size_each
    object_code_groups = [object_code_list[i: i + objects_each] for i in range(0, len(object_code_list), objects_each)]

    process_args = []
    for id, object_code_group in enumerate(object_code_groups):
        process_args.append((args, object_code_group, id + 1, gpu_list))

    with multiprocessing.Pool(len(gpu_list)) as p:
        it = tqdm(p.imap(generate, process_args), total=len(process_args), desc='generating bimanual grasps', maxinterval=1000)
        list(it)

    print('Bimanual grasp generation completed!')
    print(f'Results saved to: {args.result_path}') 