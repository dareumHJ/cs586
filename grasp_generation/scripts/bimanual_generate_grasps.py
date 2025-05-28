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
import wandb

# from utils.hand_model import HandModel
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
    
    # Initialize wandb for this worker
    if not args.disable_wandb:
        wandb.init(
            project=args.wandb_project,
            group=f"worker_{id}",
            name=f"worker_{id}_objects_{len(object_code_list)}",
            config={
                # Optimization hyperparameters
                "worker_id": id,
                "n_objects": len(object_code_list),
                "batch_size_each": args.batch_size_each,
                "n_iter": args.n_iter,
                "n_contact": args.n_contact,
                "seed": args.seed,
                
                # Annealing parameters
                "switch_possibility": args.switch_possibility,
                "starting_temperature": args.starting_temperature,
                "temperature_decay": args.temperature_decay,
                "annealing_period": args.annealing_period,
                "step_size": args.step_size,
                "stepsize_period": args.stepsize_period,
                "mu": args.mu,
                
                # Energy weights
                "w_dis": args.w_dis,
                "w_pen": args.w_pen,
                "w_spen": args.w_spen,
                "w_joints": args.w_joints,
                "w_bimpen": args.w_bimpen,
                "w_vew": args.w_vew,
                
                # Initialization settings
                "jitter_strength": args.jitter_strength,
                "distance_lower": args.distance_lower,
                "distance_upper": args.distance_upper,
                "theta_lower": args.theta_lower,
                "theta_upper": args.theta_upper,
                
                # Thresholds
                "thres_fc": args.thres_fc,
                "thres_dis": args.thres_dis,
                "thres_pen": args.thres_pen,
                
                # Wandb settings
                "wandb_log_freq": args.wandb_log_freq,
                
                # Object list
                "object_codes": object_code_list
            },
            tags=["bimanual", "grasp_generation", f"gpu_{gpu_list[id-1] if id <= len(gpu_list) else 'unknown'}"]
        )

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
        n_surface_points=100,  # Add: Minimal surface points to reduce memory
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
    bimanual_hand_model.bimanual_pose.requires_grad_(True)  # 이건 왜?
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
    
    # Log initial energy values
    if not args.disable_wandb:
        wandb.log({
            "initial/total_energy_mean": energy.mean().item(),
            "initial/total_energy_std": energy.std().item(),
            "initial/E_fc_mean": E_fc.mean().item(),
            "initial/E_dis_mean": E_dis.mean().item(),
            "initial/E_pen_mean": E_pen.mean().item(),
            "initial/E_spen_mean": E_spen.mean().item(),
            "initial/E_joints_mean": E_joints.mean().item(),
            "initial/E_bimpen_mean": E_bimpen.mean().item(),
            "initial/E_vew_mean": E_vew.mean().item(),
            "step": 0
        })

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
            
            # Log energy values and optimization metrics every few steps
            if not args.disable_wandb and (step % args.wandb_log_freq == 0 or step == 1):
                accept_rate = accept.float().mean().item()
                
                # Log current energy statistics
                log_dict = {
                    "step": step,
                    "optimization/temperature": t,
                    "optimization/accept_rate": accept_rate,
                    "optimization/step_norm": torch.norm(s, dim=1).mean().item(),
                    
                    # Current energy means and stds
                    "energy/total_mean": energy.mean().item(),
                    "energy/total_std": energy.std().item(),
                    "energy/total_min": energy.min().item(),
                    "energy/total_max": energy.max().item(),
                    
                    "energy/E_fc_mean": E_fc.mean().item(),
                    "energy/E_fc_std": E_fc.std().item(),
                    
                    "energy/E_dis_mean": E_dis.mean().item(),
                    "energy/E_dis_std": E_dis.std().item(),
                    
                    "energy/E_pen_mean": E_pen.mean().item(),
                    "energy/E_pen_std": E_pen.std().item(),
                    
                    "energy/E_spen_mean": E_spen.mean().item(),
                    "energy/E_spen_std": E_spen.std().item(),
                    
                    "energy/E_joints_mean": E_joints.mean().item(),
                    "energy/E_joints_std": E_joints.std().item(),
                    
                    "energy/E_bimpen_mean": E_bimpen.mean().item(),
                    "energy/E_bimpen_std": E_bimpen.std().item(),
                    
                    "energy/E_vew_mean": E_vew.mean().item(),
                    "energy/E_vew_std": E_vew.std().item(),
                }
                
                # Add percentage of successful grasps (below thresholds)
                good_fc = (E_fc < args.thres_fc).float().mean().item()
                good_dis = (E_dis < args.thres_dis).float().mean().item()
                good_pen = (E_pen < args.thres_pen).float().mean().item()
                
                log_dict.update({
                    "success_rate/good_fc_pct": good_fc * 100,
                    "success_rate/good_dis_pct": good_dis * 100,
                    "success_rate/good_pen_pct": good_pen * 100,
                    "success_rate/overall_good_pct": (good_fc * good_dis * good_pen) * 100
                })
                
                wandb.log(log_dict)

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

    # Log final results summary
    if not args.disable_wandb:
        wandb.log({
            "final/total_energy_mean": energy.mean().item(),
            "final/total_energy_std": energy.std().item(),
            "final/total_energy_min": energy.min().item(),
            "final/total_energy_max": energy.max().item(),
            "final/E_fc_mean": E_fc.mean().item(),
            "final/E_dis_mean": E_dis.mean().item(),
            "final/E_pen_mean": E_pen.mean().item(),
            "final/E_spen_mean": E_spen.mean().item(),
            "final/E_joints_mean": E_joints.mean().item(),
            "final/E_bimpen_mean": E_bimpen.mean().item(),
            "final/E_vew_mean": E_vew.mean().item(),
            "final/good_fc_pct": (E_fc < args.thres_fc).float().mean().item() * 100,
            "final/good_dis_pct": (E_dis < args.thres_dis).float().mean().item() * 100,
            "final/good_pen_pct": (E_pen < args.thres_pen).float().mean().item() * 100,
            "final/overall_success_pct": ((E_fc < args.thres_fc) & (E_dis < args.thres_dis) & (E_pen < args.thres_pen)).float().mean().item() * 100,
            "step": args.n_iter
        })
        
        # Close wandb run
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # experiment settings
    parser.add_argument('--result_path', default="../data/bimanual_graspdata", type=str)
    parser.add_argument('--data_root_path', default="../data/meshdata_one", type=str)
    parser.add_argument('--object_code_list', nargs='*', type=str)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--todo', action='store_true')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--n_contact', default=8, type=int)  # Changed from 4 to 8 for bimanual
    parser.add_argument('--batch_size_each', default=500, type=int)  # Extremely reduced for bimanual memory usage
    parser.add_argument('--max_total_batch_size', default=1000, type=int)  # Extremely reduced for bimanual
    parser.add_argument('--n_iter', default=6000, type=int)  # Reduced for testing bimanual
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
    # wandb settings
    parser.add_argument('--wandb_project', default='bimanual-grasp-generation', type=str)
    parser.add_argument('--wandb_log_freq', default=50, type=int, help='Log frequency for wandb (every N steps)')
    parser.add_argument('--disable_wandb', action='store_true', help='Disable wandb logging')

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