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
import time
import logging
import json

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


def load_config(config_path='../config_test.json'):
    """Load configuration from JSON file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


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
        # Save current pose for potential rollback (keep gradient)
        self.prev_pose = self.bimanual_hand_model.bimanual_pose.detach().clone()
        
        # Generate random step for bimanual pose
        batch_size = self.bimanual_hand_model.bimanual_pose.shape[0]
        step = torch.randn_like(self.bimanual_hand_model.bimanual_pose) * self.config['step_size']
        
        # Apply step with gradient consideration (with proper scaling)
        if self.bimanual_hand_model.bimanual_pose.grad is not None:
            # Normalize gradient to prevent extremely large steps
            grad = self.bimanual_hand_model.bimanual_pose.grad
            grad_norm = torch.norm(grad, dim=1, keepdim=True)
            
            # Clip gradient norm to reasonable values
            max_grad_norm = 10.0  # Maximum allowed gradient norm per batch
            grad_norm_clipped = torch.clamp(grad_norm, max=max_grad_norm)
            
            # Normalize gradient and scale properly
            grad_normalized = grad / (grad_norm + 1e-8) * grad_norm_clipped
            
            # Apply gradient-based step with much smaller mu for stability
            step -= self.config['mu'] * grad_normalized * self.config['step_size']
        
        # Update pose while maintaining gradient tracking using in-place operation
        self.bimanual_hand_model.bimanual_pose.data.add_(step)
        
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
        
        # Rollback rejected steps only (without breaking gradient)
        if (~accept).any():
            # Use in-place operation to maintain gradient tracking
            reject_mask = ~accept
            self.bimanual_hand_model.bimanual_pose.data[reject_mask] = self.prev_pose.data[reject_mask]
            
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


def setup_logging(worker_id):
    """Setup logging to file"""
    # Create log directory if it doesn't exist
    log_dir = "log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_filename = os.path.join(log_dir, f"bimanual_optimization_worker_{worker_id}.log")
    
    # Remove existing log file
    if os.path.exists(log_filename):
        os.remove(log_filename)
    
    # Setup logger
    logger = logging.getLogger(f'worker_{worker_id}')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler
    file_handler = logging.FileHandler(log_filename, mode='w')
    file_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - Worker %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    return logger


def setup_wandb_logging(worker_id):
    """Setup separate wandb logging to file"""
    log_dir = "log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    wandb_log_filename = os.path.join(log_dir, f"bimanual_wandb_log_worker_{worker_id}.log")
    
    # Remove existing wandb log file
    if os.path.exists(wandb_log_filename):
        os.remove(wandb_log_filename)
    
    # Setup wandb logger
    wandb_logger = logging.getLogger(f'wandb_worker_{worker_id}')
    wandb_logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    for handler in wandb_logger.handlers[:]:
        wandb_logger.removeHandler(handler)
    
    # File handler for wandb logs
    wandb_file_handler = logging.FileHandler(wandb_log_filename, mode='w')
    wandb_file_handler.setLevel(logging.INFO)
    
    # Simple formatter for wandb data
    wandb_formatter = logging.Formatter('%(asctime)s - %(message)s')
    wandb_file_handler.setFormatter(wandb_formatter)
    
    wandb_logger.addHandler(wandb_file_handler)
    
    return wandb_logger


def generate(args_list):
    args, object_code_list, id, gpu_list = args_list

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Setup logging
    logger = setup_logging(id)
    wandb_logger = setup_wandb_logging(id)  # NEW: Setup separate wandb logging
    
    logger.info(f"Starting bimanual grasp generation for worker {id}")
    logger.info(f"Objects: {object_code_list}")
    logger.info(f"Configuration: batch_size_each={args.batch_size_each}, n_iter={args.n_iter}")
    logger.info(f"Hyperparameters: mu={args.mu}, step_size={args.step_size}, starting_temp={args.starting_temperature}")
    logger.info(f"Energy weights: w_dis={args.w_dis}, w_pen={args.w_pen}, w_spen={args.w_spen}, w_joints={args.w_joints}, w_bimpen={args.w_bimpen}, w_vew={args.w_vew}")
    
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

    logger.info(f"Using GPU: {gpu_list[worker - 1]}")

    # Create bimanual hand model with minimal surface points for memory efficiency
    bimanual_hand_model = BimanualHandModel(
        model_path='models',
        mesh_path='models/meshes',
        contact_points_path='models',
        penetration_points_path='models/penetration_points.json',
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
    
    # Print initial location and rotation information
    print(f"\n=== Initial Bimanual Hand Poses ===")
    print(f"Total batch size: {bimanual_hand_model.bimanual_pose.shape[0]}")
    
    # Show first sample as example
    sample_pose = bimanual_hand_model.bimanual_pose[0]
    left_translation = sample_pose[:3]
    left_rotation = sample_pose[3:9]
    right_translation = sample_pose[31:34]
    right_rotation = sample_pose[34:40]
    
    print(f"\nSample [0] - Left Hand:")
    print(f"  Translation: [{left_translation[0]:.4f}, {left_translation[1]:.4f}, {left_translation[2]:.4f}]")
    print(f"  Rotation6D: [{left_rotation[0]:.4f}, {left_rotation[1]:.4f}, {left_rotation[2]:.4f}, {left_rotation[3]:.4f}, {left_rotation[4]:.4f}, {left_rotation[5]:.4f}]")
    
    print(f"\nSample [0] - Right Hand:")
    print(f"  Translation: [{right_translation[0]:.4f}, {right_translation[1]:.4f}, {right_translation[2]:.4f}]")
    print(f"  Rotation6D: [{right_rotation[0]:.4f}, {right_rotation[1]:.4f}, {right_rotation[2]:.4f}, {right_rotation[3]:.4f}, {right_rotation[4]:.4f}, {right_rotation[5]:.4f}]")
    
    # Show statistics for all samples
    all_left_trans = bimanual_hand_model.bimanual_pose[:, :3]
    all_right_trans = bimanual_hand_model.bimanual_pose[:, 31:34]
    
    print(f"\nAll Samples Statistics:")
    print(f"  Left Hand Translation - Mean: [{all_left_trans.mean(0)[0]:.4f}, {all_left_trans.mean(0)[1]:.4f}, {all_left_trans.mean(0)[2]:.4f}]")
    print(f"  Right Hand Translation - Mean: [{all_right_trans.mean(0)[0]:.4f}, {all_right_trans.mean(0)[1]:.4f}, {all_right_trans.mean(0)[2]:.4f}]")
    print(f"  Distance between hands: {torch.norm(all_left_trans - all_right_trans, dim=1).mean():.4f} ± {torch.norm(all_left_trans - all_right_trans, dim=1).std():.4f}")
    print("===================================\n")
    
    # Enable gradient tracking BEFORE saving initial pose
    bimanual_hand_model.bimanual_pose.requires_grad_(True)  # 이건 왜?
    bimanual_pose_st = bimanual_hand_model.bimanual_pose.detach().clone()

    # Initialize storage for epoch results
    epoch_poses = {}  # Store poses at different epochs
    epoch_energies = {}  # Store energies at different epochs
    
    # NEW: Calculate save_interval dynamically based on save_num
    # save_num개의 중간 지점 + 처음(0) + 마지막(n_iter) = 총 save_num+2개 지점
    if args.save_num > 0:
        save_interval = max(1, args.n_iter // (args.save_num + 1))
    else:
        save_interval = args.n_iter  # Only save initial and final if save_num is 0
    
    logger.info(f"Using save_num={args.save_num}, calculated save_interval={save_interval}")
    print(f"Save configuration: save_num={args.save_num}, save_interval={save_interval}")
    
    # Store initial state
    epoch_poses[0] = bimanual_pose_st.clone()
    
    logger.info(f"Initial pose shape: {bimanual_hand_model.bimanual_pose.shape}")
    logger.info(f"Initial pose requires_grad: {bimanual_hand_model.bimanual_pose.requires_grad}")
    
    # DEBUG: Check contact points initialization
    if bimanual_hand_model.contact_point_indices is not None:
        logger.info(f"Contact point indices shape: {bimanual_hand_model.contact_point_indices.shape}")
        logger.info(f"Contact point indices range: [{bimanual_hand_model.contact_point_indices.min().item()}, {bimanual_hand_model.contact_point_indices.max().item()}]")
        logger.info(f"Left hand contact candidates: {bimanual_hand_model.left_hand.n_contact_candidates}")
        logger.info(f"Right hand contact candidates: {bimanual_hand_model.right_hand.n_contact_candidates}")
    else:
        logger.error("CRITICAL: Contact point indices are None!")
        
    if bimanual_hand_model.contact_points is not None:
        logger.info(f"Contact points shape: {bimanual_hand_model.contact_points.shape}")
        logger.info(f"Contact points mean: {bimanual_hand_model.contact_points.mean(dim=[0,1])}")
    else:
        logger.error("CRITICAL: Contact points are None!")

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
    
    # IMPORTANT: Reset gradient to None before initial calculation
    bimanual_hand_model.bimanual_pose.grad = None
    
    # Calculate initial energy with proper gradient tracking
    energy, E_fc, E_dis, E_pen, E_spen, E_joints, E_bimpen, E_vew, E_contact_sep = cal_bimanual_energy(
        bimanual_hand_model, object_model, verbose=True, **weight_dict)

    # Store initial energy data
    epoch_energies[0] = {
        'total': energy.clone(),
        'E_fc': E_fc.clone(),
        'E_dis': E_dis.clone(),
        'E_pen': E_pen.clone(),
        'E_spen': E_spen.clone(),
        'E_joints': E_joints.clone(),
        'E_bimpen': E_bimpen.clone(),
        'E_vew': E_vew.clone(),
        'E_contact_sep': E_contact_sep.clone()
    }

    # Ensure gradient calculation is working by computing scalar loss
    total_loss = energy.sum()
    # total_loss.backward(retain_graph=True)
    total_loss.backward(retain_graph=False)
    
    # Check initial gradient
    initial_grad_norm = 0.0
    if bimanual_hand_model.bimanual_pose.grad is not None:
        initial_grad_norm = torch.norm(bimanual_hand_model.bimanual_pose.grad).item()
        logger.info(f"Initial gradient norm: {initial_grad_norm}")
        print(f"Initial gradient norm: {initial_grad_norm}")
    else:
        logger.error("CRITICAL: No gradient computed for initial pose! Check energy calculation.")
        print("CRITICAL ERROR: No gradient computed for initial pose!")
        
        # Try to debug what's wrong
        logger.info(f"Energy requires_grad: {energy.requires_grad}")
        logger.info(f"Pose requires_grad: {bimanual_hand_model.bimanual_pose.requires_grad}")
        
        # Force gradient calculation
        dummy_loss = bimanual_hand_model.bimanual_pose.sum()
        dummy_loss.backward()
        if bimanual_hand_model.bimanual_pose.grad is not None:
            logger.info(f"Dummy gradient norm: {torch.norm(bimanual_hand_model.bimanual_pose.grad).item()}")
        else:
            logger.error("Even dummy gradient failed!")
    
    logger.info(f"Initial energy - Total: {energy.mean().item():.4f}, FC: {E_fc.mean().item():.4f}, Dis: {E_dis.mean().item():.4f}")
    
    # Clear gradients before starting optimization
    optimizer.zero_grad()
    
    # Log initial energy values after gradient check
    if not args.disable_wandb:
        initial_wandb_data = {
            "initial/total_energy_mean": energy.mean().item(),
            "initial/total_energy_std": energy.std().item(),
            "initial/E_fc_mean": E_fc.mean().item(),
            "initial/E_dis_mean": E_dis.mean().item(),
            "initial/E_pen_mean": E_pen.mean().item(),
            "initial/E_spen_mean": E_spen.mean().item(),
            "initial/E_joints_mean": E_joints.mean().item(),
            "initial/E_bimpen_mean": E_bimpen.mean().item(),
            "initial/E_vew_mean": E_vew.mean().item(),
            "initial/E_contact_sep_mean": E_contact_sep.mean().item(),
            "initial/gradient_norm": initial_grad_norm,
            "step": 0
        }
        wandb.log(initial_wandb_data)
        
        # Also log to separate wandb file
        wandb_logger.info(f"INITIAL: {initial_wandb_data}")

    # Optimization loop with progress bar
    for step in tqdm(range(1, args.n_iter + 1), desc=f'Optimizing grasps (Worker {id})'):
        # Store pose before step for comparison (WITHOUT detach to maintain gradient tracking)
        pose_before_step = bimanual_hand_model.bimanual_pose.clone()
        
        s = optimizer.try_step()
        
        # Store pose after try_step (before accept/reject decision) - keep gradients
        pose_after_try_step = bimanual_hand_model.bimanual_pose.clone()
        with torch.no_grad():  # Only detach for logging calculations
            actual_step_taken = torch.norm(pose_after_try_step.detach() - pose_before_step.detach(), dim=1).mean().item()

        optimizer.zero_grad()
        new_energy, new_E_fc, new_E_dis, new_E_pen, new_E_spen, new_E_joints, new_E_bimpen, new_E_vew, new_E_contact_sep = cal_bimanual_energy(
            bimanual_hand_model, object_model, verbose=True, **weight_dict)

        # new_energy.sum().backward(retain_graph=True)
        new_energy.sum().backward(retain_graph=False)

        # Accept step without disabling gradients
        accept, t = optimizer.accept_step(energy, new_energy)

        # Check if pose actually changed after accept/reject decision - only detach for logging
        with torch.no_grad():
            pose_after_step = bimanual_hand_model.bimanual_pose.clone()
            final_pose_change = torch.norm(pose_after_step.detach() - pose_before_step.detach(), dim=1).mean().item()

        # Update energy values for accepted steps only (keep gradient tracking)
        energy = torch.where(accept, new_energy, energy)
        E_dis = torch.where(accept, new_E_dis, E_dis)
        E_fc = torch.where(accept, new_E_fc, E_fc)
        E_pen = torch.where(accept, new_E_pen, E_pen)
        E_spen = torch.where(accept, new_E_spen, E_spen)
        E_joints = torch.where(accept, new_E_joints, E_joints)
        E_bimpen = torch.where(accept, new_E_bimpen, E_bimpen)
        E_vew = torch.where(accept, new_E_vew, E_vew)
        E_contact_sep = torch.where(accept, new_E_contact_sep, E_contact_sep)
        
        # Detailed step analysis for logging (use detach only for logging)
        with torch.no_grad():
            step_norm = torch.norm(s, dim=1).mean().item()
            step_norm_std = torch.norm(s, dim=1).std().item()
            step_norm_max = torch.norm(s, dim=1).max().item()
            step_norm_min = torch.norm(s, dim=1).min().item()
            
            # Gradient analysis
            grad_norm = 0.0
            grad_norm_std = 0.0
            grad_norm_max = 0.0
            grad_norm_min = 0.0
            if bimanual_hand_model.bimanual_pose.grad is not None:
                grad_norms = torch.norm(bimanual_hand_model.bimanual_pose.grad, dim=1)
                grad_norm = grad_norms.mean().item()
                grad_norm_std = grad_norms.std().item()
                grad_norm_max = grad_norms.max().item()
                grad_norm_min = grad_norms.min().item()
            
            # Energy difference analysis
            energy_diff = new_energy - energy
            energy_diff_mean = energy_diff.mean().item()
            energy_diff_std = energy_diff.std().item()
            energy_diff_min = energy_diff.min().item()
            energy_diff_max = energy_diff.max().item()
            
            # Accept rate and rejection analysis
            accept_rate = accept.float().mean().item()
            n_accepted = accept.sum().item()
            n_rejected = (~accept).sum().item()
            
            # Metropolis probability analysis
            accept_prob = torch.exp(-energy_diff / t)
            avg_accept_prob = accept_prob.mean().item()
            min_accept_prob = accept_prob.min().item()
            max_accept_prob = accept_prob.max().item()
        
        # Log to file every step for detailed analysis
        if step % 10 == 0 or step <= 5:  # Log first 5 steps and every 10th step
            logger.info(f"Step {step}: temp={t:.4f}, accept_rate={accept_rate:.3f} ({n_accepted}/{n_accepted+n_rejected})")
            logger.info(f"  Step: norm={step_norm:.6f}, actual_taken={actual_step_taken:.6f}, final_change={final_pose_change:.6f}")
            logger.info(f"  Grad: norm={grad_norm:.6f}")
            logger.info(f"  Energy: old={energy.mean().item():.4f}, new={new_energy.mean().item():.4f}, diff={energy_diff_mean:.4f}")
            logger.info(f"  Energy_diff: min={energy_diff_min:.4f}, max={energy_diff_max:.4f}")
            logger.info(f"  Accept_prob: avg={avg_accept_prob:.6f}, min={min_accept_prob:.6f}, max={max_accept_prob:.6f}")
            
            # DEBUG: Log individual energy components to see what's changing
            logger.info(f"  Energy Components - FC: {new_E_fc.mean().item():.4f}, Dis: {new_E_dis.mean().item():.4f}, Pen: {new_E_pen.mean().item():.4f}")
            logger.info(f"  Energy Components - SPen: {new_E_spen.mean().item():.4f}, Joints: {new_E_joints.mean().item():.4f}, Bim: {new_E_bimpen.mean().item():.4f}, VEW: {new_E_vew.mean().item():.4f}, SEP: {new_E_contact_sep.mean().item():.4f}")
            
            # DEBUG: Check if contact points are actually changing
            if step > 1 and hasattr(bimanual_hand_model, 'contact_points') and bimanual_hand_model.contact_points is not None:
                contact_change = torch.norm(bimanual_hand_model.contact_points - getattr(bimanual_hand_model, '_prev_contact_points', bimanual_hand_model.contact_points), dim=-1).mean().item()
                logger.info(f"  Contact point change: {contact_change:.6f}")
            
            # Store contact points for next comparison
            if hasattr(bimanual_hand_model, 'contact_points') and bimanual_hand_model.contact_points is not None:
                bimanual_hand_model._prev_contact_points = bimanual_hand_model.contact_points.clone().detach()
        
        # Log energy values and optimization metrics every few steps
        if not args.disable_wandb and (step % args.wandb_log_freq == 0 or step == 1):
            with torch.no_grad():  # Only disable gradients for logging
                
                # Log current energy statistics
                log_dict = {
                    "step": step,
                    "optimization/temperature": t,
                    "optimization/accept_rate": accept_rate,
                    "optimization/n_accepted": n_accepted,
                    "optimization/n_rejected": n_rejected,
                    
                    # Step analysis (NEW)
                    "optimization/step_norm": step_norm,
                    "optimization/step_norm_std": step_norm_std,
                    "optimization/step_norm_max": step_norm_max,
                    "optimization/step_norm_min": step_norm_min,
                    "optimization/actual_step_taken": actual_step_taken,
                    "optimization/final_pose_change": final_pose_change,
                    
                    # Gradient analysis (NEW)
                    "optimization/gradient_norm": grad_norm,
                    "optimization/gradient_norm_std": grad_norm_std,
                    "optimization/gradient_norm_max": grad_norm_max,
                    "optimization/gradient_norm_min": grad_norm_min,
                    
                    # Energy difference analysis (NEW)
                    "optimization/energy_diff_mean": energy_diff_mean,
                    "optimization/energy_diff_std": energy_diff_std,
                    "optimization/energy_diff_min": energy_diff_min,
                    "optimization/energy_diff_max": energy_diff_max,
                    
                    # Metropolis probability analysis (NEW)
                    "optimization/avg_accept_prob": avg_accept_prob,
                    "optimization/min_accept_prob": min_accept_prob,
                    "optimization/max_accept_prob": max_accept_prob,
                    
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
                    
                    "energy/E_contact_sep_mean": E_contact_sep.mean().item(),
                    "energy/E_contact_sep_std": E_contact_sep.std().item(),
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
                
                # Also log to separate wandb file
                wandb_logger.info(f"STEP_{step}: {log_dict}")

        # Save intermediate results every save_interval epochs
        if step % save_interval == 0:
            epoch_poses[step] = bimanual_hand_model.bimanual_pose.detach().clone()
            epoch_energies[step] = {
                'total': energy.clone(),
                'E_fc': E_fc.clone(),
                'E_dis': E_dis.clone(),
                'E_pen': E_pen.clone(),
                'E_spen': E_spen.clone(),
                'E_joints': E_joints.clone(),
                'E_bimpen': E_bimpen.clone(),
                'E_vew': E_vew.clone(),
                'E_contact_sep': E_contact_sep.clone()
            }
            logger.info(f"Saved intermediate result at epoch {step}")

    # Final logging
    logger.info(f"Optimization completed for worker {id}")
    logger.info(f"Final energy: {energy.mean().item():.4f}, Final accept rate: {accept_rate:.3f}")
    logger.info(f"Final gradient norm: {grad_norm:.6f}")

    # Store final epoch results
    epoch_poses[args.n_iter] = bimanual_hand_model.bimanual_pose.detach().clone()
    epoch_energies[args.n_iter] = {
        'total': energy.clone(),
        'E_fc': E_fc.clone(),
        'E_dis': E_dis.clone(),
        'E_pen': E_pen.clone(),
        'E_spen': E_spen.clone(),
        'E_joints': E_joints.clone(),
        'E_bimpen': E_bimpen.clone(),
        'E_vew': E_vew.clone(),
        'E_contact_sep': E_contact_sep.clone()
    }

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
            left_rot = robust_compute_rotation_matrix_from_ortho6d(left_pose[3:9].unsqueeze(0), hand_side='left')[0]
            left_euler = transforms3d.euler.mat2euler(left_rot, axes='sxyz')
            left_qpos.update(dict(zip(left_rot_names, left_euler)))
            left_qpos.update(dict(zip(left_translation_names, left_pose[:3].tolist())))
            
            # Process right hand pose
            right_qpos = dict(zip(right_joint_names, right_pose[9:].tolist()))
            right_rot = robust_compute_rotation_matrix_from_ortho6d(right_pose[3:9].unsqueeze(0), hand_side='right')[0]
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
            left_rot_st = robust_compute_rotation_matrix_from_ortho6d(left_pose_st[3:9].unsqueeze(0), hand_side='left')[0]
            left_euler_st = transforms3d.euler.mat2euler(left_rot_st, axes='sxyz')
            left_qpos_st.update(dict(zip(left_rot_names, left_euler_st)))
            left_qpos_st.update(dict(zip(left_translation_names, left_pose_st[:3].tolist())))
            
            right_qpos_st = dict(zip(right_joint_names, right_pose_st[9:].tolist()))
            right_rot_st = robust_compute_rotation_matrix_from_ortho6d(right_pose_st[3:9].unsqueeze(0), hand_side='right')[0]
            right_euler_st = transforms3d.euler.mat2euler(right_rot_st, axes='sxyz')
            right_qpos_st.update(dict(zip(right_rot_names, right_euler_st)))
            right_qpos_st.update(dict(zip(right_translation_names, right_pose_st[:3].tolist())))
            
            qpos_st = {**left_qpos_st, **right_qpos_st}
            
            # Process epoch poses
            epoch_qpos = {}
            for epoch, epoch_pose_batch in epoch_poses.items():
                epoch_bimanual_pose = epoch_pose_batch[idx].detach().cpu()
                left_epoch_pose = epoch_bimanual_pose[:31]
                right_epoch_pose = epoch_bimanual_pose[31:]
                
                # Process left hand epoch pose
                left_epoch_qpos = dict(zip(left_joint_names, left_epoch_pose[9:].tolist()))
                left_epoch_rot = robust_compute_rotation_matrix_from_ortho6d(left_epoch_pose[3:9].unsqueeze(0), hand_side='left')[0]
                left_epoch_euler = transforms3d.euler.mat2euler(left_epoch_rot, axes='sxyz')
                left_epoch_qpos.update(dict(zip(left_rot_names, left_epoch_euler)))
                left_epoch_qpos.update(dict(zip(left_translation_names, left_epoch_pose[:3].tolist())))
                
                # Process right hand epoch pose
                right_epoch_qpos = dict(zip(right_joint_names, right_epoch_pose[9:].tolist()))
                right_epoch_rot = robust_compute_rotation_matrix_from_ortho6d(right_epoch_pose[3:9].unsqueeze(0), hand_side='right')[0]
                right_epoch_euler = transforms3d.euler.mat2euler(right_epoch_rot, axes='sxyz')
                right_epoch_qpos.update(dict(zip(right_rot_names, right_epoch_euler)))
                right_epoch_qpos.update(dict(zip(right_translation_names, right_epoch_pose[:3].tolist())))
                
                epoch_qpos[epoch] = {**left_epoch_qpos, **right_epoch_qpos}
                
            # Process epoch energies
            epoch_energy_data = {}
            for epoch, energy_dict in epoch_energies.items():
                epoch_energy_data[epoch] = {
                    'total': energy_dict['total'][idx].item(),
                    'E_fc': energy_dict['E_fc'][idx].item(),
                    'E_dis': energy_dict['E_dis'][idx].item(),
                    'E_pen': energy_dict['E_pen'][idx].item(),
                    'E_spen': energy_dict['E_spen'][idx].item(),
                    'E_joints': energy_dict['E_joints'][idx].item(),
                    'E_bimpen': energy_dict['E_bimpen'][idx].item(),
                    'E_vew': energy_dict['E_vew'][idx].item(),
                    'E_contact_sep': energy_dict['E_contact_sep'][idx].item()
                }
            
            data_list.append(dict(
                scale=scale,
                qpos=qpos,
                qpos_st=qpos_st,
                epoch_qpos=epoch_qpos,  # NEW: Store all epoch poses
                epoch_energies=epoch_energy_data,  # NEW: Store all epoch energies
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

    logger.info(f"Results saved for {len(object_code_list)} objects")

    # Log final results summary
    if not args.disable_wandb:
        final_wandb_data = {
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
        }
        wandb.log(final_wandb_data)
        
        # Also log to separate wandb file
        wandb_logger.info(f"FINAL: {final_wandb_data}")
        
        # Close wandb run
        wandb.finish()
    
    logger.info("Bimanual grasp generation completed successfully")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # experiment settings
    parser.add_argument('--result_path', default="../data/bimanual_graspdata_test", type=str)
    parser.add_argument('--data_root_path', default="../data/meshdata_one", type=str)
    parser.add_argument('--object_code_list', nargs='*', type=str)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--todo', action='store_true')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--n_contact', default=8, type=int)  # Changed from 4 to 8 for bimanual
    parser.add_argument('--batch_size_each', default=500, type=int)  # Extremely reduced for bimanual memory usage
    parser.add_argument('--max_total_batch_size', default=1000, type=int)  # Extremely reduced for bimanual
    # hyper parameters
    parser.add_argument('--switch_possibility', default=0.5, type=float)
    parser.add_argument('--mu', default=0.1, type=float)  # Reduced from 0.98 for stability with gradient clipping
    parser.add_argument('--step_size', default=0.01, type=float)  # Increased from 0.003 since gradient is now clipped
    parser.add_argument('--stepsize_period', default=50, type=int)
    parser.add_argument('--starting_temperature', default=100, type=float)  # Increased for bimanual to allow more exploration
    parser.add_argument('--annealing_period', default=40, type=int)
    parser.add_argument('--temperature_decay', default=0.95, type=float)
    # Energy weights
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

    # Load configuration from JSON file
    try:
        config = load_config()
        # Add config values to args
        args.n_iter = config.get('n_iter', 1000)  # Default to 1000 if not in config
        args.w_dis = config.get('w_dis', 100.0)
        args.w_pen = config.get('w_pen', 100.0)
        args.w_spen = config.get('w_spen', 10.0)
        args.w_joints = config.get('w_joints', 1.0)
        args.w_bimpen = config.get('w_bimpen', 50.0)
        args.w_vew = config.get('w_vew', 1.0)
        args.save_num = config.get('save_num', 4)  # NEW: Load save_num from config
        print(f'Loaded config: n_iter = {args.n_iter}')
        print(f'Loaded config: w_dis = {args.w_dis}, w_pen = {args.w_pen}, w_spen = {args.w_spen}')
        print(f'Loaded config: w_joints = {args.w_joints}, w_bimpen = {args.w_bimpen}, w_vew = {args.w_vew}')
        print(f'Loaded config: save_num = {args.save_num}')
    except FileNotFoundError:
        print("Warning: config_test.json not found, using default values")
        args.n_iter = 1000  # Default value
        args.w_dis = 100.0
        args.w_pen = 100.0
        args.w_spen = 10.0
        args.w_joints = 1.0
        args.w_bimpen = 50.0
        args.w_vew = 1.0
        args.save_num = 4  # Default value

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