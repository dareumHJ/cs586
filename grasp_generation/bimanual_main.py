"""
Last modified date: 2024.12.16
Author: Bimanual Extension
Description: Entry of the bimanual grasp generation program, based on BimanGrasp paper
"""

import os

os.chdir(os.path.dirname(__file__))

import argparse
import shutil
import numpy as np
import torch
from tqdm import tqdm
import math
import transforms3d

from utils.bimanual_hand_model import BimanualHandModel
from utils.object_model import ObjectModel
from utils.bimanual_initializations import initialize_bimanual_convex_hull
from utils.bimanual_energy import cal_bimanual_energy
from utils.optimizer import Annealing
from utils.logger import Logger
from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d


# prepare arguments

parser = argparse.ArgumentParser()
# experiment settings
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--gpu', default="0", type=str)
parser.add_argument('--object_code_list', default=
    [
        'sem-Car-2f28e2bd754977da8cfac9da0ff28f62',
        'sem-Car-27e267f0570f121869a949ac99a843c4',
        'sem-Car-669043a8ce40d9d78781f76a6db4ab62',
        'sem-Car-58379002fbdaf20e61a47cff24512a0',
        'sem-Car-aeeb2fb31215f3249acee38782dd9680',
    ], type=list)
parser.add_argument('--name', default='bimanual_exp_1', type=str)
parser.add_argument('--n_contact', default=8, type=int)  # Changed from 4 to 8 for bimanual
parser.add_argument('--batch_size', default=64, type=int)  # Reduced due to higher complexity
parser.add_argument('--n_iter', default=8000, type=int)   # Increased for bimanual convergence
# hyper parameters (** Magic, don't touch! **)
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

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

np.seterr(all='raise')
np.random.seed(args.seed)
torch.manual_seed(args.seed)


# prepare models

total_batch_size = len(args.object_code_list) * args.batch_size

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('running on', device)
print('Bimanual grasp generation started')

# Create bimanual hand model
bimanual_hand_model = BimanualHandModel(
    mjcf_path='mjcf/shadow_hand_wrist_free.xml',
    mesh_path='mjcf/meshes',
    contact_points_path='mjcf/contact_points.json',
    penetration_points_path='mjcf/penetration_points.json',
    device=device
)

object_model = ObjectModel(
    data_root_path='../data/meshdata',
    batch_size_each=args.batch_size,
    num_samples=2000, 
    device=device
)
object_model.initialize(args.object_code_list)

# Initialize bimanual poses
initialize_bimanual_convex_hull(bimanual_hand_model, object_model, args)

print('n_contact_candidates (total)', bimanual_hand_model.n_contact_candidates)
print('n_contact_candidates (left)', bimanual_hand_model.left_hand.n_contact_candidates)
print('n_contact_candidates (right)', bimanual_hand_model.right_hand.n_contact_candidates)
print('total batch size', total_batch_size)
bimanual_pose_st = bimanual_hand_model.bimanual_pose.detach()

# Create optimizer adapted for bimanual poses
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

# We need to adapt the optimizer to work with bimanual_hand_model
# For now, we'll create a custom optimizer for bimanual case
class BimanualAnnealing:
    def __init__(self, bimanual_hand_model, **config):
        self.bimanual_hand_model = bimanual_hand_model
        self.config = config
        self.device = config['device']
        self.temperature = config['starting_temperature']
        self.step_count = 0
        self.prev_pose = None
        
    def try_step(self):
        # Save current pose for potential rollback
        self.prev_pose = self.bimanual_hand_model.bimanual_pose.clone()
        
        # Generate random step for bimanual pose
        batch_size = self.bimanual_hand_model.bimanual_pose.shape[0]
        step = torch.randn_like(self.bimanual_hand_model.bimanual_pose) * self.config['step_size']
        
        # Apply step with gradient consideration
        if self.bimanual_hand_model.bimanual_pose.grad is not None:
            step -= self.config['mu'] * self.bimanual_hand_model.bimanual_pose.grad * self.config['step_size']
        
        self.proposed_pose = self.bimanual_hand_model.bimanual_pose + step
        
        # Update hand model with proposed pose
        self.bimanual_hand_model.set_parameters(self.proposed_pose, 
                                               self.bimanual_hand_model.contact_point_indices)
        
        return step
    
    def accept_step(self, old_energy, new_energy):
        batch_size = old_energy.shape[0]
        
        # Metropolis criterion
        energy_diff = new_energy - old_energy
        accept_prob = torch.exp(-energy_diff / self.temperature)
        random_vals = torch.rand(batch_size, device=self.device)
        accept = (random_vals < accept_prob) | (energy_diff <= 0)
        
        # Update poses for accepted steps and rollback rejected ones
        self.bimanual_hand_model.bimanual_pose[accept] = self.proposed_pose[accept]
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

optimizer = BimanualAnnealing(bimanual_hand_model, **optim_config)

# Setup logging
try:
    shutil.rmtree(os.path.join('../data/experiments', args.name, 'logs'))
except FileNotFoundError:
    pass
os.makedirs(os.path.join('../data/experiments', args.name, 'logs'), exist_ok=True)
logger_config = {
    'thres_fc': args.thres_fc,
    'thres_dis': args.thres_dis,
    'thres_pen': args.thres_pen
}
logger = Logger(log_dir=os.path.join('../data/experiments', args.name, 'logs'), **logger_config)


# log settings

with open(os.path.join('../data/experiments', args.name, 'output.txt'), 'w') as f:
    f.write(str(args) + '\n')


# optimize

weight_dict = dict(
    w_dis=args.w_dis,
    w_pen=args.w_pen,
    w_spen=args.w_spen,
    w_joints=args.w_joints,
    w_bimpen=args.w_bimpen,  # NEW
    w_vew=args.w_vew,        # NEW
)

# Enable gradient tracking for bimanual pose
bimanual_hand_model.bimanual_pose.requires_grad_(True)

energy, E_fc, E_dis, E_pen, E_spen, E_joints, E_bimpen, E_vew = cal_bimanual_energy(
    bimanual_hand_model, object_model, verbose=True, **weight_dict)

energy.sum().backward(retain_graph=True)
logger.log(energy, E_fc, E_dis, E_pen, E_spen, E_joints, E_bimpen, E_vew, 0, show=False)

print(f'Initial energy: {energy.mean().item():.4f}')
print(f'E_fc: {E_fc.mean().item():.4f}, E_dis: {E_dis.mean().item():.4f}')
print(f'E_pen: {E_pen.mean().item():.4f}, E_spen: {E_spen.mean().item():.4f}')
print(f'E_joints: {E_joints.mean().item():.4f}, E_bimpen: {E_bimpen.mean().item():.4f}')
print(f'E_vew: {E_vew.mean().item():.4f}')

for step in tqdm(range(1, args.n_iter + 1), desc='optimizing bimanual grasps'):
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

        # Log every 100 steps
        if step % 100 == 0:
            logger.log(energy, E_fc, E_dis, E_pen, E_spen, E_joints, E_bimpen, E_vew, step, show=False)
            
        # Print progress every 1000 steps
        if step % 1000 == 0:
            print(f'Step {step}: Energy: {energy.mean().item():.4f}, Temperature: {t:.4f}')
            print(f'  E_bimpen: {E_bimpen.mean().item():.4f}, Accept rate: {accept.float().mean().item():.3f}')


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

try:
    shutil.rmtree(os.path.join('../data/experiments', args.name, 'results'))
except FileNotFoundError:
    pass
os.makedirs(os.path.join('../data/experiments', args.name, 'results'), exist_ok=True)
result_path = os.path.join('../data/experiments', args.name, 'results')

for i in range(len(args.object_code_list)):
    data_list = []
    for j in range(args.batch_size):
        idx = i * args.batch_size + j
        scale = object_model.object_scale_tensor[i][j].item()
        
        # Current bimanual pose
        bimanual_pose = bimanual_hand_model.bimanual_pose[idx].detach().cpu()
        
        # Split into left and right poses
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
    np.save(os.path.join(result_path, 'bimanual_' + args.object_code_list[i] + '.npy'), data_list, allow_pickle=True)

print('Bimanual grasp generation completed!')
print(f'Results saved to: {result_path}')
print(f'Final average energy: {energy.mean().item():.4f}')
print(f'Final inter-hand penetration: {E_bimpen.mean().item():.4f}') 