"""
Last modified date: 2024.12.16
Author: Bimanual Extension
Description: visualize bimanual grasp result using plotly.graph_objects
"""

import os
import sys

# os.chdir(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import torch
import numpy as np
import transforms3d
import plotly.graph_objects as go

from utils.bimanual_hand_model import BimanualHandModel
from utils.object_model import ObjectModel

# Original joint and pose names
translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
rot_names = ['WRJRx', 'WRJRy', 'WRJRz']
joint_names = [
    'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
    'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
    'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
    'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
    'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
]

# Bimanual naming convention
left_translation_names = ['L_' + name for name in translation_names]
left_rot_names = ['L_' + name for name in rot_names]
left_joint_names = ['L_' + name for name in joint_names]

right_translation_names = ['R_' + name for name in translation_names]
right_rot_names = ['R_' + name for name in rot_names]
right_joint_names = ['R_' + name for name in joint_names]


def extract_bimanual_pose(qpos, device='cpu'):
    """Extract bimanual pose from qpos dictionary"""
    # Extract left hand pose
    left_rot = np.array(transforms3d.euler.euler2mat(*[qpos[name] for name in left_rot_names]))
    left_rot = left_rot[:, :2].T.ravel().tolist()
    left_pose = torch.tensor([qpos[name] for name in left_translation_names] + left_rot + 
                            [qpos[name] for name in left_joint_names], dtype=torch.float, device=device)
    
    # Extract right hand pose
    right_rot = np.array(transforms3d.euler.euler2mat(*[qpos[name] for name in right_rot_names]))
    right_rot = right_rot[:, :2].T.ravel().tolist()
    right_pose = torch.tensor([qpos[name] for name in right_translation_names] + right_rot + 
                             [qpos[name] for name in right_joint_names], dtype=torch.float, device=device)
    
    # Combine into bimanual pose (62 dimensions)
    bimanual_pose = torch.cat([left_pose, right_pose], dim=0)
    
    return bimanual_pose


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--object_code', type=str, default='sem-Camera-7bff4fd4dc53de7496dece3f86cb5dd5')
    parser.add_argument('--num', type=int, default=0)
    parser.add_argument('--result_path', type=str, default='../data/bimanual_graspdata_test')
    parser.add_argument('--output_file', type=str, default='bimanual_grasp_result_test.html')
    args = parser.parse_args()

    device = 'cpu'

    # Load bimanual grasp results
    grasp_file = os.path.join(args.result_path, f'bimanual_{args.object_code}.npy')
    if not os.path.exists(grasp_file):
        print(f"Error: Grasp file not found: {grasp_file}")
        exit(1)
        
    data_dict_list = np.load(grasp_file, allow_pickle=True)
    
    if args.num >= len(data_dict_list):
        print(f"Error: Index {args.num} out of range (0-{len(data_dict_list)-1})")
        exit(1)
        
    data_dict = data_dict_list[args.num]
    qpos = data_dict['qpos']
    
    # Extract bimanual poses
    bimanual_pose = extract_bimanual_pose(qpos, device)
    bimanual_pose_st = None
    if 'qpos_st' in data_dict:
        qpos_st = data_dict['qpos_st']
        bimanual_pose_st = extract_bimanual_pose(qpos_st, device)

    # Create bimanual hand model
    bimanual_hand_model = BimanualHandModel(
        model_path='models',
        mesh_path='models/meshes',
        contact_points_path='models',
        penetration_points_path='models/penetration_points.json',
        device=device
    )

    # Create object model
    object_model = ObjectModel(
        data_root_path='../data/meshdata',
        batch_size_each=1,
        num_samples=2000, 
        device=device
    )
    object_model.initialize(args.object_code)
    object_model.object_scale_tensor = torch.tensor(data_dict['scale'], dtype=torch.float, device=device).reshape(1, 1)

    # Visualize
    plotly_data = []

    # Check if epoch data exists
    if 'epoch_qpos' in data_dict:
        # Visualize saved epochs only (not every 200th)
        epoch_qpos_dict = data_dict['epoch_qpos']
        epoch_energies_dict = data_dict.get('epoch_energies', {})
        
        # Sort epochs for proper visualization
        sorted_epochs = sorted(epoch_qpos_dict.keys())
        n_epochs = len(sorted_epochs)
        
        # Show all saved epochs (no interval filtering)
        for i, epoch in enumerate(sorted_epochs):
            # Calculate opacity (older epochs are more transparent)
            if n_epochs == 1:
                opacity = 1.0
            else:
                # Start from 0.2 and go to 1.0 (more visible progression)
                opacity = 0.2 + 0.8 * (i / (n_epochs - 1))
            
            # Extract bimanual pose for this epoch
            epoch_qpos = epoch_qpos_dict[epoch]
            epoch_bimanual_pose = extract_bimanual_pose(epoch_qpos, device)
            
            # Set parameters and get plotly data
            bimanual_hand_model.set_parameters(epoch_bimanual_pose.unsqueeze(0))
            
            # Color coding: start light and become darker (more saturated)
            if epoch == 0:
                # Initial pose in very light colors
                left_color = 'rgba(173,216,230,0.5)'  # Light blue
                right_color = 'rgba(255,182,193,0.5)'  # Light pink
            elif epoch == sorted_epochs[-1]:
                # Final pose in full colors
                left_color = 'rgba(0,0,255,1.0)'  # Full blue
                right_color = 'rgba(255,0,0,1.0)'  # Full red
            else:
                # Intermediate epochs: gradually increase color intensity
                color_intensity = 0.3 + 0.7 * (i / max(1, n_epochs - 1))
                left_color = f'rgba(0,0,255,{color_intensity})'  # Blue with varying alpha
                right_color = f'rgba(255,0,0,{color_intensity})'  # Red with varying alpha
            
            epoch_plotly = bimanual_hand_model.get_plotly_data(
                i=0, 
                opacity=opacity, 
                left_color=left_color, 
                right_color=right_color
            )
            plotly_data.extend(epoch_plotly)
            
        print(f"Visualized {n_epochs} saved epochs: {sorted_epochs}")
        
    else:
        # Fallback to original behavior (initial + final only)
        # Add initial pose (if available)
        if bimanual_pose_st is not None:
            bimanual_hand_model.set_parameters(bimanual_pose_st.unsqueeze(0))
            hand_st_plotly = bimanual_hand_model.get_plotly_data(i=0, opacity=0.5, 
                                                                 left_color='lightblue', right_color='lightcoral')
            plotly_data.extend(hand_st_plotly)

        # Add final pose
        bimanual_hand_model.set_parameters(bimanual_pose.unsqueeze(0))
        hand_en_plotly = bimanual_hand_model.get_plotly_data(i=0, opacity=1.0, 
                                                             left_color='blue', right_color='red')
        plotly_data.extend(hand_en_plotly)

    # Add object
    object_plotly = object_model.get_plotly_data(i=0, color='lightgreen', opacity=1)
    plotly_data.extend(object_plotly)

    # Create figure
    fig = go.Figure(plotly_data)
    
    # Add energy information if available
    if 'energy' in data_dict:
        if 'epoch_energies' in data_dict:
            # Show epoch-based energy progression
            epoch_energies_dict = data_dict['epoch_energies']
            sorted_epochs = sorted(epoch_energies_dict.keys())
            
            result_text = f'Bimanual Grasp {args.num} - Epoch Progression'
            result_text += f'<br>Epochs: {sorted_epochs[0]} → {sorted_epochs[-1]}'
            
            # Show initial and final energies
            if len(sorted_epochs) >= 2:
                initial_energy = epoch_energies_dict[sorted_epochs[0]]['total']
                final_energy = epoch_energies_dict[sorted_epochs[-1]]['total']
                result_text += f'<br>Energy: {round(initial_energy, 3)} → {round(final_energy, 3)}'
                
                # Show energy components for final epoch
                final_epoch_data = epoch_energies_dict[sorted_epochs[-1]]
                energy_components = []
                energy_components.append(f'E_fc: {round(final_epoch_data["E_fc"], 3)}')
                energy_components.append(f'E_dis: {round(final_epoch_data["E_dis"], 5)}')
                energy_components.append(f'E_pen: {round(final_epoch_data["E_pen"], 5)}')
                energy_components.append(f'E_spen: {round(final_epoch_data["E_spen"], 5)}')
                energy_components.append(f'E_joints: {round(final_epoch_data["E_joints"], 5)}')
                energy_components.append(f'E_bimpen: {round(final_epoch_data["E_bimpen"], 5)}')
                energy_components.append(f'E_vew: {round(final_epoch_data["E_vew"], 5)}')
                result_text += '<br>' + '  '.join(energy_components)
            elif len(sorted_epochs) == 1:
                # Only one epoch available
                energy_epoch = sorted_epochs[0]
                energy_value = epoch_energies_dict[energy_epoch]['total']
                result_text += f'<br>Energy at epoch {energy_epoch}: {round(energy_value, 3)}'
        else:
            # Original single energy display
            energy = data_dict['energy']
            result_text = f'Bimanual Grasp {args.num}  Total Energy: {round(energy, 3)}'
            
            # Add individual energy components if available
            energy_components = []
            if 'E_fc' in data_dict:
                energy_components.append(f'E_fc: {round(data_dict["E_fc"], 3)}')
            if 'E_dis' in data_dict:
                energy_components.append(f'E_dis: {round(data_dict["E_dis"], 5)}')
            if 'E_pen' in data_dict:
                energy_components.append(f'E_pen: {round(data_dict["E_pen"], 5)}')
            if 'E_spen' in data_dict:
                energy_components.append(f'E_spen: {round(data_dict["E_spen"], 5)}')
            if 'E_joints' in data_dict:
                energy_components.append(f'E_joints: {round(data_dict["E_joints"], 5)}')
            if 'E_bimpen' in data_dict:
                energy_components.append(f'E_bimpen: {round(data_dict["E_bimpen"], 5)}')
            if 'E_vew' in data_dict:
                energy_components.append(f'E_vew: {round(data_dict["E_vew"], 5)}')
                
            if energy_components:
                result_text += '<br>' + '  '.join(energy_components)
                
        fig.add_annotation(text=result_text, x=0.5, y=0.1, xref='paper', yref='paper',
                          bgcolor="white", bordercolor="black", borderwidth=1)

    # Set layout
    title_text = f'Bimanual Grasp Visualization - {args.object_code} (Index {args.num})'
    if 'epoch_qpos' in data_dict:
        epoch_qpos_dict = data_dict['epoch_qpos']
        sorted_epochs = sorted(epoch_qpos_dict.keys())
        if len(sorted_epochs) > 1:
            # Calculate actual interval from saved epochs
            intervals = [sorted_epochs[i+1] - sorted_epochs[i] for i in range(len(sorted_epochs)-1)]
            avg_interval = sum(intervals) / len(intervals)
            title_text += f' - Epochs: {sorted_epochs[0]} to {sorted_epochs[-1]} (Avg Interval: {avg_interval:.1f})'
        else:
            title_text += f' - Single Epoch: {sorted_epochs[0]}'
    
    fig.update_layout(
        scene_aspectmode='data',
        title=title_text,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )
    
    # Save result
    fig.write_html(args.output_file, auto_open=False)
    print(f"Bimanual grasp visualization saved to: {args.output_file}")
    print(f"Object: {args.object_code}")
    print(f"Grasp index: {args.num}")
    print(f"Scale: {data_dict['scale']}")
    
    # Print epoch information if available
    if 'epoch_qpos' in data_dict:
        epoch_qpos_dict = data_dict['epoch_qpos']
        sorted_epochs = sorted(epoch_qpos_dict.keys())
        print(f"Epochs visualized: {sorted_epochs} (Total: {len(sorted_epochs)} epochs)")
        if 'epoch_energies' in data_dict:
            epoch_energies_dict = data_dict['epoch_energies']
            energy_epochs = sorted(epoch_energies_dict.keys())
            if len(energy_epochs) >= 2:
                initial_energy = epoch_energies_dict[energy_epochs[0]]['total']
                final_energy = epoch_energies_dict[energy_epochs[-1]]['total']
                print(f"Energy progression (epochs {energy_epochs[0]}-{energy_epochs[-1]}): {round(initial_energy, 3)} → {round(final_energy, 3)}")
            elif len(energy_epochs) == 1:
                energy_epoch = energy_epochs[0]
                energy_value = epoch_energies_dict[energy_epoch]['total']
                print(f"Energy at epoch {energy_epoch}: {round(energy_value, 3)}")
            else:
                print("No energy data available for epochs")
    else:
        print("Note: Using fallback visualization (initial + final poses only)") 