"""
Last modified date: 2023.02.23
Author: Jialiang Zhang
Description: visualize grasp result with animation using plotly.graph_objects
Enhanced: Added trajectory animation and video export
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
from plotly.offline import plot
import cv2
from PIL import Image
import tempfile
import shutil

from utils.hand_model import HandModel
from utils.object_model import ObjectModel

translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
rot_names = ['WRJRx', 'WRJRy', 'WRJRz']
joint_names = [
    'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
    'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
    'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
    'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
    'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
]

def interpolate_poses(qpos_st, qpos, num_frames=30):
    """Create interpolated poses between start and end positions"""
    interpolated_poses = []
    
    for i in range(num_frames):
        alpha = i / (num_frames - 1)  # 0 to 1
        interpolated_qpos = {}
        
        # Interpolate translations
        for name in translation_names:
            interpolated_qpos[name] = qpos_st[name] * (1 - alpha) + qpos[name] * alpha
        
        # Interpolate rotations (euler angles)
        for name in rot_names:
            interpolated_qpos[name] = qpos_st[name] * (1 - alpha) + qpos[name] * alpha
        
        # Interpolate joint angles
        for name in joint_names:
            interpolated_qpos[name] = qpos_st[name] * (1 - alpha) + qpos[name] * alpha
            
        interpolated_poses.append(interpolated_qpos)
    
    return interpolated_poses

def qpos_to_hand_pose(qpos, device):
    """Convert qpos dict to hand_pose tensor"""
    rot = np.array(transforms3d.euler.euler2mat(*[qpos[name] for name in rot_names]))
    rot = rot[:, :2].T.ravel().tolist()
    hand_pose = torch.tensor([qpos[name] for name in translation_names] + rot + [qpos[name] for name in joint_names], dtype=torch.float, device=device)
    return hand_pose

def create_animated_visualization(hand_model, object_model, interpolated_poses, data_dict, args, device):
    """Create animated plotly figure"""
    frames = []
    
    for i, qpos in enumerate(interpolated_poses):
        hand_pose = qpos_to_hand_pose(qpos, device)
        hand_model.set_parameters(hand_pose.unsqueeze(0))
        
        # Get hand and object data for this frame
        hand_data = hand_model.get_plotly_data(i=0, opacity=1, color='lightblue', with_contact_points=False)
        object_data = object_model.get_plotly_data(i=0, color='lightgreen', opacity=1)
        
        frame = go.Frame(
            data=hand_data + object_data,
            name=f'frame_{i}',
            layout=go.Layout(title=f'Frame {i}')
        )
        frames.append(frame)
    
    # Create initial figure with first frame
    hand_pose_0 = qpos_to_hand_pose(interpolated_poses[0], device)
    hand_model.set_parameters(hand_pose_0.unsqueeze(0))
    initial_hand_data = hand_model.get_plotly_data(i=0, opacity=1, color='lightblue', with_contact_points=False)
    initial_object_data = object_model.get_plotly_data(i=0, color='lightgreen', opacity=1)
    
    fig = go.Figure(
        data=initial_hand_data + initial_object_data,
        frames=frames
    )
    
    # Add animation controls
    fig.update_layout(
        scene_aspectmode='data',
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {
                    'label': '▶️ Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 100, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 50}
                    }]
                },
                {
                    'label': '⏸️ Pause',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
            ]
        }],
        sliders=[{
            'steps': [
                {
                    'args': [[f'frame_{i}'], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }],
                    'label': f'{i}',
                    'method': 'animate'
                }
                for i in range(len(interpolated_poses))
            ],
            'active': 0,
            'currentvalue': {'prefix': 'Frame: '},
            'len': 0.9,
            'x': 0.1,
            'xanchor': 'left',
            'y': 0,
            'yanchor': 'top'
        }]
    )
    
    # Add energy information if available
    if 'energy' in data_dict:
        E_fc = round(data_dict['E_fc'], 3)
        E_dis = round(data_dict['E_dis'], 5)
        E_pen = round(data_dict['E_pen'], 5)
        E_spen = round(data_dict['E_spen'], 5)
        E_joints = round(data_dict['E_joints'], 5)
        result = f'Index {args.num}  E_fc {E_fc}  E_dis {E_dis}  E_pen {E_pen}'
        fig.add_annotation(text=result, x=0.5, y=0.95, xref='paper', yref='paper')
    
    return fig

def create_video_from_frames(hand_model, object_model, interpolated_poses, output_path, device, fps=10):
    """Create video file from interpolated frames"""
    try:
        import kaleido  # For plotly image export
    except ImportError:
        print("Warning: kaleido not installed. Cannot create video. Install with: pip install kaleido")
        return
    
    temp_dir = tempfile.mkdtemp()
    frame_paths = []
    
    try:
        for i, qpos in enumerate(interpolated_poses):
            hand_pose = qpos_to_hand_pose(qpos, device)
            hand_model.set_parameters(hand_pose.unsqueeze(0))
            
            # Create figure for this frame
            hand_data = hand_model.get_plotly_data(i=0, opacity=1, color='lightblue', with_contact_points=False)
            object_data = object_model.get_plotly_data(i=0, color='lightgreen', opacity=1)
            
            fig = go.Figure(data=hand_data + object_data)
            fig.update_layout(
                scene_aspectmode='data',
                showlegend=False,
                scene=dict(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False)
                )
            )
            
            # Save frame as image
            frame_path = os.path.join(temp_dir, f'frame_{i:04d}.png')
            fig.write_image(frame_path, width=800, height=600)
            frame_paths.append(frame_path)
            print(f"Generated frame {i+1}/{len(interpolated_poses)}")
        
        # Create video from frames
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (800, 600))
        
        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            video_writer.write(frame)
        
        video_writer.release()
        print(f"Video saved to: {output_path}")
        
    finally:
        # Clean up temporary files
        shutil.rmtree(temp_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--object_code', type=str, default='core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03')
    parser.add_argument('--num', type=int, default=0)
    parser.add_argument('--result_path', type=str, default='../data/graspdata')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (auto-generated if not specified)')
    parser.add_argument('--num_frames', type=int, default=30, help='Number of interpolation frames')
    parser.add_argument('--fps', type=int, default=10, help='Video frames per second')
    args = parser.parse_args()

    device = 'cpu'

    # Auto-generate output directory if not specified
    if args.output_dir is None:
        result_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'result')
        args.output_dir = os.path.join(result_dir, f'{args.object_code}_grasp_{args.num}')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"📁 Output directory: {args.output_dir}")
    
    # Load results
    data_dict = np.load(os.path.join(args.result_path, args.object_code + '.npy'), allow_pickle=True)[args.num]
    qpos = data_dict['qpos']
    
    # Check if we have initial pose
    if 'qpos_st' not in data_dict:
        print("Warning: No initial pose (qpos_st) found. Using static visualization only.")
        # Fall back to static visualization
        rot = np.array(transforms3d.euler.euler2mat(*[qpos[name] for name in rot_names]))
        rot = rot[:, :2].T.ravel().tolist()
        hand_pose = torch.tensor([qpos[name] for name in translation_names] + rot + [qpos[name] for name in joint_names], dtype=torch.float, device=device)
    else:
        qpos_st = data_dict['qpos_st']
        
        # Create interpolated poses
        print(f"Creating {args.num_frames} interpolated frames...")
        interpolated_poses = interpolate_poses(qpos_st, qpos, args.num_frames)

    # Initialize models
    hand_model = HandModel(
        mjcf_path='mjcf/shadow_hand_wrist_free.xml',
        mesh_path='mjcf/meshes',
        contact_points_path='mjcf/contact_points.json',
        penetration_points_path='mjcf/penetration_points.json',
        device=device
    )

    object_model = ObjectModel(
        data_root_path='../data/meshdata',
        batch_size_each=1,
        num_samples=2000, 
        device=device
    )
    object_model.initialize(args.object_code)
    object_model.object_scale_tensor = torch.tensor(data_dict['scale'], dtype=torch.float, device=device).reshape(1, 1)

    if 'qpos_st' in data_dict:
        # Create animated visualization
        print("Creating animated HTML...")
        animated_fig = create_animated_visualization(hand_model, object_model, interpolated_poses, data_dict, args, device)
        
        # Save animated HTML
        animated_html_path = os.path.join(args.output_dir, 'grasp_animated.html')
        animated_fig.write_html(animated_html_path, auto_open=False)
        print(f"Animated HTML saved to: {animated_html_path}")
        
        # Create video
        print("Creating video...")
        video_path = os.path.join(args.output_dir, 'grasp_video.mp4')
        create_video_from_frames(hand_model, object_model, interpolated_poses, video_path, device, args.fps)
    
    # Create static HTML (original version)
    print("Creating static HTML...")
    hand_pose_final = qpos_to_hand_pose(qpos, device)
    hand_model.set_parameters(hand_pose_final.unsqueeze(0))
    
    if 'qpos_st' in data_dict:
        hand_pose_initial = qpos_to_hand_pose(qpos_st, device)
        hand_model.set_parameters(hand_pose_initial.unsqueeze(0))
        hand_st_plotly = hand_model.get_plotly_data(i=0, opacity=0.5, color='lightblue', with_contact_points=False)
    else:
        hand_st_plotly = []
    
    hand_model.set_parameters(hand_pose_final.unsqueeze(0))
    hand_en_plotly = hand_model.get_plotly_data(i=0, opacity=1, color='lightblue', with_contact_points=False)
    object_plotly = object_model.get_plotly_data(i=0, color='lightgreen', opacity=1)
    
    static_fig = go.Figure(hand_st_plotly + hand_en_plotly + object_plotly)
    if 'energy' in data_dict:
        E_fc = round(data_dict['E_fc'], 3)
        E_dis = round(data_dict['E_dis'], 5)
        E_pen = round(data_dict['E_pen'], 5)
        E_spen = round(data_dict['E_spen'], 5)
        E_joints = round(data_dict['E_joints'], 5)
        result = f'Index {args.num}  E_fc {E_fc}  E_dis {E_dis}  E_pen {E_pen}'
        static_fig.add_annotation(text=result, x=0.5, y=0.1, xref='paper', yref='paper')
    
    static_fig.update_layout(scene_aspectmode='data')
    static_html_path = os.path.join(args.output_dir, 'grasp_static.html')
    static_fig.write_html(static_html_path, auto_open=False)
    print(f"Static HTML saved to: {static_html_path}")
    
    print(f"\n✅ All files saved to: {args.output_dir}/")
    print(f"   - Static visualization: grasp_static.html")
    if 'qpos_st' in data_dict:
        print(f"   - Animated visualization: grasp_animated.html")
        print(f"   - Video: grasp_video.mp4") 