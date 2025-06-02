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

# Add monkey patch for ObjectModel to support direct path initialization
def initialize_from_path(self, object_path, object_code):
    """
    Initialize Object Model with direct mesh path
    
    Parameters
    ----------
    object_path: str
        direct path to mesh file (.obj or .urdf)
    object_code: str
        object code for identification
    """
    import trimesh as tm
    import pytorch3d.structures
    import pytorch3d.ops
    from torchsdf import index_vertices_by_faces
    
    self.object_code_list = [object_code]
    self.object_scale_tensor = []
    self.object_mesh_list = []
    self.object_face_verts_list = []
    self.surface_points_tensor = []
    
    # Handle scale
    if self.fixed_scale is not None:
        scale_tensor = torch.full((self.batch_size_each,), self.fixed_scale, device=self.device)
        self.object_scale_tensor.append(scale_tensor)
    else:
        self.object_scale_tensor.append(self.scale_choice[torch.randint(0, self.scale_choice.shape[0], (self.batch_size_each, ), device=self.device)])
    
    # Load mesh from direct path
    # Handle both .urdf and .obj files
    if object_path.endswith('.urdf'):
        # For .urdf files, look for decomposed.obj in the same directory
        mesh_path = os.path.join(os.path.dirname(object_path), "decomposed.obj")
        if not os.path.exists(mesh_path):
            # If decomposed.obj doesn't exist, try to find any .obj file in the directory
            obj_files = [f for f in os.listdir(os.path.dirname(object_path)) if f.endswith('.obj')]
            if obj_files:
                mesh_path = os.path.join(os.path.dirname(object_path), obj_files[0])
            else:
                raise FileNotFoundError(f"No .obj file found in {os.path.dirname(object_path)}")
    else:
        mesh_path = object_path
    
    print(f"Loading mesh from: {mesh_path}")
    self.object_mesh_list.append(tm.load(mesh_path, force="mesh", process=False))
    
    object_verts = torch.Tensor(self.object_mesh_list[-1].vertices).to(self.device)
    object_faces = torch.Tensor(self.object_mesh_list[-1].faces).long().to(self.device)
    self.object_face_verts_list.append(index_vertices_by_faces(object_verts, object_faces))
    
    if self.num_samples != 0:
        vertices = torch.tensor(self.object_mesh_list[-1].vertices, dtype=torch.float, device=self.device)
        faces = torch.tensor(self.object_mesh_list[-1].faces, dtype=torch.float, device=self.device)
        mesh = pytorch3d.structures.Meshes(vertices.unsqueeze(0), faces.unsqueeze(0))
        dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(mesh, num_samples=100 * self.num_samples)
        surface_points = pytorch3d.ops.sample_farthest_points(dense_point_cloud, K=self.num_samples)[0][0]
        surface_points.to(dtype=float, device=self.device)
        self.surface_points_tensor.append(surface_points)
    
    self.object_scale_tensor = torch.stack(self.object_scale_tensor, dim=0)
    if self.num_samples != 0:
        self.surface_points_tensor = torch.stack(self.surface_points_tensor, dim=0).repeat_interleave(self.batch_size_each, dim=0)

# Add the method to ObjectModel class
ObjectModel.initialize_from_path = initialize_from_path

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
    # parser.add_argument('--object_code', type=str, default='core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03')
    # parser.add_argument('--object_code', type=str, default='sem-Bottle-437678d4bc6be981c8724d5673a063a6')
    parser.add_argument('--object_path', type=str, default='/home/dareum/cs586/data/meshdata_exp/bottle/coacd/decomposed.obj')
    parser.add_argument('--result_path', type=str, default='../data/bimanual_graspdata')
    parser.add_argument('--output_file', type=str, default='bimanual_grasp_result.html')
    args = parser.parse_args()

    # Extract object_code from path for compatibility
    # Example: /home/dareum/cs586/data/meshdata_one/shark/coacd/coacd.urdf -> shark
    object_code = os.path.basename(os.path.dirname(os.path.dirname(args.object_path)))
    print(f"Using object path: {args.object_path}")
    print(f"Extracted object_code: {object_code}")

    # Create list of grasp indices to visualize
    numlist = [0, 1, 2, 3, 10, 30, 50, 70, 90]
    # numlist = [0]

    device = 'cpu'

    # Create saved_html directory if it doesn't exist
    output_dir = 'saved_html'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    else:
        print(f"Directory already exists: {output_dir}")

    # Load bimanual grasp results
    grasp_file = os.path.join(args.result_path, f'bimanual_{object_code}.npy')
    if not os.path.exists(grasp_file):
        print(f"Error: Grasp file not found: {grasp_file}")
        exit(1)
        
    data_dict_list = np.load(grasp_file, allow_pickle=True)
    
    # Check if all indices in numlist are valid
    max_index = max(numlist)
    if max_index >= len(data_dict_list):
        print(f"Error: Index {max_index} out of range (0-{len(data_dict_list)-1})")
        exit(1)

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
        data_root_path='../data/meshdata_exp',
        batch_size_each=1,
        num_samples=2000, 
        device=device
    )
    
    # Initialize with direct path instead of object_code
    object_model.initialize_from_path(args.object_path, object_code)

    # Generate HTML files for each grasp in numlist
    for num in numlist:
        print(f"Generating visualization for grasp {num}...")
        
        data_dict = data_dict_list[num]
        qpos = data_dict['qpos']
        
        # Extract bimanual poses
        bimanual_pose = extract_bimanual_pose(qpos, device)
        bimanual_pose_st = None
        if 'qpos_st' in data_dict:
            qpos_st = data_dict['qpos_st']
            bimanual_pose_st = extract_bimanual_pose(qpos_st, device)

        # Set object scale for this specific grasp
        object_model.object_scale_tensor = torch.tensor(data_dict['scale'], dtype=torch.float, device=device).reshape(1, 1)

        # Visualize
        plotly_data = []

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
            energy = data_dict['energy']
            result_text = f'Bimanual Grasp {num}  Total Energy: {round(energy, 3)}'
            
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
        fig.update_layout(
            scene_aspectmode='data',
            title=f'Bimanual Grasp Visualization - {object_code} (Index {num})',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )
        
        # Create output filename with index
        base_name = args.output_file.split('.')[0]  # Remove .html extension
        output_filename = f"{base_name}_{num}.html"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save result
        fig.write_html(output_path, auto_open=False)
        print(f"Bimanual grasp visualization saved to: {output_path}")
        print(f"  Object: {object_code}")
        print(f"  Grasp index: {num}")
        print(f"  Scale: {data_dict['scale']}")
    
    print(f"\nAll visualizations completed! Generated {len(numlist)} HTML files.") 