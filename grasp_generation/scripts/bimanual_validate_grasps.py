"""
Last modified date: 2024.12.16
Author: Bimanual Extension
Description: Validate bimanual grasps using Isaac Gym simulation
"""

import os
import sys
sys.path.append(os.path.realpath('.'))

from utils.isaac_validator import IsaacValidator
import argparse
import torch
import numpy as np
import transforms3d
from utils.bimanual_hand_model import BimanualHandModel
from utils.object_model import ObjectModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--val_batch', default=100, type=int)  # Smaller batch for bimanual
    parser.add_argument('--mesh_path', default="../data/meshdata", type=str)
    parser.add_argument('--grasp_file', default='../data/bimanual_graspdata', help='Path to bimanual grasp directory or .npy file')
    parser.add_argument('--result_path', default='../data/bimanual_dataset', help='Output directory for validation results')
    parser.add_argument('--object_code', default=None, help='Object code identifier (if None, processes all files)')
    # if index is received, then the debug mode is on
    parser.add_argument('--index', type=int)
    parser.add_argument('--no_force', action='store_true')
    parser.add_argument('--thres_cont', default=0.001, type=float)
    parser.add_argument('--dis_move', default=0.001, type=float)
    parser.add_argument('--grad_move', default=500, type=float)
    parser.add_argument('--penetration_threshold', default=0.001, type=float)

    args = parser.parse_args()

    # Joint and pose names for bimanual hands
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

    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.makedirs(args.result_path, exist_ok=True)

    # Determine files to process
    if args.object_code:
        # Process single object
        grasp_file = os.path.join(args.grasp_file, f'bimanual_{args.object_code}.npy')
        if not os.path.exists(grasp_file):
            print(f"Error: Grasp file not found: {grasp_file}")
            exit(1)
        files_to_process = [(grasp_file, args.object_code)]
    else:
        # Process all bimanual grasp files in directory
        if os.path.isfile(args.grasp_file):
            # Single file provided
            object_code = os.path.basename(args.grasp_file).replace('bimanual_', '').replace('.npy', '')
            files_to_process = [(args.grasp_file, object_code)]
        else:
            # Directory provided
            if not os.path.exists(args.grasp_file):
                print(f"Error: Directory not found: {args.grasp_file}")
                exit(1)
            grasp_files = [f for f in os.listdir(args.grasp_file) if f.startswith('bimanual_') and f.endswith('.npy')]
            files_to_process = []
            for f in grasp_files:
                object_code = f.replace('bimanual_', '').replace('.npy', '')
                files_to_process.append((os.path.join(args.grasp_file, f), object_code))

    if not files_to_process:
        print("No bimanual grasp files found to process.")
        exit(1)

    print(f"Found {len(files_to_process)} files to validate")

    total_successful = 0
    total_grasps = 0

    for grasp_file_path, object_code in files_to_process:
        print(f"\nProcessing: {object_code}")
        
        # Load grasp data
        data_dict = np.load(grasp_file_path, allow_pickle=True)
        batch_size = data_dict.shape[0]
        
        if batch_size == 0:
            print(f"  No grasps found in {object_code}")
            continue

        # Process bimanual grasp data - only validate left hand for now
        # (Isaac validator expects single hand data)
        left_hand_poses = []
        left_rotations = []
        left_translations = []
        E_pen_array = []
        scale_array = []

        for i in range(batch_size):
            qpos = data_dict[i]['qpos']
            scale = data_dict[i]['scale']
            
            # Extract left hand data only (for compatibility with IsaacValidator)
            left_rot = [qpos[name] for name in left_rot_names]
            left_rot = transforms3d.euler.euler2quat(*left_rot)
            left_rotations.append(left_rot)
            
            left_translations.append(np.array([qpos[name] for name in left_translation_names]))
            left_hand_poses.append(np.array([qpos[name] for name in left_joint_names]))
            scale_array.append(scale)
            
            # Use combined energy for threshold (if available)
            if 'E_pen' in data_dict[i]:
                E_pen_array.append(data_dict[i]['E_pen'])
            elif 'energy' in data_dict[i]:
                E_pen_array.append(data_dict[i]['energy'])
            else:
                E_pen_array.append(0.0)  # Default value

        E_pen_array = np.array(E_pen_array)

        # Create Isaac simulator
        sim = IsaacValidator(gpu=args.gpu)
        if args.index is not None:
            sim = IsaacValidator(gpu=args.gpu, mode="gui")

        if args.index is not None:
            # Debug mode - single grasp
            sim.set_asset("open_ai_assets", "hand/shadow_hand.xml",
                         os.path.join(args.mesh_path, object_code, "coacd"), "coacd.urdf")
            index = args.index
            if index < batch_size:
                sim.add_env_single(left_rotations[index], left_translations[index], left_hand_poses[index],
                                  scale_array[index], 0)
                result = sim.run_sim()
                print(f"Debug result for grasp {index}: {result}")
            else:
                print(f"Error: Index {index} out of range (0-{batch_size-1})")
        else:
            # Batch validation
            simulated = np.zeros(batch_size, dtype=np.bool8)
            offset = 0
            result = []
            
            for batch in range((batch_size + args.val_batch - 1) // args.val_batch):
                offset_ = min(offset + args.val_batch, batch_size)
                sim.set_asset("open_ai_assets", "hand/shadow_hand.xml",
                             os.path.join(args.mesh_path, object_code, "coacd"), "coacd.urdf")
                             
                for index in range(offset, offset_):
                    sim.add_env(left_rotations[index], left_translations[index], left_hand_poses[index],
                               scale_array[index])
                               
                batch_results = sim.run_sim()
                result.extend(batch_results)
                sim.reset_simulator()
                offset = offset_

            # Process results
            for i in range(batch_size):
                # Each grasp has 6 tests (different orientations)
                start_idx = i * 6
                end_idx = (i + 1) * 6
                if end_idx <= len(result):
                    simulated[i] = np.array(sum(result[start_idx:end_idx]) == 6)

            # Apply energy threshold
            estimated = E_pen_array < args.penetration_threshold
            valid = simulated & estimated

            successful_grasps = valid.sum()
            total_successful += successful_grasps
            total_grasps += batch_size

            print(f"  {object_code}: {successful_grasps}/{batch_size} successful ({successful_grasps/batch_size:.2%})")
            print(f"    Estimated: {estimated.sum()}/{batch_size}")
            print(f"    Simulated: {simulated.sum()}/{batch_size}")

            # Save validated grasps
            if successful_grasps > 0:
                validated_grasps = []
                for i in range(batch_size):
                    if valid[i]:
                        validated_grasps.append({
                            "qpos": data_dict[i]["qpos"],
                            "scale": data_dict[i]["scale"]
                        })
                        
                output_file = os.path.join(args.result_path, f'{object_code}.npy')
                np.save(output_file, validated_grasps, allow_pickle=True)
                print(f"    Saved {len(validated_grasps)} validated grasps to: {output_file}")

        sim.destroy()

    # Overall statistics
    if total_grasps > 0:
        overall_success_rate = total_successful / total_grasps
        print(f"\nOverall Results:")
        print(f"Total objects: {len(files_to_process)}")
        print(f"Total grasps: {total_grasps}")
        print(f"Total successful: {total_successful}")
        print(f"Overall success rate: {overall_success_rate:.2%}")
    else:
        print("\nNo grasps processed.") 