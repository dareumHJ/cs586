# CS586 Team 2

## Overview
This repository contains the codebase for the CS586 Team 2 project, implementing algorithms synthesizing bimanual dexterous grasping poses and its dataset

## Repository Structure
+-- asset_process
+-- grasp_generation
+-- data
|  +-- meshdata     # Linked to the output folder of asset processing
|  +-- experiments
|  +-- graspdata
|  +-- dataset
+-- thirdparty
|  +-- pytorch_kinematics
|  +-- CoACD
|  +-- ManifoldPlus
|  +-- TorchSDF

### Dataset

### Generation Code

### Utility Functions

## Getting Started
### Quick Example
'''conda create -n your_env python=3.7
conda activate your_env

# for quick example, cpu version is OK.
conda install -c conda-forge pytorch3d
conda install ipykernel
pip install transforms3d
pip install trimesh
pip install pyyaml
pip install lxml

cd thirdparty/pytorch_kinematics
pip install -e .'''

### Grasp Generation
'''conda create -n generation python=3.10
conda activate generation

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
conda install numpy=1.26.4
conda install pytorch3d -c pytorch -c conda-forge
conda install transforms3d -c pytorch -c conda-forge
conda install trimesh -c pytorch -c conda-forge
conda install plotly

pip install urdf_parser_py
pip install scipy
pip install networkx
conda install rtree
conda install six

cd TorchSDF
export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9
export CUDA_HOME=/usr/local/cuda-11.3 && bash install.sh

cd thirdparty/pytorch_kinematics
pip install -e .

export CUDA_VISIBLE_DEVICES=0
python scripts/generate_grasps.py --all'''

The codes used in the repository was initially cloned from DexGraspNet!
