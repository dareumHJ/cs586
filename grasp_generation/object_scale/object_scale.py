import trimesh
import argparse
import os

def scale_obj_file(obj_code, scale_factor):
    """
    Loads an OBJ file, scales it, and saves it as a new OBJ file.

    Parameters
    ----------
    input_obj_path : str
        Path to the input OBJ file.
    output_obj_path : str
        Path to save the scaled output OBJ file.
    scale_factor : float or list/tuple of 3 floats
        Scaling factor. If a single float, uniform scaling is applied.
        If a list/tuple of 3 floats (e.g., [sx, sy, sz]), non-uniform scaling is applied.
    """
    # 입력 파일 존재 여부 확인
    input_obj_path = f'../data/meshdata/{obj_code}/coacd/decomposed.obj'
    output_obj_path = f'../data/meshdata/{obj_code}/coacd/decomposed.obj'
    if not os.path.exists(input_obj_path):
        print(f"Error: Input file not found at {input_obj_path}")
        return

    # 메쉬 로드
    try:
        mesh = trimesh.load_mesh(input_obj_path, process=False) # process=False로 원본 유지 시도
        print(f"Successfully loaded mesh from {input_obj_path}")
    except Exception as e:
        print(f"Error loading mesh from {input_obj_path}: {e}")
        return

    # 메쉬 스케일링
    try:
        if isinstance(mesh, trimesh.Trimesh): # 단일 메쉬인 경우
            mesh.apply_scale(scale_factor)
        elif isinstance(mesh, trimesh.Scene): # 여러 메쉬가 씬으로 로드된 경우
            # 씬의 모든 지오메트리에 스케일 적용
            for geom_name in mesh.geometry:
                mesh.geometry[geom_name].apply_scale(scale_factor)
        else:
            print(f"Loaded object is not a Trimesh or Scene. Type: {type(mesh)}")
            return
            
        print(f"Applied scale factor: {scale_factor}")
    except Exception as e:
        print(f"Error applying scale: {e}")
        return

    # 스케일된 메쉬를 새 파일로 저장
    try:
        # 출력 디렉토리 생성 (필요한 경우)
        output_dir = os.path.dirname(output_obj_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        mesh.export(output_obj_path)
        print(f"Successfully saved scaled mesh to {output_obj_path}")
    except Exception as e:
        print(f"Error exporting mesh to {output_obj_path}: {e}")
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Scale an OBJ file.")
    parser.add_argument("obj_code", type=str, help="input object code")
    parser.add_argument("--scale", type=float, nargs='+', default=[1.0], 
                        help="Scaling factor(s). Single value for uniform scaling (e.g., 2.0), "
                             "or three values for non-uniform scaling (e.g., 1.0 2.0 0.5 for X Y Z).")

    args = parser.parse_args()

    # 스케일 인자 처리
    if len(args.scale) == 1:
        scale_value = args.scale[0]
    elif len(args.scale) == 3:
        scale_value = args.scale
    else:
        print("Error: --scale argument must be a single value or three values.")
        exit(1)

    scale_obj_file(args.obj_code, scale_value)