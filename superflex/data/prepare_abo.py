import argparse
import glob
import os
import multiprocessing
import numpy as np
import trimesh
import torch
from torch_geometric.nn import fps
from tqdm import tqdm
multiprocessing.set_start_method("spawn", force=True)

def process_file(args):
    input_file, output_root, n_points_surf, n_points_vol = args
    
    try:
        # Determine output directory
        file_name = os.path.basename(input_file)
        obj_id = os.path.splitext(file_name)[0]
        output_dir = os.path.join(output_root, obj_id)
        
        pc_path = os.path.join(output_dir, 'pointcloud.npz')
        points_path = os.path.join(output_dir, 'points.npz')
        path_4096 = os.path.join(output_dir, 'pointcloud_4096.npz')
        
        if os.path.exists(pc_path) and os.path.exists(points_path) and os.path.exists(path_4096):
            return # Skip if already exists
        
        # Load mesh
        # Handle Scene vs Mesh
        mesh_or_scene = trimesh.load(input_file)
        if isinstance(mesh_or_scene, trimesh.Scene):
            if len(mesh_or_scene.geometry) == 0:
                print(f"Empty scene {input_file}")
                return
            # Concatenate all geometries
            mesh = trimesh.util.concatenate(tuple(mesh_or_scene.geometry.values()))
        else:
            mesh = mesh_or_scene

        if not isinstance(mesh, trimesh.Trimesh):
            print(f"Not a mesh: {input_file} {type(mesh)}")
            return
        
        if mesh.bounds is None: return
        center = mesh.bounding_box.centroid
        max_extent = np.max(mesh.extents)

        # 1. Surface points
        points_surface, face_indices = trimesh.sample.sample_surface(mesh, n_points_surf)
        normals_surface = mesh.face_normals[face_indices]

        # 2. Volume points
        # Keep relative resolution (1% of bounding box + padding)
        pitch = (max_extent * 1.1) / 100
        voxel_grid = mesh.voxelized(pitch=pitch)
        voxel_grid = voxel_grid.fill()

        # Random uniform points in bbox + 5% padding
        half_size = (max_extent * 1.1) / 2
        min_bound = center - half_size
        max_bound = center + half_size
        points_vol = np.random.uniform(min_bound, max_bound, (n_points_vol, 3))
        
        # Check occupancy
        occupancies = voxel_grid.is_filled(points_vol)
        packed_occ = np.packbits(occupancies)
        
        # FPS downsampling with torch_geometric
        points_tensor = torch.from_numpy(points_surface)
        ratio = 4096 / points_surface.shape[0]
        indices = fps(points_tensor, ratio=ratio)
        indices = indices[:4096].numpy() #make sure it's 4096
        points_4096 = points_surface[indices]
        normals_4096 = normals_surface[indices]

        path_4096 = os.path.join(output_dir, 'pointcloud_4096.npz')
        
        # Save to files
        os.makedirs(output_dir, exist_ok=True)
        np.savez_compressed(pc_path, points=points_surface.astype(np.float32), normals=normals_surface.astype(np.float32))
        np.savez_compressed(points_path, points=points_vol.astype(np.float32), occupancies=packed_occ)
        np.savez_compressed(path_4096, points=points_4096.astype(np.float32), normals=normals_4096.astype(np.float32))
        
    except Exception as e:
        print(f"Error processing {input_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Process ABO dataset.")
    parser.add_argument("--input_dir", type=str, default="data/ABO/raw-complete/3dmodels/original", help="Path to input GLB files")
    parser.add_argument("--output_dir", type=str, default="data/ABO/processed-complete", help="Path to output directory")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of worker processes")
    parser.add_argument("--n_surf", type=int, default=100000, help="Number of surface points")
    parser.add_argument("--n_vol", type=int, default=100000, help="Number of volume points")
    
    args = parser.parse_args()
    
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    print(f"Searching for glb files in {input_dir}...")
    files = glob.glob(os.path.join(input_dir, "**/*.glb"), recursive=True)
    if not files:
        files = glob.glob(os.path.join(input_dir, "*.glb"))
    files.sort()

    print(f"Found {len(files)} files.")
    
    if not files:
        return

    os.makedirs(output_dir, exist_ok=True)
    
    tasks = [(f, output_dir, args.n_surf, args.n_vol) for f in files]
    
    with multiprocessing.Pool(args.num_workers, maxtasksperchild=10) as pool:
        list(tqdm(pool.imap_unordered(process_file, tasks), total=len(tasks)))

if __name__ == "__main__":
    main()
