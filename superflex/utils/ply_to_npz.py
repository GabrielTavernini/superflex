#I want to call it in this way:
#python ply_to_npz.py --input_path $(OBJECTS_SCENE_DIR) --scene_name $(SCENE_NAME)

import os
import argparse
import numpy as np
import trimesh
from tqdm import tqdm
import shutil
import glob
import multiprocessing
from functools import partial
import logging
from sklearn.cluster import DBSCAN


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_dbscan_clusters(points, eps=0.2, min_samples=20):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clustering = dbscan.fit(points)
    return clustering.labels_


def process_ply_to_npz(input_path, output_path, ply_file, eps=0.2, min_samples=50):
    try:
        mesh = trimesh.load(os.path.join(input_path, ply_file))
        # no need to normalize, only get points
        mesh_np = np.array(mesh.vertices, dtype=np.float32)
        # Apply DBSCAN to filter out noise points (label == -1)
        labels = get_dbscan_clusters(mesh_np, eps=eps, min_samples=min_samples)
        mesh_np = mesh_np[labels != -1]
        logger.info(f"{ply_file}: {len(labels)} -> {len(mesh_np)} points after DBSCAN filtering")
        if len(mesh_np) == 0:
            logger.warning(f"Skipping {ply_file}: 0 points remaining after DBSCAN filtering")
            return
        np.savez_compressed(
            os.path.join(output_path, f"{os.path.splitext(ply_file)[0]}.npz"),
            points=mesh_np
        )
    except Exception as e:
        logger.error(f"Error processing {ply_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PLY files to NPZ format")
    parser.add_argument('--input_path', type=str, required=True, help='Path to the directory containing PLY files')
    parser.add_argument('--scene_name', type=str, required=True, help='Name of the scene (used for output directory)')
    args = parser.parse_args()

    input_path = args.input_path
    scene_name = args.scene_name

    output_path = os.path.join('data', scene_name, 'pc_gt')
    os.makedirs(output_path, exist_ok=True)

    ply_files = [f for f in os.listdir(input_path) if f.endswith('.ply')]
    logger.info(f"Found {len(ply_files)} PLY files in {input_path}")

    # Use multiprocessing to speed up the conversion
    num_workers = min(multiprocessing.cpu_count(), 8)
    with multiprocessing.Pool(num_workers) as pool:
        func = partial(process_ply_to_npz, input_path, output_path, eps=0.2, min_samples=50)
        list(tqdm(pool.imap(func, ply_files), total=len(ply_files)))

    logger.info(f"Conversion completed. NPZ files saved in {output_path}")