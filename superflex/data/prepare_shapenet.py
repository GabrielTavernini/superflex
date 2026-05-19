import os
import argparse
import multiprocessing as mp

import torch
from torch_geometric.nn import fps
import numpy as np
from tqdm import tqdm
mp.set_start_method("spawn", force=True)

def process_model(model_path):
    try:
        out_path = os.path.join(model_path, "pointcloud_4096.npz")
        if os.path.exists(out_path):
            return ("skipped", model_path)
        pc_file = os.path.join(model_path, "pointcloud.npz")
        if not os.path.exists(pc_file):
            return ("missing", model_path)
        pc_data = np.load(pc_file)
        points = pc_data["points"]
        normals = pc_data["normals"]
        
        # FPS downsampling with torch_geometric
        points_tensor = torch.from_numpy(points)
        ratio = 4096 / points.shape[0]
        indices = fps(points_tensor, ratio=ratio)
        indices = indices[:4096].numpy() #make sure it's 4096
        points_4096 = points[indices]
        normals_4096 = normals[indices]
        
        points = np.asarray(points_4096)
        normals = np.asarray(normals_4096)
        np.savez(out_path, points=points, normals=normals)
        return ("ok", model_path)
    except Exception as e:
        return ("error", model_path, str(e))


def gather_model_paths(data_root):
    categories = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    model_paths = []
    for c in categories:
        category_path = os.path.join(data_root, c)
        models = [m for m in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, m))]
        for m in models:
            model_paths.append(os.path.join(category_path, m))
    return model_paths


def main():
    parser = argparse.ArgumentParser(description="Downsample ShapeNet pointclouds to 4096 using Open3D (parallel)")
    parser.add_argument("--data-root", default="data/ShapeNet", help="root folder containing ShapeNet categories")
    parser.add_argument("--jobs", type=int, default=64, help="number of parallel worker processes (default: cpu count)")
    args = parser.parse_args()

    model_paths = gather_model_paths(args.data_root)
    total = len(model_paths)
    if total == 0:
        print(f"No models found in {args.data_root}")
        return

    if args.jobs <= 1:
        results = []
        for mpth in tqdm(model_paths, desc="Processing models"):
            results.append(process_model(mpth))
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=args.jobs) as pool:
            results = []
            for r in tqdm(pool.imap_unordered(process_model, model_paths), total=total, desc="Processing models"):
                results.append(r)

    # simple summary
    counts = {"ok": 0, "skipped": 0, "missing": 0, "error": 0}
    for r in results:
        if r[0] in counts:
            counts[r[0]] += 1
        else:
            counts["error"] += 1
    print(f"Done. processed={counts['ok']} skipped={counts['skipped']} missing={counts['missing']} errors={counts['error']}")


if __name__ == "__main__":
    main()
