import os
import torch
import numpy as np
from tqdm import tqdm
from superflex.utils.predictions_handler import PredictionHandler
import random
import hydra
from omegaconf import DictConfig
import csv
from omegaconf import OmegaConf

from superflex.utils.evaluation import eval_mesh, build_dataloader, compute_ious_sdf_from_handler


def main(cfg: DictConfig):
    print("\n========== SuperFlex Evaluation ==========")
    print("Config:\n" + OmegaConf.to_yaml(cfg))

    dataset = cfg.dataloader.get("dataset", "shapenet")
    source_folder = cfg.get("source_folder", dataset)
    split = cfg.get(dataset).get("split", "test")
    use_occlusion = cfg.get("use_occlusion", False)

    print(
        f"Evaluating on {dataset} {split}{'(occlusion mode)' if use_occlusion else ''}..."
    )
    input_npz = f"data/output_npz/{source_folder}/{dataset}_{split}.npz"
    output_dir = os.path.join("data", "output_npz", source_folder, "results")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {input_npz}...")
    pred_handler = PredictionHandler.from_npz(input_npz)

    # Build dataloader from config (expects cfg.dataloader.*)
    dataloader = build_dataloader(cfg)
    assert (
        dataloader.dataset.normalize == False
    ), "Eval dataset should not be normalized."
    # Sanity-check: ensure prediction names align with dataset model_ids
    dataset_names = [m["model_id"] for m in dataloader.dataset.models]
    assert np.array_equal(
        pred_handler.names, dataset_names
    ), f"Object order differs between pred_handler and dataset."
    print(
        f"Loaded {len(dataloader.dataset)} objects from all categories out of {pred_handler.scale.shape[0]}."
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Store aggregated metrics (all metrics for both modes)
    aggregated_metrics = {
        "chamfer-L1": 0.0,
        "chamfer-L2": 0.0,
        "iou": 0.0,
        "f-score": 0.0,
        "f-score-15": 0.0,
        "f-score-20": 0.0,
        "normals": 0.0,
        "normals completeness": 0.0,
        "normals accuracy": 0.0,
        "completeness": 0.0,
        "accuracy": 0.0,
        "completeness2": 0.0,
        "accuracy2": 0.0,
        "num_primitives": 0.0,
        "count": 0,
    }

    # Store per-object metrics for ranking
    object_metrics = []
    for batch in tqdm(dataloader, desc="Processing batches"):
        batch_indices = batch["idx"]

        # Move tensors to device (differ based on occlusion mode)
        if use_occlusion:
            points = batch["abo_points"].to(device)
            normals = batch["abo_normals"].to(device)
        else:
            points = batch["points"].to(device)
            normals = batch["normals"].to(device)

        points_iou = batch["points_iou"].to(device)
        occupancies = batch["occupancies"].to(device)

        # Compute IoU using SDF
        with torch.no_grad():
            batch_ious = compute_ious_sdf_from_handler(
                pred_handler, batch_indices, points_iou, occupancies, device=device
            )

        # Evaluate mesh
        for b_idx in range(len(batch_indices)):
            idx = batch_indices[b_idx]
            try:
                mesh = pred_handler.get_mesh(idx, resolution=100, colors=False)
            except Exception as e:
                print(f"Error generating mesh for object {idx}: {e}")

            try:
                gt_pc = points[b_idx].cpu().numpy()
                gt_normal = normals[b_idx].cpu().numpy()
                out_dict_cur = eval_mesh(mesh, gt_pc, gt_normal, None, None)
                out_dict_cur["iou"] = (
                    float(batch_ious[b_idx]) if batch_ious is not None else 0.0
                )
            except Exception as e:
                print(f"Eval mesh failed: {e}")

            num_prim = (pred_handler.exist[idx] > 0.5).sum()
            aggregated_metrics["chamfer-L1"] += out_dict_cur["chamfer-L1"]
            aggregated_metrics["chamfer-L2"] += out_dict_cur["chamfer-L2"]
            aggregated_metrics["iou"] += out_dict_cur.get("iou", 0.0)
            aggregated_metrics["f-score"] += out_dict_cur.get("f-score", 0.0)
            aggregated_metrics["f-score-15"] += out_dict_cur.get("f-score-15", 0.0)
            aggregated_metrics["f-score-20"] += out_dict_cur.get("f-score-20", 0.0)
            aggregated_metrics["normals"] += out_dict_cur.get("normals", 0.0)
            aggregated_metrics["normals completeness"] += out_dict_cur.get(
                "normals completeness", 0.0
            )
            aggregated_metrics["normals accuracy"] += out_dict_cur.get(
                "normals accuracy", 0.0
            )
            aggregated_metrics["completeness"] += out_dict_cur.get("completeness", 0.0)
            aggregated_metrics["accuracy"] += out_dict_cur.get("accuracy", 0.0)
            aggregated_metrics["completeness2"] += out_dict_cur.get(
                "completeness2", 0.0
            )
            aggregated_metrics["accuracy2"] += out_dict_cur.get("accuracy2", 0.0)
            aggregated_metrics["num_primitives"] += num_prim
            aggregated_metrics["count"] += 1

            object_metrics.append(
                {
                    "index": idx.item(),
                    "name": pred_handler.names[idx],
                    "chamfer-L1": out_dict_cur["chamfer-L1"],
                    "chamfer-L2": out_dict_cur["chamfer-L2"],
                    "iou": out_dict_cur.get("iou", 0.0),
                    "f-score": out_dict_cur.get("f-score", 0.0),
                    "f-score-15": out_dict_cur.get("f-score-15", 0.0),
                    "f-score-20": out_dict_cur.get("f-score-20", 0.0),
                    "normals": out_dict_cur.get("normals", 0.0),
                    "normals completeness": out_dict_cur.get(
                        "normals completeness", 0.0
                    ),
                    "normals accuracy": out_dict_cur.get("normals accuracy", 0.0),
                    "completeness": out_dict_cur.get("completeness", 0.0),
                    "accuracy": out_dict_cur.get("accuracy", 0.0),
                    "completeness2": out_dict_cur.get("completeness2", 0.0),
                    "accuracy2": out_dict_cur.get("accuracy2", 0.0),
                    "num_primitives": int(num_prim),
                }
            )

    # Save results
    metrics_csv = os.path.join(output_dir, f"{dataset}_{split}_metrics.csv")
    print(f"Saving per-object metrics to {metrics_csv}...")

    fieldnames = [
        "index",
        "name",
        "chamfer-L1",
        "chamfer-L2",
        "iou",
        "f-score",
        "f-score-15",
        "f-score-20",
        "normals",
        "normals completeness",
        "normals accuracy",
        "completeness",
        "accuracy",
        "completeness2",
        "accuracy2",
        "num_primitives",
    ]

    with open(metrics_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in object_metrics:
            writer.writerow(data)

    # Print Metrics
    count = aggregated_metrics["count"]
    if count > 0:
        mean_chamfer_l1 = aggregated_metrics["chamfer-L1"] / count
        mean_chamfer_l2 = aggregated_metrics["chamfer-L2"] / count
        mean_iou = aggregated_metrics["iou"] / count
        mean_fscore = aggregated_metrics["f-score"] / count
        mean_fscore_15 = aggregated_metrics["f-score-15"] / count
        mean_fscore_20 = aggregated_metrics["f-score-20"] / count
        mean_normals = aggregated_metrics["normals"] / count
        mean_normals_completeness = aggregated_metrics["normals completeness"] / count
        mean_normals_accuracy = aggregated_metrics["normals accuracy"] / count
        mean_completeness = aggregated_metrics["completeness"] / count
        mean_accuracy = aggregated_metrics["accuracy"] / count
        mean_completeness2 = aggregated_metrics["completeness2"] / count
        mean_accuracy2 = aggregated_metrics["accuracy2"] / count
        mean_num_primitives = aggregated_metrics["num_primitives"] / count

        print("\n----- Evaluation Results -----")
        print(f"{'mean_chamfer_l1':>25}: {mean_chamfer_l1:.6f}")
        print(f"{'mean_chamfer_l2':>25}: {mean_chamfer_l2:.6f}")
        print(f"{'mean_iou':>25}: {mean_iou:.6f}")
        print(f"{'mean_f-score':>25}: {mean_fscore:.6f}")
        print(f"{'mean_f-score-15':>25}: {mean_fscore_15:.6f}")
        print(f"{'mean_f-score-20':>25}: {mean_fscore_20:.6f}")
        print(f"{'mean_normals':>25}: {mean_normals:.6f}")
        print(f"{'mean_normals_completeness':>25}: {mean_normals_completeness:.6f}")
        print(f"{'mean_normals_accuracy':>25}: {mean_normals_accuracy:.6f}")
        print(f"{'mean_completeness':>25}: {mean_completeness:.6f}")
        print(f"{'mean_accuracy':>25}: {mean_accuracy:.6f}")
        print(f"{'mean_completeness2':>25}: {mean_completeness2:.6f}")
        print(f"{'mean_accuracy2':>25}: {mean_accuracy2:.6f}")
        print(f"{'avg_num_primitives':>25}: {mean_num_primitives:.6f}")

        # Sort by Chamfer-L1 descending (worst first)
        object_metrics.sort(key=lambda x: x["chamfer-L1"], reverse=True)
        print("\n----- Top 10 Worst Objects (by Chamfer-L1) -----")
        print(f"{'Index':<10} {'Name':<40} {'Chamfer-L1':<15}")
        for item in object_metrics[:10]:
            print(f"{item['index']:<10} {item['name']:<40} {item['chamfer-L1']:.6f}")
    else:
        print("No valid objects evaluated.")


if __name__ == "__main__":

    @hydra.main(version_base=None, config_path="../../configs/evaluate", config_name="eval")
    def run_main(cfg: DictConfig):
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        main(cfg)

    run_main()
