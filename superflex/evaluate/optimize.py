import os
import torch
import numpy as np
from tqdm import tqdm
from superflex.utils.predictions_handler import PredictionHandler
import random
import hydra
from omegaconf import DictConfig, OmegaConf
import gc

from superflex.optimization import SuperQOptimization
from superflex.utils.evaluation import build_dataloader


def main(cfg: DictConfig):
    print("\n========== SuperFlex Optimization ==========")
    print("Config:\n" + OmegaConf.to_yaml(cfg))

    dataset = cfg.dataloader.get("dataset", "shapenet")
    source_folder = cfg.get("source_folder", dataset)
    split = cfg.get(dataset).get("split", "test")

    print(f"Optimizing on {dataset} {split}...")
    input_npz = f"data/output_npz/{source_folder}/{dataset}_{split}.npz"
    folder_name = "optim"
    output_dir = os.path.join("data", "output_npz", source_folder, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    output_npz = os.path.join(output_dir, f"{dataset}_{split}.npz")

    print(f"Loading {input_npz}...")
    print(f"Will save results to {output_npz}...")
    pred_handler = PredictionHandler.from_npz(input_npz)

    # Build dataloader from config (expects cfg.dataloader.*)
    dataloader = build_dataloader(cfg)
    assert (
        dataloader.dataset.normalize == False
    ), "Optimization dataset should not be normalized."
    # Sanity-check: ensure prediction names align with dataset model_ids
    dataset_names = [m["model_id"] for m in dataloader.dataset.models]
    assert np.array_equal(
        pred_handler.names, dataset_names
    ), f"Object order differs between pred_handler and dataset."
    print(
        f"Loaded {len(dataloader.dataset)} objects from all categories out of {pred_handler.scale.shape[0]}."
    )

    device = cfg.get("device", "cuda")
    num_epochs = cfg.get("num_epochs", 1000)

    for batch in tqdm(dataloader, desc="Processing batches"):
        batch_indices = batch["idx"]

        # Move tensors to device
        points = batch["points"].to(device)
        normals = batch["normals"].to(device)

        points_iou = batch["points_iou"].to(device)
        occupancies = batch["occupancies"].to(device)

        superq = SuperQOptimization(
            pred_handler=pred_handler,
            indices=batch_indices,
            device=device,
            external_data=(
                {
                    "points": points,
                    "normals": normals,
                    "points_iou": points_iou,
                    "occupancies": occupancies,
                }
            ),
            cfg=cfg.optimization,
        )

        param_groups = superq.get_param_groups()
        optimizer = torch.optim.Adam(param_groups)

        best_losses = [float("inf")] * len(batch_indices)
        best_params = [None] * len(batch_indices)

        # Optimization Loop
        for epoch in range(num_epochs):
            optimizer.zero_grad()

            # forward: returns a dict output
            forward_out = superq.forward()

            # Compute per-batch losses using shared function
            loss, _ = superq.compute_losses(forward_out)
            batch_loss = loss.mean()

            if torch.isnan(batch_loss):
                print(f"nan loss at epoch {epoch}")
                break

            batch_loss.backward()
            optimizer.step()

            # Save best parameters (based on Total Loss per object)
            with torch.no_grad():
                for b in range(len(batch_indices)):
                    current_loss = loss[b].item()
                    if current_loss < best_losses[b]:
                        best_losses[b] = current_loss
                        best_params[b] = {
                            "raw_scale": superq.raw_scale[b].clone(),
                            "raw_exponents": superq.raw_exponents[b].clone(),
                            "raw_rotation": superq.raw_rotation[b].clone(),
                            "translation": superq.translation[b].clone(),
                        }
                        if hasattr(superq, "raw_tapering"):
                            best_params[b]["raw_tapering"] = superq.raw_tapering[
                                b
                            ].clone()
                        if hasattr(superq, "raw_bending"):
                            best_params[b]["raw_bending"] = superq.raw_bending[
                                b
                            ].clone()

        # Restore best parameters
        with torch.no_grad():
            for b in range(len(batch_indices)):
                if best_params[b] is not None:
                    superq.raw_scale[b].copy_(best_params[b]["raw_scale"])
                    superq.raw_exponents[b].copy_(best_params[b]["raw_exponents"])
                    superq.raw_rotation[b].copy_(best_params[b]["raw_rotation"])
                    superq.translation[b].copy_(best_params[b]["translation"])
                    if hasattr(superq, "raw_tapering"):
                        superq.raw_tapering[b].copy_(best_params[b]["raw_tapering"])
                    if hasattr(superq, "raw_bending"):
                        superq.raw_bending[b].copy_(best_params[b]["raw_bending"])

        superq.update_handler(compute_meshes=False)

        # Cleanup to avoid OOM
        del superq
        del optimizer
        del param_groups
        del best_params
        del best_losses
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save results
    print(f"Saving optimized results to {output_npz}...")
    pred_handler.save_npz(output_npz)
    print(f"Optimization complete!")


if __name__ == "__main__":

    @hydra.main(version_base=None, config_path="../../configs/evaluate", config_name="optimize")
    def run_main(cfg: DictConfig):
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        main(cfg)

    run_main()
