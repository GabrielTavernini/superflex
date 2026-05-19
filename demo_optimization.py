import glob
import torch
import numpy as np
import viser
import random
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

from superflex.utils.predictions_handler import PredictionHandler
from superflex.utils.visualizations import generate_ncolors
from superflex.optimization import SuperQOptimization

def main():
    # Example Configuration / Model Outputs Configuration
    npz_path = "examples/table/predictions.npz" # "data/output_npz/superflex/shapenet_test.npz"
    ply_path = "examples/table/pointcloud.npz" # None
    num_epochs = 1000
    index = 0
    
    # Load predictions
    pred_handler = PredictionHandler.from_npz(npz_path)
    object_name = pred_handler.names[index]
    if not ply_path:
      shapenet_matches = glob.glob(f"data/ShapeNet/*/{object_name}/pointcloud*.npz")
      if not shapenet_matches:
          raise FileNotFoundError(
              f"Could not find pointcloud.npz for '{object_name}' in any data/ShapeNet subdirectory"
          )
      ply_path = shapenet_matches[0]

    superq = SuperQOptimization(
        pred_handler=pred_handler,
        indices=[index],
        ply_paths=[ply_path],
    )
    print(f"Optimizing {object_name} (index={index})")
    
    # Setup optimization
    param_groups = superq.get_param_groups()
    optimizer = torch.optim.Adam(param_groups)

    pred_handler, meshes = superq.update_handler(denormalize=False)
    orig_mesh = meshes[0]

    # Setup visualization server
    server = viser.ViserServer()
    server.scene.set_up_direction([0.0, 1.0, 0.0])
    server.scene.add_mesh_trimesh("original_superquadrics", mesh=orig_mesh, visible=False)

    # Add segmented pointcloud
    points = pred_handler.pc[superq.indices[0]] - superq.normalization_translation.cpu().numpy()
    points /= superq.normalization_scale.cpu().numpy()
    assign_matrix = pred_handler.assign_matrix[superq.indices[0]]
    colors = generate_ncolors(assign_matrix.shape[1])
    segmentation = np.argmax(assign_matrix, axis=1)
    colored_pc = colors[segmentation]
    server.scene.add_point_cloud(
        name="/segmented_pointcloud",
        points=points,
        colors=colored_pc,
        point_size=0.005,
        visible=False,
    )
    
    # Optimization loop
    pbar = tqdm(range(num_epochs), desc="Fitting Superquadrics")
    best_loss = float('inf')
    best_params = None
    
    for epoch in pbar:
        optimizer.zero_grad()
        forward_out = superq.forward()
        sdf_vals = forward_out.get('sdfs')

        loss, losses = superq.compute_losses(forward_out)
        loss = loss[0]

        if torch.isnan(loss):
            print("Failed optimization with nan values")
            return

        loss.backward()
        optimizer.step()

        # Track best parameters
        with torch.no_grad():
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_params = {
                    "raw_scale": superq.raw_scale.clone(),
                    "raw_exponents": superq.raw_exponents.clone(),
                    "raw_rotation": superq.raw_rotation.clone(),
                    "raw_tapering": superq.raw_tapering.clone(),
                    "raw_bending": superq.raw_bending.clone(),
                    "translation": superq.translation.clone(),
                    "iou": losses['iou'][0].item(),
                    "epoch": epoch,
                }

        # Visualize every 50 epochs
        if epoch % 50 == 0:
            visualize_handler(server, superq, sdf_vals)
        
        pbar.set_postfix({
            "IoU": f"{losses['iou'][0].item():.2f}",
            "Sdf": f"{losses['sdf'][0].item():.4f}",
            "Bbox": f"{losses['bbox'][0].item():.4f}",
            "Over": f"{losses['overlap'][0].item():.4f}",
            "Loss": f"{loss.item():.4f}"
        })

    # Restore and visualize best parameters
    if best_params is not None:
        with torch.no_grad():
            superq.raw_scale.copy_(best_params["raw_scale"])
            superq.raw_exponents.copy_(best_params["raw_exponents"])
            superq.raw_rotation.copy_(best_params["raw_rotation"])
            superq.raw_tapering.copy_(best_params["raw_tapering"])
            superq.raw_bending.copy_(best_params["raw_bending"])
            superq.translation.copy_(best_params["translation"])
        
        forward_out = superq.forward()
        sdf_vals = forward_out.get('sdfs')
        visualize_handler(server, superq, sdf_vals)
        print(f"Best IoU: {best_params['iou']:.4f} (epoch {best_params['epoch']})")
    
    # Keep server running
    while True:
        time.sleep(10.0)

def visualize_handler(server, superq, sdf_values):
    """Visualize superquadrics mesh and SDF point cloud."""
    sdf_values = sdf_values.detach().cpu()

    _, meshes = superq.update_handler(denormalize=False)
    server.scene.add_mesh_trimesh("superquadrics", mesh=meshes[0], visible=True)

    M_ps = superq.M_points_surf
    points = superq.points[0].detach().cpu().numpy()
    cmap = plt.get_cmap('RdBu')
    norm = plt.Normalize(vmin=-superq.truncation, vmax=superq.truncation)
    sdf_arr = sdf_values[0].numpy()
    sdf_colors = cmap(norm(sdf_arr))[:, :3]
    server.scene.add_point_cloud(
        name="/sdf_pointcloud",
        points=points[-M_ps:],
        colors=sdf_colors[-M_ps:],
        point_size=0.005,
        visible=True,
    )

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    main()