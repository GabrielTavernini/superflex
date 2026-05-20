import glob
import torch
import numpy as np
import viser
import random
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import trimesh
import os
from omegaconf import OmegaConf
from torch_geometric.nn import fps

from superflex.utils.predictions_handler import PredictionHandler
from superflex.utils.visualizations import generate_ncolors
from superflex.optimization import SuperQOptimization
from superflex.superflex import SuperFlex
from superflex.data.dataloader import denormalize_outdict, denormalize_points, normalize_points
from superflex.data.transform import rotate_around_axis

def main():
    checkpoint_folder = "checkpoints/superflex"  # specify your checkpoints folder
    checkpoint_file = "ckpt.pt"  # specify your checkpoint file 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mesh_path = "examples/chair.glb"  # specify your input mesh path
    z_up = False  # specify if your input point cloud is in z-up orientation
    normalize = True  # specify if you want to normalize the input point cloud
    max_iterations = 1000 # specify how many optimization iterations to run
    
    if not os.path.isfile(mesh_path):
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
    
    # Extract points from mesh
    print(f"Loading mesh from {mesh_path}...")
    points_4096, points_iou, occupancies = extract_points_from_mesh(mesh_path)
    print(f"Extracted {points_4096.shape[0]} surface points and {points_iou.shape[0]} volume points")
    
    # Run predictions
    print("Running predictions...")
    mesh_name = os.path.splitext(os.path.basename(mesh_path))[0]
    pred_handler, translation, scale = run_predictions(
        points_4096, 
        checkpoint_folder=checkpoint_folder,
        checkpoint_file=checkpoint_file,
        device=device,
        normalize=normalize,
        z_up=z_up,
        object_name=mesh_name
    )
    
    index = 0
    object_name = mesh_name
    print(f"Optimizing {object_name}")
    
    # Prepare external data with volume and surface points
    external_data = {
        'points_iou': [torch.tensor(points_iou, dtype=torch.float32, device=device)],
        'occupancies': [torch.tensor(occupancies, dtype=torch.bool, device=device)],
        'points': [torch.tensor(points_4096, dtype=torch.float32, device=device)]
    }
    
    # Setup optimization with external point data
    superq = SuperQOptimization(
        pred_handler=pred_handler,
        indices=[index],
        ply_paths=None,
        external_data=external_data,
        device=device
    )
    
    param_groups = superq.get_param_groups()
    optimizer = torch.optim.Adam(param_groups)

    pred_handler, meshes = superq.update_handler(denormalize=False)
    orig_mesh = meshes[0]
    
    
    # Setup visualization server for raw extracted points
    server = viser.ViserServer()
    server.scene.set_up_direction([0.0, 1.0, 0.0])

    # Add mesh to existing visualization server
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
    pbar = tqdm(range(max_iterations), desc="Fitting Superquadrics")
    best_loss = float('inf')
    best_params = None
    
    for iteration in pbar:
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
                    "iteration": iteration,
                }

        # Visualize every 50 iterations
        if iteration % 50 == 0:
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
        print(f"Best IoU: {best_params['iou']:.4f} (iteration {best_params['iteration']})")
    
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

def extract_points_from_mesh(mesh_path, n_points_surf=100000, n_points_4096=4096, n_points_vol=100000):
    """Extract surface and volume points from a mesh file using trimesh.
    
    Args:
        mesh_path: Path to mesh file (GLB, GLTF, OBJ, etc.)
        n_points_surf: Number of surface points to sample
        n_points_4096: Number of points for FPS downsampling
        n_points_vol: Number of volume points to sample
        
    Returns:
        points_4096: Downsampled surface points (n_points_4096, 3)
        points_iou: Volume points for IoU calculation (n_points_vol, 3)
        occupancies: Occupancy of volume points
    """
    # Load mesh
    mesh_or_scene = trimesh.load(mesh_path)
    if isinstance(mesh_or_scene, trimesh.Scene):
        if len(mesh_or_scene.geometry) == 0:
            raise ValueError(f"Empty scene in {mesh_path}")
        mesh = trimesh.util.concatenate(tuple(mesh_or_scene.geometry.values()))
    else:
        mesh = mesh_or_scene
    
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Not a mesh: {mesh_path} {type(mesh)}")
    
    if mesh.bounds is None:
        raise ValueError(f"Mesh has invalid bounds: {mesh_path}")
    
    # Compute mesh bounds
    center = mesh.bounding_box.centroid
    max_extent = np.max(mesh.extents)
    
    # Sample surface points
    points_surface, _ = trimesh.sample.sample_surface(mesh, n_points_surf)
    
    # FPS downsampling using torch_geometric
    points_tensor = torch.from_numpy(points_surface).float()
    ratio = n_points_4096 / points_surface.shape[0]
    indices = fps(points_tensor, ratio=ratio)
    points_4096 = points_surface[indices.numpy()]
    
    # Sample volume points (10% padding on bounding box)
    half_size = (max_extent * 1.1) / 2
    min_bound = center - half_size
    max_bound = center + half_size
    points_vol = np.random.uniform(min_bound, max_bound, (n_points_vol, 3))
    
    # Compute occupancy using voxelized mesh
    pitch = (max_extent * 1.1) / 100
    voxel_grid = mesh.voxelized(pitch=pitch)
    voxel_grid = voxel_grid.fill()
    occupancies = voxel_grid.is_filled(points_vol)
    
    return points_4096, points_vol.astype(np.float32), occupancies

def run_predictions(points, checkpoint_folder="checkpoints/superflex", 
                   checkpoint_file="ckpt.pt", device='cuda', 
                   normalize=True, z_up=False, object_name="object"):
    """Run SuperFlex predictions on point cloud.
    
    Args:
        points: Input point cloud (N, 3)
        checkpoint_folder: Path to checkpoint folder
        checkpoint_file: Checkpoint filename
        device: Device to use (cuda/cpu)
        normalize: Whether to normalize points
        z_up: Whether input is z-up oriented
        object_name: Object name for metadata
        
    Returns:
        pred_handler: PredictionHandler with predictions
        translation: Normalization translation
        scale: Normalization scale
    """
    ckp_path = os.path.join(checkpoint_folder, checkpoint_file)
    config_path = os.path.join(checkpoint_folder, 'config.yaml')
    
    if not os.path.isfile(ckp_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckp_path}")
    
    checkpoint = torch.load(ckp_path, map_location=device, weights_only=False)
    with open(config_path) as f:
        configs = OmegaConf.load(f)
    
    model = SuperFlex(configs.superflex).to(device)
    print("Loading checkpoint from:", ckp_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Ensure 4096 points
    n_points = points.shape[0]
    if n_points != 4096:
        replace = n_points < 4096
        idxs = np.random.choice(n_points, 4096, replace=replace)
        points = points[idxs]
    
    # Normalize
    if normalize:
        points_norm, translation, scale = normalize_points(points)
    else:
        points_norm = points
        translation = np.zeros(3)
        scale = 1.0
    
    if z_up:
        points_norm = rotate_around_axis(points_norm, axis=(1, 0, 0), 
                                        angle=-np.pi/2, center_point=np.zeros(3))
    
    # Convert to tensor
    points_tensor = torch.from_numpy(points_norm).unsqueeze(0).to(device).float()
        
    # Run model
    with torch.no_grad():
        outdict = model(points_tensor)
        for key in outdict:
            if isinstance(outdict[key], torch.Tensor):
                outdict[key] = outdict[key].cpu()
        
        translation_arr = np.array([translation])
        scale_arr = np.array([scale])
        outdict = denormalize_outdict(outdict, translation_arr, scale_arr, z_up)
        points = denormalize_points(points_tensor.cpu(), translation_arr, scale_arr, z_up)
    
    # Create prediction handler
    pred_handler = PredictionHandler.from_outdict(outdict, points, [object_name])
    
    return pred_handler, translation, scale

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    main()