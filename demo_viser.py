import os
import torch
import numpy as np
from omegaconf import OmegaConf
from superflex.superflex import SuperFlex
from superflex.utils.predictions_handler import PredictionHandler
from superflex.data.dataloader import denormalize_outdict, denormalize_points
import open3d as o3d
import viser
from superflex.data.dataloader import normalize_points, denormalize_outdict
from superflex.data.transform import rotate_around_axis
import time

def main():
    checkpoints_folder = "checkpoints/superflex"  # specify your checkpoints folder
    checkpoint_file = "ckpt.pt"  # specify your checkpoint file 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    path_to_point_cloud = "examples/lamp.ply"  # specify your input point cloud path
    z_up = False  # specify if your input point cloud is in z-up orientation
    normalize = True  # specify if you want to normalize the input point cloud
    resolution = 30  # specify the resolution for mesh extraction

    ckp_path = os.path.join(checkpoints_folder, checkpoint_file)
    config_path = os.path.join(checkpoints_folder, 'config.yaml')
    if not os.path.isfile(ckp_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckp_path}")
    checkpoint = torch.load(ckp_path, map_location=device, weights_only=False)
    with open(config_path) as f:
        configs = OmegaConf.load(f)

    model = SuperFlex(configs.superflex).to(device)
    print("Loading checkpoint from:", ckp_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    pc = o3d.io.read_point_cloud(path_to_point_cloud) 
    points_tmp = np.asarray(pc.points)
    n_points = points_tmp.shape[0]  

    if n_points != 4096:  
        replace = n_points < 4096
        idxs = np.random.choice(n_points, 4096, replace=replace)
        points = points_tmp[idxs]
    else:
        points = points_tmp
    

    if normalize:
        points, translation, scale  = normalize_points(points)
    else:
        translation = np.zeros(3)
        scale = 1.0
    if z_up:
        points = rotate_around_axis(points, axis=(1,0,0), angle = -np.pi/2, center_point=np.zeros(3))

    points = torch.from_numpy(points).unsqueeze(0).to(device).float()

    with torch.no_grad():
        outdict = model(points)
        for key in outdict:
            if isinstance(outdict[key], torch.Tensor):
                outdict[key] = outdict[key].cpu()
        translation = np.array([translation])
        scale = np.array([scale])
        outdict = denormalize_outdict(outdict, translation, scale, z_up)
        points = denormalize_points(points.cpu(), translation, scale, z_up)

    pred_handler = PredictionHandler.from_outdict(outdict, points, ['chair'])
    mesh = pred_handler.get_meshes(resolution=resolution)[0]

    pcs = pred_handler.get_segmented_pcs()[0]
    pred_handler.save_npz("test.npz")

    server = viser.ViserServer()
    server.scene.add_mesh_trimesh("superquadrics", mesh=mesh, visible=True)
    
    server.scene.add_point_cloud(
        name="/segmented_pointcloud",
        points=np.array(pcs.points),
        colors=np.array(pcs.colors),
        point_size=0.005,
    )
    if z_up:
        server.scene.set_up_direction([0.0, 0.0, 1.0])
    else:
        server.scene.set_up_direction([0.0, 1.0, 0.0])

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = (0.8, 0.8, 0.8)
        client.camera.look_at = (0., 0., 0.)
    while True:
        time.sleep(10.0)

if __name__ == "__main__":
    main()