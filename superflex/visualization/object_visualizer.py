import numpy as np
import os
import viser
import time
from superflex.utils.predictions_handler import PredictionHandler 
from superflex.utils.visualizations import generate_ncolors
import torch
import trimesh
import hydra
from omegaconf import DictConfig, OmegaConf

RESOLUTION = 30

def main(cfg: DictConfig) -> None:
  server = viser.ViserServer()
  if cfg.dataset not in ['shapenet', 'abo', 'scene']:
    raise NotImplementedError(f"Dataset {cfg.dataset} not implemented for visualization yet.")

  if cfg.dataset == 'shapenet' or cfg.dataset == 'abo':
    input_path = os.path.join(cfg.npz_folder, f'{cfg.dataset}_{cfg.split}.npz')
  elif cfg.dataset == 'scene':
    input_path = os.path.join(cfg.npz_folder, f'{cfg.split}.npz')
  print("Opening npz...")
  predictions_sq = PredictionHandler.from_npz(input_path)
  print("Computing meshes...")
  meshes = predictions_sq.get_meshes(resolution=RESOLUTION)

  existence_mesh = torch.ones(len(meshes), dtype = torch.bool)
  print("Computing point clouds...")
  pcs = predictions_sq.get_segmented_pcs()
  print("Done!")
  names = predictions_sq.names

  if cfg.dataset == 'scene': # visualization for scenes
    server.scene.set_up_direction([0.0, 0.0, 1.0])
    num_objects = len(meshes)
    colors = generate_ncolors(num_objects)/255 #(max(int_ids)+1)/255
    for idx in range(len(meshes)):
      if meshes[idx] == None or not existence_mesh[idx]:# int(names[idx]) >= 50 ormeshes[idx] == None or int(names[idx]) >= 50: # TODO for now I am only taking the first 200 objects, modify this
        continue
      meshes[idx].visual.face_colors = np.ones((meshes[idx].visual.face_colors.shape[0], 3)) * colors[idx]
      meshes[idx].visual.vertex_colors = np.ones((meshes[idx].visual.vertex_colors.shape[0], 3)) * colors[idx]
      server.scene.add_mesh_trimesh(f"superquadrics_{names[idx]}", mesh=meshes[idx], visible=True)
    
      server.scene.add_point_cloud(
            name=f"/segmented_pointcloud_{names[idx]}",
            points=np.array(pcs[idx].points),
            colors=pcs[idx].colors,
            point_size=0.005,
            visible = False
        )
  elif cfg.dataset == 'shapenet' or cfg.dataset == 'abo': # visualization for objects
    def draw_superquadric_and_segmentation():
      idx = int(gui_model_selection.value)
      server.scene.add_mesh_trimesh("superquadrics", mesh=meshes[idx], visible=True)
      
      server.scene.add_point_cloud(
            name="/segmented_pointcloud",
            points=np.array(pcs[idx].points),
            colors=np.array(pcs[idx].colors),
            point_size=0.005,
        )
    server.scene.set_up_direction([0.0, 1.0, 0.0])
    gui_model_selection = server.gui.add_dropdown("Model index", [str(i) for i in range(len(names))], initial_value='0')
    gui_model_selection.on_update(lambda _: draw_superquadric_and_segmentation())
    draw_superquadric_and_segmentation()
  else:
    raise NotImplementedError(f"Dataset {cfg.dataset} not implemented for visualization yet.")
  
    
  @server.on_client_connect
  def _(client: viser.ClientHandle) -> None:
    client.camera.position = (0.8, 0.8, 0.8)
    client.camera.look_at = (0., 0., 0.)
    
  
  while True:
      time.sleep(10.0)

if __name__ == "__main__":
    @hydra.main(version_base=None, config_path="../../configs", config_name="object_visualizer")
    def run_main(cfg: DictConfig):
        main(cfg)
    run_main()