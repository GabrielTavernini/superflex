from superflex.utils.predictions_handler import PredictionHandler
from superflex.utils.visualizations import export_mesh_trimesh, export_o3d_pc

RESOLUTION = 10

epoch = 9
path = f"outputs/epoch_{epoch}.npz"  # Path to your predictions file
pred_handler = PredictionHandler.from_npz(path)

meshes = pred_handler.get_meshes(resolution=RESOLUTION)
names = pred_handler.names
pcs = pred_handler.get_segmented_pcs()

num_meshes = 5


for i in range(num_meshes):  # Assuming you want to visualize the first batch
    export_mesh_trimesh(meshes[i], f"{names[i]}_sq.ply")
    export_o3d_pc(pcs[i], f"{names[i]}_seg.ply")