"""
Wandb visualization utilities for SuperFlow.
 
Logs 3D superquadric meshes and point clouds to wandb.
"""
 
import io
import tempfile
import os
 
import numpy as np
import torch
 
import wandb
import trimesh
 
from superflex.utils.visualizations import generate_ncolors
from superflex.utils.predictions_handler import PredictionHandler

class WandbViser:
    def __init__(self, wandb_run):
        self.wandb_run = wandb_run
        self.wandb_val_objects ={'pred': [], 'gt': []}
        self.wandb_val_temp_files = []
        self.log_gt = True
         
    def _point_cloud_to_spheres(self, points, colors=None, default_color=(180, 180, 180, 255), radius=0.008, max_points=512):
        """Convert a point cloud to a mesh of small spheres for glb export."""
        if trimesh is None:
            return None
    
        if points.shape[0] > max_points:
            indices = np.random.choice(points.shape[0], max_points, replace=False)
            points = points[indices]
            if colors is not None:
                colors = colors[indices]
    
        N = points.shape[0]
        sphere = trimesh.creation.icosphere(subdivisions=1, radius=radius)
        template_verts = sphere.vertices  # (V, 3)
        template_faces = sphere.faces     # (F, 3)
        V, F = len(template_verts), len(template_faces)
    
        # Tile and offset vertices
        all_verts = np.tile(template_verts, (N, 1, 1)) + points[:, None, :]
        all_verts = all_verts.reshape(-1, 3)
    
        # Tile faces with per-point vertex offset
        offsets = np.arange(N)[:, None, None] * V
        all_faces = (np.tile(template_faces, (N, 1, 1)) + offsets).reshape(-1, 3)
    
        if colors is not None:
            # colors is (N, 3) or (N, 4)
            if colors.shape[1] == 3:
                colors = np.concatenate([colors, np.full((N, 1), 255, dtype=colors.dtype)], axis=1)
            # Repeat each point's color for all faces of its sphere
            face_colors = np.repeat(colors, F, axis=0)
        else:
            face_colors = np.tile(np.array(default_color, dtype=np.uint8), (N * F, 1))
            
        return trimesh.Trimesh(vertices=all_verts, faces=all_faces, face_colors=face_colors)
    
    def log_accumulated_wandb_objects(self, epoch):
        """Log accumulated wandb objects and clear the list."""
        if self.wandb_run is not None:
            log_dict = {}
            if self.wandb_val_objects['pred']:
                log_dict["visual/pred"] = self.wandb_val_objects['pred']
            if self.wandb_val_objects['gt'] and self.log_gt:
                log_dict["visual/gt"] = self.wandb_val_objects['gt']
                self.log_gt = False
            if log_dict:
                self.wandb_run.log(log_dict, step=epoch)
            self.wandb_val_objects = {'pred': [], 'gt': []}
            
            for f in self.wandb_val_temp_files:
                try:
                    os.remove(f)
                except OSError:
                    pass
            self.wandb_val_temp_files = []

    def accumulate_wandb_objects(self, epoch, outdict, batch, num_samples=1):
        """Accumulate wandb objects from a batch into the self."""
        # Get predictions
        pred_objs, pred_temp_files = self.get_wandb_objects(outdict, batch['points'], num_samples=num_samples, resolution=30)
        self.wandb_val_objects['pred'].extend(pred_objs)
        self.wandb_val_temp_files.extend(pred_temp_files)
        
        # Get ground truth if available
        if 'gt_scale' in batch and self.log_gt:
            B, N, _ = batch['points'].shape
            P = batch['gt_scale'].shape[1]
            gt_outdict = {
                'scale': batch['gt_scale'],
                'shape': batch['gt_shape'],
                'rotate': batch['gt_rotate'],
                'trans': batch['gt_trans'],
                'exist': batch['gt_exist'],
                'assign_matrix': torch.zeros(B, N, P, device=batch['points'].device)
            }
            if 'gt_tapering' in batch:
                gt_outdict['tapering'] = batch['gt_tapering']
            if 'gt_bending' in batch:
                gt_outdict['bending'] = batch['gt_bending']
                
            gt_objs, gt_temp_files = self.get_wandb_objects(gt_outdict, batch['points'], num_samples=num_samples, resolution=30)
            self.wandb_val_objects['gt'].extend(gt_objs)
            self.wandb_val_temp_files.extend(gt_temp_files)

    def get_wandb_objects(self, outdict, pc, num_samples = 4, resolution = 30):
        """
        Get wandb Object3D instances for superquadric meshes overlaid with point clouds.
    
        Args:
            outdict: model output dict (batched)
            pc: [B, 3, N] point cloud (channels first) or [B, N, 3]
            num_samples: number of samples to visualize
            resolution: mesh resolution per superquadric
        Returns:
            Tuple of (List of wandb.Object3D, List of temp file paths)
        """
        if wandb is None:
            return [], []
    
        B = outdict['trans'].shape[0]
        num_samples = min(num_samples, B)
    
        # Ensure pc is [B, N, 3]
        if pc.shape[1] == 3 and pc.shape[2] != 3:
            pc = pc.transpose(1, 2)
        pc_np = pc.detach().cpu().numpy()
    
        temp_files = []
        wandb_objects = []
        
        names = np.arange(B)
        pred_handler = PredictionHandler.from_outdict(outdict, pc, names)
        sq_meshes = pred_handler.get_meshes(resolution=resolution, colors=True)
        pred_pcs = pred_handler.get_segmented_pcs()
    
        for i in range(num_samples):
            pc_points = pc_np[i]  # [N, 3]
            pc_colors = None
            if pred_pcs[i] is not None:
                pc_colors = (pred_pcs[i].colors * 255).astype(np.uint8)
                
            pc_mesh = self._point_cloud_to_spheres(pc_points, colors=pc_colors)
    
            sq_mesh = sq_meshes[i]
    
            parts = [m for m in (sq_mesh, pc_mesh) if m is not None]
            if parts:
                combined = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
                obj3d = self._export_object3d(combined, temp_files)
                if obj3d is not None:
                    wandb_objects.append(obj3d)
        return wandb_objects, temp_files
    
    def _export_object3d(self, mesh, temp_files):
        """Export a trimesh to a wandb Object3D and return it."""
        try:
            with tempfile.NamedTemporaryFile(suffix='.glb', delete=False) as f:
                mesh.export(f.name, file_type='glb')
                temp_files.append(f.name)
                return wandb.Object3D(open(f.name, 'rb'))
        except Exception:
            verts = mesh.vertices
            vert_data = np.zeros((verts.shape[0], 6))
            vert_data[:, :3] = verts
            vert_data[:, 3:] = 180
            return wandb.Object3D(vert_data)