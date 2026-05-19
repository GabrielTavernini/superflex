import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Enable TensorFloat32 for better performance on Ampere+ GPUs
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

from typing import Optional
from superflex.utils.transforms import quat2mat, mat2quat
from superflex.utils.safe_operations import safe_pow, safe_mul
from superflex.utils.predictions_handler import PredictionHandler

class SuperQOptimization(nn.Module):
    def __init__(
        self, 
        pred_handler: PredictionHandler,
        indices: list[int],
        ply_paths: list[str] = None,
        truncation: float = 0.05,
        device: str = "cuda",
        external_data: dict = None,
        cfg: Optional[object] = None,
    ):
        super().__init__()
        self.indices = indices
        self.device = device
        self.truncation = truncation
        self.pred_handler = pred_handler
        self.minS = 0.001
        self.minE, self.maxE = 0.1, 1.9
        
        self.cfg = cfg
        self.enable_tapering = self._parse_flag('tapering', True)
        self.enable_bending = self._parse_flag('bending', True)
        self.enable_reorient = self._parse_flag('reorient', False)
        self.enable_pruning = self._parse_flag('pruning', False)

        B = len(indices)
        self.N_max = pred_handler.scale.shape[1]
        self.M_points_iou = 100_000
        self.M_points_surf = 4096
        self.M_points = self.M_points_iou + self.M_points_surf

        scale_list = []
        exp_list = []
        rot_list = []
        trans_list = []
        tapering_list = []
        bending_list = []
        self.masks = [] 

        self.points = torch.zeros(B, self.M_points, 3, device=device)
        self.occupancies = torch.zeros(B, self.M_points_iou, dtype=torch.bool, device=device)
        
        self.w_overlap = 1.0
        self.w_bbox = 1.0
        self.w_sdf = 1.0
        if cfg is not None:
            self.w_sdf = getattr(cfg, 'w_sdf', 1.0)
            self.w_bbox = getattr(cfg, 'w_bbox', 1.0)
            self.w_overlap = getattr(cfg, 'w_overlap', 1.0)
        
        for i, idx in enumerate(indices):
            # --- Params ---
            mask = (pred_handler.exist[idx] > 0.5)
            self.masks.append(torch.tensor(mask, dtype=torch.bool, device=device).reshape(-1))

            s = torch.tensor(pred_handler.scale[idx], dtype=torch.float, device=device).reshape(-1, 3)
            e = torch.tensor(pred_handler.exponents[idx], dtype=torch.float, device=device).reshape(-1, 2)
            r = torch.tensor(pred_handler.rotation[idx], dtype=torch.float, device=device).reshape(-1, 3, 3)
            t = torch.tensor(pred_handler.translation[idx], dtype=torch.float, device=device).reshape(-1, 3)
            tap = torch.tensor(pred_handler.tapering[idx], dtype=torch.float, device=device).reshape(-1, 2)
            ben = torch.tensor(pred_handler.bending[idx], dtype=torch.float, device=device).reshape(-1, 6)
            
            s[~mask.reshape(-1)] = 1.0 
            scale_list.append(torch.log(torch.clamp(s - self.minS, min=1e-6)))
            e = torch.clamp(e, self.minE + 1e-4, self.maxE - 1e-4)
            exp_list.append(torch.logit((e - self.minE) / (self.maxE - self.minE)))
            rot_list.append(mat2quat(r))
            trans_list.append(t)
            
            if torch.count_nonzero(tap) == 0:
                raw_tap = torch.full_like(tap, 1e-4)
            else:
                tap = torch.clamp(tap, -0.9999, 0.9999)
                raw_tap = torch.atanh(tap)
            tapering_list.append(raw_tap)
            
            if torch.count_nonzero(ben) == 0:
                raw_ben_k = torch.full_like(ben[..., 0::2], 1e-3)
                raw_ben_k = torch.logit(raw_ben_k)
                raw_ben_a = torch.zeros(*ben.shape[:-1], 6, device=device)
                raw_ben_a[..., 1::2] = 1.0
                raw_ben = torch.cat([raw_ben_k, raw_ben_a], dim=-1)
            else:
                min_radius = self._compute_min_radius(s, tap)
                raw_ben_k = torch.logit(torch.clamp(ben[..., 0::2] * min_radius, 1e-6, 1 - 1e-6))
                alpha = ben[..., 1::2]
                raw_ben_a = torch.empty(*ben.shape[:-1], 6, device=device)
                raw_ben_a[..., 0::2] = torch.sin(alpha)
                raw_ben_a[..., 1::2] = torch.cos(alpha)
                raw_ben = torch.cat([raw_ben_k, raw_ben_a], dim=-1)
            bending_list.append(raw_ben)
            
            # --- Points ---
            try:
                if external_data is not None:
                    pts_iou = external_data['points_iou'][i]
                    occ = external_data['occupancies'][i]
                    pts_surf = external_data['points'][i] # This is 4096 surface points
                    pts = torch.cat([pts_iou, pts_surf], dim=0)
                else:
                    ply = ply_paths[i] if ply_paths else None
                    points_file = ply.replace("pointcloud.npz", "points.npz")
                    points_file = points_file.replace("pointcloud_4096.npz", "points.npz")
                    points_dict = np.load(points_file)
                    points_iou = points_dict['points']
                    occ_tgt = points_dict['occupancies']
                    if np.issubdtype(occ_tgt.dtype, np.uint8):
                        occ_tgt = np.unpackbits(occ_tgt)[:points_iou.shape[0]]
                    occ = torch.tensor(occ_tgt, dtype=torch.bool, device=device)
                    pts_iou = torch.tensor(points_iou, dtype=torch.float, device=device)
                    pts_surf = torch.tensor(pred_handler.pc[idx], dtype=torch.float, device=device)
                    pts = torch.cat([pts_iou, pts_surf], dim=0)
            except Exception as e:
                print(f"Error loading {points_file}: {e}")
                exit()
            
            # Ensure pts has correct shape if not loaded from ply
            if pts.shape[0] != self.M_points and occ.shape[0] != self.M_points_iou:
                print(f"Error points shape missmatch")
                exit()
            
            self.points[i] = pts
            self.occupancies[i] = occ
            
        self.raw_scale = nn.Parameter(torch.stack(scale_list)) # (B, N, 3)
        self.raw_exponents = nn.Parameter(torch.stack(exp_list)) # (B, N, 2)
        self.raw_rotation = nn.Parameter(torch.stack(rot_list)) # (B, N, 4)
        self.translation = nn.Parameter(torch.stack(trans_list)) # (B, N, 3)
        self.raw_tapering = nn.Parameter(torch.stack(tapering_list)) # (B, N, 2)
        self.raw_bending = nn.Parameter(torch.stack(bending_list)) # (B, N, 9)
        self.normalize()
        if self.enable_reorient:
            self.reorient()
        
        self.exist_mask = torch.stack(self.masks) # (B, N)
        self.bbox_min = torch.min(self.points[:, self.M_points_iou:], dim=1).values.unsqueeze(1).unsqueeze(2)  # (B,3)
        self.bbox_max = torch.max(self.points[:, self.M_points_iou:], dim=1).values.unsqueeze(1).unsqueeze(2)  # (B,3)
        self.bbox_min -= (self.bbox_max - self.bbox_min) * .025
        self.bbox_max += (self.bbox_max - self.bbox_min) * .025

    def _parse_flag(self, key, default=True):
        if self.cfg is None:
            return default
        return bool(getattr(self.cfg, key, default))
    
    @torch.compile
    def normalize(self):
        # normalize based on surface points
        surf_points = self.points[:, -self.M_points_surf:]
        translation = torch.mean(surf_points, dim=1)  # (B,3)
        self.points = self.points - translation.unsqueeze(1)

        scale = 2 * torch.amax(torch.abs(surf_points), dim=(1, 2))  # (B,)
        self.points = self.points / scale.view(-1, 1, 1)

        self.normalization_translation = translation
        self.normalization_scale = scale

        with torch.no_grad():
            self.translation.data = (self.translation.data - translation.unsqueeze(1)) / scale.view(-1, 1, 1)
            # Update raw_scale so that scale() reflects the normalization
            new_s = self.scale() / scale.view(-1, 1, 1)
            self.raw_scale.data = torch.log(torch.clamp(new_s - self.minS, min=1e-6))
        
        return translation, scale
    
    @torch.compile
    def reorient(self):
        with torch.no_grad():
            s = self.scale()       # (B,N,3)
            R = self.rotation()    # (B,N,3,3)
            
            sx = s[..., 0]
            sy = s[..., 1]
            sz = s[..., 2]

            ratio = 2.5
            mask_swap_xz = (sx > ratio * sz) & (sx > ratio * sy)
            mask_swap_yz = (sy > ratio * sz) & (sy > ratio * sx)

            new_s = s.clone()
            
            if mask_swap_xz.any():
                new_s[..., 0][mask_swap_xz] = sz[mask_swap_xz]
                new_s[..., 2][mask_swap_xz] = sx[mask_swap_xz]
                
            if mask_swap_yz.any():
                new_s[..., 1][mask_swap_yz] = sz[mask_swap_yz]
                new_s[..., 2][mask_swap_yz] = sy[mask_swap_yz]

            self.raw_scale.data = torch.log(torch.clamp(new_s - self.minS, min=1e-6))
            
            new_R = R.clone()

            if mask_swap_xz.any():
                u, v, w = R.unbind(-1) # Splits (B, N, 3, 3) into three (B, N, 3) tensors
                new_u = torch.where(mask_swap_xz.unsqueeze(-1), -w, u)
                new_w = torch.where(mask_swap_xz.unsqueeze(-1), u, w)
                new_R = torch.stack([new_u, v, new_w], dim=-1)

            if mask_swap_yz.any():
                u, v, w = new_R.unbind(-1) # Splits (B, N, 3, 3) into three (B, N, 3) tensors
                new_v = torch.where(mask_swap_yz.unsqueeze(-1), -w, v)
                new_w = torch.where(mask_swap_yz.unsqueeze(-1), v, w)
                new_R = torch.stack([u, new_v, new_w], dim=-1)

            self.raw_rotation.data = mat2quat(new_R)

    @torch.compile
    def scale(self):
        return torch.exp(self.raw_scale) + self.minS

    @torch.compile   
    def exponents(self):
        return (torch.sigmoid(self.raw_exponents) * (self.maxE - self.minE)) + self.minE

    @torch.compile  
    def rotation(self):
        return quat2mat(self.raw_rotation)

    @torch.compile 
    def tapering(self):
        return torch.tanh(self.raw_tapering)
    
    def _compute_min_radius(self, s, tap):
        """Compute min_radius constraints for bending."""
        s_x, s_y, s_z = s[..., 0:1], s[..., 1:2], s[..., 2:3]
        
        # Apply tapering to effective scales
        if self.enable_tapering:
            s_x_eff = s_x * (1 + torch.abs(tap[..., 0:1]))
            s_y_eff = s_y * (1 + torch.abs(tap[..., 1:2]))
            s_z_eff = s_z
        else:
            s_x_eff, s_y_eff, s_z_eff = s_x, s_y, s_z
        
        # Thickness Constraint
        t_x = torch.amax(torch.cat([s_y_eff, s_z_eff], dim=-1), dim=-1, keepdim=True)
        t_y = torch.amax(torch.cat([s_x_eff, s_z_eff], dim=-1), dim=-1, keepdim=True)
        t_z = torch.amax(torch.cat([s_x_eff, s_y_eff], dim=-1), dim=-1, keepdim=True)
        thickness_limit = torch.cat([t_x, t_y, t_z], dim=-1)
        
        # Global Loop Constraint (Circumference > Length)
        loop_limit = torch.cat([s_x_eff, s_y_eff, s_z_eff], dim=-1) / torch.pi
        
        min_radius_xyz = torch.maximum(thickness_limit, loop_limit)
        
        # Reorder to (z, x, y) for bending channels
        min_radius = torch.cat([
            min_radius_xyz[..., 2:3],
            min_radius_xyz[..., 0:1],
            min_radius_xyz[..., 1:2],
        ], dim=-1)
        
        return min_radius
    
    def bending(self):
        s = self.scale()#.detach()
        t = self.tapering()#.detach()
        
        min_radius = self._compute_min_radius(s, t)
        kb = torch.sigmoid(self.raw_bending[..., :3]) / min_radius
        
        alpha_raw = self.raw_bending[..., 3:].view(*self.raw_bending.shape[:-1], 3, 2)
        alpha = torch.atan2(alpha_raw[..., 0], alpha_raw[..., 1])
        return kb, alpha
    
    def get_param_groups(self):
        lrs = {
            "raw_scale": 2e-2,
            "raw_exponents": 1e-2,
            "raw_tapering": 7e-3,
            "raw_bending": 6e-3,
        }
        groups = []
        for name, param in self.named_parameters():
            if not param.requires_grad: continue
            lr = lrs.get(name, 2e-3)
            groups.append({"params": [param], "lr": lr})
        return groups

    @torch.compile
    def inverse_bending_axis(self, x, y, z, kb, alpha, axis):
        if axis == 'z':
            u, v, w = x, y, z
        elif axis == 'x':
            u, v, w = y, z, x
        elif axis == 'y':
            u, v, w = z, x, y
        else:
             return x, y, z

        # Precompute reciprocal efficiently
        inv_kb = 1.0 / (kb + 1e-6)
        
        angle_offset = torch.atan2(v, u)
        R = torch.sqrt(u**2 + v**2) * torch.cos(alpha - angle_offset)
        gamma = torch.atan2(w, inv_kb - R)
        r = inv_kb - torch.sqrt(w**2 + (inv_kb - R)**2)
        
        u = u - (R - r) * torch.cos(alpha)
        v = v - (R - r) * torch.sin(alpha)
        w = inv_kb * gamma

        if axis == 'z':
            return u, v, w
        elif axis == 'x':
            return w, u, v
        elif axis == 'y':
            return v, w, u
    
    @torch.compile
    def sdf_batch(self, points):
        B, _, M = points.shape
        N = self.N_max
        
        points_expanded = points.unsqueeze(1) # (B, 1, 3, M)
        t = self.translation.unsqueeze(-1) # (B, N, 3, 1)
        points_centered = points_expanded - t # (B, N, 3, M)
        
        X = torch.matmul(self.rotation().transpose(-2, -1), points_centered) # (B, N, 3, M)
        
        e1 = self.exponents()[..., 0].unsqueeze(-1)
        e2 = self.exponents()[..., 1].unsqueeze(-1)
        sx = self.scale()[..., 0].unsqueeze(-1)
        sy = self.scale()[..., 1].unsqueeze(-1)
        sz = self.scale()[..., 2].unsqueeze(-1)
        
        x = X[:, :, 0, :]
        y = X[:, :, 1, :]
        z = X[:, :, 2, :]
        
        eps = 1e-6
        # Use torch.where and clamp for speed and avoiding tensor creation
        # Original: ((x > 0).float() * 2 - 1) maps >0 to 1, <=0 to -1
        x = torch.where(x > 0, 1.0, -1.0) * torch.clamp(torch.abs(x), min=eps)
        y = torch.where(y > 0, 1.0, -1.0) * torch.clamp(torch.abs(y), min=eps)
        z = torch.where(z > 0, 1.0, -1.0) * torch.clamp(torch.abs(z), min=eps)
        
        r0 = torch.sqrt(x**2 + y**2 + z**2)
        
        # Apply bending if enabled
        if self.enable_bending:
            kb, alpha = self.bending()
            x, y, z = self.inverse_bending_axis(x, y, z, kb[..., 0].unsqueeze(-1), alpha[..., 0].unsqueeze(-1), 'z')
            x, y, z = self.inverse_bending_axis(x, y, z, kb[..., 1].unsqueeze(-1), alpha[..., 1].unsqueeze(-1), 'x')
            x, y, z = self.inverse_bending_axis(x, y, z, kb[..., 2].unsqueeze(-1), alpha[..., 2].unsqueeze(-1), 'y')

        # Apply tapering if enabled
        if self.enable_tapering:
            kx = self.tapering()[..., 0].unsqueeze(-1)
            ky = self.tapering()[..., 1].unsqueeze(-1)
            
            fx = safe_mul(kx/sz, z) + 1
            fy = safe_mul(ky/sz, z) + 1
            
            fx = torch.where(fx > 0, 1.0, -1.0) * torch.clamp(torch.abs(fx), min=eps)
            fy = torch.where(fy > 0, 1.0, -1.0) * torch.clamp(torch.abs(fy), min=eps)
            
            x = x / fx
            y = y / fy
        
        term1 = safe_pow(safe_pow(x / sx, 2), 1 / e2)
        term2 = safe_pow(safe_pow(y / sy, 2), 1 / e2)
        term3 = safe_pow(safe_pow(z / sz, 2), 1 / e1)
        
        f_func = safe_pow(safe_pow(term1 + term2, e2 / e1) + term3, -e1 / 2)
        
        sdf = safe_mul(r0, (1 - f_func))
        
        # Enforce proper distance growth outside the primitive:
        # SDF should be at least r0 - radius_approx (linear growth outside bounding sphere)
        # This prevents tiny primitives from generating incorrect near-zero SDF values
        radius = torch.sqrt(sx.squeeze(-1)**2 + sy.squeeze(-1)**2 + sz.squeeze(-1)**2)  # (B, N)
        floor = r0 - radius.unsqueeze(-1)
        sdf = torch.where(floor > sdf, floor, sdf)

        return sdf
    
    @torch.compile
    def poles(self):
        B, N, _ = self.scale().shape

        s = self.scale()      # (B,N,3)
        R = self.rotation()   # (B,N,3,3)
        t = self.translation  # (B,N,3)
        
        if self.enable_tapering:
            taper = self.tapering() # (B,N,2)
            sx = s[..., 0] * (1 + torch.abs(taper[..., 0]))
            sy = s[..., 1] * (1 + torch.abs(taper[..., 1]))
            sz = s[..., 2]
        else:
            sx = s[..., 0]
            sy = s[..., 1]
            sz = s[..., 2]

        local = torch.zeros(B, N, 6, 3, device=s.device)
        local[:,:,0,0] =  sx  # +x
        local[:,:,1,0] = -sx  # -x
        local[:,:,2,1] =  sy  # +y
        local[:,:,3,1] = -sy  # -y
        local[:,:,4,2] =  sz  # +z
        local[:,:,5,2] = -sz  # -z

        # rotate
        poles_world = torch.matmul(
            R.unsqueeze(2),     # (B,N,1,3,3)
            local.unsqueeze(-1) # (B,N,6,3,1)
        ).squeeze(-1)

        # translate
        poles_world = poles_world + t.unsqueeze(2)
        return poles_world
    
    @torch.compile
    def _compute_losses(self, sdfs, overlap):
        temperature = 1e-3
        sdfs_iou = sdfs[:, :self.M_points_iou]
        pred_occ = torch.sigmoid(-sdfs_iou / temperature)
        gt_occ = self.occupancies.float()

        intersection = (pred_occ * gt_occ).sum(dim=1)
        union = (pred_occ + gt_occ - pred_occ * gt_occ).sum(dim=1)
        iou = intersection / torch.clamp(union, min=1.0)

        # SDF Loss on surface points
        temperature = 1e-2
        sdfs_surf = sdfs[:, self.M_points_iou:]
        # sdfs_surf = (torch.sigmoid(sdfs_surf / temperature) - 0.5) * 2 * self.truncation
        sdfs_surf = self.truncation * torch.tanh(sdfs_surf / self.truncation)
        # Lsdf = 8 * torch.mean(torch.abs(sdfs_surf), dim=1) # (B,)
        # Lsdf = 16 * torch.mean(torch.relu(sdfs_surf), dim=1) # (B,)
        Lsdf = 16 * torch.mean(torch.abs(torch.nn.LeakyReLU()(sdfs_surf)), dim=1) # (B,)
        
        poles = self.poles()   # (B,N,6,3)
        mask = self.exist_mask.unsqueeze(-1)
        violation = torch.relu(self.bbox_min - poles) + torch.relu(poles - self.bbox_max)   # (B,N,6,3)
        dist = torch.linalg.norm(violation, dim=-1)  # (B,N,6)
        L_bbox = (dist * mask).sum(dim=2).mean(dim=1)

        L_overlap = 1e-1 * overlap.mean(dim=1)
        
        loss = -torch.log(iou) + self.w_sdf * Lsdf + self.w_bbox * L_bbox + self.w_overlap * L_overlap
        return loss, iou, Lsdf, L_bbox, L_overlap
    
    def compute_losses(self, forward_out):
        sdfs = forward_out.get('sdfs')
        overlap = forward_out.get('overlap')
        loss, iou, Lsdf, L_bbox, L_overlap = self._compute_losses(sdfs, overlap)
        return loss, {"iou": iou, "sdf": Lsdf, "bbox": L_bbox, "overlap": L_overlap}

    def forward(self):
        all_sdfs = self.sdf_batch(self.points.transpose(1, 2).contiguous()) # (B, N, M_total)
    
        # mask out non-existing primitives
        mask = self.exist_mask.unsqueeze(-1).expand_as(all_sdfs)
        all_sdfs[~mask] = float('inf')

        # Union is min(sdf)
        # min_sdf, _ = torch.min(all_sdfs, dim=1) # (B, M)
        tau = 0.01  # smaller = sharper min
        min_sdf = -tau * torch.logsumexp(-all_sdfs / tau, dim=1)  # (B, M)

        temperature = 1e-3
        indicator = torch.sigmoid(-(all_sdfs+self.truncation) / temperature)
        overlap = torch.relu(torch.sum(indicator, dim=1) -1)
        return {
            'sdfs': min_sdf,
            'overlap': overlap,
        }

    @torch.compile
    def prune(self):
        with torch.no_grad():
            all_sdfs = self.sdf_batch(self.points.transpose(1, 2)[:, :, self.M_points_iou:].contiguous())
            mask = self.exist_mask.unsqueeze(-1).expand_as(all_sdfs)
            all_sdfs[~mask] = float('inf')
            idx_points = torch.argmin(all_sdfs, dim=1) 
            counts_points = torch.zeros(all_sdfs.shape[0], all_sdfs.shape[1], device=all_sdfs.device)
            for b in range(all_sdfs.shape[0]):
                counts_points[b] = torch.bincount(idx_points[b], minlength=self.N_max).float()
            return (counts_points <= 5) & self.exist_mask

    def update_handler(self, denormalize=True, compute_meshes=True):
        if self.enable_pruning:
            prune_mask = self.prune()
            self.exist_mask &= ~prune_mask
            prune_mask = prune_mask.cpu()
        for i, idx in enumerate(self.indices):
            if self.enable_pruning: self.pred_handler.exist[idx][prune_mask[i]] = 0.
            mask = self.exist_mask[i].cpu().numpy()
            if not np.any(mask): continue
            
            # Denormalize
            s = self.scale()[i][mask].detach().cpu().numpy()
            t = self.translation[i][mask].detach().cpu().numpy()
            if denormalize:
                norm_trans = self.normalization_translation[i].detach().cpu().numpy()
                norm_scale = self.normalization_scale[i].detach().cpu().numpy()
                s = s * norm_scale
                t = (t * norm_scale) + norm_trans

            self.pred_handler.scale[idx][mask] = s
            self.pred_handler.translation[idx][mask] = t
            self.pred_handler.exponents[idx][mask] = self.exponents()[i][mask].detach().cpu().numpy()
            self.pred_handler.rotation[idx][mask] = self.rotation()[i][mask].detach().cpu().numpy()

            if self.enable_tapering:
                self.pred_handler.tapering[idx][mask] = self.tapering()[i][mask].detach().cpu().numpy()
            if self.enable_bending:
                kb, alpha = self.bending()
                combined = torch.stack([kb[i], alpha[i]], dim=-1).flatten(start_dim=1)
                self.pred_handler.bending[idx][mask] = combined[mask].detach().cpu().numpy()
            
        if compute_meshes:
            meshes = [self.pred_handler.get_mesh(idx, resolution=30) for idx in self.indices]
            return self.pred_handler, meshes
        else:
            return self.pred_handler

