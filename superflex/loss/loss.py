import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment

from superflex.loss.utils import parametric_to_points_extended, sdf_batch
from superflex.loss.sampler import EqualDistanceSamplerSQ

class LossGeometric(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # Loss weights
        self.w_exist = getattr(cfg, 'w_exist', 1.0)
        self.w_geometric = getattr(cfg, 'w_geometric', 1.0)
        self.w_geometric_f = getattr(cfg, 'w_geometric_f', 1.0)
        self.w_iou = getattr(cfg, 'w_iou', 0.0)
        
        # Hungarian matching costs
        self.c_geometric = getattr(cfg, 'c_geometric', 1.0)
        self.c_z_align = getattr(cfg, 'c_z_align', 0.0)
        self.c_exist = getattr(cfg, 'c_exist', 0.0)
        
        # Sampling
        self.sampler = EqualDistanceSamplerSQ(n_samples=cfg.n_samples, D_eta=0.05, D_omega=0.05)
        self.n_samples_cost = getattr(cfg, 'n_samples_cost', 64)
    
    def geometric_error_cd(self, pred, gt, mask=None):
        """Chamfer distance for geometric matching."""
        if mask is None:
            B, P, _, _ = gt.shape
            gt_ext = gt.unsqueeze(1).expand(B, P, P, -1, 3)
            dists = torch.norm(pred.unsqueeze(4) - gt_ext.unsqueeze(3), dim=-1)
            mean_pred_to_gt = dists.min(dim=4)[0].mean(dim=3)  # (B, P_pred, P_gt)
            mean_gt_to_pred = dists.min(dim=3)[0].mean(dim=3)  # (B, P_pred, P_gt)
            chamfer = mean_pred_to_gt + mean_gt_to_pred
            return chamfer

        # Masked mode: compute per-primitive chamfer and aggregate using mask
        dists = torch.norm(pred.unsqueeze(3) - gt.unsqueeze(2), dim=-1)  # (B, P, Np, Ng)
        mean_pred_to_gt = dists.min(dim=3)[0].mean(dim=2)  # (B, P)
        mean_gt_to_pred = dists.min(dim=2)[0].mean(dim=2)  # (B, P)
        chamfer_per_prim = mean_pred_to_gt + mean_gt_to_pred  # (B, P)
        return (chamfer_per_prim * mask.squeeze(-1)).sum() / (mask.sum() + 1e-6)
        
    def sample(self, scale, shape):
        """Sample eta/omega parameters for superquadric reconstruction."""
        etas, omegas = self.sampler.sample_on_batch(
            scale.detach().cpu().numpy(),
            shape.detach().cpu().numpy()
        )
        # Make sure we don't get nan for gradients
        etas[etas == 0] += 1e-6
        omegas[omegas == 0] += 1e-6

        # Move to tensors
        etas = scale.new_tensor(etas)
        omegas = scale.new_tensor(omegas)
        return etas, omegas 

    def forward(self, batch, out_dict):
        gt_scale = batch['gt_scale'].cuda().float()
        gt_shape = batch['gt_shape'].cuda().float()
        gt_trans = batch['gt_trans'].cuda().float()
        gt_rotate = batch['gt_rotate'].cuda().float()
        gt_rotate_q = batch['gt_rotate_q'].cuda().float()
        gt_exist = batch['gt_exist'].cuda().float()
        gt_tapering = batch['gt_tapering'].cuda().float()
        gt_bending = batch['gt_bending'].cuda().float()
        
        B, P = gt_scale.shape[:2]
        
        # Sample and compute points for GT
        gt_sq_etas, gt_sq_omegas = self.sample(gt_scale, gt_shape)
        gt_pts = parametric_to_points_extended(
            gt_trans,
            gt_rotate,
            gt_scale,
            gt_shape,
            gt_tapering,
            gt_bending[..., 0::2],
            gt_bending[..., 1::2],
            gt_sq_etas,
            gt_sq_omegas
        )

        # Sample and compute points for Pred
        pred_sq_etas, pred_sq_omegas = self.sample(out_dict['scale'], out_dict['shape'])
        pred_pts = parametric_to_points_extended(
            out_dict['trans'], 
            out_dict['rotate'], 
            out_dict['scale'],
            out_dict['shape'],
            out_dict['tapering'],
            out_dict['bending_k'],
            out_dict['bending_a'],
            pred_sq_etas,
            pred_sq_omegas
        )

        # Hungarian matching
        with torch.no_grad():
            device = gt_scale.device
            C = torch.zeros((B, P, P), device=device)

            if self.c_geometric > 0:
                # For cost computation downsample points
                n_pts = gt_pts.shape[-2]
                gt_pts_ds = gt_pts
                pred_pts_ds = pred_pts
                if n_pts > 64:
                    idx = torch.linspace(0, n_pts - 1, steps=self.n_samples_cost, dtype=torch.int, device=gt_pts.device)
                    gt_pts_ds = gt_pts.index_select(-2, idx)
                    pred_pts_ds = pred_pts.index_select(-2, idx)
                
                pred_pts_ds_ext = pred_pts_ds.unsqueeze(2).expand(-1, -1, P, -1, -1) # [B, P_pred, P_gt, N_ds, 3]
                cost_geometric = self.c_geometric * self.geometric_error_cd(pred_pts_ds_ext, gt_pts_ds)
                C += cost_geometric

            if self.c_z_align > 0:
                z_pred = out_dict['rotate'][..., 2]  # (B, P_pred, 3)
                z_gt = gt_rotate[..., 2]            # (B, P_gt, 3)
                # pairwise dot product -> (B, P_pred, P_gt)
                dot = torch.einsum('bpi,bqi->bpq', z_pred, z_gt)
                cost_z = self.c_z_align * (1.0 - torch.abs(dot))
                C += cost_z

            if self.c_exist > 0:
                pred_exist_exp = out_dict['exist'].view(B, P, 1).expand(B, P, P)
                gt_exist_exp = gt_exist.view(B, 1, P).expand(B, P, P)
                cost_exist = self.c_exist * F.binary_cross_entropy(pred_exist_exp, gt_exist_exp, reduction='none')
                C += cost_exist

            C_np = C.cpu().numpy()
            
            indices = []
            for b in range(B):
                row_ind, col_ind = linear_sum_assignment(C_np[b])
                indices.append(col_ind)
                
            indices = torch.tensor(np.array(indices), device=gt_scale.device)
            batch_idx = torch.arange(B, device=gt_scale.device).unsqueeze(1).expand(B, P)
            gt_pts = gt_pts[batch_idx, indices]
            gt_scale = gt_scale[batch_idx, indices]
            gt_shape = gt_shape[batch_idx, indices]
            gt_trans = gt_trans[batch_idx, indices]
            gt_rotate = gt_rotate[batch_idx, indices]
            gt_exist = gt_exist[batch_idx, indices]
            gt_tapering = gt_tapering[batch_idx, indices]
            gt_bending = gt_bending[batch_idx, indices]
            gt_sq_etas = gt_sq_etas[batch_idx, indices]
            gt_sq_omegas = gt_sq_omegas[batch_idx, indices]
        
        loss = 0
        loss_dict = {}            
        mask = (gt_exist > 0.5).float()

        # Existance (binary cross-entropy)
        exist_loss = nn.BCELoss()(out_dict['exist'], (gt_exist > 0.5).float())
        loss += self.w_exist * exist_loss
        loss_dict['param_exist'] = exist_loss.item()
        
        # Geometric
        # geometric_loss = self.geometric_error_cd(pred_pts, gt_pts, gt_exist)
        geometric_loss = self.geometric_error_cd(pred_pts, gt_pts, mask)
        loss += self.w_geometric * geometric_loss
        loss_dict['param_geometric'] = geometric_loss.item()
        
        if self.w_geometric_f > 0:
            z_t = torch.zeros_like(gt_tapering)
            z_bk = torch.zeros_like(gt_bending[..., 0::2])
            z_ba = torch.zeros_like(gt_bending[..., 1::2])
            gt_pts = parametric_to_points_extended(
                gt_trans,
                gt_rotate,
                gt_scale,
                gt_shape,
                z_t,
                z_bk,
                z_ba,
                gt_sq_etas,
                gt_sq_omegas
            )
            
            pred_pts_forced = parametric_to_points_extended(
                gt_trans, 
                out_dict['rotate'], 
                out_dict['scale'],
                out_dict['shape'],
                z_t,
                z_bk,
                z_ba,
                pred_sq_etas,
                pred_sq_omegas
            )
            # geometric_loss_forced = self.geometric_error_cd(pred_pts_forced, gt_pts, gt_exist)
            geometric_loss_forced = self.geometric_error_cd(pred_pts_forced, gt_pts, mask)
            loss += self.w_geometric_f * geometric_loss_forced
            loss_dict['param_geometric_forced'] = geometric_loss_forced.item()
        
        # IoU Loss
        if self.w_iou > 0:
            points_iou = batch['points_iou'].cuda().float()
            gt_occ = batch['occupancies'].cuda().bool()
            exist_weights = out_dict['exist']  # (B, N, 1)
            all_sdfs_iou = sdf_batch(points_iou, out_dict)  # (B, N, M_iou)
            temperature_iou = 1e-3
            prob_prim = torch.sigmoid(-all_sdfs_iou / temperature_iou)
            p_occupied = exist_weights * prob_prim  # (B, N, M_iou)
            p_occupied = torch.clamp(p_occupied, min=0.0, max=0.9999)
            pred_occ = 1.0 - torch.exp(torch.sum(torch.log(1.0 - p_occupied), dim=1))  # (B, M_iou)
            intersection = (pred_occ * gt_occ).sum(dim=1)
            union = (pred_occ + gt_occ - pred_occ * gt_occ).sum(dim=1)
            iou = (intersection + 1e-6) / (torch.clamp(union, min=1.0) + 1e-6)
            L_iou = -torch.log(iou).mean()
            loss += self.w_iou * L_iou
            loss_dict['iou'] = iou.mean().item()
            loss_dict['iou_loss'] = L_iou.item()

        # stop trainer from complaining about unused parameters (TODO fix this)
        loss += 0 * out_dict['assign_matrix'].mean()
        loss_dict['all'] = loss.item()
        return loss, loss_dict

class IoULoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # BaseLoss parameters
        self.w_sps = getattr(cfg, 'w_sps', 0.0)
        self.w_ext = getattr(cfg, 'w_ext', 0.0)
        
        # IoULoss specific parameters
        self.truncation = getattr(cfg, 'truncation', 0.05)
        self.leaky_relu = getattr(cfg, 'leaky_relu', True)
        self.w_iou = getattr(cfg, 'w_iou', 1.0)
        self.w_sdf = getattr(cfg, 'w_sdf', 1.0)
        self.w_bbox = getattr(cfg, 'w_bbox', 1.0)
        self.w_overlap = getattr(cfg, 'w_overlap', 1.0)
    
    def compute_existence_loss(self, assign_matrix, exist):
        thred = 24
        loss = nn.BCELoss().cuda()
        gt = (assign_matrix.sum(1) > thred).to(torch.float32).detach()
        entropy = loss(exist.squeeze(-1), gt)
        return entropy

    def get_sparsity_loss(self, assign_matrix):
        num_points = assign_matrix.shape[1]
        norm_05 = (assign_matrix.sum(1)/num_points + 0.01).sqrt().mean(1).pow(2)
        norm_05 = torch.mean(norm_05)
        return norm_05

    def compute_common_losses(self, out_dict, loss_dict):
        loss = 0
        loss_dict['expected_prim_num'] = out_dict['exist'].squeeze(-1).sum(-1).mean().data.detach().item()

        if self.w_ext > 0:
            exist_loss = self.compute_existence_loss(out_dict['assign_matrix'], out_dict['exist'])
            loss += self.w_ext * exist_loss
            loss_dict['exist_loss'] = exist_loss.item()

        if self.w_sps > 0:
            sparsity_loss = self.get_sparsity_loss(out_dict['assign_matrix'])
            loss += self.w_sps * sparsity_loss
            loss_dict['sparsity_loss'] = sparsity_loss.item()
            
        return loss
    
    def poles(self, scale, rotate, trans):
        B, N, _ = scale.shape
        local = torch.zeros(B, N, 6, 3, device=scale.device)
        local[:,:,0,0] =  scale[:,:,0]  # +x
        local[:,:,1,0] = -scale[:,:,0]  # -x
        local[:,:,2,1] =  scale[:,:,1]  # +y
        local[:,:,3,1] = -scale[:,:,1]  # -y
        local[:,:,4,2] =  scale[:,:,2]  # +z
        local[:,:,5,2] = -scale[:,:,2]  # -z

        # rotate
        poles_world = torch.matmul(
            rotate.unsqueeze(2),     # (B,N,1,3,3)
            local.unsqueeze(-1) # (B,N,6,3,1)
        ).squeeze(-1)

        # translate
        poles_world = poles_world + trans.unsqueeze(2)
        return poles_world

    def forward(self, batch, out_dict):
        pc, normals = batch['points'].cuda().float(), batch['normals'].cuda().float()
        points_iou, gt_occ = batch['points_iou'].cuda().float(), batch['occupancies'].cuda().bool()

        # pc: (B, M_surf, 3)
        loss = 0
        loss_dict = {}
        
        # 1. SDF on Surface Points
        out_dict['scale'] = torch.clamp(out_dict['scale'], min=0.01) # when scale is too low sdf calculations fail
        all_sdfs_surf = sdf_batch(pc, out_dict, use_floor=True) #(B, N, M_surf)
        exist_probs = out_dict['exist'].reshape(all_sdfs_surf.shape[0], all_sdfs_surf.shape[1]) # (B, N)
        exist_weights = exist_probs.unsqueeze(-1) # (B, N, 1)
        assign_probs = out_dict['assign_matrix'].transpose(1, 2) #(B, N, M_surf)
        
        # Soft truncation using tanh preserves unit gradients near the surface (SDF=0)
        sdfs_surf_trunc = self.truncation * torch.tanh(all_sdfs_surf / self.truncation)
        if self.leaky_relu:
            if isinstance(self.leaky_relu, float):
                relu_sdfs_trunc = torch.abs(torch.nn.LeakyReLU(negative_slope=self.leaky_relu)(sdfs_surf_trunc))
            else:
                relu_sdfs_trunc = torch.abs(torch.nn.LeakyReLU()(sdfs_surf_trunc)) 
        else:
            relu_sdfs_trunc = torch.abs(sdfs_surf_trunc) 
        weighted_relu_sdf = (assign_probs * relu_sdfs_trunc).sum(dim=1) #(B, M_surf)
        L_sdf = 16 * weighted_relu_sdf.mean()

        loss += self.w_sdf * L_sdf
        loss_dict['sdf'] = L_sdf.item()
        
        # 2. BBox Loss
        bbox_min = torch.min(pc, dim=1).values.unsqueeze(1).unsqueeze(2) # (B, 1, 1, 3)
        bbox_max = torch.max(pc, dim=1).values.unsqueeze(1).unsqueeze(2)
        margin = (bbox_max - bbox_min) * .025
        bbox_min -= margin
        bbox_max += margin
        
        poles = self.poles(out_dict['scale'], out_dict['rotate'], out_dict['trans']) # (B, N, 6, 3)
        violation = torch.relu(bbox_min - poles) + torch.relu(poles - bbox_max) # (B, N, 6, 3)
        dist = torch.linalg.norm(violation, dim=-1) # (B, N, 6)
        L_bbox = (dist * exist_weights).sum(dim=2).mean() # Mean over batch
        
        loss += self.w_bbox * L_bbox
        loss_dict['bbox'] = L_bbox.item()

        # 3. IoU Loss
        all_sdfs_iou = sdf_batch(points_iou, out_dict, use_floor=True) # (B, N, M_iou)
        
        temperature_iou = 1e-3
        prob_prim = torch.sigmoid(-all_sdfs_iou / temperature_iou)
        p_occupied = exist_weights * prob_prim # (B, N, M_iou)
        
        # Union probability: P(union) = 1 - prod(1 - p_i)
        # implementation: 1 - exp(sum(log(1 - p_i)))
        # clamp p_occupied to max 1-eps to avoid log(0)
        p_occupied = torch.clamp(p_occupied, min=0.0, max=0.9999)
        pred_occ = 1.0 - torch.exp(torch.sum(torch.log(1.0 - p_occupied), dim=1)) # (B, M_iou)
        intersection = (pred_occ * gt_occ).sum(dim=1)
        union = (pred_occ + gt_occ - pred_occ * gt_occ).sum(dim=1)
        iou = (intersection + 1e-6) / (torch.clamp(union, min=1.0) + 1e-6)
        L_iou = -torch.log(iou).mean()
        
        loss += self.w_iou * L_iou
        loss_dict['iou'] = iou.mean().item()
        loss_dict['iou_loss'] = L_iou.item()

        # 4. Overlap/Regularization Loss
        temperature_overlap = 1e-3
        indicator = torch.sigmoid(-(all_sdfs_iou + self.truncation) / temperature_overlap)
        weighted_indicator = (exist_weights > 0.5) * indicator # (B, N, M)
        overlap_count = torch.relu(torch.sum(weighted_indicator, dim=1) - 1) # (B, M)
        L_overlap = overlap_count.mean()
        
        loss += self.w_overlap * L_overlap
        loss_dict['overlap'] = L_overlap.item()

        loss += self.compute_common_losses(out_dict, loss_dict)
        loss_dict['all'] = loss.item()
        return loss, loss_dict

class Loss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        loss_type = getattr(cfg, 'type', 'original')
        print(f"Initializing Loss: {loss_type}")
        
        if loss_type == 'geom':
            self.impl = LossGeometric(cfg)
        elif loss_type == 'iou':
            self.impl = IoULoss(cfg)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, batch, out_dict):
        return self.impl(batch, out_dict)