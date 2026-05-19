import torch
import torch.nn as nn
import trimesh
import numpy as np
from scipy.spatial import KDTree
from superflex.data.dataloader import ShapeNet, ScenesDataset, ABO, ASE_Object
from torch.utils.data import DataLoader
from superflex.utils.safe_operations import safe_pow, safe_mul

def build_dataloader(cfg):
    if cfg.dataloader.dataset == 'shapenet':
        ds = ShapeNet(split=cfg.shapenet.split, cfg=cfg)
    elif cfg.dataloader.dataset == 'abo':
        ds = ABO(split=cfg.abo.split, cfg=cfg)
    elif cfg.dataloader.dataset == 'ase_object':
        ds = ASE_Object(split=cfg.ase_object.split, cfg=cfg)
    elif cfg.dataloader.dataset == 'scenes_dataset':
        ds = ScenesDataset(cfg=cfg)
    else:
        raise ValueError(f"Unsupported dataset {cfg.dataloader.dataset}")

    return _build_dataloader(cfg, ds)

def _build_dataloader(cfg, ds):
    dl = DataLoader(
        ds, batch_size=cfg.dataloader.batch_size, shuffle=False,
        num_workers=cfg.dataloader.num_workers, pin_memory=True
    )
    return dl

def check_mesh_contains(mesh, points):
    return mesh.contains(points)

def compute_iou(occ1, occ2):
    ''' Computes the Intersection over Union (IoU) value for two sets of
    occupancy values.

    Args:
        occ1 (tensor): first set of occupancy values
        occ2 (tensor): second set of occupancy values
    '''
    # occ1 = np.asarray(occ1)
    # occ2 = np.asarray(occ2)

    # Put all data in second dimension
    # Also works for 1-dimensional data
    if occ1.ndim >= 2:
        occ1 = occ1.reshape(occ1.shape[0], -1)
    if occ2.ndim >= 2:
        occ2 = occ2.reshape(occ2.shape[0], -1)

    # Convert to boolean values
    occ1 = (occ1 >= 0.5)
    occ2 = (occ2 >= 0.5)

    # Compute IOU
    area_union = (occ1 | occ2).astype(np.float32).sum(axis=-1)
    # Avoid division by zero
    area_union = np.maximum(area_union, 1e-6)
    
    area_intersect = (occ1 & occ2).astype(np.float32).sum(axis=-1)

    iou = (area_intersect / area_union)

    return iou

def eval_mesh(mesh, pointcloud_tgt, normals_tgt,
              points_iou=None, occ_tgt=None):
    ''' Evaluates a mesh.

    Args:
        mesh (trimesh): mesh which should be evaluated
        pointcloud_tgt (numpy array): target point cloud
        normals_tgt (numpy array): target normals
        points_iou (numpy_array): points tensor for IoU evaluation
        occ_tgt (numpy_array): GT occupancy values for IoU points
    '''
    if len(mesh.vertices) != 0 and len(mesh.faces) != 0:
        pointcloud, idx = mesh.sample(pointcloud_tgt.shape[0], return_index=True)
        pointcloud = pointcloud.astype(np.float32)
        normals = mesh.face_normals[idx]
    else:
        pointcloud = np.empty((0, 3))
        normals = np.empty((0, 3))

    out_dict = get_outdict(pointcloud_tgt, normals_tgt, pointcloud, normals)

    if points_iou is not None and occ_tgt is not None and len(mesh.vertices) != 0 and len(mesh.faces) != 0:
        occ = check_mesh_contains(mesh, points_iou)
        out_dict['iou'] = compute_iou(occ, occ_tgt)
    else:
        out_dict['iou'] = 0.

    return out_dict

def get_threshold_percentage(dist, thresholds):
    ''' Evaluates a point cloud.

    Args:
        dist (numpy array): calculated distance
        thresholds (numpy array): threshold values for the F-score calculation
    '''
    in_threshold = [
        (dist <= t).mean() for t in thresholds
    ]
    return in_threshold

def distance_p2p(points_src, normals_src, points_tgt, normals_tgt): # from convolutional occupancy network
    ''' Computes minimal distances of each point in points_src to points_tgt.

    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    kdtree = KDTree(points_tgt)
    dist, idx = kdtree.query(points_src)

    if normals_src is not None and normals_tgt is not None:
        normals_src = \
            normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = \
            normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array(
            [np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product

def get_outdict(pc_gt, normals_gt, pc_pred, normals_pred):
    thresholds = np.linspace(1./1000, 1, 1000)
    completeness, completeness_normals = distance_p2p(pc_gt, normals_gt, pc_pred, normals_pred)
    recall = get_threshold_percentage(completeness, thresholds)
    completeness2 = completeness**2
    
    completeness = completeness.mean()
    completeness2 = completeness2.mean()
    completeness_normals = completeness_normals.mean()
    
    # Accuracy: how far are th points of the predicted pointcloud
    # from the target pointcloud
    accuracy, accuracy_normals = distance_p2p(pc_pred, normals_pred, pc_gt, normals_gt)
    precision = get_threshold_percentage(accuracy, thresholds)
    accuracy2 = accuracy**2
    
    accuracy = accuracy.mean()
    accuracy2 = accuracy2.mean()
    accuracy_normals = accuracy_normals.mean()
    
    # Chamfer distance
    chamferL2 = 0.5 * (completeness2 + accuracy2)
    normals_correctness = (
        0.5 * completeness_normals + 0.5 * accuracy_normals
    )
    chamferL1 = 0.5 * (completeness + accuracy)
    F = [
        2 * precision[i] * recall[i] / (precision[i] + recall[i] + 0.000001)
        for i in range(len(precision))
    ]

    
    out_dict_cur = {
            'completeness': completeness,
            'accuracy': accuracy,
            'normals completeness': completeness_normals,
            'normals accuracy': accuracy_normals,
            'normals': normals_correctness,
            'completeness2': completeness2,
            'accuracy2': accuracy2,
            'chamfer-L2': chamferL2,
            'chamfer-L1': chamferL1,
            'f-score': F[9], # threshold = 1.0%
            'f-score-15': F[14], # threshold = 1.5%
            'f-score-20': F[19], # threshold = 2.0%
        }
    return out_dict_cur

def sdfs_from_outdict(outdict, points, device='cuda'):
    """Compute per-primitive SDFs for a batch of objects from an outdict.

    Args:
        outdict: Dictionary containing primitive parameters as torch tensors.
        points: torch tensor or numpy array with shape (B, M, 3) of query points.
        device: torch device string.

    Returns:
        sdfs: torch tensor of shape (B, N, M) with per-primitive SDF values.
    """
    if isinstance(points, np.ndarray):
        points = torch.tensor(points, dtype=torch.float32, device=device)
    else:
        points = points.to(device=device, dtype=torch.float32)

    B = points.shape[0]
    M = points.shape[1]
    N = outdict['scale'].shape[1]

    scales = outdict['scale'].to(device)
    exps = outdict['shape'].to(device)
    R = outdict['rotate'].to(device)
    t = outdict['trans'].to(device)
    
    if 'tapering' in outdict:
        taper = outdict['tapering'].to(device)
    else:
        taper = torch.zeros((B, N, 2), dtype=torch.float32, device=device)
        
    if 'bending' in outdict and outdict['bending'].shape[-1] == 6:
        bending = outdict['bending'].to(device)
    elif 'bending_k' in outdict and 'bending_a' in outdict:
        bending = torch.stack([
            outdict['bending_k'][..., 0], outdict['bending_a'][..., 0],
            outdict['bending_k'][..., 1], outdict['bending_a'][..., 1],
            outdict['bending_k'][..., 2], outdict['bending_a'][..., 2]
        ], -1).to(device)
    else:
        bending = torch.zeros((B, N, 6), dtype=torch.float32, device=device)

    # Prepare points: (B, M, 3) -> (B, 1, 3, M)
    points_expanded = points.permute(0, 2, 1).unsqueeze(1)
    t_uns = t.unsqueeze(-1)
    points_centered = points_expanded - t_uns

    # Rotate into local primitive frames: (B,N,3,M)
    X = torch.matmul(R.transpose(-2, -1), points_centered)

    e1 = exps[..., 0].unsqueeze(-1)
    e2 = exps[..., 1].unsqueeze(-1)
    sx = scales[..., 0].unsqueeze(-1)
    sy = scales[..., 1].unsqueeze(-1)
    sz = scales[..., 2].unsqueeze(-1)

    x = X[:, :, 0, :]
    y = X[:, :, 1, :]
    z = X[:, :, 2, :]

    eps = 1e-6
    x = torch.where(x > 0, 1.0, -1.0) * torch.clamp(torch.abs(x), min=eps)
    y = torch.where(y > 0, 1.0, -1.0) * torch.clamp(torch.abs(y), min=eps)
    z = torch.where(z > 0, 1.0, -1.0) * torch.clamp(torch.abs(z), min=eps)

    r0 = torch.sqrt(x**2 + y**2 + z**2)

    # Bending stored as [kb_z, alpha_z, kb_x, alpha_x, kb_y, alpha_y]
    kb = bending[..., 0::2]
    alpha = bending[..., 1::2]

    def inverse_bending_axis(x, y, z, kb_axis, alpha_axis, axis):
        if axis == 'z':
            u, v, w = x, y, z
        elif axis == 'x':
            u, v, w = y, z, x
        elif axis == 'y':
            u, v, w = z, x, y
        else:
            return x, y, z

        mask = torch.abs(kb_axis) > 1e-3

        inv_kb = 1.0 / (kb_axis + 1e-6)

        angle_offset = torch.atan2(v, u)
        Rval = torch.sqrt(u**2 + v**2) * torch.cos(alpha_axis - angle_offset)
        gamma = torch.atan2(w, inv_kb - Rval)
        r = inv_kb - torch.sqrt(w**2 + (inv_kb - Rval)**2)

        u_new = torch.where(mask, u - (Rval - r) * torch.cos(alpha_axis), u)
        v_new = torch.where(mask, v - (Rval - r) * torch.sin(alpha_axis), v)
        w_new = torch.where(mask, inv_kb * gamma, w)

        if axis == 'z':
            return u_new, v_new, w_new
        elif axis == 'x':
            return w_new, u_new, v_new
        elif axis == 'y':
            return v_new, w_new, u_new

    x, y, z = inverse_bending_axis(x, y, z, kb[..., 0].unsqueeze(-1), alpha[..., 0].unsqueeze(-1), 'z')
    x, y, z = inverse_bending_axis(x, y, z, kb[..., 1].unsqueeze(-1), alpha[..., 1].unsqueeze(-1), 'x')
    x, y, z = inverse_bending_axis(x, y, z, kb[..., 2].unsqueeze(-1), alpha[..., 2].unsqueeze(-1), 'y')

    # Tapering
    kx = taper[..., 0].unsqueeze(-1)
    ky = taper[..., 1].unsqueeze(-1)

    fx = safe_mul(kx / sz, z) + 1
    fy = safe_mul(ky / sz, z) + 1

    fx = torch.where(fx > 0, 1.0, -1.0) * torch.clamp(torch.abs(fx), min=eps)
    fy = torch.where(fy > 0, 1.0, -1.0) * torch.clamp(torch.abs(fy), min=eps)

    x = x / fx
    y = y / fy

    term1 = safe_pow(safe_pow(x / sx, 2), 1 / e2)
    term2 = safe_pow(safe_pow(y / sy, 2), 1 / e2)
    term3 = safe_pow(safe_pow(z / sz, 2), 1 / e1)

    f_func = safe_pow(safe_pow(term1 + term2, e2 / e1) + term3, -e1 / 2)

    sdf = safe_mul(r0, (1 - f_func))
    
    # mask out non-existing primitives
    exist_masks = outdict['exist'] > 0.5
    mask = exist_masks.expand_as(sdf)
    sdf[~mask] = float('inf')
    return sdf

def sdfs_from_pred_handler(pred_handler, indices, points, device='cuda'):
    """Compute per-primitive SDFs for a batch of objects from a PredictionHandler.

    Args:
        pred_handler: PredictionHandler with primitive parameters stored as numpy arrays.
        indices: list of object indices to evaluate (length B).
        points: torch tensor or numpy array with shape (B, M, 3) of query points.
        device: torch device string.

    Returns:
        sdfs: torch tensor of shape (B, N, M) with per-primitive SDF values.
    """
    if isinstance(points, np.ndarray):
        points = torch.tensor(points, dtype=torch.float32, device=device)
    else:
        points = points.to(device=device, dtype=torch.float32)

    B = len(indices)
    M = points.shape[1]
    N = pred_handler.scale.shape[1]

    exist = torch.stack([torch.tensor(pred_handler.exist[i], dtype=torch.float32, device=device) for i in indices], dim=0)
    scales = torch.stack([torch.tensor(pred_handler.scale[i], dtype=torch.float32, device=device) for i in indices], dim=0)
    exps = torch.stack([torch.tensor(pred_handler.exponents[i], dtype=torch.float32, device=device) for i in indices], dim=0)
    R = torch.stack([torch.tensor(pred_handler.rotation[i], dtype=torch.float32, device=device) for i in indices], dim=0)
    t = torch.stack([torch.tensor(pred_handler.translation[i], dtype=torch.float32, device=device) for i in indices], dim=0)
    taper = torch.stack([torch.tensor(pred_handler.tapering[i], dtype=torch.float32, device=device) for i in indices], dim=0)
    bending = torch.stack([torch.tensor(pred_handler.bending[i], dtype=torch.float32, device=device) for i in indices], dim=0)

    outdict = {
        'exist': exist,
        'scale': scales,
        'shape': exps,
        'rotate': R,
        'trans': t,
        'tapering': taper,
        'bending': bending
    }
    return sdfs_from_outdict(outdict, points, device)

def compute_ious_sdf(sdfs, occupancies):
    min_sdf, _ = torch.min(sdfs, dim=1)
    pred_occ = (min_sdf <= 0)

    intersection = (pred_occ & occupancies).sum(dim=1).float()
    union = (pred_occ | occupancies).sum(dim=1).float()
    ious = (intersection / torch.clamp(union, min=1e-6)).cpu().numpy()
    return ious

def compute_ious_sdf_from_outdict(outdict, points_iou, occupancies, device='cuda'):
    """Compute IoU per object using SDF union from `outdict` parameters."""
    assert occupancies.dtype == torch.bool
    
    sdfs = sdfs_from_outdict(outdict, points_iou, device=device)
    return compute_ious_sdf(sdfs, occupancies)

def compute_ious_sdf_from_handler(pred_handler, indices, points_iou, occupancies, device='cuda'):
    """Compute IoU per object using SDF union from `pred_handler` parameters."""
    assert occupancies.dtype == torch.bool
    
    sdfs = sdfs_from_pred_handler(pred_handler, indices, points_iou, device=device)
    return compute_ious_sdf(sdfs, occupancies)