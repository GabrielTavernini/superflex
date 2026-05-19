import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment

from superflex.loss.sampler import EqualDistanceSamplerSQ
from superflex.utils.safe_operations import safe_pow, safe_mul

def parametric_to_points_extended(
    translation,
    rotation,
    scale,
    shape,
    tapering,
    bending_k,
    bending_a,
    etas,
    omegas
):
    """
    Given the sampling steps in the parametric space, get the actual 3D points
    with tapering and bending deformations applied.
    """
    def fexp(x, p):
        return torch.sign(x)*(torch.abs(x)**p)
        
    def apply_bending_axis_pt(x_c, y_c, z_c, val_kb, val_alpha, axis):
        """Batched, differentiable bending along a given axis."""
        if axis == 'z':
            u, v, w = x_c, y_c, z_c
        elif axis == 'x':
            u, v, w = y_c, z_c, x_c
        elif axis == 'y':
            u, v, w = z_c, x_c, y_c
            
        sin_alpha = torch.sin(val_alpha)
        cos_alpha = torch.cos(val_alpha)
        
        beta = torch.atan2(v, u)
        r = torch.sqrt(u**2 + v**2) * torch.cos(val_alpha - beta)

        # Differentiable check to avoid division by zero when kb ~ 0
        mask = (torch.abs(val_kb) >= 1e-3)
        kb_safe = torch.where(mask, val_kb, val_kb.new_tensor(1e-3))
        
        inv_kb = 1.0 / kb_safe
        gamma = w * kb_safe
        rho = inv_kb - r
        R = inv_kb - rho * torch.cos(gamma)

        expr = (R - r)
        u_bent = u + expr * cos_alpha
        v_bent = v + expr * sin_alpha
        w_bent = rho * torch.sin(gamma)
        
        # Apply transformation only where kb >= 1e-3
        u_out = torch.where(mask, u_bent, u)
        v_out = torch.where(mask, v_bent, v)
        w_out = torch.where(mask, w_bent, w)
        
        if axis == 'z':
            return u_out, v_out, w_out
        elif axis == 'x':
            return w_out, u_out, v_out
        elif axis == 'y':
            return v_out, w_out, u_out

    B = scale.shape[0]  # batch size
    M = scale.shape[1]  # number of primitives

    # Make sure that all tensors have the right shape
    a1 = scale[:, :, 0].unsqueeze(-1)  # size BxMx1
    a2 = scale[:, :, 1].unsqueeze(-1)  # size BxMx1
    a3 = scale[:, :, 2].unsqueeze(-1)  # size BxMx1
    e1 = shape[:, :, 0].unsqueeze(-1)  # size BxMx1
    e2 = shape[:, :, 1].unsqueeze(-1)  # size BxMx1

    x = a1 * fexp(torch.cos(etas), e1) * fexp(torch.cos(omegas), e2)
    y = a2 * fexp(torch.cos(etas), e1) * fexp(torch.sin(omegas), e2)
    z = a3 * fexp(torch.sin(etas), e1)

    # Clamp magnitude to avoid NaN/Inf in normal computation
    x = ((x > 0).float() * 2 - 1) * torch.max(torch.abs(x), x.new_tensor(1e-6))
    y = ((y > 0).float() * 2 - 1) * torch.max(torch.abs(y), y.new_tensor(1e-6))
    z = ((z > 0).float() * 2 - 1) * torch.max(torch.abs(z), z.new_tensor(1e-6))

    # --- Apply Tapering ---
    kx = tapering[:, :, 0].unsqueeze(-1)
    
    ky = tapering[:, :, 1].unsqueeze(-1)
    z_norm = z / a3
    fx = kx * z_norm + 1.0
    fy = ky * z_norm + 1.0
    x = x * fx
    y = y * fy
    
    # --- Apply Bending ---
    x, y, z = apply_bending_axis_pt(
        x, y, z, 
        bending_k[:, :, 2].unsqueeze(-1), 
        bending_a[:, :, 2].unsqueeze(-1), 
        'y'
    )
    x, y, z = apply_bending_axis_pt(
        x, y, z, 
        bending_k[:, :, 1].unsqueeze(-1), 
        bending_a[:, :, 1].unsqueeze(-1), 
        'x'
    )
    x, y, z = apply_bending_axis_pt(
        x, y, z, 
        bending_k[:, :, 0].unsqueeze(-1), 
        bending_a[:, :, 0].unsqueeze(-1), 
        'z'
    )

    pts = torch.stack([x, y, z], -1)
    pts_world = torch.matmul(
        rotation.unsqueeze(2),    # (B, P, 1, 3, 3)
        pts.unsqueeze(-1)        # (B, P, N, 3, 1)
    ).squeeze(-1) + translation.unsqueeze(2)
    return pts_world

def inverse_bending_axis(x, y, z, kb, alpha, axis):
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

def sdf_batch(points, out_dict, use_floor=False):
    # points: (B, M, 3) -> expects transpose for math -> (B, 3, M) roughly
    points = points.transpose(1, 2) # (B, 3, M)
    
    scale = out_dict['scale'] # (B, N, 3)
    exponents = out_dict['shape'] # (B, N, 2)
    rotate = out_dict['rotate'] # (B, N, 3, 3)
    trans = out_dict['trans'] # (B, N, 3)

    B, _, M = points.shape
    N = scale.shape[1]
    
    points_expanded = points.unsqueeze(1) # (B, 1, 3, M)
    t = trans.unsqueeze(-1) # (B, N, 3, 1)
    points_centered = points_expanded - t # (B, N, 3, M)
    
    X = torch.matmul(rotate.transpose(-2, -1), points_centered) # (B, N, 3, M)
    
    e1 = exponents[..., 0].unsqueeze(-1)
    e2 = exponents[..., 1].unsqueeze(-1)
    sx = scale[..., 0].unsqueeze(-1)
    sy = scale[..., 1].unsqueeze(-1)
    sz = scale[..., 2].unsqueeze(-1)
    
    x = X[:, :, 0, :]
    y = X[:, :, 1, :]
    z = X[:, :, 2, :]
    
    eps = 1e-6
    x = torch.where(x > 0, 1.0, -1.0) * torch.clamp(torch.abs(x), min=eps)
    y = torch.where(y > 0, 1.0, -1.0) * torch.clamp(torch.abs(y), min=eps)
    z = torch.where(z > 0, 1.0, -1.0) * torch.clamp(torch.abs(z), min=eps)
    
    r0 = torch.sqrt(x**2 + y**2 + z**2)
    
    # Apply bending
    if 'bending_k' in out_dict and 'bending_a' in out_dict:
        bending_k = out_dict['bending_k'] # (B, N, 3)
        bending_a = out_dict['bending_a'] # (B, N, 3)
        
        x, y, z = inverse_bending_axis(x, y, z, bending_k[..., 0].unsqueeze(-1), bending_a[..., 0].unsqueeze(-1), 'z')
        x, y, z = inverse_bending_axis(x, y, z, bending_k[..., 1].unsqueeze(-1), bending_a[..., 1].unsqueeze(-1), 'x')
        x, y, z = inverse_bending_axis(x, y, z, bending_k[..., 2].unsqueeze(-1), bending_a[..., 2].unsqueeze(-1), 'y')
    
    # Apply tapering
    if 'tapering' in out_dict:
        tapering = out_dict['tapering'] # (B, N, 2)
        kx = tapering[..., 0].unsqueeze(-1)
        ky = tapering[..., 1].unsqueeze(-1)
        
        fx = safe_mul(kx/sz, z) + 1
        fy = safe_mul(ky/sz, z) + 1
        
        # Check tapering signs
        fx = torch.where(fx > 0, 1.0, -1.0) * torch.clamp(torch.abs(fx), min=eps)
        fy = torch.where(fy > 0, 1.0, -1.0) * torch.clamp(torch.abs(fy), min=eps)
        
        x = x / fx
        y = y / fy
    
    term1 = safe_pow(safe_pow(x / sx, 2), 1 / e2)
    term2 = safe_pow(safe_pow(y / sy, 2), 1 / e2)
    term3 = safe_pow(safe_pow(z / sz, 2), 1 / e1)
    
    f_func = safe_pow(safe_pow(term1 + term2, e2 / e1) + term3, -e1 / 2)
    
    sdf = safe_mul(r0, (1 - f_func))

    if use_floor:
        # Enforce proper distance growth outside the primitive:
        # SDF should be at least r0 - radius_approx (linear growth outside bounding sphere)
        # This prevents tiny primitives from generating incorrect near-zero SDF values
        radius = torch.sqrt(sx.squeeze(-1)**2 + sy.squeeze(-1)**2 + sz.squeeze(-1)**2)  # (B, N)
        floor = r0 - radius.unsqueeze(-1)
        sdf = torch.where(floor > sdf, floor, sdf)
    return sdf # (B, N, M)