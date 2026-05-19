import torch
import torch.nn.functional as F

def transform_to_primitive_frame(pc_or_normals, trans, rotate):
    B, N, _ = pc_or_normals.shape
    P = trans.shape[1]

    centered = pc_or_normals.unsqueeze(1).repeat(1, P, 1, 1) - trans.unsqueeze(2).repeat(1,1,N,1)
    centered = centered.permute(0,1,3,2)
    rotate_T = rotate.permute(0, 1, 3, 2)
    transformed = torch.einsum('abcd,abde->abce', rotate_T, centered.float()).permute(0,1,3,2)
    return transformed

# https://behavior.stanford.edu/reference/utils/transform_utils.html#utils.transform_utils.mat2quat
def mat2quat(rmat: torch.Tensor) -> torch.Tensor:
    """
    Converts given rotation matrix to quaternion.
    Args:
        rmat (torch.Tensor): (3, 3) or (..., 3, 3) rotation matrix
    Returns:
        torch.Tensor: (4,) or (..., 4) (x,y,z,w) float quaternion angles
    """
    assert torch.allclose(torch.linalg.det(rmat), torch.tensor(1.0)), "Rotation matrix must not be scaled"

    # Check if input is a single matrix or a batch
    is_single = rmat.dim() == 2
    if is_single:
        rmat = rmat.unsqueeze(0)

    batch_shape = rmat.shape[:-2]
    mat_flat = rmat.reshape(-1, 3, 3)

    m00, m01, m02 = mat_flat[:, 0, 0], mat_flat[:, 0, 1], mat_flat[:, 0, 2]
    m10, m11, m12 = mat_flat[:, 1, 0], mat_flat[:, 1, 1], mat_flat[:, 1, 2]
    m20, m21, m22 = mat_flat[:, 2, 0], mat_flat[:, 2, 1], mat_flat[:, 2, 2]

    trace = m00 + m11 + m22

    trace_positive = trace > 0
    cond1 = (m00 > m11) & (m00 > m22) & ~trace_positive
    cond2 = (m11 > m22) & ~(trace_positive | cond1)
    cond3 = ~(trace_positive | cond1 | cond2)

    # Trace positive condition
    sq = torch.where(trace_positive, torch.sqrt(trace + 1.0) * 2.0, torch.zeros_like(trace))
    qw = torch.where(trace_positive, 0.25 * sq, torch.zeros_like(trace))
    qx = torch.where(trace_positive, (m21 - m12) / sq, torch.zeros_like(trace))
    qy = torch.where(trace_positive, (m02 - m20) / sq, torch.zeros_like(trace))
    qz = torch.where(trace_positive, (m10 - m01) / sq, torch.zeros_like(trace))

    # Condition 1
    sq = torch.where(cond1, torch.sqrt(1.0 + m00 - m11 - m22) * 2.0, sq)
    qw = torch.where(cond1, (m21 - m12) / sq, qw)
    qx = torch.where(cond1, 0.25 * sq, qx)
    qy = torch.where(cond1, (m01 + m10) / sq, qy)
    qz = torch.where(cond1, (m02 + m20) / sq, qz)

    # Condition 2
    sq = torch.where(cond2, torch.sqrt(1.0 + m11 - m00 - m22) * 2.0, sq)
    qw = torch.where(cond2, (m02 - m20) / sq, qw)
    qx = torch.where(cond2, (m01 + m10) / sq, qx)
    qy = torch.where(cond2, 0.25 * sq, qy)
    qz = torch.where(cond2, (m12 + m21) / sq, qz)

    # Condition 3
    sq = torch.where(cond3, torch.sqrt(1.0 + m22 - m00 - m11) * 2.0, sq)
    qw = torch.where(cond3, (m10 - m01) / sq, qw)
    qx = torch.where(cond3, (m02 + m20) / sq, qx)
    qy = torch.where(cond3, (m12 + m21) / sq, qy)
    qz = torch.where(cond3, 0.25 * sq, qz)

    quat = torch.stack([qx, qy, qz, qw], dim=-1)

    # Normalize the quaternion
    quat = quat / torch.norm(quat, dim=-1, keepdim=True)

    # Reshape to match input batch shape
    quat = quat.reshape(batch_shape + (4,))

    # If input was a single matrix, remove the batch dimension
    if is_single:
        quat = quat.squeeze(0)

    return quat

# https://behavior.stanford.edu/reference/utils/transform_utils.html#utils.transform_utils.quat2mat
def quat2mat(quaternion):
    """
    Convert quaternions into rotation matrices.

    Args:
        quaternion (torch.Tensor): A tensor of shape (..., 4) representing batches of quaternions (x, y, z, w).

    Returns:
        torch.Tensor: A tensor of shape (..., 3, 3) representing batches of rotation matrices.
    """
    quaternion = quaternion / torch.norm(quaternion, dim=-1, keepdim=True)

    outer = quaternion.unsqueeze(-1) * quaternion.unsqueeze(-2)

    # Extract the necessary components
    xx = outer[..., 0, 0]
    yy = outer[..., 1, 1]
    zz = outer[..., 2, 2]
    xy = outer[..., 0, 1]
    xz = outer[..., 0, 2]
    yz = outer[..., 1, 2]
    xw = outer[..., 0, 3]
    yw = outer[..., 1, 3]
    zw = outer[..., 2, 3]

    rmat = torch.empty(quaternion.shape[:-1] + (3, 3), dtype=quaternion.dtype, device=quaternion.device)

    rmat[..., 0, 0] = 1 - 2 * (yy + zz)
    rmat[..., 0, 1] = 2 * (xy - zw)
    rmat[..., 0, 2] = 2 * (xz + yw)

    rmat[..., 1, 0] = 2 * (xy + zw)
    rmat[..., 1, 1] = 1 - 2 * (xx + zz)
    rmat[..., 1, 2] = 2 * (yz - xw)

    rmat[..., 2, 0] = 2 * (xz - yw)
    rmat[..., 2, 1] = 2 * (yz + xw)
    rmat[..., 2, 2] = 1 - 2 * (xx + yy)

    return rmat