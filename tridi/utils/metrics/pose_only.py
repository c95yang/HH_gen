# tridi/utils/metrics/pose_only.py
import numpy as np

def align_to_root(joints: np.ndarray, root_idx: int = 0):
    # joints: (..., J, 3)
    return joints - joints[..., root_idx:root_idx + 1, :]

def _rot_y(yaw: np.ndarray):
    # yaw: (...,)
    c = np.cos(-yaw)
    s = np.sin(-yaw)
    # R around +Y (y-up), apply inverse yaw
    R = np.zeros(yaw.shape + (3, 3), dtype=np.float32)
    R[..., 0, 0] = c
    R[..., 0, 2] = s
    R[..., 1, 1] = 1.0
    R[..., 2, 0] = -s
    R[..., 2, 2] = c
    return R

def canonicalize_pose_only(
    joints: np.ndarray,
    root_idx: int = 0,
    lhip: int = 1, rhip: int = 2,
    lsho: int = 16, rsho: int = 17,
):
    """
    return：root-center + yaw-normalized joints（ignore translation + global yaw）
    y-up，yaw on x-z 
    if joint index different，change lhip/rhip/lsho/rsho。
    """
    J = joints.astype(np.float32)
    J = align_to_root(J, root_idx=root_idx)

    
    across = (J[..., rhip, :] - J[..., lhip, :])
    if (lsho is not None) and (rsho is not None) and (J.shape[-2] > max(lsho, rsho)):
        across2 = (J[..., rsho, :] - J[..., lsho, :])
        across = 0.5 * (across + across2)

    
    across_h = across.copy()
    across_h[..., 1] = 0.0
    across_h = across_h / (np.linalg.norm(across_h, axis=-1, keepdims=True) + 1e-8)

    # forward = up x across
    up = np.zeros_like(across_h)
    up[..., 1] = 1.0
    forward = np.cross(up, across_h)
    forward[..., 1] = 0.0
    forward = forward / (np.linalg.norm(forward, axis=-1, keepdims=True) + 1e-8)

    # yaw = atan2(forward_x, forward_z)
    yaw = np.arctan2(forward[..., 0], forward[..., 2]).astype(np.float32)

    R = _rot_y(yaw)  # (...,3,3)

    Jn = J @ R
    return Jn
