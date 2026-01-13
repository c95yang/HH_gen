import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import smplx


def to_torch(x, device):
    return torch.tensor(x, dtype=torch.float32, device=device)


@torch.no_grad()
def smplx_joints_no_transl(g, prefix: str, smpl_layer, start: int, end: int, device: str):
    betas = to_torch(g[f"{prefix}_smpl_betas"][start:end], device)

    # transl 置 0：我们要用 root joint 对齐来反推正确 transl
    transl = torch.zeros((end - start, 3), dtype=torch.float32, device=device)

    glob = to_torch(g[f"{prefix}_smpl_global"][start:end], device)
    body = to_torch(g[f"{prefix}_smpl_body"][start:end], device)
    lh   = to_torch(g[f"{prefix}_smpl_lh"][start:end], device)
    rh   = to_torch(g[f"{prefix}_smpl_rh"][start:end], device)

    out = smpl_layer(
        betas=betas,
        transl=transl,
        global_orient=glob,
        body_pose=body,
        left_hand_pose=lh,
        right_hand_pose=rh,
        pose2rot=False,
        get_skin=True,
        return_full_pose=True,
    )
    return out.joints.detach().cpu().numpy()  # (B, J, 3)


@torch.no_grad()
def smplx_joints_with_transl(g, prefix: str, smpl_layer, transl_np: np.ndarray, start: int, end: int, device: str):
    betas = to_torch(g[f"{prefix}_smpl_betas"][start:end], device)
    transl = to_torch(transl_np, device)

    glob = to_torch(g[f"{prefix}_smpl_global"][start:end], device)
    body = to_torch(g[f"{prefix}_smpl_body"][start:end], device)
    lh   = to_torch(g[f"{prefix}_smpl_lh"][start:end], device)
    rh   = to_torch(g[f"{prefix}_smpl_rh"][start:end], device)

    out = smpl_layer(
        betas=betas,
        transl=transl,
        global_orient=glob,
        body_pose=body,
        left_hand_pose=lh,
        right_hand_pose=rh,
        pose2rot=False,
        get_skin=True,
        return_full_pose=True,
    )
    return out.joints.detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--smpl_folder", required=True)
    ap.add_argument("--prefix", default="second_sbj", choices=["sbj", "second_sbj"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk", type=int, default=256)
    args = ap.parse_args()

    h5_path = Path(args.h5)
    device = args.device
    chunk = args.chunk

    # 只用 male layer（和你 MeshModel 的 get_smpl_th_single 一致）
    smpl_m = smplx.build_layer(
        model_path=str(args.smpl_folder),
        model_type="smplx",
        gender="male",
        use_pca=False,
        num_betas=10,
        batch_size=chunk
    ).to(device)

    with h5py.File(h5_path, "r+") as f:
        seqs = list(f.keys())
        for si, seq in enumerate(seqs):
            g = f[seq]
            T = int(g.attrs["T"])

            if f"{args.prefix}_j" not in g:
                raise KeyError(f"{seq} missing {args.prefix}_j")
            if f"{args.prefix}_smpl_transl" not in g:
                raise KeyError(f"{seq} missing {args.prefix}_smpl_transl")

            for start in range(0, T, chunk):
                end = min(T, start + chunk)

                # 读当前“锚点 joints”
                j_ref = g[f"{args.prefix}_j"][start:end]  # (B,J,3)

                # 用 transl=0 重建 joints
                j0 = smplx_joints_no_transl(g, args.prefix, smpl_m, start, end, device)

                # root 对齐（默认 joint[0] 是 root/pelvis，和你之前用法一致）
                delta = j_ref[:, 0, :] - j0[:, 0, :]  # (B,3)

                # 写回修正后的 transl（覆盖原来的）
                g[f"{args.prefix}_smpl_transl"][start:end] = delta.astype(np.float32)

                # 用修正 transl 重建 joints，再写回 joints（保证 params/joints 一致）
                j_new = smplx_joints_with_transl(g, args.prefix, smpl_m, delta, start, end, device)
                g[f"{args.prefix}_j"][start:end] = j_new.astype(np.float32)

            if (si + 1) % 10 == 0 or si == len(seqs) - 1:
                print(f"[INFO] fixed {si+1}/{len(seqs)} sequences")

        f.flush()

    print("[DONE] fixed transl + joints for", args.prefix, "in", h5_path)


if __name__ == "__main__":
    main()
