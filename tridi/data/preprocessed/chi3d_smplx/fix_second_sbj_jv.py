import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import smplx


def build_smplx_layers(model_path: str, batch_size: int, device: str):
    smpl_m = smplx.build_layer(
        model_path=model_path, model_type="smplx", gender="male",
        use_pca=False, num_betas=10, batch_size=batch_size
    ).to(device)
    smpl_f = smplx.build_layer(
        model_path=model_path, model_type="smplx", gender="female",
        use_pca=False, num_betas=10, batch_size=batch_size
    ).to(device)
    return smpl_m, smpl_f


def _as_torch(x, device):
    return torch.tensor(x, dtype=torch.float32, device=device)


@torch.no_grad()
def forward_smplx(g, prefix: str, smpl_layer, start: int, end: int, device: str):
    betas  = _as_torch(g[f"{prefix}_smpl_betas"][start:end], device)
    transl = _as_torch(g[f"{prefix}_smpl_transl"][start:end], device)

    glob = g[f"{prefix}_smpl_global"][start:end]
    body = g[f"{prefix}_smpl_body"][start:end]
    lh   = g[f"{prefix}_smpl_lh"][start:end]
    rh   = g[f"{prefix}_smpl_rh"][start:end]

    # 兼容两种存法：(B,*,9) 或 (B,*,3,3)
    def norm_rot(arr):
        arr = np.asarray(arr)
        if arr.ndim >= 3 and arr.shape[-1] == 9:
            return _as_torch(arr, device)
        if arr.ndim >= 4 and arr.shape[-2:] == (3, 3):
            return _as_torch(arr, device)
        raise ValueError(f"Unexpected rotation shape: {arr.shape}")

    glob_t = norm_rot(glob)
    body_t = norm_rot(body)
    lh_t   = norm_rot(lh)
    rh_t   = norm_rot(rh)

    out = smpl_layer(
        betas=betas,
        transl=transl,
        global_orient=glob_t,
        body_pose=body_t,
        left_hand_pose=lh_t,
        right_hand_pose=rh_t,
        pose2rot=False,
        get_skin=True,
        return_full_pose=False,
    )
    v = out.vertices.detach().cpu().numpy()
    j = out.joints.detach().cpu().numpy()
    return v, j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True, help="path to dataset_*.hdf5 (in-place modify)")
    ap.add_argument("--smpl_folder", required=True, help="SMPL-X model folder (cfg.env.smpl_folder)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--prefix", default="second_sbj", choices=["sbj", "second_sbj"])
    ap.add_argument("--fix_vertices", action="store_true", help="also overwrite/create *_v")
    ap.add_argument("--create_missing_vertices", action="store_true",
                    help="if *_v missing and --fix_vertices, create it")
    args = ap.parse_args()

    h5_path = Path(args.h5)
    assert h5_path.exists(), h5_path

    device = args.device
    chunk = args.chunk

    # 预读一个 seq，确认 T / joints key
    with h5py.File(h5_path, "r") as f:
        seq0 = list(f.keys())[0]
        g0 = f[seq0]
        T0 = int(g0.attrs["T"])
        if f"{args.prefix}_j" not in g0:
            raise KeyError(f"[FATAL] {seq0} missing {args.prefix}_j")
        print(f"[INFO] example seq={seq0}, T={T0}")
        has_v0 = (f"{args.prefix}_v" in g0)
        print(f"[INFO] has {args.prefix}_v: {has_v0}")

    # 只有需要 vertices 时才建 SMPL layer（省点初始化开销）
    smpl_m, smpl_f = build_smplx_layers(args.smpl_folder, batch_size=chunk, device=device)

    with h5py.File(h5_path, "r+") as f:
        seqs = list(f.keys())
        for si, seq in enumerate(seqs):
            g = f[seq]
            T = int(g.attrs["T"])
            gender = g.attrs.get("gender", "male")
            smpl = smpl_m if str(gender).lower().startswith("m") else smpl_f

            # joints 必须存在
            if f"{args.prefix}_j" not in g:
                raise KeyError(f"{seq}: missing {args.prefix}_j")

            # vertices: 只有要修才处理
            need_v = args.fix_vertices
            have_v = (f"{args.prefix}_v" in g)

            # 如果用户要修 v，但 HDF5 没 v
            if need_v and (not have_v) and (not args.create_missing_vertices):
                # 直接跳过 vertices，不炸
                print(f"[WARN] {seq}: missing {args.prefix}_v, skip vertices (add --create_missing_vertices to create)")
                need_v = False

            # 如果要创建缺失的 v，先跑 1 帧推断 V
            if need_v and (not have_v) and args.create_missing_vertices:
                v1, _ = forward_smplx(g, args.prefix, smpl, 0, 1, device=device)
                V = v1.shape[1]
                g.create_dataset(
                    f"{args.prefix}_v",
                    shape=(T, V, 3),
                    dtype=np.float32,
                    chunks=(min(chunk, T), V, 3),
                    compression="gzip",
                    compression_opts=4,
                )
                have_v = True
                print(f"[INFO] {seq}: created {args.prefix}_v with shape {(T, V, 3)}")

            # 主循环：chunk 计算 + 写回
            for start in range(0, T, chunk):
                end = min(T, start + chunk)
                v, j = forward_smplx(g, args.prefix, smpl, start, end, device=device)

                # overwrite joints
                g[f"{args.prefix}_j"][start:end] = j

                # overwrite vertices (optional)
                if need_v and have_v:
                    g[f"{args.prefix}_v"][start:end] = v

            if (si + 1) % 10 == 0 or si == len(seqs) - 1:
                print(f"[INFO] fixed {si+1}/{len(seqs)} sequences")

    print("[DONE] fix complete:", h5_path)


if __name__ == "__main__":
    main()
