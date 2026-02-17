#!/usr/bin/env python3
"""
how to run
1) choose case（generate cases.json）:
python tridi/tools/vis/select_cases_from_hdf5.py \
  --samples_dir /media/uv/Data/workspace/HH_gen/experiments/021_chi3d/artifacts/step_20000_samples/chi3d/sbj \
  --dataset_root /media/uv/Data/workspace/HH_gen/tridi/data/preprocessed/chi3d_smplx \
  --mode 10

2) then run export_cases_to_ply.py export ply for visualization:
"""

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import argparse
import glob
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np

logger = logging.getLogger(__name__)


def _fallback_compute_similarity_transform(S1: np.ndarray, S2: np.ndarray) -> np.ndarray:
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert S2.shape[1] == S1.shape[1]

    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    var1 = np.sum(X1 ** 2)
    K = X1.dot(X2.T)

    U, _, Vh = np.linalg.svd(K)
    V = Vh.T
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    R = V.dot(Z.dot(U.T))

    scale = np.trace(R.dot(K)) / var1
    t = mu2 - scale * (R.dot(mu1))
    S1_hat = scale * R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T
    return S1_hat


try:
    from tridi.utils.metrics.reconstruction import get_mpjpe_pa as _repo_get_mpjpe_pa

    def _get_mpjpe_pa(pred_joints: np.ndarray, gt_joints: np.ndarray) -> float:
        val = _repo_get_mpjpe_pa(pred_joints, gt_joints)
        return float(np.asarray(val))

    logger.info("Using tridi.utils.metrics.reconstruction.get_mpjpe_pa")
except Exception as e:  # noqa: BLE001
    logger.warning("Falling back to local MPJPE_PA implementation, reason: %s", e)

    def _get_mpjpe_pa(pred_joints: np.ndarray, gt_joints: np.ndarray) -> float:
        pred = np.asarray(pred_joints, dtype=np.float32)
        gt = np.asarray(gt_joints, dtype=np.float32)

        pred = pred - pred[[0]]
        gt = gt - gt[[0]]

        pred_aligned = _fallback_compute_similarity_transform(pred, gt)
        mpjpe_pa = np.sqrt(np.sum((gt - pred_aligned) ** 2, axis=-1)).mean(-1)
        return float(mpjpe_pa)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select random/best/worst cases from samples_rep_*.hdf5")
    parser.add_argument("--samples_dir", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--mode", type=str, default="10", choices=["10", "01", "11"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_random", type=int, default=10)
    parser.add_argument("--n_best", type=int, default=5)
    parser.add_argument("--n_worst", type=int, default=5)
    parser.add_argument("--max_eval", type=int, default=-1)
    return parser.parse_args()


def _find_rep_files(samples_dir: Path) -> List[Path]:
    rep_files = sorted(Path(p) for p in glob.glob(str(samples_dir / "samples_rep_*.hdf5")))
    assert rep_files, (
        f"No rep files found at {samples_dir}/samples_rep_*.hdf5. "
        "pls make sure if sampling output correct."
    )
    return rep_files


def _rep_id_from_path(path: Path) -> int:
    stem = path.stem  # samples_rep_00
    tail = stem.split("_")[-1]
    return int(tail)


def _required_joint_keys(mode: str) -> List[str]:
    if mode == "10":
        return ["sbj_j"]
    if mode == "01":
        return ["second_sbj_j"]
    if mode == "11":
        return ["sbj_j", "second_sbj_j"]
    raise ValueError(f"Unsupported mode: {mode}")


def _assert_group_has_keys(group: h5py.Group, keys: List[str], context: str) -> None:
    missing = [k for k in keys if k not in group]
    assert not missing, (
        f"HDF5 structure doesnt match: {context} missing key={missing}. "
        f"available keys={sorted(list(group.keys()))}"
    )


def _get_group_T(group: h5py.Group, joint_keys: List[str]) -> int:
    _assert_group_has_keys(group, joint_keys, context="sequence group")
    t_from_data = min(int(group[k].shape[0]) for k in joint_keys)
    t_attr = int(group.attrs.get("T", t_from_data))
    return min(t_from_data, t_attr)


def _score_case_for_rep(
    gt_group: h5py.Group,
    pred_group: h5py.Group,
    t: int,
    mode: str,
) -> float:
    if mode == "10":
        return _get_mpjpe_pa(pred_group["sbj_j"][t], gt_group["sbj_j"][t])
    if mode == "01":
        return _get_mpjpe_pa(pred_group["second_sbj_j"][t], gt_group["second_sbj_j"][t])
    if mode == "11":
        # use 0.5*(sbj + second_sbj) as overall score for ranking, to find cases where both are good or both are bad.
        sbj_score = _get_mpjpe_pa(pred_group["sbj_j"][t], gt_group["sbj_j"][t])
        second_score = _get_mpjpe_pa(pred_group["second_sbj_j"][t], gt_group["second_sbj_j"][t])
        return 0.5 * (sbj_score + second_score)
    raise ValueError(f"Unsupported mode: {mode}")


def _compute_case_scores_all_reps(
    seq: str,
    t: int,
    mode: str,
    gt_h5: h5py.File,
    rep_h5_map: Dict[int, h5py.File],
) -> Dict[int, float]:
    if seq not in gt_h5:
        return {}
    gt_group = gt_h5[seq]

    req_keys = _required_joint_keys(mode)
    _assert_group_has_keys(gt_group, req_keys, f"GT[{seq}]")

    scores = {}
    for rep_id, rep_h5 in rep_h5_map.items():
        if seq not in rep_h5:
            continue
        pred_group = rep_h5[seq]
        try:
            _assert_group_has_keys(pred_group, req_keys, f"REP{rep_id}[{seq}]")
        except AssertionError as e:
            logger.warning("Skip rep=%d seq=%s because key mismatch: %s", rep_id, seq, e)
            continue

        t_gt = _get_group_T(gt_group, req_keys)
        t_pred = _get_group_T(pred_group, req_keys)
        if t >= min(t_gt, t_pred):
            continue

        scores[rep_id] = _score_case_for_rep(gt_group, pred_group, t, mode)
    return scores


def _load_split_sequences(dataset_root: Path, split: str) -> List[str]:
    split_path = dataset_root / f"chi3d_{split}.json"
    assert split_path.exists(), f"split json does not exist: {split_path}"
    with split_path.open("r", encoding="utf-8") as f:
        seqs = json.load(f)
    assert isinstance(seqs, list) and seqs, f"split json is empty or malformed: {split_path}"
    return seqs


def _candidate_cases_from_split(
    gt_h5: h5py.File,
    split_sequences: List[str],
    mode: str,
) -> List[Tuple[str, int]]:
    req_keys = _required_joint_keys(mode)
    cases: List[Tuple[str, int]] = []
    missing_seq = 0

    for seq in split_sequences:
        if seq not in gt_h5:
            missing_seq += 1
            continue
        gt_group = gt_h5[seq]
        _assert_group_has_keys(gt_group, req_keys, f"GT[{seq}]")
        T = _get_group_T(gt_group, req_keys)
        for t in range(T):
            cases.append((seq, t))

    if missing_seq > 0:
        logger.warning("%d split sequences not found in GT hdf5", missing_seq)
    assert cases, "No valid cases found, please check if GT hdf5 and split json match."
    return cases


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args()

    samples_dir = Path(args.samples_dir)
    dataset_root = Path(args.dataset_root)
    out_dir = samples_dir / "_viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "cases.json"

    logger.info("Scanning rep files from: %s", samples_dir)
    rep_files = _find_rep_files(samples_dir)
    rep_ids = [_rep_id_from_path(p) for p in rep_files]
    logger.info("Found %d rep files: %s", len(rep_files), rep_ids)

    gt_hdf5_path = dataset_root / f"dataset_{args.split}_25fps.hdf5"
    assert gt_hdf5_path.exists(), f"GT hdf5 does not exist: {gt_hdf5_path}"

    split_sequences = _load_split_sequences(dataset_root, args.split)
    logger.info("Loaded split=%s with %d sequences", args.split, len(split_sequences))

    rng = np.random.default_rng(args.seed)

    with h5py.File(gt_hdf5_path, "r") as gt_h5:
        rep_h5_map: Dict[int, h5py.File] = {}
        try:
            for path in rep_files:
                rep_h5_map[_rep_id_from_path(path)] = h5py.File(path, "r")

            all_candidates = _candidate_cases_from_split(gt_h5, split_sequences, args.mode)
            logger.info("Total candidate cases from split: %d", len(all_candidates))

            rank_candidates = list(all_candidates)
            if args.max_eval is not None and args.max_eval > 0 and len(rank_candidates) > args.max_eval:
                idx = rng.choice(len(rank_candidates), size=args.max_eval, replace=False)
                rank_candidates = [rank_candidates[i] for i in idx]
                logger.info("Using max_eval=%d cases for ranking", len(rank_candidates))

            case_bestofk_scores: Dict[Tuple[str, int], float] = {}
            for i, (seq, t) in enumerate(rank_candidates):
                if i % 500 == 0 and i > 0:
                    logger.info("Scored %d/%d ranking cases", i, len(rank_candidates))
                rep_scores = _compute_case_scores_all_reps(seq, t, args.mode, gt_h5, rep_h5_map)
                if rep_scores:
                    case_bestofk_scores[(seq, t)] = float(min(rep_scores.values()))

            assert case_bestofk_scores, "No case in any rep can be scored with MPJPE_PA, please check input files."
            sorted_items = sorted(case_bestofk_scores.items(), key=lambda kv: kv[1])

            n_best = min(args.n_best, len(sorted_items))
            best_cases = [k for (k, _) in sorted_items[:n_best]]
            best_set = set(best_cases)

            remaining_after_best = [k for (k, _) in sorted_items if k not in best_set]
            n_worst = min(args.n_worst, len(remaining_after_best))
            worst_cases = remaining_after_best[-n_worst:] if n_worst > 0 else []

            taken = set(best_cases) | set(worst_cases)
            pool = [c for c in all_candidates if c not in taken]
            rng.shuffle(pool)

            random_cases: List[Tuple[str, int]] = []
            for seq, t in pool:
                if len(random_cases) >= args.n_random:
                    break
                # make sure random case in at least one rep is scoreable, to avoid failure in later export
                rep_scores = _compute_case_scores_all_reps(seq, t, args.mode, gt_h5, rep_h5_map)
                if rep_scores:
                    random_cases.append((seq, t))

            if len(random_cases) < args.n_random:
                logger.warning(
                    "random cases only got %d (requested %d).",
                    len(random_cases),
                    args.n_random,
                )

            payload = {
                "meta": {
                    "dataset": "chi3d",
                    "split": args.split,
                    "mode": args.mode,
                    "seed": args.seed,
                    "n_random": args.n_random,
                    "n_best": args.n_best,
                    "n_worst": args.n_worst,
                    "max_eval": args.max_eval,
                    "score": "best_of_K(min over reps) using MPJPE_PA; mode=11 uses 0.5*(sbj+second_sbj)",
                    "reps": sorted(rep_ids),
                },
                "random": [{"seq": s, "t": int(t)} for s, t in random_cases],
                "best": [{"seq": s, "t": int(t)} for s, t in best_cases],
                "worst": [{"seq": s, "t": int(t)} for s, t in worst_cases],
            }

            with out_json.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            logger.info("Saved cases json to: %s", out_json)
            logger.info(
                "Selected random=%d best=%d worst=%d",
                len(random_cases),
                len(best_cases),
                len(worst_cases),
            )
        finally:
            for h5 in rep_h5_map.values():
                h5.close()


if __name__ == "__main__":
    main()
