from logging import getLogger
from pathlib import Path
from collections import defaultdict
import csv
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import h5py

import faiss

from config.config import ProjectConfig
from tridi.utils.metrics import generation
from tridi.utils.metrics import reconstruction

logger = getLogger(__name__)


class Evaluator:
    def __init__(self, cfg: ProjectConfig):
        self.cfg = cfg

    # -------------------------
    # helpers
    # -------------------------
    def _get_cfg_eval_samples_folder(self) -> Optional[Path]:
        """
        Optional override:
          eval:
            samples_folder: "experiments/006_xxx/artifacts/step_20000_samples"
        You can also pass from CLI:
          eval.samples_folder=experiments/006_xxx/artifacts/step_20000_samples
        """
        try:
            v = getattr(self.cfg.eval, "samples_folder", None)
        except Exception:
            v = None
        if v is None or str(v).strip() == "":
            return None
        return Path(v)

    def _resolve_base_samples_folder(self) -> Path:
        override = self._get_cfg_eval_samples_folder()
        if override is not None:
            return override
        # default: current run folder
        return Path(self.cfg.run.path) / "artifacts" / f"step_{self.cfg.resume.step}_samples"

        # -------------------------
    # NN baseline helpers
    # -------------------------
    def _get_dataset_cfg(self, dataset_name: str):
        return getattr(self.cfg, dataset_name)

    def _get_split_file(self, dataset_cfg, split: str) -> str:
        if split == "train":
            return dataset_cfg.train_split_file
        if split == "val":
            return getattr(dataset_cfg, "val_split_file", None) or dataset_cfg.test_split_file
        if split == "test":
            return dataset_cfg.test_split_file
        raise ValueError(f"Unknown split: {split}")

    def _get_h5_path(self, dataset_cfg, split: str, fps: int) -> Path:
        return Path(dataset_cfg.root) / f"dataset_{split}_{fps}fps.hdf5"

    def _load_split_list(self, split_file: str) -> List[str]:
        with open(split_file, "r") as f:
            return json.load(f)

    def _extract_features_and_joints(
        self,
        h5_path: Path,
        seq_list: List[str],
        prefix: str,
        downsample_factor: int = 1,
        max_timestamps: Optional[int] = None,
    ):
        """
        Return:
          feats: (N, D) float32
          joints: (N, J, 3) float32   (for MPJPE)
          meta: list of (seq_name, t)
        Feature = root-centered joints flattened.
        """
        feats_all = []
        joints_all = []
        meta = []

        with h5py.File(str(h5_path), "r") as f:
            available = set(f.keys())
            valid_seqs = [s for s in seq_list if s in available]
            if len(valid_seqs) == 0:
                raise RuntimeError(f"[NN baseline] No valid sequences in {h5_path} for split list.")

            key_j = f"{prefix}_j"
            for seq in valid_seqs:
                g = f[seq]
                if key_j not in g:
                    raise RuntimeError(f"[NN baseline] Missing key {key_j} in {h5_path}:{seq}")

                T = int(g.attrs.get("T", g[key_j].shape[0]))
                t_stamps = list(range(T))
                if max_timestamps is not None:
                    t_stamps = t_stamps[:max_timestamps]
                if downsample_factor is not None and downsample_factor > 1:
                    t_stamps = t_stamps[::downsample_factor]

                if len(t_stamps) == 0:
                    continue

                J = g[key_j][t_stamps].astype(np.float32)  # (n, J, 3)
                # root-center to reduce translation dominance
                J0 = J[:, 0:1, :]
                Jc = J - J0
                feats = Jc.reshape(Jc.shape[0], -1).astype(np.float32)  # (n, D)

                feats_all.append(feats)
                joints_all.append(J.astype(np.float32))
                meta.extend([(seq, int(t)) for t in t_stamps])

        feats_all = np.concatenate(feats_all, axis=0)
        joints_all = np.concatenate(joints_all, axis=0)
        return feats_all, joints_all, meta

    def _faiss_knn(self, train_feats: np.ndarray, query_feats: np.ndarray, k: int = 1):
        """
        Returns (distances, indices), both shape (N_query, k).
        distances are L2 (squared L2 if IndexFlatL2).
        """
        if faiss is None:
            raise RuntimeError("faiss is not available but NN baseline needs it. Install faiss-cpu or use your KnnWrapper.")

        D = train_feats.shape[1]
        index = faiss.IndexFlatL2(D)
        index.add(train_feats)
        dist, idx = index.search(query_feats, k)
        return dist, idx

    def _mpjpe(self, pred: np.ndarray, gt: np.ndarray):
        # pred/gt: (N, J, 3)
        mean = float(np.mean(np.linalg.norm(pred - gt, axis=-1)))
        return mean

    def _pa_mpjpe(self, pred: np.ndarray, gt: np.ndarray):
        """
        Procrustes aligned MPJPE (scale+R+t), per-sample, then mean.
        pred/gt: (N, J, 3)
        """
        N = pred.shape[0]
        errs = []

        for i in range(N):
            X = pred[i].astype(np.float64)  # (J,3)
            Y = gt[i].astype(np.float64)

            muX = X.mean(axis=0, keepdims=True)
            muY = Y.mean(axis=0, keepdims=True)
            X0 = X - muX
            Y0 = Y - muY

            normX = np.sqrt((X0 ** 2).sum())
            normY = np.sqrt((Y0 ** 2).sum())
            if normX < 1e-12 or normY < 1e-12:
                errs.append(np.mean(np.linalg.norm(X - Y, axis=-1)))
                continue

            X0 /= normX
            Y0 /= normY

            H = X0.T @ Y0
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T

            s = (S.sum() * normY) / normX
            X_aligned = (s * (X - muX) @ R) + muY

            err = np.mean(np.linalg.norm(X_aligned - Y, axis=-1))
            errs.append(err)

        mean = float(np.mean(errs))

        return mean

    def _compute_1nna(self, A: np.ndarray, B: np.ndarray) -> float:
        """
        1-NNA between two sets A and B.
        Returns accuracy in [0,1]. Uses nearest neighbor excluding self for each point.
        """
        X = np.concatenate([A, B], axis=0).astype(np.float32)
        y = np.concatenate([np.zeros(len(A), dtype=np.int32), np.ones(len(B), dtype=np.int32)], axis=0)

        # Use faiss K=2 then take the first non-self neighbor
        dist, idx = self._faiss_knn(X, X, k=2)
        nn = idx[:, 1]
        acc = float(np.mean(y[nn] == y))
        return acc

    # -------------------------
    # main entry
    # -------------------------
    def evaluate(self):
        base_samples_folder = self._resolve_base_samples_folder()
        # print("Base samples folder:", base_samples_folder)

        rows: List[Dict[str, Any]] = []
        exp_name = str(self.cfg.run.name)
        step_str = str(self.cfg.resume.step)

        logger.info(f"Experiment: {self.cfg.run.name} step: {self.cfg.resume.step}")

        # =========================
        # Generation metrics
        # =========================
        if self.cfg.eval.use_gen_metrics:
            logger.info("Evaluating generation")
            for dataset in self.cfg.run.datasets:
                logger.info(f"\ton {dataset}")
                samples_folder = base_samples_folder / dataset

                for sample_target in self.cfg.eval.sampling_target:
                    logger.info(f"\t  sampling target: {sample_target}")
                    metrics = {"1-NNA": [], "COV": [], "MMD": [], "SD": []}
                    samples_files = list(samples_folder.glob(f"{sample_target}/samples_rep_*.hdf5"))

                    for samples_file in samples_files:
                        metrics["1-NNA"].append(generation.nearest_neighbor_accuracy(
                            self.cfg, samples_file, dataset, "test", sample_target,
                        ))
                        metrics["COV"].append(generation.coverage(
                            self.cfg, samples_file, dataset, "test", sample_target,
                        ))
                        metrics["MMD"].append(generation.minimum_matching_distance(
                            self.cfg, samples_file, dataset, "test", sample_target,
                        ))
                        metrics["SD"].append(generation.sample_distance(
                            self.cfg, samples_file, dataset, "train", sample_target,
                        ))

                    for k, v in metrics.items():
                        if k in ["1-NNA", "COV"]:
                            mean = float(np.mean(v))*100
                            std = float(np.std(v))*100
                            logger.info(f"\t\t{k:<6s} - {sample_target}: {mean:.2f} ± {std:.2f}")
                            rows.append({
                                "exp": exp_name,
                                "step": step_str,
                                "phase": "gen",
                                "dataset": dataset,
                                "target": sample_target,
                                "metric": k,
                                "mean_std": f"{mean:.2f} ± {std:.2f}"
                            })
                        elif k in ["MMD", "SD"]:
                            mean = float(np.mean(v))
                            std = float(np.std(v))
                            logger.info(f"\t\t{k:<6s} - {sample_target}: {mean:.2f} ± {std:.2f}")
                            rows.append({
                                "exp": exp_name,
                                "step": step_str,
                                "phase": "gen",
                                "dataset": dataset,
                                "target": sample_target,
                                "metric": k,
                                "mean_std": f"{mean:.2f} ± {std:.2f}"
                            })

        # =========================
        # Reconstruction metrics
        # =========================
        if self.cfg.eval.use_rec_metrics:
            logger.info("Evaluating reconstruction")
            for dataset in self.cfg.run.datasets:
                logger.info(f"\ton {dataset}")
                samples_folder = base_samples_folder / dataset

                for sample_target in self.cfg.eval.sampling_target:
                    if sample_target == "sbj" or sample_target == "second_sbj":
                        samples_files = list(samples_folder.glob(f"{sample_target}/samples_rep_*.hdf5"))
                        metrics = {f"{sample_target}_MPJPE": [], f"{sample_target}_MPJPE_PA": []}
                        for samples_file in samples_files:
                            mpjpe, mpjpe_pa = reconstruction.get_sbj_metrics(self.cfg, samples_file, dataset, sample_target)
                            metrics[f"{sample_target}_MPJPE"].append(mpjpe)
                            metrics[f"{sample_target}_MPJPE_PA"].append(mpjpe_pa)

                
                    for k, v in metrics.items():
                        # print(f"\t\t{k}: {v}")
                        val = np.mean(np.min(np.stack(v, 1), axis=1))*100
                        logger.info(f"\t\t{k:<20s}: {val:.1f}")

                        rows.append({
                            "exp": exp_name,
                            "step": step_str,
                            "phase": "rec",
                            "dataset": dataset,
                            "target": sample_target,
                            "metric": k,
                            "value": f"{val:.1f}"
                        })
        # =========================
        # NN Baseline (copy-from-train)
        # =========================
        if getattr(self.cfg.eval, "nn_baseline", False):
            logger.info("Evaluating NN baseline (copy-from-train)")
            k = int(getattr(self.cfg.eval, "nn_baseline_k", 1))
            ref_split = str(getattr(self.cfg.eval, "nn_baseline_ref_split", "train"))

            for dataset in self.cfg.run.datasets:
                dataset_cfg = self._get_dataset_cfg(dataset)

                # query split：default = test
                query_split = "test"
                fps_ref = int(dataset_cfg.fps_train) if ref_split == "train" else int(dataset_cfg.fps_eval)
                fps_q = int(dataset_cfg.fps_eval)

                ref_split_file = self._get_split_file(dataset_cfg, ref_split)
                q_split_file = self._get_split_file(dataset_cfg, query_split)

                ref_seqs = self._load_split_list(ref_split_file)
                q_seqs = self._load_split_list(q_split_file)

                ref_h5 = self._get_h5_path(dataset_cfg, ref_split, fps_ref)
                q_h5 = self._get_h5_path(dataset_cfg, query_split, fps_q)

                for sample_target in self.cfg.eval.sampling_target:
                    # only to sbj / second_sbj 
                    if sample_target not in ["sbj", "second_sbj"]:
                        continue

                    prefix = "sbj" if sample_target == "sbj" else "second_sbj"

                    logger.info(f"\tNN baseline target: {sample_target} (ref={ref_split}, query={query_split})")

                    # 1) build features for ref/train and query/test
                    ref_feats, ref_joints, _ = self._extract_features_and_joints(
                        ref_h5, ref_seqs, prefix,
                        downsample_factor=int(getattr(dataset_cfg, "downsample_factor", 1)),
                        max_timestamps=getattr(dataset_cfg, "max_timestamps", None),
                    )
                    q_feats, q_joints, _ = self._extract_features_and_joints(
                        q_h5, q_seqs, prefix,
                        downsample_factor=int(getattr(dataset_cfg, "downsample_factor", 1)),
                        max_timestamps=getattr(dataset_cfg, "max_timestamps", None),
                    )

                    # 2) NN search: query -> ref
                    dist, idx = self._faiss_knn(ref_feats, q_feats, k=max(1, k))
                    nn_idx = idx[:, 0]
                    nn_dist = dist[:, 0]

                    # ===== gen-like metrics for baseline =====
                    # MMD: mean NN distance (note: dist is squared L2 for IndexFlatL2)
                    
                    mmd_mean = float(np.mean(np.sqrt(nn_dist)))
                    mmd_std = float(np.std(np.sqrt(nn_dist)))


                    cov = float(len(np.unique(nn_idx)) / len(ref_feats))  # coverage over ref set

                    nna = self._compute_1nna(q_feats, ref_feats[nn_idx])  # test vs retrieved

                    # SD for baseline: avoid self-match=0 by querying baseline points into ref and taking k=2
                    if len(ref_feats) > 1:
                        dist2, idx2 = self._faiss_knn(ref_feats, ref_feats[nn_idx], k=2)
                        sd_mean = float(np.mean(np.sqrt(dist2[:, 1])))
                        sd_std = float(np.std(np.sqrt(dist2[:, 1])))

                    logger.info(f"\t\t1-NNA - {sample_target}: {nna*100:.2f}")
                    logger.info(f"\t\tCOV   - {sample_target}: {cov*100:.2f}")
                    logger.info(f"\t\tMMD   - {sample_target}: {mmd_mean:.2f} ± {mmd_std:.2f}")
                    logger.info(f"\t\tSD    - {sample_target}: {sd_mean:.2f} ± {sd_std:.2f}")

                    rows.append({
                        "exp": exp_name, "step": step_str, "phase": "nn",
                        "dataset": dataset, "target": sample_target,
                        "metric": "1-NNA", "value": f"{nna*100:.2f}"
                    })
                    rows.append({
                        "exp": exp_name, "step": step_str, "phase": "nn",
                        "dataset": dataset, "target": sample_target,
                        "metric": "COV", "value": f"{cov*100:.2f}"
                    })
                    rows.append({
                        "exp": exp_name, "step": step_str, "phase": "nn",
                        "dataset": dataset, "target": sample_target,
                        "metric": "MMD", "mean_std": f"{mmd_mean:.2f} ± {mmd_std:.2f}"
                    })
                    rows.append({
                        "exp": exp_name, "step": step_str, "phase": "nn",
                        "dataset": dataset, "target": sample_target,
                        "metric": "SD", "mean_std": f"{sd_mean:.2f} ± {sd_std:.2f}"
                    })

                    # ===== reconstruction baseline =====
                    pred_joints = ref_joints[nn_idx]  # (Nq, J, 3)
                    mpjpe = self._mpjpe(pred_joints, q_joints) * 100.0 
                    mpjpe_pa = self._pa_mpjpe(pred_joints, q_joints) * 100.0

                    logger.info(f"\t\tMPJPE     - {sample_target}: {mpjpe:.2f}")
                    logger.info(f"\t\tMPJPE_PA  - {sample_target}: {mpjpe_pa:.2f}")

                    rows.append({
                        "exp": exp_name, "step": step_str, "phase": "nn",
                        "dataset": dataset, "target": sample_target,
                        "metric": "MPJPE", "value": f"{mpjpe:.2f}"
                    })
                    rows.append({
                        "exp": exp_name, "step": step_str, "phase": "nn",
                        "dataset": dataset, "target": sample_target,
                        "metric": "MPJPE_PA", "value": f"{mpjpe_pa:.2f}"
                    })

        # =========================
        # Dump tables
        # =========================
        out_dir = base_samples_folder / "_eval"
        self._save_tables(rows, out_dir)

    def _save_tables(self, rows: List[Dict[str, Any]], out_dir: Path):
        # only save target,metric,value,mean_std, split sbj and second_sbj in two rows
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "evaluation_results.csv"
        with open(csv_path, "w", newline="") as csvfile:
            fieldnames = ["method", "target", "metric", "value", "mean_std"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    "method": "HHgen" if row.get("phase", "") != "nn" else "NN",
                    "target": row.get("target", ""),
                    "metric": row.get("metric", ""),
                    "value": row.get("value", ""),
                    "mean_std": row.get("mean_std", ""),
                })
        logger.info(f"Saved evaluation results to {csv_path}")

