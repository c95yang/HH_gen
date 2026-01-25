from logging import getLogger
from pathlib import Path
from collections import defaultdict
import csv
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import h5py

try:
    import faiss
except Exception:
    faiss = None

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

    def _save_tables(self, rows: List[Dict[str, Any]], out_dir: Path) -> None:
        """
        Writes:
          - eval_metrics_long.csv  (merge + upsert if already exists)
          - eval_metrics_wide.csv  (pivot from merged long)
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        long_path = out_dir / "eval_metrics_long.csv"

        # (A) Load existing long csv if present
        existing: List[Dict[str, str]] = []
        if long_path.exists():
            try:
                with open(long_path, "r", newline="") as f:
                    r = csv.DictReader(f)
                    for row in r:
                        existing.append(dict(row))
            except Exception as e:
                logger.warning(f"[Eval] Failed reading existing CSV ({long_path}): {e}")

        # (B) Merge + upsert
        # Key: (exp, step, phase, dataset, target, metric)
        def make_key(d: Dict[str, Any]) -> Tuple[str, str, str, str, str, str]:
            return (
                str(d.get("exp", "")),
                str(d.get("step", "")),
                str(d.get("phase", "")),
                str(d.get("dataset", "")),
                str(d.get("target", "")),
                str(d.get("metric", "")),
            )

        merged_map: Dict[Tuple[str, str, str, str, str, str], Dict[str, Any]] = {}
        for r0 in existing:
            merged_map[make_key(r0)] = r0
        for r1 in rows:
            merged_map[make_key(r1)] = r1  # newest wins

        merged_rows = list(merged_map.values())
        merged_rows.sort(key=lambda d: make_key(d))

        # (C) Fieldnames = union of keys (stable order)
        preferred = [
            "exp", "step", "phase", "dataset", "target", "metric",
            "value", "mean", "std", "mean_std", "raw_values"
        ]
        all_keys = set()
        for r in merged_rows:
            all_keys.update(r.keys())

        fieldnames = [k for k in preferred if k in all_keys] + sorted([k for k in all_keys if k not in preferred])

        # (D) Write merged long table (overwrite file, but content is merged)
        with open(long_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in merged_rows:
                out = {k: r.get(k, "") for k in fieldnames}
                w.writerow(out)

        logger.info(f"[Eval] Updated long table (merge+upsert): {long_path}")

        # (E) Wide table (pivot from merged long)
        grouped = defaultdict(dict)  # (exp,step,phase,dataset,target) -> {metric: value}
        metric_cols = sorted({r.get("metric", "") for r in merged_rows if r.get("metric", "")})

        for r in merged_rows:
            key = (
                r.get("exp", ""),
                r.get("step", ""),
                r.get("phase", ""),
                r.get("dataset", ""),
                r.get("target", ""),
            )
            m = r.get("metric", "")
            if not m:
                continue
            val = r.get("mean_std", "")
            if val is None or str(val).strip() == "":
                val = r.get("value", "")

            grouped[key][m] = val

        wide_rows = []
        for (exp, step, phase, dataset, target), mdict in grouped.items():
            row = {"exp": exp, "step": step, "phase": phase, "dataset": dataset, "target": target}
            for m in metric_cols:
                row[m] = mdict.get(m, "")
            wide_rows.append(row)

        wide_path = out_dir / "eval_metrics_wide.csv"
        with open(wide_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["exp", "step", "phase", "dataset", "target"] + metric_cols)
            w.writeheader()
            w.writerows(wide_rows)

        logger.info(f"[Eval] Updated wide table: {wide_path}")

        # (F) Print readable table
        if len(wide_rows) > 0:
            headers = ["exp", "step", "phase", "dataset", "target"] + metric_cols
            col_w = {h: len(h) for h in headers}
            for r in wide_rows:
                for h in headers:
                    col_w[h] = max(col_w[h], len(str(r.get(h, ""))))

            def fmt_row(rr):
                return " | ".join(str(rr.get(h, "")).ljust(col_w[h]) for h in headers)

            sep = "-+-".join("-" * col_w[h] for h in headers)
            logger.info("[Eval] Summary table (wide):")
            print(fmt_row({h: h for h in headers}))
            print(sep)
            for r in wide_rows:
                print(fmt_row(r))
        # (G) Paper-like table: rows=method, cols=target_metric
        metric_order = ["1-NNA", "COV", "MMD", "SD", "MPJPE", "MPJPE_PA"]
        targets = sorted({r.get("target", "") for r in merged_rows if r.get("target", "")})

        def pick_val(r):
            v = r.get("mean_std", "")
            if v is None or str(v).strip() == "":
                v = r.get("value", "")
            return "" if v is None else str(v)

        def parse_method_and_metric(r):
            target = r.get("target", "")
            metric = r.get("metric", "")
            if not target or not metric:
                return None, None, None

            method = "TriDi"
            base_metric = metric

            if metric.startswith("NNBASE_"):
                method = "NNBASE"
                base_metric = metric[len("NNBASE_"):]
            else:
                # rec metrics are like "sbj_MPJPE", "second_sbj_MPJPE_PA"
                prefix = f"{target}_"
                if metric.startswith(prefix):
                    base_metric = metric[len(prefix):]

            return method, target, base_metric

        paper_map = defaultdict(dict)  # (exp,step,dataset,method) -> {target_metric: val}

        for r in merged_rows:
            method, target, base_metric = parse_method_and_metric(r)
            if method is None:
                continue
            if base_metric not in metric_order:
                continue
            key = (r.get("exp", ""), r.get("step", ""), r.get("dataset", ""), method)
            paper_map[key][f"{target}_{base_metric}"] = pick_val(r)

        paper_rows = []
        for (exp, step, dataset, method), mdict in sorted(paper_map.items()):
            row = {"exp": exp, "step": step, "dataset": dataset, "method": method}
            for t in targets:
                for m in metric_order:
                    row[f"{t}_{m}"] = mdict.get(f"{t}_{m}", "")
            paper_rows.append(row)

        paper_cols = ["exp", "step", "dataset", "method"] + [f"{t}_{m}" for t in targets for m in metric_order]
        paper_path = out_dir / "eval_metrics_paper.csv"
        with open(paper_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=paper_cols)
            w.writeheader()
            w.writerows(paper_rows)

        logger.info(f"[Eval] Wrote paper-like table: {paper_path}")

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

    def _mpjpe(self, pred: np.ndarray, gt: np.ndarray) -> float:
        # pred/gt: (N, J, 3)
        return float(np.mean(np.linalg.norm(pred - gt, axis=-1)))

    def _pa_mpjpe(self, pred: np.ndarray, gt: np.ndarray) -> float:
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

        return float(np.mean(errs))

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
                        print(f"\t\t{k}: {v}")
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
                    
                    mmd = float(np.mean(np.sqrt(nn_dist)))

                    cov = float(len(np.unique(nn_idx)) / len(ref_feats))  # coverage over ref set

                    nna = self._compute_1nna(q_feats, ref_feats[nn_idx])  # test vs retrieved

                    # SD for baseline: avoid self-match=0 by querying baseline points into ref and taking k=2
                    sd = 0.0
                    if len(ref_feats) > 1:
                        dist2, idx2 = self._faiss_knn(ref_feats, ref_feats[nn_idx], k=2)
                        sd = float(np.mean(np.sqrt(dist2[:, 1])))

                    logger.info(f"\t\tNNBASE 1-NNA - {sample_target}: {nna*100:.2f}")
                    logger.info(f"\t\tNNBASE COV   - {sample_target}: {cov*100:.2f}")
                    logger.info(f"\t\tNNBASE MMD   - {sample_target}: {mmd:.4f}")
                    logger.info(f"\t\tNNBASE SD    - {sample_target}: {sd:.4f}")

                    rows.append({
                        "exp": exp_name, "step": step_str, "phase": "nn",
                        "dataset": dataset, "target": sample_target,
                        "metric": "NNBASE_1-NNA", "mean_std": f"{nna*100:.2f} ± 0.00"
                    })
                    rows.append({
                        "exp": exp_name, "step": step_str, "phase": "nn",
                        "dataset": dataset, "target": sample_target,
                        "metric": "NNBASE_COV", "mean_std": f"{cov*100:.2f} ± 0.00"
                    })
                    rows.append({
                        "exp": exp_name, "step": step_str, "phase": "nn",
                        "dataset": dataset, "target": sample_target,
                        "metric": "NNBASE_MMD", "mean_std": f"{mmd:.4f} ± 0.0000"
                    })
                    rows.append({
                        "exp": exp_name, "step": step_str, "phase": "nn",
                        "dataset": dataset, "target": sample_target,
                        "metric": "NNBASE_SD", "mean_std": f"{sd:.4f} ± 0.0000"
                    })

                    # ===== reconstruction baseline =====
                    pred_joints = ref_joints[nn_idx]  # (Nq, J, 3)
                    mpjpe = self._mpjpe(pred_joints, q_joints) * 100.0      # to cm
                    mpjpe_pa = self._pa_mpjpe(pred_joints, q_joints) * 100.0

                    logger.info(f"\t\tNNBASE MPJPE     - {sample_target}: {mpjpe:.3f} cm")
                    logger.info(f"\t\tNNBASE MPJPE_PA  - {sample_target}: {mpjpe_pa:.3f} cm")

                    rows.append({
                        "exp": exp_name, "step": step_str, "phase": "nn",
                        "dataset": dataset, "target": sample_target,
                        "metric": "NNBASE_MPJPE", "value": f"{mpjpe:.3f}"
                    })
                    rows.append({
                        "exp": exp_name, "step": step_str, "phase": "nn",
                        "dataset": dataset, "target": sample_target,
                        "metric": "NNBASE_MPJPE_PA", "value": f"{mpjpe_pa:.3f}"
                    })

        # =========================
        # Dump tables
        # =========================
        out_dir = base_samples_folder / "_eval"
        self._save_tables(rows, out_dir)
