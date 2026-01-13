from logging import getLogger
from pathlib import Path
from collections import defaultdict
import csv
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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
            if r.get("phase", "") == "gen":
                val = r.get("mean_std", "")
            else:
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

    # -------------------------
    # main entry
    # -------------------------
    def evaluate(self):
        base_samples_folder = self._resolve_base_samples_folder()
        print(base_samples_folder)

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
                    print(samples_files)

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
                        if len(v) > 0:
                            mean = float(np.mean(v))
                            std = float(np.std(v))
                            logger.info(f"\t\t{k:<6s} - {sample_target}: {mean:.4f} ± {std:.4f}")
                            rows.append({
                                "exp": exp_name,
                                "step": step_str,
                                "phase": "gen",
                                "dataset": dataset,
                                "target": sample_target,
                                "metric": k,
                                "mean": f"{mean:.6f}",
                                "std": f"{std:.6f}",
                                "mean_std": f"{mean:.4f} ± {std:.4f}",
                                "raw_values": json.dumps([float(x) for x in v]),
                            })
                        else:
                            logger.warning(f"\t\t{k:<6s} - {sample_target}: No data (empty list)")
                            rows.append({
                                "exp": exp_name,
                                "step": step_str,
                                "phase": "gen",
                                "dataset": dataset,
                                "target": sample_target,
                                "metric": k,
                                "mean": "",
                                "std": "",
                                "mean_std": "",
                                "raw_values": "[]",
                            })

        # =========================
        # Reconstruction metrics
        # =========================
        if self.cfg.eval.use_rec_metrics:
            logger.info("Evaluating reconstruction")
            for dataset in self.cfg.run.datasets:
                logger.info(f"\ton {dataset}")
                samples_folder = base_samples_folder / dataset

                metrics = {
                    "MPJPE": [], "MPJPE_PA": [],
                    "MPJPE_SECOND_SBJ": [], "MPJPE_PA_SECOND_SBJ": []
                }

                for sample_target in self.cfg.eval.sampling_target:
                    if "sbj" in sample_target:
                        samples_files = list(samples_folder.glob(f"{sample_target}/samples_rep_*.hdf5"))
                        for samples_file in samples_files:
                            mpjpe, mpjpe_pa, mpjpe_second_sbj, mpjpe_pa_second_sbj = \
                                reconstruction.get_sbj_metrics(self.cfg, samples_file, dataset)

                            metrics["MPJPE"].append(mpjpe)
                            metrics["MPJPE_PA"].append(mpjpe_pa)
                            metrics["MPJPE_SECOND_SBJ"].append(mpjpe_second_sbj)
                            metrics["MPJPE_PA_SECOND_SBJ"].append(mpjpe_pa_second_sbj)

                # keep your original aggregation logic:
                # for each sample -> min over reps -> mean over samples
                for k, v in metrics.items():
                    if len(v) == 0:
                        logger.warning(f"\t\t{k:<20s}: No data")
                        rows.append({
                            "exp": exp_name,
                            "step": step_str,
                            "phase": "rec",
                            "dataset": dataset,
                            "target": "sbj/second_sbj",
                            "metric": k,
                            "value": "",
                            "raw_values": "[]",
                        })
                        continue

                    stacked = np.stack(v, axis=1)
                    val = float(np.mean(np.min(stacked, axis=1)))
                    logger.info(f"\t\t{k:<20s}: {val:.4f}")

                    # store per-rep mean as raw_values for debugging
                    per_rep_means = [float(np.mean(x)) for x in v]

                    rows.append({
                        "exp": exp_name,
                        "step": step_str,
                        "phase": "rec",
                        "dataset": dataset,
                        "target": "sbj/second_sbj",
                        "metric": k,
                        "value": f"{val:.6f}",
                        "raw_values": json.dumps(per_rep_means),
                    })

        # =========================
        # Dump tables
        # =========================
        out_dir = base_samples_folder / "_eval"
        self._save_tables(rows, out_dir)
