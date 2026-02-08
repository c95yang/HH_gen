from logging import getLogger
from pathlib import Path
import csv
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import h5py

from config.config import ProjectConfig
from tridi.utils.metrics import generation
from tridi.utils.metrics import reconstruction

logger = getLogger(__name__)


class Evaluator:
    def __init__(self, cfg: ProjectConfig):
        self.cfg = cfg


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

                    if not samples_files:
                        logger.warning(f"\t\tNo sample files found at {samples_folder / sample_target}/samples_rep_*.hdf5, skipping.")
                        continue

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
        if getattr(self.cfg.eval, "sanity_gt_train_test", False):
            for dataset in self.cfg.run.datasets:
                for sample_target in self.cfg.eval.sampling_target:
                    acc = generation.sanity_nna_gt_train_vs_test(
                        self.cfg, dataset=dataset, sample_target=sample_target, max_per_split=5000
                    )
                    logger.info(f"[SANITY] GT train vs GT test 1-NNA ({dataset}, {sample_target}) = {acc*100:.2f}")


        if getattr(self.cfg.eval, "sanity_gt_test_test", False):
            logger.info("Running sanity: GT(test) vs GT(test) split-half 1-NNA")
            for dataset in self.cfg.run.datasets:
                for sample_target in self.cfg.eval.sampling_target:
                    val = generation.sanity_gt_test_test_1nna(
                        self.cfg,
                        reference_dataset=dataset,
                        reference_set=self.cfg.eval.split,   
                        sample_target=sample_target,
                        seed=getattr(self.cfg.eval, "sanity_seed", 42),
                        max_n=getattr(self.cfg.eval, "sanity_max_n", -1),
                    )
                    logger.info(f"[SANITY] GT {self.cfg.eval.split} vs GT {self.cfg.eval.split} (split-half) 1-NNA "
                                f"({dataset}, {sample_target}) = {val*100:.2f}")

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
                        if not samples_files:
                            logger.warning(f"\t\tNo sample files found at {samples_folder / sample_target}/samples_rep_*.hdf5, skipping reconstruction.")
                            continue
                        for samples_file in samples_files:
                            mpjpe, mpjpe_pa = reconstruction.get_sbj_metrics(self.cfg, samples_file, dataset, sample_target)
                            metrics[f"{sample_target}_MPJPE"].append(mpjpe)
                            metrics[f"{sample_target}_MPJPE_PA"].append(mpjpe_pa)

                        for k, v in metrics.items():
                            if not v:
                                continue
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

