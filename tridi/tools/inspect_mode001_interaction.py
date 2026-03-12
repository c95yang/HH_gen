#!/usr/bin/env python3
"""
Inspect mode=001 behavior for full H1/H2/Interaction model.

This script:
  1) runs sampling with mode=001 (generate interaction branch only)
  2) collects predicted interaction_latent
  3) decodes latent -> action label via cosine to action prototypes
  4) prints readable rows and saves jsonl artifact
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import torch

from tridi.utils.exp import init_exp
from tridi.model import get_model
from tridi.core.sampler import Sampler


def _to_int(x: Any) -> int:
    if isinstance(x, torch.Tensor):
        return int(x.detach().cpu().item())
    return int(x)


def _to_str(x: Any) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    return str(x)


def _as_list(x: Any) -> List[Any]:
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    return [x]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Inspect mode=001 interaction decoding")
    parser.add_argument("--config", "-c", type=str, nargs="+", required=True, help="YAML config files")
    parser.add_argument("overrides", type=str, nargs="*", help="OmegaConf dotlist overrides")

    parser.add_argument("--max_items", type=int, default=200, help="Max number of samples to inspect")
    parser.add_argument("--topk", type=int, default=3, help="Top-k decoded labels")
    parser.add_argument(
        "--sampling_strategy",
        type=str,
        choices=["sequential", "random", "stratified"],
        default="stratified",
        help="How to choose rows when --max_items is smaller than dataset size",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for random/stratified sampling")
    parser.add_argument(
        "--out_jsonl",
        type=str,
        default="",
        help="Optional output jsonl path. Defaults to <run.path>/artifacts/mode001_decode_step_<step>.jsonl",
    )
    return parser.parse_args()


def _split_config_and_overrides(config_items: List[str], overrides: List[str]):
    """
    Robustly support both styles:
      1) ... -c config/env.yaml scenarios/chi3d.yaml -- run.name=...
      2) ... -c config/env.yaml scenarios/chi3d.yaml run.name=...
    Any token containing '=' is treated as override.
    Any non-existing config path token is also treated as override.
    """
    cfg_files: List[str] = []
    ov: List[str] = list(overrides or [])

    for item in list(config_items or []):
        s = str(item)
        if "=" in s:
            ov.append(s)
            continue
        p = Path(s)
        if p.exists() and p.is_file():
            cfg_files.append(s)
        else:
            ov.append(s)

    return cfg_files, ov


def _group_by_gt(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = row.get("gt_label")
        key = "<None>" if key is None else str(key)
        groups[key].append(row)
    return dict(groups)


def _select_rows(rows: List[Dict[str, Any]], max_items: int, strategy: str, seed: int) -> List[Dict[str, Any]]:
    if max_items <= 0 or len(rows) <= max_items:
        return list(rows)

    if strategy == "sequential":
        return rows[:max_items]

    rng = random.Random(seed)

    if strategy == "random":
        picked = list(rows)
        rng.shuffle(picked)
        return picked[:max_items]

    # stratified
    groups = _group_by_gt(rows)
    labels = sorted(groups.keys())
    for label in labels:
        rng.shuffle(groups[label])

    selected: List[Dict[str, Any]] = []
    while len(selected) < max_items:
        made_progress = False
        for label in labels:
            if groups[label]:
                selected.append(groups[label].pop())
                made_progress = True
                if len(selected) >= max_items:
                    break
        if not made_progress:
            break

    rng.shuffle(selected)
    return selected[:max_items]


def _print_summary(rows: List[Dict[str, Any]], title: str) -> None:
    seq_count = len({str(r.get("seq")) for r in rows})
    gt_counter = Counter("<None>" if r.get("gt_label") is None else str(r.get("gt_label")) for r in rows)
    print(f"{title}")
    print(f"  rows={len(rows)}")
    print(f"  unique_sequences={seq_count}")
    print("  gt_label_distribution=")
    for label, count in sorted(gt_counter.items(), key=lambda x: (-x[1], x[0])):
        print(f"    {label}: {count}")


def main() -> None:
    args = parse_args()

    cfg_files, parsed_overrides = _split_config_and_overrides(args.config, args.overrides)
    if len(cfg_files) == 0:
        raise FileNotFoundError(
            "No valid config files were found. Pass existing files after -c, e.g. '-c config/env.yaml scenarios/chi3d.yaml'."
        )

    overrides = list(parsed_overrides)
    force_overrides = [
        "run.job=sample",
        "logging.wandb=false",
        "sample.mode=001",
        "sample.target=hdf5",
    ]
    for ov in force_overrides:
        key = ov.split("=", 1)[0]
        if not any(x.startswith(key + "=") for x in overrides):
            overrides.append(ov)

    cfg_args = argparse.Namespace(config=cfg_files, overrides=overrides)
    cfg = init_exp(cfg_args)

    model = get_model(cfg)
    sampler = Sampler(cfg, model)

    out_path = (
        Path(args.out_jsonl)
        if args.out_jsonl
        else Path(cfg.run.path) / "artifacts" / f"mode001_decode_step_{cfg.resume.step}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    all_rows = []
    allow_early_break = args.sampling_strategy == "sequential"

    for dataloader in sampler.dataloaders:
        for batch in dataloader:
            output, _ = sampler.sample_step(batch)
            if output.interaction_latent is None:
                raise RuntimeError(
                    "interaction_latent is None. Ensure model.use_interaction_diffusion=true and sample.mode=001."
                )

            decoded = model.conditioning_model.decode_interaction_latent_with_scores(
                output.interaction_latent,
                topk=args.topk,
            )

            seqs = _as_list(batch.sbj)
            t_stamps = _as_list(batch.t_stamp)
            gt_labels = _as_list(getattr(batch, "interaction_label", [None] * len(seqs)))
            latents = output.interaction_latent.detach().cpu().numpy()

            for i in range(len(seqs)):
                row = {
                    "seq": _to_str(seqs[i]),
                    "t_stamp": _to_int(t_stamps[i]),
                    "gt_label": None if gt_labels[i] is None else _to_str(gt_labels[i]),
                    "pred_label_top1": decoded["top1"][i],
                    "pred_topk_labels": decoded["topk_labels"][i],
                    "pred_topk_scores": [float(x) for x in decoded["topk_scores"][i]],
                    "pred_similarity_scores": [float(x) for x in decoded["similarities"][i]],
                    "decoder_candidate_actions": decoded["candidate_actions"],
                    "decoder_state_signature": decoded["decoder_state_signature"],
                    "interaction_latent": latents[i].astype(float).tolist(),
                }
                all_rows.append(row)
                n += 1

                if allow_early_break and n >= args.max_items:
                    break
            if allow_early_break and n >= args.max_items:
                break
        if allow_early_break and n >= args.max_items:
            break

    rows = _select_rows(all_rows, args.max_items, args.sampling_strategy, args.seed)

    _print_summary(rows, "Selection summary (before writing JSONL):")

    for i, row in enumerate(rows, start=1):
        print(
            f"[{i:04d}] seq={row['seq']} t={row['t_stamp']} "
            f"gt={row['gt_label']} pred={row['pred_label_top1']} topk={row['pred_topk_labels']}"
        )

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
