#!/usr/bin/env python3
"""
Analyze mode001 decode behavior to distinguish:
- decoder/prototype-space mismatch
- interaction-latent collapse
- likely training weakness
"""

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from tridi.core.sampler import Sampler
from tridi.model import get_model
from tridi.utils.exp import init_exp


def _as_list(x: Any) -> List[Any]:
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    return [x]


def _to_str(x: Any) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    return str(x)


def _split_config_and_overrides(config_items: List[str], overrides: List[str]):
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


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _pairwise_stats(latents: np.ndarray, max_pairs: int, seed: int) -> Dict[str, float]:
    n = int(latents.shape[0])
    if n <= 1:
        return {
            "avg_pairwise_cosine": float("nan"),
            "avg_pairwise_l2": float("nan"),
            "num_pairs_used": 0,
        }

    total_pairs = n * (n - 1) // 2
    rng = random.Random(seed)

    if total_pairs <= max_pairs:
        # full matrix
        x = torch.from_numpy(latents).float()
        x_norm = F.normalize(x, dim=1)
        sim = x_norm @ x_norm.T
        l2 = torch.cdist(x, x, p=2)
        iu = torch.triu_indices(n, n, offset=1)
        sim_vals = sim[iu[0], iu[1]]
        l2_vals = l2[iu[0], iu[1]]
        return {
            "avg_pairwise_cosine": float(sim_vals.mean().item()),
            "avg_pairwise_l2": float(l2_vals.mean().item()),
            "num_pairs_used": int(total_pairs),
        }

    # sampled pairs for large N
    x = torch.from_numpy(latents).float()
    x_norm = F.normalize(x, dim=1)

    pairs = set()
    while len(pairs) < max_pairs:
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        pairs.add((i, j))

    idx_i = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    idx_j = torch.tensor([p[1] for p in pairs], dtype=torch.long)

    sim_vals = (x_norm[idx_i] * x_norm[idx_j]).sum(dim=1)
    l2_vals = torch.norm(x[idx_i] - x[idx_j], dim=1, p=2)

    return {
        "avg_pairwise_cosine": float(sim_vals.mean().item()),
        "avg_pairwise_l2": float(l2_vals.mean().item()),
        "num_pairs_used": int(len(pairs)),
    }


def _prototype_similarity_stats(
    sims: torch.Tensor,
    candidate_actions: List[str],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    avg_score_per_proto = sims.mean(dim=0).detach().cpu().tolist()

    topk_vals, topk_idxs = torch.topk(sims, k=min(2, sims.shape[1]), dim=1)
    top1_idxs = topk_idxs[:, 0]
    top1_counts = Counter(int(i) for i in top1_idxs.detach().cpu().tolist())

    if sims.shape[1] >= 2:
        margins = (topk_vals[:, 0] - topk_vals[:, 1]).detach().cpu().numpy()
        avg_margin = float(np.mean(margins))
        median_margin = float(np.median(margins))
    else:
        margins = np.array([math.nan], dtype=np.float32)
        avg_margin = float("nan")
        median_margin = float("nan")

    per_proto_rows: List[Dict[str, Any]] = []
    n = int(sims.shape[0])
    for idx, label in enumerate(candidate_actions):
        c = int(top1_counts.get(idx, 0))
        per_proto_rows.append(
            {
                "label": label,
                "avg_similarity": float(avg_score_per_proto[idx]),
                "top1_count": c,
                "top1_rate": float(c / n) if n > 0 else 0.0,
            }
        )

    per_proto_rows.sort(key=lambda r: (-r["top1_count"], r["label"]))

    top_label = per_proto_rows[0]["label"] if per_proto_rows else "<none>"
    top_rate = per_proto_rows[0]["top1_rate"] if per_proto_rows else 0.0

    margin_regime = "tiny"
    if not math.isnan(avg_margin):
        if avg_margin >= 0.05:
            margin_regime = "large"
        elif avg_margin >= 0.01:
            margin_regime = "moderate"

    summary = {
        "num_samples": int(sims.shape[0]),
        "num_prototypes": int(sims.shape[1]),
        "avg_top1_minus_top2_margin": avg_margin,
        "median_top1_minus_top2_margin": median_margin,
        "dominant_top1_label": top_label,
        "dominant_top1_rate": top_rate,
        "dominance_margin_regime": margin_regime,
    }
    return summary, per_proto_rows


def _extract_saved_decode(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if len(rows) == 0:
        return None
    if not all("pred_similarity_scores" in r and "decoder_candidate_actions" in r for r in rows):
        return None

    candidate_actions = rows[0].get("decoder_candidate_actions")
    if not isinstance(candidate_actions, list) or len(candidate_actions) == 0:
        return None
    candidate_actions = [_to_str(x) for x in candidate_actions]

    sims: List[List[float]] = []
    sigs = set()
    for r in rows:
        ca = r.get("decoder_candidate_actions")
        if list(ca) != list(candidate_actions):
            return None
        vals = r.get("pred_similarity_scores")
        if not isinstance(vals, list) or len(vals) != len(candidate_actions):
            return None
        sims.append([float(x) for x in vals])
        if "decoder_state_signature" in r and r.get("decoder_state_signature"):
            sigs.add(_to_str(r.get("decoder_state_signature")))

    return {
        "candidate_actions": candidate_actions,
        "similarities": np.asarray(sims, dtype=np.float32),
        "decoder_state_signatures": sorted(sigs),
    }


def _decoder_checks(conditioning_model, sampler, candidate_actions: List[str], latent_dim: int, max_gt_items: int):
    device = next(conditioning_model.parameters()).device

    # Check 1: prototype self-consistency
    proto_latent = conditioning_model.get_interaction_latent(
        interaction_label=candidate_actions,
        batch=None,
        device=device,
        latent_dim=latent_dim,
    )
    proto_decoded = conditioning_model.decode_interaction_latent_with_scores(
        proto_latent,
        topk=1,
        candidate_actions=candidate_actions,
    )["top1"]
    proto_total = len(candidate_actions)
    proto_ok = sum(1 for gt, pred in zip(candidate_actions, proto_decoded) if gt == pred)

    # Check 2: GT interaction latent from current pipeline labels
    used = 0
    gt_ok = 0
    gt_skipped_unknown = 0

    for dataloader in sampler.dataloaders:
        for batch in dataloader:
            labels = [_to_str(x) for x in _as_list(getattr(batch, "interaction_label", []))]
            if len(labels) == 0:
                continue

            # keep only labels that exist in candidate actions
            keep_idx = [i for i, lbl in enumerate(labels) if lbl in candidate_actions]
            gt_skipped_unknown += (len(labels) - len(keep_idx))
            if len(keep_idx) == 0:
                continue

            selected_labels = [labels[i] for i in keep_idx]
            gt_latent = conditioning_model.get_interaction_latent(
                interaction_label=selected_labels,
                batch=batch,
                device=device,
                latent_dim=latent_dim,
            )
            dec = conditioning_model.decode_interaction_latent_with_scores(
                gt_latent,
                topk=1,
                candidate_actions=candidate_actions,
            )["top1"]
            gt_ok += sum(1 for gt, pred in zip(selected_labels, dec) if gt == pred)
            used += len(selected_labels)

            if used >= max_gt_items:
                break
        if used >= max_gt_items:
            break

    checks = {
        "prototype_self_decode": {
            "total": proto_total,
            "correct": proto_ok,
            "accuracy": float(proto_ok / proto_total) if proto_total > 0 else float("nan"),
        },
        "gt_latent_decode": {
            "total": used,
            "correct": gt_ok,
            "accuracy": float(gt_ok / used) if used > 0 else float("nan"),
            "skipped_unknown_labels": gt_skipped_unknown,
        },
    }
    return checks


def _diagnose(decoder_checks: Dict[str, Any], diversity: Dict[str, Any], proto_summary: Dict[str, Any]) -> str:
    proto_acc = float(decoder_checks["prototype_self_decode"]["accuracy"])
    gt_acc = float(decoder_checks["gt_latent_decode"]["accuracy"])
    replay_acc = float(decoder_checks["jsonl_pred_redecode_consistency"]["accuracy"])

    decoder_bad = (
        (not math.isnan(proto_acc) and proto_acc < 0.95)
        or (not math.isnan(gt_acc) and gt_acc < 0.90)
        or (not math.isnan(replay_acc) and replay_acc < 0.95)
    )
    if decoder_bad:
        return "decoder-space mismatch"

    mean_dim_std = float(diversity["latent_mean_std_across_dims"])
    avg_cos = float(diversity["avg_pairwise_cosine"])
    avg_l2 = float(diversity["avg_pairwise_l2"])

    collapsed = (mean_dim_std < 0.02) or (avg_cos > 0.98 and avg_l2 < 1.0)
    if collapsed:
        return "latent collapse"

    return "likely training weakness"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Analyze mode001 decode collapse")
    parser.add_argument("--config", "-c", type=str, nargs="+", required=True, help="YAML config files")
    parser.add_argument("overrides", type=str, nargs="*", help="OmegaConf dotlist overrides")
    parser.add_argument("--jsonl", type=str, required=True, help="Path to mode001_decode_step_*.jsonl")
    parser.add_argument("--max_gt_items", type=int, default=500, help="How many GT-label samples to use for GT latent decode check")
    parser.add_argument("--max_pairwise_pairs", type=int, default=200000, help="Maximum pairwise pairs for diversity stats")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

    cfg_files, parsed_overrides = _split_config_and_overrides(args.config, args.overrides)
    if len(cfg_files) == 0:
        raise FileNotFoundError("No valid config files were found after -c")

    # enforce sample job in same style as existing script
    overrides = list(parsed_overrides)
    force_overrides = ["run.job=sample", "logging.wandb=false", "sample.mode=001", "sample.target=hdf5"]
    for ov in force_overrides:
        key = ov.split("=", 1)[0]
        if not any(x.startswith(key + "=") for x in overrides):
            overrides.append(ov)

    cfg = init_exp(argparse.Namespace(config=cfg_files, overrides=overrides))
    model = get_model(cfg)
    sampler = Sampler(cfg, model)

    conditioning_model = model.conditioning_model
    if not getattr(conditioning_model, "use_interaction_conditioning", False):
        raise RuntimeError("Interaction conditioning must be enabled for this analysis")

    rows = _read_jsonl(jsonl_path)

    # Parse predicted latents
    latents = []
    for row in rows:
        v = row.get("interaction_latent", None)
        if v is None:
            continue
        latents.append(v)
    if len(latents) == 0:
        raise ValueError("No interaction_latent found in JSONL")

    pred_latents_np = np.asarray(latents, dtype=np.float32)
    n, d = pred_latents_np.shape

    dim_mean = pred_latents_np.mean(axis=0)
    dim_std = pred_latents_np.std(axis=0)

    diversity = {
        "num_samples": int(n),
        "latent_dim": int(d),
        "latent_mean_abs_mean_across_dims": float(np.mean(np.abs(dim_mean))),
        "latent_mean_std_across_dims": float(np.mean(dim_std)),
        "latent_min_std_across_dims": float(np.min(dim_std)),
        "latent_max_std_across_dims": float(np.max(dim_std)),
    }
    diversity.update(_pairwise_stats(pred_latents_np, args.max_pairwise_pairs, args.seed))

    device = next(model.parameters()).device
    pred_latents_t = torch.from_numpy(pred_latents_np).to(device=device, dtype=torch.float32)

    candidate_actions = [a for a in sorted(conditioning_model.interaction_prompts.keys()) if a != "Unknown"] or ["Unknown"]
    saved_decode = _extract_saved_decode(rows)

    jsonl_pred = [_to_str(r.get("pred_label_top1")) for r in rows]
    if saved_decode is not None:
        candidate_actions = saved_decode["candidate_actions"]
        sims_t = torch.from_numpy(saved_decode["similarities"]).to(device=device, dtype=torch.float32)
        saved_idxs = torch.argmax(sims_t, dim=1).detach().cpu().tolist()
        saved_top1 = [candidate_actions[int(i)] for i in saved_idxs]
        saved_ok = sum(1 for a, b in zip(jsonl_pred, saved_top1) if a == b)
        saved_total = min(len(jsonl_pred), len(saved_top1))
    else:
        dec_for_saved = conditioning_model.decode_interaction_latent_with_scores(
            pred_latents_t,
            topk=1,
            candidate_actions=candidate_actions,
        )
        sims_t = torch.tensor(dec_for_saved["similarities"], device=device, dtype=torch.float32)
        saved_top1 = dec_for_saved["top1"]
        saved_ok = sum(1 for a, b in zip(jsonl_pred, saved_top1) if a == b)
        saved_total = min(len(jsonl_pred), len(saved_top1))

    proto_summary, proto_rows = _prototype_similarity_stats(
        sims=sims_t,
        candidate_actions=candidate_actions,
    )

    decoder_checks = _decoder_checks(
        conditioning_model=conditioning_model,
        sampler=sampler,
        candidate_actions=candidate_actions,
        latent_dim=d,
        max_gt_items=args.max_gt_items,
    )

    # Check 3a: consistency between stored JSONL top1 and saved decode info (if provided).
    decoder_checks["jsonl_saved_decode_consistency"] = {
        "total": int(saved_total),
        "correct": int(saved_ok),
        "accuracy": float(saved_ok / saved_total) if saved_total > 0 else float("nan"),
        "used_saved_similarity_scores": bool(saved_decode is not None),
    }

    # Check 3b: consistency between stored JSONL top1 labels and current decoder replay.
    replay_decoded = conditioning_model.decode_interaction_latent_with_scores(
        pred_latents_t,
        topk=1,
        candidate_actions=candidate_actions,
    )
    replay = replay_decoded["top1"]
    replay_ok = sum(1 for a, b in zip(jsonl_pred, replay) if a == b)
    replay_total = min(len(jsonl_pred), len(replay))
    decoder_checks["jsonl_pred_redecode_consistency"] = {
        "total": int(replay_total),
        "correct": int(replay_ok),
        "accuracy": float(replay_ok / replay_total) if replay_total > 0 else float("nan"),
    }

    diagnosis = _diagnose(decoder_checks, diversity, proto_summary)

    # Save outputs next to input JSONL
    summary_json = jsonl_path.with_name(f"{jsonl_path.stem}_analysis_summary.json")
    proto_csv = jsonl_path.with_name(f"{jsonl_path.stem}_prototype_similarity.csv")
    dim_csv = jsonl_path.with_name(f"{jsonl_path.stem}_latent_dim_stats.csv")

    with proto_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "avg_similarity", "top1_count", "top1_rate"])
        writer.writeheader()
        for row in proto_rows:
            writer.writerow(row)

    with dim_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dim", "mean", "std"])
        for i in range(d):
            writer.writerow([i, float(dim_mean[i]), float(dim_std[i])])

    gt_counts = Counter(_to_str(r.get("gt_label")) for r in rows if r.get("gt_label") is not None)
    pred_counts = Counter(_to_str(r.get("pred_label_top1")) for r in rows if r.get("pred_label_top1") is not None)

    out = {
        "input_jsonl": str(jsonl_path),
        "decoder_checks": decoder_checks,
        "decoder_state": {
            "jsonl_decoder_state_signatures": [] if saved_decode is None else saved_decode["decoder_state_signatures"],
            "replay_decoder_state_signature": replay_decoded.get("decoder_state_signature", ""),
            "candidate_actions": candidate_actions,
        },
        "predicted_latent_diversity": diversity,
        "prototype_similarity_summary": proto_summary,
        "jsonl_gt_label_distribution": dict(sorted(gt_counts.items())),
        "jsonl_pred_top1_distribution": dict(sorted(pred_counts.items())),
        "diagnosis": diagnosis,
        "output_files": {
            "summary_json": str(summary_json),
            "prototype_similarity_csv": str(proto_csv),
            "latent_dim_stats_csv": str(dim_csv),
        },
    }

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # Terminal summary
    print("=== mode001 diagnostic summary ===")
    print(f"input: {jsonl_path}")
    print(
        "decoder_checks: "
        f"prototype_self_acc={decoder_checks['prototype_self_decode']['accuracy']:.6f}, "
        f"gt_latent_acc={decoder_checks['gt_latent_decode']['accuracy']:.6f} "
        f"(n={decoder_checks['gt_latent_decode']['total']}), "
        f"jsonl_saved_decode_consistency={decoder_checks['jsonl_saved_decode_consistency']['accuracy']:.6f}, "
        f"jsonl_redecode_consistency={decoder_checks['jsonl_pred_redecode_consistency']['accuracy']:.6f}"
    )
    print(
        "latent_diversity: "
        f"mean_dim_std={diversity['latent_mean_std_across_dims']:.6f}, "
        f"avg_pairwise_cos={diversity['avg_pairwise_cosine']:.6f}, "
        f"avg_pairwise_l2={diversity['avg_pairwise_l2']:.6f}"
    )
    print(
        "prototype_scores: "
        f"dominant_top1={proto_summary['dominant_top1_label']} "
        f"(rate={proto_summary['dominant_top1_rate']:.6f}), "
        f"avg_top1_top2_margin={proto_summary['avg_top1_minus_top2_margin']:.6f} "
        f"[{proto_summary['dominance_margin_regime']}]"
    )
    print(f"diagnosis: {diagnosis}")
    print(f"summary_json: {summary_json}")
    print(f"prototype_similarity_csv: {proto_csv}")
    print(f"latent_dim_stats_csv: {dim_csv}")


if __name__ == "__main__":
    main()
