import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _pick_field(keys: Iterable[str], candidates: List[str]) -> Optional[str]:
    keyset = set(keys)
    for name in candidates:
        if name in keyset:
            return name
    return None


def _normalize_label(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


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
    return rows


def evaluate(path: Path) -> Tuple[Dict[str, Any], Dict[str, Dict[str, int]]]:
    rows = _read_jsonl(path)
    if not rows:
        raise ValueError(f"No rows found in {path}")

    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())

    gt_field = _pick_field(all_keys, ["gt_label", "label_gt", "gt", "target_label", "label"])
    pred_field = _pick_field(all_keys, ["pred_label_top1", "pred_label", "top1_label", "prediction", "pred"])
    topk_field = _pick_field(all_keys, ["pred_topk_labels", "topk_labels", "pred_labels_topk", "topk"])

    if gt_field is None or pred_field is None:
        raise ValueError(
            f"Could not identify required fields. Found keys: {sorted(all_keys)}"
        )

    total = 0
    matches = 0
    mismatches = 0
    skipped = 0
    topk_total = 0
    topk_hits = 0

    confusion: Dict[str, Counter] = defaultdict(Counter)

    for row in rows:
        gt = _normalize_label(row.get(gt_field))
        pred = _normalize_label(row.get(pred_field))

        if gt is None or pred is None:
            skipped += 1
            continue

        total += 1
        confusion[gt][pred] += 1

        if gt == pred:
            matches += 1
        else:
            mismatches += 1

        if topk_field is not None:
            topk_vals = row.get(topk_field)
            if isinstance(topk_vals, list):
                normalized = {_normalize_label(v) for v in topk_vals}
                normalized.discard(None)
                topk_total += 1
                if gt in normalized:
                    topk_hits += 1

    accuracy = (matches / total) if total else 0.0
    mismatch_rate = (mismatches / total) if total else 0.0
    topk_hit_rate = (topk_hits / topk_total) if topk_total else None

    summary: Dict[str, Any] = {
        "input_file": str(path),
        "detected_fields": {
            "gt_field": gt_field,
            "pred_field": pred_field,
            "topk_field": topk_field,
        },
        "num_rows_raw": len(rows),
        "num_rows_used": total,
        "num_rows_skipped_missing_labels": skipped,
        "matches": matches,
        "mismatches": mismatches,
        "top1_accuracy": accuracy,
        "mismatch_rate": mismatch_rate,
    }
    if topk_field is not None:
        summary.update(
            {
                "topk_rows_used": topk_total,
                "topk_hits": topk_hits,
                "topk_hit_rate": topk_hit_rate,
            }
        )

    confusion_out = {
        gt: dict(sorted(pred_counts.items(), key=lambda x: (-x[1], x[0])))
        for gt, pred_counts in sorted(confusion.items(), key=lambda x: x[0])
    }

    return summary, confusion_out


def save_outputs(input_path: Path, summary: Dict[str, Any], confusion: Dict[str, Dict[str, int]]) -> Tuple[Path, Path]:
    summary_path = input_path.with_name(f"{input_path.stem}_summary.json")
    confusion_path = input_path.with_name(f"{input_path.stem}_confusion.csv")

    summary["summary_file"] = str(summary_path)
    summary["confusion_file"] = str(confusion_path)

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    label_space = sorted({label for label in confusion.keys()} | {p for preds in confusion.values() for p in preds.keys()})

    with confusion_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gt_label", *label_space, "row_total"])
        for gt in sorted(confusion.keys()):
            row_total = sum(confusion[gt].values())
            row = [gt] + [confusion[gt].get(pred, 0) for pred in label_space] + [row_total]
            writer.writerow(row)

    return summary_path, confusion_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate mode001 decoded interaction labels from JSONL")
    parser.add_argument("jsonl", type=str, help="Path to mode001_decode_step_*.jsonl")
    args = parser.parse_args()

    input_path = Path(args.jsonl)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    summary, confusion = evaluate(input_path)
    summary_path, confusion_path = save_outputs(input_path, summary, confusion)

    print("=== mode001 decode evaluation ===")
    print(f"input: {input_path}")
    print(f"rows(raw/used/skipped): {summary['num_rows_raw']}/{summary['num_rows_used']}/{summary['num_rows_skipped_missing_labels']}")
    print(f"matches: {summary['matches']}")
    print(f"mismatches: {summary['mismatches']}")
    print(f"top1_accuracy: {summary['top1_accuracy']:.6f}")
    print(f"mismatch_rate: {summary['mismatch_rate']:.6f}")
    if "topk_hit_rate" in summary:
        print(f"topk_rows_used: {summary['topk_rows_used']}")
        print(f"topk_hits: {summary['topk_hits']}")
        print(f"topk_hit_rate: {summary['topk_hit_rate']:.6f}")
    print(f"summary_file: {summary_path}")
    print(f"confusion_file: {confusion_path}")


if __name__ == "__main__":
    main()
