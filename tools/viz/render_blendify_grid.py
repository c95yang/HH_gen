import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import logging
from pathlib import Path
import re
from typing import Dict, List

from PIL import Image

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build prediction grids from rendered pred_repXX images")
    parser.add_argument("--renders_root", type=str, required=True)
    parser.add_argument("--pattern", type=str, default="pred_rep*.png")
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--out_name", type=str, default="pred_grid.png")
    parser.add_argument("--dry_run", action="store_true", help="Only print discovered case dirs and image counts")
    return parser.parse_args()


def _natural_key(path: Path):
    s = path.name.lower()
    return [int(tok) if tok.isdigit() else tok for tok in re.split(r"(\d+)", s)]


def _discover_case_dirs(root: Path, pattern: str) -> Dict[Path, List[Path]]:
    """
    Discovery logic:
    1) If root has direct matches, treat root as a single case directory.
    2) Else recursively find matches and group by parent directory.
    """
    direct = sorted([p for p in root.glob(pattern) if p.is_file()], key=_natural_key)
    if direct:
        return {root: direct}

    grouped: Dict[Path, List[Path]] = {}
    for p in root.rglob(pattern):
        if not p.is_file():
            continue
        grouped.setdefault(p.parent, []).append(p)

    for d in list(grouped.keys()):
        grouped[d] = sorted(grouped[d], key=_natural_key)
    return dict(sorted(grouped.items(), key=lambda kv: str(kv[0])))


def _make_image_grid(img_paths: List[Path], out_path: Path, cols: int) -> None:
    if not img_paths:
        raise ValueError("img_paths is empty")

    cols = max(1, int(cols))
    with Image.open(img_paths[0]) as im0:
        w, h = im0.size
        mode = im0.mode

    rows = (len(img_paths) + cols - 1) // cols
    canvas = Image.new(mode, (cols * w, rows * h), color=(0, 0, 0) if mode in {"RGB", "RGBA"} else 0)

    for idx, p in enumerate(img_paths):
        with Image.open(p) as im:
            if im.size != (w, h) or im.mode != mode:
                im = im.convert(mode).resize((w, h))
            r = idx // cols
            c = idx % cols
            canvas.paste(im, (c * w, r * h))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
    args = parse_args()

    root = Path(args.renders_root)
    if not root.exists():
        raise FileNotFoundError(f"renders_root not found: {root}")

    cases = _discover_case_dirs(root, args.pattern)
    if not cases:
        logger.warning("No images matching pattern '%s' found under %s", args.pattern, root)
        return

    logger.info("Found %d case folder(s)", len(cases))
    if args.dry_run:
        for d, imgs in cases.items():
            logger.info("[dry_run] %s -> %d files", d, len(imgs))
        return

    for d, imgs in cases.items():
        if not imgs:
            logger.warning("Skip case folder with no matching images: %s", d)
            continue
        out = d / args.out_name
        try:
            _make_image_grid(imgs, out, cols=max(1, int(args.cols)))
            logger.info("Saved %s", out)
        except Exception as e:  # noqa: BLE001
            logger.warning("Skip case %s due to grid build error: %s", d, e)


if __name__ == "__main__":
    main()
