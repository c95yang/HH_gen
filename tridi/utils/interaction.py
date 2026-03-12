import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


_CANONICAL_ACTIONS = {
    "grab": "Grab",
    "handshake": "Handshake",
    "hit": "Hit",
    "push": "Push",
    "holdinghands": "HoldingHands",
    "posing": "Posing",
    "hug": "Hug",
    "kick": "Kick",
}


def _normalize_action_token(token: str) -> str:
    return re.sub(r"[^a-z]", "", token.lower())


def canonicalize_action_label(token: str) -> Optional[str]:
    if token is None:
        return None
    key = _normalize_action_token(str(token))
    if not key:
        return None
    return _CANONICAL_ACTIONS.get(key)


def parse_action_from_seq(seq: str) -> Optional[str]:
    if not seq:
        return None

    stem = Path(str(seq)).stem
    # Common case: s03_Hit_7 -> Hit
    for token in re.split(r"[_\-\s]+", stem):
        action = canonicalize_action_label(token)
        if action is not None:
            return action

        # Handle compact tokens like Hit7 / HoldingHands12
        m = re.match(r"^([A-Za-z]+)\d+$", token)
        if m:
            action = canonicalize_action_label(m.group(1))
            if action is not None:
                return action

    return None


def parse_action_from_video(filename: str) -> Optional[str]:
    if not filename:
        return None

    stem = Path(str(filename)).stem
    for token in re.split(r"[_\-\s]+", stem):
        action = canonicalize_action_label(token)
        if action is not None:
            return action

        m = re.match(r"^([A-Za-z]+)\d+$", token)
        if m:
            action = canonicalize_action_label(m.group(1))
            if action is not None:
                return action

    return None


class InteractionLabelResolver:
    def __init__(self, interaction_source: str = "both", raw_root: Optional[Path] = None):
        if interaction_source not in {"seq_name", "video_name", "both"}:
            raise ValueError(
                f"Unsupported interaction_source={interaction_source}. "
                f"Expected one of seq_name|video_name|both"
            )

        self.interaction_source = interaction_source
        self.raw_root = Path(raw_root) if raw_root else None

        self._seq_to_video: Dict[str, Optional[str]] = {}
        self._subject_video_cache: Dict[str, List[Path]] = {}

    def resolve(self, seq_name: str, video_name: Optional[str] = None) -> Optional[str]:
        action_seq = parse_action_from_seq(seq_name) if self.interaction_source in {"seq_name", "both"} else None
        if action_seq is not None:
            return action_seq

        if self.interaction_source in {"video_name", "both"}:
            if video_name:
                action_video = parse_action_from_video(video_name)
                if action_video is not None:
                    return action_video

            guessed_video = self._resolve_video_for_seq(seq_name)
            if guessed_video is not None:
                action_video = parse_action_from_video(guessed_video)
                if action_video is not None:
                    return action_video

        return None

    def _resolve_video_for_seq(self, seq_name: str) -> Optional[str]:
        if seq_name in self._seq_to_video:
            return self._seq_to_video[seq_name]

        if self.raw_root is None or (not self.raw_root.exists()):
            self._seq_to_video[seq_name] = None
            return None

        subject_match = re.match(r"^(s\d+)", seq_name, flags=re.IGNORECASE)
        subject = subject_match.group(1).lower() if subject_match else None
        index_match = re.search(r"(\d+)$", seq_name)
        seq_index = int(index_match.group(1)) if index_match else None

        if subject is None:
            self._seq_to_video[seq_name] = None
            return None

        videos = self._get_subject_videos(subject)
        if not videos:
            self._seq_to_video[seq_name] = None
            return None

        selected: Optional[Path] = None
        if seq_index is not None:
            with_same_index = []
            for p in videos:
                stem = p.stem
                m = re.search(r"(\d+)$", stem)
                if m and int(m.group(1)) == seq_index:
                    with_same_index.append(p)
            if with_same_index:
                selected = sorted(with_same_index)[0]

        if selected is None:
            selected = sorted(videos)[0]

        out = selected.name if selected is not None else None
        self._seq_to_video[seq_name] = out
        return out

    def _get_subject_videos(self, subject: str) -> List[Path]:
        if subject in self._subject_video_cache:
            return self._subject_video_cache[subject]

        videos: List[Path] = []
        patterns = [
            f"**/{subject}/videos/*.mp4",
            f"**/{subject}/videos/*.MP4",
            f"**/{subject}/video/*.mp4",
            f"**/{subject}/video/*.MP4",
        ]
        for pattern in patterns:
            videos.extend(self.raw_root.glob(pattern))

        videos = sorted(set(videos))
        self._subject_video_cache[subject] = videos
        return videos


def _extract_numeric_vector(obj) -> Optional[np.ndarray]:
    if isinstance(obj, list):
        if len(obj) == 0:
            return None
        if all(isinstance(v, (int, float)) for v in obj):
            return np.asarray(obj, dtype=np.float32)
        return None

    if isinstance(obj, dict):
        for key in ["interaction_contact_signature", "contact_signature", "signature", "vector", "values"]:
            if key in obj:
                v = _extract_numeric_vector(obj[key])
                if v is not None:
                    return v

        if len(obj) > 0 and all(isinstance(v, (int, float)) for v in obj.values()):
            keys = sorted(obj.keys())
            return np.asarray([obj[k] for k in keys], dtype=np.float32)

    return None


def load_action_contact_signatures(
    raw_root: Optional[Path],
    signature_path: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    signatures: Dict[str, List[np.ndarray]] = {}

    candidate_files: List[Path] = []
    if signature_path:
        p = Path(signature_path)
        if p.exists() and p.is_file():
            candidate_files = [p]
        elif p.exists() and p.is_dir():
            candidate_files = sorted(p.rglob("interaction_contact_signature.json"))
        else:
            return {}
    else:
        if raw_root is None:
            return {}
        rr = Path(raw_root)
        if not rr.exists():
            return {}
        candidate_files = sorted(rr.rglob("interaction_contact_signature.json"))

    for file_path in candidate_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        vec = _extract_numeric_vector(data)
        if vec is None:
            continue

        action = None
        if isinstance(data, dict) and "action" in data:
            action = canonicalize_action_label(str(data["action"]))

        if action is None:
            # Try sequence-like folder names up the tree.
            for parent in [file_path.parent.name, file_path.parent.parent.name if file_path.parent.parent else ""]:
                action = parse_action_from_seq(parent) or parse_action_from_video(parent)
                if action is not None:
                    break

        if action is None:
            continue

        signatures.setdefault(action, []).append(vec.astype(np.float32))

    # Aggregate repeated entries per action by mean.
    out: Dict[str, np.ndarray] = {}
    for action, vecs in signatures.items():
        if len(vecs) == 0:
            continue
        out[action] = np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)

    return out
