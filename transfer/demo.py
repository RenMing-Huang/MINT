"""Local one-shot demonstration loading and chunk construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Demonstration:
    name: str
    actions: np.ndarray
    eef_positions: np.ndarray


def resolve_demo_path(explicit_path: str) -> Path:
    path = Path(explicit_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Extracted demonstration trajectory not found: {path}")
    if path.suffix != ".npz":
        raise ValueError(f"demonstration trajectory must be an NPZ file, got: {path}")
    return path


def load_demonstration(path: str | Path) -> Demonstration:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        missing = {"actions", "eef_positions"} - set(data.files)
        if missing:
            raise ValueError(f"demonstration trajectory is missing arrays: {sorted(missing)}")
        actions = np.asarray(data["actions"], dtype=np.float32)
        eef_positions = np.asarray(data["eef_positions"])

    if actions.ndim != 2:
        raise ValueError(f"actions must have shape (T, D), got {actions.shape}")
    if eef_positions.ndim != 2 or eef_positions.shape[1] != 3:
        raise ValueError(f"eef_positions must have shape (T, 3), got {eef_positions.shape}")
    if len(actions) != len(eef_positions):
        raise ValueError(
            "actions and eef_positions must have equal trajectory lengths, got "
            f"{len(actions)} and {len(eef_positions)}"
        )
    return Demonstration(path.stem, actions, eef_positions)


def action_chunk(actions: np.ndarray, start: int, horizon: int) -> np.ndarray | None:
    """Return a fixed-horizon chunk, padding the demonstration's final action."""
    if start >= len(actions):
        return None
    chunk = actions[start : start + horizon]
    if len(chunk) < horizon:
        chunk = np.concatenate(
            [chunk, np.repeat(chunk[-1:], horizon - len(chunk), axis=0)], axis=0
        )
    return np.asarray(chunk, dtype=np.float32)
