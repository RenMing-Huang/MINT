#!/usr/bin/env python3
"""Generate playlist JSON manifests for the static project page.

Why this exists:
- Browsers can't reliably list directories on a static site.
- We generate a JSON manifest by scanning the `static/videos/...` folders.

Outputs:
- static/videos/main_results/playlist.json
- static/videos/generalization/playlist.json
- static/videos/one_shot/playlist.json

Usage:
  python3 scripts/generate_playlists.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


VIDEO_EXTS = {".mp4", ".webm", ".mov", ".m4v"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
MEDIA_EXTS = VIDEO_EXTS | IMAGE_EXTS


def _is_visible_dir(path: Path) -> bool:
    return path.is_dir() and not path.name.startswith(".")


def _iter_video_files(directory: Path, recursive: bool) -> list[Path]:
    if not directory.exists():
        return []

    if recursive:
        candidates: Iterable[Path] = directory.rglob("*")
    else:
        candidates = directory.iterdir()

    files: list[Path] = []
    for p in candidates:
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        if p.suffix.lower() not in VIDEO_EXTS:
            continue
        files.append(p)

    return sorted(files, key=lambda x: x.as_posix())


def _iter_media_files(directory: Path, recursive: bool) -> list[Path]:
    """Return both video and image files (used for one-shot where a cell may be an image)."""
    if not directory.exists():
        return []

    if recursive:
        candidates: Iterable[Path] = directory.rglob("*")
    else:
        candidates = directory.iterdir()

    files: list[Path] = []
    for p in candidates:
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        if p.suffix.lower() not in MEDIA_EXTS:
            continue
        files.append(p)

    return sorted(files, key=lambda x: x.as_posix())


def _to_web_path(repo_root: Path, file_path: Path) -> str:
    return file_path.relative_to(repo_root).as_posix()


def generate_main_results(repo_root: Path) -> dict[str, list[str]]:
    base = repo_root / "static" / "videos" / "main_results"
    data: dict[str, list[str]] = {}

    if not base.exists():
        return data

    for benchmark_dir in sorted([p for p in base.iterdir() if _is_visible_dir(p)], key=lambda p: p.name.lower()):
        files = _iter_video_files(benchmark_dir, recursive=False)
        if not files:
            data[benchmark_dir.name] = []
            continue
        data[benchmark_dir.name] = [_to_web_path(repo_root, f) for f in files]

    return data


def generate_generalization(repo_root: Path) -> dict[str, dict[str, list[str]]]:
    base = repo_root / "static" / "videos" / "generalization"
    data: dict[str, dict[str, list[str]]] = {}

    if not base.exists():
        return data

    for category_dir in sorted([p for p in base.iterdir() if _is_visible_dir(p)], key=lambda p: p.name.lower()):
        tasks: dict[str, list[str]] = {}
        for task_dir in sorted([p for p in category_dir.iterdir() if _is_visible_dir(p)], key=lambda p: p.name.lower()):
            files = _iter_video_files(task_dir, recursive=True)
            if not files:
                continue
            tasks[task_dir.name] = [_to_web_path(repo_root, f) for f in files]

        if tasks:
            data[category_dir.name] = tasks

    return data


def generate_one_shot(repo_root: Path) -> dict[str, dict[str, dict[str, str]]]:
        """Generate a manifest for one-shot transfer videos.

        Expected folder structure:
            static/videos/one_shot/<Category>/<Task>/<Method>/*.{mp4,png,...}

        Output schema:
            {
                "Category": {
                    "Task": {
                        "fine_tuning": ".../file.mp4",
                        "intent_injection": ".../file.mp4"
                    }
                }
            }

        Notes:
        - Prefer videos; fall back to images if a method directory has no videos.
        - If multiple files exist, pick the lexicographically first.
        """
        base = repo_root / "static" / "videos" / "one_shot"
        data: dict[str, dict[str, dict[str, str]]] = {}

        if not base.exists():
                return data

        def method_key(method_dir_name: str) -> str | None:
                name = method_dir_name.lower()
                if "fine" in name:
                        return "fine_tuning"
                if "intent" in name:
                        return "intent_injection"
                return None

        for category_dir in sorted([p for p in base.iterdir() if _is_visible_dir(p)], key=lambda p: p.name.lower()):
                tasks: dict[str, dict[str, str]] = {}
                for task_dir in sorted([p for p in category_dir.iterdir() if _is_visible_dir(p)], key=lambda p: p.name.lower()):
                        methods: dict[str, str] = {}
                        for method_dir in sorted([p for p in task_dir.iterdir() if _is_visible_dir(p)], key=lambda p: p.name.lower()):
                                key = method_key(method_dir.name)
                                if not key:
                                        continue

                                media_files = _iter_media_files(method_dir, recursive=True)
                                if not media_files:
                                        continue

                                # Prefer videos; otherwise first image.
                                videos = [p for p in media_files if p.suffix.lower() in VIDEO_EXTS]
                                chosen = videos[0] if videos else media_files[0]
                                methods[key] = _to_web_path(repo_root, chosen)

                        if methods:
                                tasks[task_dir.name] = methods

                if tasks:
                        data[category_dir.name] = tasks

        return data


def generate_real_world(repo_root: Path) -> dict[str, dict[str, str]]:
    """Generate a manifest for real-world videos.

    Expected folder structure:
        static/videos/real_world/<Task>/{MINT,ACT,Pi0,Pi05*}.mp4

    Output schema:
        {
            "task_name": {
                "MINT": ".../MINT.mp4",
                "ACT": ".../ACT.mp4",
                "Pi0": ".../Pi0.mp4",
                "Pi05": ".../Pi05*.mp4"
            }
        }
    """
    base = repo_root / "static" / "videos" / "real_world"
    data: dict[str, dict[str, str]] = {}

    if not base.exists():
        return data

    def method_key(file_stem: str) -> str | None:
        name = file_stem.lower()
        if name == "mint":
            return "MINT"
        if name == "act":
            return "ACT"
        if name == "pi0":
            return "Pi0"
        if name.startswith("pi05"):
            return "Pi05"
        return None

    for task_dir in sorted([p for p in base.iterdir() if _is_visible_dir(p)], key=lambda p: p.name.lower()):
        files = _iter_video_files(task_dir, recursive=False)
        methods: dict[str, str] = {}
        for f in files:
            key = method_key(f.stem)
            if not key:
                continue
            methods[key] = _to_web_path(repo_root, f)
        if methods:
            data[task_dir.name] = methods

    return data


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    main_results = generate_main_results(repo_root)
    generalization = generate_generalization(repo_root)
    one_shot = generate_one_shot(repo_root)
    real_world = generate_real_world(repo_root)

    _write_json(repo_root / "static" / "videos" / "main_results" / "playlist.json", main_results)
    _write_json(repo_root / "static" / "videos" / "generalization" / "playlist.json", generalization)
    _write_json(repo_root / "static" / "videos" / "one_shot" / "playlist.json", one_shot)
    _write_json(repo_root / "static" / "videos" / "real_world" / "playlist.json", real_world)

    print("Wrote:")
    print("- static/videos/main_results/playlist.json")
    print("- static/videos/generalization/playlist.json")
    print("- static/videos/one_shot/playlist.json")
    print("- static/videos/real_world/playlist.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
