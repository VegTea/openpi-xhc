#!/usr/bin/env python3
"""Build expert + success + HIL-suffix mixed LeRobot datasets for challenge tasks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_RECIPES = {
    "insert-mouse-battery": {"expert": 1, "hil_suffix": 2, "success": 4},
    "seal-water-bottle-cap": {"expert": 1, "hil_suffix": 3, "success": 1},
    "tower-of-hanoi-game": {"expert": 1, "hil_suffix": 4, "success": 2},
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_stats(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    return {row["episode_index"]: row["stats"] for row in read_jsonl(path)}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def data_path(root: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunk = episode_index // int(info["chunks_size"])
    rel = info["data_path"].format(episode_chunk=chunk, episode_index=episode_index)
    return root / rel


def video_path(root: Path, info: dict[str, Any], episode_index: int, video_key: str) -> Path:
    chunk = episode_index // int(info["chunks_size"])
    rel = info["video_path"].format(episode_chunk=chunk, video_key=video_key, episode_index=episode_index)
    return root / rel


def patch_indices(df: pd.DataFrame, new_episode_index: int, start_global_index: int) -> pd.DataFrame:
    df = df.copy()
    n_rows = len(df)
    df["episode_index"] = np.full(n_rows, new_episode_index, dtype=np.int64)
    df["index"] = np.arange(start_global_index, start_global_index + n_rows, dtype=np.int64)
    df["task_index"] = np.zeros(n_rows, dtype=np.int64)
    return df


def as_array(series: pd.Series) -> np.ndarray:
    first = series.iloc[0]
    if isinstance(first, np.ndarray):
        return np.stack(series.to_numpy())
    if isinstance(first, (list, tuple)):
        return np.asarray(series.to_list())
    return series.to_numpy()


def stat_for_array(arr: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(arr)
    arr2 = arr[:, None] if arr.ndim == 1 else arr.reshape(arr.shape[0], -1)
    out_shape = arr.shape[1:] or (1,)
    return {
        "min": np.min(arr2, axis=0).reshape(out_shape).tolist(),
        "max": np.max(arr2, axis=0).reshape(out_shape).tolist(),
        "mean": np.mean(arr2, axis=0).reshape(out_shape).tolist(),
        "std": np.std(arr2, axis=0).reshape(out_shape).tolist(),
        "count": [int(arr.shape[0])],
        "q01": np.quantile(arr2, 0.01, axis=0).reshape(out_shape).tolist(),
        "q10": np.quantile(arr2, 0.10, axis=0).reshape(out_shape).tolist(),
        "q50": np.quantile(arr2, 0.50, axis=0).reshape(out_shape).tolist(),
        "q90": np.quantile(arr2, 0.90, axis=0).reshape(out_shape).tolist(),
        "q99": np.quantile(arr2, 0.99, axis=0).reshape(out_shape).tolist(),
    }


def compute_episode_stats(df: pd.DataFrame, original_stats: dict[str, Any] | None, video_keys: list[str]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for key in ("observation.state", "action", "timestamp", "frame_index", "episode_index", "index", "task_index"):
        if key in df:
            stats[key] = stat_for_array(as_array(df[key]))
    if original_stats:
        for key in video_keys:
            if key in original_stats:
                stats[key] = original_stats[key]
    return stats


def commander_state_counts(df: pd.DataFrame) -> dict[str, int]:
    values, counts = np.unique(df["observation.commander_state"].to_numpy(), return_counts=True)
    return {str(value): int(count) for value, count in zip(values, counts, strict=True)}


def build_dataset(
    task: str,
    raw_root: Path,
    output_name: str,
    repeats: dict[str, int],
    force: bool,
) -> Path:
    task_root = raw_root / task
    expert_root = task_root / "expert-data"
    success_hil_root = task_root / "success-and-hil-data"
    target_root = task_root / output_name

    if target_root.exists() and any(target_root.iterdir()):
        if not force:
            raise SystemExit(f"Target exists and is not empty: {target_root}")
        shutil.rmtree(target_root)

    expert_info = json.loads((expert_root / "meta/info.json").read_text())
    success_hil_info = json.loads((success_hil_root / "meta/info.json").read_text())
    if expert_info["features"] != success_hil_info["features"]:
        raise SystemExit(f"{task}: source feature schemas differ; refusing to merge.")

    video_keys = [key for key, value in expert_info["features"].items() if value.get("dtype") == "video"]
    target_info = dict(expert_info)
    target_info["splits"] = {"train": "0:0"}
    target_info["total_episodes"] = 0
    target_info["total_frames"] = 0
    target_info["total_tasks"] = 1

    target_root.mkdir(parents=True, exist_ok=True)
    (target_root / "meta").mkdir(parents=True, exist_ok=True)
    shutil.copy2(expert_root / "meta/tasks.jsonl", target_root / "meta/tasks.jsonl")

    expert_eps = read_jsonl(expert_root / "meta/episodes.jsonl")
    expert_stats = read_stats(expert_root / "meta/episodes_stats.jsonl")
    success_hil_eps = read_jsonl(success_hil_root / "meta/episodes.jsonl")
    success_hil_stats = read_stats(success_hil_root / "meta/episodes_stats.jsonl")

    episodes_out: list[dict[str, Any]] = []
    stats_out: list[dict[str, Any]] = []
    sources_out: list[dict[str, Any]] = []
    frame_counts = {"expert": 0, "success": 0, "hil_suffix": 0}
    episode_counts = {"expert": 0, "success": 0, "hil_suffix": 0}

    new_ep = 0
    global_index = 0

    def append_episode(
        src_root: Path,
        src_info: dict[str, Any],
        src_ep: int,
        df: pd.DataFrame,
        source: str,
        original_stats: dict[str, Any] | None,
        repeat_index: int,
        hil_start_frame: int | None = None,
    ) -> None:
        nonlocal new_ep, global_index
        patched = patch_indices(df, new_ep, global_index)
        dst_parquet = data_path(target_root, target_info, new_ep)
        dst_parquet.parent.mkdir(parents=True, exist_ok=True)
        patched.to_parquet(dst_parquet, index=False)

        for video_key in video_keys:
            link_or_copy(
                video_path(src_root, src_info, src_ep, video_key),
                video_path(target_root, target_info, new_ep, video_key),
            )

        length = len(patched)
        episodes_out.append({"episode_index": new_ep, "tasks": task, "length": length})
        stats_out.append(
            {"episode_index": new_ep, "stats": compute_episode_stats(patched, original_stats, video_keys)}
        )
        sources_out.append(
            {
                "episode_index": new_ep,
                "source": source,
                "source_episode_index": src_ep,
                "source_path": str(src_root),
                "repeat_index": repeat_index,
                "hil_start_frame": hil_start_frame,
                "length": length,
                "commander_state_counts": commander_state_counts(patched),
            }
        )

        frame_counts[source] += length
        episode_counts[source] += 1
        new_ep += 1
        global_index += length

    for repeat_index in range(repeats["expert"]):
        for ep in tqdm(expert_eps, desc=f"{task} expert x{repeat_index + 1}/{repeats['expert']}"):
            src_ep = int(ep["episode_index"])
            df = pd.read_parquet(data_path(expert_root, expert_info, src_ep))
            append_episode(expert_root, expert_info, src_ep, df, "expert", expert_stats.get(src_ep), repeat_index)

    success_items: list[tuple[int, pd.DataFrame, str, int | None]] = []
    for ep in tqdm(success_hil_eps, desc=f"{task} classify success/hil"):
        src_ep = int(ep["episode_index"])
        df = pd.read_parquet(data_path(success_hil_root, success_hil_info, src_ep))
        teleop_indices = np.flatnonzero(df["observation.commander_state"].to_numpy() == "teleop")
        if len(teleop_indices) == 0:
            success_items.append((src_ep, df, "success", None))
        else:
            start = int(teleop_indices[0])
            success_items.append((src_ep, df.iloc[start:].reset_index(drop=True), "hil_suffix", start))

    for source in ("hil_suffix", "success"):
        source_items = [item for item in success_items if item[2] == source]
        for repeat_index in range(repeats[source]):
            for src_ep, df, _, hil_start_frame in tqdm(
                source_items, desc=f"{task} {source} x{repeat_index + 1}/{repeats[source]}"
            ):
                append_episode(
                    success_hil_root,
                    success_hil_info,
                    src_ep,
                    df,
                    source,
                    success_hil_stats.get(src_ep),
                    repeat_index,
                    hil_start_frame,
                )

    target_info["total_episodes"] = len(episodes_out)
    target_info["total_frames"] = global_index
    target_info["splits"] = {"train": f"0:{len(episodes_out)}"}
    (target_root / "meta/info.json").write_text(json.dumps(target_info, indent=4) + "\n")
    write_jsonl(target_root / "meta/episodes.jsonl", episodes_out)
    write_jsonl(target_root / "meta/episodes_stats.jsonl", stats_out)
    write_jsonl(target_root / "meta/sources.jsonl", sources_out)
    write_jsonl(
        target_root / "meta/mix_recipe.jsonl",
        [
            {
                "task": task,
                "repeats": repeats,
                "episode_counts": episode_counts,
                "frame_counts": frame_counts,
                "frame_ratios": {
                    key: value / global_index if global_index else 0.0 for key, value in frame_counts.items()
                },
            }
        ],
    )

    print(f"Created {target_root}")
    print(f"episodes={len(episodes_out)} frames={global_index}")
    print(f"episode_counts={episode_counts}")
    print(f"frame_counts={frame_counts}")
    print(f"frame_ratios={ {key: round(value / global_index, 4) for key, value in frame_counts.items()} }")
    return target_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/inspire/qb-ilm/project/gjjproject/public/xl/data/rss_challenge/raw"),
    )
    parser.add_argument("--tasks", nargs="+", choices=sorted(DEFAULT_RECIPES), default=sorted(DEFAULT_RECIPES))
    parser.add_argument("--output-name", default="expert-success-hil-suffix-mix-data")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for task in args.tasks:
        build_dataset(
            task=task,
            raw_root=args.raw_root.resolve(),
            output_name=args.output_name,
            repeats=DEFAULT_RECIPES[task],
            force=args.force,
        )


if __name__ == "__main__":
    main()
