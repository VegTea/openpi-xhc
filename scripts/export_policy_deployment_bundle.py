"""Export one LeRobot episode into a policy_deployment sim bundle.

The output pkl supports:
  - --mode replay: uses actions / joint_positions
  - --mode policy: uses top-level images + timestamps
  - --mode compare: uses slim-style samples
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import pickle

import cv2
import imageio.v3 as iio
import numpy as np
from PIL import Image
import polars as pl

CAMERA_MAP = {
    "top_camera": "observation.images.cam_high",
    "left_camera": "observation.images.cam_left_wrist",
    "right_camera": "observation.images.cam_right_wrist",
}


def _episode_chunk(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


def _load_info(dataset_root: Path) -> dict:
    with (dataset_root / "meta/info.json").open() as f:
        return json.load(f)


def _read_episode_table(dataset_root: Path, episode_index: int, info: dict) -> pl.DataFrame:
    chunk = _episode_chunk(episode_index, int(info.get("chunks_size", 1000)))
    parquet_path = dataset_root / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)
    return pl.read_parquet(parquet_path)


def _encode_rgb_to_jpeg_bytes(rgb: np.ndarray, quality: int) -> bytes:
    image = Image.fromarray(rgb)
    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG", quality=quality)
        return buffer.getvalue()


def _read_video_as_jpegs(video_path: Path, *, expected_frames: int, quality: int) -> list[bytes]:
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    try:
        frames = []
        for rgb in iio.imiter(video_path):
            frames.append(_encode_rgb_to_jpeg_bytes(np.asarray(rgb), quality))
            if len(frames) >= expected_frames:
                break
        if len(frames) >= expected_frames:
            return frames
    except Exception as exc:
        print(f"imageio failed for {video_path}: {exc}; falling back to OpenCV")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames: list[bytes] = []
    try:
        while len(frames) < expected_frames:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(_encode_rgb_to_jpeg_bytes(rgb, quality))
    finally:
        cap.release()

    if len(frames) < expected_frames:
        raise RuntimeError(f"{video_path} has {len(frames)} frames, expected at least {expected_frames}")
    return frames


def _read_images(dataset_root: Path, episode_index: int, info: dict, *, expected_frames: int, quality: int) -> dict:
    chunk = _episode_chunk(episode_index, int(info.get("chunks_size", 1000)))
    images = {}
    for bundle_key, video_key in CAMERA_MAP.items():
        video_path = dataset_root / f"videos/chunk-{chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        images[bundle_key] = _read_video_as_jpegs(video_path, expected_frames=expected_frames, quality=quality)
    return images


def _make_samples(
    *,
    states: np.ndarray,
    actions: np.ndarray,
    images: dict,
    action_horizon: int,
    num_samples: int,
) -> list[dict]:
    max_start = len(actions) - action_horizon
    if max_start < 0:
        return []
    frame_indices = [0] if num_samples <= 1 else np.linspace(0, max_start, num_samples, dtype=np.int64).tolist()

    return [
        {
            "frame_idx": int(frame_idx),
            "state": states[frame_idx].astype(np.float64),
            "action_chunk": actions[frame_idx : frame_idx + action_horizon].astype(np.float64),
            "images": {key: frames[frame_idx] for key, frames in images.items()},
        }
        for frame_idx in frame_indices
    ]


def export_bundle(args: argparse.Namespace) -> None:
    dataset_root = args.dataset_root.resolve()
    info = _load_info(dataset_root)
    table = _read_episode_table(dataset_root, args.episode_index, info)

    if args.max_frames is not None:
        table = table.head(args.max_frames)

    states = np.asarray(table["observation.state"].to_list(), dtype=np.float32)
    actions = np.asarray(table["action"].to_list(), dtype=np.float32)
    timestamps_s = np.asarray(table["timestamp"].to_list(), dtype=np.float64)
    timestamps_ns = (timestamps_s * 1_000_000_000).astype(np.int64)

    images = _read_images(
        dataset_root,
        args.episode_index,
        info,
        expected_frames=len(table),
        quality=args.jpeg_quality,
    )
    samples = _make_samples(
        states=states,
        actions=actions,
        images=images,
        action_horizon=args.action_horizon,
        num_samples=args.num_samples,
    )

    bundle = {
        "meta": {
            "source": "lerobot",
            "dataset_root": str(dataset_root),
            "episode_index": args.episode_index,
            "n_frames": len(table),
            "duration_s": float(timestamps_s[-1] - timestamps_s[0]) if len(timestamps_s) else 0.0,
            "rate_hz": int(info.get("fps", 60)),
            "cameras": CAMERA_MAP,
            "action_layout": "[L_j1..6, L_gripper, R_j1..6, R_gripper]",
            "chunk_horizon": args.action_horizon,
            "n_samples": len(samples),
        },
        "actions": actions,
        "joint_positions": states,
        "timestamps_ns": timestamps_ns,
        "images": images,
        "samples": samples,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Wrote {args.output}")
    print(f"  episode={args.episode_index} frames={len(table)} samples={len(samples)}")
    print(f"  actions={actions.shape} states={states.shape}")
    print("  images=" + ", ".join(f"{k}:{len(v)}" for k, v in images.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()
    export_bundle(args)


if __name__ == "__main__":
    main()
