"""Run a policy_deployment MuJoCo compare without opening a viewer or renderer.

This is useful on CPU/headless machines where MuJoCo dynamics works but
OpenGL rendering is unavailable.
"""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
import sys
import tempfile

import numpy as np


class NullViewer:
    def __init__(self) -> None:
        self.steps = 0

    def is_running(self) -> bool:
        return True

    def sync(self) -> None:
        self.steps += 1


@contextlib.contextmanager
def _scene_with_autolimits(scene: Path):
    xml = scene.read_text()
    if "autolimits=" in xml:
        yield scene
        return
    if '<compiler angle="radian"/>' not in xml:
        yield scene
        return

    patched = xml.replace('<compiler angle="radian"/>', '<compiler angle="radian" autolimits="true"/>', 1)
    with tempfile.NamedTemporaryFile("w", suffix=".xml", dir=scene.parent, delete=False) as f:
        f.write(patched)
        patched_path = Path(f.name)
    try:
        yield patched_path
    finally:
        patched_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-deployment-root", type=Path, default=Path("third_party/policy_deployment"))
    parser.add_argument("--bundle", type=Path, default=Path("sim/assets/example_slim.pkl"))
    parser.add_argument(
        "--scene",
        type=Path,
        default=Path("sim/assets/robot_models/arm/dual_yam/dual_yam_bimanual.xml"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--ctrl-hz", type=float, default=60.0)
    args = parser.parse_args()

    root = args.policy_deployment_root.resolve()
    sys.path.insert(0, str(root))

    from sim import check_in_sim as sim

    from scripts._client import WebsocketPolicyClient

    bundle_path = args.bundle if args.bundle.is_absolute() else root / args.bundle
    scene_path = args.scene if args.scene.is_absolute() else root / args.scene

    bundle = sim.load_bundle(bundle_path)
    samples = bundle.get("samples")
    if not samples:
        raise ValueError(f"{bundle_path} does not contain slim-bundle samples")

    with _scene_with_autolimits(scene_path) as load_scene:
        model, data = sim.build_model(load_scene)
    qpos_addrs, ctrl_idxs = sim._resolve_indices(model)  # noqa: SLF001

    client = WebsocketPolicyClient(host=args.host, port=args.port, api_key=args.api_key)
    print(f"metadata: {client.metadata}")

    count = min(args.num_samples, len(samples))
    for sample_idx, sample in enumerate(samples[:count]):
        recorded = np.asarray(sample["action_chunk"])
        sim._snap_to_state14(model, data, qpos_addrs, ctrl_idxs, sample["state"])  # noqa: SLF001
        policy_input = {
            "images": {
                "cam_high": sim.jpeg_to_chw(sample["images"]["top_camera"]),
                "cam_left_wrist": sim.jpeg_to_chw(sample["images"]["left_camera"]),
                "cam_right_wrist": sim.jpeg_to_chw(sample["images"]["right_camera"]),
            },
            "state": np.ascontiguousarray(np.asarray(sample["state"], dtype=np.float64)),
            "prompt": args.prompt,
        }
        chunk = np.asarray(client.infer(policy_input)["actions"])
        viewer = NullViewer()
        sim._run_chunk(  # noqa: SLF001
            model,
            data,
            viewer,
            ctrl_idxs,
            chunk,
            ctrl_hz=args.ctrl_hz,
            realtime=False,
            speed=1.0,
        )

        horizon = min(len(recorded), len(chunk))
        diff = chunk[:horizon] - recorded[:horizon]
        print(
            f"sample[{sample_idx:2d}] steps={viewer.steps} shape={chunk.shape} "
            f"L2={float(np.linalg.norm(diff)):.4f} "
            f"L2/sqrt(N*14)={float(np.sqrt(np.mean(diff * diff))):.4f} "
            f"max|.|={float(np.max(np.abs(diff))):.4f}"
        )
        print("            max|.| per dim: " + " ".join(f"{v:.3f}" for v in np.max(np.abs(diff), axis=0)))

    client.close()


if __name__ == "__main__":
    main()
