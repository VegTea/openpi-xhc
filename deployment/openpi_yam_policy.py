"""Adapter from policy_deployment's WebSocket protocol to an OpenPI YAM policy.

Launch with policy_deployment's server, for example:

    PYTHONPATH=/path/to/policy_deployment:/path/to/openpi-xhc \
        uv run python /path/to/policy_deployment/scripts/launch.py \
        --policy deployment.openpi_yam_policy:OpenPiYamPolicy \
        --policy-kwargs config=pi05_tower-of-hanoi-game_with_val_loss \
        --policy-kwargs checkpoint_dir=checkpoints/pi05_tower-of-hanoi-game_with_val_loss/pi05_tower-of-hanoi-game_with_val_loss_2h200/40000
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

try:
    from server.policy import BasePolicy
    from server.schema import InferenceRequest
    from server.schema import InferenceResponse
    from server.schema import ServerMetadata
except ModuleNotFoundError as exc:  # pragma: no cover - only hit when launched without policy_deployment.
    raise ModuleNotFoundError(
        "deployment.openpi_yam_policy must be launched with policy_deployment on PYTHONPATH. "
        "Example: PYTHONPATH=/path/to/policy_deployment:/path/to/openpi-xhc python "
        "/path/to/policy_deployment/scripts/launch.py --policy "
        "deployment.openpi_yam_policy:OpenPiYamPolicy ..."
    ) from exc

from openpi.policies import policy_config
from openpi.training import config as openpi_config

_CAMERA_ALIASES = {
    "cam_high": ("cam_high", "top_camera", "base_0_rgb"),
    "cam_left_wrist": ("cam_left_wrist", "left_camera", "left_wrist_0_rgb"),
    "cam_right_wrist": ("cam_right_wrist", "right_camera", "right_wrist_0_rgb"),
}


class OpenPiYamPolicy(BasePolicy):
    """Serve a fine-tuned OpenPI dual-YAM checkpoint via policy_deployment."""

    def __init__(
        self,
        config: str,
        checkpoint_dir: str,
        *,
        default_prompt: str = "",
        policy_name: str | None = None,
        pytorch_device: str | None = None,
    ) -> None:
        self._config_name = config
        self._checkpoint_dir = checkpoint_dir
        self._default_prompt = default_prompt
        self._train_config = openpi_config.get_config(config)
        self._policy = policy_config.create_trained_policy(
            self._train_config,
            checkpoint_dir,
            default_prompt=default_prompt,
            pytorch_device=pytorch_device,
        )
        self._policy_name = policy_name or config

    @property
    def metadata(self) -> ServerMetadata:
        model_config = self._train_config.model
        return {
            "protocol_version": "1.0",
            "policy_name": self._policy_name,
            "control_mode": "joints",
            "action_horizon": int(model_config.action_horizon),
            "action_dim": 14,
            "state_dim": 14,
            "image_keys": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
            "image_shape": [3, 224, 224],
            "expects_prompt": True,
            "extra": {
                "openpi_config": self._config_name,
                "checkpoint_dir": self._checkpoint_dir,
                "action_layout": "[L_j1..6, L_gripper, R_j1..6, R_gripper]",
                "gripper_range": "[0, 1]",
            },
        }

    def infer(self, obs: InferenceRequest) -> InferenceResponse:
        images = self._normalize_images(obs.get("images", {}))
        state = np.asarray(obs["state"], dtype=np.float32)
        if state.shape != (14,):
            raise ValueError(f"Expected 14-D state, got shape {state.shape}")

        prompt = obs.get("prompt") or self._default_prompt
        openpi_obs = {"images": images, "state": state, "prompt": prompt}
        result = self._policy.infer(openpi_obs)
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 14:
            raise ValueError(f"OpenPI policy returned actions with shape {actions.shape}, expected (H, 14)")

        response: InferenceResponse = {"actions": np.ascontiguousarray(actions)}
        if "policy_timing" in result:
            response["extra"] = {"openpi_policy_timing": result["policy_timing"]}
        if request_id := obs.get("request_id"):
            response["request_id"] = request_id
        return response

    def _normalize_images(self, images: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        normalized: dict[str, np.ndarray] = {}
        for canonical, aliases in _CAMERA_ALIASES.items():
            image = next((images[name] for name in aliases if name in images), None)
            if image is None:
                raise KeyError(f"Missing image for {canonical}; accepted keys are {aliases}")
            arr = np.asarray(image)
            if arr.ndim != 3:
                raise ValueError(f"Image {canonical} must be 3-D CHW or HWC, got shape {arr.shape}")
            if arr.shape[0] != 3 and arr.shape[-1] == 3:
                arr = np.transpose(arr, (2, 0, 1))
            if arr.shape[0] != 3:
                raise ValueError(f"Image {canonical} must have 3 channels, got shape {arr.shape}")
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.clip(arr, 0.0, 1.0) * 255.0
            normalized[canonical] = np.ascontiguousarray(arr.astype(np.uint8, copy=False))
        return normalized
