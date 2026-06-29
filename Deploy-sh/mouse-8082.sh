cd /home/user/Workspace/openpi-xhc

XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run scripts/serve_policy.py \
  --port=8082 \
  policy:checkpoint \
  --policy.config=pi05_insert-mouse-battery_mixed \
  --policy.dir=checkpoints/mouse_80k_assets_params