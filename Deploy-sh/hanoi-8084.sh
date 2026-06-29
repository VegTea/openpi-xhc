cd /home/user/Workspace/openpi-xhc

XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run scripts/serve_policy.py \
  --port=8084 \
  policy:checkpoint \
  --policy.config=pi05_tower-of-hanoi-game_mixed \
  --policy.dir=checkpoints/hanoi_200k_assets_params