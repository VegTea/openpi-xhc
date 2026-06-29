cd /home/user/Workspace/openpi-xhc

DENOISING_STEPS=10 # defaut 10

PYTHONPATH=$PWD/third_party/policy_deployment:$PWD \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python third_party/policy_deployment/scripts/launch.py \
  --policy deployment.openpi_yam_policy:OpenPiYamPolicy \
  --policy-kwargs config=pi05_tower-of-hanoi-game_mixed \
  --policy-kwargs checkpoint_dir=/home/user/Workspace/openpi-xhc/checkpoints/hanoi_200k_assets_params \
  --policy-kwargs default_prompt="Place the rings on the middle pillar under Tower of Hanoi constraints, ensuring the smaller ring ends up on top." \
  --policy-kwargs num_steps="$DENOISING_STEPS" \
  --host 127.0.0.1 \
  --port 8084 \
  --max-message-size 67108864