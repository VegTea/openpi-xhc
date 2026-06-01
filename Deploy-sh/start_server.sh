cd /inspire/ssd/project/gjjproject/czxs24230043/openpi-xhc

PYTHONPATH=$PWD/third_party/policy_deployment:$PWD \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python third_party/policy_deployment/scripts/launch.py \
  --policy deployment.openpi_yam_policy:OpenPiYamPolicy \
  --policy-kwargs config=pi05_tower-of-hanoi-game_with_val_loss \
  --policy-kwargs checkpoint_dir=/inspire/qb-ilm/project/gjjproject/czxs24230043/checkpoints/pi05_tower-of-hanoi-game_with_val_loss/pi05_tower-of-hanoi-game_with_val_loss_4h200/160000 \
  --policy-kwargs default_prompt="play the tower of hanoi game" \
  --host 127.0.0.1 \
  --port 8010 \
  --max-message-size 67108864
  