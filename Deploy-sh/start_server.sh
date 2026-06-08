cd /inspire/ssd/project/gjjproject/czxs24230043/openpi-xhc

DENOISING_STEPS=10 # defaut 10

PYTHONPATH=$PWD/third_party/policy_deployment:$PWD \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python third_party/policy_deployment/scripts/launch.py \
  --policy deployment.openpi_yam_policy:OpenPiYamPolicy \
  --policy-kwargs config=pi05_insert-mouse-battery_mixed \
  --policy-kwargs checkpoint_dir=/inspire/qb-ilm/project/gjjproject/czxs24230043/checkpoints/pi05_insert-mouse-battery_mixed/pi05_insert-mouse-battery_mixed_2h200/199999 \
  --policy-kwargs default_prompt="Insert the battery into the mouse." \
  --policy-kwargs num_steps="$DENOISING_STEPS" \
  --host 127.0.0.1 \
  --port 8010 \
  --max-message-size 67108864
  