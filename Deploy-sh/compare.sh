cd /home/user/Workspace/openpi-xhc

PYTHONPATH=$PWD/third_party/policy_deployment:$PWD \
MUJOCO_GL=egl \
uv run python third_party/policy_deployment/sim/check_in_sim.py \
  --mode compare \
  --bundle out/bundles/water_episode_000000.pkl \
  --scene third_party/policy_deployment/sim/assets/robot_models/arm/dual_yam/dual_yam_bimanual.xml \
  --host 127.0.0.1 \
  --port 8082 \
  --prompt "Place the lid on the cup, align the threads, and twist clockwise to tighten." \
  --action-horizon 50 \
  --output out/openpi_compare.mp4 \
  --render-camera front \
  --fps 30
