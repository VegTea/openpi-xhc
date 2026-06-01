cd /inspire/ssd/project/gjjproject/czxs24230043/openpi-xhc

PYTHONPATH=$PWD/third_party/policy_deployment:$PWD \
MUJOCO_GL=osmesa \
uv run python third_party/policy_deployment/sim/check_in_sim.py \
  --mode replay \
  --bundle out/bundles/tower_episode_000000.pkl \
  --scene third_party/policy_deployment/sim/assets/robot_models/arm/dual_yam/dual_yam_bimanual.xml \
  --output out/replay.mp4 \
  --render-camera front \
  --fps 30