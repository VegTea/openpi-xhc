cd /inspire/ssd/project/gjjproject/czxs24230043/openpi-xhc

PYTHONPATH=$PWD/third_party/policy_deployment:$PWD \
MUJOCO_GL=osmesa \
uv run python third_party/policy_deployment/sim/check_in_sim.py \
  --mode compare \
  --bundle third_party/policy_deployment/sim/assets/example_slim.pkl \
  --scene third_party/policy_deployment/sim/assets/robot_models/arm/dual_yam/dual_yam_bimanual.xml \
  --host 127.0.0.1 \
  --port 8010 \
  --prompt "Insert the battery into the mouse." \
  --action-horizon 50 \
  --output out/openpi_compare.mp4 \
  --render-camera front \
  --fps 30