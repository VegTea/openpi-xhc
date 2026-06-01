# OpenPI 策略部署与 MuJoCo Sim 测试

本文档说明如何把当前仓库微调好的 OpenPI dual-YAM 策略通过
`policy_deployment` 的 WebSocket server 部署起来，并在它提供的
MuJoCo sim 中做 replay、policy rollout 和 compare 视频测试。

当前仓库已经包含 `policy_deployment`：

```text
third_party/policy_deployment
```

OpenPI 到 `policy_deployment` 协议的适配器在：

```text
deployment/openpi_yam_policy.py
```

## 1. 环境准备

进入当前仓库：

```bash
cd /inspire/ssd/project/gjjproject/czxs24230043/openpi-xhc
```

安装 Python 依赖：

```bash
uv pip install -r third_party/policy_deployment/requirements.txt
uv pip install mujoco pillow imageio
```

如果需要在无显示器的服务器上渲染 mp4，还需要系统 OpenGL/OSMesa 依赖。
当前机器的 apt 源走内网 nexus，安装时不要走 `127.0.0.1:7897` 代理：

```bash
env -u http_proxy -u https_proxy -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  apt-get update

env -u http_proxy -u https_proxy -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
  xvfb libosmesa6 libosmesa6-dev libegl1 libegl1-mesa-dev \
  libgles2-mesa-dev libgl1-mesa-dri libglx-mesa0 libglfw3 mesa-utils
```

本机已验证 `MUJOCO_GL=osmesa` 可以正常生成 mp4。

## 2. 策略与动作约定

`policy_deployment` sim 使用 14 维 state/action：

```text
[0:6]   left arm joints 1-6
[6]     left gripper, normalized [0, 1]
[7:13]  right arm joints 1-6
[13]    right gripper, normalized [0, 1]
```

当前适配器会把 sim 的输入转换成 OpenPI dual-YAM policy 需要的格式，并返回
sim 所需的 14 维绝对 joint/gripper action。

相机 key：

```text
cam_high
cam_left_wrist
cam_right_wrist
```

适配器也兼容 bundle 里的原始命名：

```text
top_camera
left_camera
right_camera
```

## 3. 启动 Policy Server

下面命令会加载当前训练好的 `40000` step checkpoint，并在 `127.0.0.1:8010`
启动 WebSocket server。

```bash
cd /inspire/ssd/project/gjjproject/czxs24230043/openpi-xhc

PYTHONPATH=$PWD/third_party/policy_deployment:$PWD \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python third_party/policy_deployment/scripts/launch.py \
  --policy deployment.openpi_yam_policy:OpenPiYamPolicy \
  --policy-kwargs config=pi05_tower-of-hanoi-game_with_val_loss \
  --policy-kwargs checkpoint_dir=checkpoints/pi05_tower-of-hanoi-game_with_val_loss/pi05_tower-of-hanoi-game_with_val_loss_2h200/40000 \
  --policy-kwargs default_prompt="play the tower of hanoi game" \
  --host 127.0.0.1 \
  --port 8010 \
  --max-message-size 67108864
```

看到下面日志说明 server 已经启动：

```text
WebSocketPolicyServer listening on ws://127.0.0.1:8010
```

如果要换 checkpoint，只改 `checkpoint_dir` 最后的 step 目录，例如：

```text
.../10000
.../20000
.../30000
.../40000
```

如果要部署到另一台机器访问，把 `--host` 改成 `0.0.0.0`。公网或共享网络上建议加 API key：

```bash
export POLICY_SERVER_API_KEYS="$(openssl rand -hex 32)"
```

然后 sim 端命令加上：

```bash
--api-key "$POLICY_SERVER_API_KEYS"
```

## 4. 检查 Server 是否可用

另开一个 terminal：

```bash
cd /inspire/ssd/project/gjjproject/czxs24230043/openpi-xhc

PYTHONPATH=$PWD/third_party/policy_deployment:$PWD \
uv run python third_party/policy_deployment/scripts/ping.py \
  --host 127.0.0.1 \
  --port 8010
```

成功时会看到：

```text
[PASS] handshake OK
```

再跑一次完整 smoke test：

```bash
PYTHONPATH=$PWD/third_party/policy_deployment:$PWD \
uv run python third_party/policy_deployment/scripts/smoke_test.py \
  --host 127.0.0.1 \
  --port 8010
```

成功时应看到：

```text
[PASS] actions shape=(50, 14) dtype=float32
```

## 5. Sim 模式 1：Replay

Replay 不调用策略，只把 bundle 中记录的 action 在 MuJoCo 中重放。先用它确认
sim 场景、XML、mesh 和 mp4 渲染正常。

```bash
cd /inspire/ssd/project/gjjproject/czxs24230043/openpi-xhc

PYTHONPATH=$PWD/third_party/policy_deployment:$PWD \
MUJOCO_GL=osmesa \
uv run python third_party/policy_deployment/sim/check_in_sim.py \
  --mode replay \
  --bundle third_party/policy_deployment/sim/assets/example_slim.pkl \
  --scene third_party/policy_deployment/sim/assets/robot_models/arm/dual_yam/dual_yam_bimanual.xml \
  --output out/replay.mp4 \
  --render-camera front \
  --fps 30
```

输出：

```text
out/replay.mp4
```

## 6. Sim 模式 2：Compare

Compare 是主要检查方式。它会对每个 sample 做两段 rollout：

1. 从同一个起始状态 replay 记录的 action chunk。
2. 从同一个起始状态调用 policy server，执行预测 action chunk。

输出视频是左右并排：

```text
left:  recorded replay
right: policy rollout
```

命令：

```bash
cd /inspire/ssd/project/gjjproject/czxs24230043/openpi-xhc

PYTHONPATH=$PWD/third_party/policy_deployment:$PWD \
MUJOCO_GL=osmesa \
uv run python third_party/policy_deployment/sim/check_in_sim.py \
  --mode compare \
  --bundle third_party/policy_deployment/sim/assets/example_slim.pkl \
  --scene third_party/policy_deployment/sim/assets/robot_models/arm/dual_yam/dual_yam_bimanual.xml \
  --host 127.0.0.1 \
  --port 8010 \
  --prompt "play the tower of hanoi game" \
  --action-horizon 50 \
  --output out/openpi_compare.mp4 \
  --render-camera front \
  --fps 30
```

输出：

```text
out/openpi_compare.mp4
```

运行时会打印每个 sample 的误差，例如：

```text
sample[ 0] frame=   0
  recorded=50 actions  policy=50 actions
  diff over 50 steps: L2=4.6143  max|.|=0.7973
```

这些指标是 policy action chunk 与 recorded action chunk 的差异，只是 sanity
check，不等同于最终任务成功率。

## 7. Sim 模式 3：Policy Rollout

Policy 模式会持续用 bundle 的图像流和 sim 当前 state 调用策略，让策略驱动 sim：

```bash
cd /inspire/ssd/project/gjjproject/czxs24230043/openpi-xhc

PYTHONPATH=$PWD/third_party/policy_deployment:$PWD \
MUJOCO_GL=osmesa \
uv run python third_party/policy_deployment/sim/check_in_sim.py \
  --mode policy \
  --bundle third_party/policy_deployment/sim/assets/example_slim.pkl \
  --scene third_party/policy_deployment/sim/assets/robot_models/arm/dual_yam/dual_yam_bimanual.xml \
  --host 127.0.0.1 \
  --port 8010 \
  --prompt "play the tower of hanoi game" \
  --action-horizon 50 \
  --output out/openpi_policy.mp4 \
  --render-camera front \
  --fps 30
```

## 8. 无渲染 Headless Compare

如果某台机器不能配置 OpenGL，仍然可以跑不生成视频的 dynamics + policy 检查：

```bash
cd /inspire/ssd/project/gjjproject/czxs24230043/openpi-xhc

PYTHONPATH=$PWD/third_party/policy_deployment:$PWD \
uv run python deployment/headless_policy_deployment_compare.py \
  --host 127.0.0.1 \
  --port 8010 \
  --prompt "play the tower of hanoi game" \
  --num-samples 1
```

它不会创建 MuJoCo renderer，但会：

```text
load scene
load bundle sample
query policy server
execute action chunk in MuJoCo dynamics
print L2 / max error metrics
```

## 9. 从 Expert Data 导出自己的 Bundle

`policy_deployment` 的 sim 使用 `.pkl` bundle 作为输入。当前仓库提供了导出脚本：

```text
scripts/export_policy_deployment_bundle.py
```

它会从 LeRobot 格式的 `expert-data` 里读取：

```text
data/chunk-xxx/episode_xxxxxx.parquet
videos/chunk-xxx/observation.images.cam_high/episode_xxxxxx.mp4
videos/chunk-xxx/observation.images.cam_left_wrist/episode_xxxxxx.mp4
videos/chunk-xxx/observation.images.cam_right_wrist/episode_xxxxxx.mp4
```

并生成一个同时支持 `replay`、`policy`、`compare` 的 pkl，包含：

```text
actions
joint_positions
timestamps_ns
images
samples
meta
```

导出一条完整轨迹，例如 episode 0：

```bash
cd /inspire/ssd/project/gjjproject/czxs24230043/openpi-xhc

PYTHONPATH=$PWD \
uv run python scripts/export_policy_deployment_bundle.py \
  --dataset-root /inspire/qb-ilm/project/gjjproject/public/xl/data/rss_challenge/raw/tower-of-hanoi-game/expert-data \
  --episode-index 0 \
  --output out/bundles/tower_episode_000000.pkl \
  --num-samples 10 \
  --action-horizon 50
```

如果只是快速测试导出流程，可以限制帧数：

```bash
PYTHONPATH=$PWD \
uv run python scripts/export_policy_deployment_bundle.py \
  --dataset-root /inspire/qb-ilm/project/gjjproject/public/xl/data/rss_challenge/raw/tower-of-hanoi-game/expert-data \
  --episode-index 0 \
  --output out/bundles/tower_episode_000000_120f.pkl \
  --max-frames 120 \
  --num-samples 3 \
  --action-horizon 50
```

导出后可以直接 replay：

```bash
PYTHONPATH=$PWD/third_party/policy_deployment:$PWD \
MUJOCO_GL=osmesa \
uv run python third_party/policy_deployment/sim/check_in_sim.py \
  --mode replay \
  --bundle out/bundles/tower_episode_000000.pkl \
  --scene third_party/policy_deployment/sim/assets/robot_models/arm/dual_yam/dual_yam_bimanual.xml \
  --output out/bundles/tower_episode_000000_replay.mp4 \
  --render-camera front \
  --fps 30
```

或者用同一个 bundle 做 policy rollout：

```bash
PYTHONPATH=$PWD/third_party/policy_deployment:$PWD \
MUJOCO_GL=osmesa \
uv run python third_party/policy_deployment/sim/check_in_sim.py \
  --mode policy \
  --bundle out/bundles/tower_episode_000000.pkl \
  --scene third_party/policy_deployment/sim/assets/robot_models/arm/dual_yam/dual_yam_bimanual.xml \
  --host 127.0.0.1 \
  --port 8010 \
  --prompt "play the tower of hanoi game" \
  --action-horizon 50 \
  --output out/bundles/tower_episode_000000_policy.mp4 \
  --render-camera front \
  --fps 30
```

注意：完整 episode 会把三路视频逐帧编码进 pkl，文件可能较大。`--max-frames`
适合调试，正式测试时去掉它。

## 10. CPU 与 4090/GPU 机器的差别

当前 CPU 节点可以完整跑通部署、推理、sim 和 mp4 渲染，但 `pi05` 推理较慢。
实测 CPU 单次 WebSocket inference 是秒级到十秒级。

如果换到 RTX 4090，并且 JAX CUDA 环境正确，主要会加速 policy inference。
MuJoCo dynamics 本身主要还是 CPU，mp4 渲染如果能用 EGL/GPU backend 也会更快。

检查 JAX 是否使用 GPU：

```bash
PYTHONPATH=$PWD uv run python - <<'PY'
import jax
print(jax.default_backend())
print(jax.devices())
PY
```

如果看到 `cpu`，说明策略仍然在 CPU 上跑；如果看到 GPU/CUDA device，才是真正用到了 4090。

## 11. 常见问题

### `apt-get update` 访问 nexus 报 502

当前 shell 可能开了代理：

```text
http_proxy=http://127.0.0.1:7897
```

apt 访问内网 nexus 时需要去掉这些代理变量：

```bash
env -u http_proxy -u https_proxy -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY apt-get update
```

### `Renderer` / OpenGL / EGL / OSMesa 报错

优先使用：

```bash
MUJOCO_GL=osmesa
```

并确认安装了：

```text
libosmesa6
libosmesa6-dev
libegl1
libglfw3
xvfb
```

### `joint has range but not limited`

这是 MuJoCo XML 兼容问题。当前 vendored 的 XML 已经加了：

```xml
<compiler angle="radian" autolimits="true"/>
```

如果换了新的 `policy_deployment` 副本，也需要保留这个字段。

### Smoke test 返回 shape 不是 `(50, 14)`

检查 server 加载的 config 和 checkpoint 是否匹配：

```text
config=pi05_tower-of-hanoi-game_with_val_loss
checkpoint_dir=.../pi05_tower-of-hanoi-game_with_val_loss_2h200/40000
```

### 端口占用

换一个端口即可，例如 `8011`：

```bash
--port 8011
```

sim、ping、smoke test 里的 `--port` 也要同步改成 `8011`。

## 12. 已验证输出

当前机器已验证生成：

```text
out/replay.mp4
out/openpi_compare.mp4
```

验证过的链路：

```text
OpenPI checkpoint -> OpenPiYamPolicy adapter -> policy_deployment WebSocket server
-> policy_deployment MuJoCo sim -> mp4 output
```
