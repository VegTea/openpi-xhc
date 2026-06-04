# Challenge 混合数据集构建说明

本文档说明 `scripts/build_challenge_mixed_dataset.py` 的用途和用法。

这个脚本用于为三个 RSS challenge 任务生成混合后的 LeRobot 数据集：

- `insert-mouse-battery`
- `seal-water-bottle-cap`
- `tower-of-hanoi-game`

脚本会在每个任务目录下创建新的混合数据集：

```text
<raw-root>/<task>/expert-success-hil-suffix-mix-data
```

脚本不会修改原始的 `expert-data` 或 `success-and-hil-data` 目录。

## 混合了哪些数据

每个任务的输出数据集包含三类数据：

- 全部 `expert-data`
- `success-and-hil-data` 中没有人类接管的成功 rollout
- `success-and-hil-data` 中 HIL episode 的后半段，也就是从第一次 `observation.commander_state == "teleop"` 开始到 episode 结束

`failure-data` 不会放进 BC 训练数据里。

脚本会统一 task prompt：输出数据集会复制 `expert-data/meta/tasks.jsonl`，避免 HIL 数据里的 task 名称形成单独 prompt 分布。

## 当前混合比例

混合比例是通过“重复完整 episode”实现的，而不是训练时改 sampler。

混合前，如果只把 `expert-data`、纯成功 rollout、HIL suffix 各用一次直接拼接，原始帧数和自然比例是：

```text
insert-mouse-battery:
  expert      2,085,520 frames, 78.6%
  HIL suffix    462,527 frames, 17.4%
  success       103,549 frames, 3.9%

seal-water-bottle-cap:
  expert      2,036,650 frames, 76.7%
  HIL suffix    311,588 frames, 11.7%
  success       309,110 frames, 11.6%

tower-of-hanoi-game:
  expert      2,143,353 frames, 82.9%
  HIL suffix    284,579 frames, 11.0%
  success       157,824 frames, 6.1%
```

因为自然拼接后 expert 占比仍然很高，所以当前脚本对 HIL suffix 和 success 做了整数倍过采样。

当前脚本里的配置是：

```python
DEFAULT_RECIPES = {
    "insert-mouse-battery": {"expert": 1, "hil_suffix": 2, "success": 4},
    "seal-water-bottle-cap": {"expert": 1, "hil_suffix": 3, "success": 1},
    "tower-of-hanoi-game": {"expert": 1, "hil_suffix": 4, "success": 2},
}
```

对应生成后的帧比例约为：

```text
insert-mouse-battery:  expert 60.9%, HIL suffix 27.0%, success 12.1%
seal-water-bottle-cap: expert 62.1%, HIL suffix 28.5%, success 9.4%
tower-of-hanoi-game:   expert 59.6%, HIL suffix 31.6%, success 8.8%
```

每次生成后，脚本会把实际统计写到：

```text
<mixed-dataset>/meta/mix_recipe.jsonl
```

每个输出 episode 的来源会写到：

```text
<mixed-dataset>/meta/sources.jsonl
```

## 基本用法

重新生成三个任务的混合数据集：

```bash
uv run scripts/build_challenge_mixed_dataset.py --force
```

只生成某一个任务：

```bash
uv run scripts/build_challenge_mixed_dataset.py \
  --tasks tower-of-hanoi-game \
  --force
```

只生成两个任务：

```bash
uv run scripts/build_challenge_mixed_dataset.py \
  --tasks insert-mouse-battery seal-water-bottle-cap \
  --force
```

指定原始数据根目录：

```bash
uv run scripts/build_challenge_mixed_dataset.py \
  --raw-root /path/to/rss_challenge/raw \
  --force
```

指定输出数据集名称：

```bash
uv run scripts/build_challenge_mixed_dataset.py \
  --output-name expert-success-hil-suffix-mix-v2-data \
  --force
```

注意：`--force` 会删除并重建已有的输出目录。

## 如何调整混合比例

修改 `scripts/build_challenge_mixed_dataset.py` 里的 `DEFAULT_RECIPES`。

例如，如果想提高 tower 任务里 HIL 后半段的权重，可以改成：

```python
"tower-of-hanoi-game": {"expert": 1, "hil_suffix": 5, "success": 2}
```

然后重新生成 tower 数据集：

```bash
uv run scripts/build_challenge_mixed_dataset.py \
  --tasks tower-of-hanoi-game \
  --force
```

生成完成后，查看实际比例：

```bash
cat /inspire/qb-ilm/project/gjjproject/public/xl/data/rss_challenge/raw/tower-of-hanoi-game/expert-success-hil-suffix-mix-data/meta/mix_recipe.jsonl
```

## 对应训练配置

当前已经添加了三个 mixed config：

```text
pi05_insert-mouse-battery_mixed
pi05_seal-water-bottle-cap_mixed
pi05_tower-of-hanoi-game_mixed
```

可以直接用 `train-mgpu.sh` 训练：

```bash
bash train-mgpu.sh pi05_insert-mouse-battery_mixed
bash train-mgpu.sh pi05_seal-water-bottle-cap_mixed
bash train-mgpu.sh pi05_tower-of-hanoi-game_mixed
```

`train-mgpu.sh` 会在训练前自动运行：

```bash
uv run scripts/compute_norm_stats.py --config-name "$CONFIG"
```

所以混合数据构建脚本不会预先计算 norm stats。

## 存储说明

脚本会尽量使用硬链接复用原始视频文件。

因此，单独对 mixed 目录运行 `du -sh` 时，目录看起来可能很大；但视频通常不是重新复制的一份，而是和原始数据共享同一个 inode。

实际新增空间主要来自新写的 parquet 和 meta 文件。

建议用下面的文件查看数据规模和混合比例：

```text
<mixed-dataset>/meta/mix_recipe.jsonl
```
