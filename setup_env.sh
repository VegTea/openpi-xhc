#!/bin/bash
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENPI_DATA_HOME=/inspire/ssd/project/gjjproject/czxs24230043/openpi_cache
export HF_LEROBOT_HOME=/inspire/qb-ilm/project/gjjproject/public/xl/data/rss_challenge/raw
export HF_HOME=/inspire/ssd/project/gjjproject/czxs24230043/hf_cache

echo "Environment variables set:"
echo "OPENPI_DATA_HOME: $OPENPI_DATA_HOME"
echo "HF_LEROBOT_HOME: $HF_LEROBOT_HOME"
echo "HF_HOME: $HF_HOME"