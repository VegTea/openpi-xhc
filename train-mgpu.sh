source setup_env.sh

export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}

export CUDA_VISIBLE_DEVICES=0,1,2,3

CONFIG="pi05_tower-of-hanoi-game_with_val_loss"
EXP_ID=$(date "+%Y%m%d-%H%M%S")  
EXP_NAME="${CONFIG}_4h200"

mkdir -p logs
LOG_FILE="logs/${EXP_NAME}_${EXP_ID}.log"

uv run scripts/compute_norm_stats.py --config-name "$CONFIG"

wandb offline
uv run scripts/train.py "$CONFIG" \
    --exp-name="$EXP_NAME" \
    --fsdp-devices=4 \
    --checkpoint-base-dir="/inspire/qb-ilm/project/gjjproject/czxs24230043/checkpoints" \
    --overwrite  2>&1 | tee -a "$LOG_FILE"
