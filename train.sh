source setup_env.sh

export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export UV_OFFLINE=1

CONFIG="pi05_tower-of-hanoi-game_with_val_loss"
EXP_ID=$(date "+%Y%m%d-%H%M%S")  
EXP_NAME="${CONFIG}_with_val_loss"

mkdir -p logs
LOG_FILE="logs/${EXP_NAME}_${EXP_ID}.log"

# uv run scripts/compute_norm_stats.py --config-name "$CONFIG"

uv --offline run scripts/train.py "$CONFIG" --exp-name="$EXP_NAME" --overwrite  2>&1 | tee -a "$LOG_FILE"
