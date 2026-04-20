export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${MODEL_PATH:-}"   # Path to pretrained HunyuanVideo-1.5 model
NORMALIZED_MANIFEST="${NORMALIZED_MANIFEST:-}"   # Path to normalized origami manifest produced by origami_step_precompute.py
LOAD_FROM_DIR="${LOAD_FROM_DIR:-}"   # Path to pretrained transformer directory
AR_ACTION_LOAD_FROM_DIR="${AR_ACTION_LOAD_FROM_DIR:-}"   # Optional: path to pretrained AR action model directory
OUTPUT_DIR="${OUTPUT_DIR:-}"   # Path to output directory
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
WANDB_KEY="${WANDB_KEY:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
TRACKER_PROJECT_NAME="${TRACKER_PROJECT_NAME:-origami_steps}"
LOG_STEPS="${LOG_STEPS:-10}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-500}"
TRAINING_STATE_CHECKPOINT_STEPS="${TRAINING_STATE_CHECKPOINT_STEPS:-0}"
WEIGHT_ONLY_CHECKPOINT_STEPS="${WEIGHT_ONLY_CHECKPOINT_STEPS:-0}"
SAVE_LIMIT="${SAVE_LIMIT:-3}"
NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29612}"

NUM_GPUS=$NPROC_PER_NODE

count_visible_devices() {
  local value="$1"
  python3 - "$value" <<'PY'
import sys
value = sys.argv[1]
items = [item.strip() for item in value.split(",") if item.strip()]
print(len(items))
PY
}

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && -z "${HIP_VISIBLE_DEVICES:-}" && -z "${ROCR_VISIBLE_DEVICES:-}" ]]; then
  DEFAULT_VISIBLE_DEVICES=$(seq -s, 0 $((NPROC_PER_NODE - 1)))
  export CUDA_VISIBLE_DEVICES="$DEFAULT_VISIBLE_DEVICES"
  export HIP_VISIBLE_DEVICES="$DEFAULT_VISIBLE_DEVICES"
  export ROCR_VISIBLE_DEVICES="$DEFAULT_VISIBLE_DEVICES"
else
  ACTIVE_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-${ROCR_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$ACTIVE_VISIBLE_DEVICES}"
  export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-$ACTIVE_VISIBLE_DEVICES}"
  export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-$ACTIVE_VISIBLE_DEVICES}"
fi

VISIBLE_DEVICE_COUNT=$(count_visible_devices "${HIP_VISIBLE_DEVICES:-${ROCR_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}}")
if (( VISIBLE_DEVICE_COUNT < NPROC_PER_NODE )); then
  echo "Visible device mismatch: NPROC_PER_NODE=$NPROC_PER_NODE but only $VISIBLE_DEVICE_COUNT visible device(s)." >&2
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" >&2
  echo "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-}" >&2
  echo "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-}" >&2
  echo "Unset the single-device env vars or set them to a full list such as 0,1,2,3,4,5,6,7 before launching." >&2
  exit 1
fi

if [[ -z "$MODEL_PATH" ]]; then
  echo "MODEL_PATH is required." >&2
  exit 1
fi

if [[ -z "$NORMALIZED_MANIFEST" ]]; then
  echo "NORMALIZED_MANIFEST is required." >&2
  exit 1
fi

if [[ -z "$LOAD_FROM_DIR" ]]; then
  echo "LOAD_FROM_DIR is required." >&2
  exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  echo "OUTPUT_DIR is required." >&2
  exit 1
fi

training_args=(
  --data-path $NORMALIZED_MANIFEST
  --json_path $NORMALIZED_MANIFEST
  --dataset_type origami_steps
  --causal
  --action
  --i2v_rate 1.0
  --train_time_shift 3.0
  --window_frames 24
  --max_train_steps 200000
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 1
  --num_latent_t 9
  --num_height 480
  --num_width 832
  --num_frames 77
  --enable_gradient_checkpointing_type "full"
  --seed 3208
  --log-steps $LOG_STEPS
  --weighting_scheme "logit_normal"
  --logit_mean 0.0
  --logit_std 1.0
  --output_dir $OUTPUT_DIR
)

if [[ -n "$WANDB_KEY" ]]; then
  training_args+=(--wandb_key "$WANDB_KEY")
fi

if [[ -n "$WANDB_ENTITY" ]]; then
  training_args+=(--wandb_entity "$WANDB_ENTITY")
fi

if [[ -n "$TRACKER_PROJECT_NAME" ]]; then
  training_args+=(--tracker_project_name "$TRACKER_PROJECT_NAME")
fi

if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  training_args+=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
fi

parallel_args=(
  --num_gpus $((NNODES * NPROC_PER_NODE))
  --sp_size 4
  --tp_size 1
  --hsdp_replicate_dim 1
  --hsdp_shard_dim $((NNODES * NPROC_PER_NODE))
)

model_args=(
  --cls_name "HunyuanTransformer3DARActionModel"
  --load_from_dir $LOAD_FROM_DIR
  --model_path $MODEL_PATH
  --pretrained_model_name_or_path $MODEL_PATH
)

if [[ -n "$AR_ACTION_LOAD_FROM_DIR" ]]; then
  model_args+=(--ar_action_load_from_dir "$AR_ACTION_LOAD_FROM_DIR")
fi

dataset_args=(
  --dataloader_num_workers 1
)

optimizer_args=(
  --learning_rate 1e-5
  --mixed_precision "bf16"
  --checkpointing-steps $CHECKPOINT_STEPS
  --training-state-checkpointing-steps $TRAINING_STATE_CHECKPOINT_STEPS
  --weight-only-checkpointing-steps $WEIGHT_ONLY_CHECKPOINT_STEPS
  --weight_decay 1e-4
  --max_grad_norm 1.0
)

miscellaneous_args=(
  --inference_mode False
  --checkpoints-total-limit $SAVE_LIMIT
  --training_cfg_rate 0.1
  --multi_phased_distill_schedule "4000-1"
  --not_apply_cfg_solver
  --dit_precision "fp32"
  --num_euler_timesteps 50
  --ema_start_step 0
)

echo "Distributed config: NNODES=$NNODES NPROC_PER_NODE=$NPROC_PER_NODE NODE_RANK=$NODE_RANK MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "Visible devices: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-} ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-}"

torchrun \
        --nnodes=$NNODES \
        --nproc_per_node=$NPROC_PER_NODE \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        trainer/training/ar_hunyuan_w_mem_training_pipeline.py \
        "${parallel_args[@]}" \
        "${model_args[@]}" \
        "${dataset_args[@]}" \
        "${training_args[@]}" \
        "${optimizer_args[@]}" \
        "${miscellaneous_args[@]}"
