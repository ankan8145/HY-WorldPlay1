


<!-- Environment activate -->

## Command to run the model 
```bash 
export PATH=/vast/users/ankan.deria/Document/envs/yume/bin:$PATH
export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"

source /vast/users/ankan.deria/Document/envs/yume/bin/activate
```

## miopen
```bash
export MIOPEN_USER_DB_PATH="/vast/users/ankan.deria/Document/.cache/miopen1"
mkdir -p ${MIOPEN_USER_DB_PATH}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
```

## Run the code modified
```bash
export PYTHONPATH=/vast/users/ankan.deria/Document/HY-WorldPlay
cd /vast/users/ankan.deria/Document/HY-WorldPlay
```

<!-- bash scripts/inference -->





# Origami Fine-Tuning Commands

## 1. Precompute normalized manifest and cached features

```bash
cd /vast/users/ankan.deria/Document/HY-WorldPlay
export PYTHONPATH=$(pwd):$PYTHONPATH

python trainer/dataset/origami_step_precompute.py \
  --raw-manifest /vast/users/ankan.deria/DATA/vla_processed/train_manifest.json \
  --output-manifest /vast/users/ankan.deria/DATA/vla_processed/origami_train_manifest_normalized.json \
  --feature-cache-dir /vast/users/ankan.deria/DATA/vla_processed/origami_feature_cache \
  --model-path /vast/users/ankan.deria/Document/YUME_VLA/Pretrain_ckpt/HunyuanVideo-1.5 \
  --device cuda \
  --height 480 \
  --width 832 \
  --target-fps 16 \
  --history-keep 3 \
  --reference-mode latest_past_or_global \
  --shard-strategy balanced_duration \
  --dist-timeout-seconds 7200 \
  --save-every 100




export NNODES=2
export NPROC_PER_NODE=8
export NODE_RANK=1
export MASTER_ADDR=<ip-of-node0>
export MASTER_PORT=29500

torchrun \
  --nnodes $NNODES \
  --nproc_per_node $NPROC_PER_NODE \
  --node_rank $NODE_RANK \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT \
  /vast/users/ankan.deria/Document/HY-WorldPlay/trainer/dataset/origami_step_precompute.py \
  --raw-manifest /vast/users/ankan.deria/DATA/vla_processed/train_manifest.json \
  --output-manifest /vast/users/ankan.deria/DATA/vla_processed/origami_train_manifest_normalized.json \
  --feature-cache-dir /vast/users/ankan.deria/DATA/vla_processed/origami_feature_cache \
  --model-path /vast/users/ankan.deria/Document/YUME_VLA/Pretrain_ckpt/HunyuanVideo-1.5 \
  --device cuda \
  --height 480 \
  --width 832 \
  --target-fps 16 \
  --history-keep 3 \
  --reference-mode latest_past_or_global \
  --shard-strategy balanced_duration \
  --dist-timeout-seconds 7200 \
  --save-every 100 \
  --progress-every 10
```

Notes:
- `history-keep 3` only affects the saved text metadata field `history_last_3`.
- Visual conditioning still uses only one reference clip:
  - `global_clip_path` for the first step
  - otherwise `past_clip_paths[-1]`
- `save-every` writes per-rank shard progress into `origami_train_manifest_normalized.json.shards/`, which makes large runs easier to resume and inspect.
- `shard-strategy balanced_duration` is the new default and uses `duration_sec` to keep distributed work balanced.
- `dist-timeout-seconds 7200` gives slow preprocessing jobs a longer control-plane timeout before distributed syncs fail.
- If `.pt` feature files already exist and `--overwrite` is not set, the script now rebuilds the normalized manifest without loading the encoders again.


Notes:
- Each rank now binds itself to `cuda:${LOCAL_RANK}` automatically, so the same command works under `torchrun`.
- Distributed coordination now uses a long-lived control-plane timeout and duration-balanced shard assignment to avoid rank-0 idling while long clips finish elsewhere.
- Rank 0 owns `negative_prompt.pt` and merges all shard manifests into the final `origami_train_manifest_normalized.json` after every rank finishes.
- The output manifest order is restored to the original raw-manifest order after the merge.
- Restarting the same command is cheap when many `.pt` files already exist, because only missing samples are re-encoded by their assigned rank.


## 2. Launch training

```bash
cd /vast/users/ankan.deria/Document/HY-WorldPlay
export PYTHONPATH=$(pwd):$PYTHONPATH

# Make sure this shell exposes all 8 local GPUs before torchrun starts.
# If your scheduler previously pinned the shell to one GPU, unset first.
unset CUDA_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES

export MODEL_PATH=/vast/users/ankan.deria/Document/YUME_VLA/Pretrain_ckpt/HunyuanVideo-1.5
export NORMALIZED_MANIFEST=/vast/users/ankan.deria/DATA/vla_processed/origami_train_manifest_normalized.json
export LOAD_FROM_DIR=/vast/users/ankan.deria/Document/YUME_VLA/Pretrain_ckpt/HunyuanVideo-1.5/transformer/480p_i2v
export AR_ACTION_LOAD_FROM_DIR=/vast/users/ankan.deria/Document/YUME_VLA/Pretrain_ckpt/HY-WorldPlay/ar_model/diffusion_pytorch_model.safetensors
export OUTPUT_DIR=/vast/users/ankan.deria/Document/HY-WorldPlay/Checkpoint/origami_ft
export LOG_STEPS=10
export CHECKPOINT_STEPS=20
export TRAINING_STATE_CHECKPOINT_STEPS=0
export WEIGHT_ONLY_CHECKPOINT_STEPS=0
export SAVE_LIMIT=3
export WANDB_KEY=""
export WANDB_ENTITY=""
export TRACKER_PROJECT_NAME=origami_steps

export NNODES=1
export NPROC_PER_NODE=8
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29612

bash scripts/training/hyvideo15/run_ar_hunyuan_origami_steps.sh
```

Notes:
- The launcher now prints `Distributed config: ...` and `Visible devices: ...` before `torchrun`; confirm that all 8 local GPUs are visible there.
- If you prefer explicit device selection instead of `unset`, set all three env vars to the full list:
  `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
  `HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
  `ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
- If the launcher reports only one visible device, fix the job allocation or shell environment first; do not start an 8-process run from a 1-GPU-visible shell.
- `LOG_STEPS` controls how often detailed logger lines are emitted.
- `CHECKPOINT_STEPS` controls how often model-weight snapshots are saved.
- `TRAINING_STATE_CHECKPOINT_STEPS` controls resumable distributed training-state checkpoints.
- `WEIGHT_ONLY_CHECKPOINT_STEPS` controls explicit weight-only exports.
- `SAVE_LIMIT` keeps only the most recent checkpoint folders.

## 3. Resume training

### Exact resume from a distributed training-state checkpoint

Use this only if the checkpoint folder contains `distributed_checkpoint/`.

```bash
export RESUME_FROM_CHECKPOINT=/vast/users/ankan.deria/Document/HY-WorldPlay/outputs/origami_ft/checkpoint-100
bash scripts/training/hyvideo15/run_ar_hunyuan_origami_steps.sh
```

### Continue training from saved weights only

Use this when the checkpoint folder only contains `transformer/`.
This restores model weights, but optimizer/scheduler/dataloader state starts fresh.

```bash
export LOAD_FROM_DIR=/vast/users/ankan.deria/Document/HY-WorldPlay/outputs/origami_ft/checkpoint-100/transformer
bash scripts/training/hyvideo15/run_ar_hunyuan_origami_steps.sh
```

## 4. What the training script uses

- `MODEL_PATH`
  `/vast/users/ankan.deria/Document/YUME_VLA/Pretrain_ckpt/HunyuanVideo-1.5`
- `NORMALIZED_MANIFEST`
  `/vast/users/ankan.deria/DATA/vla_processed/origami_train_manifest_normalized.json`
- `LOAD_FROM_DIR`
  `/vast/users/ankan.deria/Document/YUME_VLA/Pretrain_ckpt/HunyuanVideo-1.5/transformer/480p_i2v`
- `AR_ACTION_LOAD_FROM_DIR`
  `/vast/users/ankan.deria/Document/YUME_VLA/Pretrain_ckpt/HY-WorldPlay/ar_model/diffusion_pytorch_model.safetensors`



# Inference and Evaluation Commands

```bash
cd /vast/users/ankan.deria/Document/HY-WorldPlay
export PYTHONPATH=$(pwd):$PYTHONPATH
export HIP_VISIBLE_DEVICES=0

torchrun --nproc_per_node=1 hyvideo/generate.py \
  --prompt "Fold the paper vertically in half." \
  --image_path /vast/users/ankan.deria/Document/HY-WorldPlay/assets/img/airplane_glider_005_clip_first_frame.png \
  --resolution 480p \
  --aspect_ratio 16:9 \
  --video_length 29 \
  --seed 1 \
  --rewrite false \
  --sr false \
  --pose "./assets/pose/static_8_latents.json" \
  --output_path ./outputs/origami_ft_infer_step120_fold_vertical_29f \
  --model_path /vast/users/ankan.deria/Document/YUME_VLA/Pretrain_ckpt/HunyuanVideo-1.5 \
  --action_ckpt /vast/users/ankan.deria/Document/HY-WorldPlay/outputs/origami_ft/checkpoint-120/transformer/diffusion_pytorch_model.safetensors \
  --few_step false \
  --width 832 \
  --height 480 \
  --model_type ar \
  --group_offloading false \
  --use_vae_parallel false
```
