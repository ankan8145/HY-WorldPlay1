# Training Model-Loading Analysis

## Script under analysis

`scripts/training/hyvideo15/run_ar_hunyuan_action_mem.sh`

---

## Short answer

| Component | Loaded during training? | How its output appears in training |
|---|---|---|
| **Qwen2.5-VL-7B-Instruct** (LLM text encoder) | ❌ **No** | `prompt_embeds` field read from precomputed `.pt` file |
| **google/byt5-small** (ByT5 encoder) | ❌ **No** | `byt5_text_states` / `byt5_text_mask` fields read from precomputed `.pt` file |
| **SigLIP / FLUX vision encoder** | ❌ **No** | `vision_states` field read from precomputed `.pt` file |
| **VAE** | ❌ **No** | `latent` field read from precomputed `.pt` file |
| **Transformer DiT** (`HunyuanTransformer3DARActionModel`) | ✅ **Yes** | Loaded via `TransformerLoader` / FSDP loader |

**Latent strategy: fully offline / precomputed.** The training loop never calls VAE, Qwen, ByT5, or SigLIP at runtime. All encoder outputs were computed in a separate preprocessing step and saved to `.pt` files on disk.

---

## Step-by-step evidence

### 1 – Shell script entry-point

**File:** `scripts/training/hyvideo15/run_ar_hunyuan_action_mem.sh`, lines 96–107

```bash
torchrun \
    --master_port=$MASTER_PORT \
    --nproc_per_node=$NUM_GPUS \
    --nnodes 1 \
    trainer/training/ar_hunyuan_w_mem_training_pipeline.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    ...
```

Key model arguments passed (lines 51–57):

```bash
model_args=(
  --cls_name "HunyuanTransformer3DARActionModel"
  --load_from_dir          # pretrained transformer directory
  --ar_action_load_from_dir  # AR action model directory
  --model_path $MODEL_PATH
  --pretrained_model_name_or_path $MODEL_PATH
)
```

`MODEL_PATH` is only used to locate the transformer checkpoint; it is **not** passed to any text/image encoder loader.

---

### 2 – Training entrypoint: only the transformer is required

**File:** `trainer/training/ar_hunyuan_w_mem_training_pipeline.py`, line 21

```python
class HunyuanTrainingPipeline(TrainingPipeline):
    _required_config_modules = ["transformer"]   # line 21 — only the DiT
```

---

### 3 – Pipeline `__init__` dispatches to `load_hunyuan_modules` (transformer only)

**File:** `trainer/pipelines/composed_pipeline_base.py`, lines 76–77

```python
if "HunyuanTransformer" in trainer_args.cls_name:          # line 76
    self.modules = self.load_hunyuan_modules(trainer_args)  # line 77
```

`load_hunyuan_modules` (lines 228–254 of the same file) loads **only** the `"transformer"` key:

```python
def load_hunyuan_modules(self, trainer_args):   # line 228
    modules = {}
    module_name = "transformer"                 # line 238
    component_model_path = trainer_args.load_from_dir
    module = PipelineComponentLoader.load_module(
        module_name=module_name,
        component_model_path=component_model_path,
        transformers_or_diffusers="diffusers",
        trainer_args=trainer_args,
    )                                           # line 242-247
    modules[module_name] = module               # line 253
    return modules                              # line 254
```

No VAE, no text encoder, no image encoder is loaded here.

---

### 4 – Transformer (DiT) load site

**File:** `trainer/models/loader/component_loader.py`

`TransformerLoader.load()` (line 397) is called for the DiT and internally calls
`maybe_load_fsdp_model()` (line 434) with:

```python
load_from_dir=trainer_args.load_from_dir,
ar_action_load_from_dir=trainer_args.ar_action_load_from_dir,
cls_name="HunyuanTransformer3DARActionModel",
```

This is the **only** model that gets instantiated and weight-loaded at training startup.

---

### 5 – Dataset loader: everything else is precomputed

**File:** `trainer/dataset/ar_camera_hunyuan_w_mem_dataset.py`

Dataset init loads two precomputed negative-prompt tensors (lines 405–415):

```python
self.neg_prompt_pt = torch.load(                           # line 405
    "/your_path/to/hunyuan_neg_prompt.pt", ...)
self.neg_byt5_pt = torch.load(                             # line 411
    "/your_path/to/hunyuan_neg_byt5_prompt.pt", ...)
```

`__getitem__` (line 457) loads a per-clip precomputed file and unpacks all encoder outputs from it:

```python
latent_pt_path = json_data['latent_path']                  # line 461
latent_pt = torch.load(                                    # line 464
    os.path.join(latent_pt_path), map_location="cpu", weights_only=True)

latent          = latent_pt['latent'][0]                   # line 469  ← VAE latents
prompt_embed    = latent_pt['prompt_embeds'][0]            # line 482  ← Qwen LLM embeddings
prompt_mask     = latent_pt['prompt_mask'][0]              # line 483
image_cond      = latent_pt['image_cond'][0]               # line 485
vision_states   = latent_pt['vision_states'][0]            # line 486  ← SigLIP states
byt5_text_states= latent_pt['byt5_text_states'][0]         # line 487  ← ByT5 states
byt5_text_mask  = latent_pt['byt5_text_mask'][0]           # line 488
```

None of these keys are computed at training time — they are simply read from disk.

---

### 6 – Where the precomputed files are produced (offline, before training)

**File:** `worldcompass/prepare_dataset/prepare_image_text_latent_simple.py`

The `LatentExtractor._load_models()` method (line 106) is where **all four encoders are loaded** — but this is a *preprocessing script*, not the trainer:

```python
# VAE — line ~117
self.vae = hunyuanvideo_15_vae_w_cache.AutoencoderKLConv3D.from_pretrained(
    os.path.join(checkpoint_path, "vae"), torch_dtype=torch.float32)

# SigLIP vision encoder — line ~124
self.vision_encoder = VisionEncoder(
    vision_encoder_type="siglip",
    vision_encoder_path=os.path.join(checkpoint_path, "vision_encoder/siglip"), ...)

# Qwen2.5-VL LLM text encoder — loaded further in the same method
# ByT5 encoder — loaded further in the same method
```

Running this script produces the `.pt` files that the training dataset later reads.

---

## Summary

When you run `scripts/training/hyvideo15/run_ar_hunyuan_action_mem.sh`:

- **Only the `HunyuanTransformer3DARActionModel` (the DiT / AR transformer) is instantiated and loaded into GPU memory.**
  - Load site: `trainer/models/loader/component_loader.py`, `TransformerLoader.load()` → `maybe_load_fsdp_model()` (line 434)
- **Qwen2.5-VL-7B, ByT5, SigLIP, and VAE are never loaded during training.** Their outputs were computed in a prior offline step by `worldcompass/prepare_dataset/prepare_image_text_latent_simple.py` and saved to `.pt` files.
- **Latent vectors are fully precomputed / offline.** The training dataloader reads `latent`, `prompt_embeds`, `vision_states`, `byt5_text_states`, and `image_cond` directly from these `.pt` files (`trainer/dataset/ar_camera_hunyuan_w_mem_dataset.py`, lines 464–488) — no VAE encoding, no text encoding, no vision encoding happens during a training step.
