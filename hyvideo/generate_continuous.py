  # Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.


'''
cd /vast/users/ankan.deria/Document/HY-WorldPlay
export PYTHONPATH=$(pwd):$PYTHONPATH

HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0  torchrun --nproc_per_node=1 hyvideo/generate_continuous.py \
  --prompt "Fold the paper to make the paper plane." \
  --image_path /vast/users/ankan.deria/Document/HY-WorldPlay/assets/img/airplane_glider_005_clip_first_frame.png \
  --resolution 480p \
  --aspect_ratio 16:9 \
  --video_length 831 \
  --seed 1 \
  --rewrite false \
  --sr false \
  --pose "./assets/pose/static_504_latents.json" \
  --output_path ./outputs/continuous_fold \
  --stage_dir ./outputs/continuous_fold/stages \
  --final_output_name gen.mp4 \
  --model_path /vast/users/ankan.deria/Document/YUME_VLA/Pretrain_ckpt/HunyuanVideo-1.5 \
  --action_ckpt /vast/users/ankan.deria/Document/HY-WorldPlay/outputs/origami_ft/checkpoint-120/transformer/diffusion_pytorch_model.safetensors \
  --few_step false \
  --width 832 \
  --height 480 \
  --model_type ar \
  --group_offloading false \
  --use_vae_parallel false \
--keep_stage_artifacts false


'''


import argparse
import gc
import json
import os
import shutil
from pathlib import Path

if (
    "PYTORCH_ALLOC_CONF" not in os.environ
    and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ
):
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import einops
import imageio.v2 as imageio
import loguru
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from scipy.spatial.transform import Rotation as R

from hyvideo.commons.infer_state import initialize_infer_state
from hyvideo.commons.parallel_states import initialize_parallel_state
from hyvideo.generate_custom_trajectory import generate_camera_trajectory_local
from hyvideo.pipelines.worldplay_video_pipeline import HunyuanVideo_1_5_Pipeline

mapping = {
    (0, 0, 0, 0): 0,
    (1, 0, 0, 0): 1,
    (0, 1, 0, 0): 2,
    (0, 0, 1, 0): 3,
    (0, 0, 0, 1): 4,
    (1, 0, 1, 0): 5,
    (1, 0, 0, 1): 6,
    (0, 1, 1, 0): 7,
    (0, 1, 0, 1): 8,
}

FPS = 24


def one_hot_to_one_dimension(one_hot):
    return torch.tensor([mapping[tuple(row.tolist())] for row in one_hot])


def parse_pose_string(pose_string):
    forward_speed = 0.08
    yaw_speed = np.deg2rad(3)
    pitch_speed = np.deg2rad(3)

    motions = []
    commands = [cmd.strip() for cmd in pose_string.split(",")]
    for cmd in commands:
        if not cmd:
            continue

        parts = cmd.split("-")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid pose command: {cmd}. Expected format: 'action-duration'"
            )

        action = parts[0].strip()
        duration = int(float(parts[1].strip()))

        if action == "w":
            motions.extend([{"forward": forward_speed}] * duration)
        elif action == "s":
            motions.extend([{"forward": -forward_speed}] * duration)
        elif action == "a":
            motions.extend([{"right": -forward_speed}] * duration)
        elif action == "d":
            motions.extend([{"right": forward_speed}] * duration)
        elif action == "up":
            motions.extend([{"pitch": pitch_speed}] * duration)
        elif action == "down":
            motions.extend([{"pitch": -pitch_speed}] * duration)
        elif action == "left":
            motions.extend([{"yaw": -yaw_speed}] * duration)
        elif action == "right":
            motions.extend([{"yaw": yaw_speed}] * duration)
        else:
            raise ValueError(
                f"Unknown action: {action}. Supported actions: w, s, a, d, up, down, left, right"
            )
    return motions


def pose_string_to_json(pose_string):
    motions = parse_pose_string(pose_string)
    poses = generate_camera_trajectory_local(motions)
    intrinsic = [
        [969.6969696969696, 0.0, 960.0],
        [0.0, 969.6969696969696, 540.0],
        [0.0, 0.0, 1.0],
    ]
    pose_json = {}
    for i, pose in enumerate(poses):
        pose_json[str(i)] = {"extrinsic": pose.tolist(), "K": intrinsic}
    return pose_json


def pose_to_input(pose_data, latent_num, tps=False):
    if isinstance(pose_data, str):
        if pose_data.endswith(".json"):
            pose_json = json.load(open(pose_data, "r"))
        else:
            pose_json = pose_string_to_json(pose_data)
    elif isinstance(pose_data, dict):
        pose_json = pose_data
    else:
        raise ValueError(
            f"Invalid pose_data type: {type(pose_data)}. Expected str or dict."
        )

    pose_keys = list(pose_json.keys())
    latent_num_from_pose = len(pose_keys)
    assert latent_num_from_pose == latent_num, (
        f"pose corresponds to {latent_num_from_pose * 4 - 3} frames, num_frames "
        f"must be set to {latent_num_from_pose * 4 - 3} to ensure alignment."
    )

    intrinsic_list = []
    w2c_list = []
    for i in range(latent_num):
        t_key = pose_keys[i]
        c2w = np.array(pose_json[t_key]["extrinsic"])
        w2c = np.linalg.inv(c2w)
        w2c_list.append(w2c)
        intrinsic = np.array(pose_json[t_key]["K"])
        intrinsic[0, 0] /= intrinsic[0, 2] * 2
        intrinsic[1, 1] /= intrinsic[1, 2] * 2
        intrinsic[0, 2] = 0.5
        intrinsic[1, 2] = 0.5
        intrinsic_list.append(intrinsic)

    w2c_list = np.array(w2c_list)
    intrinsic_list = torch.tensor(np.array(intrinsic_list))

    c2ws = np.linalg.inv(w2c_list)
    c_inv = np.linalg.inv(c2ws[:-1])
    relative_c2w = np.zeros_like(c2ws)
    relative_c2w[0, ...] = c2ws[0, ...]
    relative_c2w[1:, ...] = c_inv @ c2ws[1:, ...]
    trans_one_hot = np.zeros((relative_c2w.shape[0], 4), dtype=np.int32)
    rotate_one_hot = np.zeros((relative_c2w.shape[0], 4), dtype=np.int32)

    move_norm_valid = 0.0001
    for i in range(1, relative_c2w.shape[0]):
        move_dirs = relative_c2w[i, :3, 3]
        move_norms = np.linalg.norm(move_dirs)
        if move_norms > move_norm_valid:
            move_norm_dirs = move_dirs / move_norms
            angles_rad = np.arccos(move_norm_dirs.clip(-1.0, 1.0))
            trans_angles_deg = angles_rad * (180.0 / torch.pi)
        else:
            trans_angles_deg = torch.zeros(3)

        rot_angles_deg = R.from_matrix(relative_c2w[i, :3, :3]).as_euler(
            "xyz", degrees=True
        )

        if move_norms > move_norm_valid:
            if (not tps) or (
                tps
                and abs(rot_angles_deg[1]) < 5e-2
                and abs(rot_angles_deg[0]) < 5e-2
            ):
                if trans_angles_deg[2] < 60:
                    trans_one_hot[i, 0] = 1
                elif trans_angles_deg[2] > 120:
                    trans_one_hot[i, 1] = 1

                if trans_angles_deg[0] < 60:
                    trans_one_hot[i, 2] = 1
                elif trans_angles_deg[0] > 120:
                    trans_one_hot[i, 3] = 1

        if rot_angles_deg[1] > 5e-2:
            rotate_one_hot[i, 0] = 1
        elif rot_angles_deg[1] < -5e-2:
            rotate_one_hot[i, 1] = 1

        if rot_angles_deg[0] > 5e-2:
            rotate_one_hot[i, 2] = 1
        elif rot_angles_deg[0] < -5e-2:
            rotate_one_hot[i, 3] = 1

    trans_one_hot = torch.tensor(trans_one_hot)
    rotate_one_hot = torch.tensor(rotate_one_hot)
    trans_one_label = one_hot_to_one_dimension(trans_one_hot)
    rotate_one_label = one_hot_to_one_dimension(rotate_one_hot)
    action_one_label = trans_one_label * 9 + rotate_one_label

    return torch.as_tensor(w2c_list), torch.as_tensor(intrinsic_list), action_one_label


def save_video(video, path, fps=FPS):
    if video.ndim == 5:
        assert video.shape[0] == 1
        video = video[0]
    vid = (video * 255).clamp(0, 255).to(torch.uint8)
    vid = einops.rearrange(vid, "c f h w -> f h w c")
    imageio.mimwrite(path, vid, fps=fps)


def str_to_bool(value):
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.lower().strip()
        if value in ("true", "1", "yes", "on"):
            return True
        if value in ("false", "0", "no", "off"):
            return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {value}")


def cleanup_tensors(finalize_distributed=False):
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

    if finalize_distributed and dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def gpu_memory_mb():
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return (
        torch.cuda.memory_allocated() / (1024**2),
        torch.cuda.memory_reserved() / (1024**2),
    )


def create_pipeline(args):
    transformer_version = f"{args.resolution}_i2v"
    if args.dtype == "bf16":
        transformer_dtype = torch.bfloat16
    elif args.dtype == "fp32":
        transformer_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}")

    return HunyuanVideo_1_5_Pipeline.create_pipeline(
        pretrained_model_name_or_path=args.model_path,
        transformer_version=transformer_version,
        enable_offloading=args.offloading,
        enable_group_offloading=args.group_offloading,
        create_sr_pipeline=False,
        force_sparse_attn=False,
        transformer_dtype=transformer_dtype,
        action_ckpt=args.action_ckpt,
    )


def ensure_single_process():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != 1 or rank != 0 or local_rank != 0:
        raise ValueError(
            "generate_continuous.py v1 supports only single-process / single-GPU inference."
        )


def load_latent_frames(frame_indices, chunk_cache=None, latents_dir: Path = None):
    if not frame_indices:
        return None

    tensors_by_chunk = {}
    frames = []
    for frame_idx in frame_indices:
        chunk_idx = frame_idx // 4
        offset = frame_idx % 4
        if chunk_idx not in tensors_by_chunk:
            if chunk_cache is not None and chunk_idx in chunk_cache:
                tensors_by_chunk[chunk_idx] = chunk_cache[chunk_idx]
            else:
                if latents_dir is None:
                    raise FileNotFoundError(
                        f"Chunk {chunk_idx} not found in memory and no latents_dir was provided."
                    )
                chunk_path = latents_dir / f"chunk_{chunk_idx:04d}.pt"
                if not chunk_path.exists():
                    raise FileNotFoundError(f"Missing latent chunk: {chunk_path}")
                tensors_by_chunk[chunk_idx] = torch.load(chunk_path, map_location="cpu")
        frames.append(tensors_by_chunk[chunk_idx][:, :, offset : offset + 1])
    return torch.cat(frames, dim=2)


def save_conditioning_cache(args, cache_path: Path):
    pipe = None
    try:
        pipe = create_pipeline(args)
        cache = pipe.build_continuous_generation_cache(
            prompt=args.prompt,
            aspect_ratio=args.aspect_ratio,
            video_length=args.video_length,
            prompt_rewrite=args.rewrite,
            num_inference_steps=args.num_inference_steps,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            reference_image=args.image_path,
            user_height=args.height,
            user_width=args.width,
            chunk_latent_frames=4,
        )
        latent_num = cache["latent_target_length"]
        viewmats, Ks, action = pose_to_input(args.pose, latent_num)
        cache["viewmats"] = viewmats.unsqueeze(0).cpu()
        cache["Ks"] = Ks.unsqueeze(0).cpu()
        cache["action"] = action.unsqueeze(0).cpu()
        cache["fps"] = FPS
        torch.save(cache, cache_path)
        return cache
    finally:
        del pipe
        cleanup_tensors()


def write_stage_clip(pipe, stage_idx, stage_latents, chunk_cache, latents_dir, clip_path):
    prev_last_latent = None
    latent_window = None
    decoded = None
    try:
        pipe.prepare_for_decode_only()
        if stage_idx == 0:
            decoded = pipe.decode_latent_window(stage_latents)
        else:
            prev_last_latent = load_latent_frames(
                [stage_idx * 4 - 1], chunk_cache=chunk_cache, latents_dir=latents_dir
            )
            latent_window = torch.cat([prev_last_latent, stage_latents], dim=2)
            decoded = pipe.decode_latent_window(latent_window)
            decoded = decoded[:, :, 1:, :, :]

        # decoded is already on CPU here; save from RAM, not GPU
        save_video(decoded, clip_path)
    finally:
        del decoded
        del latent_window
        del prev_last_latent
        cleanup_tensors()


def concatenate_stage_clips(clips_dir: Path, final_output_path: Path, target_frames: int):
    writer = imageio.get_writer(final_output_path, fps=FPS)
    written = 0
    try:
        for stage_clip in sorted(clips_dir.glob("stage_*.mp4")):
            reader = imageio.get_reader(stage_clip)
            try:
                for frame in reader:
                    if written >= target_frames:
                        break
                    writer.append_data(frame)
                    written += 1
            finally:
                reader.close()
            if written >= target_frames:
                break
    finally:
        writer.close()

    if written != target_frames:
        raise RuntimeError(
            f"Expected {target_frames} frames in final video, wrote {written}."
        )


def completed_frame_count(completed_chunks: int, target_frames: int):
    if completed_chunks <= 0:
        return 0
    return min(target_frames, completed_chunks * 16 - 3)


def run_continuous_generation(args):
    ensure_single_process()

    if args.model_type != "ar":
        raise ValueError("generate_continuous.py v1 supports only --model_type ar.")
    if args.sr:
        raise ValueError("generate_continuous.py v1 requires --sr false.")
    if args.image_path is None:
        raise ValueError("generate_continuous.py v1 currently requires --image_path.")
    if ((args.video_length - 1) // 4 + 1) % 4 != 0:
        raise ValueError("number of latents must be divisible by 4")

    stage_dir = Path(args.stage_dir or args.output_path or "./outputs/continuous")
    latents_dir = stage_dir / "latents"
    clips_dir = stage_dir / "clips"
    cache_path = stage_dir / "conditioning_cache.pt"
    final_output_path = stage_dir / args.final_output_name
    rolling_output_path = stage_dir / "full_video.mp4"
    stage_dir.mkdir(parents=True, exist_ok=True)
    latents_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu")
        loguru.logger.info(f"Loaded conditioning cache from {cache_path}")
    else:
        loguru.logger.info("Building conditioning cache for continuous generation")
        cache = save_conditioning_cache(args, cache_path)
        loguru.logger.info(f"Saved conditioning cache to {cache_path}")

    resume_stage = args.resume_stage if args.resume_stage is not None else 0
    chunk_num = cache["chunk_num"]
    if resume_stage < 0 or resume_stage >= chunk_num:
        raise ValueError(f"resume_stage must be in [0, {chunk_num - 1}]")

    chunk_cache = {}
    for stage_idx in range(resume_stage):
        chunk_path = latents_dir / f"chunk_{stage_idx:04d}.pt"
        if not chunk_path.exists():
            raise FileNotFoundError(
                f"Missing latent artifact for resumed stage {stage_idx}: "
                f"{latents_dir / f'chunk_{stage_idx:04d}.pt'}"
            )
        chunk_cache[stage_idx] = torch.load(chunk_path, map_location="cpu")
        if not (clips_dir / f"stage_{stage_idx:03d}.mp4").exists():
            raise FileNotFoundError(
                f"Missing clip artifact for resumed stage {stage_idx}: "
                f"{clips_dir / f'stage_{stage_idx:03d}.mp4'}"
            )

    pipe = None
    try:
        pipe = create_pipeline(args)
        for stage_idx in range(resume_stage, chunk_num):
            stage_chunk_path = latents_dir / f"chunk_{stage_idx:04d}.pt"
            stage_clip_path = clips_dir / f"stage_{stage_idx:03d}.mp4"
            start_idx = stage_idx * cache["chunk_latent_frames"]
            end_idx = start_idx + cache["chunk_latent_frames"]

            alloc_mb, reserved_mb = gpu_memory_mb()
            loguru.logger.info(
                f"Stage {stage_idx + 1}/{chunk_num} starting | "
                f"gpu_alloc={alloc_mb:.2f} MB | gpu_reserved={reserved_mb:.2f} MB"
            )

            selected_indices = None
            context_latents = None
            context_cond_latents = None
            stage_latents = None
            try:
                pipe.prepare_for_generation_only()
                selected_indices = pipe.get_ar_chunk_context_indices(
                    cache["viewmats"],
                    stage_idx,
                    chunk_latent_frames=cache["chunk_latent_frames"],
                    device=pipe.execution_device,
                )
                context_latents = load_latent_frames(
                    selected_indices,
                    chunk_cache=chunk_cache,
                    latents_dir=latents_dir,
                )
                context_cond_latents = (
                    cache["cond_latents"][:, :, selected_indices]
                    if selected_indices
                    else None
                )

                stage_latents = pipe.run_continuous_ar_chunk(
                    cache=cache,
                    chunk_idx=stage_idx,
                    viewmats=cache["viewmats"],
                    Ks=cache["Ks"],
                    action=cache["action"],
                    current_chunk_latents=cache["latents"][:, :, start_idx:end_idx],
                    current_chunk_cond_latents=cache["cond_latents"][:, :, start_idx:end_idx],
                    selected_frame_indices=selected_indices,
                    context_latents=context_latents,
                    context_cond_latents=context_cond_latents,
                    show_progress=True,
                )

                chunk_cache[stage_idx] = stage_latents
                torch.save(stage_latents, stage_chunk_path)
                del context_latents
                context_latents = None
                del context_cond_latents
                context_cond_latents = None
                del selected_indices
                selected_indices = None
                cleanup_tensors()

                write_stage_clip(
                    pipe,
                    stage_idx,
                    stage_latents,
                    chunk_cache,
                    latents_dir,
                    stage_clip_path,
                )
                cleanup_tensors()
                concatenate_stage_clips(
                    clips_dir,
                    rolling_output_path,
                    completed_frame_count(stage_idx + 1, cache["video_length"]),
                )
                loguru.logger.info(
                    f"Stage {stage_idx + 1}/{chunk_num} saved | "
                    f"latents={stage_chunk_path} | clip={stage_clip_path} | "
                    f"rolling_video={rolling_output_path}"
                )
            finally:
                del selected_indices
                del context_latents
                del context_cond_latents
                del stage_latents
                cleanup_tensors()
                alloc_mb, reserved_mb = gpu_memory_mb()
                loguru.logger.info(
                    f"Stage {stage_idx + 1}/{chunk_num} cleaned | "
                    f"gpu_alloc={alloc_mb:.2f} MB | gpu_reserved={reserved_mb:.2f} MB"
                )
    finally:
        del pipe
        cleanup_tensors()

    concatenate_stage_clips(clips_dir, final_output_path, cache["video_length"])
    loguru.logger.info(f"Saved concatenated video to {final_output_path}")

    shutil.rmtree(latents_dir, ignore_errors=True)
    if cache_path.exists():
        cache_path.unlink()
    loguru.logger.info("Removed intermediate latent artifacts and conditioning cache")

    if not args.keep_stage_artifacts:
        shutil.rmtree(clips_dir, ignore_errors=True)
        loguru.logger.info("Removed intermediate stage clips")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Continuous disk-backed video generation for HunyuanWorld-1.5"
    )
    parser.add_argument(
        "--pose",
        type=str,
        default="./assets/pose/test_forward_32_latents.json",
        help="Path to pose JSON file or pose string",
    )
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument(
        "--resolution",
        type=str,
        required=True,
        choices=["480p", "720p"],
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--action_ckpt", type=str, required=True)
    parser.add_argument("--aspect_ratio", type=str, default="16:9")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--video_length", type=int, default=125)
    parser.add_argument(
        "--sr",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--rewrite",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--offloading",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument(
        "--group_offloading",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=None,
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp32"],
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument(
        "--enable_torch_compile",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--few_step",
        type=str_to_bool,
        nargs="?",
        const=False,
        default=False,
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["bi", "ar"],
    )
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument(
        "--use_sageattn",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument("--sage_blocks_range", type=str, default="0-53")
    parser.add_argument(
        "--use_vae_parallel",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--use_fp8_gemm",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument("--quant_type", type=str, default="fp8-per-block")
    parser.add_argument("--include_patterns", type=str, default="double_blocks")

    parser.add_argument(
        "--stage_dir",
        type=str,
        default=None,
        help="Directory for conditioning cache, chunk latents, stage clips, and final video.",
    )
    parser.add_argument(
        "--keep_stage_artifacts",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Keep per-stage latent and clip files after final concatenation.",
    )
    parser.add_argument(
        "--resume_stage",
        type=int,
        default=None,
        help="Resume generation from this latent chunk index.",
    )
    parser.add_argument(
        "--final_output_name",
        type=str,
        default="gen.mp4",
        help="Final concatenated output filename under stage_dir.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    ensure_single_process()
    initialize_parallel_state(sp=1)
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    initialize_infer_state(args)

    try:
        run_continuous_generation(args)
    finally:
        cleanup_tensors(finalize_distributed=True)


if __name__ == "__main__":
    main()


# torchrun --nproc_per_node=1 hyvideo/generate_continuous1.py   --prompt "Fold the paper to make the paper plane."   --image_path /vast/users/ankan.deria/Document/HY-WorldPlay/assets/img/airplane_glider_005_clip_first_frame.png   --resolution 480p   --aspect_ratio 16:9   --video_length 381   --seed 1   --rewrite false   --sr false   --pose "./assets/pose/static_96_latents.json"   --output_path ./outputs/continuous_fold   --stage_dir ./outputs/continuous_fold/stages   --final_output_name gen.mp4   --model_path /vast/users/ankan.deria/Document/YUME_VLA/Pretrain_ckpt/HunyuanVideo-1.5   --action_ckpt /vast/users/ankan.deria/Document/HY-WorldPlay/outputs/origami_ft/checkpoint-120/transformer/diffusion_pytorch_model.safetensors   --few_step false   --width 832   --height 480   --model_type ar   --group_offloading false   --use_vae_parallel true  --keep_stage_artifacts false