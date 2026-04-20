# SPDX-License-Identifier: Apache-2.0
import json
import random
from multiprocessing import Manager

import torch
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from trainer.dataset.ar_camera_hunyuan_w_mem_dataset import (
    DP_SP_BatchSampler,
    latent_collate_function,
)
from trainer.distributed import get_local_torch_device
from trainer.distributed.parallel_state import (
    get_sp_world_size,
    get_world_rank,
    get_world_size,
)


def _build_static_w2c(latent_length: int) -> torch.Tensor:
    return torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(latent_length, 1, 1)


def _build_static_intrinsic(latent_length: int) -> torch.Tensor:
    intrinsic = torch.tensor(
        [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]], dtype=torch.float32
    )
    return intrinsic.unsqueeze(0).repeat(latent_length, 1, 1)


class OrigamiStepDataset(Dataset):
    def __init__(
        self,
        json_path,
        causal,
        window_frames,
        batch_size,
        cfg_rate,
        i2v_rate,
        drop_last,
        drop_first_row,
        seed,
        device,
        shared_state,
    ):
        del causal, i2v_rate, device
        with open(json_path, "r", encoding="utf-8") as fp:
            self.json_data = json.load(fp)
        self.all_length = len(self.json_data)
        self.window_frames = window_frames
        self.cfg_rate = cfg_rate
        self.rng = random.Random(seed)
        self.shared_state = shared_state

        self.sampler = DP_SP_BatchSampler(
            batch_size=batch_size,
            dataset_size=self.all_length,
            num_sp_groups=get_world_size() // get_sp_world_size(),
            sp_world_size=get_sp_world_size(),
            global_rank=get_world_rank(),
            drop_last=drop_last,
            drop_first_row=drop_first_row,
            seed=seed,
        )

    def __len__(self):
        return self.all_length

    def update_max_frames(self, training_step):
        if training_step < 500:
            self.shared_state["max_frames"] = 32
        elif training_step < 1000:
            self.shared_state["max_frames"] = 64
        elif training_step < 2000:
            self.shared_state["max_frames"] = 96
        elif training_step < 3000:
            self.shared_state["max_frames"] = 128
        else:
            self.shared_state["max_frames"] = 160

    def _get_negative_text_features(self, json_data, feature_pt):
        negative_feature_path = json_data.get("negative_feature_path")
        if negative_feature_path:
            negative_pt = torch.load(
                negative_feature_path, map_location="cpu", weights_only=True
            )
            return (
                negative_pt["prompt_embeds"][0],
                negative_pt["prompt_mask"][0],
                negative_pt["byt5_text_states"][0],
                negative_pt["byt5_text_mask"][0],
            )

        prompt_embed = torch.zeros_like(feature_pt["prompt_embeds"][0])
        prompt_mask = torch.zeros_like(feature_pt["prompt_mask"][0])
        byt5_text_states = torch.zeros_like(feature_pt["byt5_text_states"][0])
        byt5_text_mask = torch.zeros_like(feature_pt["byt5_text_mask"][0])
        return prompt_embed, prompt_mask, byt5_text_states, byt5_text_mask

    def __getitem__(self, idx):
        while True:
            try:
                json_data = self.json_data[idx]
                feature_pt = torch.load(
                    json_data["feature_pt_path"], map_location="cpu", weights_only=True
                )

                latent = feature_pt["latent"][0]
                latent_length = latent.shape[1]
                max_frames = int(self.shared_state["max_frames"]) // 4 * 4
                max_length = min(max_frames, latent_length // 4 * 4)
                if max_length < 4:
                    idx = self.rng.randint(0, self.all_length - 1)
                    continue

                latent = latent[:, :max_length, ...]
                prompt_embed = feature_pt["prompt_embeds"][0]
                prompt_mask = feature_pt["prompt_mask"][0]
                image_cond = feature_pt["image_cond"][0]
                vision_states = feature_pt["vision_states"][0]
                byt5_text_states = feature_pt["byt5_text_states"][0]
                byt5_text_mask = feature_pt["byt5_text_mask"][0]

                if self.rng.random() < self.cfg_rate:
                    (
                        prompt_embed,
                        prompt_mask,
                        byt5_text_states,
                        byt5_text_mask,
                    ) = self._get_negative_text_features(json_data, feature_pt)

                latent_t = latent.shape[1]
                w2c = _build_static_w2c(latent_t)
                intrinsic = _build_static_intrinsic(latent_t)
                action = torch.zeros(latent_t, dtype=torch.long)
                i2v_mask = torch.ones_like(latent)

                batch = {
                    "i2v_mask": i2v_mask,
                    "latent": latent,
                    "prompt_embed": prompt_embed,
                    "w2c": w2c,
                    "intrinsic": intrinsic,
                    "action": action,
                    "action_for_pe": action,
                    "context_frames_list": None,
                    "select_window_out_flag": 0,
                    "video_path": json_data["clip_path"],
                    "max_length": max_frames,
                    "image_cond": image_cond,
                    "vision_states": vision_states,
                    "prompt_mask": prompt_mask,
                    "byt5_text_states": byt5_text_states,
                    "byt5_text_mask": byt5_text_mask,
                }
                break
            except Exception as exc:
                print("error:", exc, json_data.get("feature_pt_path"), flush=True)
                idx = self.rng.randint(0, self.all_length - 1)
        return batch


def build_origami_step_dataloader(
    json_path,
    causal,
    window_frames,
    batch_size,
    num_data_workers,
    drop_last,
    drop_first_row,
    seed,
    cfg_rate,
    i2v_rate,
) -> tuple[OrigamiStepDataset, StatefulDataLoader]:
    manager = Manager()
    shared_state = manager.dict()
    shared_state["max_frames"] = window_frames

    dataset = OrigamiStepDataset(
        json_path,
        causal,
        window_frames,
        batch_size,
        cfg_rate,
        i2v_rate,
        drop_last=drop_last,
        drop_first_row=drop_first_row,
        seed=seed,
        device=get_local_torch_device(),
        shared_state=shared_state,
    )

    loader = StatefulDataLoader(
        dataset,
        batch_sampler=dataset.sampler,
        collate_fn=latent_collate_function,
        num_workers=num_data_workers,
        pin_memory=True,
        persistent_workers=num_data_workers > 0,
    )
    return dataset, loader
