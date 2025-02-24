import ast
import asyncio
from datetime import datetime
import gc
import importlib
import argparse
import math
import os
import pathlib
import re
import sys
import random
import time
import json
from multiprocessing import Value
from typing import Any, Dict, List, Optional
import accelerate
import numpy as np
from packaging.version import Version
from pathlib import Path

import huggingface_hub
import toml

import torch
from tqdm import tqdm
from accelerate.utils import set_seed
from accelerate import Accelerator, InitProcessGroupKwargs, DistributedDataParallelKwargs, PartialState
from safetensors.torch import load_file
import transformers
from diffusers.optimization import (
    SchedulerType as DiffusersSchedulerType,
    TYPE_TO_SCHEDULER_FUNCTION as DIFFUSERS_TYPE_TO_SCHEDULER_FUNCTION,
)
from transformers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION
from diffusers.utils.torch_utils import randn_tensor
from dataset import config_utils
from hunyuan_model.models import load_transformer, get_rotary_pos_embed_by_shape, HYVideoDiffusionTransformer
import hunyuan_model.text_encoder as text_encoder_module
from hunyuan_model.vae import load_vae, VAE_VER
import hunyuan_model.vae as vae_module
from modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
import networks.lora as lora_module
from networks.controlnet_adapter import ControlNetAdapter, DiffusionTransformerWithControl, ConditionedVideoDataset

import logging

from utils import huggingface_utils, model_utils, train_utils, sai_model_spec

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ----- Removed Control Type Definitions and Adapter Classes -----
# (Now handled in controlnet_adapter.py)

BASE_MODEL_VERSION_HUNYUAN_VIDEO = "hunyuan_video"

SS_METADATA_KEY_BASE_MODEL_VERSION = "ss_base_model_version"
SS_METADATA_KEY_NETWORK_MODULE = "ss_network_module"
SS_METADATA_KEY_NETWORK_DIM = "ss_network_dim"
SS_METADATA_KEY_NETWORK_ALPHA = "ss_network_alpha"
SS_METADATA_KEY_NETWORK_ARGS = "ss_network_args"

SS_METADATA_MINIMUM_KEYS = [
    SS_METADATA_KEY_BASE_MODEL_VERSION,
    SS_METADATA_KEY_NETWORK_MODULE,
    SS_METADATA_KEY_NETWORK_DIM,
    SS_METADATA_KEY_NETWORK_ALPHA,
    SS_METADATA_KEY_NETWORK_ARGS,
]

def clean_memory_on_device(device: torch.device):
    r"""
    Clean memory on the specified device, will be called from training scripts.
    """
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if device.type == "xpu":
        torch.xpu.empty_cache()
    if device.type == "mps":
        torch.mps.empty_cache()

class collator_class:
    def __init__(self, current_epoch, current_step, dataset):  # Rename parameters
        self.current_epoch = current_epoch
        self.current_step = current_step
        self.dataset = dataset

    def __call__(self, examples):
        return examples[0]  # Remove the dataset.set_current_epoch calls


def prepare_accelerator(args: argparse.Namespace) -> Accelerator:
    if args.logging_dir is None:
        logging_dir = None
    else:
        log_prefix = "" if args.log_prefix is None else args.log_prefix
        logging_dir = args.logging_dir + "/" + log_prefix + time.strftime("%Y%m%d%H%M%S", time.localtime())

    if args.log_with is None:
        log_with = "tensorboard" if logging_dir is not None else None
    else:
        log_with = args.log_with

    kwargs_handlers = [
        (
            InitProcessGroupKwargs(
                backend="gloo" if os.name == "nt" or not torch.cuda.is_available() else "nccl",
                init_method=(
                    "env://?use_libuv=False" if os.name == "nt" and Version(torch.__version__) >= Version("2.4.0") else None
                ),
                timeout=datetime.timedelta(minutes=args.ddp_timeout) if args.ddp_timeout else None,
            )
            if torch.cuda.device_count() > 1
            else None
        ),
        (
            DistributedDataParallelKwargs(
                gradient_as_bucket_view=args.ddp_gradient_as_bucket_view, static_graph=args.ddp_static_graph
            )
            if args.ddp_gradient_as_bucket_view or args.ddp_static_graph
            else None
        ),
    ]
    kwargs_handlers = [i for i in kwargs_handlers if i is not None]

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=log_with,
        project_dir=logging_dir,
        kwargs_handlers=kwargs_handlers,
    )
    print("accelerator device:", accelerator.device)
    return accelerator

# ----- Removed Control-Specific Dataset Class -----
# (Now handled in controlnet_adapter.py)

def line_to_prompt_dict(line: str) -> dict:
    prompt_args = line.split(" --")
    prompt_dict = {"prompt": prompt_args[0]}
    
    for parg in prompt_args:
        try:
            if m := re.match(r"w (\d+)", parg, re.IGNORECASE):
                prompt_dict["width"] = int(m.group(1))
            elif m := re.match(r"h (\d+)", parg, re.IGNORECASE):
                prompt_dict["height"] = int(m.group(1))
            elif m := re.match(r"f (\d+)", parg, re.IGNORECASE):
                prompt_dict["frame_count"] = int(m.group(1))
            elif m := re.match(r"d (\d+)", parg, re.IGNORECASE):
                prompt_dict["seed"] = int(m.group(1))
            elif m := re.match(r"s (\d+)", parg, re.IGNORECASE):
                prompt_dict["sample_steps"] = max(1, min(1000, int(m.group(1))))
            elif m := re.match(r"g ([\d\.]+)", parg, re.IGNORECASE):
                prompt_dict["guidance_scale"] = float(m.group(1))
        except ValueError as ex:
            logger.error(f"Exception in parsing: {parg}")
            logger.error(ex)
    return prompt_dict

def load_prompts(prompt_file: str) -> list[Dict]:
    if prompt_file.endswith(".txt"):
        with open(prompt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        prompts = [line.strip() for line in lines if len(line.strip()) > 0 and line[0] != "#"]
    elif prompt_file.endswith(".toml"):
        with open(prompt_file, "r", encoding="utf-8") as f:
            data = toml.load(f)
        prompts = [dict(**data["prompt"], **subset) for subset in data["prompt"]["subset"]]
    elif prompt_file.endswith(".json"):
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompts = json.load(f)

    for i in range(len(prompts)):
        prompt_dict = prompts[i]
        if isinstance(prompt_dict, str):
            prompt_dict = line_to_prompt_dict(prompt_dict)
            prompts[i] = prompt_dict
        prompt_dict["enum"] = i
        prompt_dict.pop("subset", None)
    return prompts

def compute_density_for_timestep_sampling(
    weighting_scheme: str, batch_size: int, logit_mean: float = None, logit_std: float = None, mode_scale: float = None
):
    if weighting_scheme == "logit_normal":
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device="cpu")
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device="cpu")
    return u

def get_sigmas(noise_scheduler, timesteps, device, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)

    if any([(schedule_timesteps == t).sum() == 0 for t in timesteps]):
        logger.warning("Some timesteps not in schedule, rounding to nearest")
        step_indices = [torch.argmin(torch.abs(schedule_timesteps - t)).item() for t in timesteps]
    else:
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma

def compute_loss_weighting_for_sd3(weighting_scheme: str, noise_scheduler, timesteps, device, dtype):
    if weighting_scheme == "sigma_sqrt" or weighting_scheme == "cosmap":
        sigmas = get_sigmas(noise_scheduler, timesteps, device, n_dim=5, dtype=dtype)
        weighting = (sigmas**-2.0).float() if weighting_scheme == "sigma_sqrt" else 2 / (math.pi * (1 - 2 * sigmas + 2 * sigmas**2))
    else:
        weighting = None
    return weighting

def should_sample_images(args, steps, epoch=None):
    if steps == 0:
        return False if not args.sample_at_first else True
    should_sample = (args.sample_every_n_steps and steps % args.sample_every_n_steps == 0) or \
                   (args.sample_every_n_epochs and epoch and epoch % args.sample_every_n_epochs == 0)
    return should_sample

def sample_images(accelerator, args, epoch, steps, vae, transformer, sample_parameters, dit_dtype):
    if not should_sample_images(args, steps, epoch):
        return

    logger.info(f"Generating sample images at step: {steps}")
    if not sample_parameters:
        logger.error(f"No prompt file: {args.sample_prompts}")
        return

    distributed_state = PartialState()
    save_dir = os.path.join(args.output_dir, "sample")
    os.makedirs(save_dir, exist_ok=True)

    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

    transformer: HYVideoDiffusionTransformer = accelerator.unwrap_model(transformer)
    transformer.switch_block_swap_for_inference()

    try:
        if distributed_state.num_processes <= 1:
            for params in sample_parameters:
                sample_image_inference(accelerator, args, transformer, dit_dtype, vae, save_dir, params, epoch, steps)
                clean_memory_on_device(accelerator.device)
        else:
            per_process_params = [sample_parameters[i::distributed_state.num_processes] for i in range(distributed_state.num_processes)]
            with distributed_state.split_between_processes(per_process_params) as params_list:
                for params in params_list[0]:
                    sample_image_inference(accelerator, args, transformer, dit_dtype, vae, save_dir, params, epoch, steps)
                    clean_memory_on_device(accelerator.device)
    finally:
        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state)
        transformer.switch_block_swap_for_training()
        clean_memory_on_device(accelerator.device)

def sample_image_inference(
    accelerator, args, transformer, dit_dtype, vae, save_dir, sample_parameter, epoch, steps
):
    logger = logging.getLogger(__name__)
    logger.info("Entering sample_image_inference()")
    
    # Show what args.dit_dtype is before we do anything
    logger.info(f"args.dit_dtype is: {args.dit_dtype!r}")
    
    # Convert string => torch.dtype (if needed):
    if args.dit_dtype is not None:
        # Log the pre-conversion str
        logger.info(f"Converting string '{args.dit_dtype}' to real torch.dtype")
        real_dit_dtype = str_to_dtype(args.dit_dtype)
        logger.info(f"str_to_dtype('{args.dit_dtype}') => {real_dit_dtype}")
        dit_dtype = real_dit_dtype
    else:
        # Default to bfloat16 if not given
        dit_dtype = torch.bfloat16
        logger.info("args.dit_dtype is None; defaulting to torch.bfloat16")
    
    # Just to double-check
    logger.info(f"Final dit_dtype (should be a torch.dtype): {dit_dtype}")
    
    # Retrieve prompt and sampling parameters
    num_videos_per_prompt = sample_parameter.get("num_videos_per_prompt", 1)
    width = sample_parameter.get("width", 272)
    height = sample_parameter.get("height", 480)
    frame_count = sample_parameter.get("frame_count", 1)
    guidance_scale = sample_parameter.get("guidance_scale", 6.0)
    discrete_flow_shift = sample_parameter.get("discrete_flow_shift", 14.5)
    seed = sample_parameter.get("seed")
    prompt = sample_parameter.get("prompt", "")
    
    # Determine latent video length
    if "884" in vae.config._class_name:
        latent_video_length = (frame_count - 1) // 4 + 1
    elif "888" in vae.config._class_name:
        latent_video_length = (frame_count - 1) // 8 + 1
    else:
        latent_video_length = frame_count
    
    device = accelerator.device
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        generator = torch.Generator(device=device).manual_seed(seed)
        logger.info(f"Set manual seed to {seed}")
    else:
        generator = torch.Generator(device=device).manual_seed(torch.initial_seed())
        logger.info(f"No explicit seed; using torch.initial_seed() => {torch.initial_seed()}")
    
    logger.info(f"Sampling prompt: {prompt}")
    logger.info(f"width={width}, height={height}, frame_count={frame_count}, sample_steps={sample_parameter.get('sample_steps', 10)}")
    
    # Prepare scheduler
    logger.info("Creating FlowMatchDiscreteScheduler")
    scheduler = FlowMatchDiscreteScheduler(shift=discrete_flow_shift, reverse=True, solver="euler")
    sample_steps = sample_parameter.get("sample_steps", 10)
    scheduler.set_timesteps(sample_steps, device=device)
    timesteps = scheduler.timesteps
    logger.info(f"Prepared scheduler with sample_steps={sample_steps}; timesteps={timesteps}")
    
    # Build the latent noise tensor
    comp_ratio = vae.config.spatial_compression_ratio
    latent_height = int(height) // comp_ratio
    latent_width = int(width) // comp_ratio
    num_channels_latents = 16
    shape_or_frame = (
        num_videos_per_prompt,
        num_channels_latents,
        1,
        latent_height,
        latent_width,
    )
    logger.info(f"VAE comp_ratio={comp_ratio}, shape for each frame={shape_or_frame}, total frames={latent_video_length}")
    
    latents_noise_list = []
    for i in range(latent_video_length):
        noise_frame = randn_tensor(shape_or_frame, generator=generator, device=device, dtype=dit_dtype)
        latents_noise_list.append(noise_frame)
    latents_noise = torch.cat(latents_noise_list, dim=2)
    logger.info(f"Built noise latents shape={latents_noise.shape}, dtype={latents_noise.dtype}")
    
    # Build noise latents as before:
    latents = latents_noise

    # Now define pose_latent (None by default)
    # ----------------------------------------------------------------
    # 1) Load the pose latent from disk
    # ----------------------------------------------------------------
    pose_latent = None
    if args.pose is not None:
        from safetensors.torch import load_file
        logger.info(f"Loading pose from {args.pose}")
        pose_data = load_file(args.pose)
        pose_latent = pose_data["latent"].to(accelerator.device, dtype=dit_dtype)
        
        # -- Add logging for its shape/dtype:
        logger.info(f"(DEBUG) Pose latent loaded from disk has shape={pose_latent.shape}, dtype={pose_latent.dtype}")

        # If your code expects (B,C,T,H,W) but the file is (T,C,H,W), fix it here:
        if pose_latent.ndim == 4:
            # interpret dim0 as frames => shape (F, C, H, W)
            # Insert a batch dimension and reorder to (B=1, C, T=F, H, W):
            pose_latent = pose_latent.unsqueeze(0)
    
    guidance_expand = torch.tensor([guidance_scale * 1000.0], dtype=torch.float32, device=device).to(dit_dtype)
    logger.info(f"Guidance expand shape={guidance_expand.shape}, dtype={guidance_expand.dtype}, data={guidance_expand}")
    
    # Obtain rotary pos embeddings
    freqs_cos, freqs_sin = get_rotary_pos_embed_by_shape(transformer, latents.shape[2:])
    freqs_cos = freqs_cos.to(device=device, dtype=dit_dtype)
    freqs_sin = freqs_sin.to(device=device, dtype=dit_dtype)
    logger.info(f"freqs_cos shape={freqs_cos.shape}, dtype={freqs_cos.dtype}; freqs_sin shape={freqs_sin.shape}, dtype={freqs_sin.dtype}")
    
    # Retrieve additional embeddings
    prompt_embeds = sample_parameter["llm_embeds"].to(device=device, dtype=dit_dtype)
    prompt_mask = sample_parameter["llm_mask"].to(device=device)
    prompt_embeds_2 = sample_parameter["clipL_embeds"].to(device=device, dtype=dit_dtype)
    logger.info(f"prompt_embeds shape={prompt_embeds.shape}, dtype={prompt_embeds.dtype}")
    logger.info(f"prompt_mask shape={prompt_mask.shape}, dtype={prompt_mask.dtype}")
    logger.info(f"prompt_embeds_2 shape={prompt_embeds_2.shape}, dtype={prompt_embeds_2.dtype}")
    
    prompt_idx = sample_parameter.get("enum", 0)
    logger.info(f"Beginning diffusion sampling; timesteps={timesteps}, len={len(timesteps)}")
    
    def get_pose_alpha(step, total_steps, start=1.0, end=0.0):
        return 1.0
    
    latents = latents.clone()
    
    from tqdm import tqdm
    with torch.no_grad():
        for i, t in enumerate(tqdm(timesteps, desc=f"Sampling prompt_idx={prompt_idx}")):
            alpha_t = 1.0
            
            latents_step = latents.clone()
            
            # Log shapes/dtypes at each step if needed
            logger.info(
                f"Step {i}: alpha_t={alpha_t:.4f}, latents_step shape={latents_step.shape}, dtype={latents_step.dtype}, t={t}"
            )
            
            scaled_input = scheduler.scale_model_input(latents_step, t)
            
            # If for some reason t is still a python scalar or if dtype is str:
            logger.info(f" t.repeat => {t.repeat(latents_step.shape[0])}, device={device}, dtype={dit_dtype}")
            time_tensor = t.repeat(latents_step.shape[0]).to(device=device, dtype=dit_dtype)
            logger.info(f" time_tensor shape={time_tensor.shape}, dtype={time_tensor.dtype}, data={time_tensor}")
            
            # Now call the transformer
            noise_pred = transformer(
                x=scaled_input,                # The usual diffusion latents
                t=time_tensor,
                text_states=None,
                text_states_2=None,
                freqs_cos=freqs_cos,
                freqs_sin=freqs_sin,
                pose_input=pose_latent,        # Pass your skeleton or pose encoding here
                pose_alpha=alpha_t,            # How strongly to apply that skeleton
                guidance=guidance_expand,
                return_dict=False,
            )

            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    
    # Decode latents
    logger.info("Decoding latents...")
    vae.to(device)
    vae.eval()
    if hasattr(vae.config, "shift_factor") and vae.config.shift_factor:
        latents = latents / vae.config.scaling_factor + vae.config.shift_factor
    else:
        latents = latents / vae.config.scaling_factor
    latents = latents.to(device=device, dtype=vae.dtype)
    logger.info(f"After scaling, latents shape={latents.shape}, dtype={latents.dtype}")
    
    with torch.no_grad():
        video = vae.decode(latents, return_dict=False)[0]
    video = (video / 2 + 0.5).clamp(0, 1)
    video = video.cpu().float()
    logger.info(f"Decoded video shape={video.shape}, dtype={video.dtype}")
    
    # Save video or image
    import time, os
    ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
    seed_suffix = "" if seed is None else f"_{seed}"
    save_path = (
        f"{'' if args.output_name is None else args.output_name + '_'}{num_suffix}_{prompt_idx:02d}_{ts_str}{seed_suffix}"
    )
    
    # Save the final result
    if video.shape[2] == 1:
        save_images_grid(
            videos=video,
            parent_dir=save_dir,     # Could be just the directory
            image_name=save_path,    # The file/stem name
            create_subdir=False
        )
    else:
        save_videos_grid(video, os.path.join(save_dir, save_path) + ".mp4")
    logger.info(f"Saved result to {save_path}")
    vae.to("cpu")
    logger.info("Exiting sample_image_inference()")

CONTROL_TYPES = ['pose', 'depth', 'canny']


class NetworkTrainer:
    def __init__(self):
        self.control_type = None

    def generate_step_logs(
        self,
        args: argparse.Namespace,
        current_loss,
        avr_loss,
        lr_scheduler,
        lr_descriptions,
        optimizer=None,
        keys_scaled=None,
        mean_norm=None,
        maximum_norm=None,
    ):
        network_train_unet_only = True
        logs = {"loss/current": current_loss, "loss/average": avr_loss}

        if keys_scaled is not None:
            logs["max_norm/keys_scaled"] = keys_scaled
            logs["max_norm/average_key_norm"] = mean_norm
            logs["max_norm/max_key_norm"] = maximum_norm

        lrs = lr_scheduler.get_last_lr()
        for i, lr in enumerate(lrs):
            if lr_descriptions is not None:
                lr_desc = lr_descriptions[i]
            else:
                idx = i - (0 if network_train_unet_only else -1)
                if idx == -1:
                    lr_desc = "textencoder"
                else:
                    if len(lrs) > 2:
                        lr_desc = f"group{idx}"
                    else:
                        lr_desc = "unet"

            logs[f"lr/{lr_desc}"] = lr

            if args.optimizer_type.lower().startswith("DAdapt".lower()) or args.optimizer_type.lower() == "Prodigy".lower():
                logs[f"lr/d*lr/{lr_desc}"] = (
                    lr_scheduler.optimizers[-1].param_groups[i]["d"] * lr_scheduler.optimizers[-1].param_groups[i]["lr"]
                )
            if (
                args.optimizer_type.lower().endswith("ProdigyPlusScheduleFree".lower()) and optimizer is not None
            ):
                logs["lr/d*lr"] = optimizer.param_groups[0]["d"] * optimizer.param_groups[0]["lr"]
        else:
            idx = 0
            if not network_train_unet_only:
                logs["lr/textencoder"] = float(lrs[0])
                idx = 1

            for i in range(idx, len(lrs)):
                logs[f"lr/group{i}"] = float(lrs[i])
                if args.optimizer_type.lower().startswith("DAdapt".lower()) or args.optimizer_type.lower() == "Prodigy".lower():
                    logs[f"lr/d*lr/group{i}"] = (
                        lr_scheduler.optimizers[-1].param_groups[i]["d"] * lr_scheduler.optimizers[-1].param_groups[i]["lr"]
                    )
                if args.optimizer_type.lower().endswith("ProdigyPlusScheduleFree".lower()) and optimizer is not None:
                    logs[f"lr/d*lr/group{i}"] = optimizer.param_groups[i]["d"] * optimizer.param_groups[i]["lr"]

        return logs

    def process_sample_prompts(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        sample_prompts: str,
        text_encoder1: str,
        text_encoder2: str,
        fp8_llm: bool,
    ):
        logger.info(f"cache Text Encoder outputs for sample prompt: {sample_prompts}")
        prompts = load_prompts(sample_prompts)

        def encode_for_text_encoder(text_encoder, is_llm=True):
            sample_prompts_te_outputs = {}  # (prompt) -> (embeds, mask)
            with accelerator.autocast(), torch.no_grad():
                for prompt_dict in prompts:
                    for p in [prompt_dict.get("prompt", "")]:
                        if p not in sample_prompts_te_outputs:
                            logger.info(f"cache Text Encector outputs for prompt: {p}")

                            data_type = "video"
                            text_inputs = text_encoder.text2tokens(p, data_type=data_type)

                            prompt_outputs = text_encoder.encode(text_inputs, data_type=data_type)
                            sample_prompts_te_outputs[p] = (prompt_outputs.hidden_state, prompt_outputs.attention_mask)

            return sample_prompts_te_outputs

        # Load Text Encoder 1 and encode
        text_encoder_dtype = torch.float16 if args.text_encoder_dtype is None else model_utils.str_to_dtype(args.text_encoder_dtype)
        logger.info(f"loading text encoder 1: {text_encoder1}")
        text_encoder_1 = text_encoder_module.load_text_encoder_1(text_encoder1, accelerator.device, fp8_llm, text_encoder_dtype)

        logger.info("encoding with Text Encector 1")
        te_outputs_1 = encode_for_text_encoder(text_encoder_1)
        del text_encoder_1

        # Load Text Encoder 2 and encode
        logger.info(f"loading text encoder 2: {text_encoder2}")
        text_encoder_2 = text_encoder_module.load_text_encoder_2(text_encoder2, accelerator.device, text_encoder_dtype)

        logger.info("encoding with Text Encoder 2")
        te_outputs_2 = encode_for_text_encoder(text_encoder_2, is_llm=False)
        del text_encoder_2

        # prepare sample parameters
        sample_parameters = []
        for prompt_dict in prompts:
            prompt_dict_copy = prompt_dict.copy()
            p = prompt_dict.get("prompt", "")
            prompt_dict_copy["llm_embeds"] = te_outputs_1[p][0]
            prompt_dict_copy["llm_mask"] = te_outputs_1[p][1]
            prompt_dict_copy["clipL_embeds"] = te_outputs_2[p][0]
            prompt_dict_copy["clipL_mask"] = te_outputs_2[p][1]
            sample_parameters.append(prompt_dict_copy)

        clean_memory_on_device(accelerator.device)

        return sample_parameters

    def get_optimizer(self, args, trainable_params: list[torch.nn.Parameter]) -> tuple[str, str, torch.optim.Optimizer]:
        optimizer_type = args.optimizer_type.lower()

        # split optimizer_type and optimizer_args
        optimizer_kwargs = {}
        if args.optimizer_args is not None and len(args.optimizer_args) > 0:
            for arg in args.optimizer_args:
                key, value = arg.split("=")
                value = ast.literal_eval(value)
                optimizer_kwargs[key] = value

        lr = args.learning_rate
        optimizer = None
        optimizer_class = None

        if optimizer_type.endswith("8bit".lower()):
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError("No bitsandbytes / bitsandbytesがインストールされていないようです")

            if optimizer_type == "AdamW8bit".lower():
                logger.info(f"use 8-bit AdamW optimizer | {optimizer_kwargs}")
                optimizer_class = bnb.optim.AdamW8bit
                optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

        elif optimizer_type == "Adafactor".lower():
            if "relative_step" not in optimizer_kwargs:
                optimizer_kwargs["relative_step"] = True
            if not optimizer_kwargs["relative_step"] and optimizer_kwargs.get("warmup_init", False):
                logger.info(
                    f"set relative_step to True because warmup_init is True / warmup_initがTrueのためrelative_stepをTrueにします"
                )
                optimizer_kwargs["relative_step"] = True
            logger.info(f"use Adafactor optimizer | {optimizer_kwargs}")

            if optimizer_kwargs["relative_step"]:
                logger.info(f"relative_step is true / relative_stepがtrueです")
                if lr != 0.0:
                    logger.warning(f"learning rate is used as initial_lr / 指定したlearning rateはinitial_lrとして使用されます")
                args.learning_rate = None

                if args.lr_scheduler != "adafactor":
                    logger.info(f"use adafactor_scheduler / スケジューラにadafactor_schedulerを使用します")
                args.lr_scheduler = f"adafactor:{lr}"

                lr = None
            else:
                if args.max_grad_norm != 0.0:
                    logger.warning(
                        f"because max_grad_norm is set, clip_grad_norm is enabled. consider set to 0 / max_grad_normが設定されているためclip_grad_normが有効になります。0に設定して無効にしたほうがいいかもしれません"
                    )
                if args.lr_scheduler != "constant_with_warmup":
                    logger.warning(f"constant_with_warmup will be good / スケジューラはconstant_with_warmupが良いかもしれません")
                if optimizer_kwargs.get("clip_threshold", 1.0) != 1.0:
                    logger.warning(f"clip_threshold=1.0 will be good / clip_thresholdは1.0が良いかもしれません")

            optimizer_class = transformers.optimization.Adafactor
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

        elif optimizer_type == "AdamW".lower():
            logger.info(f"use AdamW optimizer | {optimizer_kwargs}")
            optimizer_class = torch.optim.AdamW
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

        if optimizer is None:
            case_sensitive_optimizer_type = args.optimizer_type
            logger.info(f"use {case_sensitive_optimizer_type} | {optimizer_kwargs}")

            if "." not in case_sensitive_optimizer_type:
                optimizer_module = torch.optim
            else:
                values = case_sensitive_optimizer_type.split(".")
                optimizer_module = importlib.import_module(".".join(values[:-1]))
                case_sensitive_optimizer_type = values[-1]

            optimizer_class = getattr(optimizer_module, case_sensitive_optimizer_type)
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

        optimizer_name = optimizer_class.__module__ + "." + optimizer_class.__name__
        optimizer_args = ",".join([f"{k}={v}" for k, v in optimizer_kwargs.items()])

        if hasattr(optimizer, "train") and callable(optimizer.train):
            train_fn = optimizer.train
            eval_fn = optimizer.eval
        else:
            train_fn = lambda: None
            eval_fn = lambda: None

        return optimizer_name, optimizer_args, optimizer, train_fn, eval_fn

    def is_schedulefree_optimizer(self, optimizer: torch.optim.Optimizer, args: argparse.Namespace) -> bool:
        return args.optimizer_type.lower().endswith("schedulefree".lower())

    def get_dummy_scheduler(optimizer: torch.optim.Optimizer) -> Any:
        class DummyScheduler:
            def __init__(self, optimizer: torch.optim.Optimizer):
                self.optimizer = optimizer

            def step(self):
                pass

            def get_last_lr(self):
                return [group["lr"] for group in self.optimizer.param_groups]

        return DummyScheduler(optimizer)

    def get_scheduler(self, args, optimizer: torch.optim.Optimizer, num_processes: int):
        if self.is_schedulefree_optimizer(optimizer, args):
            return self.get_dummy_scheduler(optimizer)

        name = args.lr_scheduler
        num_training_steps = args.max_train_steps * num_processes
        num_warmup_steps: Optional[int] = (
            int(args.lr_warmup_steps * num_training_steps) if isinstance(args.lr_warmup_steps, float) else args.lr_warmup_steps
        )
        num_decay_steps: Optional[int] = (
            int(args.lr_decay_steps * num_training_steps) if isinstance(args.lr_decay_steps, float) else args.lr_decay_steps
        )
        num_stable_steps = num_training_steps - num_warmup_steps - num_decay_steps
        num_cycles = args.lr_scheduler_num_cycles
        power = args.lr_scheduler_power
        timescale = args.lr_scheduler_timescale
        min_lr_ratio = args.lr_scheduler_min_lr_ratio

        lr_scheduler_kwargs = {}
        if args.lr_scheduler_args is not None and len(args.lr_scheduler_args) > 0:
            for arg in args.lr_scheduler_args:
                key, value = arg.split("=")
                value = ast.literal_eval(value)
                lr_scheduler_kwargs[key] = value

        def wrap_check_needless_num_warmup_steps(return_vals):
            if num_warmup_steps is not None and num_warmup_steps != 0:
                raise ValueError(f"{name} does not require `num_warmup_steps`. Set None or 0.")
            return return_vals

        if args.lr_scheduler_type:
            lr_scheduler_type = args.lr_scheduler_type
            logger.info(f"use {lr_scheduler_type} | {lr_scheduler_kwargs} as lr_scheduler")
            if "." not in lr_scheduler_type:
                lr_scheduler_module = torch.optim.lr_scheduler
            else:
                values = lr_scheduler_type.split(".")
                lr_scheduler_module = importlib.import_module(".".join(values[:-1]))
                lr_scheduler_type = values[-1]
            lr_scheduler_class = getattr(lr_scheduler_module, lr_scheduler_type)
            lr_scheduler = lr_scheduler_class(optimizer, **lr_scheduler_kwargs)
            return lr_scheduler

        if name.startswith("adafactor"):
            assert (
                type(optimizer) == transformers.optimization.Adafactor
            ), f"adafactor scheduler must be used with Adafactor optimizer / adafactor schedulerはAdafactorオプティマイザと同時に使ってください"
            initial_lr = float(name.split(":")[1])
            return wrap_check_needless_num_warmup_steps(transformers.optimization.AdafactorSchedule(optimizer, initial_lr))

        if name == DiffusersSchedulerType.PIECEWISE_CONSTANT.value:
            name = DiffusersSchedulerType(name)
            schedule_func = DIFFUSERS_TYPE_TO_SCHEDULER_FUNCTION[name]
            return schedule_func(optimizer, **lr_scheduler_kwargs)

        name = SchedulerType(name)
        schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]

        if name == SchedulerType.CONSTANT:
            return wrap_check_needless_num_warmup_steps(schedule_func(optimizer, **lr_scheduler_kwargs))

        if num_warmup_steps is None:
            raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

        if name == SchedulerType.CONSTANT_WITH_WARMUP:
            return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, **lr_scheduler_kwargs)

        if name == SchedulerType.INVERSE_SQRT:
            return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, timescale=timescale, **lr_scheduler_kwargs)

        if num_training_steps is None:
            raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

        if name == SchedulerType.COSINE_WITH_RESTARTS:
            return schedule_func(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=num_cycles,
                **lr_scheduler_kwargs,
            )

        if name == SchedulerType.POLYNOMIAL:
            return schedule_func(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                power=power,
                **lr_scheduler_kwargs,
            )

        if name == SchedulerType.COSINE_WITH_MIN_LR:
            return schedule_func(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=num_cycles / 2,
                min_lr_rate=min_lr_ratio,
                **lr_scheduler_kwargs,
            )

        if name == SchedulerType.LINEAR or name == SchedulerType.COSINE:
            return schedule_func(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                **lr_scheduler_kwargs,
            )

        if num_decay_steps is None:
            raise ValueError(f"{name} requires `num_decay_steps`, please provide that argument.")
        if name == SchedulerType.WARMUP_STABLE_DECAY:
            return schedule_func(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_stable_steps=num_stable_steps,
                num_decay_steps=num_decay_steps,
                num_cycles=num_cycles / 2,
                min_lr_ratio=min_lr_ratio if min_lr_ratio is not None else 0.0,
                **lr_scheduler_kwargs,
            )

        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_decay_steps=num_decay_stops,
            **lr_scheduler_kwargs,
        )

    def resume_from_local_or_hf_if_specified(self, accelerator: Accelerator, args: argparse.Namespace) -> bool:
        if not args.resume:
            return False

        if not args.resume_from_huggingface:
            logger.info(f"resume training from local state: {args.resume}")
            accelerator.load_state(args.resume)
            return True

        logger.info(f"resume training from huggingface state: {args.resume}")
        repo_id = args.resume.split("/")[0] + "/" + args.resume.split("/")[1]
        path_in_repo = "/".join(args.resume.split("/")[2:])
        revision = None
        repo_type = None
        if ":" in path_in_repo:
            divided = path_in_repo.split(":")
            if len(divided) == 2:
                path_in_repo, revision = divided
                repo_type = "model"
            else:
                path_in_repo, revision, repo_type = divided
        logger.info(f"Downloading state from huggingface: {repo_id}/{path_in_repo}@{revision}")

        list_files = huggingface_utils.list_dir(
            repo_id=repo_id,
            subfolder=path_in_repo,
            revision=revision,
            token=args.huggingface_token,
            repo_type=repo_type,
        )

        async def download(filename) -> str:
            def task():
                return huggingface_hub.hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    revision=revision,
                    repo_type=repo_type,
                    token=args.huggingface_token,
                )

            return await asyncio.get_event_loop().run_in_executor(None, task)

        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(asyncio.gather(*[download(filename=filename.rfilename) for filename in list_files]))
        if len(results) == 0:
            raise ValueError(
                "No files found in the specified repo id/path/revision / 指定されたリポジトリID/パス/リビジョンにファイルが見つかりませんでした"
            )
        dirname = os.path.dirname(results[0])
        accelerator.load_state(dirname)

        return True

    def get_noisy_model_input_and_timesteps(
        self,
        args: argparse.Namespace,
        noise: torch.Tensor,
        latents: torch.Tensor,
        noise_scheduler: FlowMatchDiscreteScheduler,
        device: torch.device,
        dtype: torch.dtype,
    ):
        batch_size = noise.shape[0]

        if args.timestep_sampling == "uniform" or args.timestep_sampling == "sigmoid" or args.timestep_sampling == "shift":
            if args.timestep_sampling == "uniform" or args.timestep_sampling == "sigmoid":
                if args.timestep_sampling == "sigmoid":
                    t = torch.sigmoid(args.sigmoid_scale * torch.randn((batch_size,), device=device))
                else:
                    t = torch.rand((batch_size,), device=device)

            elif args.timestep_sampling == "shift":
                shift = args.discrete_flow_shift
                logits_norm = torch.randn(batch_size, device=device)
                logits_norm = logits_norm * args.sigmoid_scale
                t = logits_norm.sigmoid()
                t = (t * shift) / (1 + (shift - 1) * t)

            t_min = args.min_timestep if args.min_timestep is not None else 0
            t_max = args.max_timestep if args.max_timestep is not None else 1000.0
            t_min /= 1000.0
            t_max /= 1000.0
            t = t * (t_max - t_min) + t_min

            timesteps = t * 1000.0
            t = t.view(-1, 1, 1, 1, 1)
            noisy_model_input = (1 - t) * latents + t * noise

            timesteps += 1
        else:
            u = compute_density_for_timestep_sampling(
                weighting_scheme=args.weighting_scheme,
                batch_size=batch_size,
                logit_mean=args.logit_mean,
                logit_std=args.logit_std,
                mode_scale=args.mode_scale,
            )
            t_min = args.min_timestep if args.min_timestep is not None else 0
            t_max = args.max_timestep if args.max_timestep is not None else 1000
            indices = (u * (t_max - t_min) + t_min).long()

            timesteps = noise_scheduler.timesteps[indices].to(device=device)

            sigmas = get_sigmas(noise_scheduler, timesteps, device, n_dim=latents.ndim, dtype=dtype)
            noisy_model_input = sigmas * noise + (1.0 - sigmas) * latents

        return noisy_model_input, timesteps

    def show_timesteps(self, args: argparse.Namespace):
        N_TRY = 100000
        BATCH_SIZE = 1000
        CONSOLE_WIDTH = 64
        N_TIMESTEPS_PER_LINE = 25

        noise_scheduler = FlowMatchDiscreteScheduler(shift=args.discrete_flow_shift, reverse=True, solver="euler")

        latents = torch.zeros(BATCH_SIZE, 1, 1, 1, 1, dtype=torch.float16)
        noise = torch.ones_like(latents)

        sampled_timesteps = [0] * noise_scheduler.config.num_train_timesteps
        for i in tqdm(range(N_TRY // BATCH_SIZE)):
            actual_timesteps, _ = self.get_noisy_model_input_and_timesteps(
                args, noise, latents, noise_scheduler, "cpu", torch.float16
            )
            actual_timesteps = actual_timesteps[:, 0, 0, 0, 0] * 1000
            for t in actual_timesteps:
                t = int(t.item())
                sampled_timesteps[t] += 1

        sampled_weighting = [0] * noise_scheduler.config.num_train_timesteps
        for i in tqdm(range(len(sampled_weighting))):
            timesteps = torch.tensor([i + 1], device="cpu")
            weighting = compute_loss_weighting_for_sd3(args.weighting_scheme, noise_scheduler, timesteps, "cpu", torch.float16)
            if weighting is None:
                weighting = torch.tensor(1.0, device="cpu")
            elif torch.isinf(weighting).any():
                weighting = torch.tensor(1.0, device="cpu")
            sampled_weighting[i] = weighting.item()

        if args.show_timesteps == "image":
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.bar(range(len(sampled_timesteps)), sampled_timesteps, width=1.0)
            plt.title("Sampled timesteps")
            plt.xlabel("Timestep")
            plt.ylabel("Count")

            plt.subplot(1, 2, 2)
            plt.bar(range(len(sampled_weighting)), sampled_weighting, width=1.0)
            plt.title("Sampled loss weighting")
            plt.xlabel("Timestep")
            plt.ylabel("Weighting")

            plt.tight_layout()
            plt.show()

        else:
            sampled_timesteps = np.array(sampled_timesteps)
            sampled_weighting = np.array(sampled_weighting)

            sampled_timesteps = sampled_timesteps.reshape(-1, N_TIMESTEPS_PER_LINE).mean(axis=1)
            sampled_weighting = sampled_weighting.reshape(-1, N_TIMESTEPS_PER_LINE).mean(axis=1)

            max_count = max(sampled_timesteps)
            print(f"Sampled timesteps: max count={max_count}")
            for i, t in enumerate(sampled_timesteps):
                line = f"{(i)*N_TIMESTEPS_PER_LINE:4d}-{(i+1)*N_TIMESTEPS_PER_LINE-1:4d}: "
                line += "#" * int(t / max_count * CONSOLE_WIDTH)
                print(line)

            max_weighting = max(sampled_weighting)
            print(f"Sampled loss weighting: max weighting={max_weighting}")
            for i, w in enumerate(sampled_weighting):
                line = f"{i*N_TIMESTEPS_PER_LINE:4d}-{(i+1)*N_TIMESTEPS_PER_LINE-1:4d}: {w:8.2f} "
                line += "#" * int(w / max_weighting * CONSOLE_WIDTH)
                print(line)

    def train(self, args):
        accelerator = prepare_accelerator(args)
        
        # Define the loading device and data type
        loading_device = accelerator.device
        dit_weight_dtype = torch.bfloat16 if args.dit_dtype is None else model_utils.str_to_dtype(args.dit_dtype)
        
        if args.sdpa:
            attn_mode = "sdpa"
        elif args.flash_attn:
            attn_mode = "flash_attn"
        elif args.sage_attn:
            attn_mode = "sage_attn"
        elif args.xformers:
            attn_mode = "xformers"
        
        if args.dataset_config is None:
            raise ValueError("dataset_config is required / dataset_configが必要です")
        if args.dit is None:
            raise ValueError("path to DiT model is required / DiTモデルのパスが必要です")

        control_type = next(t for t in CONTROL_TYPES if getattr(args, t))
        args.control_type = control_type
        logger.info(f"Initializing training with {control_type} control conditioning")
        
        if not hasattr(args, 'cache_directory'):
            raise ValueError("cache_directory is not set. Ensure it is defined in dataset.toml.")
        logger.info(f"cache_directory: {args.cache_directory}")

        # Define the cache and conditioning cache directories
        config = {
            'cache_dir': args.cache_directory,
            'conditioning_cache_dir': args.conditioning_cache_directory
        }
        
        # Initialize the dataset
        train_dataset_group = ConditionedVideoDataset(config, control_type=args.control_type)
        
        # Initialize shared variables for epoch and step
        current_epoch = Value("i", 0)
        current_step = Value("i", 0)
        
        # Initialize the collator
        collator = collator_class(current_epoch=Value("i", 0), current_step=Value("i", 0), dataset=train_dataset_group)
        
        # Debug: Print the number of pairs found
        print(f"Number of valid pairs in dataset: {len(train_dataset_group)}")
        
        # Create the DataLoader
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset_group,
            batch_size=1,
            shuffle=True,
            collate_fn=lambda x: x[0],
            num_workers=0,
            persistent_workers=False
        )

        session_id = random.randint(0, 2**32)
        training_started_at = time.time()
        set_seed(args.seed)

        current_epoch = Value("i", 0)
        current_step = Value("i", 0)
        ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
        collator = collator_class(current_epoch, current_step, ds_for_collator)

        logger.info("preparing accelerator")
        accelerator = prepare_accelerator(args)
        is_main_process = accelerator.is_main_process

        epoch_to_start = 0
        global_step = 0
        
        if args.dit_dtype is None:
            default_dtype = torch.bfloat16
        else:
            default_dtype = model_utils.str_to_dtype(args.dit_dtype)

        dit_dtype = default_dtype
        dit_weight_dtype = dit_dtype

        logger.info(f"Loading DiT model from {args.dit} with {control_type} conditioning")
        transformer = load_transformer(
            dit_path=args.dit,
            attn_mode=attn_mode,
            split_attn=args.split_attn,
            device=loading_device,
            dtype=dit_weight_dtype,
            use_nf4=args.nf4,  # Pass the --nf4 flag
        )
        
        # Create ControlNetAdapter instance
        control_adapter = ControlNetAdapter(
            control_type=control_type,
            dtype=dit_weight_dtype,
            device=accelerator.device
        )
        
        # Pass the adapter instance to DiffusionTransformerWithControl
        transformer = DiffusionTransformerWithControl(transformer, control_adapter)
        
        if hasattr(transformer.img_in, "flatten"):
            transformer.img_in.flatten = False
        transformer.eval()

        blocks_to_swap = args.blocks_to_swap
        
        if blocks_to_swap > 0:
            logger.info(f"enable swap {blocks_to_swap} blocks to CPU from device: {accelerator.device}")
            transformer.enable_block_swap(blocks_to_swap, accelerator.device, supports_backward=True)  # CHANGE: Set to True
            transformer.move_to_device_except_swap_blocks(accelerator.device)
            transformer.prepare_block_swap_before_forward()  # CHANGE: Add this line
        else:
            logger.info(f"Moving and casting model to {accelerator.device} and {dit_weight_dtype}")
            transformer.to(device=accelerator.device)

        sys.path.append(os.path.dirname(__file__))
        accelerator.print("import network module:", args.network_module)
        network_module = importlib.import_module(args.network_module)
        
        # Initialize the network
        if hasattr(network_module, 'create_network_hunyuan_video'):
            network = network_module.create_network_hunyuan_video(
                1.0,
                args.network_dim,
                args.network_alpha,
                vae=None,
                text_encoders=None,
                unet=transformer,
                neuron_dropout=args.network_dropout,
            )
        else:
            network = network_module.create_network(
                1.0,
                args.network_dim,
                args.network_alpha,
                vae=None,
                text_encoders=None,
                unet=transformer,
            )
        
        
        # Enable gradient checkpointing
        if args.gradient_checkpointing:  # CHANGE: Add this block
            transformer.enable_gradient_checkpointing()
            network.enable_gradient_checkpointing()
        
        # num workers for data loader: if 0, persistent_workers is not available
        n_workers = min(args.max_data_loader_n_workers, os.cpu_count())
        
        logger.info(f"Dataset length: {len(train_dataset_group)}")
        
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset_group,
            batch_size=1,
            shuffle=True,
            collate_fn=collator,
            num_workers=n_workers,
            persistent_workers=args.persistent_data_loader_workers,
        )

        # calculate max_train_steps
        if args.max_train_epochs is not None:
            args.max_train_steps = args.max_train_epochs * math.ceil(
                len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
            )
            accelerator.print(
                f"override steps. steps for {args.max_train_epochs} epochs is / 指定エポックまでのステップ数: {args.max_train_steps}"
            )
        
        # epoch数を計算する
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        
        noise_scheduler = FlowMatchDiscreteScheduler(
            shift=args.discrete_flow_shift,
            reverse=True,
            solver="euler"
        )

        for epoch in range(epoch_to_start, num_train_epochs):
            accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}")
            current_epoch.value = epoch + 1

            for step, batch in enumerate(train_dataloader):
                orig_latents = batch['original'].to(accelerator.device)
                cond_latents = batch['conditioned'].to(accelerator.device)
                bsz = orig_latents.shape[0]
                current_step.value = global_step
                
                print(f"Original latents shape: {orig_latents.shape}")
                print(f"Conditioned latents shape: {cond_latents.shape}")

                # Generate noise tensor
                noise = torch.randn_like(cond_latents)
                
                # Define current_pose_alpha
                current_pose_alpha = args.pose_alpha if hasattr(args, 'pose_alpha') else 1.0
                
                # Define llm_mask
                llm_mask = batch.get('llm_mask', None)
                
                # Generate rotary positional embeddings
                freqs_cos, freqs_sin = get_rotary_pos_embed_by_shape(
                    transformer.transformer, cond_latents.shape[2:]
                )
                freqs_cos = freqs_cos.to(accelerator.device, dtype=dit_dtype)
                freqs_sin = freqs_sin.to(accelerator.device, dtype=dit_dtype)

                with accelerator.accumulate(transformer):
                    noisy_model_input, timesteps = self.get_noisy_model_input_and_timesteps(
                        args, noise, cond_latents, noise_scheduler, accelerator.device, dit_dtype
                    )
                    
                    # Explicitly cast tensors to dit_dtype
                    noisy_model_input = noisy_model_input.to(dit_dtype)  # CHANGE: Add this line
                    cond_latents = cond_latents.to(dit_dtype)  # CHANGE: Add this line
                    guidance_vec = torch.tensor([args.guidance_scale], dtype=dit_dtype, device=accelerator.device)  # CHANGE: Add this line

                    with accelerator.autocast():
                        model_pred = transformer(
                            x=noisy_model_input,
                            t=timesteps,
                            text_states=None,
                            control_input=cond_latents,
                            pose_alpha=current_pose_alpha,
                            control_alpha=1.0,
                            text_mask=llm_mask,
                            text_states_2=None,
                            freqs_cos=freqs_cos,
                            freqs_sin=freqs_sin,
                            guidance=guidance_vec,
                            return_dict=False,
                        )

                    # Free memory immediately
                    del orig_latents, cond_latents, batch  # CHANGE: Add this line
                    torch.cuda.empty_cache()  # CHANGE: Add this line
                    
                    # Compute loss
                    target = (noise - orig_latents).to(network_dtype)
                    loss = torch.nn.functional.mse_loss(model_pred.to(network_dtype), target, reduction="none")
                    loss = loss.mean()

                    accelerator.backward(loss)
                    if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                        params_to_clip = accelerator.unwrap_model(network).get_trainable_params()
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

            accelerator.wait_for_everyone()
            if args.save_every_n_epochs is not None:
                saving = (epoch + 1) % args.save_every_n_epochs == 0 and (epoch + 1) < num_train_epochs
                if is_main_process and saving:
                    ckpt_name = train_utils.get_epoch_ckpt_name(args.output_name, epoch + 1)
                    save_model(ckpt_name, accelerator.unwrap_model(network), global_step, epoch + 1)

        accelerator.end_training()
        if is_main_process:
            ckpt_name = train_utils.get_last_ckpt_name(args.output_name)
            save_model(ckpt_name, network, global_step, num_train_epochs, force_sync_upload=True)
            logger.info("model saved.")
            
def setup_parser():
    parser = argparse.ArgumentParser()
    
    def int_or_float(value):
        try:
            return int(value)
        except ValueError:
            return float(value)
    
    # Control type flags
    control_group = parser.add_mutually_exclusive_group(required=True)
    control_group.add_argument("--pose", action="store_true", help="Use pose control")
    control_group.add_argument("--depth", action="store_true", help="Use depth control")
    control_group.add_argument("--canny", action="store_true", help="Use canny edge control")
                             
    
    # general settings
    parser.add_argument(
        "--dataset_config",
        type=pathlib.Path,
        default=None,
        help="config file for dataset / データセットの設定ファイル",
    )
    # Add the --nf4 flag
    parser.add_argument(
        "--nf4",
        action="store_true",
        help="Enable NF4 quantization for the model to reduce memory usage.",
    )

    # training settings
    parser.add_argument(
        "--sdpa",
        action="store_true",
        help="use sdpa for CrossAttention (requires PyTorch 2.0) / CrossAttentionにsdpaを使う（PyTorch 2.0が必要）",
    )
    parser.add_argument(
        "--flash_attn",
        action="store_true",
        help="use FlashAttention for CrossAttention, requires FlashAttention / CrossAttentionにFlashAttentionを使う、FlashAttentionが必要",
    )
    parser.add_argument(
        "--sage_attn",
        action="store_true",
        help="use SageAttention. requires SageAttention / SageAttentionを使う。SageAttentionが必要",
    )
    parser.add_argument(
        "--xformers",
        action="store_true",
        help="use xformers for CrossAttention, requires xformers / CrossAttentionにxformersを使う、xformersが必要",
    )
    parser.add_argument(
        "--split_attn",
        action="store_true",
        help="use split attention for attention calculation (split batch size=1, affects memory usage and speed)"
        " / attentionを分割して計算する（バッチサイズ=1に分割、メモリ使用量と速度に影響）",
    )

    parser.add_argument("--max_train_steps", type=int, default=1600, help="training steps / 学習ステップ数")
    parser.add_argument(
        "--max_train_epochs",
        type=int,
        default=None,
        help="training epochs (overrides max_train_steps) / 学習エポック数（max_train_stepsを上書きします）",
    )
    parser.add_argument(
        "--max_data_loader_n_workers",
        type=int,
        default=8,
        help="max num workers for DataLoader (lower is less main RAM usage, faster epoch start and slower data loading) / DataLoaderの最大プロセス数（小さい値ではメインメモリの使用量が減りエポック間の待ち時間が減りますが、データ読み込みは遅くなります）",
    )
    parser.add_argument(
        "--persistent_data_loader_workers",
        action="store_true",
        help="persistent DataLoader workers (useful for reduce time gap between epoch, but may use more memory) / DataLoader のワーカーを持続させる (エポック間の時間差を少なくするのに有効だが、より多くのメモリを消費する可能性がある)",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed for training / 学習時の乱数のseed")
    parser.add_argument(
        "--gradient_checkpointing", action="store_true", help="enable gradient checkpointing / gradient checkpointingを有効にする"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass / 学習時に逆伝播をする前に勾配を合計するステップ数",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="use mixed precision / 混合精度を使う場合、その精度",
    )

    parser.add_argument(
        "--logging_dir",
        type=str,
        default=None,
        help="enable logging and output TensorBoard log to this directory / ログ出力を有効にしてこのディレクトリにTensorBoard用のログを出力する",
    )
    parser.add_argument(
        "--log_with",
        type=str,
        default=None,
        choices=["tensorboard", "wandb", "all"],
        help="what logging tool(s) to use (if 'all', TensorBoard and WandB are both used) / ログ出力に使用するツール (allを指定するとTensorBoardとWandBの両方が使用される)",
    )
    parser.add_argument(
        "--log_prefix", type=str, default=None, help="add prefix for each log directory / ログディレクトリ名の先頭に追加する文字列"
    )
    parser.add_argument(
        "--log_tracker_name",
        type=str,
        default=None,
        help="name of tracker to use for logging, default is script-specific default name / ログ出力に使用するtrackerの名前、省略時はスクリプトごとのデフォルト名",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="The name of the specific wandb session / wandb ログに表示される特定の実行の名前",
    )
    parser.add_argument(
        "--log_tracker_config",
        type=str,
        default=None,
        help="path to tracker config file to use for logging / ログ出力に使用するtrackerの設定ファイルのパス",
    )
    parser.add_argument(
        "--wandb_api_key",
        type=str,
        default=None,
        help="specify WandB API key to log in before starting training (optional). / WandB APIキーを指定して学習開始前にログインする（オプション）",
    )
    parser.add_argument("--log_config", action="store_true", help="log training configuration / 学習設定をログに出力する")

    parser.add_argument(
        "--ddp_timeout",
        type=int,
        default=None,
        help="DDP timeout (min, None for default of accelerate) / DDPのタイムアウト（分、Noneでaccelerateのデフォルト）",
    )
    parser.add_argument(
        "--ddp_gradient_as_bucket_view",
        action="store_true",
        help="enable gradient_as_bucket_view for DDP / DDPでgradient_as_bucket_viewを有効にする",
    )
    parser.add_argument(
        "--ddp_static_graph",
        action="store_true",
        help="enable static_graph for DDP / DDPでstatic_graphを有効にする",
    )

    parser.add_argument(
        "--sample_every_n_steps",
        type=int,
        default=None,
        help="generate sample images every N steps / 学習中のモデルで指定ステップごとにサンプル出力する",
    )
    parser.add_argument(
        "--sample_at_first", action="store_true", help="generate sample images before training / 学習前にサンプル出力する"
    )
    parser.add_argument(
        "--sample_every_n_epochs",
        type=int,
        default=None,
        help="generate sample images every N epochs (overwrites n_steps) / 学習中のモデルで指定エポックごとにサンプル出力する（ステップ数指定を上書きします）",
    )
    parser.add_argument(
        "--sample_prompts",
        type=str,
        default=None,
        help="file for prompts to generate sample images / 学習中モデルのサンプル出力用プロンプトのファイル",
    )

    # optimizer and lr scheduler settings
    parser.add_argument(
        "--optimizer_type",
        type=str,
        default="",
        help="Optimizer to use / オプティマイザの種類: AdamW (default), AdamW8bit, AdaFactor. "
        "Also, you can use any optimizer by specifying the full path to the class, like 'torch.optim.AdamW', 'bitsandbytes.optim.AdEMAMix8bit' or 'bitsandbytes.optim.PagedAdEMAMix8bit' etc. / ",
    )
    parser.add_argument(
        "--optimizer_args",
        type=str,
        default=None,
        nargs="*",
        help='additional arguments for optimizer (like "weight_decay=0.01 betas=0.9,0.999 ...") / オプティマイザの追加引数（例： "weight_decay=0.01 betas=0.9,0.999 ..."）',
    )
    parser.add_argument("--learning_rate", type=float, default=2.0e-6, help="learning rate / 学習率")
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm, 0 for no clipping / 勾配正規化の最大norm、0でclippingを行わない",
    )

    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help="scheduler to use for learning rate / 学習率のスケジューラ: linear, cosine, cosine_with_restarts, polynomial, constant (default), constant_with_warmup, adafactor",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int_or_float,
        default=0,
        help="Int number of steps for the warmup in the lr scheduler (default is 0) or float with ratio of train steps"
        " / 学習率のスケジューラをウォームアップするステップ数（デフォルト0）、または学習ステップの比率（1未満のfloat値の場合）",
    )
    parser.add_argument(
        "--lr_decay_steps",
        type=int_or_float,
        default=0,
        help="Int number of steps for the decay in the lr scheduler (default is 0) or float (<1) with ratio of train steps"
        " / 学習率のスケジューラを減衰させるステップ数（デフォルト0）、または学習ステップの比率（1未満のfloat値の場合）",
    )
    parser.add_argument(
        "--lr_scheduler_num_cycles",
        type=int,
        default=1,
        help="Number of restarts for cosine scheduler with restarts / cosine with restartsスケジューラでのリスタート回数",
    )
    parser.add_argument(
        "--lr_scheduler_power",
        type=float,
        default=1,
        help="Polynomial power for polynomial scheduler / polynomialスケジューラでのpolynomial power",
    )
    parser.add_argument(
        "--lr_scheduler_timescale",
        type=int,
        default=None,
        help="Inverse sqrt timescale for inverse sqrt scheduler,defaults to `num_warmup_steps`"
        + " / 逆平方根スケジューラのタイムスケール、デフォルトは`num_warmup_steps`",
    )
    parser.add_argument(
        "--lr_scheduler_min_lr_ratio",
        type=float,
        default=None,
        help="The minimum learning rate as a ratio of the initial learning rate for cosine with min lr scheduler and warmup decay scheduler"
        + " / 初期学習率の比率としての最小学習率を指定する、cosine with min lr と warmup decay スケジューラ で有効",
    )
    parser.add_argument("--lr_scheduler_type", type=str, default="", help="custom scheduler module / 使用するスケジューラ")
    parser.add_argument(
        "--lr_scheduler_args",
        type=str,
        default=None,
        nargs="*",
        help='additional arguments for scheduler (like "T_max=100") / スケジューラの追加引数（例： "T_max100"）',
    )

    # model settings
    parser.add_argument("--dit", type=str, help="DiT checkpoint path / DiTのチェックポイントのパス")
    parser.add_argument("--dit_dtype", type=str, default=None, help="data type for DiT, default is bfloat16")
    parser.add_argument("--vae", type=str, help="VAE checkpoint path / VAEのチェックポイントのパス")
    parser.add_argument("--vae_dtype", type=str, default=None, help="data type for VAE, default is float16")
    parser.add_argument(
        "--vae_tiling",
        action="store_true",
        help="enable spatial tiling for VAE, default is False. If vae_spatial_tile_sample_min_size is set, this is automatically enabled."
        " / VAEの空間タイリングを有効にする、デフォルトはFalse。vae_spatial_tile_sample_min_sizeが設定されている場合、自動的に有効になります。",
    )
    parser.add_argument("--vae_chunk_size", type=int, default=None, help="chunk size for CausalConv3d in VAE")
    parser.add_argument(
        "--vae_spatial_tile_sample_min_size", type=int, default=None, help="spatial tile sample min size for VAE, default 256"
    )
    parser.add_argument("--text_encoder1", type=str, help="Text Encoder 1 directory / テキストエンコーダ1のディレクトリ")
    parser.add_argument("--text_encoder2", type=str, help="Text Encoder 2 directory / テキストエンコーダ2のディレクトリ")
    parser.add_argument("--text_encoder_dtype", type=str, default=None, help="data type for Text Encector, default is float16")
    parser.add_argument("--fp8_llm", action="store_true", help="use fp8 for LLM / LLMにfp8を使う")
    parser.add_argument("--fp8_base", action="store_true", help="use fp8 for base model / base modelにfp8を使う")
    # parser.add_argument("--full_fp16", action="store_true", help="fp16 training including gradients / 勾配も含めてfp16で学習する")
    # parser.add_argument("--full_bf16", action="store_true", help="bf16 training including gradients / 勾配も含めてbf16で学習する")

    parser.add_argument(
        "--blocks_to_swap",
        type=int,
        default=0,
        help="number of blocks to swap in the model, max XXX / モデル内のブロックの数、最大XXX",
    )
    parser.add_argument(
        "--img_in_txt_in_offloading",
        action="store_true",
        help="offload img_in and txt_in to cpu / img_inとtxt_inをCPUにオフロードする",
    )

    # parser.add_argument("--flow_shift", type=float, default=7.0, help="Shift factor for flow matching schedulers")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="Embeded classifier free guidance scale.")
    parser.add_argument(
        "--timestep_sampling",
        choices=["sigma", "uniform", "sigmoid", "shift"],
        default="sigma",
        help="Method to sample timesteps: sigma-based, uniform random, sigmoid of random normal and shift of sigmoid."
        " / タイムステップをサンプリングする方法：sigma、random uniform、random normalのsigmoid、sigmoidのシフト。",
    )
    parser.add_argument(
        "--discrete_flow_shift",
        type=float,
        default=1.0,
        help="Discrete flow shift for the Euler Discrete Scheduler, default is 1.0. / Euler Discrete Schedulerの離散フローシフト、デフォルトは1.0。",
    )
    parser.add_argument(
        "--sigmoid_scale",
        type=float,
        default=1.0,
        help='Scale factor for sigmoid timestep sampling (only used when timestep-sampling is "sigmoid" or "shift"). / sigmoidタイムステップサンプリングの倍率（timestep-samplingが"sigmoid"または"shift"の場合のみ有効）。',
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["logit_normal", "mode", "cosmap", "sigma_sqrt", "none"],
        help="weighting scheme for timestep distribution. Default is none"
        " / タイムステップ分布の重み付けスキーム、デフォルトはnone",
    )
    parser.add_argument(
        "--logit_mean",
        type=float,
        default=0.0,
        help="mean to use when using the `'logit_normal'` weighting scheme / `'logit_normal'`重み付けスキームを使用する場合の平均",
    )
    parser.add_argument(
        "--logit_std",
        type=float,
        default=1.0,
        help="std to use when using the `'logit_normal'` weighting scheme / `'logit_normal'`重み付けスキームを使用する場合のstd",
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme` / モード重み付けスキームのスケール",
    )
    parser.add_argument(
        "--min_timestep",
        type=int,
        default=None,
        help="set minimum time step for training (0~999, default is 0) / 学習時のtime stepの最小値を設定する（0~999で指定、省略時はデフォルト値(0)） ",
    )
    parser.add_argument(
        "--max_timestep",
        type=int,
        default=None,
        help="set maximum time step for training (1~1000, default is 1000) / 学習時のtime stepの最大値を設定する（1~1000で指定、省略時はデフォルト値(1000)）",
    )

    parser.add_argument(
        "--show_timesteps",
        type=str,
        default=None,
        choices=["image", "console"],
        help="show timesteps in image or console, and return to console / タイムステップを画像またはコンソールに表示し、コンソールに戻る",
    )

    # network settings
    parser.add_argument(
        "--no_metadata", action="store_true", help="do not save metadata in output model / メタデータを出力先モデルに保存しない"
    )
    parser.add_argument(
        "--network_weights", type=str, default=None, help="pretrained weights for network / 学習するネットワークの初期重み"
    )
    parser.add_argument(
        "--network_module", type=str, default=None, help="network module to train / 学習対象のネットワークのモジュール"
    )
    parser.add_argument(
        "--network_dim",
        type=int,
        default=None,
        help="network dimensions (depends on each network) / モジュールの次元数（ネットワークにより定義は異なります）",
    )
    parser.add_argument(
        "--network_alpha",
        type=float,
        default=1,
        help="alpha for LoRA weight scaling, default 1 (same as network_dim for same behavior as old version) / LoRaの重み調整のalpha値、デフォルト1（旧バージョンと同じ動作をするにはnetwork_dimと同じ値を指定）",
    )
    parser.add_argument(
        "--network_dropout",
        type=float,
        default=None,
        help="Drops neurons out of training every step (0 or None is default behavior (no dropout), 1 would drop all neurons) / 訓練時に毎ステップでニューロンをdropする（0またはNoneはdropoutなし、1は全ニューロンをdropout）",
    )
    parser.add_argument(
        "--network_args",
        type=str,
        default=None,
        nargs="*",
        help="additional arguments for network (key=value) / ネットワークへの追加の引数",
    )
    parser.add_argument(
        "--training_comment",
        type=str,
        default=None,
        help="arbitrary comment string stored in metadata / メタデータに記録する任意のコメント文字列",
    )
    parser.add_argument(
        "--dim_from_weights",
        action="store_true",
        help="automatically determine dim (rank) from network_weights / dim (rank)をnetwork_weightsで指定した重みから自動で決定する",
    )
    parser.add_argument(
        "--scale_weight_norms",
        type=float,
        default=None,
        help="Scale the weight of each key pair to help prevent overtraing via exploding gradients. (1 is a good starting point) / 重みの値をスケーリングして勾配爆発を防ぐ（1が初期値としては適当）",
    )
    parser.add_argument(
        "--base_weights",
        type=str,
        default=None,
        nargs="*",
        help="network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みファイル",
    )
    parser.add_argument(
        "--base_weights_multiplier",
        type=float,
        default=None,
        nargs="*",
        help="multiplier for network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みの倍率",
    )

    # save and load settings
    parser.add_argument(
        "--output_dir", type=str, default=None, help="directory to output trained model / 学習後のモデル出力先ディレクトリ"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="base name of trained model file / 学習後のモデルの拡張子を除くファイル名",
    )
    parser.add_argument("--resume", type=str, default=None, help="saved state to resume training / 学習再開するモデルのstate")

    parser.add_argument(
        "--save_eevery_n_epochs",
        type=int,
        default=None,
        help="save checkpoint every N epochs / 学習中のモデルを指定エポックごとに保存する",
    )
    parser.add_argument(
        "--save_every_n_steps",
        type=int,
        default=None,
        help="save checkpoint every N steps / 学習中のモデルを指定ステップごとに保存する",
    )
    parser.add_argument(
        "--save_last_n_epochs",
        type=int,
        default=None,
        help="save last N checkpoints when saving every N epochs (remove older checkpoints) / 指定エポックごとにモデルを保存するとき最大Nエポック保存する（古いチェックポイントは削除する）",
    )
    parser.add_argument(
        "--save_last_n_epochs_state",
        type=int,
        default=None,
        help="save last N checkpoints of state (overrides the value of --save_last_n_epochs)/ 最大Nエポックstateを保存する（--save_last_n_epochsの指定を上書きする）",
    )
    parser.add_argument(
        "--save_last_n_steps",
        type=int,
        default=None,
        help="save checkpoints until N steps elapsed (remove older checkpoints if N steps elapsed) / 指定ステップごとにモデルを保存するとき、このステップ数経過するまで保存する（このステップ数経過したら削除する）",
    )
    parser.add_argument(
        "--save_last_n_steps_state",
        type=int,
        default=None,
        help="save states until N steps elapsed (remove older states if N steps elapsed, overrides --save_last_n_steps) / 指定ステップごとにstateを保存するとき、このステップ数経過するまで保存する（このステップ数経過したら削除する。--save_last_n_stepsを上書きする）",
    )
    parser.add_argument(
        "--save_state",
        action="store_true",
        help="save training state additionally (including optimizer states etc.) when saving model / optimizerなど学習状態も含めたstateをモデル保存時に追加で保存する",
    )
    parser.add_argument(
        "--save_state_on_train_end",
        action="store_true",
        help="save training state (including optimizer states etc.) on train end even if --save_state is not specified"
        " / --save_stateが未指定時にもoptimizerなど学習状態も含めたstateを学習終了時に保存する",
    )

    # SAI Model spec
    parser.add_argument(
        "--metadata_title",
        type=str,
        default=None,
        help="title for model metadata (default is output_name) / メタデータに書き込まれるモデルタイトル、省略時はoutput_name",
    )
    parser.add_argument(
        "--metadata_author",
        type=str,
        default=None,
        help="author name for model metadata / メタデータに書き込まれるモデル作者名",
    )
    parser.add_argument(
        "--metadata_description",
        type=str,
        default=None,
        help="description for model metadata / メタデータに書き込まれるモデル説明",
    )
    parser.add_argument(
        "--metadata_license",
        type=str,
        default=None,
        help="license for model metadata / メタデータに書き込まれるモデルライセンス",
    )
    parser.add_argument(
        "--metadata_tags",
        type=str,
        default=None,
        help="tags for model metadata, separated by comma / メタデータに書き込まれるモデルタグ、カンマ区切り",
    )

    # huggingface settings
    parser.add_argument(
        "--huggingface_repo_id",
        type=str,
        default=None,
        help="huggingface repo name to upload / huggingfaceにアップロードするリポジトリ名",
    )
    parser.add_argument(
        "--huggingface_repo_type",
        type=str,
        default=None,
        help="huggingface repo type to upload / huggingfaceにアップロードするリポジトリの種類",
    )
    parser.add_argument(
        "--huggingface_path_in_repo",
        type=str,
        default=None,
        help="huggingface model path to upload files / huggingfaceにアップロードするファイルのパス",
    )
    parser.add_argument("--huggingface_token", type=str, default=None, help="huggingface token / huggingfaceのトークン")
    parser.add_argument(
        "--huggingface_repo_visibility",
        type=str,
        default=None,
        help="huggingface repository visibility ('public' for public, 'private' or None for private) / huggingfaceにアップロードするリポジトリの公開設定（'public'で公開、'private'またはNoneで非公開）",
    )
    parser.add_argument(
        "--save_state_to_huggingface", action="store_true", help="save state to huggingface / huggingfaceにstateを保存する"
    )
    parser.add_argument(
        "--resume_from_huggingface",
        action="store_true",
        help="resume from huggingface (ex: --resume {repo_id}/{path_in_repo}:{revision}:{repo_type}) / huggingfaceから学習を再開する(例: --resume {repo_id}/{path_in_repo}:{revision}:{repo_type})",
    )
    parser.add_argument(
        "--async_upload",
        action="store_true",
        help="upload to huggingface asynchronously / huggingfaceに非同期でアップロードする",
    )
    
    parser.add_argument("--use_pose", action="store_true", help="Enable pose guidance injection")
    parser.add_argument("--pose_alpha", type=float, default=1.0, help="Scaling factor to weight the pose latent injection (default: 1.0)")

    return parser


def read_config_from_file(args: argparse.Namespace, parser: argparse.ArgumentParser):
   # Read dataset.toml and assign cache_directory and conditioning_cache_directory to args
    if hasattr(args, 'dataset_config') and args.dataset_config:
        dataset_config_path = args.dataset_config
        if os.path.exists(dataset_config_path):
            with open(dataset_config_path, "r", encoding="utf-8") as f:
                dataset_config = toml.load(f)
            # Assign cache_directory from the first valid dataset
            for dataset in dataset_config.get('datasets', []):
                if not hasattr(args, 'cache_directory') and 'cache_directory' in dataset:
                    args.cache_directory = dataset['cache_directory']
                    logger.info(f"cache_directory set to {args.cache_directory}")
                if not hasattr(args, 'conditioning_cache_directory') and 'conditioning_cache' in dataset:
                    args.conditioning_cache_directory = dataset['conditioning_cache']
                    logger.info(f"conditioning_cache_directory set to {args.conditioning_cache_directory}")
                if 'general' in dataset_config and 'batch_size' in dataset_config['general']:
                    args.batch_size = dataset_config['general']['batch_size']
                    logger.info(f"batch_size set to {args.batch_size}")
                # Break after processing the first dataset with valid directories
                if hasattr(args, 'cache_directory') and hasattr(args, 'conditioning_cache_directory'):
                    break
        else:
            logger.error(f"Dataset config file not found: {dataset_config_path}")
            raise FileNotFoundError(f"Dataset config file not found: {dataset_config_path}")
    return args



if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    # Debug: Check if cache_directory is set
    if not hasattr(args, 'cache_directory'):
        raise ValueError("cache_directory is not set. Ensure it is defined in dataset.toml.")
    logger.info(f"cache_directory: {args.cache_directory}")

    trainer = NetworkTrainer()
    trainer.train(args)
