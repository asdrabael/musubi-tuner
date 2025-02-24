import argparse
import os
import glob
import re
from typing import Optional, Union, Tuple
import numpy as np
import cv2
import torch
from tqdm import tqdm
from PIL import Image
import logging
from safetensors.torch import save_file

from dataset import config_utils
from dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from dataset.image_video_dataset import BaseDataset, ItemInfo, save_latent_cache
from hunyuan_model.vae import load_vae
from hunyuan_model.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from utils.model_utils import str_to_dtype

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def encode_and_save_batch(vae: AutoencoderKLCausal3D, batch: list[ItemInfo]):
    contents = torch.stack([torch.from_numpy(item.content) for item in batch])
    if len(contents.shape) == 4:
        contents = contents.unsqueeze(1)
    contents = contents.permute(0, 4, 1, 2, 3).contiguous()
    contents = contents.to(vae.device, dtype=vae.dtype)
    contents = contents / 127.5 - 1.0
    with torch.no_grad():
        latent = vae.encode(contents).latent_dist.sample()
    for item, l in zip(batch, latent):
        # Extract the base name from latent_cache_path and remove dataset suffixes
        latent_cache_basename = os.path.basename(item.latent_cache_path)
        # Use regex to strip suffixes like "_00000-149_0144x0256_hv"
        original_name = re.sub(r'(_\d+-\d+_\d+x\d+).*$', '', latent_cache_basename)
        # Remove existing extension (e.g., .safetensors) and append .safetensors
        original_name = os.path.splitext(original_name)[0] + ".safetensors"
        # Override latent_cache_path to use the clean name
        item.latent_cache_path = os.path.join(os.path.dirname(item.latent_cache_path), original_name)
        # Save with the corrected name
        save_file({"latent": l.cpu()}, item.latent_cache_path)  # <-- Replace save_latent_cache 

def main(args):
    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Load dataset config
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets

    if args.debug_mode is not None:
        show_datasets(datasets, args.debug_mode, args.console_width, args.console_back, args.console_num_images)
        return

    assert args.vae is not None, "vae checkpoint is required"

    # Load VAE model
    vae_dtype = torch.float16 if args.vae_dtype is None else str_to_dtype(args.vae_dtype)
    vae, _, s_ratio, t_ratio = load_vae(vae_dtype=vae_dtype, device=device, vae_path=args.vae)
    vae.eval()
    logger.info(f"Loaded VAE: {vae.config}, dtype: {vae.dtype}")

    if args.vae_chunk_size is not None:
        vae.set_chunk_size_for_causal_conv_3d(args.vae_chunk_size)
        logger.info(f"Set chunk_size to {args.vae_chunk_size} for CausalConv3d in VAE")
    if args.vae_spatial_tile_sample_min_size is not None:
        vae.enable_spatial_tiling(True)
        vae.tile_sample_min_size = args.vae_spatial_tile_sample_min_size
        vae.tile_latent_min_size = args.vae_spatial_tile_sample_min_size // 8
    elif args.vae_tiling:
        vae.enable_spatial_tiling(True)

    # Encode main datasets
    num_workers = args.num_workers if args.num_workers is not None else max(1, os.cpu_count() - 1)
    for i, dataset in enumerate(datasets):
        logger.info(f"Encoding dataset [{i}]")
        all_latent_cache_paths = []
        for _, batch in tqdm(dataset.retrieve_latent_cache_batches(num_workers)):
            # Store ORIGINAL paths before any modifications
            original_paths = [item.latent_cache_path for item in batch]  # <-- NEW LINE
            all_latent_cache_paths.extend(original_paths)  # <-- Use original_paths instead of item.latent_cache_path

            if args.skip_existing:
                filtered_batch = [item for item in batch if not os.path.exists(item.latent_cache_path)]
                if len(filtered_batch) == 0:
                    continue
                batch = filtered_batch

            bs = args.batch_size if args.batch_size is not None else len(batch)
            for i in range(0, len(batch), bs):
                encode_and_save_batch(vae, batch[i : i + bs])

        # Cleanup old caches
        all_latent_cache_paths = [os.path.normpath(p) for p in all_latent_cache_paths]
        all_latent_cache_paths = set(all_latent_cache_paths)
        all_cache_files = dataset.get_all_latent_cache_files()
        for cache_file in all_cache_files:
            normalized = os.path.normpath(cache_file)
            if normalized not in all_latent_cache_paths:  # <-- Now compares against original paths
                if args.keep_cache:
                    logger.info(f"Keep cache file not in dataset: {cache_file}")
                else:
                    os.remove(cache_file)
                    logger.info(f"Removed old cache: {cache_file}")

    # ====== Process conditioning datasets ======
    conditioning_dir = user_config.get('conditioning_directory')
    if conditioning_dir:
        logger.info("Processing conditioning datasets...")
        # Generate blueprint with conditioning paths
        conditioning_blueprint = blueprint_generator.generate_conditioning_blueprint(
            user_config, args, conditioning_dir
        )
        conditioning_dataset_group = config_utils.generate_dataset_group_by_blueprint(
            conditioning_blueprint.dataset_group
        )
        conditioning_datasets = conditioning_dataset_group.datasets

        # Process conditioning datasets
        for i, dataset in enumerate(conditioning_datasets):
            logger.info(f"Encoding conditioning dataset [{i}]")
            all_cond_cache_paths = []
            for _, batch in tqdm(dataset.retrieve_latent_cache_batches(num_workers)):
                all_cond_cache_paths.extend([item.latent_cache_path for item in batch])

                if args.skip_existing:
                    filtered_batch = [item for item in batch if not os.path.exists(item.latent_cache_path)]
                    if len(filtered_batch) == 0:
                        continue
                    batch = filtered_batch

                bs = args.batch_size if args.batch_size is not None else len(batch)
                for j in range(0, len(batch), bs):
                    encode_and_save_batch(vae, batch[j : j + bs])

            # Cleanup old conditioning caches
            all_cond_cache_paths = [os.path.normpath(p) for p in all_cond_cache_paths]
            all_cond_cache_paths = set(all_cond_cache_paths)
            cond_cache_files = dataset.get_all_latent_cache_files()
            for cache_file in cond_cache_files:
                if os.path.normpath(cache_file) not in all_cond_cache_paths:
                    if args.keep_cache:
                        logger.info(f"Keep conditioning cache: {cache_file}")
                    else:
                        os.remove(cache_file)
                        logger.info(f"Removed old conditioning cache: {cache_file}")

def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_config", type=str, required=True, help="path to dataset config .toml file")
    parser.add_argument("--vae", type=str, required=False, default=None, help="path to vae checkpoint")
    parser.add_argument("--vae_dtype", type=str, default=None, help="data type for VAE, default is float16")
    parser.add_argument("--vae_tiling", action="store_true", help="enable spatial tiling for VAE")
    parser.add_argument("--vae_chunk_size", type=int, default=None, help="chunk size for CausalConv3d in VAE")
    parser.add_argument("--vae_spatial_tile_sample_min_size", type=int, default=None, help="spatial tile sample min size for VAE, default 256")
    parser.add_argument("--device", type=str, default=None, help="device to use, default is cuda if available")
    parser.add_argument("--batch_size", type=int, default=None, help="batch size; overrides dataset config if dataset batch size > this")
    parser.add_argument("--num_workers", type=int, default=None, help="number of workers for dataset; default is cpu count-1")
    parser.add_argument("--skip_existing", action="store_true", help="skip existing cache files")
    parser.add_argument("--keep_cache", action="store_true", help="keep cache files not in dataset")
    parser.add_argument("--debug_mode", type=str, default=None, choices=["image", "console"], help="debug mode")
    parser.add_argument("--console_width", type=int, default=80, help="debug mode: console width")
    parser.add_argument("--console_back", type=str, default=None, help="debug mode: console background; choice from ascii_magic.Back")
    parser.add_argument("--console_num_images", type=int, default=None, help="debug mode: number of images to show for each dataset")
    return parser

if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    main(args)
