#!/usr/bin/env python

"""Utility to convert a folder of videos and prompts into a LMDB dataset.

This script wraps the repository's ``compute_vae_latent.py`` and
``create_lmdb_14b_shards.py`` scripts. It first computes VAE latents for all
videos and then packs them into an LMDB dataset ready for training.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    """Run a command and raise ``CalledProcessError`` if it fails."""
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LMDB dataset from videos")
    parser.add_argument("--video_folder", required=True, help="Path to source videos")
    parser.add_argument("--prompt_folder", required=True, help="Folder containing prompt txt files")
    parser.add_argument("--latent_folder", required=True, help="Folder to store computed latents")
    parser.add_argument("--lmdb_folder", required=True, help="Output LMDB directory")
    parser.add_argument("--model_name", default="Wan2.1-T2V-14B", help="Pretrained model name")
    parser.add_argument("--num_gpus", type=int, default=1, help="GPUs for latent computation")
    parser.add_argument("--num_shards", type=int, default=16, help="Number of LMDB shards")
    args = parser.parse_args()

    Path(args.latent_folder).mkdir(parents=True, exist_ok=True)
    Path(args.lmdb_folder).mkdir(parents=True, exist_ok=True)

    compute_cmd = [
        "torchrun",
        f"--nproc_per_node={args.num_gpus}",
        "scripts/compute_vae_latent.py",
        "--input_video_folder",
        args.video_folder,
        "--output_latent_folder",
        args.latent_folder,
        "--model_name",
        args.model_name,
        "--prompt_folder",
        args.prompt_folder,
    ]
    run(compute_cmd)

    create_cmd = [
        "python",
        "scripts/create_lmdb_14b_shards.py",
        "--data_path",
        args.latent_folder,
        "--prompt_path",
        args.prompt_folder,
        "--video_path",
        args.video_folder,
        "--lmdb_path",
        args.lmdb_folder,
        "--num_shards",
        str(args.num_shards),
    ]
    run(create_cmd)


if __name__ == "__main__":
    main()
