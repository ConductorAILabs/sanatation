#!/usr/bin/env python3
"""Decode subprocess: load the Sana VAE only, read a Stage-1 latent from
disk, decode it to an MP4, exit.

By skipping the DiT and text encoder, this stage's peak memory is bounded
by the VAE alone. On macOS the parent process should have exited Stage 1
before invoking this, so the previous stage's pages have been released.

Inputs: <in-dir>/stage1.latent.pt + stage1.meta.json
Outputs: <output_dir>/<name>_generated.mp4 (path from meta)

Usage:
    python -m stages.decode --in-dir <path>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
for k in (
    "SANA_WM_FORCE_PURE_PYTORCH", "FLA_USE_PURE_PYTORCH", "DISABLE_XFORMERS",
    "GDN_DISABLE_COMPILE", "SANA_WM_NO_COMPLEX_ROPE", "SANA_WM_RESTORE_GDN",
):
    os.environ.setdefault(k, "1")

import numpy as np
import pyrallis
import torch

REPO = Path(__file__).resolve().parents[1] / "repo"
sys.path.insert(0, str(REPO))

from diffusion.model.builder import get_vae, vae_decode  # noqa: E402
from diffusion.model.utils import get_weight_dtype  # noqa: E402
from inference_video_scripts.inference_sana_wm import (  # noqa: E402
    InferenceConfig,
    _empty_cache,
    _pick_device,
    get_root_logger,
    write_video,
)
from sana.tools import resolve_hf_path  # noqa: E402

HF_REPO = "Efficient-Large-Model/SANA-WM_bidirectional"
DEFAULT_CONFIG = f"hf://{HF_REPO}/config.yaml"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", type=Path, required=True)
    args = ap.parse_args()

    logger = get_root_logger()
    meta_path = args.in_dir / "stage1.meta.json"
    latent_path = args.in_dir / "stage1.latent.pt"
    if not latent_path.exists() or not meta_path.exists():
        raise SystemExit(f"missing stage1 outputs in {args.in_dir}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    output_dir = Path(meta["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    name = meta["name"]
    fps = meta["fps"]

    config_path = meta.get("config", DEFAULT_CONFIG)
    config: InferenceConfig = pyrallis.parse(
        config_class=InferenceConfig, config_path=resolve_hf_path(config_path), args=[]
    )
    device = _pick_device()
    vae_dtype = get_weight_dtype(config.vae.weight_dtype)

    t_load0 = time.time()
    config.vae.vae_pretrained = resolve_hf_path(config.vae.vae_pretrained)
    vae = get_vae(
        config.vae.vae_type,
        config.vae.vae_pretrained,
        device=device,
        dtype=vae_dtype,
        config=config.vae,
    )
    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
    if hasattr(vae, "use_framewise_encoding"):
        vae.use_framewise_encoding = True
        vae.use_framewise_decoding = True
    load_s = time.time() - t_load0
    logger.info(f"[decode] VAE loaded in {load_s:.1f}s  type={config.vae.vae_type}")

    t_load_lat0 = time.time()
    sana_latent = torch.load(latent_path, map_location="cpu")
    logger.info(f"[decode] latent loaded in {time.time() - t_load_lat0:.2f}s  shape={tuple(sana_latent.shape)}")

    t_dec0 = time.time()
    samples = sana_latent.to(device=device, dtype=vae_dtype)
    decoded = vae_decode(config.vae.vae_type, vae, samples)
    if isinstance(decoded, list):
        decoded = torch.stack(decoded, dim=0)
    video_hwc = (
        torch.clamp(127.5 * decoded + 127.5, 0, 255)
        .permute(0, 2, 3, 4, 1)
        .to("cpu", dtype=torch.uint8)
        .numpy()[0]
    )
    del samples, decoded
    _empty_cache(device)
    dec_s = time.time() - t_dec0

    mp4 = write_video(output_dir, name, video_hwc, fps, logger)
    logger.info(f"[decode] decoded in {dec_s:.1f}s  → {mp4}")

    # Write a finishing breadcrumb so render.py can read timings without re-parsing logs.
    (args.in_dir / "decode.meta.json").write_text(json.dumps({
        "vae_load_s": round(load_s, 2),
        "decode_s": round(dec_s, 2),
        "mp4_path": str(mp4),
        "frames": int(video_hwc.shape[0]),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
