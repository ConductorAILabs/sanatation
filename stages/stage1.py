#!/usr/bin/env python3
"""Stage 1 subprocess: load full SANA-WM pipeline, generate a Stage-1 latent,
write it to disk, exit.

This is the FIRST stage of subprocess-staged rendering. Memory is released
when the process exits — the next stage (decode or refine) starts fresh.

Inputs: manifest JSON describing the render (image, prompt, action,
num_frames, steps, cfg, etc.).
Outputs: <out_dir>/stage1.latent.pt (the SANA-WM latent tensor)
         <out_dir>/stage1.meta.json (shape, dtype, generation params)

Usage:
    python -m stages.stage1 --manifest <path> --out-dir <path>
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
from PIL import Image

REPO = Path(__file__).resolve().parents[1] / "repo"
sys.path.insert(0, str(REPO))

from inference_video_scripts.inference_sana_wm import (  # noqa: E402
    GenerationParams,
    InferenceConfig,
    SanaWMPipeline,
    _default_intrinsics,
    _snap_num_frames,
    action_string_to_c2w,
    get_root_logger,
    resize_and_center_crop,
    transform_intrinsics_for_crop,
)
from sana.tools import resolve_hf_path  # noqa: E402

HF_REPO = "Efficient-Large-Model/SANA-WM_bidirectional"
DEFAULT_CONFIG = f"hf://{HF_REPO}/config.yaml"
DEFAULT_MODEL = f"hf://{HF_REPO}/dit/sana_wm_1600m_720p.safetensors"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    logger = get_root_logger()

    # Resolve paths and config.
    config_path = manifest.get("config", DEFAULT_CONFIG)
    model_path = manifest.get("model_path", DEFAULT_MODEL)
    image = Image.open(manifest["image"]).convert("RGB")
    prompt = manifest["prompt"]

    t_load0 = time.time()
    config: InferenceConfig = pyrallis.parse(
        config_class=InferenceConfig, config_path=resolve_hf_path(config_path), args=[]
    )
    pipeline = SanaWMPipeline(
        config=config,
        model_path=resolve_hf_path(model_path),
        refiner=None,
        offload_vae=True,
        offload_refiner=True,
        logger=logger,
    )
    load_s = time.time() - t_load0
    logger.info(f"[stage1] pipeline ready in {load_s:.1f}s")

    # Build camera + intrinsics. Snap num_frames to the LTX VAE's 8k+1 grid
    # — anything else risks a degenerate latent_T=1 that triggers fla's
    # autoregressive `step()` path through an unpatched triton kernel.
    c2w = action_string_to_c2w(manifest["action"])
    requested = manifest.get("num_frames", c2w.shape[0])
    num_frames = _snap_num_frames(min(requested, c2w.shape[0]), stride=8, upper_bound=c2w.shape[0])
    if num_frames != requested:
        logger.info(f"[stage1] snapped num_frames {requested} → {num_frames} (LTX 8k+1 grid)")
    c2w = c2w[:num_frames]
    cropped, src_size, resized_size, crop_offset = resize_and_center_crop(image)
    intr_one = _default_intrinsics(image, logger)
    intr_src = np.broadcast_to(intr_one, (num_frames, 4)).copy()
    intrinsics_vec4 = transform_intrinsics_for_crop(intr_src, src_size, resized_size, crop_offset)

    params = GenerationParams(
        num_frames=num_frames,
        fps=manifest.get("fps", 24),
        step=manifest.get("steps", 20),
        cfg_scale=manifest.get("cfg_scale", 1.0),
        flow_shift=manifest.get("flow_shift", 8.0),
        seed=manifest.get("seed", 0),
        negative_prompt=manifest.get("negative_prompt", ""),
        sampling_algo="flow_euler_ltx",
    )

    # Reach into the pipeline to run only Stage 1 (skip the bundled decode).
    vae_stride = config.vae.vae_stride
    latent_T = (num_frames - 1) // vae_stride[0] + 1
    from inference_video_scripts.inference_sana_wm import TARGET_HEIGHT, TARGET_WIDTH, prepare_camera
    latent_h, latent_w = TARGET_HEIGHT // vae_stride[-1], TARGET_WIDTH // vae_stride[-1]
    camera = prepare_camera(c2w, intrinsics_vec4, target_size=(TARGET_HEIGHT, TARGET_WIDTH), vae_stride=vae_stride)

    t_sample0 = time.time()
    sana_latent = pipeline._sample_stage1(cropped, prompt, camera, params, latent_T, latent_h, latent_w)
    sample_s = time.time() - t_sample0
    logger.info(f"[stage1] sampled in {sample_s:.1f}s  shape={tuple(sana_latent.shape)}  dtype={sana_latent.dtype}")

    # Move to CPU before serialization to avoid MPS↔disk weirdness.
    sana_latent_cpu = sana_latent.detach().to("cpu", dtype=torch.float32)

    latent_path = args.out_dir / "stage1.latent.pt"
    meta_path = args.out_dir / "stage1.meta.json"
    torch.save(sana_latent_cpu, latent_path)
    meta = {
        "shape": list(sana_latent_cpu.shape),
        "dtype": "float32",
        "num_frames": num_frames,
        "fps": params.fps,
        "vae_type": config.vae.vae_type,
        "prompt": prompt,
        "name": manifest.get("name", "render"),
        "output_dir": manifest.get("output_dir", str(args.out_dir.parent)),
        "load_s": round(load_s, 2),
        "sample_s": round(sample_s, 2),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info(f"[stage1] wrote {latent_path}  ({latent_path.stat().st_size / 1024:.1f} KB)")
    logger.info(f"[stage1] wrote {meta_path}")


if __name__ == "__main__":
    main()
