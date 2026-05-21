#!/usr/bin/env python3
"""Refiner subprocess: load LTX-2 refiner (transformer + Gemma3 text encoder),
read a Stage-1 latent from disk, run the 3-step Euler refinement, write the
refined latent to disk, exit.

This stage releases its memory on exit so the final decode stage starts fresh.
Per junafinity's M3 Max measurements the refiner peaks at ~38 GB MPS — on a
128 GB machine this lives comfortably alongside the OS, but only because the
previous stage (DiT + Sana VAE + text encoder, ~22 GB) has already exited.

Inputs:  <dir>/stage1.latent.pt + stage1.meta.json
Outputs: <dir>/refine.latent.pt + refine.meta.json

Usage:
    python -m stages.refine --dir <staging-dir>
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

import pyrallis
import torch

REPO = Path(__file__).resolve().parents[1] / "repo"
sys.path.insert(0, str(REPO))

from diffusion.model.utils import get_weight_dtype  # noqa: E402
from diffusion.refiner.diffusers_ltx2_refiner import DiffusersLTX2Refiner  # noqa: E402
from inference_video_scripts.inference_sana_wm import (  # noqa: E402
    InferenceConfig,
    _empty_cache,
    _pick_device,
    get_root_logger,
)
from sana.tools import resolve_hf_path  # noqa: E402

HF_REPO = "Efficient-Large-Model/SANA-WM_bidirectional"
DEFAULT_CONFIG = f"hf://{HF_REPO}/config.yaml"
DEFAULT_REFINER_ROOT = f"hf://{HF_REPO}/refiner"
DEFAULT_GEMMA_ROOT = f"hf://{HF_REPO}/refiner/text_encoder"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True, help="Staging dir containing stage1.latent.pt")
    ap.add_argument("--sink-size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logger = get_root_logger()
    stage1_latent = args.dir / "stage1.latent.pt"
    stage1_meta = args.dir / "stage1.meta.json"
    if not stage1_latent.exists() or not stage1_meta.exists():
        raise SystemExit(f"missing stage1 outputs in {args.dir}")

    meta = json.loads(stage1_meta.read_text(encoding="utf-8"))
    prompt = meta["prompt"]
    fps = float(meta["fps"])
    config_path = meta.get("config", DEFAULT_CONFIG)

    config: InferenceConfig = pyrallis.parse(
        config_class=InferenceConfig, config_path=resolve_hf_path(config_path), args=[]
    )
    if "LTX2VAE_diffusers" not in config.vae.vae_type:
        raise SystemExit(
            f"Refiner requires LTX2VAE_diffusers in the config; got {config.vae.vae_type!r}. "
            "This config doesn't use the LTX-2 VAE, so the refiner cannot run."
        )

    device = _pick_device()
    weight_dtype = get_weight_dtype(config.model.mixed_precision)

    refiner_root = resolve_hf_path(DEFAULT_REFINER_ROOT)
    gemma_root = resolve_hf_path(DEFAULT_GEMMA_ROOT)

    t_load0 = time.time()
    refiner = DiffusersLTX2Refiner(
        refiner_root=refiner_root,
        gemma_root=gemma_root,
        dtype=weight_dtype,
        device=device,
    )
    load_s = time.time() - t_load0
    logger.info(f"[refine] LTX-2 refiner loaded in {load_s:.1f}s")

    t_lat0 = time.time()
    sana_latent = torch.load(stage1_latent, map_location="cpu")
    logger.info(f"[refine] stage1 latent loaded in {time.time() - t_lat0:.2f}s  shape={tuple(sana_latent.shape)}")
    if sana_latent.shape[2] <= args.sink_size:
        raise SystemExit(
            f"Stage 1 produced {sana_latent.shape[2]} latent frames; need > sink_size={args.sink_size}. "
            "Increase --num-frames in render.py so latent_T is at least 2."
        )

    t_ref0 = time.time()
    refined = refiner.refine_latents(
        sana_latent,
        prompt,
        fps=fps,
        sink_size=args.sink_size,
        seed=args.seed,
        progress=True,
    )
    refine_s = time.time() - t_ref0
    logger.info(f"[refine] refined in {refine_s:.1f}s  shape={tuple(refined.shape)}")

    refined_cpu = refined.detach().to("cpu", dtype=torch.float32)
    del refined, sana_latent
    _empty_cache(device)

    out_latent = args.dir / "refine.latent.pt"
    out_meta = args.dir / "refine.meta.json"
    torch.save(refined_cpu, out_latent)
    out_meta.write_text(json.dumps({
        "shape": list(refined_cpu.shape),
        "dtype": "float32",
        "sink_size": args.sink_size,
        "seed": args.seed,
        "load_s": round(load_s, 2),
        "refine_s": round(refine_s, 2),
        # Decode should drop the first frame (sink anchor) post-decode.
        "drop_first_frame": True,
    }, indent=2), encoding="utf-8")
    logger.info(f"[refine] wrote {out_latent}  ({out_latent.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
