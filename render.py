#!/usr/bin/env python3
"""Subprocess-staged SANA-WM renderer.

Architecture cribbed from junafinity's osmapi/SANA-WM-Bidirectional-on-Apple-Silicon
(https://huggingface.co/osmapi/SANA-WM-Bidirectional-on-Apple-Silicon).
Their insight: keep Stage 1, refiner, and VAE in disjoint subprocesses so
macOS can reclaim each model's pages between stages.

Our skeleton ships two stages — `stage1` and `decode` — and leaves a slot
for `refine` once we validate it end-to-end. Each stage runs as `python -m
stages.<name>`; we orchestrate via subprocess.run + manifest JSON on disk.

Usage:
    python render.py                                                # cave demo
    python render.py --image my.png --prompt "..." --action "w-40,jw-20"
    python render.py --num-frames 17 --steps 20 --name short_test
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "repo"
# Default venv path matches what apply-patches.sh creates (./.venv next to this
# script). Override with the VENV_PY env var if your venv lives elsewhere.
VENV_PY = Path(os.environ.get("VENV_PY", ROOT / ".venv" / "bin" / "python"))

DEFAULT_IMAGE = REPO / "asset/sana_wm/demo_1.png"
DEFAULT_PROMPT_FILE = REPO / "asset/sana_wm/demo_1.txt"
DEFAULT_OUTPUT_DIR = ROOT / "outputs"
HF_REPO = "Efficient-Large-Model/SANA-WM_bidirectional"
DEFAULT_CONFIG = f"hf://{HF_REPO}/config.yaml"
DEFAULT_MODEL = f"hf://{HF_REPO}/dit/sana_wm_1600m_720p.safetensors"


def run_stage(module: str, *args: str) -> float:
    """Run `python -m <module> <args>` as a subprocess. Return wall-clock seconds.
    Aborts if exit code != 0; stdout/stderr stream to the parent terminal so
    the user sees what's happening."""
    cmd = [str(VENV_PY), "-m", module, *args]
    print(f"\n▶  {' '.join(cmd)}\n", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT)
    wall = time.time() - t0
    if proc.returncode != 0:
        sys.exit(f"\n✗  stage {module} failed (exit {proc.returncode}) after {wall:.1f}s")
    print(f"\n✓  {module} done in {wall:.1f}s", flush=True)
    return wall


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    ap.add_argument("--prompt", type=str, default=None, help="Inline prompt; else --prompt-file.")
    ap.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    ap.add_argument("--action", type=str, default="w-40,jw-20,w-20")
    ap.add_argument("--num-frames", type=int, default=80)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--flow-shift", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    ap.add_argument("--model-path", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--refine", action="store_true",
                    help="Run the LTX-2 refiner stage between Stage 1 and decode. "
                         "Adds substantial wall-clock and memory cost (~38 GB MPS peak per junafinity's "
                         "M3 Max measurements) but gives output parity with the upstream CUDA pipeline.")
    ap.add_argument("--sink-size", type=int, default=1, help="Refiner sink anchor frames.")
    ap.add_argument("--refiner-seed", type=int, default=42)
    ap.add_argument("--keep-intermediates", action="store_true",
                    help="Don't delete the staging dir after success.")
    args = ap.parse_args()

    if not VENV_PY.exists():
        sys.exit(f"venv python not found at {VENV_PY}; set VENV_PY env var to point at it.")

    prompt = args.prompt or args.prompt_file.read_text(encoding="utf-8").strip()
    name = args.name or f"render_{int(time.time())}"
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="sana-staging-", dir=str(output_dir)) as staging:
        staging_path = Path(staging)
        manifest = {
            "image": str(args.image.resolve()),
            "prompt": prompt,
            "action": args.action,
            "num_frames": args.num_frames,
            "steps": args.steps,
            "cfg_scale": args.cfg,
            "flow_shift": args.flow_shift,
            "seed": args.seed,
            "fps": args.fps,
            "negative_prompt": "",
            "name": name,
            "output_dir": str(output_dir.resolve()),
            "config": args.config,
            "model_path": args.model_path,
        }
        manifest_path = staging_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        timings = {}
        timings["stage1"] = run_stage("stages.stage1", "--manifest", str(manifest_path),
                                       "--out-dir", str(staging_path))
        if args.refine:
            timings["refine"] = run_stage("stages.refine", "--dir", str(staging_path),
                                           "--sink-size", str(args.sink_size),
                                           "--seed", str(args.refiner_seed))
        timings["decode"] = run_stage("stages.decode", "--in-dir", str(staging_path))

        # Pull stage metadata for a one-line summary.
        decode_meta = json.loads((staging_path / "decode.meta.json").read_text(encoding="utf-8"))
        mp4_path = decode_meta["mp4_path"]
        total_s = sum(timings.values())
        line = "  " + "   ".join(f"{stage}: {sec:.1f}s" for stage, sec in timings.items())
        print(
            f"\n{'='*68}\n"
            f"  Render complete: {mp4_path}\n"
            f"{line}   total: {total_s:.1f}s\n"
            f"{'='*68}"
        )

        if args.keep_intermediates:
            preserved = output_dir / f"{name}_staging"
            preserved.mkdir(parents=True, exist_ok=True)
            for f in staging_path.iterdir():
                f.rename(preserved / f.name)
            print(f"  intermediates preserved at: {preserved}")


if __name__ == "__main__":
    main()
