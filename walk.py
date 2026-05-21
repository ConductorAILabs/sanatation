#!/usr/bin/env python3
"""Walk around inside a SANA-WM scene with single keypresses.

Loads the pipeline once (~60-90s), then each key generates an 8-frame chunk
(~10-15s on M5) seeded from the last frame of the previous chunk.

Keys:
    w/a/s/d  walk forward/strafe-left/back/strafe-right
    j/l      look left / right (yaw)
    i/k      look up / down (pitch, limited)
    space    same as w (alias)
    z        replay the last chunk
    q        quit

Usage:
    python walk.py                                       # cave demo
    python walk.py --image ~/Pictures/room.jpg \
                   --prompt "wide shot of an empty hallway"
"""
from __future__ import annotations

import argparse
import os
import sys
import termios
import tty
from pathlib import Path

# Set MPS fallback before any torch import.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
for k in (
    "SANA_WM_FORCE_PURE_PYTORCH", "FLA_USE_PURE_PYTORCH", "DISABLE_XFORMERS",
    "GDN_DISABLE_COMPILE", "SANA_WM_NO_COMPLEX_ROPE", "SANA_WM_RESTORE_GDN",
):
    os.environ.setdefault(k, "1")

import numpy as np
import pyrallis
from PIL import Image

REPO = Path(__file__).parent / "repo"
sys.path.insert(0, str(REPO))

from inference_video_scripts.inference_sana_wm import (  # noqa: E402
    GenerationParams,
    InferenceConfig,
    SanaWMPipeline,
    _default_intrinsics,
    action_string_to_c2w,
    get_root_logger,
    resize_and_center_crop,
    transform_intrinsics_for_crop,
    write_video,
)
from sana.tools import resolve_hf_path  # noqa: E402

DEFAULT_IMAGE = REPO / "asset/sana_wm/demo_1.png"
DEFAULT_PROMPT_FILE = REPO / "asset/sana_wm/demo_1.txt"
HF_REPO = "Efficient-Large-Model/SANA-WM_bidirectional"
DEFAULT_CONFIG = f"hf://{HF_REPO}/config.yaml"
DEFAULT_MODEL = f"hf://{HF_REPO}/dit/sana_wm_1600m_720p.safetensors"
OUT = Path(__file__).parent / "outputs" / "walk"

CHUNK_FRAMES = 8  # one move = 8 generated frames + the seed = 9 total in c2w
KEY_TO_ACTION = {
    "w": "w", "a": "a", "s": "s", "d": "d",
    "j": "j", "l": "l", "i": "i", "k": "k",
    " ": "w",
}


def read_one_key() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def extract_last_frame(video_hwc: np.ndarray) -> Image.Image:
    return Image.fromarray(video_hwc[-1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    ap.add_argument("--prompt", type=str, default=None,
                    help="Inline prompt. If omitted, reads --prompt-file.")
    ap.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    ap.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    ap.add_argument("--model-path", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--steps", type=int, default=4, help="Diffusion steps per chunk (4=preview, 20=quality)")
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--flow-shift", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--auto-open", action="store_true",
                    help="Run `open` on each chunk's MP4 as it lands.")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    logger = get_root_logger()
    prompt = args.prompt or args.prompt_file.read_text(encoding="utf-8").strip()

    print(f"loading SANA-WM Stage-1 on first device available... (one-time, ~60-90s)")
    config: InferenceConfig = pyrallis.parse(
        config_class=InferenceConfig, config_path=resolve_hf_path(args.config), args=[]
    )
    pipeline = SanaWMPipeline(
        config=config,
        model_path=resolve_hf_path(args.model_path),
        refiner=None,
        offload_vae=True,
        offload_refiner=True,
        logger=logger,
    )

    current_image = Image.open(args.image).convert("RGB")
    turn = 0

    print(
        f"\nLoaded. Image: {args.image.name}\n"
        f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}\n"
        f"Chunks land in: {OUT}\n"
        f"\nKeys: w/a/s/d move, j/l yaw, i/k pitch, space=walk, z=replay, q=quit"
    )

    last_action = "w"

    while True:
        sys.stdout.write(f"\n[turn {turn}] move? > ")
        sys.stdout.flush()
        key = read_one_key().lower()
        sys.stdout.write(f"{key}\n")
        sys.stdout.flush()

        if key == "q" or key == "\x03":
            print("bye.")
            return
        if key == "z":
            action = last_action
        elif key in KEY_TO_ACTION:
            action = KEY_TO_ACTION[key]
        else:
            print(f"unknown key {key!r}; try w/a/s/d/j/l/i/k/space/z/q")
            continue
        last_action = action

        c2w = action_string_to_c2w(f"{action}-{CHUNK_FRAMES}")  # → (CHUNK_FRAMES+1, 4, 4)
        cropped, src_size, resized_size, crop_offset = resize_and_center_crop(current_image)
        intr_one = _default_intrinsics(current_image, logger)
        intr_src = np.broadcast_to(intr_one, (c2w.shape[0], 4)).copy()
        intrinsics_vec4 = transform_intrinsics_for_crop(intr_src, src_size, resized_size, crop_offset)

        params = GenerationParams(
            num_frames=c2w.shape[0],
            fps=24,
            step=args.steps,
            cfg_scale=args.cfg,
            flow_shift=args.flow_shift,
            seed=args.seed + turn,
            negative_prompt="",
            sampling_algo="flow_euler_ltx",
        )

        out = pipeline.generate(cropped, prompt, c2w, intrinsics_vec4, params)
        video = out["video"]

        name = f"turn{turn:03d}_{action}"
        write_video(OUT, name, video, params.fps, logger)
        last_frame = extract_last_frame(video)
        last_frame.save(OUT / f"{name}_last.png")
        current_image = last_frame

        if args.auto_open:
            os.system(f"open {OUT / f'{name}.mp4'}")  # noqa: S605

        turn += 1


if __name__ == "__main__":
    main()
