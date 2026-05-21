#!/usr/bin/env python3
"""Text adventure game with AI-generated visuals.

Qwen (via Ollama) runs the narrative, SANA-WM generates the video for each
scene. Each turn:
    1. Show the player the latest scene visual.
    2. Qwen narrates and offers 2-3 choices.
    3. Player picks one.
    4. Qwen produces a visual prompt + camera move for the next scene.
    5. SANA-WM generates ~9 frames seeded from the previous scene's last frame.
    6. Auto-open the MP4. Loop.

Requirements:
    - Ollama running locally with a Qwen model pulled. Defaults to
      qwen3:30b-a3b which you already have. Pull others with:
        ollama pull qwen3:30b-a3b
    - The same SANA-WM venv you've been using for run.sh / walk.py.

Usage:
    python adventure.py                                      # cave seed
    python adventure.py --seed-image ~/Pictures/cabin.jpg \
                        --seed-prompt "an old cabin at dusk, fireflies"
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

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

DEFAULT_SEED_IMAGE = REPO / "asset/sana_wm/demo_1.png"
DEFAULT_SEED_PROMPT = "A torch-lit cave passage, damp stone walls glistening, narrow path winding into darkness."
HF_REPO = "Efficient-Large-Model/SANA-WM_bidirectional"
DEFAULT_CONFIG = f"hf://{HF_REPO}/config.yaml"
DEFAULT_MODEL = f"hf://{HF_REPO}/dit/sana_wm_1600m_720p.safetensors"
OUT = Path(__file__).parent / "outputs" / "adventure"

VALID_CAMERAS = {
    "w-8", "a-8", "s-8", "d-8",         # walk / strafe
    "jw-8", "lw-8", "rw-8",              # turn-walks (j=yaw-left, l=yaw-right)
    "w-16",                              # long forward
    "j-4", "l-4",                        # just turn
}

QWEN_SYSTEM = textwrap.dedent("""\
    You are the dungeon master of a video-adventure game. Each turn you produce
    one JSON object — no prose around it, no markdown fences, just JSON.

    The world is rendered by a video diffusion model that takes:
      - the last frame of the previous scene as input image
      - a SHORT visual prompt (one sentence, concrete, painterly, no
        camera-direction words)
      - a camera action string from this exact set:
            w-8     walk forward
            a-8     strafe left
            s-8     walk backward (use rarely)
            d-8     strafe right
            jw-8    turn left while walking
            lw-8    turn right while walking
            rw-8    same as lw-8 (legacy alias)
            w-16    long forward push (use for grand reveals)
            j-4     turn left in place
            l-4     turn right in place

    Schema:
      {
        "narration":     "2-3 sentences of second-person prose describing what
                          happens as a result of the player's choice.",
        "visual_prompt": "ONE evocative sentence describing what the player
                          sees next. Concrete nouns, lighting, colors. Avoid
                          camera verbs like 'pan' or 'tracking shot'.",
        "camera":        "one of the allowed camera strings",
        "choices":       ["short choice 1", "short choice 2", "short choice 3"]
      }

    Rules:
      - The visual must be plausibly the *next* moment in the same physical
        space as the previous scene. Don't teleport unless the choice
        explicitly involves a door/portal/sleep.
      - Keep tonal continuity. Don't switch genres mid-game.
      - Choices should be different in kind, not just rephrasings.
      - If the player wrote a free-form action, interpret it generously.
""")


def ollama_generate(url: str, model: str, system: str, prompt: str) -> dict[str, Any]:
    body = json.dumps({
        "model": model,
        "system": system,
        "prompt": prompt + "\n\nRespond with ONLY a single JSON object matching the schema. /no_think",
        "stream": False,
        "options": {"temperature": 0.8, "top_p": 0.95},
    }).encode("utf-8")
    req = urllib.request.Request(
        url.rstrip("/") + "/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"Could not reach Ollama at {url}: {exc}.\n"
            "Start it with `ollama serve` and pull a model: `ollama pull qwen2.5:7b-instruct`."
        )
    raw = payload.get("response", "").strip()
    return _coerce_json(raw)


def _coerce_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise SystemExit(f"Qwen returned no JSON object. Raw:\n{text[:400]}")
        return json.loads(m.group(0))


def fmt_turn(qwen_out: dict[str, Any], turn: int) -> str:
    narration = qwen_out.get("narration", "").strip()
    choices = qwen_out.get("choices") or []
    width = 78
    lines = [
        "",
        "─" * width,
        f"  Turn {turn}",
        "─" * width,
        textwrap.fill(narration, width=width),
        "",
    ]
    for i, c in enumerate(choices, 1):
        lines.append(f"  {i}) {c}")
    lines.append(f"  f) free-form action      r) replay last scene      q) quit")
    return "\n".join(lines)


def extract_last_frame(video_hwc: np.ndarray) -> Image.Image:
    return Image.fromarray(video_hwc[-1])


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--seed-image", type=Path, default=DEFAULT_SEED_IMAGE)
    ap.add_argument("--seed-prompt", type=str, default=DEFAULT_SEED_PROMPT,
                    help="Visual prompt for the opening scene.")
    ap.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    ap.add_argument("--model-path", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--qwen-url", type=str, default="http://localhost:11434")
    ap.add_argument("--qwen-model", type=str, default="qwen3:30b-a3b")
    ap.add_argument("--steps", type=int, default=4,
                    help="SANA diffusion steps per scene (4=preview, 20=quality, 6 is a nice middle).")
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--flow-shift", type=float, default=8.0)
    ap.add_argument("--no-open", action="store_true",
                    help="Do not auto-open generated MP4s.")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    logger = get_root_logger()

    print(f"loading SANA-WM (one-time, ~60-90s)...")
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
    print("\nSANA loaded.")

    # Probe Ollama with a tiny request so we fail fast.
    try:
        urllib.request.urlopen(args.qwen_url + "/api/tags", timeout=5).read()
    except Exception as exc:
        raise SystemExit(
            f"Could not reach Ollama at {args.qwen_url}: {exc}.\n"
            "Run `ollama serve` in another terminal."
        )

    current_image = Image.open(args.seed_image).convert("RGB")
    visual_prompt = args.seed_prompt
    camera = "w-8"
    history: list[str] = []
    turn = 0

    print("\n" + "═" * 78)
    print("  An adventure begins.")
    print("═" * 78)
    print(textwrap.fill(visual_prompt, width=78))
    print()

    # Generate the opening scene before asking Qwen anything.
    while True:
        c2w = action_string_to_c2w(camera)
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
            seed=turn,
            negative_prompt="",
            sampling_algo="flow_euler_ltx",
        )
        out = pipeline.generate(cropped, visual_prompt, c2w, intrinsics_vec4, params)
        video = out["video"]
        name = f"turn{turn:03d}"
        mp4 = write_video(OUT, name, video, params.fps, logger)
        last = extract_last_frame(video)
        last.save(OUT / f"{name}_last.png")

        if not args.no_open:
            os.system(f"open {mp4}")  # noqa: S605

        # Ask Qwen to describe what just happened and offer choices.
        story_context = "\n".join(f"- {h}" for h in history[-6:]) or "(beginning of adventure)"
        prompt = textwrap.dedent(f"""\
            Story so far:
            {story_context}

            What the player just saw (rendered as a {len(video)}-frame video):
            "{visual_prompt}"
            Camera move: {camera}

            Narrate this moment in second person and offer 3 choices for what the
            player can do next. Respond with a single JSON object matching the
            schema you were given.
        """)
        try:
            qwen = ollama_generate(args.qwen_url, args.qwen_model, QWEN_SYSTEM, prompt)
        except SystemExit:
            raise
        except Exception as exc:
            print(f"[warning] Qwen call failed: {exc}. Using fallback choices.")
            qwen = {
                "narration": "You stand for a moment, considering what to do.",
                "choices": ["press forward", "look around", "turn back"],
                "visual_prompt": visual_prompt,
                "camera": "w-8",
            }

        history.append(f"turn {turn}: saw \"{visual_prompt[:60]}\" (cam {camera})")
        history.append(f"          narrator: {qwen.get('narration', '')[:120]}")

        print(fmt_turn(qwen, turn))
        sys.stdout.write("\n> ")
        sys.stdout.flush()
        choice_input = sys.stdin.readline().strip().lower()

        if choice_input in ("q", "quit", "exit"):
            print("the adventure pauses.")
            return
        if choice_input == "r":
            # Re-render the same prompt with a different seed.
            turn += 1
            continue

        if choice_input == "f":
            sys.stdout.write("describe what you do: ")
            sys.stdout.flush()
            player_action = sys.stdin.readline().strip()
            if not player_action:
                continue
        else:
            try:
                idx = int(choice_input) - 1
                player_action = qwen["choices"][idx]
            except (ValueError, IndexError, KeyError):
                print("not a valid choice; try 1/2/3 or 'f'/'r'/'q'.")
                continue

        history.append(f"          player chose: {player_action}")

        # Ask Qwen for the NEXT scene given the chosen action.
        next_prompt = textwrap.dedent(f"""\
            Story so far:
            {chr(10).join('- ' + h for h in history[-8:])}

            The player just chose: "{player_action}"

            Generate the next scene. Respond with a single JSON object: narration
            of the immediate outcome, visual_prompt for the next image, camera
            action, and 3 fresh choices.
        """)
        try:
            qwen_next = ollama_generate(args.qwen_url, args.qwen_model, QWEN_SYSTEM, next_prompt)
        except Exception as exc:
            print(f"[warning] Qwen call failed: {exc}. Repeating last scene.")
            qwen_next = {
                "narration": "Nothing changes.",
                "choices": ["try again", "wait", "leave"],
                "visual_prompt": visual_prompt,
                "camera": camera,
            }

        # Validate camera before passing to SANA.
        cam = (qwen_next.get("camera") or "w-8").strip()
        if cam not in VALID_CAMERAS:
            # Drop suffix tokens, keep direction. Fall back to w-8.
            cam = "w-8"
        visual_prompt = (qwen_next.get("visual_prompt") or visual_prompt).strip()
        camera = cam
        current_image = last  # seed the next render from the just-finished video's last frame
        turn += 1


if __name__ == "__main__":
    main()
