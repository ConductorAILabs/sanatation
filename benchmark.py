#!/usr/bin/env python3
"""Benchmark SANA-WM and Qwen on this machine.

Loads SANA once, then sweeps a small matrix of (steps, num_frames) configs.
Also pings Ollama for Qwen latency + JSON-validity rate. Writes results to
BENCHMARKS.md incrementally so partial data survives an interrupt.

Usage:
    python benchmark.py                          # full sweep
    python benchmark.py --short                  # 3 quick configs only
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
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
)
from sana.tools import resolve_hf_path  # noqa: E402

ROOT = Path(__file__).parent
REPORT = ROOT / "BENCHMARKS.md"
SEED_IMAGE = REPO / "asset/sana_wm/demo_1.png"
SEED_PROMPT = "A torch-lit cave passage, damp stone walls glistening."
HF_REPO = "Efficient-Large-Model/SANA-WM_bidirectional"
MODEL = f"hf://{HF_REPO}/dit/sana_wm_1600m_720p.safetensors"
CONFIG_HF = f"hf://{HF_REPO}/config.yaml"
OLLAMA_URL = "http://localhost:11434"

# (steps, num_frames) — keep small enough that the matrix runs in ~5-10 min
# after model load. num_frames must be 8k+1 → 9, 17, 33, 81.
MATRIX_FULL = [
    (4, 9),
    (6, 9),
    (8, 9),
    (12, 9),
    (20, 9),
    (4, 17),
    (20, 17),
    (4, 33),
    (20, 33),
    (20, 81),
]
MATRIX_SHORT = [(4, 9), (8, 9), (20, 9)]


def append_md(line: str) -> None:
    with REPORT.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def init_report() -> None:
    fresh = not REPORT.exists()
    if fresh:
        append_md(f"# SANA-WM + Qwen benchmarks")
        append_md("")
        append_md(f"Host: `{os.uname().nodename}` / `{os.uname().sysname} {os.uname().release}`")
        append_md(f"Python: `{sys.version.split()[0]}` · Torch: `{torch.__version__}` · MPS: `{torch.backends.mps.is_available()}`")
        append_md("")
    append_md(f"## Run `{datetime.now().isoformat(timespec='seconds')}`")
    append_md("")


def mps_peak_mb() -> float:
    if not torch.backends.mps.is_available():
        return float("nan")
    try:
        return torch.mps.driver_allocated_memory() / 1024 / 1024
    except Exception:
        return float("nan")


def mps_reset_peak() -> None:
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    gc.collect()


def bench_sana(pipeline, matrix, logger) -> list[dict]:
    image = Image.open(SEED_IMAGE).convert("RGB")
    cropped, src_size, resized_size, crop_offset = resize_and_center_crop(image)
    intr_one = _default_intrinsics(image, logger)

    append_md("## SANA-WM Stage-1 latency")
    append_md("")
    append_md("Camera: `w-N`. Image: `demo_1.png`. CFG=1.0, flow_shift=8.0.")
    append_md("")
    append_md("| steps | frames | wall-clock (s) | per-frame (s) | per-step-frame (ms) | peak MPS alloc (MB) |")
    append_md("| ---:|  ---: | ---: | ---: | ---: | ---: |")

    rows = []
    for steps, frames in matrix:
        mps_reset_peak()
        cam = f"w-{frames - 1}"
        c2w = action_string_to_c2w(cam)
        intr_src = np.broadcast_to(intr_one, (c2w.shape[0], 4)).copy()
        intrinsics_vec4 = transform_intrinsics_for_crop(intr_src, src_size, resized_size, crop_offset)
        params = GenerationParams(
            num_frames=c2w.shape[0],
            fps=24,
            step=steps,
            cfg_scale=1.0,
            flow_shift=8.0,
            seed=0,
            negative_prompt="",
            sampling_algo="flow_euler_ltx",
        )

        t0 = time.time()
        out = pipeline.generate(cropped, SEED_PROMPT, c2w, intrinsics_vec4, params)
        wall = time.time() - t0
        peak = mps_peak_mb()
        per_frame = wall / frames
        per_sf_ms = wall / (steps * frames) * 1000.0

        row = {
            "steps": steps,
            "frames": frames,
            "wall_s": round(wall, 2),
            "per_frame_s": round(per_frame, 2),
            "per_step_frame_ms": round(per_sf_ms, 1),
            "peak_mps_mb": round(peak, 1),
        }
        rows.append(row)
        append_md(f"| {steps} | {frames} | {wall:.2f} | {per_frame:.2f} | {per_sf_ms:.1f} | {peak:.0f} |")
        print(f"[sana] steps={steps:2d} frames={frames:2d}  wall={wall:6.2f}s  peak={peak:6.0f}MB", flush=True)

        # Free the returned video so it doesn't accumulate in memory between configs.
        del out
        mps_reset_peak()

    append_md("")
    return rows


def bench_cfg_sweep(pipeline, cfg_values: list[float], logger) -> list[dict]:
    """Generate 9 frames at each cfg_scale. Report a 'frame-stillness' metric
    that flags the black-frame collapse (variance of pixel values drops near 0
    on collapsed frames)."""
    image = Image.open(SEED_IMAGE).convert("RGB")
    cropped, src_size, resized_size, crop_offset = resize_and_center_crop(image)
    intr_one = _default_intrinsics(image, logger)
    cam = "w-8"
    c2w = action_string_to_c2w(cam)
    intr_src = np.broadcast_to(intr_one, (c2w.shape[0], 4)).copy()
    intrinsics_vec4 = transform_intrinsics_for_crop(intr_src, src_size, resized_size, crop_offset)

    append_md("## CFG-scale sweep (4 steps × 9 frames)")
    append_md("")
    append_md("`frame_std` is the mean per-frame std-dev of pixel values (uint8). "
              "Frames that collapse to near-black have std → 0. Healthy frames are typically 40-80.")
    append_md("")
    append_md("| cfg_scale | wall (s) | frame_std (mean) | last-frame std | likely collapsed |")
    append_md("| ---: | ---: | ---: | ---: | :---: |")

    rows = []
    for cfg in cfg_values:
        params = GenerationParams(
            num_frames=c2w.shape[0],
            fps=24, step=4, cfg_scale=cfg, flow_shift=8.0,
            seed=0, negative_prompt="", sampling_algo="flow_euler_ltx",
        )
        t0 = time.time()
        out = pipeline.generate(cropped, SEED_PROMPT, c2w, intrinsics_vec4, params)
        wall = time.time() - t0
        video = out["video"]  # (T, H, W, 3) uint8
        per_frame_std = np.std(video.reshape(video.shape[0], -1).astype(np.float32), axis=1)
        last_std = float(per_frame_std[-1])
        mean_std = float(per_frame_std.mean())
        collapsed = "YES" if last_std < 5.0 else "no"
        rows.append({"cfg": cfg, "wall_s": round(wall, 2), "mean_std": round(mean_std, 1),
                     "last_std": round(last_std, 1), "collapsed": collapsed})
        append_md(f"| {cfg} | {wall:.2f} | {mean_std:.1f} | {last_std:.1f} | {collapsed} |")
        print(f"[cfg] cfg={cfg:4.2f}  wall={wall:5.2f}s  mean_std={mean_std:5.1f}  last_std={last_std:5.1f}  "
              f"{'COLLAPSED' if collapsed == 'YES' else 'ok'}", flush=True)
        del out
        mps_reset_peak()

    append_md("")
    return rows


def bench_e2e_adventure(pipeline, model: str, turns: int, logger) -> dict:
    """Simulate `turns` adventure-game iterations end-to-end. Time each phase."""
    import textwrap as _tw

    image = Image.open(SEED_IMAGE).convert("RGB")
    seed_prompt = SEED_PROMPT
    history: list[str] = []

    qwen_system = _tw.dedent("""\
        You are the narrator of a video adventure game. Respond with a single
        JSON object: {"narration": str, "visual_prompt": str (one sentence,
        concrete, no camera-direction words), "camera": one of
        ["w-8","a-8","s-8","d-8","jw-8","lw-8","w-16"], "choices": [str, str, str]}.
        Keep tonal continuity. The visual must plausibly be the next moment in
        the same physical space as the previous scene.""")

    append_md(f"## End-to-end adventure turn timing (`{model}`)")
    append_md("")
    append_md(f"Simulating {turns} player turns. SANA at 4 steps × 9 frames per turn.")
    append_md("")
    append_md("| turn | qwen (s) | sana (s) | total (s) | camera | json_ok |")
    append_md("| ---: | ---: | ---: | ---: | :--- | :---: |")

    current_image = image
    visual_prompt = seed_prompt
    camera = "w-8"
    timings = []
    for t_idx in range(turns):
        # 1. Render the current scene with SANA.
        cropped, src_size, resized_size, crop_offset = resize_and_center_crop(current_image)
        intr_one = _default_intrinsics(current_image, logger)
        c2w = action_string_to_c2w(camera)
        intr_src = np.broadcast_to(intr_one, (c2w.shape[0], 4)).copy()
        intrinsics_vec4 = transform_intrinsics_for_crop(intr_src, src_size, resized_size, crop_offset)
        params = GenerationParams(
            num_frames=c2w.shape[0], fps=24, step=4, cfg_scale=1.0,
            flow_shift=8.0, seed=t_idx, negative_prompt="",
            sampling_algo="flow_euler_ltx",
        )
        t_s0 = time.time()
        out = pipeline.generate(cropped, visual_prompt, c2w, intrinsics_vec4, params)
        sana_s = time.time() - t_s0

        # 2. Ask Qwen for the next scene from the new last frame.
        last = Image.fromarray(out["video"][-1])
        del out
        mps_reset_peak()

        history.append(f"saw \"{visual_prompt[:50]}\" via {camera}")
        ctx = "\n".join("- " + h for h in history[-6:])
        user_prompt = _tw.dedent(f"""\
            Story so far:
            {ctx}

            The player chose: "press onward"

            Generate the next scene as a JSON object.
        """)
        body = json.dumps({
            "model": model, "system": qwen_system,
            "prompt": user_prompt + "\n\nRespond with ONLY a single JSON object. /no_think",
            "stream": False, "options": {"temperature": 0.8},
        }).encode()
        t_q0 = time.time()
        try:
            with urllib.request.urlopen(
                urllib.request.Request(OLLAMA_URL + "/api/generate", data=body,
                                       headers={"Content-Type": "application/json"}),
                timeout=60,
            ) as resp:
                payload = json.loads(resp.read().decode())
            raw = payload.get("response", "").strip()
            try:
                parsed = json.loads(re.search(r"\{.*\}", raw, flags=re.DOTALL).group(0))
                json_ok = {"narration", "visual_prompt", "camera", "choices"}.issubset(parsed.keys())
                if json_ok:
                    visual_prompt = (parsed.get("visual_prompt") or visual_prompt).strip()
                    cam_candidate = (parsed.get("camera") or "w-8").strip()
                    if cam_candidate in {"w-8","a-8","s-8","d-8","jw-8","lw-8","rw-8","w-16","j-4","l-4"}:
                        camera = cam_candidate
            except (AttributeError, json.JSONDecodeError):
                json_ok = False
        except urllib.error.URLError:
            json_ok = False
        qwen_s = time.time() - t_q0

        total_s = sana_s + qwen_s
        timings.append({"turn": t_idx, "sana_s": sana_s, "qwen_s": qwen_s,
                        "total_s": total_s, "camera": camera, "json_ok": json_ok})
        append_md(f"| {t_idx} | {qwen_s:.2f} | {sana_s:.2f} | {total_s:.2f} | "
                  f"`{camera}` | {'✓' if json_ok else '✗'} |")
        print(f"[e2e] turn {t_idx}: sana={sana_s:.2f}s qwen={qwen_s:.2f}s total={total_s:.2f}s "
              f"cam={camera} json={'ok' if json_ok else 'bad'}", flush=True)

        current_image = last

    sana_times = [r["sana_s"] for r in timings]
    qwen_times = [r["qwen_s"] for r in timings]
    total_times = [r["total_s"] for r in timings]
    summary = {
        "turns": turns,
        "json_ok_rate": round(100 * sum(r["json_ok"] for r in timings) / turns, 1),
        "sana_mean_s": round(float(np.mean(sana_times)), 2),
        "qwen_mean_s": round(float(np.mean(qwen_times)), 2),
        "total_mean_s": round(float(np.mean(total_times)), 2),
        "total_p95_s": round(float(np.percentile(total_times, 95)), 2),
    }
    append_md("")
    append_md(f"- JSON-ok rate: **{summary['json_ok_rate']}%**")
    append_md(f"- SANA mean: **{summary['sana_mean_s']}s** · Qwen mean: **{summary['qwen_mean_s']}s**")
    append_md(f"- Total per-turn mean: **{summary['total_mean_s']}s** (p95 {summary['total_p95_s']}s)")
    append_md("")
    return summary


def bench_qwen(model: str, n: int = 5) -> dict:
    """Time `n` JSON-formatted Qwen calls on a fixed prompt; report
    p50 latency and JSON-validity rate."""
    prompt = (
        "You are testing a JSON output. Reply with this exact schema, no prose:\n"
        '{"narration": "the wind picks up", "visual_prompt": '
        '"a dim hallway lit by a single bulb", "camera": "w-8", '
        '"choices": ["enter", "wait", "leave"]}'
    )
    durations = []
    valid = 0
    for i in range(n):
        body = json.dumps({
            "model": model,
            "prompt": prompt + "\n\n/no_think",
            "stream": False,
            "options": {"temperature": 0.7},
        }).encode()
        req = urllib.request.Request(
            OLLAMA_URL + "/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode())
        except urllib.error.URLError as exc:
            print(f"[qwen] call {i} failed: {exc}", flush=True)
            continue
        d = time.time() - t0
        durations.append(d)
        raw = payload.get("response", "").strip()
        try:
            parsed = json.loads(raw)
            keys = {"narration", "visual_prompt", "camera", "choices"}
            if keys.issubset(parsed.keys()):
                valid += 1
        except json.JSONDecodeError:
            pass
        print(f"[qwen] call {i}: {d:.2f}s, valid={raw and json.loads(raw) is not None}", flush=True)

    durations.sort()
    return {
        "model": model,
        "n": n,
        "valid_json_pct": round(100 * valid / max(1, len(durations)), 1),
        "p50_s": round(durations[len(durations) // 2], 2) if durations else None,
        "p95_s": round(durations[int(0.95 * (len(durations) - 1))], 2) if durations else None,
        "min_s": round(durations[0], 2) if durations else None,
        "max_s": round(durations[-1], 2) if durations else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--short", action="store_true", help="3-config sweep instead of full 10.")
    ap.add_argument("--qwen-model", default="qwen3:30b-a3b")
    ap.add_argument("--qwen-n", type=int, default=5)
    ap.add_argument("--no-qwen", action="store_true")
    ap.add_argument("--no-sana", action="store_true")
    ap.add_argument("--cfg-sweep", action="store_true",
                    help="Sweep cfg_scale to find black-frame collapse threshold.")
    ap.add_argument("--e2e", type=int, default=0, metavar="TURNS",
                    help="Run end-to-end adventure-turn benchmark for N turns.")
    args = ap.parse_args()

    init_report()
    logger = get_root_logger()

    needs_pipeline = not args.no_sana or args.cfg_sweep or args.e2e > 0
    pipeline = None
    if needs_pipeline:
        print("Loading SANA-WM (~60-90s)...", flush=True)
        t_load = time.time()
        config: InferenceConfig = pyrallis.parse(
            config_class=InferenceConfig, config_path=resolve_hf_path(CONFIG_HF), args=[]
        )
        pipeline = SanaWMPipeline(
            config=config,
            model_path=resolve_hf_path(MODEL),
            refiner=None,
            offload_vae=True,
            offload_refiner=True,
            logger=logger,
        )
        load_s = time.time() - t_load
        print(f"Loaded in {load_s:.1f}s", flush=True)
        append_md(f"SANA-WM cold-load: **{load_s:.1f}s**")
        append_md("")

    if not args.no_sana and pipeline is not None:
        matrix = MATRIX_SHORT if args.short else MATRIX_FULL
        sana_rows = bench_sana(pipeline, matrix, logger)

        # Quick derived analysis.
        append_md("### Derived")
        append_md("")
        # Linear regression of wall_s vs (steps × frames)
        if len(sana_rows) >= 3:
            xs = np.array([r["steps"] * r["frames"] for r in sana_rows], dtype=float)
            ys = np.array([r["wall_s"] for r in sana_rows], dtype=float)
            slope, intercept = np.polyfit(xs, ys, 1)
            append_md(
                f"- Fitted: `wall_s ≈ {slope*1000:.1f}ms × (steps × frames) + {intercept:.2f}s` "
                f"(overhead = intercept; per-step-frame cost = slope)."
            )
            fastest = min(sana_rows, key=lambda r: r["wall_s"])
            slowest = max(sana_rows, key=lambda r: r["wall_s"])
            append_md(f"- Fastest config: **{fastest['steps']}×{fastest['frames']}** in {fastest['wall_s']}s.")
            append_md(f"- Slowest config: **{slowest['steps']}×{slowest['frames']}** in {slowest['wall_s']}s.")
        append_md("")

    if args.cfg_sweep and pipeline is not None:
        bench_cfg_sweep(pipeline, [1.0, 1.25, 1.5, 2.0, 3.0, 5.0], logger)

    if args.e2e > 0 and pipeline is not None:
        bench_e2e_adventure(pipeline, args.qwen_model, args.e2e, logger)

    if pipeline is not None:
        del pipeline
        mps_reset_peak()

    if not args.no_qwen:
        append_md("## Qwen (via Ollama) latency")
        append_md("")
        append_md(f"Model: `{args.qwen_model}`. Prompt format: `format=\"json\"`.")
        append_md("")
        qwen = bench_qwen(args.qwen_model, n=args.qwen_n)
        append_md(f"- n = {qwen['n']}")
        append_md(f"- JSON validity: **{qwen['valid_json_pct']}%**")
        append_md(f"- p50 / p95: **{qwen['p50_s']}s** / **{qwen['p95_s']}s**")
        append_md(f"- min / max: {qwen['min_s']}s / {qwen['max_s']}s")
        append_md("")

    append_md("---")
    append_md(f"Finished at `{datetime.now().isoformat(timespec='seconds')}`")
    print(f"\nReport written to {REPORT}", flush=True)


if __name__ == "__main__":
    main()
