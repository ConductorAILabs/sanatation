# SANA-WM + Qwen benchmarks

Run at `2026-05-20T20:52:56`
Host: `REDACTED-host` / `Darwin 25.4.0`
Python: `3.12.13`
Torch: `2.12.0`
MPS available: `True`

## Qwen (via Ollama) latency

Model: `qwen3:30b-a3b`. Prompt format: `format="json"`.

- n = 10
- JSON validity: **100.0%**
- p50 / p95: **3.81s** / **4.83s**
- min / max: 2.46s / 6.5s

---
Finished at `2026-05-20T20:53:35`
## Run `2026-05-20T20:54:22`

SANA-WM cold-load: **11.8s**

## SANA-WM Stage-1 latency

Camera: `w-N`. Image: `demo_1.png`. CFG=1.0, flow_shift=8.0.

| steps | frames | wall-clock (s) | per-frame (s) | per-step-frame (ms) | peak MPS alloc (MB) |
| ---:|  ---: | ---: | ---: | ---: | ---: |
| 4 | 9 | 9.08 | 1.01 | 252.2 | 22705 |
| 6 | 9 | 10.75 | 1.19 | 199.1 | 22707 |
| 8 | 9 | 12.94 | 1.44 | 179.7 | 22707 |
| 12 | 9 | 16.53 | 1.84 | 153.1 | 22707 |
| 20 | 9 | 23.46 | 2.61 | 130.4 | 22715 |
| 4 | 17 | 14.54 | 0.86 | 213.8 | 24372 |
| 20 | 17 | 36.30 | 2.14 | 106.8 | 24372 |
| 4 | 33 | 25.68 | 0.78 | 194.5 | 28789 |
| 20 | 33 | 62.95 | 1.91 | 95.4 | 28789 |
| 20 | 81 | 160.58 | 1.98 | 99.1 | 41926 |

### Derived

- Fitted: `wall_s ≈ 93.7ms × (steps × frames) + 6.65s` (overhead = intercept; per-step-frame cost = slope).
- Fastest config: **4×9** in 9.08s.
- Slowest config: **20×81** in 160.58s.

---
Finished at `2026-05-20T21:00:49`
## Run `2026-05-20T21:01:31`

## Qwen (via Ollama) latency

Model: `qwen3:30b-a3b`. Prompt format: `format="json"`.

- n = 10
- JSON validity: **100.0%**
- p50 / p95: **3.95s** / **5.65s**
- min / max: 2.54s / 6.12s

---
Finished at `2026-05-20T21:02:13`
## Run `2026-05-20T21:02:17`

## Qwen (via Ollama) latency

Model: `qwen3.6:35b-a3b-nvfp4`. Prompt format: `format="json"`.

- n = 10
- JSON validity: **100.0%**
- p50 / p95: **9.52s** / **14.32s**
- min / max: 4.29s / 15.34s

---
Finished at `2026-05-20T21:03:50`
## Run `2026-05-20T21:03:55`

## Qwen (via Ollama) latency

Model: `gemma4:31b-it-q8_0`. Prompt format: `format="json"`.

- n = 10
- JSON validity: **100.0%**
- p50 / p95: **18.01s** / **20.56s**
- min / max: 11.21s / 31.2s

---
Finished at `2026-05-20T21:06:57`
## Run `2026-05-20T21:07:27`

SANA-WM cold-load: **15.5s**

## CFG-scale sweep (4 steps × 9 frames)

`frame_std` is the mean per-frame std-dev of pixel values (uint8). Frames that collapse to near-black have std → 0. Healthy frames are typically 40-80.

| cfg_scale | wall (s) | frame_std (mean) | last-frame std | likely collapsed |
| ---: | ---: | ---: | ---: | :---: |
| 1.0 | 9.84 | 27.2 | 27.0 | no |
| 1.25 | 14.03 | 27.2 | 27.0 | no |
| 1.5 | 13.09 | 27.1 | 27.0 | no |
| 2.0 | 13.07 | 27.1 | 27.0 | no |
| 3.0 | 13.13 | 27.1 | 27.0 | no |
| 5.0 | 13.22 | 27.0 | 27.1 | no |

## End-to-end adventure turn timing (`qwen3:30b-a3b`)

Simulating 5 player turns. SANA at 4 steps × 9 frames per turn.

| turn | qwen (s) | sana (s) | total (s) | camera | json_ok |
| ---: | ---: | ---: | ---: | :--- | :---: |
| 0 | 10.69 | 9.50 | 20.20 | `d-8` | ✓ |
| 1 | 17.17 | 10.01 | 27.18 | `s-8` | ✓ |
| 2 | 23.92 | 9.56 | 33.48 | `s-8` | ✓ |
| 3 | 14.47 | 9.96 | 24.44 | `s-8` | ✓ |
| 4 | 12.81 | 10.08 | 22.89 | `d-8` | ✓ |

- JSON-ok rate: **100.0%**
- SANA mean: **9.82s** · Qwen mean: **15.81s**
- Total per-turn mean: **25.64s** (p95 32.22s)

---
Finished at `2026-05-20T21:11:08`
## Run `2026-05-20T22:18` — subprocess-staged renderer

End-to-end via `render.py` (Stage 1 → Decode subprocesses isolated).

| config | stage1 (s) | refine (s) | decode (s) | total (s) | output |
| --- | ---: | ---: | ---: | ---: | --- |
| 9f × 4s, no refine | 30.5 | — | 9.9 | **40.4** | 230 KB MP4 |
| 9f × 4s, --refine | 32.2 | 42.1 | 9.7 | **84.1** | 150 KB MP4 (sink dropped) |
| 17f × 4s, no refine | OOM (SIGKILL after step 3) | — | — | — | — |
| 17f × 4s, --refine | OOM (SIGKILL after step 3) | — | — | — | — |

The 9-frame configs match expectations — Stage 1 is identical to the
in-process path (~9s sample + ~21s pipeline build), refine adds 42s, decode
10s.

At 17 frames in subprocess the OS kills Stage 1 after the 3rd denoising
step. The same config runs cleanly in-process via `benchmark.py` (14.5s)
because that pipeline stays resident across the matrix sweep. The
subprocess version pays the full pipeline-build cost on a cold start while
swap is already saturated from prior runs — there's no headroom for the
peak memory burst during sampling.

**Workaround**: cold-start the OS (reboot, or `sudo purge`) between long
sessions when running staged renders. junafinity's "96 GB rule" is exactly
the constraint that bites here; their architecture pre-allocates the
budget instead of relying on dynamic swap expansion.

**Refiner per-frame cost**: 42.1s / 9 frames ≈ **4.7s/frame** on this Mac
(M5 Pro Max). Extrapolated, a full 81-frame refined clip would land
around **6.3 min** — substantially faster than junafinity's per-frame
extrapolation suggests, though that's apples-to-oranges (different
hardware, different chunk strategy, different swap state).

---
