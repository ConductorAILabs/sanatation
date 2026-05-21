# SANA-WM on Apple Silicon — interactive runtime

NVIDIA released [SANA-WM](https://github.com/NVlabs/Sana/pull/379), a bidirectional
video "world model" — give it a still image and a camera path, get back a few
seconds of video moving through that scene. The upstream code is CUDA-only:
it imports `triton`, `mmcv`, `xformers`, `flash-linear-attention`,
`bitsandbytes`, and `liger_kernel` at module load. None of those have macOS
arm64 wheels.

This repo is the set of patches that make SANA-WM import, load, and generate
on an M-series Mac via PyTorch MPS, plus an interactive layer on top — walk
the camera around with WASD, or run an LLM-driven adventure game where the
visuals come from SANA. No CUDA, no Linux box, no cloud GPU.

## Related work

If you want **full-pipeline rendering** with the LTX-2 refiner (321-frame,
20-second 720p clips), use [`osmapi/SANA-WM-Bidirectional-on-Apple-Silicon`](https://huggingface.co/osmapi/SANA-WM-Bidirectional-on-Apple-Silicon)
by [junafinity](https://huggingface.co/blog/junafinity/sana-wm-bidirectional-on-apple-silicon).
Their work uses subprocess staging to enforce a hard 96 GB memory budget so
all three stages (Stage 1 DiT, LTX-2 refiner, LTX-2 VAE) never co-reside in
memory. That's the right architecture for one-shot cinematic output.

This repo focuses on the opposite axis: **short chunks, fast iteration, and
interactive control loops** — explicitly listed as future work in their
README. The two are complementary; we link to their patch set in
`PATCHES_TECHNICAL.md` and plan to borrow their staging approach for our
cinematic-finale mode.

## Quickstart

```bash
git clone https://github.com/ConductorAILabs/sanatation
cd sanatation
./apply-patches.sh                              # clone NVlabs/Sana@485a6bb, apply patches, build venv
./run.sh                                        # default demo
./run.sh "w-30,a-20,jw-20,d-10"                  # custom camera trajectory
CFG=1.0 STEPS=20 NAME=mytest ./run.sh "rw-80"
```

`apply-patches.sh` clones the upstream PR #379 HEAD into `./repo/`, applies
`patches/repo.patch`, creates `./.venv/`, installs the macOS-compatible
dependency set, and copies the four `patches/venv/*.py` files into the venv.

`run.sh` sets every env flag SANA-WM needs to find the pure-PyTorch code paths
and runs the Stage-1 inference script. Output lands in `outputs/<name>_generated.mp4`.

Tested on M5 Pro Max, 128 GB unified memory. Will work with less RAM but with
shorter clips.

## Two runtime modes

Pick based on what you're doing — they share the same patched checkout and
venv, only the orchestration differs.

| Mode | Entry point | Latency | When to use |
|---|---|---|---|
| **In-process** | `run.sh`, `walk.py`, `adventure.py` | ~10s per 9-frame chunk | Interactive — chained turns, WASD play, LLM-driven scenes. Pipeline loads once. |
| **Subprocess-staged** | `render.py` | ~40s per 9-frame clip today; designed to scale to long clips + refiner | One-shot rendering where each stage's memory must release before the next loads. |

For day-to-day play, in-process. Subprocess staging is for the cinematic
finale path and (once we add it) the LTX-2 refiner stage — see
`PATCHES_TECHNICAL.md`.

## What works

| | Status |
|---|---|
| Stage-1 SANA-WM DiT (1.6B) on MPS | ✅ at `cfg_scale=1.0` |
| Trained 15-GDN + 5-softmax hybrid topology | ✅ via `SANA_WM_RESTORE_GDN=1` |
| 1280×704 video, 81 frames, 20 steps | ✅ ~2:20 end-to-end on M5 Pro Max |
| Camera control from WASD / trajectory strings | ✅ |
| LTX-2 refiner (Stage-2) | ⚠ patched but not validated end-to-end |
| `cfg_scale > 1.0` | ❌ produces black frames — `null_embed` workaround pending |
| Pi3X intrinsics estimation | ⚠ bypassed, use `--intrinsics` or default 55° FOV |
| Real-time playback | ❌ each step is ~3–5 s; chess-pace at best |

Reports on M1/M2/M3/M4 welcome — file an issue with your timings.

## Limitations to be honest about

- **Speed.** ~2:20 for a 5-second 1280×704 clip on M5 Pro Max. CUDA does this
  in ~30 s. M1/M2/M3 will be slower than M5.
- **`cfg_scale=1.0` only.** Classifier-free guidance needs an
  `uncond_prompt_embeds.pt` that the bidirectional snapshot doesn't ship.
  With CFG > 1 the recurrence amplifies numerical noise and produces
  black/streaked frames past frame 0. See `PATCHES_TECHNICAL.md §11`.
- **Resolution fixed at 1280×704.** Model architecture, not a port
  limitation.
- **`--num_frames` must be 8k+1.** LTX-2 VAE constraint; `run.sh` auto-snaps to
  the nearest.
- **Quality is at the Stage-1 level.** No parity claim against the full
  CUDA pipeline (refiner stage is patched but not validated).

## How (the interesting bits)

Patches that other CUDA-only video models will likely need too:

1. **Triton stubs.** `_triton_stub.py` registers no-op `triton` /
   `triton.language` / `.runtime` / `.compiler` modules in `sys.modules` before
   anything tries to import them, so `import diffusion` succeeds without
   touching the model code.
2. **Real-math RoPE.** MPS lacks `torch.view_as_complex` for some shapes. The
   rotary embedding paths in `sana_blocks.py`, `sana_gdn_blocks.py`,
   `sana_camctrl_blocks.py`, and `wan/model.py` are rewritten to compute the
   rotation manually:
   `o_re = x_re·cos − x_im·sin; o_im = x_re·sin + x_im·cos`.
   Numerically equivalent to <1e-5 vs the complex path.
3. **fp64 → fp32 on MPS.** MPS doesn't implement `aten::*` for float64.
   The RoPE freqs and a couple of LTX-2 connector paths are downgraded to fp32
   with an explicit cast.
4. **Pure-PyTorch GDN paths.** Two trained block types
   (`BidirectionalGDNUCPESinglePathLiteLABothTriton`, `BidirectionalGDNTriton`)
   are remapped to their non-Triton siblings at construction time; chunkwise
   GDN is forced to recurrent because `@torch.compile` + a tricky
   `(I − k_beta @ k_rotᵀ)` matrix construction is unstable on MPS.
5. **`view` → `reshape`.** MPS sometimes returns non-contiguous tensors where
   CUDA gives contiguous; `.view(B, -1, C)` after attention transposes fails.
6. **`fla` library fallbacks.** `causal_conv1d` gets a pure-PyTorch path
   (depthwise `F.conv1d` + left-pad). `custom_device_ctx` returns a
   `nullcontext` on devices that don't expose `.device(index)`.
7. **CUDA deps gated as `[cuda]` extras.** `pyproject.toml` is patched so the
   main install set has no CUDA-only wheels — install works clean on macOS arm64.

Full file-level rundown with the original CUDA behavior each patch replaces is
in [`PATCHES_TECHNICAL.md`](PATCHES_TECHNICAL.md).

## Repo layout

```
sanatation/
├── README.md                     ← this file
├── LICENSE                       ← Apache 2.0 (matches upstream)
├── PATCHES_TECHNICAL.md          ← per-file, per-line patch notes
├── BENCHMARKS.md                 ← latency / memory measurements on M5 Pro Max
├── apply-patches.sh              ← clone NVlabs/Sana@485a6bb, apply patches, build venv
├── run.sh                        ← one-shot Stage-1 inference with all env vars set
├── walk.py                       ← WASD camera walking; one keypress = one short chunk
├── adventure.py                  ← LLM-driven adventure game (Qwen via Ollama + SANA)
├── render.py                     ← subprocess-staged renderer (Stage-1 → decode for now)
├── stages/                       ← subprocess entry points used by render.py
│   ├── stage1.py                 ←   loads pipeline, samples Stage-1 latent, exits
│   └── decode.py                 ←   loads VAE only, decodes latent → MP4, exits
├── benchmark.py                  ← latency / memory / e2e harness used for BENCHMARKS.md
├── patches/
│   ├── repo.patch                ← unified diff against NVlabs/Sana@485a6bb
│   └── venv/                     ← drop-in replacements for fla and diffusers
│       ├── fla__utils.py
│       ├── fla_modules_conv__causal_conv1d.py
│       ├── diffusers_ltx2__connectors.py
│       └── diffusers_transformers__transformer_ltx2.py
├── repo/                         ← gitignored; created by apply-patches.sh
├── .venv/                        ← gitignored; created by apply-patches.sh
└── outputs/                      ← gitignored; your generated videos
```

`repo/` itself is not vendored — `apply-patches.sh` clones it fresh from
upstream and applies `patches/repo.patch`. Smaller download, cleaner provenance.

## Camera DSL

The `--action` argument (or first positional arg to `run.sh`) is a
comma-separated list of camera moves:

| token | meaning |
|---|---|
| `w`, `s` | walk forward / backward |
| `a`, `d` | strafe left / right |
| `l`, `r` | look up / right (combine with walks like `lw-20`, `rw-30`) |
| `j` | jump (combine like `jw-40`) |
| `-N` | apply this move for N frames |

Examples:

```bash
./run.sh "w-40,jw-20,w-20"        # walk, jump-walk, walk
./run.sh "rw-30,d-10,la-20"       # turn-right-walk, strafe, turn-left-strafe
```

## Who this is for

- You want to **drive SANA-WM interactively** — walk the camera around, chain
  short chunks, or have an LLM author scenes on the fly. (Pick this repo.)
- You're porting some other CUDA-only video diffusion model to MPS and want to
  see what the workarounds look like — the patterns here generalize.
- You work on PyTorch MPS at Apple and want a real-world stress test that
  exercises ~15 different MPS gaps simultaneously.

If you want **full-pipeline rendering with the LTX-2 refiner** in one shot,
use [junafinity's port](https://huggingface.co/osmapi/SANA-WM-Bidirectional-on-Apple-Silicon)
instead — they've worked out the memory contract for keeping all stages
below 96 GB on a 128 GB Mac. We focus on Stage-1 only for now and plan to
borrow their staging architecture for our cinematic-finale mode.

## Credits

- Upstream: [NVlabs/Sana](https://github.com/NVlabs/Sana), PR #379 at
  `485a6bbf7084001b3a6f736a89d217e4bb5749c3`. Apache 2.0, license preserved.
- Model weights: [Efficient-Large-Model/SANA-WM_bidirectional](https://huggingface.co/Efficient-Large-Model/SANA-WM_bidirectional)
  (~96 GB).
- Paper: [SANA-WM](https://arxiv.org/abs/2605.15178).
- Bridge: Conductor AI Labs. PRs welcome.
