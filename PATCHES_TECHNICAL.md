# Bridge patches — what we changed to make SANA-WM run on MPS

Every patch lives in `patches/` and is applied by `apply-patches.sh`. This doc
explains each one with file paths and the upstream CUDA behavior it replaces,
so the bridge can be re-created against a future rebase of PR #379 or adapted
to a different CUDA-only video model.

## 0. Source state

- Upstream: `NVlabs/Sana`, PR #379 (`feat/sana-wm` branch), head SHA
  `485a6bbf7084001b3a6f736a89d217e4bb5749c3`.
- Hard requirements that don't have macOS arm64 wheels:
  `xformers==0.0.33.post2`, `triton==3.5.1`, `flash-linear-attention>=0.4.2`,
  `mmcv==1.7.2`, `flash-attn>=2.7.0`. All CUDA-only.

## 1. `pyproject.toml`

- Moved `xformers`, `triton`, `flash-linear-attention`, `mmcv`, `bitsandbytes`,
  `liger_kernel`, `torch==2.9.1+cu128`, `torchvision`, `torchaudio` into a new
  `[project.optional-dependencies] cuda` group. The required block now installs
  cleanly on macOS arm64.
- Commented out `[tool.pip] extra-index-url = [".../whl/cu128"]` so pip doesn't
  try to clobber MPS torch.
- `mmcv==1.7.2` is still needed (model code uses legacy `Registry`/`Config` APIs).
  Install it as: `pip install --no-build-isolation mmcv==1.7.2`.

## 2. Triton/xformers/flash-attn module-level imports

Created `diffusion/model/nets/_triton_stub.py`. It installs no-op
`triton` / `triton.language` / `.runtime` / `.compiler` / `.backends` / `.knobs`
packages into `sys.modules` so import-time `import triton` calls in
`fla`, `dc_ae`, and the SANA repo's own `*_triton.py` modules don't explode.
Called from the top of `sana_gdn_blocks.py` and `sana_gdn_camctrl_blocks.py`.

Also patched: `diffusion/model/ops/{fused_gdn,fused_cam_gdn,fused_gdn_chunkwise,fused_gdn_chunkwise_bwd}.py`
and `diffusion/model/dc_ae/efficientvit/models/nn/triton_rms_norm.py`
to wrap `import triton` in try/except.

## 3. Force pure-PyTorch GDN paths

`diffusion/model/nets/sana_multi_scale_video_camctrl.py`:
- Added `_force_no_triton_block_type()` helper at top of file. When
  `SANA_WM_FORCE_PURE_PYTORCH=1` or no CUDA is available, remaps trained block
  types:
  - `BidirectionalGDNUCPESinglePathLiteLABothTriton` → `BidirectionalGDNUCPESinglePathLiteLA`
    (when `SANA_WM_RESTORE_GDN=1`) OR `BidirectionalSoftmaxUCPESinglePathLiteLA`
    (default — quality drop).
  - `BidirectionalGDNTriton` → `BidirectionalGDN`.
- Added construction branch (around line 209) for the pure-PyTorch
  `BidirectionalGDNUCPESinglePathLiteLA` class. The trained-model `__init__`
  originally only had branches for `BothTriton` and `Softmax`.

`diffusion/model/nets/sana_gdn_blocks.py`:
- Dispatch logic forces `update_rule_func = "torch_recurrent_sana_gdn"` (not
  chunkwise) when `SANA_WM_FORCE_PURE_PYTORCH=1`. Chunkwise has `@torch.compile`
  + `(I − k_beta @ k_rotᵀ)` matrix construction that's unstable on MPS.
- `@torch.compile` on `torch_chunk_sana_gdn` (line 193) wrapped with
  `disable=os.environ.get("GDN_DISABLE_COMPILE","0")=="1"`.

`diffusion/model/nets/sana_gdn_camctrl_blocks.py`:
- Same: force `cam_update_rule_func = "torch_recurrent"` when env is set.

## 4. Real-math RoPE (replaces `view_as_complex`)

MPS doesn't support `torch.view_as_complex` for some shapes. Replaced with
explicit real arithmetic: `freqs_cos = freqs.real`, `freqs_sin = freqs.imag`,
then `o_re = x_re*cos − x_im*sin; o_im = x_re*sin + x_im*cos`.

Patched in:
- `diffusion/model/nets/sana_gdn_blocks.py` — `_apply_rotary_emb_real()`
- `diffusion/model/nets/sana_camctrl_blocks.py` — `_apply_complex_rope_real()`
- `diffusion/model/nets/sana_blocks.py` — 8 separate call sites, 3 layout
  helpers added (`_apply_rotary_emb_real_perm`, `_apply_rotary_emb_real_transpose`,
  `_apply_rotary_emb_real_lumina`)
- `diffusion/model/wan/model.py` — `_rope_apply_real_wan` inside `rope_apply`

All gated by `device.type == "mps"` or `SANA_WM_NO_COMPLEX_ROPE=1`. Numerical
equivalence vs the complex path verified to <1e-5.

## 5. fp64 → fp32 freqs on MPS

MPS doesn't support `float64` tensors. Three sites cast to fp32 on MPS:

- `diffusion/model/nets/sana_blocks.py:1488` (RoPE forward — downcast
  complex128 → complex64 on MPS)
- `patches/venv/diffusers_ltx2__connectors.py:128`
  (`freqs_dtype = float64 if (self.double_precision and _dev_type != "mps") else float32`)
- `patches/venv/diffusers_transformers__transformer_ltx2.py:1012` (same fix)

## 6. `view` → `reshape` on non-contiguous tensors

MPS sometimes returns non-contiguous tensors where CUDA gives contiguous. After
attention transposes, `.view(B, -1, C)` fails.

- `diffusion/model/nets/sana_blocks.py:176` — `x = x.reshape(B, -1, C)`

## 7. Device-agnostic inference script

`inference_video_scripts/inference_sana_wm.py`:
- Added `_pick_device()` and `_empty_cache(device)` helpers at top.
- Replaced 7 `torch.cuda.empty_cache()` with `_empty_cache(self.device)`.
- Replaced hardcoded `device="cuda"` defaults with `device=None` + `_pick_device()`
  fallback.
- `torch.amp.autocast("cuda", ...)` → `torch.amp.autocast(device.type, ...)`.
- Wrapped `torch.Generator(device=self.device)` in try/except (MPS generator
  occasionally errors on some torch versions); falls back to CPU generator.
- Added `_default_intrinsics()` fallback that estimates `(fx, fy, cx, cy)` from
  image dimensions at 55° FOV when `--intrinsics` is omitted and Pi3X is
  unavailable.
- `os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")` at top.

## 8. LTX-2 refiner patches

`diffusion/refiner/diffusers_ltx2_refiner.py`:
- Added `_pick_device()` + `_make_generator()` helpers.
- Replaced `torch.cuda.empty_cache()` with `_empty_device_cache(self.device)`.
- Wrapped `torch.Generator(device=self.device)` similarly.

Refiner is patched but not validated end-to-end on MPS — only the Stage-1 path
has been smoke-tested.

## 9. fla library patches (venv-side)

`patches/venv/fla__utils.py`:
- `custom_device_ctx(index)` now returns `contextlib.nullcontext()` when
  `device_torch_lib` doesn't expose `.device(index)` (which is the case for
  `torch.cpu` and `torch.mps`).

`patches/venv/fla_modules_conv__causal_conv1d.py`:
- Added a pure-PyTorch fallback at the top of `causal_conv1d()`. When x is on
  MPS/CPU or `SANA_WM_FORCE_PURE_PYTORCH=1`, computes depthwise causal Conv1d
  via `F.conv1d` with left-padding. **Op order: conv → +bias → silu → +residual**
  (matches the Triton/CUDA reference kernel — initially had `silu(conv +
  residual)` which is non-equivalent and was the first attempt at the blur fix).

## 10. HF cache symlink

```
~/.cache/huggingface/hub/models--Efficient-Large-Model--SANA-WM_bidirectional
  → ~/.cache/sana-wm/models--Efficient-Large-Model--SANA-WM_bidirectional
```

So the inference script's `from_pretrained` doesn't re-download the 96 GB
snapshot — it points at the existing local copy.

## 11. CFG=1.0 workaround for the missing null_embed

`diffusion/model/nets/sana.py:398` tries to load `uncond_prompt_embeds[0]` from
a `.pt` file that doesn't ship in the snapshot. Fails silently. The trained
`y_embedding` (loaded from safetensors) then serves as both conditional and
unconditional. With `cfg_scale=5.0`:

```
final_eps = uncond + 5×(cond − uncond)
        = uncond + 5×(≈0)  ← (cond - uncond) is just numerical noise for
                              non-conditioning frames
        = uncond + amplified noise
```

…which corrupts the bidirectional GDN recurrence over the 11-latent sequence
and produces black/streaked frames past frame 0.

**Workaround**: pass `--cfg_scale 1.0`. Disables CFG entirely; output uses the
conditional eps directly. Quality is fine, prompt adherence is reduced.

**Future fix**: pre-compute the empty-prompt embedding from the text encoder
and patch `diffusion/model/nets/sana.py:396-409` to use it when no
`null_embed_path` is configured.

## Env vars (all required for `cfg=1.0` to work end-to-end)

```
PYTORCH_ENABLE_MPS_FALLBACK=1
SANA_WM_FORCE_PURE_PYTORCH=1
FLA_USE_PURE_PYTORCH=1
DISABLE_XFORMERS=1
GDN_DISABLE_COMPILE=1
SANA_WM_NO_COMPLEX_ROPE=1
SANA_WM_RESTORE_GDN=1
```

`run.sh` sets all of these.

## Related: subprocess staging (junafinity's approach)

For **one-shot full-pipeline rendering** (Stage 1 + LTX-2 refiner + VAE decode
in 321-frame 720p clips), the right architecture is subprocess staging. We
don't ship that here — see [`osmapi/SANA-WM-Bidirectional-on-Apple-Silicon`](https://huggingface.co/osmapi/SANA-WM-Bidirectional-on-Apple-Silicon)
for the canonical implementation. The core invariant they enforce:

> Stage 1, refiner text encoder, refiner transformer, and VAE must never all
> be resident at the same time.

Each heavy model runs in its own subprocess, writes its output latent to
disk, and exits — so macOS reclaims its pages before the next model loads.
Their reported memory profile on an M3 Max 128 GB:

| Stage | Peak RSS | MPS peak |
|---|---|---|
| Stage 1 DiT | ~12.9 GB | ~14 GB |
| LTX-2 refiner | ~28.2 GB | ~37.8 GB |
| LTX-2 VAE decode | ~3.4 GB | minimal (streaming) |

This repo's `render.py` (work-in-progress) borrows the same pattern for our
cinematic-finale mode — we keep the in-process pipeline for interactive use
(`walk.py`, `adventure.py`) but isolate each stage in its own subprocess
when rendering a final cut.

## Cleanup pass on top of the bridge

After the portability work above, an 8-agent cleanup pass made the patched code
auditable and removed several latent bugs that the original CUDA paths masked.
See the merge commits in `patches/repo.patch` for details, but the load-bearing
fixes were:

- **`assert (cond, "msg")`** in `PatchEmbed` — Python evaluates the tuple's
  truthiness, so input-shape checks never fired. Fixed to `assert cond, "msg"`.
- **Duplicate `get_scheduler`** in `longsana/utils/model_wrapper.py` — the
  second definition silently shadowed the first, so callers received an
  unbound `FlowMatchScheduler` and then called `convert_x0_to_noise` on it.
  Restored the intended `SchedulerInterface` method-binding.
- **Broken `Transformer2DModelOutput` try-imports** across 3 longsana/scheduler
  files — the guarded import was followed by an *unguarded* `isinstance` call
  that would NameError if the import had ever failed. Hoisted to top-level
  imports (`diffusers` is a hard dep).
- **Orphaned `diffusion/utils/config_wan.py`** — 150 lines of config dataclasses
  with zero consumers in the repo. Removed.
- **Import cycle** `diffusion.__init__ ↔ diffusion.utils.import_utils` — broken
  by extracting `__version__` into a leaf `diffusion/_version.py` module.
