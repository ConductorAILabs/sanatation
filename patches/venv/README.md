# Venv-side patches

These are drop-in replacements for files inside two installed packages (`fla`
and `diffusers`). They cover MPS gaps that the upstream packages haven't fixed
yet. `../../apply-patches.sh` copies them into your venv automatically; this
README exists so you can restore them by hand after rebuilding the venv.

## Restore by hand

```bash
VENV_SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"

cp fla__utils.py                              "$VENV_SITE/fla/utils.py"
cp fla_modules_conv__causal_conv1d.py         "$VENV_SITE/fla/modules/conv/causal_conv1d.py"
cp diffusers_ltx2__connectors.py              "$VENV_SITE/diffusers/pipelines/ltx2/connectors.py"
cp diffusers_transformers__transformer_ltx2.py \
                                              "$VENV_SITE/diffusers/models/transformers/transformer_ltx2.py"
```

## What each file changes

See `../../PATCHES_TECHNICAL.md` sections 5 and 9. Summary:

- **`fla/utils.py`** — `custom_device_ctx()` returns `nullcontext` on MPS/CPU
  instead of crashing because `torch.cpu.device(...)` doesn't exist.
- **`fla/modules/conv/causal_conv1d.py`** — pure-PyTorch causal-conv1d fallback
  when input is on MPS/CPU. Uses `F.conv1d` with left-padding.
- **`diffusers/pipelines/ltx2/connectors.py`** — RoPE `freqs_dtype` downcasts
  to fp32 on MPS instead of fp64.
- **`diffusers/models/transformers/transformer_ltx2.py`** — same fp64→fp32
  downcast in the transformer's RoPE.

## Version pins

These patches were authored against:

- `fla` (flash-linear-attention) `0.5.0`
- `diffusers` `0.38.0`

If your installed versions differ, the patches may not apply cleanly — open
an issue with the diff.
