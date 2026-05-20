#!/usr/bin/env bash
# Bootstrap a working SANA-WM checkout on macOS arm64.
#
# Idempotent: re-running will skip steps that are already done.
#
# Env vars:
#   REPO_DIR     where to clone NVlabs/Sana (default: ./repo)
#   VENV_DIR     where to create the Python venv (default: ./.venv)
#   WEIGHTS_DIR  where the 96 GB SANA-WM snapshot lives (default: ~/.cache/sana-wm)
#   PYTHON       python interpreter to use (default: python3.12 if present, else python3)
#
# Prereqs:
#   - macOS 14+ on Apple Silicon (any M1/M2/M3/M4/M5)
#   - Python 3.12 (3.11 also works, but the venv pin is 3.12)
#   - git
#   - ~96 GB free for the model snapshot (or pass WEIGHTS_DIR=/path/to/existing)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="${REPO_DIR:-$ROOT/repo}"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$HOME/.cache/sana-wm}"
PYTHON="${PYTHON:-$(command -v python3.12 || command -v python3)}"
UPSTREAM_SHA="485a6bbf7084001b3a6f736a89d217e4bb5749c3"

say() { printf "\033[1;34m==>\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m==>\033[0m %s\n" "$*" >&2; }

# 1. Clone upstream Sana at the pinned PR #379 HEAD.
if [ ! -d "$REPO_DIR/.git" ]; then
  say "Cloning NVlabs/Sana into $REPO_DIR"
  git clone https://github.com/NVlabs/Sana.git "$REPO_DIR"
  ( cd "$REPO_DIR" && git fetch origin pull/379/head:pr-379 && git checkout pr-379 && git reset --hard "$UPSTREAM_SHA" )
else
  say "Using existing checkout at $REPO_DIR"
  ( cd "$REPO_DIR" && git rev-parse HEAD ) | grep -q "$UPSTREAM_SHA" \
    || warn "  HEAD is not at PR #379 baseline ($UPSTREAM_SHA). Patch may not apply cleanly."
fi

# 2. Apply the repo patch.
if ! ( cd "$REPO_DIR" && git apply --check "$ROOT/patches/repo.patch" 2>/dev/null ); then
  if ( cd "$REPO_DIR" && git apply --check --reverse "$ROOT/patches/repo.patch" 2>/dev/null ); then
    say "Patch already applied — skipping"
  else
    warn "patches/repo.patch neither applies cleanly nor is already applied."
    warn "  Likely cause: upstream PR #379 has been rebased since SHA $UPSTREAM_SHA."
    warn "  Re-clone $REPO_DIR or open an issue with the git status output."
    exit 1
  fi
else
  say "Applying patches/repo.patch"
  ( cd "$REPO_DIR" && git apply "$ROOT/patches/repo.patch" )
fi

# 3. Create venv and install macOS-compatible deps.
if [ ! -x "$VENV_DIR/bin/python" ]; then
  say "Creating venv at $VENV_DIR (Python: $PYTHON)"
  "$PYTHON" -m venv "$VENV_DIR"
  "$VENV_DIR/bin/pip" install --upgrade pip wheel "setuptools<80"
fi

say "Installing macOS-compatible dependencies"
"$VENV_DIR/bin/pip" install --quiet torch torchvision torchaudio
"$VENV_DIR/bin/pip" install --quiet \
  diffusers==0.38.0 transformers==4.57.3 accelerate huggingface_hub safetensors \
  einops omegaconf opencv-python pillow "imageio[pyav,ffmpeg]" ffmpy pyrallis termcolor \
  ml_collections plyfile webdataset image-reward open_clip_torch hpsv2
"$VENV_DIR/bin/pip" install --quiet --no-build-isolation mmcv==1.7.2
"$VENV_DIR/bin/pip" install --quiet flash-linear-attention==0.5.0
"$VENV_DIR/bin/pip" install --quiet -e "$REPO_DIR"

# 4. Apply venv-side patches.
VENV_SITE="$($VENV_DIR/bin/python -c 'import site; print(site.getsitepackages()[0])')"
say "Applying venv patches into $VENV_SITE"
cp "$ROOT/patches/venv/fla__utils.py"                              "$VENV_SITE/fla/utils.py"
cp "$ROOT/patches/venv/fla_modules_conv__causal_conv1d.py"         "$VENV_SITE/fla/modules/conv/causal_conv1d.py"
cp "$ROOT/patches/venv/diffusers_ltx2__connectors.py"              "$VENV_SITE/diffusers/pipelines/ltx2/connectors.py"
cp "$ROOT/patches/venv/diffusers_transformers__transformer_ltx2.py" \
                                                                   "$VENV_SITE/diffusers/models/transformers/transformer_ltx2.py"

# 5. Wire up the weights cache.
HF_CACHE="$HOME/.cache/huggingface/hub/models--Efficient-Large-Model--SANA-WM_bidirectional"
WEIGHTS_TARGET="$WEIGHTS_DIR/models--Efficient-Large-Model--SANA-WM_bidirectional"
if [ -d "$WEIGHTS_TARGET" ] && [ ! -e "$HF_CACHE" ]; then
  say "Symlinking HF cache → $WEIGHTS_TARGET"
  mkdir -p "$(dirname "$HF_CACHE")"
  ln -s "$WEIGHTS_TARGET" "$HF_CACHE"
elif [ ! -d "$WEIGHTS_TARGET" ]; then
  warn "Model weights not found at $WEIGHTS_TARGET"
  warn "  Download with:"
  warn "    huggingface-cli download Efficient-Large-Model/SANA-WM_bidirectional --local-dir $WEIGHTS_TARGET"
  warn "  Or set WEIGHTS_DIR to point at an existing snapshot and re-run this script."
fi

say "Done. Run ./run.sh to generate a clip."
