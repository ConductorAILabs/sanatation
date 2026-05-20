#!/usr/bin/env bash
# Run SANA-WM Stage-1 inference on Apple Silicon.
#
# Env vars (all optional):
#   REPO_DIR   path to patched NVlabs/Sana checkout (default: ./repo)
#   VENV_DIR   path to venv built by apply-patches.sh (default: ./.venv)
#   STEPS      diffusion steps; 20 = full quality, 4 = preview (default: 20)
#   CFG        cfg_scale; KEEP AT 1.0 until null_embed workaround lands (default: 1.0)
#   NUM_FRAMES total frames; must be 8k+1 — script auto-snaps if not (default: 80)
#   NAME       output basename (default: sana-wm-HHMMSS)
#
# First positional arg is the camera action string; see README for the DSL.
#
# Examples:
#   ./run.sh                              # default demo
#   ./run.sh "w-30,a-20,jw-20,d-10"        # custom trajectory
#   CFG=1.0 STEPS=4 NAME=preview ./run.sh "rw-80"

set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="${REPO_DIR:-$ROOT/repo}"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
OUT="$ROOT/outputs"

if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "FATAL: venv not found at $VENV_DIR" >&2
  echo "  Run ./apply-patches.sh first." >&2
  exit 1
fi

if [ ! -d "$REPO_DIR/diffusion" ]; then
  echo "FATAL: patched repo not found at $REPO_DIR" >&2
  echo "  Run ./apply-patches.sh first." >&2
  exit 1
fi

# All bridge-required env vars in one place.
export PYTORCH_ENABLE_MPS_FALLBACK=1
export SANA_WM_FORCE_PURE_PYTORCH=1
export FLA_USE_PURE_PYTORCH=1
export DISABLE_XFORMERS=1
export GDN_DISABLE_COMPILE=1
export SANA_WM_NO_COMPLEX_ROPE=1
export SANA_WM_RESTORE_GDN=1   # use trained 15-GDN + 5-softmax hybrid (not all-softmax)

mkdir -p "$OUT"

cd "$REPO_DIR"
exec "$VENV_DIR/bin/python" inference_video_scripts/inference_sana_wm.py \
  --image      "$REPO_DIR/asset/sana_wm/demo_1.png" \
  --prompt     "$REPO_DIR/asset/sana_wm/demo_1.txt" \
  --intrinsics "$REPO_DIR/asset/sana_wm/demo_1_intrinsics.npy" \
  --action     "${1:-w-40,jw-20,w-20}" \
  --num_frames "${NUM_FRAMES:-80}" \
  --step       "${STEPS:-20}" \
  --cfg_scale  "${CFG:-1.0}" \
  --flow_shift 8.0 \
  --no_refiner --offload_vae \
  --output_dir "$OUT" \
  --name       "${NAME:-sana-wm-$(date +%H%M%S)}"
