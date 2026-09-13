#!/usr/bin/env bash
# anima-lora-studio — one-shot Colab/Linux setup + launch.
#
#   bash colab_setup.sh            # setup if needed, then (re)start the WebUI
#   bash colab_setup.sh --install  # force the expensive dependency install
#
# Everything is pulled from GitHub (this repo + Anima-Standalone-Trainer); no
# base64 payloads are pushed into the notebook.
#
# Venv note: Colab's Python 3.13 cannot bootstrap pip (`ensurepip` exits 1), so
# we build the environment with `uv`, which also fetches a CPython 3.12 — the
# version Anima-Standalone-Trainer is developed against.
set -uo pipefail

ROOT="${ALSTUDIO_ROOT:-/content/anima-lora-studio}"          # app + venv + datasets + jobs
CODE="${ALSTUDIO_CODE:-$ROOT/code}"                          # this repo's checkout
TRAINER="${ALSTUDIO_TRAINER_DIR:-/content/Anima-Standalone-Trainer}"
VENV="$ROOT/.venv"
PYVER="${ALSTUDIO_PYVER:-3.12}"
PORT="${ALSTUDIO_PORT:-8000}"
REPO_URL="${ALSTUDIO_REPO:-https://github.com/hsgwktb/anima-lora-studio.git}"
TRAINER_URL="https://github.com/gazingstars123/Anima-Standalone-Trainer.git"
LOG="$ROOT/app.log"
CFD="$ROOT/cloudflared"

say() { echo -e "\n\033[1;36m=== $* ===\033[0m"; }

mkdir -p "$ROOT"
export PATH="$HOME/.local/bin:$PATH"

# ---------------------------------------------------------------- source code
say "fetching code"
if [ -d "$CODE/.git" ]; then
  git -C "$CODE" pull --ff-only -q && echo "updated $CODE"
else
  git clone --depth 1 -q "$REPO_URL" "$CODE" && echo "cloned $CODE"
fi
if [ ! -d "$TRAINER/.git" ]; then
  say "cloning Anima-Standalone-Trainer"
  git clone --depth 1 -q "$TRAINER_URL" "$TRAINER"
fi

# ------------------------------------------------------------------------- uv
if ! command -v uv >/dev/null 2>&1; then
  say "installing uv"
  pip install -q uv 2>/dev/null || python3 -m pip install -q uv 2>/dev/null \
    || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || { echo "FATAL: uv unavailable"; exit 1; }
echo "uv: $(uv --version)"

# --------------------------------------------------------------------- install
# A half-built venv (Colab's ensurepip failure leaves bin/python behind) must not
# count as installed, so success is recorded in a marker file.
MARKER="$VENV/.alstudio-installed"
NEED_INSTALL=0
[ -f "$MARKER" ] || NEED_INSTALL=1
[ "${1:-}" = "--install" ] && NEED_INSTALL=1

if [ "$NEED_INSTALL" = "1" ]; then
  say "creating venv (python $PYVER)"
  rm -rf "$VENV"
  uv venv --python "$PYVER" "$VENV" || uv venv "$VENV" || { echo "FATAL: venv"; exit 1; }
  "$VENV/bin/python" -V

  # unsafe-best-match: the requirements use an extra index (pytorch cu128); uv
  # otherwise refuses to fall back to PyPI when a package exists on the first
  # index at a different version (it failed on setuptools==80.0.0 this way).
  vpip() { uv pip install --python "$VENV/bin/python" --index-strategy unsafe-best-match "$@"; }

  # cuda_direct_pkg / wd_parallel_pkg are optional native extras: try them, but
  # never let them abort the install. Index flags are passed explicitly because
  # they are stripped from the generated requirements file below.
  say "installing Anima-Standalone-Trainer requirements (torch cu128, ~5-8 min)"
  grep -v -e '^\s*#' -e '^\s*--' -e '^\./cuda_direct_pkg' -e '^\./wd_parallel_pkg' \
      -e '^\s*$' "$TRAINER/requirements.txt" > "$ROOT/req-core.txt"
  ( cd "$TRAINER" && vpip -r "$ROOT/req-core.txt" \
      --extra-index-url https://download.pytorch.org/whl/cu128 ) \
      || { echo "FATAL: core requirements failed"; exit 1; }
  ( cd "$TRAINER" && vpip ./cuda_direct_pkg ./wd_parallel_pkg ) \
      || echo "WARN: optional local packages failed (not required for LoRA)"

  say "installing onnxruntime for the WD14-v3 tagger"
  vpip onnx onnxruntime-gpu "numpy<2.3" \
      --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ \
      || vpip onnx onnxruntime || echo "WARN: onnxruntime install failed"

  say "install verification"
  "$VENV/bin/python" - <<'PY' || echo "WARN: some imports failed"
import importlib
for m in ("torch", "transformers", "diffusers", "accelerate", "onnxruntime", "safetensors"):
    try:
        mod = importlib.import_module(m)
        print("  %-14s %s" % (m, getattr(mod, "__version__", "ok")))
    except Exception as e:
        print("  %-14s MISSING (%s)" % (m, e))
PY
  touch "$MARKER"
else
  echo "venv already present ($VENV) — pass --install to rebuild"
fi

# ------------------------------------------------------------------ UI runtime
# The Gradio UI runs on the *system* python so it never collides with the
# trainer's pinned torch/transformers stack.
python3 -c "import gradio" 2>/dev/null || pip install -q gradio

# ---------------------------------------------------------------------- tunnel
if [ ! -x "$CFD" ]; then
  say "downloading cloudflared"
  curl -sL -o "$CFD" https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
    && chmod +x "$CFD"
fi

# ---------------------------------------------------------------------- launch
say "starting WebUI on :$PORT"
pkill -f "app.py --port $PORT" 2>/dev/null
sleep 1
ALSTUDIO_ROOT="$ROOT" \
ALSTUDIO_CODE="$CODE" \
ALSTUDIO_TRAINER_DIR="$TRAINER" \
ALSTUDIO_VENV_PYTHON="$VENV/bin/python" \
ALSTUDIO_TAGGER_SCRIPT="$CODE/vendor/tag_images_by_wd14_tagger.py" \
nohup python3 -u "$CODE/app.py" --port "$PORT" > "$LOG" 2>&1 &

for i in $(seq 1 40); do
  curl -s -o /dev/null "http://127.0.0.1:$PORT/" && break
  sleep 2
done
echo "--- app.log ---"; tail -8 "$LOG"

say "starting cloudflared quick tunnel"
pkill -f "cloudflared tunnel --url http://127.0.0.1:$PORT" 2>/dev/null
nohup "$CFD" tunnel --url "http://127.0.0.1:$PORT" --no-autoupdate > "$ROOT/tunnel.log" 2>&1 &
for i in $(seq 1 30); do
  URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$ROOT/tunnel.log" | head -1)
  [ -n "${URL:-}" ] && break
  sleep 2
done
echo
echo "================================================================"
echo "  Anima LoRA Studio:  ${URL:-<tunnel not up, see $ROOT/tunnel.log>}"
echo "  local:              http://127.0.0.1:$PORT"
echo "  venv python:        $VENV/bin/python"
echo "================================================================"
