#!/usr/bin/env bash
# anima-lora-studio — one-shot Colab/Linux setup + launch.
#
#   bash colab_setup.sh            # setup if needed, then (re)start the WebUI
#   bash colab_setup.sh --install  # force the expensive dependency install
#
# Everything is pulled from GitHub (this repo + Anima-Standalone-Trainer); no
# base64 payloads are pushed into the notebook.
set -uo pipefail

ROOT="${ALSTUDIO_ROOT:-/content/anima-lora-studio}"          # app + venv + datasets + jobs
CODE="${ALSTUDIO_CODE:-$ROOT/code}"                          # this repo's checkout
TRAINER="${ALSTUDIO_TRAINER_DIR:-/content/Anima-Standalone-Trainer}"
VENV="$ROOT/.venv"
PORT="${ALSTUDIO_PORT:-8000}"
REPO_URL="${ALSTUDIO_REPO:-https://github.com/hsgwktb/anima-lora-studio.git}"
TRAINER_URL="https://github.com/gazingstars123/Anima-Standalone-Trainer.git"
LOG="$ROOT/app.log"
CFD="$ROOT/cloudflared"

say() { echo -e "\n\033[1;36m=== $* ===\033[0m"; }

mkdir -p "$ROOT"

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

# --------------------------------------------------------------------- flask/venv
NEED_INSTALL=0
[ -x "$VENV/bin/python" ] || NEED_INSTALL=1
[ "${1:-}" = "--install" ] && NEED_INSTALL=1
command -v python3 >/dev/null || { echo "python3 missing"; exit 1; }

if [ "$NEED_INSTALL" = "1" ]; then
  say "creating venv (python3 = $(python3 -V))"
  python3 -m venv --clear "$VENV"
  "$VENV/bin/pip" install -q -U pip wheel setuptools

  say "installing Anima-Standalone-Trainer requirements (torch 2.7.0+cu128, ~5-8 min)"
  # cuda_direct_pkg / wd_parallel_pkg are optional native extras: try them, but
  # never let them abort the install.
  grep -v -e '^\./cuda_direct_pkg' -e '^\./wd_parallel_pkg' \
      "$TRAINER/requirements.txt" > "$ROOT/req-core.txt"
  ( cd "$TRAINER" && "$VENV/bin/pip" install -r "$ROOT/req-core.txt" ) \
      || { echo "FATAL: core requirements failed"; exit 1; }
  ( cd "$TRAINER" && "$VENV/bin/pip" install ./cuda_direct_pkg ./wd_parallel_pkg ) \
      || echo "WARN: optional local packages failed (not required for LoRA)"

  say "installing onnxruntime for the WD14-v3 tagger"
  "$VENV/bin/pip" install -q onnx onnxruntime-gpu "numpy<2.3" \
      --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ \
      || "$VENV/bin/pip" install -q onnx onnxruntime
else
  echo "venv already present ($VENV) — pass --install to rebuild"
fi

# ------------------------------------------------------------------ UI runtime
# The Gradio UI runs on the *system* python so it never collides with the
# trainer's pinned torch/transformers stack.
python3 - <<'PY' || pip install -q gradio
import gradio  # noqa
PY

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
  if curl -s -o /dev/null "http://127.0.0.1:$PORT/"; then break; fi
  sleep 2
done
echo "--- app.log ---"; tail -5 "$LOG"

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
