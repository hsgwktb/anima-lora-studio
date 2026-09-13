"""Anima LoRA trainer driver — wraps gazingstars123/Anima-Standalone-Trainer.

The trainer ships a Node/Electron UI; we keep only its *training engine*
(``anima_train_network.py`` + ``library/`` + ``networks/lora_anima.py``), which
is an sd-scripts fork, and drive it from Python.

Invocation is deliberately all-CLI-flags rather than the repo's sectioned TOML
(``[model_arguments]``/``[anima_arguments]``/...).  ``anima_train_network.py``
accepts every one of these through ``train_util.read_config_from_file``'s
argparse path, so no section-name guessing is involved.  Only the dataset keeps
a TOML, because ``--dataset_config`` requires one.

Official Anima finetuning advice, wired in as defaults:
  * never train the LLM adapter      -> ``--llm_adapter_lr 0``
  * low learning rate for a base model -> 2e-5 at rank 32, scaled by rank here
  * Anima LoRAs are saved in ComfyUI format (the trainer does this itself)
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

# Repo-relative paths inside huggingface.co/circlestone-labs/Anima. The ComfyUI
# split files live under split_files/, not at the repo root.
MODELS = {
    "dit": "split_files/diffusion_models/anima-base-v1.0.safetensors",
    "te": "split_files/text_encoders/qwen_3_06b_base.safetensors",
    "vae": "split_files/vae/qwen_image_vae.safetensors",
}
HF_REPO = "circlestone-labs/Anima"
HF_BASE = "https://huggingface.co/%s/resolve/main/" % HF_REPO
MIN_MODEL_BYTES = 10_000_000  # anything smaller is an error page, not weights

_PROCS: dict[str, subprocess.Popen] = {}


def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v else default


def trainer_dir() -> str:
    return _env("ALSTUDIO_TRAINER_DIR", "/content/Anima-Standalone-Trainer")


def venv_python() -> str:
    return _env("ALSTUDIO_VENV_PYTHON", sys.executable)


def models_dir() -> str:
    d = _env("ALSTUDIO_MODELS_DIR", "/content/anima-models")
    os.makedirs(d, exist_ok=True)
    return d


def model_paths() -> dict:
    d = models_dir()
    return {k: os.path.join(d, os.path.basename(v)) for k, v in MODELS.items()}


def models_ready() -> bool:
    return all(os.path.exists(p) and os.path.getsize(p) > MIN_MODEL_BYTES
               for p in model_paths().values())


def download_models(log_path: str | None = None) -> str:
    """Fetch the three ComfyUI-format Anima files. Idempotent (curl -C -)."""
    d = models_dir()
    lines = []
    for rel in MODELS.values():
        dest = os.path.join(d, os.path.basename(rel))
        if os.path.exists(dest) and os.path.getsize(dest) > MIN_MODEL_BYTES:
            lines.append("have %s (%.2f GB)" % (os.path.basename(dest), os.path.getsize(dest) / 1e9))
            continue
        # A previous failed attempt may have left an error page behind; resuming
        # onto it (curl -C -) would produce a corrupt file.
        if os.path.exists(dest) and os.path.getsize(dest) <= MIN_MODEL_BYTES:
            os.remove(dest)
        cmd = ["curl", "-L", "-C", "-", "--retry", "3", "-o", dest, HF_BASE + rel]
        lines.append("$ " + " ".join(cmd))
        p = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        size = os.path.getsize(dest) if os.path.exists(dest) else 0
        if p.returncode != 0 or size <= MIN_MODEL_BYTES:
            lines.append("ERROR downloading %s (rc=%s, %d bytes)\n%s"
                         % (rel, p.returncode, size, (p.stderr or "")[-500:]))
            return "\n".join(lines)
        lines.append("got %s (%.2f GB)" % (os.path.basename(dest), size / 1e9))
    return "\n".join(lines)


def write_dataset_toml(
    path: str,
    image_dir: str,
    resolution: int = 1024,
    batch_size: int = 1,
    num_repeats: int = 10,
    keep_tokens: int = 1,
    caption_dropout_rate: float = 0.05,
    caption_tag_dropout_rate: float = 0.0,
    # sd-scripts refuses shuffle_caption / caption_tag_dropout_rate while
    # caching text-encoder outputs, so the default (cache ON) must keep this off.
    shuffle_caption: bool = False,
    flip_aug: bool = False,
    min_bucket: int = 512,
    max_bucket: int = 1536,
) -> str:
    toml = f"""[general]
enable_bucket = true
bucket_no_upscale = true
min_bucket_reso = {min_bucket}
max_bucket_reso = {max_bucket}
bucket_reso_steps = 64

[[datasets]]
resolution = [{resolution}, {resolution}]
batch_size = {batch_size}
caption_extension = ".txt"

  [[datasets.subsets]]
  image_dir = "{image_dir}"
  num_repeats = {num_repeats}
  keep_tokens = {keep_tokens}
  flip_aug = {str(flip_aug).lower()}
  shuffle_caption = {str(shuffle_caption).lower()}
  caption_tag_dropout_rate = {caption_tag_dropout_rate}
  caption_dropout_rate = {caption_dropout_rate}
  caption_dropout_every_n_epochs = 0
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(toml)
    return path


def default_lr(rank: int) -> float:
    """Anima's card: 2e-5 at rank 32 for a base model; scale with rank."""
    return round(2e-5 * max(rank, 1) / 32, 8)


def build_train_args(
    image_dir: str,
    job_dir: str,
    output_name: str,
    paths: dict,
    dataset_toml: str,
    rank: int = 16,
    alpha: int = 16,
    epochs: int = 10,
    save_every_n_epochs: int = 1,
    learning_rate: float | None = None,
    text_encoder_lr: float = 0.0,
    llm_adapter_lr: float = 0.0,
    optimizer: str = "AdamW8bit",
    lr_scheduler: str = "cosine",
    warmup_ratio: float = 0.05,
    batch_size: int = 1,
    gradient_accumulation: int = 1,
    mixed_precision: str = "bf16",
    gradient_checkpointing: bool = True,
    blocks_to_swap: int = 0,
    seed: int = 42,
    max_train_steps: int = 0,
    discrete_flow_shift: float = 3.0,
    timestep_sample_method: str = "logit_normal",
    max_token_length: int = 512,
    save_precision: str = "bf16",
    cache: bool = True,
) -> tuple[str, list[str]]:
    """Returns (cwd, argv) for anima_train_network.py."""
    os.makedirs(job_dir, exist_ok=True)
    out_dir = os.path.join(job_dir, "output")
    log_dir = os.path.join(job_dir, "logs")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    if learning_rate is None:
        learning_rate = default_lr(rank)

    script = os.path.join(trainer_dir(), "anima_train_network.py")
    argv = [
        venv_python(), script,
        "--dit_path=%s" % paths["dit"],
        "--vae_path=%s" % paths["vae"],
        "--qwen3_path=%s" % paths["te"],
        "--dataset_config=%s" % dataset_toml,
        "--output_dir=%s" % out_dir,
        "--output_name=%s" % output_name,
        "--logging_dir=%s" % log_dir,
        "--log_with=tensorboard",
        "--save_model_as=safetensors",
        "--save_precision=%s" % save_precision,
        "--save_every_n_epochs=%d" % save_every_n_epochs,
        "--learning_rate=%s" % learning_rate,
        "--text_encoder_lr=%s" % text_encoder_lr,
        "--llm_adapter_lr=%s" % llm_adapter_lr,
        "--optimizer_type=%s" % optimizer,
        "--lr_scheduler=%s" % lr_scheduler,
        "--mixed_precision=%s" % mixed_precision,
        "--network_module=networks.lora_anima",
        "--network_dim=%d" % rank,
        "--network_alpha=%d" % alpha,
        "--network_train_unet_only",
        "--timestep_sample_method=%s" % timestep_sample_method,
        "--discrete_flow_shift=%s" % discrete_flow_shift,
        "--qwen3_max_token_length=%d" % max_token_length,
        "--seed=%d" % seed,
        "--max_data_loader_n_workers=2",
        "--persistent_data_loader_workers",
    ]
    if cache:
        # Caching the Qwen3 text-encoder outputs keeps the TE off the GPU during
        # training (big VRAM saving) at the cost of no caption shuffling.
        argv += ["--cache_latents_to_disk",
                 "--cache_text_encoder_outputs",
                 "--cache_text_encoder_outputs_to_disk"]
    if gradient_checkpointing:
        argv.append("--gradient_checkpointing")
    if gradient_accumulation > 1:
        argv.append("--gradient_accumulation_steps=%d" % gradient_accumulation)
    if batch_size > 1:
        argv.append("--max_batch_size=%d" % batch_size)
    if warmup_ratio:
        argv.append("--lr_warmup_steps=%d" % max(1, int(warmup_ratio * 100)))
    if blocks_to_swap > 0:
        argv.append("--blocks_to_swap=%d" % blocks_to_swap)
    if max_train_steps > 0:
        argv.append("--max_train_steps=%d" % max_train_steps)
    else:
        argv.append("--max_train_epochs=%d" % epochs)
    return trainer_dir(), argv


def start_training(job_id: str, cwd: str, argv: list[str], log_path: str) -> str:
    if job_id in _PROCS and _PROCS[job_id].poll() is None:
        return "already running (pid %d)" % _PROCS[job_id].pid
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("HF_HOME", "/content/.cache/huggingface")
    lf = open(log_path, "w", encoding="utf-8", errors="replace")
    lf.write("$ cd %s && %s\n\n" % (cwd, " ".join(argv)))
    lf.flush()
    p = subprocess.Popen(argv, cwd=cwd, stdout=lf, stderr=subprocess.STDOUT,
                         env=env, start_new_session=True)
    _PROCS[job_id] = p
    return "pid=%d" % p.pid


def stop_training(job_id: str) -> str:
    p = _PROCS.get(job_id)
    if not p or p.poll() is not None:
        return "not running"
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    except Exception:
        p.terminate()
    return "sent SIGTERM to pid %d" % p.pid


def status(job_id: str) -> str:
    p = _PROCS.get(job_id)
    if not p:
        return "idle"
    rc = p.poll()
    if rc is None:
        return "running (pid %d)" % p.pid
    _PROCS.pop(job_id, None)
    return "finished (exit %s)" % rc


def is_running(job_id: str) -> bool:
    p = _PROCS.get(job_id)
    return bool(p and p.poll() is None)


def tail_log(log_path: str, n: int = 120) -> str:
    if not os.path.exists(log_path):
        return "(no log yet)"
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError as e:
        return "ERR %r" % e
    return "".join(lines[-n:])


_PROG = re.compile(r"steps:\s*(\d+)%.*?(\d+)\s*/\s*(\d+)")
_LOSS = re.compile(r"avr_loss=([0-9.]+)")


def progress(log_path: str) -> str:
    """Cheap progress line for the UI (parsed from the sd-scripts log)."""
    txt = tail_log(log_path, 400)
    if not txt.strip():
        return "waiting for output…"
    m = None
    for m2 in _PROG.finditer(txt):
        m = m2
    loss = _LOSS.findall(txt)
    parts = []
    if m:
        parts.append("step %s/%s (%s%%)" % (m.group(2), m.group(3), m.group(1)))
    if loss:
        parts.append("avr_loss=%s" % loss[-1])
    last = [l for l in txt.strip().splitlines() if l.strip()][-1:]
    if last:
        parts.append(last[0][-160:])
    return " · ".join(parts) if parts else txt.strip().splitlines()[-1][-200:]


def list_outputs(job_dir: str) -> list[str]:
    out = os.path.join(job_dir, "output")
    if not os.path.isdir(out):
        return []
    return [os.path.join(out, f) for f in sorted(os.listdir(out))
            if f.endswith(".safetensors")]
