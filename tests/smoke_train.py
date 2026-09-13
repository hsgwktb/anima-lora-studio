"""Smoke test: download the Anima files and run a few training steps.

    nohup python3 tests/smoke_train.py > /content/anima-lora-studio/traintest.log 2>&1 &

Uses the same code path as the WebUI's "② 训练" tab (trainer.write_dataset_toml ->
trainer.build_train_args -> trainer.start_training), but with tiny settings so it
finishes in a few minutes on a 24 GB L4: 512 px, rank 8, 6 steps.

Downloads the three ComfyUI-format files (~5.6 GB) on first run.
"""

import os
import sys

CODE = os.environ.get("ALSTUDIO_CODE", "/content/anima-lora-studio/code")
ROOT = os.environ.get("ALSTUDIO_ROOT", "/content/anima-lora-studio")
sys.path.insert(0, CODE)

import trainer  # noqa: E402

DS = os.path.join(ROOT, "datasets", "smoke")
JOB = os.path.join(ROOT, "jobs", "smoke")
LOG = os.path.join(JOB, "train.log")
os.makedirs(JOB, exist_ok=True)

if not os.path.isdir(DS) or not os.listdir(DS):
    sys.exit("dataset %s missing — run tests/smoke_tagger.py first" % DS)

print("=== downloading models ===")
print(trainer.download_models())
assert trainer.models_ready(), "models incomplete"

cfg = trainer.write_dataset_toml(os.path.join(JOB, "dataset.toml"), DS,
                                 resolution=512, batch_size=1, num_repeats=10)
cwd, argv = trainer.build_train_args(
    image_dir=DS, job_dir=JOB, output_name="smoke",
    paths=trainer.model_paths(), dataset_toml=cfg,
    rank=8, alpha=8, epochs=1, save_every_n_epochs=1, max_train_steps=6,
)
print("\n=== command ===")
print(" ".join(argv))
print("\n=== launching ===")
print(trainer.start_training("smoke", cwd, argv, LOG))
print("log:", LOG)
