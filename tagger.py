"""Anima LoRA tagger — the Dataset_Maker tagging recipe, driven from Python.

Dataset_Maker (hollowstrawberry/kohya-colab) tags with kohya's
``finetune/tag_images_by_wd14_tagger.py`` against the SmilingWolf **v3** taggers
(Tagger v2, from 2023, was dropped upstream on 2025-03-31 — its vocabulary is too
old for Anima, whose anime data cuts off in Sept 2025).

This module reproduces that recipe exactly:

  python tag_images_by_wd14_tagger.py <dir> \
      --repo_id=<tagger> --general_threshold=<t> --character_threshold=<ct> \
      --batch_size=8 --max_data_loader_n_workers=2 --caption_extension=.txt \
      --undesired_tags "<csv>" --onnx --recursive --remove_underscore [--append_tags]

"Both" runs wd-eva02-large-tagger-v3 first and wd-vit-large-tagger-v3 second with
``--append_tags`` so the two tag sets are merged (upstream does the same).

``--remove_underscore`` is what makes the output match Anima's caption spec
(lowercase tags, spaces instead of underscores).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".jxl", ".avif"}

# v3 taggers only. Listed newest/best first.
TAGGER_CHOICES = [
    "Both (eva02 + vit-large)",
    "SmilingWolf/wd-eva02-large-tagger-v3",
    "SmilingWolf/wd-vit-large-tagger-v3",
    "SmilingWolf/wd-swinv2-tagger-v3",
]

TAGGER_MAP = {
    "Both (eva02 + vit-large)": [
        "SmilingWolf/wd-eva02-large-tagger-v3",
        "SmilingWolf/wd-vit-large-tagger-v3",
    ],
    "SmilingWolf/wd-eva02-large-tagger-v3": ["SmilingWolf/wd-eva02-large-tagger-v3"],
    "SmilingWolf/wd-vit-large-tagger-v3": ["SmilingWolf/wd-vit-large-tagger-v3"],
    "SmilingWolf/wd-swinv2-tagger-v3": ["SmilingWolf/wd-swinv2-tagger-v3"],
}

# Upstream Dataset_Maker default blacklist — generic concepts that hurt a
# single-concept/style LoRA.
DEFAULT_BLACKLIST = (
    "virtual youtuber, parody, style parody, official alternate costume, "
    "official alternate hairstyle, official alternate hair length, alternate costume, "
    "alternate hairstyle, alternate hair length, alternate hair color"
)


def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v else default


def venv_python() -> str:
    return _env("ALSTUDIO_VENV_PYTHON", sys.executable)


def tagger_script() -> str:
    """kohya's WD14 tagger. It imports ``library.dataset`` / ``library.utils``, so
    it must run from a full sd-scripts checkout (colab_setup.sh clones one)."""
    return _env("ALSTUDIO_TAGGER_SCRIPT",
                "/content/sd-scripts/finetune/tag_images_by_wd14_tagger.py")


def tagger_root() -> str:
    """Repo root the tagger must run from (its ``library/`` package lives there)."""
    return str(Path(tagger_script()).resolve().parent.parent)


def list_images(image_dir: str) -> list[str]:
    out = []
    for root, _dirs, files in os.walk(image_dir):
        for f in sorted(files):
            if Path(f).suffix.lower() in IMAGE_EXTS:
                out.append(os.path.join(root, f))
    return out


def list_captions(image_dir: str) -> list[str]:
    out = []
    for root, _dirs, files in os.walk(image_dir):
        for f in sorted(files):
            if f.lower().endswith(".txt"):
                out.append(os.path.join(root, f))
    return out


def tag_images(
    image_dir: str,
    tagger: str = "Both (eva02 + vit-large)",
    general_threshold: float = 0.25,
    character_threshold: float = 1.1,
    undesired_tags: str = DEFAULT_BLACKLIST,
    remove_underscore: bool = True,
    recursive: bool = True,
    batch_size: int = 8,
    extras: str = "",
) -> str:
    """Run the Dataset_Maker tagging pass(es). Returns the combined log."""
    models = TAGGER_MAP.get(tagger, [tagger])
    images = list_images(image_dir)
    if not images:
        return "ERROR: no images found in %s" % image_dir

    log_lines = [
        "images=%d  taggers=%s  general_threshold=%s  character_threshold=%s"
        % (len(images), ", ".join(models), general_threshold, character_threshold)
    ]
    blacklist = ",".join(t.strip() for t in (undesired_tags or "").split(",") if t.strip())

    for i, model in enumerate(models):
        cmd = [
            venv_python(), tagger_script(), image_dir,
            "--repo_id=%s" % model,
            "--general_threshold=%s" % general_threshold,
            "--character_threshold=%s" % character_threshold,
            "--batch_size=%d" % batch_size,
            "--max_data_loader_n_workers=2",
            "--caption_extension=.txt",
            "--onnx",
        ]
        if recursive:
            cmd.append("--recursive")
        if remove_underscore:
            cmd.append("--remove_underscore")
        if i > 0:
            cmd.append("--append_tags")  # merge second tagger into the same .txt
        if blacklist:
            cmd += ["--undesired_tags", blacklist]
        if extras.strip():
            cmd += extras.split()

        log_lines.append("\n$ " + " ".join(cmd))
        env = dict(os.environ)
        root = tagger_root()
        env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                              cwd=root, env=env)
        tail = (proc.stdout or "")[-4000:] + (proc.stderr or "")[-4000:]
        log_lines.append(tail.strip())
        if proc.returncode != 0:
            log_lines.append("ERROR: tagger exited %d" % proc.returncode)
            return "\n".join(log_lines)

    stats = tag_stats(image_dir)
    if stats:
        log_lines.append("\n📊 top 40 tags:")
        log_lines.append("\n".join("%s (%d)" % (t, n) for t, n in stats.most_common(40)))
    return "\n".join(log_lines)


def tag_stats(image_dir: str) -> Counter:
    c = Counter()
    for txt in list_captions(image_dir):
        try:
            with open(txt, encoding="utf-8") as f:
                for t in f.read().split(","):
                    t = t.strip()
                    if t:
                        c[t] += 1
        except OSError:
            pass
    return c


def clean_tags(
    image_dir: str,
    remove_tags: str = "",
    add_prefix: str = "",
    dedupe: bool = True,
    sort_alpha: bool = False,
    ensure_anima_prefix: bool = True,
) -> str:
    """Dataset_Maker's "extras" editing pass, plus Anima's recommended prefix.

    Anima was trained on Danbooru tags; its model card recommends the positive
    prefix ``masterpiece, best quality, score_7, safe, ``. Removing redundant
    tags and adding that prefix is the whole point of this step.
    """
    remove = {t.strip().lower() for t in (remove_tags or "").split(",") if t.strip()}
    prefix = [t.strip() for t in (add_prefix or "").split(",") if t.strip()]
    if ensure_anima_prefix:
        for t in ("masterpiece", "best quality"):
            if t not in prefix:
                prefix.append(t)

    changed = 0
    for txt in list_captions(image_dir):
        try:
            with open(txt, encoding="utf-8") as f:
                raw = f.read()
        except OSError:
            continue
        tags = [t.strip() for t in raw.split(",") if t.strip()]
        out, seen = [], set()
        for t in tags:
            key = t.lower()
            if key in remove:
                continue
            if dedupe and key in seen:
                continue
            seen.add(key)
            out.append(t)
        if sort_alpha:
            out.sort(key=str.lower)
        # prefix first, no duplicates
        ordered = [p for p in prefix if p.lower() not in seen] + out
        new = ", ".join(ordered)
        if new != raw:
            with open(txt, "w", encoding="utf-8") as f:
                f.write(new)
            changed += 1
    return "cleaned %d caption file(s)" % changed


def preview_captions(image_dir: str, limit: int = 12) -> tuple[list[str], list[str], str]:
    """(image paths, paired captions, stats text) for the gallery."""
    imgs, caps = [], []
    for img in list_images(image_dir)[:limit]:
        txt = os.path.splitext(img)[0] + ".txt"
        cap = ""
        if os.path.exists(txt):
            with open(txt, encoding="utf-8") as f:
                cap = f.read().strip()
        imgs.append(img)
        caps.append(cap)
    stats = tag_stats(image_dir)
    summary = "**%d images · %d captions**" % (len(list_images(image_dir)), len(list_captions(image_dir)))
    if stats:
        summary += "\n\nTop tags: " + ", ".join("%s(%d)" % (t, n) for t, n in stats.most_common(25))
    return imgs, caps, summary
