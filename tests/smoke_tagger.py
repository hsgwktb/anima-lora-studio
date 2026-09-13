"""Smoke test: run the tagger the same way the WebUI does.

    python3 tests/smoke_tagger.py [n_images]

Fetches a few real photos (picsum.photos) into datasets/smoke/, runs the
Dataset_Maker tagging recipe through tagger.py, and prints the captions.

Why photos and not anime: the point is to prove the *pipeline* (model download ->
onnxruntime-gpu inference -> per-image .txt -> tag stats). Tag content for a
non-anime photo is naturally poor; feed it your own dataset for real results.

Run it as a background job (it downloads ~2.4 GB of tagger weights on first run):

    nohup python3 tests/smoke_tagger.py > /content/anima-lora-studio/tagtest.log 2>&1 &
"""

import os
import subprocess
import sys

CODE = os.environ.get("ALSTUDIO_CODE", "/content/anima-lora-studio/code")
sys.path.insert(0, CODE)

os.environ.setdefault("ALSTUDIO_VENV_PYTHON", "/content/anima-lora-studio/.venv/bin/python")
os.environ.setdefault("ALSTUDIO_TAGGER_SCRIPT",
                      "/content/sd-scripts/finetune/tag_images_by_wd14_tagger.py")

import tagger  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
DST = os.path.join(os.environ.get("ALSTUDIO_ROOT", "/content/anima-lora-studio"),
                   "datasets", "smoke")
os.makedirs(DST, exist_ok=True)

for i in range(N):
    p = os.path.join(DST, "img%d.jpg" % i)
    if not os.path.exists(p):
        subprocess.run(["curl", "-sL", "-o", p,
                        "https://picsum.photos/seed/anima%d/768" % i], check=False)

imgs = tagger.list_images(DST)
print("images: %d" % len(imgs))
if not imgs:
    sys.exit("no images downloaded (network?)")

print("tagger python:", tagger.venv_python())
print("tagger script:", tagger.tagger_script())
print(tagger.tag_images(DST, tagger="Both (eva02 + vit-large)"))

preview_imgs, caps, summary = tagger.preview_captions(DST, limit=8)
for i, c in zip(preview_imgs, caps):
    print("--- %s\n%s" % (os.path.basename(i), c[:500]))
print("\n" + summary)

n_caps = len(tagger.list_captions(DST))
print("\nRESULT: %d images / %d caption files" % (len(imgs), n_caps))
print("SMOKE_TAGGER_OK" if n_caps >= len(imgs) else "SMOKE_TAGGER_FAIL")
