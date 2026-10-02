# -*- coding: utf-8 -*-
"""生成 F:\\我的文档\\Documents\\Computer\\skills\\Anima-LoRA-Colab一键部署.ipynb

单元正文一律用 raw 三引号包裹：写进生成器的内容 = 写进笔记本的内容，避免转义层层叠加。
"""
import json
import os

OUT = r"F:\我的文档\Documents\Computer\skills\Anima-LoRA-Colab一键部署.ipynb"
# 同一份内容也放进仓库（README 指向它）。
OUT_REPO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "anima_lora_studio.ipynb")

TITLE = r"""
# Anima LoRA Studio 一键部署到 Colab L4（24GB）
### Dataset_Maker 打标 + Anima-Standalone-Trainer 训练的**双选项卡 WebUI**

> 打标引擎：kohya `tag_images_by_wd14_tagger.py` + SmilingWolf **v3** 打标器
> （`wd-eva02-large-tagger-v3` + `wd-vit-large-tagger-v3`，即 [Dataset_Maker](https://github.com/hollowstrawberry/kohya-colab) 的配方）
> 训练引擎：[Anima-Standalone-Trainer](https://github.com/gazingstars123/Anima-Standalone-Trainer) 的 `anima_train_network.py`
> 代码仓库：https://github.com/hsgwktb/anima-lora-studio
> 记录日期：2026-09-13 · 环境：Colab Pro（L4 24GB）

**怎么用**：从上往下依次运行。第 1–4 格是安装（首次约 5–8 分钟），第 5 格给出 WebUI 地址。
之后在网页里：① 上传图片打标 → 点「发送到训练器」→ ② 下载模型、设参数、开始训练。

运行前确认「代码执行程序 → 更改运行时类型」是 **GPU**（本笔记本元数据已声明 GPU，
但如果你手动改过，请改回来）。全程代码从 GitHub 拉取，本笔记本不含任何 base64 载荷。
"""

ENVCHECK = r'''
import json, os, re, shutil, subprocess, sys, textwrap, time

def sh(cmd, check=False, timeout=None, cwd=None, stream=False):
    """跑一条 shell 命令，返回 (returncode, 输出)。stream=True 时实时打印。"""
    if stream:
        p = subprocess.Popen(cmd, shell=True, cwd=cwd, executable="/bin/bash",
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1)
        buf = []
        for line in p.stdout:
            print(line, end="")
            buf.append(line)
        p.wait()
        return p.returncode, "".join(buf)
    p = subprocess.run(cmd, shell=True, cwd=cwd, executable="/bin/bash",
                       capture_output=True, text=True, timeout=timeout)
    out = (p.stdout or "") + (p.stderr or "")
    if check and p.returncode != 0:
        raise RuntimeError("命令失败(%d): %s\n%s" % (p.returncode, cmd, out[-2000:]))
    return p.returncode, out

print("Python :", sys.version.split()[0], " (脚本本身只要 3.x；训练跑在 venv 的 3.12 里)")
print("GPU    :", sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null")[1].strip() or "(未检测到 GPU)")
print("内存   :", sh("free -g | awk 'NR==2{print $2\" GB 总 / \"$7\" GB 可用\"}'")[1].strip())
print("磁盘   :", sh("df -h /content | awk 'NR==2{print $4\" 可用\"}'")[1].strip())
print("Node   :", sh("node --version")[1].strip() or "(无)")
if not sh("nvidia-smi -L 2>/dev/null")[1].strip():
    print("\n⚠️ 没有 GPU。菜单「代码执行程序 → 更改运行时类型 → L4 / T4 GPU」后重跑本格。")
'''

CONFIG = r"""
ROOT      = "/content/anima-lora-studio"          # 工作目录（运行时回收即丢失，产物请及时下载）
CODE      = ROOT + "/code"                        # 本仓库（GitHub 拉取）
REPO      = "https://github.com/hsgwktb/anima-lora-studio.git"
TRAINER   = "/content/Anima-Standalone-Trainer"   # 训练引擎，由 setup 脚本克隆
SDSCRIPTS = "/content/sd-scripts"                 # kohya sd-scripts：打标脚本的宿主
MODELS    = "/content/anima-models"               # anima-base-v1.0 + Qwen3 TE + Qwen-Image VAE
PORT      = 8000                                  # WebUI 端口（别用 8080：Colab 自带 node 占着）
print("配置就绪")
"""

FETCH = r"""
if os.path.isdir(CODE + "/.git"):
    sh("git -C %s pull --ff-only" % CODE, stream=True)
else:
    sh("git clone --depth 1 %s %s" % (REPO, CODE), check=True)
print("仓库内容:", sorted(os.listdir(CODE)))
"""

SETUP_DOC = r"""
## 4. 一键安装 + 启动

`colab_setup.sh` 依次做：拉代码 → `uv` 建 **Python 3.12** venv（Colab 自带 3.13 的 `ensurepip` 是坏的）
→ 装 torch 2.7.0+cu128 / transformers / diffusers / bitsandbytes / onnxruntime-gpu 等
→ 起 WebUI(:8000) → 开 cloudflared 隧道。

* **首次约 5–8 分钟**（主要花在下 torch cu128 上），输出会实时刷出来。
* 重复运行**只重启**，不会重装；要强制重装就改成 `bash .../colab_setup.sh --install`。
"""

SETUP = r"""
rc, _ = sh("bash %s/colab_setup.sh" % CODE, stream=True)
print("\nexit code =", rc)
"""

OPEN = r"""
try:
    from google.colab import output
    print("Colab 代理地址（推荐，稳定）:", output.eval_js("google.colab.kernel.proxyPort(8000)"))
except Exception as e:
    print("(没有 Colab 代理:", e, ")")

url = ""
log = ROOT + "/tunnel.log"
if os.path.exists(log):
    txt = open(log, encoding="utf-8", errors="replace").read()
    m = re.findall(r"https://[a-z0-9-]+\.trycloudflare\.com", txt)
    url = m[0] if m else ""

# 注意：不要拿 % 去格式化含 "%{http_code}" 的字符串（会 TypeError: not enough arguments）；拼接最稳。
probe = "curl -s -o /dev/null -w 'WebUI HTTP %{http_code}\n' " + "http://127.0.0.1:" + str(PORT) + "/"
print("公网隧道地址      :", url or "(未就绪，重跑第 4 格)")
print("本机直连          : http://127.0.0.1:" + str(PORT))
print(sh(probe)[1].strip())
print("\n拿到地址后：① 打标 → ② 训练。「隧道地址即密钥」，别外传。")
"""

TAG_DOC = r"""
## 6.（可选）冒烟测试 A：打标

拉 4 张真实照片到 `datasets/smoke/`，跑一遍 Dataset_Maker 配方（首次需联网下载 v3 打标器权重）。
照片不是动漫，标签内容不用在意——这一步只验证**管道**通不通。
"""

TAG = r"""
RUN_TAG_SMOKE = True      # 不想跑就改 False
if RUN_TAG_SMOKE:
    cmd = ("cd %s && ALSTUDIO_VENV_PYTHON=%s/.venv/bin/python "
           "ALSTUDIO_TAGGER_SCRIPT=%s/finetune/tag_images_by_wd14_tagger.py "
           "python3 -u tests/smoke_tagger.py") % (CODE, ROOT, SDSCRIPTS)
    sh(cmd, stream=True)
else:
    print("已跳过")
"""

TRAIN_DOC = r"""
## 7.（可选）冒烟测试 B：训练 6 步

会先下 5.6 GB 权重（`anima-base-v1.0` 4.18 GB + `qwen_3_06b_base` 1.19 GB + `qwen_image_vae` 0.25 GB），
然后以 512px / rank 8 / 6 步跑通一次，产物落在 `jobs/smoke/output/`。
"""

TRAIN = r"""
RUN_TRAIN_SMOKE = False   # 改成 True 再跑（约 3–4 分钟，含下载权重）
if RUN_TRAIN_SMOKE:
    sh("cd %s && ALSTUDIO_VENV_PYTHON=%s/.venv/bin/python python3 -u tests/smoke_train.py" % (CODE, ROOT),
       stream=True)
    print(sh("ls -la %s/jobs/smoke/output/" % ROOT)[1])
else:
    print("已跳过（把 RUN_TRAIN_SMOKE 改成 True 再跑）")
"""

HINTS = r"""
HINTS = '''
# ── 重启 WebUI ──
pkill -f "app.py --port 8000"
ALSTUDIO_ROOT=__ROOT__ ALSTUDIO_CODE=__CODE__ ALSTUDIO_TRAINER_DIR=__TRAINER__ \\
ALSTUDIO_VENV_PYTHON=__ROOT__/.venv/bin/python \\
ALSTUDIO_TAGGER_SCRIPT=__SD__/finetune/tag_images_by_wd14_tagger.py \\
  nohup python3 -u __CODE__/app.py --port 8000 > __ROOT__/app.log 2>&1 &

# ── 看日志 ──
tail -f __ROOT__/app.log                    # WebUI
tail -f __ROOT__/jobs/<项目名>/train.log    # 训练

# ── 数据集 / 产物 ──
ls __ROOT__/datasets/<项目名>               # 图片 + 同名 .txt 标注
ls __ROOT__/jobs/<项目名>/output            # *.safetensors（ComfyUI 格式 LoRA）

# ── 把 LoRA 存到本机 ──
from google.colab import files
files.download("__ROOT__/jobs/<项目名>/output/<名字>.safetensors")

# ── 训练器自带参数说明 ──
__ROOT__/.venv/bin/python __TRAINER__/anima_train_network.py --help | head -60
'''.replace("__ROOT__", ROOT).replace("__CODE__", CODE) \
   .replace("__TRAINER__", TRAINER).replace("__SD__", SDSCRIPTS)
print(HINTS)
"""

NOTES = r"""
## 9. 注意事项（重要）

| 事项 | 说明 |
|---|---|
| **运行时是临时的** | 断开/回收后 `/content` 清空：venv、5.6 GB 权重、数据集全没。重连后从第 3 格重跑即可恢复（安装会重来一遍）。重要 LoRA 请及时 `files.download`。 |
| **一个账号通常只有一个 GPU 会话** | 别的 notebook 占着 GPU 时，这里会卡在「正在分配运行时」。 |
| **隧道地址即密钥** | quick tunnel 地址随机、重启就变；拿到地址的人就能用你的 GPU。 |
| **别用 8080** | Colab 自带 node 进程占着；这里用 8000。 |
| **显存** | L4 24 GB，默认梯度检查点 + latent/TE 缓存就够；不够时打开 `blocks_to_swap`。 |
| **打标器只用 v3** | 2023 年的 `wd-v1-4-swinv2-tagger-v2` 词表太旧；Anima 的动漫数据截止 2025-09，旧词表会整块丢掉新角色。 |
| **标注规范** | 标签小写、空格代替下划线（打标脚本已带 `--remove_underscore`）；正面前缀可用 `masterpiece, best quality, score_7, safe, `。 |
| **训练默认值来自官方建议** | `llm_adapter_lr=0`（不训 LLM adapter，它极易被训坏）；学习率按 rank 缩放（rank 32 → 2e-5）。 |
| **缓存与 shuffle 互斥** | 勾选 cache latents/TE 时不能开 `shuffle_caption`；UI 里两个勾选框已联动。 |
| **许可** | 打标/训练引擎均为第三方（Apache-2.0）；Anima 权重受 CircleStone Labs 非商业许可约束，生成图片可商用。 |
"""


def md(t):
    return {"cell_type": "markdown", "metadata": {},
            "source": t.strip("\n").splitlines(keepends=True)}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": t.strip("\n").splitlines(keepends=True)}


cells = [
    md(TITLE),
    md("## 1. 环境自检（GPU / 内存 / 磁盘 / Node）"), code(ENVCHECK),
    md("## 2. 配置"), code(CONFIG),
    md("## 3. 拉取代码"), code(FETCH),
    md(SETUP_DOC), code(SETUP),
    md("## 5. ✅ 打开 WebUI"), code(OPEN),
    md(TAG_DOC), code(TAG),
    md(TRAIN_DOC), code(TRAIN),
    md("## 8. 常用操作"), code(HINTS),
    md(NOTES),
]

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "toc_visible": True, "name": os.path.basename(OUT)},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

def validate(path):
    """JSON 能读回 + 每个 code cell 能编译。

    上一次的教训：只检查"JSON 合法"是不够的。转义写错时源码里会出现真换行，
    把一行劈成 `print("` + 余下部分——JSON 依然合法，但 Python 直接
    SyntaxError: unterminated string literal。所以必须 compile()，并额外扫一遍
    "以未闭合引号结尾"的行。
    """
    nb = json.load(open(path, encoding="utf-8"))
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        compile(src, "%s[cell%d]" % (path, i), "exec")
        for n, line in enumerate(src.split("\n"), 1):
            stripped = line.rstrip()
            if stripped.endswith(('print("', 'print(\'', '= "', "= '")):
                raise AssertionError("%s cell%d L%d 疑似被劈开: %r"
                                     % (path, i, n, line))
    return len(nb["cells"])


for path in (OUT, OUT_REPO):
    nb["metadata"]["colab"]["name"] = os.path.basename(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    n = validate(path)
    print("wrote %s (%d cells, %d bytes) ✓ 全部 code cell 编译通过"
          % (path, n, os.path.getsize(path)))
