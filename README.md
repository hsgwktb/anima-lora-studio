# Anima LoRA Studio

一个 Colab 上的 Anima base v1.0 LoRA 训练工作台：**两个选项卡的 WebUI**，上传图片 → 打标 → 一键送到训练器。

* **① 打标 / Tagger** — 复刻 [Dataset_Maker](https://github.com/hollowstrawberry/kohya-colab) 的打标配方：
  kohya 的 `tag_images_by_wd14_tagger.py` + SmilingWolf **v3** 打标器
  （`wd-eva02-large-tagger-v3` + `wd-vit-large-tagger-v3`，可切换），支持阈值、黑名单、
  去下划线、递归、标签清洗（去重 / 黑名单 / 加前缀 / Anima 推荐前缀）。
* **② 训练 / Trainer** — 驱动 [Anima-Standalone-Trainer](https://github.com/gazingstars123/Anima-Standalone-Trainer)
  的训练引擎 `anima_train_network.py`（sd-scripts 分支），只保留训练引擎，用 Python 直接调用，
  不装它的 Node/Electron 界面。

## 为什么是 v3 打标器

Dataset_Maker 在 2025-03-31 把打标器从 `wd-v1-4-swinv2-tagger-v2`（2023-01）升级到了 v3。
Anima 的动漫数据截止到 **2025 年 9 月**，旧版打标器词表里根本没有近两年半的新角色/新番，
会整块丢失。所以这里只用 v3。

## 快速开始（Colab）

```bash
git clone --depth 1 https://github.com/hsgwktb/anima-lora-studio.git /content/anima-lora-studio/code
bash /content/anima-lora-studio/code/colab_setup.sh
```

脚本会：克隆/更新 Anima-Standalone-Trainer → 建 venv 装 `requirements.txt`
（torch 2.7.0+cu128，约 5–8 分钟）→ 装 onnxruntime → 起 WebUI(:8000) → 起 cloudflared 隧道并打印公网地址。
重复执行只会重启，不会重装；强制重装加 `--install`。

## 用法

1. **① 打标**：填项目名 → 上传图片 → 选打标器/阈值 → 「开始打标」→ 用「清洗」加触发词/去杂标签 →
   「**发送到训练器**」（数据集路径会自动填进第二个选项卡）。
2. **② 训练**：点「下载模型」（`anima-base-v1.0.safetensors` + `qwen_3_06b_base` + `qwen_image_vae`，约 5.6 GB）→
   设 rank/LR/epochs → 「开始训练」。日志与 LoRA 产物实时出现在右侧。

## 默认值来自官方建议

| 项 | 值 | 出处 |
|---|---|---|
| `llm_adapter_lr` | `0`（不训练 LLM adapter） | Anima 模型卡：adapter 极易被训坏 |
| 学习率 | rank 16 → `1e-5`，rank 32 → `2e-5` | 模型卡：rank 32 从 2e-5 起，按 rank 线性缩放 |
| 打标 | `--remove_underscore` | 模型卡：标签用小写、空格代替下划线 |
| 前缀 | `masterpiece, best quality` | 模型卡推荐正面前缀 |

标签用 `dataset.toml` 传给训练器；LoRA 由训练器按 **ComfyUI 格式**保存。

## 目录

```
app.py                 Gradio 双选项卡 WebUI
tagger.py              Dataset_Maker 打标流程 + 标签清洗
trainer.py             Anima-Standalone-Trainer 驱动（dataset.toml / CLI / 进程管理）
colab_setup.sh         一键安装 + 启动 + 隧道
tests/smoke_tagger.py  打标冒烟测试
```

打标脚本本身来自运行期克隆的 `kohya-ss/sd-scripts`（其 `finetune/tag_images_by_wd14_tagger.py`
需要同仓库的 `library/` 包），本仓库不复制它。

Colab 上的运行期布局：

```
/content/anima-lora-studio/
├── code/         本仓库
├── .venv/        训练器依赖（torch 2.7.0+cu128）
├── datasets/<项目名>/    图片 + .txt 标注
├── jobs/<项目名>/        dataset.toml、logs/、output/*.safetensors
└── cloudflared, tunnel.log, app.log
```

## 注意

* 训练引擎需要 NVIDIA GPU。L4/A100 上可训；24 GB 显存建议开 `gradient_checkpointing`，
  必要时用 `blocks_to_swap`。
* 隧道地址即密钥：拿到地址的人就能用你的 GPU，别外传。
* Colab 运行时是临时的，重连后需要重新跑 `colab_setup.sh`（模型和数据集会丢）。
* 这是第三方封装，不是 CircleStone Labs 官方工具。

## 许可

本仓库代码 MIT。打标脚本与训练引擎都在运行期从上游克隆（kohya-ss/sd-scripts、
Anima-Standalone-Trainer，均 Apache-2.0），本仓库不再分发它们。
Anima 模型权重受 CircleStone Labs 非商业许可约束（生成图片可商用）。详见 `NOTICE`。
