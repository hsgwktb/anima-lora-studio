"""Anima LoRA Studio — two-tab WebUI for Anima base v1.0 LoRA datasets and training.

Tab ① 打标 (Tagger)  — Dataset_Maker's WD14-v3 tagging recipe (upload → tag → clean).
Tab ② 训练 (Trainer) — Anima-Standalone-Trainer's engine (anima_train_network.py).

"Send to trainer" hands the tagged dataset straight over to the training tab.

Run:  python3 app.py --port 8000 [--share]
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import gradio as gr

import tagger
import trainer

ROOT = os.environ.get("ALSTUDIO_ROOT", "/content/anima-lora-studio")
DATASETS = os.path.join(ROOT, "datasets")
JOBS = os.path.join(ROOT, "jobs")
for d in (ROOT, DATASETS, JOBS):
    os.makedirs(d, exist_ok=True)


def project_dir(name: str) -> str:
    name = (name or "").strip().replace(" ", "_")
    if not name:
        raise gr.Error("请先填写项目名 / set a project name")
    p = os.path.join(DATASETS, name)
    os.makedirs(p, exist_ok=True)
    return p


def job_dir(name: str) -> str:
    p = os.path.join(JOBS, (name or "job").strip().replace(" ", "_"))
    os.makedirs(p, exist_ok=True)
    return p


# --------------------------------------------------------------------------- #
# Tab ① — tagger
# --------------------------------------------------------------------------- #
def do_upload(project, files):
    if not files:
        return "没有收到文件 / no files uploaded", None, ""
    dst = project_dir(project)
    n = 0
    for f in files:
        src = f if isinstance(f, str) else getattr(f, "name", None)
        if not src or not os.path.isfile(src):
            continue
        shutil.copy2(src, os.path.join(dst, os.path.basename(src)))
        n += 1
    imgs, caps, summary = tagger.preview_captions(dst)
    return "已上传 %d 张图片到 datasets/%s" % (n, os.path.basename(dst)), \
        [(i, c or "(未打标)") for i, c in zip(imgs, caps)], summary


def do_tag(project, tagger_choice, general_t, char_t, blacklist,
           remove_underscore, recursive, batch_size, extras, progress=gr.Progress()):
    dst = project_dir(project)
    if not tagger.list_images(dst):
        return "目录里没有图片 / no images in datasets/%s" % os.path.basename(dst), None, ""
    progress(0.1, desc="打标中 / tagging…")
    log = tagger.tag_images(
        dst, tagger=tagger_choice, general_threshold=general_t,
        character_threshold=char_t, undesired_tags=blacklist,
        remove_underscore=remove_underscore, recursive=recursive,
        batch_size=int(batch_size), extras=extras,
    )
    imgs, caps, summary = tagger.preview_captions(dst)
    return log, [(i, c or "(empty)") for i, c in zip(imgs, caps)], summary


def do_clean(project, remove_tags, add_prefix, dedupe, sort_alpha, anima_prefix):
    dst = project_dir(project)
    msg = tagger.clean_tags(dst, remove_tags=remove_tags, add_prefix=add_prefix,
                            dedupe=dedupe, sort_alpha=sort_alpha,
                            ensure_anima_prefix=anima_prefix)
    imgs, caps, summary = tagger.preview_captions(dst)
    return msg, [(i, c or "(empty)") for i, c in zip(imgs, caps)], summary


def do_refresh(project):
    dst = project_dir(project)
    imgs, caps, summary = tagger.preview_captions(dst)
    return [(i, c or "(empty)") for i, c in zip(imgs, caps)], summary, dst


def send_to_trainer(project):
    dst = project_dir(project)
    n_img = len(tagger.list_images(dst))
    n_cap = len(tagger.list_captions(dst))
    return dst, "✅ 已发送到训练器：%s（%d 图 / %d 标注）" % (dst, n_img, n_cap)


# --------------------------------------------------------------------------- #
# Tab ② — trainer
# --------------------------------------------------------------------------- #
def do_download_models():
    return trainer.download_models()


def model_status():
    paths = trainer.model_paths()
    if trainer.models_ready():
        return "✅ 三个模型文件已就绪 / models ready", paths
    have = [k for k, v in paths.items() if os.path.exists(v)]
    return "⚠️ 缺少模型文件（已有: %s）。点击「下载模型」/ missing models" % (",".join(have) or "无"), paths


def do_train(ds_dir, name, rank, alpha, epochs, save_every, lr, optimizer,
             scheduler, warmup, batch, grad_accum, resolution, repeats,
             seed, blocks_to_swap, max_steps, flow_shift, grad_ckpt, use_cache):
    paths = trainer.model_paths()
    if not trainer.models_ready():
        raise gr.Error("模型文件不完整，请先点「下载模型」/ download models first")
    if not ds_dir or not os.path.isdir(ds_dir):
        raise gr.Error("数据集目录无效 / invalid dataset dir")
    if not tagger.list_images(ds_dir):
        raise gr.Error("数据集里没有图片 / dataset has no images")

    jd = job_dir(name)
    dcfg = os.path.join(jd, "dataset.toml")
    trainer.write_dataset_toml(
        dcfg, ds_dir, resolution=int(resolution), batch_size=int(batch),
        num_repeats=int(repeats), shuffle_caption=True,
    )
    _, argv = trainer.build_train_args(
        image_dir=ds_dir, job_dir=jd, output_name=name, paths=paths,
        dataset_toml=dcfg, rank=int(rank), alpha=int(alpha), epochs=int(epochs),
        save_every_n_epochs=int(save_every),
        learning_rate=(float(lr) if lr and float(lr) > 0 else None),
        optimizer=optimizer, lr_scheduler=scheduler,
        warmup_ratio=float(warmup), batch_size=int(batch),
        gradient_accumulation=int(grad_accum),
        gradient_checkpointing=bool(grad_ckpt), blocks_to_swap=int(blocks_to_swap),
        seed=int(seed), max_train_steps=int(max_steps),
        discrete_flow_shift=float(flow_shift),
    )
    log_path = os.path.join(jd, "train.log")
    msg = trainer.start_training(name, trainer.trainer_dir(), argv, log_path)
    return "%s\njob=%s\nlog=%s" % (msg, jd, log_path), log_path


def do_stop(name):
    return trainer.stop_training(name)


def refresh_train(name, log_path):
    jd = job_dir(name)
    lp = log_path or os.path.join(jd, "train.log")
    st = trainer.status(name)
    outs = trainer.list_outputs(jd)
    out_md = "**LoRA 输出 / outputs (%d):**\n" % len(outs) + \
             "\n".join("- `%s` (%.1f MB)" % (o, os.path.getsize(o) / 1e6) for o in outs)
    return st, trainer.progress(lp), trainer.tail_log(lp), out_md, outs


CSS = """
.tabnav button {font-size: 15px !important;}
"""


def build_ui():
    with gr.Blocks(title="Anima LoRA Studio") as demo:
        gr.Markdown(
            "# 🎨 Anima LoRA Studio\n"
            "**① 打标** 用 Dataset_Maker 的 WD14-v3 打标流程（`wd-eva02-large-tagger-v3` + `wd-vit-large-tagger-v3`）"
            " · **② 训练** 用 Anima-Standalone-Trainer 的引擎（`anima_train_network.py`）\n\n"
            "打标完成后点「发送到训练器」，数据集会直接填进训练选项卡。"
        )
        ds_state = gr.State("")
        log_state = gr.State("")

        with gr.Tabs():
            # ---------------- Tab 1 : tagger ----------------
            with gr.Tab("① 打标 / Tagger"):
                with gr.Row():
                    with gr.Column(scale=1):
                        project = gr.Textbox(label="项目名 / project", value="my_lora",
                                             info="数据集落在 datasets/<项目名>/")
                        files = gr.File(label="上传图片 / upload images", file_count="multiple",
                                        file_types=["image"], type="filepath")
                        up_btn = gr.Button("上传并预览 / upload", variant="secondary")

                        gr.Markdown("### 打标设置 / tagging")
                        tagger_choice = gr.Dropdown(tagger.TAGGER_CHOICES,
                                                    value=tagger.TAGGER_CHOICES[0],
                                                    label="打标模型 / tagger")
                        general_t = gr.Slider(0.0, 1.0, value=0.25, step=0.01,
                                              label="一般阈值 / general threshold",
                                              info="概念 0.25，风格 0.5")
                        char_t = gr.Slider(0.0, 1.2, value=1.1, step=0.01,
                                           label="角色阈值 / character threshold",
                                           info=">1.0 = 不输出角色名（训画风时用）")
                        blacklist = gr.Textbox(label="黑名单标签 / undesired tags",
                                               value=tagger.DEFAULT_BLACKLIST, lines=3)
                        with gr.Row():
                            remove_underscore = gr.Checkbox(value=True, label="下划线换空格 (Anima 要求)")
                            recursive = gr.Checkbox(value=True, label="递归子目录")
                        batch_size = gr.Number(value=8, label="batch size", precision=0)
                        extras = gr.Textbox(label="附加参数 / extra CLI args",
                                            placeholder="--use_quality_tags --character_tags_first", lines=1)
                        tag_btn = gr.Button("🏷️ 开始打标 / tag images", variant="primary")

                        gr.Markdown("### 清洗标签 / clean (Dataset_Maker extras)")
                        remove_tags = gr.Textbox(label="删除这些标签 / remove tags", lines=2)
                        add_prefix = gr.Textbox(label="加前缀标签 / prefix",
                                                placeholder="例如你的人设触发词",
                                                info="会自动补 Anima 推荐的 masterpiece, best quality")
                        with gr.Row():
                            dedupe = gr.Checkbox(value=True, label="去重")
                            sort_alpha = gr.Checkbox(value=False, label="按字母排序")
                            anima_prefix = gr.Checkbox(value=True, label="加 Anima 前缀")
                        clean_btn = gr.Button("🧹 清洗 / clean captions")
                        send_btn = gr.Button("➡️ 发送到训练器 / send to trainer", variant="primary")
                        send_msg = gr.Markdown()

                    with gr.Column(scale=2):
                        tag_log = gr.Textbox(label="打标日志 / tagger log", lines=16)
                        stats_md = gr.Markdown()
                        gallery = gr.Gallery(label="图片 + 标注 / images & captions",
                                             columns=4, height=560, allow_preview=True)

            # ---------------- Tab 2 : trainer ----------------
            with gr.Tab("② 训练 / Trainer"):
                with gr.Row():
                    with gr.Column(scale=1):
                        m_status, paths_state = model_status()
                        gr.Markdown("### " + m_status)
                        dl_btn = gr.Button("⬇️ 下载模型 / download anima-base-v1.0 + TE + VAE")
                        dl_out = gr.Textbox(label="下载日志 / download log", lines=6)

                        gr.Markdown("### 数据集 / dataset")
                        ds_dir = gr.Textbox(label="数据集目录 / image dir",
                                            placeholder="先在上一个选项卡点「发送到训练器」")
                        job_name = gr.Textbox(label="LoRA 输出名 / output name", value="my_lora")

                        gr.Markdown("### 参数 / hyperparameters")
                        with gr.Row():
                            rank = gr.Slider(4, 128, value=16, step=4, label="network_dim (rank)")
                            alpha = gr.Slider(1, 128, value=16, step=1, label="network_alpha")
                        with gr.Row():
                            epochs = gr.Number(value=10, label="epochs", precision=0)
                            save_every = gr.Number(value=1, label="save_every_n_epochs", precision=0)
                            max_steps = gr.Number(value=0, label="max_train_steps (0=用 epochs)", precision=0)
                        with gr.Row():
                            lr = gr.Number(value=0, label="learning_rate (0=按 rank 自动)")
                            warmup = gr.Number(value=0.05, label="warmup 比例")
                            seed = gr.Number(value=42, label="seed", precision=0)
                        with gr.Row():
                            batch = gr.Number(value=1, label="batch_size", precision=0)
                            grad_accum = gr.Number(value=1, label="grad_accum", precision=0)
                            resolution = gr.Number(value=1024, label="resolution", precision=0)
                            repeats = gr.Number(value=10, label="num_repeats", precision=0)
                        with gr.Row():
                            optimizer = gr.Dropdown(["AdamW8bit", "AdamW", "Prodigy", "Lion",
                                                     "Adafactor", "Muon"], value="AdamW8bit",
                                                    label="optimizer")
                            scheduler = gr.Dropdown(["cosine", "constant", "constant_with_warmup",
                                                     "cosine_with_restarts", "linear"],
                                                    value="cosine", label="lr_scheduler")
                            flow_shift = gr.Number(value=3.0, label="discrete_flow_shift")
                        with gr.Row():
                            grad_ckpt = gr.Checkbox(value=True, label="gradient_checkpointing (省显存)")
                            blocks_to_swap = gr.Number(value=0, label="blocks_to_swap (0=关)", precision=0)
                            use_cache = gr.Checkbox(value=True, label="cache latents/TE (推荐)")

                        with gr.Row():
                            train_btn = gr.Button("🚀 开始训练 / start training", variant="primary")
                            stop_btn = gr.Button("⏹️ 停止 / stop")
                        train_msg = gr.Textbox(label="状态 / status", lines=2)

                    with gr.Column(scale=2):
                        t_status = gr.Textbox(label="任务状态 / job state", lines=1)
                        t_progress = gr.Textbox(label="进度 / progress", lines=1)
                        t_log = gr.Textbox(label="训练日志 / training log", lines=26)
                        t_out = gr.Markdown()
                        t_files = gr.File(label="下载 LoRA / download LoRA", file_count="multiple")

            # ---------------- wiring ----------------
            up_btn.click(do_upload, [project, files], [send_msg, gallery, stats_md])
            tag_btn.click(do_tag,
                          [project, tagger_choice, general_t, char_t, blacklist,
                           remove_underscore, recursive, batch_size, extras],
                          [tag_log, gallery, stats_md])
            clean_btn.click(do_clean,
                            [project, remove_tags, add_prefix, dedupe, sort_alpha, anima_prefix],
                            [send_msg, gallery, stats_md])
            send_btn.click(send_to_trainer, [project], [ds_dir, send_msg])
            dl_btn.click(do_download_models, None, [dl_out])
            train_btn.click(do_train,
                            [ds_dir, job_name, rank, alpha, epochs, save_every, lr, optimizer,
                             scheduler, warmup, batch, grad_accum, resolution, repeats, seed,
                             blocks_to_swap, max_steps, flow_shift, grad_ckpt, use_cache],
                            [train_msg, log_state])
            stop_btn.click(do_stop, [job_name], [train_msg])
            timer = gr.Timer(4)
            timer.tick(refresh_train, [job_name, log_state],
                       [t_status, t_progress, t_log, t_out, t_files])
        return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    # Gradio >= 6 moved theme/css from Blocks() onto launch().
    build_ui().queue().launch(server_name=args.host, server_port=args.port,
                              share=args.share, allowed_paths=[ROOT],
                              theme=gr.themes.Soft(), css=CSS)


if __name__ == "__main__":
    main()
