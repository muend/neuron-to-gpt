"""Optional Gradio dashboard: recorded evidence plus trusted-checkpoint inference."""

import argparse
import html
import os
from pathlib import Path

# Keep telemetry disabled by default, before importing the optional UI package.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("GRADIO_TEMP_DIR", str(Path(__file__).resolve().parent / ".panel_cache"))

from PIL import Image
import gradio as gr
import torch

from data import HERE
from panel_runtime import MODELS, PanelService, SamplingSettings, architecture_rows


def metric_cards(service, key):
    result = service.record(key)["result"]
    items = [("FINAL VAL NLL", f"{result['final_val_nll']:.4f}", "nats / character"),
             ("PARAMETERS", f"{result['parameters'] / 1e6:.2f}M", "trained from scratch"),
             ("CONTEXT", str(result['model_config']['block_size']), "previous characters"),
             ("TRAINING", f"{result['training_seconds'] / 60:.1f} min", "A100 · 5,000 updates")]
    return '<div class="metric-grid">' + "".join(
        f'<div class="metric-card"><span>{label}</span><strong>{value}</strong><small>{caption}</small></div>'
        for label, value, caption in items) + "</div>"


def checkpoint_notice(service, key):
    if service.has_checkpoint(key):
        return ("**Checkpoint detected.** Generate will validate the model, corpus hash and best step before inference. "
                "Saved samples remain labeled separately.")
    return ("**Evidence-only mode.** The results ZIP did not contain model weights. The text shown is a recorded sample, "
            "not a live response. Connect the matching `best.pt` files with `--checkpoint-root` to enable Generate.")


def plot_image(service, key):
    # Only public plots are copied into Gradio's cache, not source files or weights.
    with Image.open(service.record(key)["plot"]) as picture:
        return picture.copy()


def detail_text(service, key):
    result = service.record(key)["result"]
    corpus, config = result["corpus"], result["model_config"]
    return (f"### {html.escape(MODELS[key][0])}\n\n"
            f"Best validation NLL: **{result['best_val_nll']:.4f}** at step **{result['sample_checkpoint_step']:,}**. "
            f"Final train NLL: **{result['final_train_nll']:.4f}**.\n\n"
            f"{corpus['characters']:,} characters · {corpus['vocabulary']} unique characters · contiguous 90/10 split.\n\n"
            f"Embedding {config['n_embd']} · {config['n_head']} heads · {config['n_layer']} blocks · dropout {config['dropout']}. "
            f"Recorded elapsed time: {result['elapsed_seconds'] / 60:.2f} min, including validation and checkpoint I/O.\n\n"
            "English and Turkish losses are not directly comparable quality scores. GPU checkpoints were not in the returned "
            "ZIP; export review checked metadata, histories, parameter counts and local corpus hashes.")


def build_panel(service):
    first = "en"
    sample, sample_status = service.saved_sample(first)
    with gr.Blocks(title="Neuron to GPT | Week 08", analytics_enabled=False) as demo:
        gr.HTML('<header class="hero"><div class="eyebrow">NEURON → GPT <span>WEEK 08 / MODEL LAB</span></div>'
                '<h1>Small model.<br><em>Real experiments.</em></h1>'
                '<p>A character-level Transformer, built from scratch.<br>Explore its writing and the evidence behind it.</p>'
                '<div class="hero-tag">256 DIMENSIONS &nbsp; / &nbsp; 8 HEADS &nbsp; / &nbsp; 6 BLOCKS</div></header>')
        cards = gr.HTML(metric_cards(service, first))
        with gr.Tab("Playground"):
            with gr.Row():
                with gr.Column(scale=4, min_width=290):
                    gr.Markdown("### Set the scene\nSelect a model and give it a starting line. This predicts characters; it is not an instruction-following chatbot.")
                    selection = gr.Dropdown(choices=[(name, key) for key, (name, _, _) in MODELS.items()],
                                            value=first, label="Model", interactive=True)
                    prompt = gr.Textbox(value=MODELS[first][2], label="Starting text", lines=3, max_lines=5,
                                        placeholder="Write an opening line...")
                    with gr.Row():
                        characters = gr.Slider(50, 1000, value=250, step=50, label="New characters")
                        temperature = gr.Slider(.1, 2, value=.8, step=.05, label="Temperature")
                    with gr.Row():
                        top_k = gr.Slider(0, 118, value=40, step=1, label="Top-k (0 = all)")
                        seed = gr.Number(value=1337, precision=0, minimum=0, maximum=2**31 - 1, label="Random seed")
                    with gr.Row():
                        generate = gr.Button("Generate continuation", variant="primary", interactive=service.has_checkpoint(first))
                        saved = gr.Button("Recorded sample", variant="secondary")
                    stop = gr.Button("Stop generation", variant="secondary", interactive=service.has_checkpoint(first))
                    notice = gr.Markdown(checkpoint_notice(service, first))
                with gr.Column(scale=6, min_width=350):
                    output = gr.Textbox(value=sample, label="Model writing", lines=20, max_lines=24, interactive=False,
                                        elem_id="writing-output")
                    status = gr.Markdown(sample_status, elem_id="generation-status")
                    gr.Markdown("Lower temperature narrows the choices; higher temperature adds variation. "
                                "Top-k limits sampling to the strongest character candidates. The same seed/settings are repeatable on the same device.")
        with gr.Tab("Training evidence"):
            gr.Markdown("### From training to held-out text\nThe selected model's original exported curve, with no invented measurements.")
            with gr.Row():
                with gr.Column(scale=6, min_width=360):
                    curve = gr.Image(value=plot_image(service, first), label="Train / validation NLL", show_label=False,
                                     alt_text="Selected model's original train and validation loss curves",
                                     interactive=False, buttons=["download", "fullscreen"])
                with gr.Column(scale=4, min_width=260):
                    details = gr.Markdown(detail_text(service, first))
            gr.Markdown("### Controlled architecture progression\nShakespeare · 4,000 updates each · fixed evaluation windows · "
                        "width 64, context 32, dropout off. Parameter counts grow, so this is not a parameter-matched comparison.")
            gr.Dataframe(headers=["Stage", "Parameters", "Final val NLL", "Training seconds"],
                         value=architecture_rows(), datatype=["str", "number", "number", "number"], interactive=False)
            gr.Markdown("**Read the curves honestly.** English validation largely plateaus late in training; its best checkpoint is step 4,500. "
                        "Turkish improves through 5,000. Different vocabularies, genres and OCR quality prevent a direct cross-language ranking. "
                        "The larger GPU model also changes several settings together; it is not a dropout-only ablation.")
        with gr.Tab("Inside the model"):
            gr.Markdown("### One block, repeated six times")
            gr.HTML('<div class="architecture-flow"><div>CHARACTERS<br><small>token + position embeddings</small></div>'
                    '<b>→</b><div>ATTENTION<br><small>pre-LN · 8 causal heads · residual</small></div>'
                    '<b>→</b><div>FEEDFORWARD<br><small>pre-LN · 4× width · residual</small></div>'
                    '<b>→</b><div>NEXT CHARACTER<br><small>final LN · logits · sampling</small></div></div>')
            gr.Markdown("Attention mixes information from earlier tokens; feedforward computes separately at each position. "
                        "Residual paths preserve a route for gradients. LayerNorm normalizes each token's channels, unlike "
                        "Week 6 BatchNorm's batch/time statistics.\n\n"
                        "During generation, only the latest 256 characters enter the model. The full starting text remains visible. "
                        "No fine-tuning, training or corpus upload is performed by this panel.\n\n"
                        "**Limits:** generated writing can contain invented words, grammar errors and incoherent passages. "
                        "Turkish OCR noise remains possible. Outputs are model samples, not quotations from the authors.")
        gr.HTML('<footer>YZ50 / WEEK 08 <span>Measured results. Private source texts. No fabricated live output.</span></footer>')

        def switch(key):
            text, message = service.saved_sample(key)
            return (metric_cards(service, key), MODELS[key][2], text, message,
                    gr.update(interactive=service.has_checkpoint(key)), gr.update(interactive=service.has_checkpoint(key)), checkpoint_notice(service, key),
                    plot_image(service, key), detail_text(service, key))

        def write(key, starting_text, amount, heat, limit, random_seed):
            try:
                settings = SamplingSettings(int(amount), float(heat), int(limit), int(random_seed))
                yield from service.stream(key, starting_text, settings)
            except (ValueError, KeyError, RuntimeError, OSError) as error:
                raise gr.Error(str(error)) from error

        event = generate.click(write, [selection, prompt, characters, temperature, top_k, seed],
                               [output, status], concurrency_limit=1, concurrency_id="generation", api_name=False)
        # Cancel an active stream before switching the language or showing a snapshot.
        selection.change(switch, selection, [cards, prompt, output, status, generate, stop, notice, curve, details],
                         cancels=[event], queue=False, api_name=False)
        saved.click(service.saved_sample, selection, [output, status], cancels=[event], queue=False, api_name=False)
        stop.click(lambda: "Generation stopped. The partial output is preserved.", outputs=status,
                   cancels=[event], queue=False, api_name=False)
    return demo.queue(max_size=8, default_concurrency_limit=1)


def launch_panel(checkpoint_root=None, *, device="auto", share=False, auth=None, port=7860, inbrowser=False,
                 prevent_thread_lock=False):
    if share and (not isinstance(auth, tuple) or len(auth) != 2 or
                  not all(isinstance(part, str) and part.strip() for part in auth) or len(auth[1]) < 8):
        raise ValueError("Public sharing requires authentication with a private password of at least 8 characters")
    torch.set_num_threads(1)
    service = PanelService(checkpoint_root, device=device)
    demo = build_panel(service)
    blocked = [str(HERE / "data"), str(HERE / "outputs"), str(HERE.parent / ".git"),
               str(HERE / ".panel_dependencies"), str(HERE.parent / ".venv"),
               str(HERE.parent / ".codex"), str(HERE.parent / ".aws")]
    if service.checkpoint_root:
        blocked.append(str(service.checkpoint_root))
    theme = gr.themes.Soft(primary_hue="emerald", secondary_hue="slate", neutral_hue="slate",
                           font=["Segoe UI", "Arial", "sans-serif"])
    demo.launch(server_name="127.0.0.1", server_port=port, share=share, auth=auth,
                inbrowser=inbrowser, prevent_thread_lock=prevent_thread_lock,
                allowed_paths=[], blocked_paths=blocked, theme=theme,
                css=(HERE / "panel.css").read_text(encoding="utf-8"), show_error=False)
    return demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, help="Owner-trusted folder containing en/tr_a100_layernorm/best.pt")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--open", action="store_true", help="Open the localhost panel in your browser")
    parser.add_argument("--share", action="store_true", help="Explicitly create an authenticated public Gradio link")
    args = parser.parse_args()
    password = os.environ.get("WEEK08_PANEL_PASSWORD")
    if args.share and not password:
        parser.error("Set WEEK08_PANEL_PASSWORD before --share. Login name: viewer.")
    launch_panel(args.checkpoint_root, device=args.device, port=args.port, share=args.share,
                 auth=("viewer", password) if args.share else None, inbrowser=args.open)


if __name__ == "__main__":
    main()
