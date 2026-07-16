"""Gradio UI for interactive streaming LTX-2 TI2V — live prompt + streaming A/V.

Launches a long-lived :class:`InteractiveStreamingSession` (DiT + Gemma + VAEs built
once, resident) and a Gradio Blocks app. While a clip streams, editing the **live
prompt** textbox rewrites the cross-attention conditioning for subsequent AR chunks
(text is not part of the cached self-attention history, so the change is clean) —
the growing video visibly changes content on the next chunk, and audio plays live.

Run (after `uv sync`, models downloaded as in the upstream README):

    uv run python -m ltx_pipelines.app_gradio \
        --checkpoint-path models/ltx-2.3/ltx-2.3-22b-dev.safetensors \
        --gemma-root models/gemma-3-12b

Then open the printed URL, upload a reference image (the sink), set the initial
prompt, hit **Generate**, and edit the live prompt mid-stream. On low-VRAM GPUs add
`--quantization fp8-cast --offload cpu` (note: Gemma stays resident — 12B).

This is a conceptual/interactive demo (the base model is bidirectionally trained,
so the causal streaming regime is a train/test mismatch — see the README gaps).
Runtime-unverified in this dev environment (no torch/GPU); validate on a GPU box.
"""

from __future__ import annotations

import logging

import torch

from ltx_core.model.video_vae.tiling import TilingConfig
from ltx_pipelines.interactive_session import InteractiveStreamingSession
from ltx_pipelines.utils.args import new_video_gen_arg_parser, resolve_cli_params

logger = logging.getLogger(__name__)

#: The single shared session (one GPU → one active generation). Built in :func:`main`.
SESSION: InteractiveStreamingSession | None = None


def _build_ui(session: InteractiveStreamingSession) -> "Blocks":  # type: ignore[name-defined]
    import gradio as gr

    def generate(  # noqa: PLR0913
        image_path: str,
        initial_prompt: str,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        steps: int,
        window_chunks: int,
        chunk_frames: int,
        seed: int,
        causal_cross_attn: bool,
        enhance_prompt: bool,
    ):
        """Generator: drive the session, yielding (video, audio, status) per chunk."""
        if not image_path:
            yield None, None, "⚠️ Please provide a reference image (the sink)."
            return
        yield None, None, "⏳ Starting generation…"
        try:
            for update in session.run(
                initial_prompt=initial_prompt,
                seed=int(seed),
                height=int(height),
                width=int(width),
                num_frames=int(num_frames),
                frame_rate=float(frame_rate),
                num_inference_steps=int(steps),
                image_path=image_path,
                window_chunks=int(window_chunks),
                chunk_frames=int(chunk_frames),
                enhance_prompt=bool(enhance_prompt),
                causal_cross_attn=bool(causal_cross_attn),
                tiling_config=TilingConfig.default(),
            ):
                yield update.video_path, update.audio, update.status
        except Exception as exc:  # noqa: BLE001 — surface to the UI status line
            logger.exception("generation failed")
            yield None, None, f"✗ error: {exc}"

    with gr.Blocks(title="LTX-2 Interactive Streaming") as demo:
        gr.Markdown(
            "## LTX-2 · Interactive Streaming Video (live prompt)\n"
            "Upload a reference image (the sink), set the initial prompt, and **Generate**. "
            "While it streams, edit the **live prompt** — the next chunk picks it up "
            "(text is cross-attention, so the change does not reset the video)."
        )
        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(type="filepath", label="Reference image (sink)")
                initial_prompt = gr.Textbox(
                    label="Initial prompt", value="A person talking calmly to the camera."
                )
                live_prompt = gr.Textbox(
                    label="Live prompt (auto-applies on next chunk)",
                    placeholder="Type a new prompt while generating…",
                )
                seed = gr.Slider(0, 2**31 - 1, value=0, step=1, label="Seed")
                enhance = gr.Checkbox(value=False, label="Enhance initial prompt (Gemma)")
                causal = gr.Checkbox(value=True, label="Causal cross-attention (AV)")
                with gr.Row():
                    start_btn = gr.Button("▶ Generate", variant="primary")
                    stop_btn = gr.Button("⏹ Stop", variant="stop")
            with gr.Column(scale=1):
                with gr.Accordion("Generation settings", open=False):
                    height = gr.Slider(256, 1024, value=512, step=32, label="Height")
                    width = gr.Slider(256, 1024, value=768, step=32, label="Width")
                    num_frames = gr.Slider(9, 257, value=33, step=8, label="Num frames")
                    frame_rate = gr.Slider(8, 60, value=30, step=1, label="Frame rate (fps)")
                    steps = gr.Slider(1, 80, value=30, step=1, label="Inference steps")
                    window_chunks = gr.Slider(1, 16, value=4, step=1, label="Window chunks")
                    chunk_frames = gr.Slider(1, 8, value=1, step=1, label="Chunk frames")
                video_out = gr.Video(label="Growing video", autoplay=True)
                audio_out = gr.Audio(label="Live audio", streaming=True, autoplay=True)
                status_out = gr.Markdown("Idle.")

        # Live prompt: each edit queues the latest value for the next chunk boundary.
        live_prompt.change(
            fn=lambda text: session.submit_prompt(text),
            inputs=[live_prompt],
            outputs=None,
        )

        gen_event = start_btn.click(
            generate,
            inputs=[
                image_in, initial_prompt, height, width, num_frames, frame_rate,
                steps, window_chunks, chunk_frames, seed, causal, enhance,
            ],
            outputs=[video_out, audio_out, status_out],
            concurrency_limit=1,
        )
        stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[gen_event])

    return demo


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    params = resolve_cli_params()
    parser = new_video_gen_arg_parser(params=params)
    # The interactive app takes the prompt from the UI and streams output (no file),
    # so relax the base parser's required --prompt / --output-path.
    for action in parser._actions:
        if action.dest in ("prompt", "output_path"):
            action.required = False
            action.default = ""
    parser.add_argument("--host", default="0.0.0.0", help="Gradio server host (0.0.0.0 for LAN access).")
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="GPU device for the DiT + VAEs (e.g. 'cuda:0'). Defaults to cuda:0.",
    )
    parser.add_argument(
        "--text-encoder-device",
        type=str,
        default=None,
        help=(
            "GPU device for the Gemma text encoder (e.g. 'cuda:1'). Defaults to --device. "
            "Set to a different GPU to split the 12B text encoder from the 22B DiT."
        ),
    )
    args = parser.parse_args()

    dit_device = torch.device(args.device) if args.device else None
    text_device = torch.device(args.text_encoder_device) if args.text_encoder_device else None

    global SESSION  # noqa: PLW0603 — single shared session for a single-GPU demo
    SESSION = InteractiveStreamingSession(
        checkpoint_path=args.checkpoint_path,
        gemma_root=args.gemma_root,
        loras=tuple(args.lora) if args.lora else (),
        device=dit_device,
        text_encoder_device=text_device,
        quantization=args.quantization,
        compilation_config=args.compile,
        offload_mode=args.offload_mode,
    )
    logger.info("Loading models (this may take a while)…")
    SESSION.start()

    demo = _build_ui(SESSION)
    try:
        demo.launch(server_name=args.host, server_port=args.port, share=args.share)
    finally:
        SESSION.stop()


if __name__ == "__main__":
    main()
