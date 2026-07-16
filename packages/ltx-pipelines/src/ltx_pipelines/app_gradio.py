"""Gradio UI for interactive streaming LTX-2 TI2V — pre-encoded prompt playlist + streaming A/V.

Launches a long-lived :class:`InteractiveStreamingSession` (DiT + Gemma + VAEs built
once, resident) and a Gradio Blocks app. Enter **several prompts** (one per line);
on **Generate** they are all pre-encoded once by Gemma and cached in memory, then the
clip streams starting from the first prompt. The **Next Prompt** button advances to
the next pre-encoded prompt — applied at the next AR chunk boundary as a pure
cross-attention swap (text is not part of the cached self-attention history, so the
switch is clean and does not reset the video). Audio plays live. Until Next is
pressed the model keeps running on the current prompt; the last prompt clamps (no loop).

Run (after `uv sync`, models downloaded as in the upstream README):

    uv run python -m ltx_pipelines.app_gradio \
        --checkpoint-path models/ltx-2.3/ltx-2.3-22b-dev.safetensors \
        --gemma-root models/gemma-3-12b

Then open the printed URL, upload a reference image (the sink), enter one prompt per
line, and hit **Generate**; press **Next Prompt** while it streams to switch prompts.
On low-VRAM GPUs add `--quantization fp8-cast --offload cpu` (note: Gemma stays resident — 12B).

This is a conceptual/interactive demo (the base model is bidirectionally trained,
so the causal streaming regime is a train/test mismatch — see the README gaps).
Runtime-unverified in this dev environment (no torch/GPU); validate on a GPU box.
"""

from __future__ import annotations

import logging

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
        prompts_text: str,
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
        """Generator: drive the session, yielding (video, audio, status, prompt_status) per chunk."""
        if not image_path:
            yield None, None, "⚠️ Please provide a reference image (the sink).", session.prompt_status()
            return
        prompts = [ln.strip() for ln in (prompts_text or "").splitlines() if ln.strip()]
        if not prompts:
            yield None, None, "⚠️ Enter at least one prompt (one per line).", session.prompt_status()
            return
        yield (
            None,
            None,
            f"⏳ Pre-encoding {len(prompts)} prompt(s)…",
            session.prompt_status(),
        )
        try:
            for update in session.run(
                prompts=prompts,
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
                yield update.video_path, update.audio, update.status, session.prompt_status()
        except Exception as exc:  # noqa: BLE001 — surface to the UI status line
            logger.exception("generation failed")
            yield None, None, f"✗ error: {exc}", session.prompt_status()

    with gr.Blocks(title="LTX-2 Interactive Streaming") as demo:
        gr.Markdown(
            "## LTX-2 · Interactive Streaming Video (prompt playlist)\n"
            "Upload a reference image (the sink), enter **one prompt per line**, and **Generate**. "
            "All prompts are pre-encoded once on start; the clip streams from the first. Press "
            "**Next Prompt** while streaming to switch — applied at the next chunk as a clean "
            "cross-attention swap (no video reset). Until then it keeps running on the current "
            "prompt; the last prompt clamps (no loop)."
        )
        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(type="filepath", label="Reference image (sink)")
                prompts_in = gr.Textbox(
                    label="Prompts (one per line — pre-encoded, advance with 'Next')",
                    value="A person talking calmly to the camera.\nA person laughing.\nA person looking surprised.",
                    lines=8,
                )
                seed = gr.Slider(0, 2**31 - 1, value=0, step=1, label="Seed")
                enhance = gr.Checkbox(value=False, label="Enhance prompts (Gemma)")
                causal = gr.Checkbox(value=True, label="Causal cross-attention (AV)")
                with gr.Row():
                    start_btn = gr.Button("▶ Generate", variant="primary")
                    next_btn = gr.Button("⏭ Next Prompt", variant="secondary")
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
                prompt_status_out = gr.Markdown("Prompts: 0 loaded.")
                video_out = gr.Video(label="Growing video", autoplay=True)
                audio_out = gr.Audio(label="Live audio", streaming=True, autoplay=True)
                status_out = gr.Markdown("Idle.")

        # Next Prompt: queue a one-step advance (clamped at the last). Thread-safe,
        # runs concurrently with the streaming generator; refreshes the status line.
        next_btn.click(
            fn=lambda: session.advance_prompt(),
            inputs=None,
            outputs=[prompt_status_out],
        )

        gen_event = start_btn.click(
            generate,
            inputs=[
                image_in, prompts_in, height, width, num_frames, frame_rate,
                steps, window_chunks, chunk_frames, seed, causal, enhance,
            ],
            outputs=[video_out, audio_out, status_out, prompt_status_out],
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
    parser.add_argument("--host", default="127.0.0.1", help="Gradio server host.")
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    args = parser.parse_args()

    global SESSION  # noqa: PLW0603 — single shared session for a single-GPU demo
    SESSION = InteractiveStreamingSession(
        checkpoint_path=args.checkpoint_path,
        gemma_root=args.gemma_root,
        loras=tuple(args.lora) if args.lora else (),
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
