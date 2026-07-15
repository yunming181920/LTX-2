"""Checkpoint-free CPU validation of the interactive streaming driver.

Mirrors ``test_streaming_tiny_model.py`` (same tiny random 2-layer model, no
checkpoint) to exercise :func:`iter_streaming_chunks_joint` — the generator that
adds the live-prompt ``context_resolver`` hook and per-chunk yielding on top of M1.

Three phases:

  * **Phase 1 — constant-context parity.** A resolver returning the same context
    every chunk must reproduce :func:`streaming_generate_joint` (M1) exactly (same
    chunk loop body). The final yielded video prefix equals M1's full video latent;
    the audio prefix matches M1's filled audio region.
  * **Phase 2 — per-chunk swap smoke.** A resolver returning a *different* context
    on chunk ≥ 1 runs to completion with finite output, and chunk 0 (no history) is
    unchanged — its prefix matches M1(ctx0) through chunk 0. This is the live-prompt-
    injection guarantee: a context change affects only subsequent chunks.
  * **Phase 3 — resolver call count.** The resolver is invoked exactly once per AR
    chunk (one cross-attention context per chunk).

Run:

    uv run python packages/ltx-pipelines/tests/test_streaming_interactive.py
"""

import torch

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ltx_core.model.transformer.model import LTXModel, X0Model
from ltx_core.tools import AudioLatentTools, VideoLatentTools
from ltx_core.types import AudioLatentShape, VideoLatentShape
from ltx_pipelines.utils.streaming import streaming_generate_joint
from ltx_pipelines.utils.streaming_interactive import StreamChunk, iter_streaming_chunks_joint

FPS = 25.0
H, W = 2, 3
CH = 8
ACH, MEL = 4, 16  # audio token dim = 64


def build_tiny() -> X0Model:
    torch.manual_seed(7)
    model = LTXModel(
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=CH,
        out_channels=CH,
        num_layers=2,
        cross_attention_dim=16,  # == video inner_dim (context is viewed to inner_dim)
        audio_num_attention_heads=2,
        audio_attention_head_dim=8,
        audio_in_channels=ACH * MEL,
        audio_out_channels=ACH * MEL,
        audio_cross_attention_dim=16,  # == audio inner_dim
    )
    # Reinitialize EVERYTHING to small gaussians so the random model is finite.
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, 0.02)
    return X0Model(model).float().eval()


def _tools(num_latent_frames: int) -> tuple[VideoLatentTools, AudioLatentTools]:
    v_shape = VideoLatentShape(1, CH, num_latent_frames, H, W)
    total_audio = int(round((num_latent_frames - 1) * 8 / FPS * 25)) + 1
    a_shape = AudioLatentShape(1, ACH, total_audio, MEL)
    video_tools = VideoLatentTools(VideoLatentPatchifier(patch_size=1), v_shape, FPS)
    audio_tools = AudioLatentTools(AudioPatchifier(patch_size=1), a_shape)
    return video_tools, audio_tools


def _base_kwargs(x0: X0Model, num_latent_frames: int, seed: int = 0) -> dict:
    """Shared kwargs for both drivers (sigmas, shapes, sink, noiser, ...)."""
    device = torch.device("cpu")
    video_tools, audio_tools = _tools(num_latent_frames)
    torch.manual_seed(11)  # so the sink tensor below is identical across runs
    return dict(
        sigmas=torch.linspace(1.0, 0.0, 5),
        num_generated_latent_frames=num_latent_frames - 1,
        chunk_frames=1,
        window_chunks=2,
        video_tools_full=video_tools,
        audio_tools_full=audio_tools,
        sink_latent_unpatchified=torch.randn(1, CH, 1, H, W),
        stepper=EulerDiffusionStep(),
        transformer=x0,
        noiser=GaussianNoiser(torch.Generator(device=device).manual_seed(seed)),
        dtype=torch.float32,
        device=device,
        causal_cross_attn=True,
        cross_attn_lookahead_sec=0.0,
    )


def run_m1(x0: X0Model, num_latent_frames: int, v_context: torch.Tensor, a_context: torch.Tensor) -> tuple:
    kw = _base_kwargs(x0, num_latent_frames)
    kw["v_context"] = v_context
    kw["a_context"] = a_context
    return streaming_generate_joint(**kw)


def run_interactive(x0: X0Model, num_latent_frames: int, resolver) -> list[StreamChunk]:
    kw = _base_kwargs(x0, num_latent_frames)
    kw["context_resolver"] = resolver
    return list(iter_streaming_chunks_joint(**kw))


def main() -> None:
    x0 = build_tiny()
    with torch.inference_mode():
        # Phase 1: constant-context parity vs M1 (2 generated chunks).
        ctx_v = torch.randn(1, 4, 16)
        ctx_a = torch.randn(1, 4, 16)
        m1_v, m1_a = run_m1(x0, 3, ctx_v, ctx_a)
        chunks = run_interactive(x0, 3, lambda i, n: (ctx_v, ctx_a))
        last = chunks[-1]
        dv = (last.video_latent_prefix - m1_v).abs().max().item()
        assert last.video_latent_prefix.shape == m1_v.shape, "video prefix shape must match M1"
        assert dv == 0.0, f"constant-context video must equal M1 bitwise, max|diff|={dv}"
        audio_len = min(last.audio_latent_prefix.shape[2], m1_a.shape[2])
        da = (last.audio_latent_prefix[:, :, :audio_len] - m1_a[:, :, :audio_len]).abs().max().item()
        assert da == 0.0, f"constant-context audio must equal M1 bitwise, max|diff|={da}"
        print(f"[phase1] constant-context parity: video max|diff|={dv:.3e} audio max|diff|={da:.3e}")

        # Phase 2: per-chunk context swap (4 generated chunks). Chunk 0 must match
        # M1(ctx0); later chunks use ctx1 and only need to be finite.
        ctx0_v, ctx0_a = torch.randn(1, 4, 16), torch.randn(1, 4, 16)
        ctx1_v, ctx1_a = torch.randn(1, 4, 16), torch.randn(1, 4, 16)

        def swap_resolver(i: int, n: int) -> tuple[torch.Tensor, torch.Tensor]:
            return (ctx0_v, ctx0_a) if i == 0 else (ctx1_v, ctx1_a)

        swap_chunks = run_interactive(x0, 5, swap_resolver)
        last_swap = swap_chunks[-1]
        assert torch.isfinite(last_swap.video_latent_prefix).all(), "swap: non-finite video"
        assert torch.isfinite(last_swap.audio_latent_prefix).all(), "swap: non-finite audio"
        # Chunk 0 (no history) is independent of later context → must equal M1(ctx0).
        m1_v0, _ = run_m1(x0, 5, ctx0_v, ctx0_a)
        first = swap_chunks[0]
        d0 = (first.video_latent_prefix - m1_v0[:, :, : first.video_latent_prefix.shape[2]]).abs().max().item()
        assert d0 == 0.0, f"chunk-0 must match M1(ctx0), max|diff|={d0}"
        print(
            f"[phase2] swap smoke: {len(swap_chunks)} chunks finite; "
            f"chunk-0 matches M1(ctx0) (max|diff|={d0:.3e})"
        )

        # Phase 3: resolver is called exactly once per chunk.
        calls = {"n": 0}

        def counting_resolver(i: int, n: int) -> tuple[torch.Tensor, torch.Tensor]:
            calls["n"] += 1
            return (ctx_v, ctx_a)

        count_chunks = run_interactive(x0, 5, counting_resolver)
        expected = 5 - 1  # num_generated_latent_frames
        assert calls["n"] == len(count_chunks) == expected, (
            f"resolver must be called once per chunk: got {calls['n']}, expected {expected}"
        )
        print(f"[phase3] resolver called {calls['n']} times for {expected} chunks")

    print("\nINTERACTIVE STREAMING VALIDATION PASSED")


if __name__ == "__main__":
    main()
