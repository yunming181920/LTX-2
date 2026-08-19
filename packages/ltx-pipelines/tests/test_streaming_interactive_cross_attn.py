"""Checkpoint-free CPU validation of ``cache_cross_attn`` on the interactive driver.

``cache_cross_attn`` has to be threaded through two independent places: the
``CausalStreamingModel`` constructor (which allocates the a2v/v2a caches) *and*
``prepare_chunk`` (which supplies their window RoPE / history lengths). Wiring
only the second is silently inert — the extra ``prepare_chunk`` arguments are
accepted and dropped, every run stays byte-identical, and the ablation quietly
compares a flag against itself. A parity-only test cannot see that, so assert
the semantics instead:

  * **Phase 1 — single chunk is a no-op.** With one AR chunk there is no history,
    so the cached cross-attn prefix *is* the current chunk and the result must be
    bitwise identical to ``cache_cross_attn=False``.
  * **Phase 2 — multiple chunks must differ.** Once history exists the current
    video chunk cross-attends to ``[past audio | current audio]``, so the output
    has to move (and stay finite).
  * **Phase 3 — cross-driver equivalence.** The interactive driver must reproduce
    the offline driver (:func:`streaming_generate_joint_cached`) exactly for
    *both* settings of the flag. This is the phase that fails when only half the
    wiring is present.

Run:

    uv run python packages/ltx-pipelines/tests/test_streaming_interactive_cross_attn.py
"""

import torch

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ltx_core.model.transformer.model import LTXModel, X0Model
from ltx_core.tools import AudioLatentTools, VideoLatentTools
from ltx_core.types import AudioLatentShape, VideoLatentShape
from ltx_pipelines.utils.streaming import streaming_generate_joint_cached
from ltx_pipelines.utils.streaming_interactive import iter_streaming_chunks_joint_cached

FPS = 25.0
H, W = 2, 3
CH = 8
ACH, MEL = 4, 16

STRATEGIES = ("twin", "clean", "noisy_steps")


def build_tiny() -> X0Model:
    torch.manual_seed(7)
    model = LTXModel(
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=CH,
        out_channels=CH,
        num_layers=2,
        cross_attention_dim=16,
        audio_num_attention_heads=2,
        audio_attention_head_dim=8,
        audio_in_channels=ACH * MEL,
        audio_out_channels=ACH * MEL,
        audio_cross_attention_dim=16,
    )
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, 0.02)
    return X0Model(model).float().eval()


def _tools(num_latent_frames: int) -> tuple[VideoLatentTools, AudioLatentTools]:
    v_shape = VideoLatentShape(1, CH, num_latent_frames, H, W)
    total_audio = int(round((num_latent_frames - 1) * 8 / FPS * 25)) + 1
    a_shape = AudioLatentShape(1, ACH, total_audio, MEL)
    return (
        VideoLatentTools(VideoLatentPatchifier(patch_size=1), v_shape, FPS),
        AudioLatentTools(AudioPatchifier(patch_size=1), a_shape),
    )


def _base_kwargs(x0: X0Model, num_latent_frames: int, *, cache_cross_attn: bool) -> dict:
    device = torch.device("cpu")
    video_tools, audio_tools = _tools(num_latent_frames)
    torch.manual_seed(11)  # identical sink tensor across runs
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
        noiser=GaussianNoiser(torch.Generator(device=device).manual_seed(0)),
        dtype=torch.float32,
        device=device,
        causal_cross_attn=True,
        cross_attn_lookahead_sec=0.0,
        cache_cross_attn=cache_cross_attn,
    )


CTX_V = torch.randn(1, 4, 16, generator=torch.Generator().manual_seed(7))
CTX_A = torch.randn(1, 4, 16, generator=torch.Generator().manual_seed(8))


def run_interactive(x0: X0Model, n: int, *, ccx: bool, strategy: str) -> torch.Tensor:
    kw = _base_kwargs(x0, n, cache_cross_attn=ccx)
    kw["context_resolver"] = lambda *_: (CTX_V, CTX_A)
    chunks = list(iter_streaming_chunks_joint_cached(**kw, strategy=strategy))
    return chunks[-1].video_latent_prefix


def run_offline(x0: X0Model, n: int, *, ccx: bool, strategy: str) -> torch.Tensor:
    kw = _base_kwargs(x0, n, cache_cross_attn=ccx)
    kw["v_context"] = CTX_V
    kw["a_context"] = CTX_A
    return streaming_generate_joint_cached(**kw, strategy=strategy)[0]


def main() -> None:
    x0 = build_tiny()

    # Phase 1: one chunk -> no history -> the flag cannot change anything.
    for s in STRATEGIES:
        off = run_interactive(x0, 2, ccx=False, strategy=s)
        on = run_interactive(x0, 2, ccx=True, strategy=s)
        d = (off - on).abs().max().item()
        print(f"[phase1] {s:12s} single chunk no-op: max|diff|={d:.3e}")
        assert d == 0.0, f"{s}: cache_cross_attn must be a no-op without history, got {d:.3e}"

    # Phase 2: four chunks -> history exists -> the flag must take effect.
    for s in STRATEGIES:
        off = run_interactive(x0, 5, ccx=False, strategy=s)
        on = run_interactive(x0, 5, ccx=True, strategy=s)
        d = (off - on).abs().max().item()
        print(f"[phase2] {s:12s} multi-chunk effect: max|diff|={d:.3e}")
        assert torch.isfinite(on).all(), f"{s}: non-finite with cache_cross_attn"
        assert d > 0.0, (
            f"{s}: cache_cross_attn had no effect across 4 chunks — the flag is probably "
            "not reaching CausalStreamingModel(...), so the a2v/v2a caches are never built"
        )

    # Phase 3: the interactive driver must match the offline driver either way.
    for s in STRATEGIES:
        for ccx in (False, True):
            itr = run_interactive(x0, 5, ccx=ccx, strategy=s)
            off = run_offline(x0, 5, ccx=ccx, strategy=s)
            d = (itr - off).abs().max().item()
            print(f"[phase3] {s:12s} ccx={ccx!s:5s} interactive vs offline: max|diff|={d:.3e}")
            assert d < 1e-6, f"{s} (ccx={ccx}): interactive must reproduce offline, got {d:.3e}"

    print("\nINTERACTIVE CROSS-ATTENTION CACHE VALIDATION PASSED")


if __name__ == "__main__":
    main()
