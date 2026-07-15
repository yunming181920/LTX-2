<!--
MAINTENANCE: When modifying any pipeline class in src/ltx_pipelines/,
update this document to reflect changes to:
- __init__ / __call__ signatures
- sigma handling or step counts
- denoiser types or guidance
- new or removed pipelines
Run: ls src/ltx_pipelines/*.py to check for new pipeline files.
-->

# ltx-pipelines

Inference pipelines for LTX-2 audio-video generation. Depends on `ltx-core` for model definitions, diffusion components, and loading. All pipelines live in `packages/ltx-pipelines/src/ltx_pipelines/`.

## Pipeline selection

| Pipeline | File | Stages | Model | Sampler | Use case |
|----------|------|--------|-------|---------|----------|
| `TI2VidOneStagePipeline` | `ti2vid_one_stage.py` | 1 | Full | Euler | Simple text/image-to-video |
| `T2AOneStagePipeline` | `t2a_one_stage.py` | 1 | Full | Euler | Text-to-audio (audio-only output, no video branch) |
| `TI2VidTwoStagesPipeline` | `ti2vid_two_stages.py` | 2 | Full + distilled LoRA | Euler | Production quality |
| `TI2VidTwoStagesHQPipeline` | `ti2vid_two_stages_hq.py` | 2 | Full + distilled LoRA (both stages) | Res2s | Highest quality, fewer steps |
| `A2VidPipelineTwoStage` | `a2vid_two_stage.py` | 2 | Full + distilled LoRA | Euler | Audio-conditioned video |
| `KeyframeInterpolationPipeline` | `keyframe_interpolation.py` | 2 | Full + distilled LoRA | Euler | Keyframe interpolation |
| `DistilledPipeline` | `distilled.py` | 2 | Distilled only | Euler | Fastest inference |
| `ICLoraPipeline` | `ic_lora.py` | 2 | Distilled only | Euler | Video-to-video with IC-LoRA control |
| `LipDubPipeline` | `lipdub.py` | 2 | Distilled only | Euler | Lip dubbing with IC-LoRA + audio ref conditioning |
| `RetakePipeline` | `retake.py` | 1 | Full or distilled | Euler | Video region regeneration |
| `TI2VidStreamingPipeline` | `ti2vid_streaming.py` | 1 | Full | Euler | Streaming causal video+audio (joint generation) |

## Guidance

- **CFG**: Blends conditioned/unconditioned predictions. Defaults: `cfg_scale=3.0` (video), `7.0` (audio).
- **STG**: Perturbs self-attention in transformer blocks. Default `stg_scale=1.0`, `stg_blocks=[28]` (LTX-2.3) / `[29]` (LTX-2). HQ disables STG (`stg_scale=0.0`).
- **Modality guidance**: Cross-modal attention scaling (`modality_scale=3.0`).
- All guidance is stage 1 only. Stage 2 always uses `SimpleDenoiser`.

## Sigma schedules and step counts

- **Scheduler-based** (full model): `self._scheduler = LTX2Scheduler()` with `execute(steps=N)` (HQ also passes `latent=` for token-count-dependent shift). Defaults: 30 steps (LTX-2.3), 40 (LTX-2), 15 (HQ).
- **Distilled**: Fixed 8-step `DISTILLED_SIGMA_VALUES` (9 values). Stage 2 uses 3-step `STAGE_2_DISTILLED_SIGMA_VALUES` (4 values). No `num_inference_steps` param.
- **Retake**: `num_inference_steps=40` default; ignored when `distilled=True` (fixed 8-step).
- **Overrides**: All pipelines accept optional sigma tensors in `__call__`: `sigmas` (one-stage), `stage_1_sigmas` + `stage_2_sigmas` (two-stage).

## LoRA conventions

- No default LoRAs. `loras` param defaults to empty list/tuple. `DEFAULT_LORA_STRENGTH = 1.0`.
- Two-stage non-distilled pipelines require `distilled_lora` (applied to stage 2 only in TI2Vid/A2Vid/Keyframe).
- HQ is unique: applies distilled LoRA to **both** stages with separate `distilled_lora_strength_stage_1` / `_stage_2` params.

## Shared building blocks (`utils/blocks.py`)

- `DiffusionStage` -- owns transformer lifecycle; builds model on call, frees on exit via `gpu_model()` context manager (moves params to meta device to release GPU/CPU memory). Accepts optional `stepper` and `loop` overrides. `__init__` takes a pre-built transformer builder; pipelines construct it via the `DiffusionStage.from_checkpoint(checkpoint_path, ..., loras=...)` classmethod, which builds the standard (and, when offloading, streaming) builders. `with_builder` / `with_loras` return a new stage with a swapped builder / LoRA set without re-specifying config.
- `PromptEncoder` -- Gemma text encoder + embeddings processor (video 4096-dim, audio 2048-dim).
- `ImageConditioner` / `AudioConditioner` -- temporary encoder scope; builds encoder, passes to callable, frees.
- `VideoUpsampler` -- 2x spatial upsampling via encoder + upsampler.
- `VideoDecoder` / `AudioDecoder` -- latent-to-pixel decoding (iterator for video, `Audio` for audio).

### Memory management

- **Model lifecycle**: All blocks build their model on call and free it on exit. `gpu_model()` moves params to `"meta"` device on exit, immediately releasing storage. No model persists between calls.
- **Block streaming**: When offloading is enabled, `DiffusionStage` wraps the transformer in `BlockStreamingWrapper`. Blocks live on pinned CPU memory; only 2 blocks are buffered on GPU at a time (one for compute, one for async H2D copy on a separate CUDA stream).
- **Batch splitting**: `BatchSplitAdapter` wraps the transformer and splits inputs exceeding `max_batch_size` into sequential chunks. If guidance needs B=4 but `max_batch_size=1`, it runs 4 sequential B=1 passes. Higher `max_batch_size` reduces layer-streaming PCIe transfers at the cost of peak memory.

## Denoisers (`utils/denoisers.py`)

- `SimpleDenoiser` -- single forward pass (B=1), no guidance. Used by distilled pipelines and all stage 2.
- `GuidedDenoiser` -- CFG/STG with static `MultiModalGuider` instances (HQ, A2Vid, Retake non-distilled).
- `FactoryGuidedDenoiser` -- per-step guider creation via factory (OneStageTI2Vid, TwoStagesTI2Vid, Keyframe).

All denoisers return a `(video_result, audio_result)` tuple of `DenoisedLatentResult` (defined in `utils/types.py`), either element may be `None` for absent modalities. `DenoisedLatentResult.denoised` is the final blended tensor. Guided denoisers additionally populate per-pass fields (`.cond`, `.uncond`, `.ptb`, `.mod`) on each result; `SimpleDenoiser` leaves these `None`.

`GuidedDenoiser` and `FactoryGuidedDenoiser` accept `force_uncond_pass=True` to run the uncond pass even when `cfg_scale=1.0` (required by CFG++ when the guidance scale is 1 but the uncond prediction is still needed for the ODE derivative). Requires `negative_context` to be set on the guider. When enabled, `DenoisedLatentResult.uncond` will be a tensor instead of `None`.

Guided denoisers batch all guidance passes into a **single transformer call**: states are repeated along the batch dimension, contexts concatenated, and a `BatchedPerturbationConfig` controls which attention ops are skipped per sample. Pass count is dynamic: B=2 for CFG-only, up to B=4 with CFG+STG+modality isolation. Results are split back and blended by the guider.

## Per-pipeline unique features

- **HQ**: Res2s second-order sampler for **both** stages, latent-dependent sigma schedule, distilled LoRA on both stages with separate strengths.
- **A2Vid**: Audio frozen in both stages (`frozen=True, noise_scale=0.0`). Returns original audio (not VAE-decoded); no `AudioDecoder`.
- **IC-LoRA**: `VideoConditionByReferenceLatent`, `reference_downscale_factor` from LoRA metadata, `skip_stage_2`, attention mask downsampling. Stage 2 is LoRA-free and uses `combined_image_conditionings` (no IC-LoRA conditioning).
- **LipDub**: Standalone pipeline; IC reference **video** helpers in `iclora_utils.py`, LipDub-only **audio** patchify/negative positions in `lipdub.py`. Appends frozen audio-reference tokens via `AudioConditionByReferenceLatent` (ltx-core), matching video token order (`[target | ref]`) while keeping reference RoPE positions negative (training-compatible). Single IC-LoRA on both stages; full IC-LoRA video conditioning at stage 1 and 2; stage-2 audio is frozen with S1 latent as initial state and uses S1-derived ref. Final audio decoded from stage 1 latent. The LipDub CLI does not expose `--conditioning-attention-mask`; use `ic_lora.py` if you need spatial IC attention masking.
- **Keyframe**: Uses `image_conditionings_by_adding_guiding_latent` in both stages (all frames as keyframe guidance, no replacement) -- unlike TI2Vid which uses `combined_image_conditionings` (frame_idx=0 replaces, others guide).
- **Retake**: `TemporalRegionMask` for selective time-window regeneration. `regenerate_video`/`regenerate_audio` flags. Conditional distilled/full behavior.
- **Distilled**: Single `self.stage` reused for both stages (not `stage_1`/`stage_2`).

## Streaming causal pipelines (experimental)

`TI2VidStreamingPipeline` (`ti2vid_streaming.py`) is a training-free reproduction of Vidu S1 §2.3 streaming inference on top of the pretrained *bidirectional* LTX-2 checkpoint (used as-is as the "causal model", no training). Primitives live in `utils/streaming.py`; the M2 KV-cache lives in `ltx-core` (`model/transformer/streaming_cache.py`, `streaming_model.py`, the `_stream_cached_forward` path in `attention.py`).

TI2V has no audio input — video **and** audio are generated in lockstep. Each video AR chunk also produces its time-aligned audio frames; audio keeps its own sliding-window FIFO history (no sink / no persistent anchor, since audio has no image conditioning) so per-step memory stays O(window) for both modalities.

Shared mechanics: persistent video reference context = encoded first-frame "sink" + first generated video chunk (always clean, never evicted); latent-level TwinCache (noisy snapshot at a mid denoising step, clean after the final step; frozen via per-token `denoise_mask` + `latent == clean_latent` → Euler velocity 0); block-causal self-attention mask on each modality's temporal axis (a structured `BlockCausalMask` from `ltx_core.model.transformer.masking` — per query frame block, the visible contiguous key prefix; served by exact *unmasked* per-block prefix attention calls so the causal paths run on FlashAttention backends, which have no additive-mask kernel); time-causal video↔audio cross-attention mask (`--causal-cross-attn`, default on) with the audio window's clock aligned to the video window's compressed clock (`_audio_window_alignment` — the video window pins sink+first at its head, so after eviction the audio grid must start at its absolute frame minus the video compression shift or AV sync breaks) and an earliest-key fallback for query rows with no causally visible key. CLI flags: `--window-chunks` (FIFO cap), `--chunk-frames` (latent video frames per AR step), `--causal-cross-attn` / `--cross-attn-lookahead-seconds`, `--use-kv-cache` (M2).

Two execution paths:
- **M1 (default)** — latent TwinCache, full per-step recompute of history features. The correct, recommended path.
- **M2 (`--use-kv-cache`)** — per-block KV cache + RoPE repositioning. Caches **both** video self-attn (`attn1`, 1-frame sink + persistent first chunk) and audio self-attn (`audio_attn1`, no-sink FIFO) via `CausalStreamingModel(..., cache_audio=True)`. The clean TwinCache K/V snapshot is captured by one extra sigma-0 forward on the finalized latents per chunk (Vidu S1: clean cache "obtained after the final denoising step"). Faster, but **conceptual/unvalidated** — run the parity test before trusting. Production bidirectional pipelines are unaffected (`attn1`/`audio_attn1.stream_cache` stay `None` unless a streaming driver attaches them).

Validation: `tests/test_streaming_joint.py` (pure-tensor, no checkpoint) + `tests/test_streaming_tiny_model.py` (CPU, random 2-layer model: single-chunk M1≡M2 bitwise parity with causal cross-attn off/on + multi-chunk eviction smoke for both paths) + `tests/test_streaming_joint_parity.py` (GPU integration: single-chunk M1≡M2 parity + multi-chunk smoke, both modalities, real checkpoint). Run the GPU parity test before trusting M2 on real weights. These are conceptual reproductions, not quality-tuned (the base model is bidirectionally trained, so the causal masks are a train/test mismatch).

## Image conditioning helpers (`utils/helpers.py`)

- `combined_image_conditionings()` -- images with `frame_idx==0` replace latent (`VideoConditionByLatentIndex`), others guide (`VideoConditionByKeyframeIndex`).
- `image_conditionings_by_adding_guiding_latent()` -- all images become keyframe guidance regardless of `frame_idx`.
