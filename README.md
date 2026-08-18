# LTX-2 · Vidu S1 §2.3 Streaming Causal TI2V Reproduction

[![Website](https://img.shields.io/badge/Website-LTX-181717?logo=google-chrome)](https://ltx.io)
[![Model](https://img.shields.io/badge/ModelScope-LTX--2.5-orange?logo=alibabacloud)](https://www.modelscope.cn/models/Lightricks/LTX-2.5)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-LTX--2.5-yellow?logo=huggingface)](https://huggingface.co/Lightricks/LTX-2.5)
[![Paper](https://img.shields.io/badge/Paper-PDF-EC1C24?logo=adobeacrobatreader&logoColor=white)](https://arxiv.org/abs/2601.03233)

> A **training-free** reproduction of the streaming inference described in
> **Vidu S1** (arXiv:[2607.03118](https://arxiv.org/abs/2607.03118), §2.3) on top of
> the pretrained **bidirectional** LTX-2 checkpoint, which is used as-is as the
> “causal model”. This is a fork of [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2),
> ported to the **LTX-2.5** split-pack API. The additions live in
> `packages/ltx-pipelines` (streaming driver + pipeline + Gradio app) and
> `packages/ltx-core` (KV-cache attention surgery).

---

## ✍️ Author's Notes / 作者见解

**English.** Both **Vidu S1** and **Wan Streamer 0.3** implement this kind of
interactive (streaming, memory-based) video generation. My guess is that
**Wan Streamer 0.3**'s condition injection uses embeddings produced by a
**Qwen3.5-Omni** multimodal model, whereas **Vidu S1** may condition on a
multimodal model — or possibly on pure text. As for this project: it has been
verified to actually run on **dual 4090 GPUs**, but the results are **not ideal** —
a natively **full-attention** model cannot be directly repurposed for
**memory-augmented causal attention**.

**中文.** **Vidu S1** 和 **Wan Streamer 0.3** 都实现了这种交互式（带记忆的流式）视频生成。
我猜测 **Wan Streamer 0.3** 的 condition 注入可能是用 **Qwen3.5-Omni** 多模态模型得到的 embedding，
而 **Vidu S1** 使用的可能是多模态模型，也可能是纯文本。至于本项目：经 **双 4090** 实测确实能跑起来，
但结果 **不太理想** —— 原生的 **全注意力** 模型无法直接套用 **带记忆的因果注意力**。

---

## English

### What this is

Vidu S1 is a real-time interactive audio→video model whose §2.3 describes a
**streaming, autoregressive, causal** inference scheme: a sliding window with a
persistent reference context (“sink”), cached historical states, **RoPE
Repositioning**, and **TwinCache**. Vidu S1 trains a dedicated causal teacher
(Stage 2) and distills it (Stage 3) to reach 3-step / 42 FPS.

This repo reproduces **only the §2.3.1 streaming inference, training-free** —
i.e. it reuses LTX-2’s pretrained bidirectional weights without any fine-tuning,
now on the **LTX-2.5** split-pack checkpoints. Per the chosen scope:

- **Inference only** — no training (Stages 1–3 are skipped).
- **Joint video + audio generation** — TI2V has no audio input; both modalities
  are generated chunk-by-chunk in lockstep.
- **Full-fidelity streaming machinery** — causal masks + sliding window +
  persistent anchor + TwinCache + KV cache + RoPE repositioning, for *both*
  video and audio. Chunk 1 is a standard **bidirectional ti2v bootstrap** (the
  reference image replaces latent frame 0, full attention); its `[image |
  chunk 1]` output is the never-evicted anchor of every later causal window
  `[anchor | history | current]`.
- **Four streaming strategies** (KV-cache TwinCache default + two history-read
  ablations + one image-conditioning baseline) — selectable per generation from
  the CLI or the webui dropdown.

### Download models from ModelScope

The LTX-2.5 weights are mirrored on [ModelScope](https://www.modelscope.cn/models/Lightricks/LTX-2.5).
The streaming pipeline needs four files (the spatial upsampler is **not** used by streaming):

| Component | Repo path | Used as |
|---|---|---|
| **Diffusion transformer (DiT)** | `diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors` | `--transformer-path` |
| **Text encoder (Gemma 4 12B + proj)** | `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` | `--text-encoder-path` |
| **Video VAE (conv, recommended for streaming)** | `vae/ltx-2.5-video-vae-conv-bf16.safetensors` | `--video-vae-path` |
| **Audio VAE + vocoder** | `vae/ltx-2.5-audio-vae-bf16.safetensors` | `--audio-vae-path` |

> The conv video VAE (`...-conv-bf16...`) keeps the 32×32×8 latent geometry the
> streaming chunk math is hardcoded for; the default diffusion video VAE
> (`ltx-2.5-video-vae-bf16.safetensors`) has a different geometry and may break
> AV sync in streaming — use the conv variant for streaming.
>
> The **distilled** transformer (specified above) runs in far fewer steps; the
> **full dev** transformer `diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors`
> gives better streaming quality (the streaming path is designed for the full
> model) — download it too if you want the higher-fidelity option.

Install the ModelScope SDK, then download exactly the four files:

```bash
pip install modelscope
modelscope download \
    --model Lightricks/LTX-2.5 \
    --include \
        "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors" \
        "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
        "vae/ltx-2.5-video-vae-conv-bf16.safetensors" \
        "vae/ltx-2.5-audio-vae-bf16.safetensors" \
    --local_dir models/ltx-2.5
```

Or with the Python SDK (preserves the repo’s folder layout under `--local_dir`):

```python
from modelscope import snapshot_download
snapshot_download(
    "Lightricks/LTX-2.5",
    allow_patterns=[
        "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
        "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
        "vae/ltx-2.5-audio-vae-bf16.safetensors",
    ],
    local_dir="models/ltx-2.5",
)
```

The resulting layout (the CLI keeps the repo’s folder structure under `--local_dir`):

```
models/ltx-2.5/
├─ diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors
├─ text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors
└─ vae/
   ├─ ltx-2.5-video-vae-conv-bf16.safetensors
   └─ ltx-2.5-audio-vae-bf16.safetensors
```

### Install (uv monorepo)

```bash
uv sync --extra natten     # natten is the fastest VAE backend (Linux + CUDA only;
                           # skipped automatically elsewhere, falls back to Triton/eager)
```

### Quick start (CLI streaming)

```bash
uv run python -m ltx_pipelines.ti2vid_streaming \
    --transformer-path       models/ltx-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
    --text-encoder-path      models/ltx-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
    --video-vae-path         models/ltx-2.5/vae/ltx-2.5-video-vae-conv-bf16.safetensors \
    --audio-vae-path         models/ltx-2.5/vae/ltx-2.5-audio-vae-bf16.safetensors \
    --prompt "A person talking calmly to the camera." \
    --image ref.jpg 0 1.0 \
    --output-path out.mp4 \
    --num-frames 33 --frame-rate 24 --height 512 --width 768 \
    --num-inference-steps 15 --window-chunks 1 --chunk-frames 3 \
    --stream-strategy kv_twin
```

There is **no audio input** — audio is generated jointly with the video and written
into the output file. On low-VRAM GPUs add `--quantization fp8-cast --offload cpu`.

> The 2.3 monolith layout (`--checkpoint-path models/ltx-2.3/ltx-2.3-22b-dev.safetensors`
> + `--gemma-root models/gemma-3-12b`) is still supported as a fallback.

### Streaming strategies (`--stream-strategy`)

Four strategies, selectable per run (ablation):

| Strategy | Description | Memory |
|---|---|---|
| `kv_twin` (default) | KV cache + RoPE repositioning; history reads **noisy** at mid steps + **clean** at the final step (Vidu S1 §2.3.1). | baseline |
| `kv_clean` (ablation A) | KV cache; history reads **clean** at *every* step (pure-clean history). | ≈ kv_twin |
| `kv_noisy_steps` (ablation B) | KV cache; step *t* reads the history's own step-*t* **noisy** snapshot (noise-level matched). No clean, no sigma-0 forward. | ~`num_steps/2`× kv_twin ⚠️ |
| `image_cond` (ablation C) | No KV cache; each chunk conditions on the **previous chunk's last frame** as the image reference (rotating sink), no attention history. | ≈ baseline |

> `--use-kv-cache` is kept as a legacy no-op (the KV-cache path is the default).
>
> **B (`kv_noisy_steps`) memory:** each history chunk stores `num_steps` noisy
> K/V snapshots — about `num_steps/2`× the KV memory of `kv_twin`. Run B with a
> lower `--num-inference-steps` (8–15), `--window-chunks 2`, or `--offload cpu`.

#### Streaming-specific CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--stream-strategy` | `kv_twin` | One of `kv_twin` / `kv_clean` / `kv_noisy_steps` / `image_cond` |
| `--window-chunks` | 1 | Sliding-window rolling-history size in AR chunks (TwinCache FIFO cap; the persistent items — video anchor `[image | chunk 1]`, audio first chunk — are not counted) |
| `--chunk-frames` | 3 | Latent video frames generated per AR step (1 image + 3 = 4 latents per window). The causal VAE decodes the window as 1 + frames×8 pixel frames: the first latent frame (the reference image) decodes to 1 frame, every later latent frame to 8, so a 3-latent chunk yields 1 + 3×8 = 25 pixel frames (≈ 1 s at 24 fps). Each step also generates time-aligned audio frames in lockstep |
| `--causal-cross-attn` / `--no-causal-cross-attn` | on | Time-causal mask on video↔audio cross-attention (default ON, paper-faithful; train/test mismatch on the bidirectional base model) |
| `--cross-attn-lookahead-seconds` | 0.0 | Seconds of future audio a video frame may attend under causal cross-attn (0 = strict causal) |

Standard LTX-2 flags (`--seed`, `--quantization`, `--offload`, `--compile`,
`--enhance-prompt`, `--lora`, `--prompt`, `--negative-prompt`, `--image PATH FRAME_IDX STRENGTH [CRF]`,
`--num-frames`, `--frame-rate`, `--height`, `--width`, `--num-inference-steps`,
`--diffvae-optimization`) behave as in the upstream 2.5 pipelines.

### The streaming path (KV cache + RoPE repositioning)

`streaming.py::streaming_generate_joint_cached` + `streaming_cache.py` +
`streaming_model.py` + `attention.py`: per-block KV cache on **both** video
self-attn (`attn1`) and audio self-attn (`audio_attn1`), history K/V spliced
from the cache, RoPE reapplied with the full-window `window_pe`. Chunk 1 is a
**bidirectional ti2v bootstrap** — the image replaces latent frame 0, no
causal masks — and its `[image | chunk 1]` K/V is committed straight into the
caches' permanent first slots (the never-evicted anchor; no separate sink
block to recompute each step). Later windows are `[anchor | history | current]`
(video) / `[first | history | current]` (audio) with block-causal query masks.
The `strategy=` arg selects which TwinCache variant each cache uses
(`kv_twin` / `kv_clean` / `kv_noisy_steps`). Production pipelines are untouched
(`*.stream_cache` stays `None` → byte-identical standard path).

### Design note: why a bidirectional anchor bootstrap (not the paper's sink)

Vidu S1 runs on its own Vidu base model — by all indications a
full-self-attention architecture over image / video / audio / text with
MMRoPE-style positional encoding (the same family as MiniMax H3). There the
§2.3.1 sink is a first-class, *trained* component. Porting it onto LTX-2 hits
concrete architectural obstacles:

1. **The sink's K/V are not stationary in LTX-2.** Video tokens cross-attend
   to audio (and vice versa) at every block, so a pinned first-frame token's
   features depend on the current audio slice: the paper's "constructed once,
   remains fixed" cache would freeze stale cross-modal state, while
   recomputing the sink's K/V every step pays full price for one frame the
   checkpoint was never trained to use that way.
2. **The base is bidirectional.** A permanently pinned single-frame reference
   inside a causal window is a train/test mismatch at the most influential
   position of the stream — the anchor that every later chunk attends to.

So chunk 1 is generated with **standard ti2v** (the reference image replaces
latent frame 0, full attention — exactly how LTX-2 was trained), and the whole
`[image | chunk 1]` output becomes the pinned anchor. The paper's
persistent-reference *content* (first frame + first generated video-audio
state) survives intact inside one pinned chunk, with no special sink
mechanics left to maintain.

On the text side: Vidu S1's interface is conversational. The closest relatives
available today — MiniMax H3, and JoyAI-Echo (jd-opensource, an autoregressive
video model built on LTX-2.3) — drive generation with formatted/structured
prompt schemas rather than dialogue. LTX-2's text conditioning (Gemma,
natural-language prompts) is exactly the simple, natural-language kind, so
*interactive* descriptions fill Vidu S1's conversational role here — see the
live-prompt Gradio app below.

### Interactive streaming (Gradio webui, live prompt injection)

`packages/ltx-pipelines/src/ltx_pipelines/app_gradio.py` is an **interactive** front
end on top of the same streaming machinery: a long-lived session (DiT + Gemma + VAEs
built once) generates chunk-by-chunk and **streams** the growing video + live audio to
a browser. While it streams, editing the **live prompt** textbox rewrites the
cross-attention conditioning for the *next* chunk — so the generated content changes
mid-clip. This is safe because text is cross-attention only: it is **not** part of the
cached self-attention history or the anchor, so a prompt change does **not** reset the
video or invalidate the sliding-window state.

The **Stream strategy** dropdown in the “Generation settings” panel exposes the same
four `--stream-strategy` options — selectable **per generation, without restarting
the app** (the KV-cache wrapper is built and detached inside the generator each run).
This makes A/B ablation between strategies convenient in one session. `--stream-strategy`
on the CLI sets the dropdown’s default.

```bash
uv run python -m ltx_pipelines.app_gradio \
    --transformer-path       models/ltx-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
    --text-encoder-path      models/ltx-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
    --video-vae-path         models/ltx-2.5/vae/ltx-2.5-video-vae-conv-bf16.safetensors \
    --audio-vae-path         models/ltx-2.5/vae/ltx-2.5-audio-vae-bf16.safetensors \
    --stream-strategy kv_clean
```

Open the printed URL, upload a reference image (it becomes the anchor's frame 0), set the initial prompt,
hit **Generate**, and edit the live prompt while it streams. `--host/--port/--share`
control serving; `--device` / `--text-encoder-device` can split the 12B text encoder
onto a second GPU.

Notes:
- `fp8-scaled-mm` is for **pre-quantized** fp8 checkpoints (on-disk `F8_E4M3` weight +
  `.weight_scale`); `fp8-cast` is for bf16 checkpoints (casts bf16→fp8 at load).
- Do **not** combine `--quantization fp8-scaled-mm` with `--offload cpu/disk` — block
  streaming only supports bf16 / `fp8-cast`. The long-lived session keeps DiT + Gemma
  (12B, bf16) + VAEs resident at once — expect high VRAM use (≈ A100/H100 class). If
  VRAM-constrained, use `--offload cpu`.
- The distilled weights are tuned for 3-step *bidirectional* generation; the causal
  streaming path is a train/test mismatch — it runs, but quality is better with the
  full dev transformer. If you do use the distilled checkpoint, set a low step count
  (≈ 3–8) in the UI.

### Tests

Checkpoint-free CPU tests cover the streaming plumbing and all four strategies:

```bash
uv run python packages/ltx-pipelines/tests/test_streaming_joint.py          # pure-tensor helpers
uv run python packages/ltx-pipelines/tests/test_streaming_tiny_model.py     # tiny model: 4-strategy parity + finiteness
uv run python packages/ltx-pipelines/tests/test_streaming_interactive.py   # interactive: 4-strategy parity + finiteness
```

`test_streaming_tiny_model.py` asserts single-chunk parity across the three KV
strategies (no history ⇒ identical; they share the bidirectional bootstrap)
with `image_cond` checked for finiteness (its all-causal rotating-sink path
legitimately differs on chunk 1), plus multi-chunk finiteness;
`test_streaming_interactive.py` does the same for the Gradio/interactive
driver, plus constant-context parity against the offline driver, the
live-prompt swap guarantee and the resolver call-count. Smoke-test at
checkpoint scale before trusting quality in production.

### Known limitations vs. the paper

1. **No training (Stages 1–3).** The bidirectional checkpoint is used as-is as the
   causal model — the largest quality gap.
2. **Bidirectional chunk-1 bootstrap.** The first chunk is denoised with full
   (non-causal) attention to match the base model's training and anchor the
   stream on a high-quality ti2v output; the paper's causal model generates it
   causally. Later chunks are causal as in the paper. (Rationale in the design
   note above.)
3. **No Stage-3 distillation.** Generation uses the full step count (default 30, or 15
   for the distilled checkpoint), not 3 steps. The paper’s 42 FPS / 540p headline is
   unreachable without distillation + the §2.3.2 infra stack.
4. **No §2.3.2 inference infrastructure.** SageAttention / SpargeAttention / SLA,
   custom W8A8 GEMM, kernel fusion, CUDA Graph, Ulysses multi-GPU parallelism are
   not implemented. LTX-2’s own `fp8-cast` / `fp8-scaled-mm` quantization and
   FlashAttention still work.
5. **AV cross-attention causal mask is ON by default** — a train/test mismatch on
   the bidirectional base model, but preferred for paper-faithful causality
   (`--no-causal-cross-attn` to disable).
6. **No CFG.** `--negative-prompt` is encoded but unused (single forward pass).

### Files added / changed by this fork

- `packages/ltx-pipelines/src/ltx_pipelines/utils/streaming.py` — the KV-cache
  joint streaming driver (incl. `streaming_generate_joint_image_cond` for
  ablation C) + shared causal-streaming primitives.
- `packages/ltx-pipelines/src/ltx_pipelines/utils/streaming_interactive.py` —
  interactive generator drivers (`iter_streaming_chunks_joint` /
  `_cached` / `_image_cond`) with per-chunk context resolver + yielding.
- `packages/ltx-pipelines/src/ltx_pipelines/ti2vid_streaming.py` — `TI2VidStreamingPipeline`
  + CLI (`--stream-strategy`).
- `packages/ltx-pipelines/src/ltx_pipelines/interactive_session.py` — long-lived
  session + `LivePromptEncoder` + incremental decode + `stream_strategy` dispatch.
- `packages/ltx-pipelines/src/ltx_pipelines/app_gradio.py` — Gradio webui (Stream
  strategy dropdown).
- `packages/ltx-core/src/ltx_core/model/transformer/streaming_cache.py` —
  `StreamingKVCache` (twin / clean / noisy_steps strategies; persistent-first
  slots — the joint TI2V video cache commits the `[image | chunk 1]` anchor,
  audio pins its first chunk; both no-sink, A2V keeps the sink-carrying
  default).
- `packages/ltx-core/src/ltx_core/model/transformer/streaming_model.py` —
  `CausalStreamingModel` (video + optional audio caches; threads strategy/step_idx).
- `packages/ltx-core/src/ltx_core/model/transformer/attention.py` — `stream_cache`
  attr + `_stream_cached_forward` (sink-less layouts for both modalities; strategy-agnostic).
- `packages/ltx-pipelines/tests/test_streaming_{joint,tiny_model,interactive}.py` —
  pure-tensor / tiny-model / interactive tests covering all four strategies.

### Upstream LTX-2

Everything else (core model, other pipelines, trainer) is unchanged upstream
LTX-2 (v1.2.0 / LTX-2.5). See the per-package READMEs under `packages/` and
<https://www.modelscope.cn/models/Lightricks/LTX-2.5>.

---

## 中文

### 这是什么

Vidu S1（arXiv:2607.03118）是一个实时交互式音→视频模型，其 §2.3 描述了一套
**流式、自回归、因果** 的推理方案：滑动窗口 + 持久参考上下文（“sink”）+ 缓存历史状态 +
**RoPE 重定位** + **TwinCache**。Vidu S1 训练了专门的因果教师（Stage 2）并蒸馏（Stage 3）
以实现 3 步 / 42 FPS。

本仓库 **仅复现 §2.3.1 的流式推理，且为 training-free** —— 直接复用 LTX-2 预训练双向权重，
现已移植到 **LTX-2.5** split-pack 接口。按既定范围：

- **仅推理** —— 不训练（跳过 Stage 1–3）。
- **音视频联合生成** —— TI2V 无音频输入；视频与音频按 chunk 同步逐段生成。
- **完整保真的流式机制** —— 因果掩码 + 滑动窗口 + 持久锚点 + TwinCache + KV cache + RoPE 重定位，**音视频皆然**。
  第一个 chunk 用标准 **双向 ti2v bootstrap**（参考图替换 latent 帧 0、全注意力）生成，其
  `[图 | chunk 1]` 输出成为后续因果窗口 `[anchor | history | current]` 永不淘汰的锚。
- **四种流式策略**（三种 KV cache 变体 + 一种 image-conditioning 基线）用于消融，
  可从 CLI 或 webui 下拉框按次生成选择。

### 从 ModelScope 下载模型

LTX-2.5 权重在 [ModelScope](https://www.modelscope.cn/models/Lightricks/LTX-2.5) 有镜像。
流式管线需要四个文件（流式**不使用**空间上采样器）：

| 组件 | 仓库路径 | 用作 |
|---|---|---|
| **扩散 transformer (DiT)** | `diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors` | `--transformer-path` |
| **文本编码器 (Gemma 4 12B + proj)** | `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` | `--text-encoder-path` |
| **视频 VAE（conv，流式推荐）** | `vae/ltx-2.5-video-vae-conv-bf16.safetensors` | `--video-vae-path` |
| **音频 VAE + vocoder** | `vae/ltx-2.5-audio-vae-bf16.safetensors` | `--audio-vae-path` |

> conv 视频 VAE（`...-conv-bf16...`）保留流式分块数学硬编码的 32×32×8 latent 几何；默认的 diffusion
> 视频 VAE（`ltx-2.5-video-vae-bf16.safetensors`）几何不同，流式下可能导致音画不同步——流式请用 conv 变体。
>
> 上面指定的**蒸馏** transformer 步数少；**完整 dev** transformer
> `diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors` 流式质量更好（流式路径是为完整模型设计的）——
> 想要更高保真也可一并下载。

安装 ModelScope SDK，下载这四个文件：

```bash
pip install modelscope
modelscope download \
    --model Lightricks/LTX-2.5 \
    --include \
        "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors" \
        "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
        "vae/ltx-2.5-video-vae-conv-bf16.safetensors" \
        "vae/ltx-2.5-audio-vae-bf16.safetensors" \
    --local_dir models/ltx-2.5
```

或用 Python SDK（保留仓库目录结构）：

```python
from modelscope import snapshot_download
snapshot_download(
    "Lightricks/LTX-2.5",
    allow_patterns=[
        "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
        "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
        "vae/ltx-2.5-audio-vae-bf16.safetensors",
    ],
    local_dir="models/ltx-2.5",
)
```

下载后的目录（CLI 在 `--local_dir` 下保留仓库目录结构）：

```
models/ltx-2.5/
├─ diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors
├─ text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors
└─ vae/
   ├─ ltx-2.5-video-vae-conv-bf16.safetensors
   └─ ltx-2.5-audio-vae-bf16.safetensors
```

### 安装（uv monorepo）

```bash
uv sync --extra natten     # natten 是最快的 VAE 后端（仅 Linux + CUDA；
                           # 其他平台自动跳过，回退到 Triton/eager）
```

### 快速开始（CLI 流式）

```bash
uv run python -m ltx_pipelines.ti2vid_streaming \
    --transformer-path       models/ltx-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
    --text-encoder-path      models/ltx-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
    --video-vae-path         models/ltx-2.5/vae/ltx-2.5-video-vae-conv-bf16.safetensors \
    --audio-vae-path         models/ltx-2.5/vae/ltx-2.5-audio-vae-bf16.safetensors \
    --prompt "一个人平静地对着镜头说话。" \
    --image ref.jpg 0 1.0 \
    --output-path out.mp4 \
    --num-frames 33 --frame-rate 24 --height 512 --width 768 \
    --num-inference-steps 15 --window-chunks 1 --chunk-frames 3 \
    --stream-strategy kv_twin
```

**无需音频输入** —— 音频与视频联合生成并写入输出文件。显存不足时加 `--quantization fp8-cast --offload cpu`。

> 2.3 monolith 布局（`--checkpoint-path models/ltx-2.3/ltx-2.3-22b-dev.safetensors` + `--gemma-root models/gemma-3-12b`）仍作为回退支持。

### 流式策略（`--stream-strategy`）

四种策略，可按次运行选择（消融）：

| 策略 | 说明 | 显存 |
|---|---|---|
| `kv_twin`（默认，M2） | KV cache；中间步读 **noisy**、末步读 **clean**（Vidu S1 §2.3.1）。 | M2 基准 |
| `kv_clean`（消融 A） | KV cache；所有步读 **clean**（纯干净 history）。 | ≈ M2 |
| `kv_noisy_steps`（消融 B） | KV cache；步 t 读 history 自身步 t 的 **noisy**（噪声级别匹配）。无 clean、无 sigma-0 额外前向。 | ~`num_steps/2`× kv_twin ⚠️ |
| `image_cond`（消融 C） | 无 KV cache；每块用**上一块末帧**做 image 参考（旋转 sink），无注意力 history。 | ≈ 基准 |

> `--use-kv-cache` 保留为 `--stream-strategy kv_twin` 的旧别名。
>
> **B（`kv_noisy_steps`）显存：** 每个 history chunk 存 `num_steps` 份 noisy K/V，约为 kv_twin 的 `num_steps/2` 倍。跑 B 请降低 `--num-inference-steps`（8–15）、`--window-chunks 2`，或 `--offload cpu`。

#### 流式专用 CLI 参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--stream-strategy` | `kv_twin` | `kv_twin` / `kv_clean` / `kv_noisy_steps` / `image_cond` 之一 |
| `--window-chunks` | 1 | 滑动窗口滚动历史大小（以 AR chunk 计；TwinCache FIFO 上限；持久项——视频锚 `[图 | chunk 1]`、音频首块——不计入） |
| `--chunk-frames` | 3 | 每个 AR 步生成的 latent 视频帧数（1 图 + 3 = 每窗口 4 个 latent）。因果 VAE 解码窗口为 1 + 帧数×8 个像素帧：第 1 个 latent 帧（参考图像）解码为 1 帧，之后每个 latent 帧解码为 8 帧，因此 3 个 latent 的 chunk 对应 1 + 3×8 = 25 个像素帧（24 fps 下约 1 秒）。每步同时生成时间对齐的音频帧 |
| `--causal-cross-attn` / `--no-causal-cross-attn` | 开 | 对 video↔audio 跨注意力施加时间因果掩码（默认开启，论文忠实；对双向基础模型是训练/测试不匹配） |
| `--cross-attn-lookahead-seconds` | 0.0 | 因果跨注意力下视频帧可看到的未来音频秒数（0 = 严格因果） |

标准 LTX-2 参数（`--seed`、`--quantization`、`--offload`、`--compile`、`--enhance-prompt`、
`--lora`、`--prompt`、`--negative-prompt`、`--image PATH FRAME_IDX STRENGTH [CRF]`、
`--num-frames`、`--frame-rate`、`--height`、`--width`、`--num-inference-steps`、
`--diffvae-optimization`）与上游 2.5 管线一致。

### 流式路径（KV cache + RoPE 重定位）

KV cache + RoPE 重定位（`streaming.py::streaming_generate_joint_cached` + `streaming_cache.py`
+ `streaming_model.py` + `attention.py`）：视频自注意力（`attn1`）**与**音频自注意力（`audio_attn1`）
各一个 KV cache，历史 K/V 从缓存拼接，用全窗 `window_pe` 重新施加 RoPE。`kv_twin` / `kv_clean` /
`kv_noisy_steps` 策略属 M2 族；`strategy=` 参数选择每个 cache 用哪种 TwinCache 变体。生产管线不受影响
（`*.stream_cache` 保持 `None` → 与标准路径逐字节一致）。

### 设计说明：为何用双向锚 bootstrap 而非论文的 sink

Vidu S1 跑在自家 Vidu 基模上——从各方迹象看，那是图像 / 视频 / 音频 / 文本全自注意力 +
MMRoPE 式位置编码的架构（与 MiniMax H3 同族）。在那样的模型里，§2.3.1 的 sink 是训练过的
原生组件；而搬到 LTX-2 上会撞上具体的架构障碍：

1. **LTX-2 里 sink 的 K/V 不平稳。** 视频 token 在每个 block 都与音频交叉注意力，钉住的
   首帧 token 特征依赖当前音频切片——论文 "constructed once, remains fixed" 的缓存会冻结
   陈旧的跨模态状态，而每步重算 sink 的 K/V（我们此前的做法）要为这一帧付出全价，且
   checkpoint 从未按这种用法训练过。
2. **基模是双向的。** 因果窗口里永久钉一帧参考，是发生在流中影响力最大位置上的训练/测试
   错配——每个后续 chunk 都要 attend 的锚。

因此 chunk 1 改用**标准 ti2v** 生成（参考图替换 latent 帧 0、全注意力——正是 LTX-2 的
训练方式），其 `[图 | chunk 1]` 输出整体成为钉住的锚。论文持久参考的**内容**（首帧 +
第一个生成的 video-audio 状态）完整保留在一个钉住的 chunk 里，无需再维护任何特殊 sink 机制。

文本侧：Vidu S1 的接口是对话式的。如今最接近的同族模型——MiniMax H3，以及 JoyAI-Echo
（jd-opensource，基于 LTX-2.3 的自回归视频模型）——驱动生成用的是格式化/结构化的提示词
模式而非对话。LTX-2 的文本端（Gemma、自然语言提示）恰好是简单的自然语言风格，因此
**交互式**描述在这里正好承担 Vidu S1 的对话式角色——见下文的 live-prompt Gradio 应用。

### 交互式流式（Gradio webui，实时改写提示词）

`packages/ltx-pipelines/src/ltx_pipelines/app_gradio.py` 是同一套流式机制之上的 **交互式** 前端：
一个长驻会话（DiT + Gemma + VAE 只构建一次）按 chunk 生成，并把不断增长的画面 + 实时音频
**流式** 推送到浏览器。生成过程中编辑 **live prompt** 文本框，会改写 *下一个* chunk 的
跨注意力条件 —— 于是画面内容会中途改变。这是安全的，因为文本只是跨注意力：它 **不属于**
缓存的自注意力历史或锚，所以改提示词 **不会** 重置画面、不会作废滑动窗口状态。

“Generation settings” 面板里的 **Stream strategy** 下拉框提供同样的四个 `--stream-strategy` 选项 ——
**按次生成可选，无需重启 app**（KV-cache wrapper 在每次 run 的 generator 内构建与 detach）。
便于在一个 session 里做策略间消融对比。CLI 上的 `--stream-strategy` 设下拉框默认值。

```bash
uv run python -m ltx_pipelines.app_gradio \
    --transformer-path       models/ltx-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
    --text-encoder-path      models/ltx-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
    --video-vae-path         models/ltx-2.5/vae/ltx-2.5-video-vae-conv-bf16.safetensors \
    --audio-vae-path         models/ltx-2.5/vae/ltx-2.5-audio-vae-bf16.safetensors \
    --stream-strategy kv_clean
```

打开打印的 URL，上传参考图（它成为锚的第 0 帧），设好初始提示词，点 **Generate**，然后在流式过程中编辑
live prompt 即可。`--host/--port/--share` 控制服务；`--device` / `--text-encoder-device` 可把
12B 文本编码器分到第二张 GPU。

注意事项：
- `fp8-scaled-mm` 用于**已量化**的 fp8 checkpoint（盘上是 `F8_E4M3` weight + `.weight_scale`）；
  `fp8-cast` 用于 bf16 checkpoint（加载时把 bf16 转 fp8）。
- **不要** 把 `--quantization fp8-scaled-mm` 与 `--offload cpu/disk` 同时用 —— 块流式只支持
  bf16 / `fp8-cast`。长驻会话会同时常驻 DiT + Gemma（12B，bf16）+ VAE，显存占用很高（约 A100/H100 级别）。
  若显存吃紧，用 `--offload cpu`。
- 蒸馏权重是为 3 步 *双向* 生成调优的；因果流式路径属于训练/测试不匹配——能跑，但用完整 dev transformer
  质量更好。若确实用蒸馏 checkpoint，请在 UI 里把步数调低（约 3–8）。

### 测试

无需 checkpoint 的 CPU 测试覆盖流式管线与全部四种策略：

```bash
uv run python packages/ltx-pipelines/tests/test_streaming_joint.py          # 纯张量 helper
uv run python packages/ltx-pipelines/tests/test_streaming_tiny_model.py     # 微型模型：4 策略 parity + finiteness
uv run python packages/ltx-pipelines/tests/test_streaming_interactive.py   # 交互：4 策略 parity + finiteness
```

`test_streaming_tiny_model.py` 断言四种策略单 chunk 逐位一致（无 history ⇒ 等价）+ 多 chunk finiteness；
`test_streaming_interactive.py` 对交互/Gradio 驱动做同样校验（外加 live-prompt 改写保证与 resolver 调用次数）。
生产环境信任 M2 前请在 GPU 上跑一遍。

### 相对论文的不足

1. **无训练（Stage 1–3）。** 双向权重被直接当作因果模型使用——最大的质量差距。
2. **chunk 1 双向 bootstrap。** 第一个 chunk 用全（非因果）注意力去噪，以匹配基础模型的训练、
   让流有一个高质量 ti2v 锚；论文的因果模型是因果地生成它的。后续 chunk 与论文一致为因果。
   （动机见上文"设计说明"。）
3. **无 Stage-3 蒸馏。** 生成用完整步数（默认 30，蒸馏 checkpoint 用 15），而非 3 步。论文 42 FPS / 540p
   headline 在没有蒸馏 + §2.3.2 基础设施栈的情况下不可达。
4. **无 §2.3.2 推理基础设施。** SageAttention / SpargeAttention / SLA、定制 W8A8 GEMM、kernel fusion、
   CUDA Graph、Ulysses 多卡并行均未实现。LTX-2 自带的 `fp8-cast` / `fp8-scaled-mm` 量化与 FlashAttention 仍可用。
5. **AV 跨模态因果掩码默认开启** —— 对双向基础模型是训练/测试不匹配，但优先论文忠实的因果性
   （`--no-causal-cross-attn` 可关闭）。
6. **无 CFG。** `--negative-prompt` 被编码但未使用（单次前向）。

### 本 fork 新增 / 修改的文件

- `packages/ltx-pipelines/src/ltx_pipelines/utils/streaming.py` —— KV cache + image_cond 联合流式驱动
  （`streaming_generate_joint_cached` / `streaming_generate_joint_image_cond`）+ 共享因果流式原语。
- `packages/ltx-pipelines/src/ltx_pipelines/utils/streaming_interactive.py` ——
  交互生成器驱动（`iter_streaming_chunks_joint_cached` / `_image_cond`，带逐 chunk context resolver + yield）。
- `packages/ltx-pipelines/src/ltx_pipelines/ti2vid_streaming.py` —— `TI2VidStreamingPipeline` + CLI（`--stream-strategy`）。
- `packages/ltx-pipelines/src/ltx_pipelines/interactive_session.py` —— 长驻会话 + `LivePromptEncoder`
  + 增量解码 + `stream_strategy` 分发。
- `packages/ltx-pipelines/src/ltx_pipelines/app_gradio.py` —— Gradio webui（Stream strategy 下拉框）。
- `packages/ltx-core/src/ltx_core/model/transformer/streaming_cache.py` ——
  `StreamingKVCache`（twin / clean / noisy_steps 策略；持久首块槽位——joint TI2V 的视频 cache
  提交 `[图 | chunk 1]` 锚，音频钉住首块；两者皆无 sink，A2V 保留带 sink 的默认）。
- `packages/ltx-core/src/ltx_core/model/transformer/streaming_model.py` ——
  `CausalStreamingModel`（视频 + 可选音频缓存；透传 strategy/step_idx）。
- `packages/ltx-core/src/ltx_core/model/transformer/attention.py` —— `stream_cache` 属性 + `_stream_cached_forward`
  （音视频皆无 sink 布局；对策略无感）。
- `packages/ltx-pipelines/tests/test_streaming_{joint,tiny_model,interactive}.py` ——
  纯张量 / 微型模型 / 交互测试，覆盖全部四种策略。

### 上游 LTX-2

其余部分（核心模型、其他管线、训练器）均为未改动的上游 LTX-2（v1.2.0 / LTX-2.5）。详见 `packages/`
下各包 README 与 <https://www.modelscope.cn/models/Lightricks/LTX-2.5>。

---

## References / 参考文献

- Vidu S1: *A Real-Time Interactive Video Generation Model*, arXiv:2607.03118 (2026).
- LTX-2 by Lightricks — <https://github.com/Lightricks/LTX-2>,
  <https://www.modelscope.cn/models/Lightricks/LTX-2.5>，
  <https://huggingface.co/Lightricks/LTX-2.5>。

## License

Inherits the upstream LTX-2 license (see `LICENSE`).
