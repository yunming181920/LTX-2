# LTX-2 · Vidu S1 §2.3 Streaming Causal A2V Reproduction

> A **training-free** reproduction of the streaming inference described in
> **Vidu S1** (arXiv:[2607.03118](https://arxiv.org/abs/2607.03118), §2.3) on top of
> the pretrained **bidirectional** LTX-2 checkpoint, which is used as-is as the
> “causal model”. This is a fork of [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2);
> everything below the *original LTX-2* line is unchanged. The additions live in
> `packages/ltx-pipelines` (streaming driver + pipeline) and `packages/ltx-core`
> (KV-cache attention surgery).

---

## English

### What this is

Vidu S1 is a real-time interactive audio→video model whose §2.3 describes a
**streaming, autoregressive, causal** inference scheme: a sliding window with a
persistent reference context (“sink”), cached historical states, **RoPE
Repositioning**, and **TwinCache**. Vidu S1 trains a dedicated causal teacher
(Stage 2) and distills it (Stage 3) to reach 3-step / 42 FPS.

This repo reproduces **only the §2.3.1 streaming inference, training-free** —
i.e. it reuses LTX-2’s pretrained bidirectional weights without any fine-tuning.
Per the chosen scope:

- **Inference only** — no training (Stages 1–3 are skipped).
- **Audio is a frozen control signal** — only video is generated.
- **Full-fidelity streaming machinery** — causal masks + sliding window + sink +
  TwinCache + KV cache + RoPE repositioning.

### Which paper features are implemented (§2.3.1)

| Vidu S1 §2.3.1 component | Status | Where |
|---|---|---|
| Sliding-window decoding (O(window) per step) | ✅ | `streaming.generate` / `streaming_generate_cached` |
| Persistent reference context = first-frame latent **+ first generated state** | ✅ | `sink_tokens` + `first_ref` (permanent slot, never evicted) |
| Cached historical states within the window | ✅ | `rolling` deque (M1) / `StreamingKVCache` FIFO ring (M2) |
| **RoPE Repositioning** (cache pre-RoPE K/V, reapply at updated window positions) | ✅ | M1: fresh `VideoLatentTools` window-relative positions · M2: `attention.py::_stream_cached_forward` caches pre-RoPE K and reapplies `window_pe` |
| **TwinCache** (noisy cache from a mid step + clean cache from the final step; stage-aware) | ✅ | `ChunkSnapshots` + `StreamingTwinDenoiser._inject_history` (M1) / cache `stash`+`commit` (M2); intermediate steps read noisy, final step reads clean |
| Causal attention mask on the temporal axis | ✅ | `block_causal_attention_mask` (frame-major: bidirectional within a frame, causal across frames) |
| Causal mask on AV **cross**-attention (video↔audio) | ✅ (default ON) | `cross_causal_attention_mask` built from LTX-2’s shared seconds-axis cross-attn RoPE positions; applied to both `audio_to_video_attn` and `video_to_audio_attn` (`--no-causal-cross-attn` to disable) |
| Per-block KV cache (video self-attention) | ✅ (M2) | `StreamingKVCache` × 48 blocks, permanent first-chunk slot + TwinCache ring |
| Frozen audio control, window-aligned | ✅ | `assemble_audio_slice` + window-relative audio positions |

The autoregressive unit is **one latent video frame** (= 8 pixel frames =
`H_lat × W_lat` tokens); `--chunk-frames` generates a few latent frames per AR
step. The full latent is decoded once at the end (seamless causal-VAE decode).

### Two milestones

- **M1 — correct-but-slow, no core changes** (`streaming.py::streaming_generate`):
  block-causal mask routed through the existing `Modality.attention_mask`
  channel, latent-level TwinCache (history snapshots injected per step,
  sink+history frozen via `denoise_mask=0` so the Euler velocity stays 0).
- **M2 — KV cache + RoPE repositioning** (`streaming.py::streaming_generate_cached`
  + `streaming_cache.py` + `streaming_model.py` + `attention.py`): per-block
  video-self-attn KV cache (48 caches), history K/V spliced from the cache,
  RoPE reapplied with the full-window `window_pe`. Production pipelines are
  untouched (`attn1.stream_cache` stays `None` → byte-identical standard path).

### Known limitations / gaps vs. the paper

These are **intentionally out of scope** or not yet done — read this before
comparing outputs to the Vidu S1 paper numbers.

1. **No training (Stages 1–3).** The bidirectional checkpoint is used as-is as
   the causal model. Vidu S1 trains a causal teacher (Teacher Forcing +
   Diffusion Forcing) and distills it (DMD + PCM). This is the **largest
   quality gap** — the base model was never adapted to the causal/streaming
   regime, so fidelity will not match the paper.
2. **Audio generation is frozen.** Only video is generated; audio is the input
   control. The paper does joint video-audio generation.
3. **AV cross-attention causal mask is ON by default.** The causal mask always
   applies to video self-attention; the video↔audio cross-attention is also masked
   time-causally by default (built from LTX-2’s shared seconds-axis RoPE
   positions; `--no-causal-cross-attn` to disable). The base model is
   bidirectionally trained, so this is a train/test mismatch — but this is a
   **conceptual reproduction** (not tuned for runtime quality), so paper-faithful
   causality is preferred. Tune the strictness with
   `--cross-attn-lookahead-seconds` (0 = strict causal).
4. **No Stage-3 distillation.** Generation uses the full step count (default
   30), not 3 steps. The paper’s **42 FPS / 540p** headline is **not reachable**
   without distillation + the §2.3.2 infra stack.
5. **No §2.3.2 inference infrastructure.** SageAttention / SpargeAttention /
   SLA, custom W8A8 GEMM, kernel fusion, CUDA Graph, and Ulysses multi-GPU
   parallelism are all **not implemented**. LTX-2’s own `fp8-cast` /
   `fp8-scaled-mm` quantization and FlashAttention still work.
6. **No online interactive serving (TurboServe).** “Streaming” here means
   per-step activation memory is O(window) and independent of total length —
   but the full latent is still decoded once at the end and written to one file.
   It is **batch-offline with internal streaming**, not online: no incremental
   frame output to a client, no mid-generation prompt/audio injection.
7. **No CFG.** `--negative-prompt` is encoded but unused (single forward pass,
   SimpleDenoiser logic).
8. **Runtime-unverified.** The code compiles (`py_compile`) and is statically
   reviewed, but **has not been run end-to-end** — the dev environment has no
   torch/GPU. **M2 in particular must pass the parity test before trusting it.**

### Quick start

Install (uv monorepo) and download models as in the upstream LTX-2 README
([Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) +
`google/gemma-3-12b-it-qat-q4_0-unquantized`).

```bash
uv run python -m ltx_pipelines.a2vid_streaming \
    --checkpoint-path models/ltx-2.3/ltx-2.3-22b-dev.safetensors \
    --gemma-root models/gemma-3-12b \
    --prompt "A person talking calmly to the camera." \
    --image ref.jpg 0 1.0 \
    --audio-path ref.wav \
    --output-path out.mp4 \
    --num-frames 33 --frame-rate 30 --height 512 --width 768 \
    --num-inference-steps 30 --window-chunks 4 --chunk-frames 1
```

Add `--use-kv-cache` for the M2 (KV-cache + RoPE-repositioning) path — faster,
but results differ slightly from the default full-recompute path (run the parity
test first). On low-VRAM GPUs add `--quantization fp8-cast --offload cpu`.

#### Streaming-specific CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--window-chunks` | 4 | Sliding-window rolling-history size in AR chunks (TwinCache FIFO cap; sink + first chunk are persistent and not counted) |
| `--chunk-frames` | 1 | Latent video frames generated per AR step (1 = finest streaming granularity) |
| `--audio-lookahead` | `ceil(200/fps)` | Extra audio latent frames visible ahead of the current chunk’s end |
| `--use-kv-cache` | off | Use the M2 KV-cache + RoPE-repositioning path |
| `--causal-cross-attn` / `--no-causal-cross-attn` | on | Time-causal mask on video↔audio cross-attention (default ON, paper-faithful; train/test mismatch on the bidirectional base model — this is a conceptual reproduction) |
| `--cross-attn-lookahead-seconds` | 0.0 | Seconds of future audio a video frame may attend under causal cross-attn (0 = strict causal) |
| `--audio-path` / `--audio-start-time` / `--audio-max-duration` | — | Frozen audio control signal |

Standard LTX-2 flags (`--seed`, `--quantization`, `--offload`, `--compile`,
`--enhance-prompt`, `--lora`, `--prompt`, `--negative-prompt`, `--image PATH FRAME_IDX STRENGTH [CRF]`,
`--num-frames`, `--frame-rate`, `--height`, `--width`, `--num-inference-steps`)
behave as in upstream pipelines.

### Validate M2 against M1 (parity test)

Loads the full checkpoint + Gemma and needs a reference image + audio. Run it on
a GPU **before trusting M2** — Phase 1 asserts strict single-chunk parity (M2 ≡
M1 modulo bf16); Phase 2 is a multi-chunk smoke test (the two paths diverge by
design once history exists).

```bash
uv run python packages/ltx-pipelines/tests/test_streaming_kv_cache_parity.py \
    --checkpoint-path models/ltx-2.3/ltx-2.3-22b-dev.safetensors \
    --gemma-root models/gemma-3-12b --image ref.jpg --audio-path ref.wav
```

### Files added / changed by this fork

- `packages/ltx-pipelines/src/ltx_pipelines/utils/streaming.py` — M1 + M2 streaming drivers, `cross_causal_attention_mask`
- `packages/ltx-pipelines/src/ltx_pipelines/a2vid_streaming.py` — `A2VidStreamingPipeline` + CLI
- `packages/ltx-core/src/ltx_core/model/transformer/streaming_cache.py` — `StreamingKVCache`
- `packages/ltx-core/src/ltx_core/model/transformer/streaming_model.py` — `CausalStreamingModel`
- `packages/ltx-core/src/ltx_core/model/transformer/attention.py` — `stream_cache` attr + `_stream_cached_forward`
- `packages/ltx-core/src/ltx_core/model/transformer/transformer.py` — `mask=` on `audio_to_video_attn`/`video_to_audio_attn` (+ shape guard)
- `packages/ltx-core/src/ltx_core/model/transformer/transformer_args.py`, `modality.py`, `ltx_core/types.py`, `ltx_pipelines/utils/helpers.py` — `cross_attention_mask` field plumbed `LatentState → Modality → TransformerArgs → block`
- `packages/ltx-core/src/ltx_core/model/transformer/__init__.py` — new exports
- `packages/ltx-pipelines/tests/test_streaming_kv_cache_parity.py` — parity/smoke test
- `packages/ltx-pipelines/tests/test_cross_causal_mask.py` — unit test for `cross_causal_attention_mask`

### Upstream LTX-2

Everything else (core model, other pipelines, trainer) is unchanged upstream
LTX-2. See the per-package READMEs under `packages/` and
<https://huggingface.co/Lightricks/LTX-2.3>.

---

## 中文

### 这是什么

Vidu S1（arXiv:2607.03118）是一个实时交互式音→视频模型，其 §2.3 描述了一套
**流式、自回归、因果** 的推理方案：滑动窗口 + 持久参考上下文（“sink”）+ 缓存历史状态 +
**RoPE 重定位（RoPE Repositioning）** + **TwinCache**。Vidu S1 训练了专门的因果教师
（Stage 2）并蒸馏（Stage 3）以实现 3 步 / 42 FPS。

本仓库 **仅复现 §2.3.1 的流式推理，且为 training-free** —— 即直接复用 LTX-2 预训练的
双向权重，不做任何微调。按既定范围：

- **仅推理** —— 不训练（跳过 Stage 1–3）。
- **音频为冻结的控制信号** —— 只生成视频。
- **完整保真的流式机制** —— 因果掩码 + 滑动窗口 + sink + TwinCache + KV cache + RoPE 重定位。

### 实现了论文中的哪些功能（§2.3.1）

| Vidu S1 §2.3.1 组件 | 状态 | 位置 |
|---|---|---|
| 滑动窗口解码（每步 O(window)） | ✅ | `streaming_generate` / `streaming_generate_cached` |
| 持久参考上下文 = 首帧 latent **+ 第一个生成状态** | ✅ | `sink_tokens` + `first_ref`（永久槽位，永不淘汰） |
| 窗口内缓存历史状态 | ✅ | `rolling` deque（M1）/ `StreamingKVCache` FIFO 环（M2） |
| **RoPE 重定位**（缓存 RoPE 前的 K/V，按更新后的窗口位置重新施加） | ✅ | M1：用 fresh `VideoLatentTools` 重建窗口相对位置 · M2：`attention.py::_stream_cached_forward` 缓存 pre-RoPE K 并用 `window_pe` 重新施加 |
| **TwinCache**（mid 步的 noisy 缓存 + 末步的 clean 缓存；阶段感知） | ✅ | `ChunkSnapshots` + `StreamingTwinDenoiser._inject_history`（M1）/ cache `stash`+`commit`（M2）；中间步读 noisy、末步读 clean |
| 时间轴因果注意力掩码 | ✅ | `block_causal_attention_mask`（帧主序：帧内双向、跨帧因果） |
| AV **跨**注意力因果掩码（video↔audio） | ✅（默认开启） | `cross_causal_attention_mask`，基于 LTX-2 跨注意力 RoPE 的共享秒轴位置构建；同时作用于 `audio_to_video_attn` 与 `video_to_audio_attn`（`--no-causal-cross-attn` 可关闭） |
| 每块 KV cache（视频自注意力） | ✅（M2） | `StreamingKVCache` × 48 块，永久首 chunk 槽 + TwinCache 环 |
| 冻结音频控制，窗口对齐 | ✅ | `assemble_audio_slice` + 窗口相对音频位置 |

自回归单元为 **一个 latent 视频帧**（= 8 个像素帧 = `H_lat × W_lat` 个 token）；
`--chunk-frames` 可每步生成多个 latent 帧。整段 latent 在最后一次性解码（因果 VAE 无缝解码）。

### 两个里程碑

- **M1 —— 正确但慢，无 core 改动**（`streaming.py::streaming_generate`）：通过既有
  `Modality.attention_mask` 通道施加块因果掩码，latent 级 TwinCache（每步注入历史快照，
  sink+历史通过 `denoise_mask=0` 冻结，使 Euler 速度为 0）。
- **M2 —— KV cache + RoPE 重定位**（`streaming.py::streaming_generate_cached` +
  `streaming_cache.py` + `streaming_model.py` + `attention.py`）：每块视频自注意力一个 KV cache
  （共 48 个），历史 K/V 从缓存拼接，用全窗 `window_pe` 重新施加 RoPE。生产管线不受影响
  （`attn1.stream_cache` 保持 `None` → 与标准路径逐字节一致）。

### 相对论文的不足 / 已知差距

在把输出和 Vidu S1 论文数字对比前，请先阅读以下内容——这些是 **有意不做** 或 **尚未完成** 的。

1. **无训练（Stage 1–3）。** 双向权重被直接当作因果模型使用。Vidu S1 训练了因果教师
   （Teacher Forcing + Diffusion Forcing）并蒸馏（DMD + PCM）。这是 **最大的质量差距** ——
   基础模型从未适配因果/流式 regime，保真度无法匹敌论文。
2. **音频生成被冻结。** 只生成视频，音频是输入控制信号。论文是音视频联合生成。
3. **AV 跨模态因果掩码默认开启。** 因果掩码始终作用于视频自注意力；video↔audio 跨注意力
   也默认施加时间因果掩码（基于 LTX-2 跨注意力 RoPE 的共享秒轴位置构建；`--no-causal-cross-attn`
   可关闭）。基础模型是双向训练的，这属于训练/测试不匹配——但本仓库是**概念复现**（未针对运行时
   质量调优），故优先采用论文忠实的因果性。用 `--cross-attn-lookahead-seconds` 调节严格度
   （0 = 严格因果）。
4. **无 Stage 3 蒸馏。** 生成使用完整步数（默认 30），而非 3 步。论文的 **42 FPS / 540p**
   headline 在没有蒸馏 + §2.3.2 基础设施栈的情况下 **不可达**。
5. **无 §2.3.2 推理基础设施。** SageAttention / SpargeAttention / SLA、定制 W8A8 GEMM、
   kernel fusion、CUDA Graph、Ulysses 多卡并行 **均未实现**。LTX-2 自带的 `fp8-cast` /
   `fp8-scaled-mm` 量化与 FlashAttention 仍可用。
6. **无在线交互式 serving（TurboServe）。** 此处“流式”指每步激活内存为 O(window) 且与总长度无关
   —— 但整段 latent 仍在最后一次性解码并写成一个文件。它是 **内部流式的离线批处理**，并非 online：
   无向客户端的增量帧输出，无生成中途注入 prompt/音频。
7. **无 CFG。** `--negative-prompt` 被编码但未使用（单次前向，SimpleDenoiser 逻辑）。
8. **未经运行时验证。** 代码通过 `py_compile` 并经静态审查，但 **未端到端运行过** —— 开发环境无
   torch/GPU。**尤其是 M2，必须先通过 parity 测试才能信任。**

### 快速开始

按上游 LTX-2 README 用 uv 安装并下载模型（[Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3)
+ `google/gemma-3-12b-it-qat-q4_0-unquantized`）。

```bash
uv run python -m ltx_pipelines.a2vid_streaming \
    --checkpoint-path models/ltx-2.3/ltx-2.3-22b-dev.safetensors \
    --gemma-root models/gemma-3-12b \
    --prompt "一个人平静地对着镜头说话。" \
    --image ref.jpg 0 1.0 \
    --audio-path ref.wav \
    --output-path out.mp4 \
    --num-frames 33 --frame-rate 30 --height 512 --width 768 \
    --num-inference-steps 30 --window-chunks 4 --chunk-frames 1
```

加 `--use-kv-cache` 走 M2（KV-cache + RoPE 重定位）路径——更快，但结果与默认的全量重算路径
略有差异（请先跑 parity 测试）。显存不足时加 `--quantization fp8-cast --offload cpu`。

#### 流式专用 CLI 参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--window-chunks` | 4 | 滑动窗口滚动历史大小（以 AR chunk 计；TwinCache FIFO 上限；sink + 首 chunk 为持久项，不计入） |
| `--chunk-frames` | 1 | 每个 AR 步生成的 latent 视频帧数（1 = 最细流式粒度） |
| `--audio-lookahead` | `ceil(200/fps)` | 在当前 chunk 末尾之后可见的额外音频 latent 帧数 |
| `--use-kv-cache` | 关 | 使用 M2 KV-cache + RoPE 重定位路径 |
| `--causal-cross-attn` / `--no-causal-cross-attn` | 开 | 对 video↔audio 跨注意力施加时间因果掩码（默认开启，论文忠实；对双向基础模型是训练/测试不匹配——本仓库为概念复现） |
| `--cross-attn-lookahead-seconds` | 0.0 | 因果跨注意力下视频帧可看到的未来音频秒数（0 = 严格因果） |
| `--audio-path` / `--audio-start-time` / `--audio-max-duration` | — | 冻结音频控制信号 |

标准 LTX-2 参数（`--seed`、`--quantization`、`--offload`、`--compile`、`--enhance-prompt`、
`--lora`、`--prompt`、`--negative-prompt`、`--image PATH FRAME_IDX STRENGTH [CRF]`、
`--num-frames`、`--frame-rate`、`--height`、`--width`、`--num-inference-steps`）与上游管线一致。

### 校验 M2 与 M1 的一致性（parity 测试）

需加载完整 checkpoint + Gemma，并需要参考图 + 音频。**在信任 M2 前** 请在 GPU 上运行——
阶段 1 断言单 chunk 严格一致（M2 ≡ M1，误差在 bf16 范围内）；阶段 2 为多 chunk 冒烟测试
（一旦存在历史，两条路径按设计会有差异）。

```bash
uv run python packages/ltx-pipelines/tests/test_streaming_kv_cache_parity.py \
    --checkpoint-path models/ltx-2.3/ltx-2.3-22b-dev.safetensors \
    --gemma-root models/gemma-3-12b --image ref.jpg --audio-path ref.wav
```

### 本 fork 新增 / 修改的文件

- `packages/ltx-pipelines/src/ltx_pipelines/utils/streaming.py` —— M1 + M2 流式驱动
- `packages/ltx-pipelines/src/ltx_pipelines/a2vid_streaming.py` —— `A2VidStreamingPipeline` + CLI
- `packages/ltx-core/src/ltx_core/model/transformer/streaming_cache.py` —— `StreamingKVCache`
- `packages/ltx-core/src/ltx_core/model/transformer/streaming_model.py` —— `CausalStreamingModel`
- `packages/ltx-core/src/ltx_core/model/transformer/attention.py` —— `stream_cache` 属性 + `_stream_cached_forward`
- `packages/ltx-core/src/ltx_core/model/transformer/__init__.py` —— 新增导出
- `packages/ltx-pipelines/tests/test_streaming_kv_cache_parity.py` —— parity/冒烟测试

### 上游 LTX-2

其余部分（核心模型、其他管线、训练器）均为未改动的上游 LTX-2。详见 `packages/` 下各包 README
与 <https://huggingface.co/Lightricks/LTX-2.3>。

---

## References / 参考文献

- Vidu S1: *A Real-Time Interactive Video Generation Model*, arXiv:2607.03118 (2026).
- LTX-2 by Lightricks — <https://github.com/Lightricks/LTX-2>, <https://huggingface.co/Lightricks/LTX-2.3>.

## License

Inherits the upstream LTX-2 license (see `LICENSE`).
