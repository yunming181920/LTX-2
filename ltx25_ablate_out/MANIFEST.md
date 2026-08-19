# LTX-2.5 streaming ablation — 8 runs

video: 512x768, 97 frames (4 chunks, ~4.04s), 8 steps (distilled), bf16, seed=42, image CRF=18

stage1 (chunks 0-1): A cat lowers its head and licks its front paw ...
stage2 (chunks 2-3): looks directly at the camera ... (no subject; interaction test)

image_cond rotates PIXEL frame (decode->CRF+VAE re-encode) as next sink.

file                                        total_s  peakGB  ratio   per_chunk_s
------------------------------------------------------------------------------------------
01_kv-twin_xattn-off.mp4                      28.06    46.6   0.14   [7.551, 4.586, 10.614, 2.973]
02_kv-twin_xattn-on.mp4                       27.76    49.5   0.15   [10.831, 3.617, 7.947, 3.651]
03_kv-clean_xattn-off.mp4                     27.00    46.6   0.15   [7.419, 7.481, 7.289, 2.994]
04_kv-clean_xattn-on.mp4                      23.36    49.5   0.17   [7.012, 4.373, 6.804, 3.848]
05_kv-noisy-steps_xattn-off.mp4               28.34    66.1   0.14   [10.647, 7.747, 5.947, 2.709]
06_kv-noisy-steps_xattn-on.mp4                27.89    78.9   0.14   [7.922, 7.458, 7.85, 2.826]
07_image-cond_xattn-off.mp4                   31.02    42.6   0.13   [11.769, 4.593, 6.476, 4.952]
08_image-cond_xattn-on.mp4                    23.09    42.6   0.17   [4.972, 7.161, 4.375, 4.225]

ratio = video_s/gen_s (>1 => realtime; none reach it here)
videos/ + embeddings.pt + timings.json
