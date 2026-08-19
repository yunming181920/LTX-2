# LTX-2.3 streaming ablation — 8 runs

video: 512x768, 97 frames (4 chunks, ~4.04s), 8 steps, bf16, seed=42

stage1 (chunks 0-1, ~2s): A cat lowers its head and licks its front paw ...
stage2 (chunks 2-3, ~2s): looks directly at the camera ... (no subject; interaction test)

image_cond now rotates the PIXEL frame (decode last latent -> CRF + VAE re-encode) as next sink,
not the latent token. kv_* unchanged (KV history + anchor).

file                                        total_s  peakGB  ratio   per_chunk_s
------------------------------------------------------------------------------------------
01_kv-twin_xattn-off.mp4                      28.95    46.6   0.14   [7.477, 8.13, 7.913, 2.989]
02_kv-twin_xattn-on.mp4                       25.32    49.5   0.16   [8.026, 3.875, 7.908, 3.636]
03_kv-clean_xattn-off.mp4                     28.42    46.6   0.14   [7.539, 4.889, 10.714, 3.009]
04_kv-clean_xattn-on.mp4                      24.69    49.5   0.16   [7.343, 4.168, 7.857, 3.537]
05_kv-noisy-steps_xattn-off.mp4               28.33    66.1   0.14   [8.289, 7.79, 7.756, 3.14]
06_kv-noisy-steps_xattn-on.mp4                28.12    78.9   0.14   [8.0, 7.691, 7.894, 2.841]
07_image-cond_xattn-off.mp4                   27.62    42.6   0.15   [7.735, 7.333, 4.568, 6.633]
08_image-cond_xattn-on.mp4                    23.86    42.6   0.17   [7.177, 4.3, 4.33, 4.074]

ratio = video_s / gen_s (>1 => faster than realtime; none reach realtime here)
videos in ./videos/ | embeddings.pt = pre-encoded stage1/stage2 text | timings.json = raw metrics
