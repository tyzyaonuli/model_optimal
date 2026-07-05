# minWM Action2V DMD Full Optimization Benchmark

- run_id: `codex_lighttae_fast3_223045`
- started_at: `2026-07-05T22:34:36+08:00`
- cwd: `/root/autodl-tmp/workspace/minWM`
- prompts: `/root/autodl-tmp/workspace/minWM/Wan21/prompts/demos.txt`
- frames: `20`
- bench_prompts: `10`
- checkpoint: `/root/autodl-tmp/workspace/minWM/ckpts/Wan21/Action2V/dmd/model.pt`
- lighttae_checkpoint: `/root/autodl-tmp/workspace/lightx2v_ckpts/lighttaew2_1.pth`
- results_csv: `outputs/benchmark_results/codex_lighttae_fast3_223045/results.csv`
- stage_csv: `outputs/benchmark_results/codex_lighttae_fast3_223045/stage_profile.csv`

## Case Meaning

- `baseline_offload`: original Wan VAE decode, generator offloaded before VAE to avoid 24GB OOM.
- `wan_chunk`: original Wan VAE with temporal chunk decode, no generator offload.
- `wan_chunk_torchao`: `wan_chunk` plus TorchAO generator weight-only quantization.
- `paper_lighttae`: LightX2V/LightTAE Wan2.1 autoencoder decode inside minWM, no generator offload.
- `paper_lighttae_torchao`: LightTAE decode plus TorchAO generator quantization.

## Results

| case | status | seconds | speedup | gen_seconds | gen_speedup | diffusion_excl_vae | vae_speedup | pipeline_speedup | backend | lightx2v_parallel | torchao | peak_alloc_gb | min_free_gb |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: |
| paper_lighttae_fast3 | success | 231.119 |  | 123.807 |  | 121.796 |  |  | lightx2v_tae | 1 | none | 14.653 | 0.296 |

## Stage Seconds

- paper_lighttae_fast3: `{"pipeline_inference": 123.80394230037928, "vae_decode": 2.008101500570774, "write_video_submit": 0.0031370073556900024}`

## Notes

This script reproduces the LightTAE-style optimization inside minWM by replacing the final Wan VAE decode. It is not the full LightX2V pipeline and does not claim true FP8 kernels unless your TorchAO/LightX2V stack enables them.
