# minWM Action2V DMD Full Optimization Benchmark

- run_id: `20260705_213020_wan`
- started_at: `2026-07-05T21:41:42+08:00`
- cwd: `/root/autodl-tmp/workspace/minWM`
- prompts: `/root/autodl-tmp/workspace/minWM/Wan21/prompts/demos.txt`
- frames: `20`
- bench_prompts: `10`
- checkpoint: `/root/autodl-tmp/workspace/minWM/ckpts/Wan21/Action2V/dmd/model.pt`
- lighttae_checkpoint: `/root/autodl-tmp/workspace/lightx2v_ckpts/lighttaew2_1.pth`
- results_csv: `outputs/benchmark_results/20260705_213020_wan/results.csv`
- stage_csv: `outputs/benchmark_results/20260705_213020_wan/stage_profile.csv`

## Case Meaning

- `baseline_offload`: original Wan VAE decode, generator offloaded before VAE to avoid 24GB OOM.
- `wan_chunk`: original Wan VAE with temporal chunk decode, no generator offload.
- `wan_chunk_torchao`: `wan_chunk` plus TorchAO generator weight-only quantization.
- `paper_lighttae`: LightX2V/LightTAE Wan2.1 autoencoder decode inside minWM, no generator offload.
- `paper_lighttae_torchao`: LightTAE decode plus TorchAO generator quantization.

## Results

| case | status | seconds | speedup | gen_seconds | gen_speedup | diffusion_excl_vae | vae_speedup | pipeline_speedup | backend | lightx2v_parallel | torchao | peak_alloc_gb | min_free_gb |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: |
| baseline_offload | success | 360.384 | 1.0000 | 268.312 | 1.0000 | 189.440 | 1.0000 | 1.0000 | wan |  | none | 15.149 | 0.176 |
| wan_chunk | success | 322.229 | 1.1184 | 231.634 | 1.1583 | 176.368 | 1.3117 | 1.1308 | wan |  | none | 17.601 | 6.329 |

## Stage Seconds

- baseline_offload: `{"pipeline_inference": 261.9270913079381, "vae_decode": 72.48709715902805, "write_video_submit": 6.385094456374645}`
- wan_chunk: `{"pipeline_inference": 231.62776447832584, "vae_decode": 55.25985224545002, "write_video_submit": 0.006609894335269928}`

## Notes

This script reproduces the LightTAE-style optimization inside minWM by replacing the final Wan VAE decode. It is not the full LightX2V pipeline and does not claim true FP8 kernels unless your TorchAO/LightX2V stack enables them.
