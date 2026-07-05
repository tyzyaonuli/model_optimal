# minWM Action2V DMD Full Optimization Benchmark

- run_id: `20260705_214326_combined`
- started_at: `2026-07-05T21:53:19+08:00`
- cwd: `/root/autodl-tmp/workspace/minWM`
- prompts: `/root/autodl-tmp/workspace/minWM/Wan21/prompts/demos.txt`
- frames: `20`
- bench_prompts: `10`
- checkpoint: `/root/autodl-tmp/workspace/minWM/ckpts/Wan21/Action2V/dmd/model.pt`
- lighttae_checkpoint: `/root/autodl-tmp/workspace/lightx2v_ckpts/lighttaew2_1.pth`
- results_csv: `outputs/benchmark_results/20260705_214326_combined/results.csv`
- stage_csv: `outputs/benchmark_results/20260705_214326_combined/stage_profile.csv`

## Case Meaning

- `baseline_offload`: original Wan VAE decode, generator offloaded before VAE to avoid 24GB OOM.
- `wan_chunk`: original Wan VAE with temporal chunk decode, no generator offload.
- `wan_chunk_torchao`: `wan_chunk` plus TorchAO generator weight-only quantization.
- `paper_lighttae`: LightX2V/LightTAE Wan2.1 autoencoder decode inside minWM, no generator offload.
- `paper_lighttae_torchao`: LightTAE decode plus TorchAO generator quantization.

## Results

| case | status | seconds | speedup | gen_seconds | gen_speedup | diffusion_excl_vae | vae_speedup | pipeline_speedup | backend | lightx2v_parallel | torchao | peak_alloc_gb | min_free_gb |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: |
| baseline_offload | success | 353.259 | 1.0000 | 264.854 | 1.0000 | 186.772 | 1.0000 | 1.0000 | wan |  | none | 15.149 | 0.176 |
| paper_lighttae | success | 239.411 | 1.4755 | 148.543 | 1.7830 | 146.574 | 36.4469 | 1.7398 | lightx2v_tae | 1 | none | 14.653 | 0.296 |

## Stage Seconds

- baseline_offload: `{"pipeline_inference": 258.43035796284676, "vae_decode": 71.65885051339865, "write_video_submit": 6.423788741230965}`
- paper_lighttae: `{"pipeline_inference": 148.54019363969564, "vae_decode": 1.9661158174276352, "write_video_submit": 0.003091573715209961}`

## Notes

This script reproduces the LightTAE-style optimization inside minWM by replacing the final Wan VAE decode. It is not the full LightX2V pipeline and does not claim true FP8 kernels unless your TorchAO/LightX2V stack enables them.
