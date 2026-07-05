# minWM Action2V DMD 单卡 4090 优化复现

本仓库用于在单张 RTX 4090 24GB 上运行和优化 `MIN-Lab/minWM` 的
`Wan21/Action2V/dmd` 推理。代码以 overlay 形式工作：不下载模型、不改权重，只在运行前给
minWM 推理代码打补丁，并记录速度和显存。

当前结果：

| 版本 | 核心思路 | 生成阶段加速 |
| --- | --- | ---: |
| `wan_chunk` | 原 Wan VAE 按时间维切块 decode，解决 OOM | 1.1583x |
| `paper_lighttae` | 用 LightTAEW 2.1 替换原 Wan VAE decode | 1.7830x |
| `paper_lighttae_fast3` | LightTAEW 2.1 + DMD 采样从 4 step 改为 3 step | 2.1393x |

`paper_lighttae_fast3` 达到了“生成阶段至少 2x”的目标，但它减少了 DMD denoise step，属于速度优先版本，需要人工检查生成质量。

## 1. 依赖仓库和目录

推荐服务器目录：

```text
/root/autodl-tmp/workspace/
├── minWM/                  # MIN-Lab/minWM 仓库
├── LightX2V/               # ModelTC/LightX2V 仓库，只需要源码
├── minwm_overlay_docs/     # 本仓库代码
└── lightx2v_ckpts/         # LightTAE / LightVAE 权重
```

需要两个外部仓库：

```bash
cd /root/autodl-tmp/workspace

git clone https://github.com/MIN-Lab/minWM.git
git clone https://github.com/ModelTC/LightX2V.git
```

LightTAE 路径只依赖 LightX2V 源码文件，不要求完整安装 LightX2V 包。

## 2. 环境配置

建议使用独立 venv：

```bash
cd /root/autodl-tmp
python3 -m venv venv
source /root/autodl-tmp/venv/bin/activate

pip install -U pip setuptools wheel
cd /root/autodl-tmp/workspace/minWM
pip install -r requirements.txt
pip install huggingface_hub hf_transfer safetensors imageio imageio-ffmpeg av
```

FlashAttention 需要和 PyTorch / CUDA ABI 匹配。如果遇到 undefined symbol，重新本机编译：

```bash
source /root/autodl-tmp/venv/bin/activate

MAX_JOBS=4 \
FLASH_ATTENTION_FORCE_BUILD=TRUE \
pip install --force-reinstall --no-cache-dir --no-build-isolation \
  flash-attn==2.7.4.post1
```

TorchAO 只用于实验，不是推荐主路径：

```bash
pip install torchao
```

## 3. 模型下载

### minWM DMD 权重

```bash
cd /root/autodl-tmp/workspace/minWM
mkdir -p ckpts

HF_ENDPOINT=https://hf-mirror.com \
HF_HUB_DISABLE_XET=1 \
huggingface-cli download MIN-Lab/minWM \
  --include "Wan21/Action2V/dmd/*" \
  --local-dir ckpts
```

推理时使用：

```text
CHECKPOINT_PATH=/root/autodl-tmp/workspace/minWM/ckpts/Wan21/Action2V/dmd/model.pt
```

### LightTAEW 2.1 权重

```bash
mkdir -p /root/autodl-tmp/workspace/lightx2v_ckpts

HF_ENDPOINT=https://hf-mirror.com \
HF_HUB_DISABLE_XET=1 \
huggingface-cli download lightx2v/Autoencoders \
  --include "lighttaew2_1.pth" "lighttaew2_1.safetensors" \
            "lightvaew2_1.pth" "lightvaew2_1.safetensors" \
            "Wan2.1_VAE.pth" "Wan2.1_VAE.safetensors" \
  --local-dir /root/autodl-tmp/workspace/lightx2v_ckpts
```

推荐使用：

```text
MINWM_LIGHTX2V_VAE_PATH=/root/autodl-tmp/workspace/lightx2v_ckpts/lighttaew2_1.pth
```

## 4. 文件说明

| 文件 | 作用 |
| --- | --- |
| `run_minwm_dmd_4090.py` | 单次推理入口，先 patch minWM，再用单卡 `torchrun` 启动推理。 |
| `minwm_4090_patch.py` | 主 patch 脚本，修改 `wan_inference.py`、`causal_inference.py`、`wan_wrapper.py`。 |
| `llv2_minwm_runtime.py` | 计时、显存记录、异步视频写入、Wan VAE temporal chunk decode。 |
| `lightx2v_minwm_runtime.py` | LightTAE 适配器，把 minWM latent 转给 LightX2V tiny VAE，并输出 `[B,T,3,H,W]`。 |
| `lightx2v_standalone_loader.py` | 只加载 LightX2V 中需要的 `tae.py` 和 `wan/vae_tiny.py`。 |
| `minwm_torchao_runtime.py` | TorchAO generator 量化实验 helper。 |
| `bench_minwm_common.py` | benchmark 公共逻辑，负责启动子进程和汇总结果。 |
| `bench_minwm_dmd_4090_all.py` | 所有 benchmark case 的调度器。 |
| `bench_minwm_dmd_4090_combined.py` | 推荐主入口：baseline、LightTAE、fast3 对比。 |
| `bench_minwm_dmd_4090.py` | 旧 Wan chunk 对比入口。 |
| `bench_minwm_dmd_4090_wan_experimental.py` | Wan chunk 和 TorchAO 实验入口。 |
| `bench_lightx2v_autoencoder.py` | 单独测 LightTAE autoencoder，不经过 minWM DMD pipeline。 |

结果目录：

| 目录 | 内容 |
| --- | --- |
| `20260705_213020_wan/` | `baseline_offload` vs `wan_chunk` |
| `20260705_214326_combined/` | `baseline_offload` vs `paper_lighttae` |
| `20260705_223045_fast3/` | `paper_lighttae_fast3` 达到生成阶段 2x 的结果 |

## 5. 推荐 benchmark 命令

### baseline + LightTAE + fast3 一起对比

```bash
cd /root/autodl-tmp/workspace/minWM

LIGHTX2V_REPO=/root/autodl-tmp/workspace/LightX2V \
DATA_PATH=/root/autodl-tmp/workspace/minWM/Wan21/prompts/demos.txt \
CHECKPOINT_PATH=/root/autodl-tmp/workspace/minWM/ckpts/Wan21/Action2V/dmd/model.pt \
BENCH_PROMPTS=10 \
NUM_OUTPUT_FRAMES=20 \
BASELINE_OFFLOAD_GENERATOR_BEFORE_VAE=1 \
MINWM_LIGHTX2V_VAE_PATH=/root/autodl-tmp/workspace/lightx2v_ckpts/lighttaew2_1.pth \
RUN_BASELINE=1 \
RUN_PAPER_LIGHTTAE=1 \
RUN_PAPER_LIGHTTAE_FAST3=1 \
python /root/autodl-tmp/workspace/minwm_overlay_docs/bench_minwm_dmd_4090_combined.py
```

### 只跑 fast3

```bash
cd /root/autodl-tmp/workspace/minWM

LIGHTX2V_REPO=/root/autodl-tmp/workspace/LightX2V \
DATA_PATH=/root/autodl-tmp/workspace/minWM/Wan21/prompts/demos.txt \
CHECKPOINT_PATH=/root/autodl-tmp/workspace/minWM/ckpts/Wan21/Action2V/dmd/model.pt \
BENCH_PROMPTS=10 \
NUM_OUTPUT_FRAMES=20 \
BASELINE_OFFLOAD_GENERATOR_BEFORE_VAE=1 \
MINWM_LIGHTX2V_VAE_PATH=/root/autodl-tmp/workspace/lightx2v_ckpts/lighttaew2_1.pth \
RUN_BASELINE=0 \
RUN_PAPER_LIGHTTAE=0 \
RUN_PAPER_LIGHTTAE_FAST3=1 \
python /root/autodl-tmp/workspace/minwm_overlay_docs/bench_minwm_dmd_4090_combined.py
```

### 旧 Wan chunk 对比

```bash
cd /root/autodl-tmp/workspace/minWM

DATA_PATH=/root/autodl-tmp/workspace/minWM/Wan21/prompts/demos.txt \
CHECKPOINT_PATH=/root/autodl-tmp/workspace/minWM/ckpts/Wan21/Action2V/dmd/model.pt \
BENCH_PROMPTS=10 \
NUM_OUTPUT_FRAMES=20 \
BASELINE_OFFLOAD_GENERATOR_BEFORE_VAE=1 \
python /root/autodl-tmp/workspace/minwm_overlay_docs/bench_minwm_dmd_4090.py
```

benchmark 输出：

```text
/root/autodl-tmp/workspace/minWM/outputs/benchmark_results/<run_id>/
├── summary.md
├── results.csv
├── stage_profile.csv
├── *.log
└── *_profile_rank0.jsonl
```

## 6. 优化版本说明

### baseline_offload

原始 minWM DMD 在 4090 上最后 VAE decode 容易 OOM，因此 baseline 使用 generator CPU offload：

```text
MINWM_VAE_BACKEND=wan
MINWM_OFFLOAD_GENERATOR_BEFORE_VAE=1
MINWM_VAE_TEMPORAL_CHUNK=0
MINWM_TORCHAO_QUANT=none
```

这个版本能稳定跑，但 generator 在 VAE 前搬到 CPU，速度较慢。

### wan_chunk

仍然使用原 Wan VAE，但把 VAE decode 按时间维切块：

```text
MINWM_VAE_BACKEND=wan
MINWM_OFFLOAD_GENERATOR_BEFORE_VAE=0
MINWM_VAE_TEMPORAL_CHUNK=2
MINWM_VAE_CHUNK_OVERLAP=0
```

这个版本主要改善显存，不是主要加速路径。

### paper_lighttae

用 LightX2V 的 LightTAEW 2.1 替换最终 Wan VAE decode：

```text
MINWM_VAE_BACKEND=lightx2v_tae
MINWM_LIGHTX2V_VAE_PATH=/root/autodl-tmp/workspace/lightx2v_ckpts/lighttaew2_1.pth
MINWM_LIGHTX2V_DTYPE=bfloat16
MINWM_LIGHTX2V_NEED_SCALED=1
MINWM_LIGHTX2V_PARALLEL=1
MINWM_LIGHTX2V_OUTPUT_DEVICE=cuda
MINWM_LAZY_WAN_VAE=1
MINWM_OFFLOAD_GENERATOR_BEFORE_VAE=0
```

这复现的是论文/截图里最核心的 autoencoder decode 加速。它不是完整 LightX2V pipeline，没有替换 DiT/generator，也没有实现 SageAttention、FP8 kernel、TeaCache 等整套推理引擎优化。

### paper_lighttae_fast3

在 LightTAE 基础上减少 DMD denoise step：

```text
MINWM_DENOISING_STEP_LIST=1000,500,250
```

原始配置是：

```text
1000,750,500,250
```

减少一步后，每个 block 的 generator forward 数从约 `4 + 1` 变为 `3 + 1`。这是达到生成阶段 2x 的关键，但会改变采样路径，需要检查质量。

## 7. 实测结果

### Wan VAE chunk

结果目录：`20260705_213020_wan/`

| case | wall time | full speedup | generation stage | generation speedup | diffusion excl. VAE | VAE decode | min free VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_offload` | 360.384s | 1.0000x | 268.312s | 1.0000x | 189.440s | 72.487s | 0.176GB |
| `wan_chunk` | 322.229s | 1.1184x | 231.634s | 1.1583x | 176.368s | 55.260s | 6.329GB |

结论：显存明显改善，但速度提升有限。

### LightTAE combined

结果目录：`20260705_214326_combined/`

| case | wall time | full speedup | generation stage | generation speedup | diffusion excl. VAE | VAE decode | min free VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_offload` | 353.259s | 1.0000x | 264.854s | 1.0000x | 186.772s | 71.659s | 0.176GB |
| `paper_lighttae` | 239.411s | 1.4755x | 148.543s | 1.7830x | 146.574s | 1.966s | 0.296GB |

结论：VAE decode 从 `71.659s` 降到 `1.966s`，VAE 部分加速 `36.45x`。但生成阶段还没到 2x，因为 DiT/generator 仍然是主要瓶颈。

### LightTAE fast3

结果目录：`20260705_223045_fast3/`

| case | wall time | full speedup | generation stage | generation speedup | diffusion excl. VAE | VAE decode | min free VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_offload` | 353.259s | 1.0000x | 264.854s | 1.0000x | 186.772s | 71.659s | 0.176GB |
| `paper_lighttae_fast3` | 231.119s | 1.5285x | 123.807s | 2.1393x | 121.796s | 2.008s | 0.296GB |

结论：`paper_lighttae_fast3` 的生成阶段加速为 `2.1393x`，达到至少 2x 的目标。代价是 DMD denoise step 从 4 步降为 3 步。

## 8. 为什么不是所有论文优化都实现

论文或 LightX2V 结果通常是整套推理栈优化，而当前 overlay 只迁移了适合接入 minWM 的部分：

- 已实现：LightTAEW 2.1 替换 VAE decode。
- 已实现：原 Wan VAE temporal chunk，解决 4090 OOM。
- 已实现：异步视频写入、lazy Wan VAE、profiling、fast3 fewer-step 版本。
- 未完整实现：完整 LightX2V pipeline。
- 未完整实现：SageAttention / FlashAttention3 / q8 kernel 等底层 attention 替换。
- 未完整实现：真正高性能 FP8 DiT/generator kernel。
- 未完整实现：TeaCache / MagCache 类跨步特征复用。

因此，`paper_lighttae` 的 VAE 部分可以达到论文级别的大幅提升，但完整 pipeline 不会自动达到同样倍数。`paper_lighttae_fast3` 通过减少 generator forward 数，把生成阶段推进到 2x 以上。

## 9. 计时字段解释

| 字段 | 含义 |
| --- | --- |
| `elapsed_seconds` | 整个子进程 wall time，包括 patch、import、模型加载、生成和视频写入。 |
| `pipeline_inference` | `pipeline.inference()` 内部耗时。VAE decode 在其中发生时，它已经包含 VAE decode。 |
| `vae_decode` | VAE decode 子阶段耗时。 |
| `write_video_submit` | 视频写入提交耗时。启用异步写入后一般很小。 |
| `generation_stage_seconds` | `pipeline_inference + write_video_submit`，避免重复计算嵌套的 VAE decode。 |
| `diffusion_excluding_vae_seconds` | `pipeline_inference - vae_decode`，用来看 DiT/generator 主体是否变快。 |
| `speedup_vs_baseline` | 全流程加速比。 |
| `generation_speedup_vs_baseline` | 生成阶段加速比。 |
| `profile_peak_allocated_gb` | PyTorch 峰值 allocated 显存。 |
| `profile_min_free_gb` | 采样到的最低剩余显存。 |

## 10. 单次推理命令

LightTAE 4-step 单次推理：

```bash
cd /root/autodl-tmp/workspace/minWM

LIGHTX2V_REPO=/root/autodl-tmp/workspace/LightX2V \
DATA_PATH=/root/autodl-tmp/workspace/minWM/Wan21/prompts/demos.txt \
CHECKPOINT_PATH=/root/autodl-tmp/workspace/minWM/ckpts/Wan21/Action2V/dmd/model.pt \
OUTPUT_FOLDER=/root/autodl-tmp/workspace/minWM/outputs/dmd_lighttae \
MAX_PROMPTS=1 \
NUM_OUTPUT_FRAMES=20 \
MINWM_VAE_BACKEND=lightx2v_tae \
MINWM_LIGHTX2V_VAE_PATH=/root/autodl-tmp/workspace/lightx2v_ckpts/lighttaew2_1.pth \
MINWM_LIGHTX2V_DTYPE=bfloat16 \
MINWM_LIGHTX2V_NEED_SCALED=1 \
MINWM_LIGHTX2V_PARALLEL=1 \
MINWM_LIGHTX2V_OUTPUT_DEVICE=cuda \
MINWM_LAZY_WAN_VAE=1 \
MINWM_COMPILE=0 \
MINWM_CLEANUP_EACH_SAMPLE=0 \
MINWM_ASYNC_VIDEO_WRITER=1 \
MINWM_LLV2_CACHE_QUANT=0 \
MINWM_OFFLOAD_GENERATOR_BEFORE_VAE=0 \
python /root/autodl-tmp/workspace/minwm_overlay_docs/run_minwm_dmd_4090.py
```

如果要单次推理启用 fast3，加：

```bash
MINWM_DENOISING_STEP_LIST=1000,500,250
```

## 11. LightTAE VAE-only 复现

只测 autoencoder，不经过 minWM DMD pipeline：

```bash
export LIGHTX2V_REPO=/root/autodl-tmp/workspace/LightX2V
cd /root/autodl-tmp/workspace/minWM

python /root/autodl-tmp/workspace/minwm_overlay_docs/bench_lightx2v_autoencoder.py \
  --case lighttae:taew2_1:/root/autodl-tmp/workspace/lightx2v_ckpts/lighttaew2_1.pth \
  --frames 81 \
  --height 480 \
  --width 832 \
  --dtype bfloat16 \
  --repeats 3 \
  --output-dir outputs/lightx2v_autoencoder_bench
```

这个结果更接近论文截图里的 “LightTAEW 2.1 路径”，因为截图通常统计 VAE/autoencoder，而不是 minWM DMD 端到端生成。

## 12. 还原 patch

如果同一个 minWM 仓库被 patch 多次，可以先还原：

```bash
cd /root/autodl-tmp/workspace/minWM

cp Wan21/wan_inference.py.before_4090_overlay Wan21/wan_inference.py 2>/dev/null || true
cp Wan21/pipeline/causal_inference.py.before_4090_overlay Wan21/pipeline/causal_inference.py 2>/dev/null || true
cp Wan21/wan_utils/wan_wrapper.py.before_4090_overlay Wan21/wan_utils/wan_wrapper.py 2>/dev/null || true
```

如果 minWM 是正常 git clone：

```bash
git restore Wan21/wan_inference.py Wan21/pipeline/causal_inference.py Wan21/wan_utils/wan_wrapper.py
```

## 13. 上传内容说明

建议上传：

```text
README.md
*.py
20260705_213020_wan/summary.md
20260705_213020_wan/results.csv
20260705_213020_wan/stage_profile.csv
20260705_214326_combined/summary.md
20260705_214326_combined/results.csv
20260705_214326_combined/stage_profile.csv
20260705_223045_fast3/summary.md
20260705_223045_fast3/results.csv
20260705_223045_fast3/stage_profile.csv
```

不建议上传：

```text
__pycache__/
*.log
*_profile_rank0.jsonl
模型权重
生成视频
```
