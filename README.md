# minWM Action2V DMD 单卡 RTX 4090 优化复现

本仓库是一套用于 `MIN-Lab/minWM` 的 Python overlay 工具，目标是在单张 RTX 4090 24GB 上运行
`Wan21/Action2V/dmd`，并对不同优化版本做速度、显存和阶段耗时对比。

代码不会下载模型，也不会修改模型权重。运行时会先给 minWM 推理代码打补丁，然后启动单卡
`torchrun` 推理，并自动写出 `summary.md`、`results.csv`、`stage_profile.csv`。

## 结论概览

| 版本 | 主要优化 | 生成阶段耗时 | 生成阶段加速 |
| --- | --- | ---: | ---: |
| `baseline_offload` | 原 Wan VAE + generator CPU offload 防 OOM | 264.854s | 1.0000x |
| `wan_chunk` | 原 Wan VAE temporal chunk decode | 231.634s | 1.1583x |
| `paper_lighttae` | LightTAEW 2.1 替换最终 VAE decode | 148.543s | 1.7830x |
| `paper_lighttae_fast3` | LightTAEW 2.1 + DMD 4 step 改 3 step | 123.807s | 2.1393x |

`paper_lighttae_fast3` 达到“生成阶段至少 2x”的目标。它减少了 DMD denoise step，属于速度优先版本，需要人工检查生成质量。

## 依赖仓库和目录

推荐服务器目录：

```text
/root/autodl-tmp/workspace/
├── minWM/                  # MIN-Lab/minWM 仓库
├── LightX2V/               # ModelTC/LightX2V 仓库，只需要源码
├── minwm_overlay_docs/     # 本仓库代码
└── lightx2v_ckpts/         # LightTAE / LightVAE 权重
```

克隆依赖仓库：

```bash
cd /root/autodl-tmp/workspace

git clone https://github.com/MIN-Lab/minWM.git
git clone https://github.com/ModelTC/LightX2V.git
```

LightTAE 路径只读取 LightX2V 源码中的 autoencoder 文件，不要求完整安装 LightX2V 包。

## 环境配置

```bash
cd /root/autodl-tmp
python3 -m venv venv
source /root/autodl-tmp/venv/bin/activate

pip install -U pip setuptools wheel
cd /root/autodl-tmp/workspace/minWM
pip install -r requirements.txt
pip install huggingface_hub hf_transfer safetensors imageio imageio-ffmpeg av
```

FlashAttention 需要和当前 PyTorch / CUDA ABI 匹配。如果遇到 undefined symbol，重新本机编译：

```bash
source /root/autodl-tmp/venv/bin/activate

MAX_JOBS=4 \
FLASH_ATTENTION_FORCE_BUILD=TRUE \
pip install --force-reinstall --no-cache-dir --no-build-isolation \
  flash-attn==2.7.4.post1
```

TorchAO 只用于实验版本：

```bash
pip install torchao
```

## 模型下载

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

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `run_minwm_dmd_4090.py` | 单次推理入口。先 patch minWM，再用单卡 `torchrun` 启动推理。 |
| `minwm_4090_patch.py` | 主 patch 脚本。修改 `wan_inference.py`、`causal_inference.py`、`wan_wrapper.py`。 |
| `llv2_minwm_runtime.py` | 计时、显存记录、异步视频写入、Wan VAE temporal chunk decode、可选 cache quant。 |
| `lightx2v_minwm_runtime.py` | LightTAE 适配器。把 minWM latent 转给 LightX2V tiny VAE，并输出 `[B,T,3,H,W]`。 |
| `lightx2v_standalone_loader.py` | 只加载 LightX2V 的 `tae.py` 和 `wan/vae_tiny.py`，避免完整导入 LightX2V pipeline。 |
| `minwm_torchao_runtime.py` | TorchAO generator weight-only quantization 实验 helper。 |
| `bench_minwm_common.py` | benchmark 公共逻辑，负责启动子进程、读取 profile、汇总结果。 |
| `bench_minwm_dmd_4090_all.py` | 所有 benchmark case 的调度器。 |
| `bench_minwm_dmd_4090_combined.py` | 推荐主入口：baseline、LightTAE、fast3 对比。 |
| `bench_minwm_dmd_4090.py` | 旧 Wan chunk 对比入口。 |
| `bench_minwm_dmd_4090_wan_experimental.py` | Wan chunk 和 TorchAO 实验入口。 |
| `bench_minwm_dmd_4090_torchao.py` | 旧 TorchAO 实验入口。 |
| `bench_lightx2v_autoencoder.py` | 单独测 LightTAE autoencoder，不经过 minWM DMD pipeline。 |

## 所有优化项说明

### 1. 4090 低显存基础配置

位置：`run_minwm_dmd_4090.py`、`minwm_4090_patch.py`

默认设置：

```text
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
CUDA_MODULE_LOADING=LAZY
OMP_NUM_THREADS=1
TOKENIZERS_PARALLELISM=false
```

作用：减少 CUDA 内存碎片、延迟加载 CUDA module、避免 OpenMP 线程异常，是单卡 4090 稳定运行的基础环境。

### 2. TF32

开关：

```text
--tf32
```

作用：允许 TF32 matmul / cudnn，对 Ada / Ampere GPU 上的部分矩阵乘法有加速。默认由 `run_minwm_dmd_4090.py` 加入。

### 3. baseline 的 generator CPU offload

case：`baseline_offload`

```text
MINWM_OFFLOAD_GENERATOR_BEFORE_VAE=1
MINWM_VAE_BACKEND=wan
```

作用：原始 Wan VAE decode 在 24GB 4090 上容易 OOM，baseline 在 VAE decode 前把 generator 搬到 CPU，腾出显存。这个版本稳定但慢，是统一对比基线。

### 4. 原 Wan VAE temporal chunk decode

case：`wan_chunk`

```text
MINWM_VAE_BACKEND=wan
MINWM_VAE_TEMPORAL_CHUNK=2
MINWM_VAE_CHUNK_OVERLAP=0
MINWM_OFFLOAD_GENERATOR_BEFORE_VAE=0
```

作用：不替换 VAE，只把原 Wan VAE decode 按时间维切块，主要解决 OOM 和降低峰值显存。实测生成阶段加速 `1.1583x`。

### 5. 异步视频写入

位置：`llv2_minwm_runtime.py`

```text
MINWM_ASYNC_VIDEO_WRITER=1
```

作用：视频写入提交给后台线程，CPU 编码可以和后续 prompt 重叠。在 LightTAE 结果中，`write_video_submit` 已降到约 `0.003s / 10 prompts`。它不改变模型推理速度。

### 6. 每个 sample 后清理 CUDA cache

```text
MINWM_CLEANUP_EACH_SAMPLE=0 或 1
```

`1` 更保守，`0` 避免频繁 `empty_cache()`，速度更好。当前 LightTAE 推荐：

```text
MINWM_CLEANUP_EACH_SAMPLE=0
```

### 7. profiling 和显存记录

位置：`llv2_minwm_runtime.py`

输出：

```text
summary.md
results.csv
stage_profile.csv
*_profile_rank0.jsonl
```

记录 `pipeline_inference`、`vae_decode`、`write_video_submit`、`generation_stage_seconds`、`diffusion_excluding_vae_seconds`、峰值显存和最低剩余显存。

### 8. LightTAEW 2.1 替换最终 VAE decode

case：`paper_lighttae`

```text
MINWM_VAE_BACKEND=lightx2v_tae
MINWM_LIGHTX2V_VAE_PATH=/root/autodl-tmp/workspace/lightx2v_ckpts/lighttaew2_1.pth
MINWM_LIGHTX2V_DTYPE=bfloat16
MINWM_LIGHTX2V_NEED_SCALED=1
MINWM_LIGHTX2V_PARALLEL=1
MINWM_LIGHTX2V_OUTPUT_DEVICE=cuda
```

作用：用 LightX2V 的 LightTAEW 2.1 替换原 Wan VAE decode。实测 VAE decode 从 `71.659s` 降到 `1.966s`，VAE 部分加速 `36.45x`。

限制：只替换最终 VAE decode，没有替换 DiT/generator，也不是完整 LightX2V pipeline。

### 9. LightTAE 输出留在 GPU

```text
MINWM_LIGHTX2V_OUTPUT_DEVICE=cuda
```

作用：LightTAE decode 后先保留在 GPU，减少中间 CPU copy。若设置为 `cpu`，可降低部分显存压力但速度更慢。

### 10. lazy Wan VAE loading

```text
MINWM_LAZY_WAN_VAE=1
```

作用：LightTAE 路径不再加载原 Wan VAE，减少启动时间和显存占用。在 `wan_wrapper.py` patch 中实现。

### 11. LLV2 cache quant / history compression

```text
MINWM_LLV2_CACHE_QUANT=1
MINWM_LLV2_CACHE_MIN_NUMEL=16384
```

作用：对较大的 cache/history tensor 做 int8 压缩，目标是降低历史状态和中间缓存占用。

当前结论：这是实验项，不是主要加速来源。推荐 benchmark 默认关闭：

```text
MINWM_LLV2_CACHE_QUANT=0
```

### 12. TorchAO weight-only / FP8 量化实验

文件：

```text
minwm_torchao_runtime.py
bench_minwm_dmd_4090_torchao.py
```

开关：

```text
MINWM_TORCHAO_QUANT=int8wo
MINWM_TORCHAO_QUANT=fp8wo
MINWM_TORCHAO_QUANT=fp8dq
```

作用：尝试对 generator 做 TorchAO weight-only 或 FP8 量化。

当前结论：服务器环境 `torch 2.6.0+cu124` + `torchao 0.15.0` 下，TorchAO 日志提示 cpp extensions 不兼容；实测没有稳定提速，部分配置还会降低剩余显存。因此保留为实验项，不作为推荐路径。

### 13. torch.compile 实验

```text
MINWM_COMPILE=reduce-overhead
```

当前结论：已尝试编译 generator，但 causal KV cache / cross-attention cache 存在原地更新，Dynamo / CUDAGraph 报错。默认关闭：

```text
MINWM_COMPILE=0
```

### 14. inference_mode 和关闭 chunk0 latency 同步

已加入：

- 外层 `pipeline.inference()` 使用 `torch.inference_mode()`。
- `MINWM_RECORD_CHUNK0_LATENCY=0` 时关闭首块延迟统计中的强制 `cuda.synchronize()`。

当前结论：这两个优化不改变输出，但实测收益很小，说明主要瓶颈是 generator 真实计算，而不是 Python 同步。

### 15. fast3 fewer-step DMD

case：`paper_lighttae_fast3`

```text
MINWM_DENOISING_STEP_LIST=1000,500,250
```

原始 DMD step：

```text
1000,750,500,250
```

作用：每个 block 的 generator forward 数从约 `4 + 1` 变为 `3 + 1`，是生成阶段超过 2x 的关键。

代价：改变采样路径，可能影响质量，需要人工检查视频。

## 推荐运行命令

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

## 实测结果

### Wan VAE chunk

结果目录：`20260705_213020_wan/`

| case | wall time | full speedup | generation stage | generation speedup | diffusion excl. VAE | VAE decode | min free VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_offload` | 360.384s | 1.0000x | 268.312s | 1.0000x | 189.440s | 72.487s | 0.176GB |
| `wan_chunk` | 322.229s | 1.1184x | 231.634s | 1.1583x | 176.368s | 55.260s | 6.329GB |

### LightTAE combined

结果目录：`20260705_214326_combined/`

| case | wall time | full speedup | generation stage | generation speedup | diffusion excl. VAE | VAE decode | min free VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_offload` | 353.259s | 1.0000x | 264.854s | 1.0000x | 186.772s | 71.659s | 0.176GB |
| `paper_lighttae` | 239.411s | 1.4755x | 148.543s | 1.7830x | 146.574s | 1.966s | 0.296GB |

### LightTAE fast3

结果目录：`20260705_223045_fast3/`

| case | wall time | full speedup | generation stage | generation speedup | diffusion excl. VAE | VAE decode | min free VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_offload` | 353.259s | 1.0000x | 264.854s | 1.0000x | 186.772s | 71.659s | 0.176GB |
| `paper_lighttae_fast3` | 231.119s | 1.5285x | 123.807s | 2.1393x | 121.796s | 2.008s | 0.296GB |

## 为什么论文里的部分优化没有完整实现

论文或 LightX2V 的结果通常来自整套推理栈优化，而当前项目只迁移了适合接入 minWM 的部分。

已实现：

- LightTAEW 2.1 替换最终 VAE decode。
- 原 Wan VAE temporal chunk，解决 4090 OOM。
- 异步视频写入。
- lazy Wan VAE loading。
- profiling 和显存记录。
- cache quant 实验开关。
- TorchAO 量化实验开关。
- torch.compile 实验开关。
- fast3 fewer-step 版本。

未完整实现：

- 完整 LightX2V pipeline。
- SageAttention / FlashAttention3 / q8 kernel 等底层 attention 替换。
- 真正高性能 FP8 DiT/generator kernel。
- TeaCache / MagCache 类跨步特征复用。

因此，`paper_lighttae` 的 VAE 部分能达到论文级别的大幅提升，但完整 pipeline 不会自动达到同样倍数。`paper_lighttae_fast3` 通过减少 generator forward 数，把生成阶段推进到 2x 以上。

## 计时字段解释

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

## 单次推理命令

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

## LightTAE VAE-only 复现

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

## 还原 patch

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

## 不建议上传的内容

```text
__pycache__/
*.log
*_profile_rank0.jsonl
模型权重
生成视频
```
