# minWM Action2V DMD 4090 优化复现说明

这个目录是一套 Python-only overlay，用于在单张 RTX 4090 24GB 上运行和对比
`MIN-Lab/minWM` 的 `Wan21/Action2V/dmd` 模型。代码不会下载模型权重，只负责给原
minWM 推理代码加低显存运行、计时、显存记录，并输出 benchmark 结果。

当前本地保存了两组对比结果：

| 对比 | run id | baseline | optimized | 全流程加速 | 生成阶段加速 | VAE decode 加速 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Wan VAE chunk | `20260705_213020_wan` | 360.384s | 322.229s | 1.1184x | 1.1583x | 1.3117x |
| LightTAEW combined | `20260705_214326_combined` | 353.259s | 239.411s | 1.4755x | 1.7830x | 36.4469x |

`生成阶段加速` 更接近真实推理链路，因为它主要统计 `pipeline.inference()` 和视频写入提交；`全流程加速` 还包含子进程启动、patch、import、模型加载等冷启动开销。

## 1. 依赖仓库

服务器建议统一放在 `/root/autodl-tmp/workspace`：

```text
/root/autodl-tmp/workspace/
├── minWM/                  # MIN-Lab/minWM 仓库
├── LightX2V/               # ModelTC/LightX2V 仓库，只需源码
├── minwm_overlay_docs/     # 本目录，优化和 benchmark 脚本
└── lightx2v_ckpts/         # LightTAE / LightVAE autoencoder 权重
```

需要的外部仓库：

| 仓库 | 用途 | 路径 |
| --- | --- | --- |
| `MIN-Lab/minWM` | 原始 Action2V DMD 推理代码和配置 | `/root/autodl-tmp/workspace/minWM` |
| `ModelTC/LightX2V` | LightTAEW 2.1 autoencoder 源码 | `/root/autodl-tmp/workspace/LightX2V` |

克隆命令：

```bash
cd /root/autodl-tmp/workspace

git clone https://github.com/MIN-Lab/minWM.git
git clone https://github.com/ModelTC/LightX2V.git
```

如果 GitHub 很慢，可以替换为可用镜像。LightTAE 路径只用到 LightX2V 的源码文件，不要求完整 `pip install -e LightX2V`。

## 2. Python 环境

建议在 AutoDL / SeetaCloud 上创建独立虚拟环境：

```bash
cd /root/autodl-tmp
python3 -m venv venv
source /root/autodl-tmp/venv/bin/activate

pip install -U pip setuptools wheel
cd /root/autodl-tmp/workspace/minWM
pip install -r requirements.txt
pip install huggingface_hub hf_transfer safetensors imageio imageio-ffmpeg av
```

FlashAttention 必须和当前 PyTorch/CUDA ABI 匹配。若已经有 `flash-attn` 但报 undefined symbol，需要重新本机编译：

```bash
source /root/autodl-tmp/venv/bin/activate

MAX_JOBS=4 \
FLASH_ATTENTION_FORCE_BUILD=TRUE \
pip install --force-reinstall --no-cache-dir --no-build-isolation \
  flash-attn==2.7.4.post1
```

TorchAO 只用于实验项，不是推荐主路径：

```bash
pip install torchao
```

## 3. 模型下载

### 3.1 minWM Action2V DMD

只下载 DMD stage：

```bash
cd /root/autodl-tmp/workspace/minWM
mkdir -p ckpts

HF_ENDPOINT=https://hf-mirror.com \
HF_HUB_DISABLE_XET=1 \
huggingface-cli download MIN-Lab/minWM \
  --include "Wan21/Action2V/dmd/*" \
  --local-dir ckpts
```

下载后推荐检查：

```bash
ls -lh /root/autodl-tmp/workspace/minWM/ckpts/Wan21/Action2V/dmd/
```

推理脚本里使用：

```text
CHECKPOINT_PATH=/root/autodl-tmp/workspace/minWM/ckpts/Wan21/Action2V/dmd/model.pt
```

### 3.2 LightX2V / LightTAEW 2.1 autoencoder

LightTAE combined 路径需要 `lighttaew2_1.pth`：

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

实际主路径使用：

```text
MINWM_LIGHTX2V_VAE_PATH=/root/autodl-tmp/workspace/lightx2v_ckpts/lighttaew2_1.pth
```

## 4. 文件说明

| 文件 | 作用 |
| --- | --- |
| `run_minwm_dmd_4090.py` | 单次推理入口。先执行 patch，再用单卡 `torchrun` 启动 `Wan21/wan_inference.py`。 |
| `minwm_4090_patch.py` | 主 patch 脚本。复制 runtime 文件，并修改 `wan_inference.py`、`pipeline/causal_inference.py`、`wan_utils/wan_wrapper.py`。 |
| `llv2_minwm_runtime.py` | 通用 runtime：stage profiler、显存记录、异步视频写入、原 Wan VAE temporal chunk decode、可选 cache quant。 |
| `lightx2v_minwm_runtime.py` | LightTAE 适配器。把 minWM latent layout 转成 LightX2V tiny VAE layout，再输出 `[B,T,3,H,W]`。 |
| `lightx2v_standalone_loader.py` | 只加载 LightX2V 中 `tae.py` 和 `wan/vae_tiny.py`，避免导入完整 LightX2V pipeline。 |
| `minwm_torchao_runtime.py` | TorchAO generator weight-only quantization 实验 helper。 |
| `bench_minwm_common.py` | benchmark 公共逻辑：启动子进程、读取 profile、汇总 CSV/Markdown。 |
| `bench_minwm_dmd_4090_all.py` | 所有 case 的调度器，供各个 split benchmark 入口复用。 |
| `bench_minwm_dmd_4090_combined.py` | 推荐主入口：统一 baseline vs LightTAEW combined。 |
| `bench_minwm_dmd_4090.py` | 旧 Wan chunk 对比入口：统一 baseline vs `wan_chunk`。 |
| `bench_minwm_dmd_4090_wan_experimental.py` | Wan chunk 和 Wan chunk + TorchAO 实验入口。 |
| `bench_minwm_dmd_4090_lighttae.py` | LightTAE 专用入口，功能接近 combined。 |
| `bench_minwm_dmd_4090_torchao.py` | 旧 TorchAO 实验入口。 |
| `bench_lightx2v_autoencoder.py` | 独立复现 LightTAE autoencoder VAE-only 速度和显存。 |
| `minwm_4090_patch_torchao.py` | 旧 TorchAO patcher，保留用于 ablation。 |
| `20260705_213020_wan/` | 本地保存的 Wan chunk 对比结果。 |
| `20260705_214326_combined/` | 本地保存的 LightTAE combined 对比结果。 |

## 5. 推荐运行：统一 baseline vs LightTAEW combined

这是当前最重要的命令，用同一个 baseline 对比 LightTAEW 2.1 替换 VAE 的优化版本：

```bash
cd /root/autodl-tmp/workspace/minWM

LIGHTX2V_REPO=/root/autodl-tmp/workspace/LightX2V \
DATA_PATH=/root/autodl-tmp/workspace/minWM/Wan21/prompts/demos.txt \
CHECKPOINT_PATH=/root/autodl-tmp/workspace/minWM/ckpts/Wan21/Action2V/dmd/model.pt \
BENCH_PROMPTS=10 \
NUM_OUTPUT_FRAMES=20 \
BASELINE_OFFLOAD_GENERATOR_BEFORE_VAE=1 \
MINWM_LIGHTX2V_VAE_PATH=/root/autodl-tmp/workspace/lightx2v_ckpts/lighttaew2_1.pth \
python /root/autodl-tmp/workspace/minwm_overlay_docs/bench_minwm_dmd_4090_combined.py
```

这个入口默认启用：

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `RUN_BASELINE` | `1` | 跑统一 baseline。 |
| `RUN_PAPER_LIGHTTAE` | `1` | 跑 LightTAEW combined 优化版本。 |
| `RUN_WAN_CHUNK` | `0` | 主对比不混入 Wan chunk。 |
| `RUN_PAPER_LIGHTTAE_TORCHAO` | `0` | 主对比不混入 TorchAO。 |
| `MINWM_COMPILE` | `0` | 避免短 benchmark 被 compile warmup 干扰。 |
| `MINWM_ASYNC_VIDEO_WRITER` | `1` | 视频写入异步提交。 |
| `MINWM_CLEANUP_EACH_SAMPLE` | `0` | 优化版本不每个 prompt 后 `empty_cache()`。 |
| `MINWM_LIGHTX2V_OUTPUT_DEVICE` | `cuda` | LightTAE decode 结果先留在 GPU，减少额外 CPU copy。 |
| `MINWM_LAZY_WAN_VAE` | `1` | LightTAE path 不加载原 Wan VAE。 |
| `MINWM_TORCHAO_QUANT` | `none` | 推荐对比不使用 TorchAO。 |

结果输出：

```text
/root/autodl-tmp/workspace/minWM/outputs/benchmark_results/<run_id>_combined/
├── summary.md
├── results.csv
├── stage_profile.csv
├── baseline_offload.log
├── paper_lighttae.log
├── baseline_offload_profile_rank0.jsonl
└── paper_lighttae_profile_rank0.jsonl
```

## 6. 实验运行：统一 baseline vs Wan VAE chunk

这个是早期低显存方案：仍然使用原 Wan VAE，只是把 VAE decode 按时间维切块。

```bash
cd /root/autodl-tmp/workspace/minWM

DATA_PATH=/root/autodl-tmp/workspace/minWM/Wan21/prompts/demos.txt \
CHECKPOINT_PATH=/root/autodl-tmp/workspace/minWM/ckpts/Wan21/Action2V/dmd/model.pt \
BENCH_PROMPTS=10 \
NUM_OUTPUT_FRAMES=20 \
BASELINE_OFFLOAD_GENERATOR_BEFORE_VAE=1 \
python /root/autodl-tmp/workspace/minwm_overlay_docs/bench_minwm_dmd_4090.py
```

结果输出：

```text
/root/autodl-tmp/workspace/minWM/outputs/benchmark_results/<run_id>_wan/
```

## 7. 各版本优化思路

### 7.1 `baseline_offload`

原 minWM 在 4090 上最后 VAE decode 容易 OOM，所以 baseline 使用：

```text
MINWM_VAE_BACKEND=wan
MINWM_OFFLOAD_GENERATOR_BEFORE_VAE=1
MINWM_VAE_TEMPORAL_CHUNK=0
MINWM_TORCHAO_QUANT=none
```

它的意义是“能稳定跑起来的原始 Wan VAE 版本”，但缺点是 generator 在 VAE 前被搬到 CPU，速度慢。

### 7.2 `wan_chunk`

仍然使用原 Wan VAE，但把 latent video 按时间维切块 decode：

```text
MINWM_VAE_BACKEND=wan
MINWM_OFFLOAD_GENERATOR_BEFORE_VAE=0
MINWM_VAE_TEMPORAL_CHUNK=2
MINWM_VAE_CHUNK_OVERLAP=0
```

优点是显存更稳，视频写入也可以异步；缺点是 VAE 本身还是原 Wan VAE，所以速度提升有限。

### 7.3 `paper_lighttae`

把最终 VAE decode 替换为 LightX2V 的 LightTAEW 2.1：

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

这里复现的是 LightTAE 论文/截图里最核心的 autoencoder decode 加速：VAE decode 从几十秒降到约 2 秒。但它不是完整 LightX2V pipeline，也没有把 DiT/generator 替换成 LightX2V 的 FP8 / SageAttention / 其他 kernel。

## 8. 实测结果

### 8.1 Wan VAE chunk 对比

结果目录：`minwm_overlay_docs/20260705_213020_wan/`

| 项 | 值 |
| --- | --- |
| prompts | `/root/autodl-tmp/workspace/minWM/Wan21/prompts/demos.txt` |
| bench prompts | 10 |
| frames | 20 |
| checkpoint | `/root/autodl-tmp/workspace/minWM/ckpts/Wan21/Action2V/dmd/model.pt` |

| case | status | wall time | full speedup | generation stage | generation speedup | diffusion excl. VAE | VAE decode | VAE speedup | min free VRAM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_offload` | success | 360.384s | 1.0000x | 268.312s | 1.0000x | 189.440s | 72.487s | 1.0000x | 0.176GB |
| `wan_chunk` | success | 322.229s | 1.1184x | 231.634s | 1.1583x | 176.368s | 55.260s | 1.3117x | 6.329GB |

结论：Wan chunk 明显改善最低剩余显存，从 `0.176GB` 提升到 `6.329GB`，但速度只提升 `1.12x` 左右，因为原 Wan VAE 仍然较重。

### 8.2 LightTAEW combined 对比

结果目录：`minwm_overlay_docs/20260705_214326_combined/`

| 项 | 值 |
| --- | --- |
| prompts | `/root/autodl-tmp/workspace/minWM/Wan21/prompts/demos.txt` |
| bench prompts | 10 |
| frames | 20 |
| checkpoint | `/root/autodl-tmp/workspace/minWM/ckpts/Wan21/Action2V/dmd/model.pt` |
| LightTAE checkpoint | `/root/autodl-tmp/workspace/lightx2v_ckpts/lighttaew2_1.pth` |

| case | status | wall time | full speedup | generation stage | generation speedup | diffusion excl. VAE | VAE decode | VAE speedup | min free VRAM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_offload` | success | 353.259s | 1.0000x | 264.854s | 1.0000x | 186.772s | 71.659s | 1.0000x | 0.176GB |
| `paper_lighttae` | success | 239.411s | 1.4755x | 148.543s | 1.7830x | 146.574s | 1.966s | 36.4469x | 0.296GB |

结论：

- VAE decode 从 `71.659s` 降到 `1.966s`，VAE 部分加速 `36.45x`，这基本符合 LightTAE 类优化的目标。
- 生成阶段从 `264.854s` 降到 `148.543s`，加速 `1.783x`。
- 全流程从 `353.259s` 降到 `239.411s`，加速 `1.476x`。
- 全流程没有到 2x 的原因是 DiT/generator、模型加载、进程启动仍然占比很大；LightTAE 只解决最后 VAE decode，不会自动加速整个 diffusion 主体。

## 9. 计时字段解释

| 字段 | 含义 |
| --- | --- |
| `elapsed_seconds` | 整个 benchmark 子进程 wall time，包括 patch、import、模型加载、生成、视频写入。 |
| `pipeline_inference` | `pipeline.inference()` 内部耗时。VAE decode 在 pipeline 内发生时，它已经包含 VAE decode。 |
| `vae_decode` | runtime hook 记录的 VAE decode 子阶段耗时。 |
| `write_video_submit` | 视频写入提交耗时。启用异步写入后一般很小。 |
| `generation_stage_seconds` | `pipeline_inference + write_video_submit`，避免把嵌套的 `vae_decode` 重复计算。 |
| `diffusion_excluding_vae_seconds` | `pipeline_inference - vae_decode`，用来看 DiT/generator 主体是否变快。 |
| `speedup_vs_baseline` | 全流程加速比。 |
| `generation_speedup_vs_baseline` | 生成阶段加速比。 |
| `profile_peak_allocated_gb` | PyTorch 峰值 allocated 显存。 |
| `profile_min_free_gb` | 采样到的最低剩余显存。 |

## 10. 单次推理命令

不跑 benchmark，只生成一个输出视频：

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

## 11. LightTAE VAE-only 复现

这个脚本只测 autoencoder，不经过 minWM DMD pipeline：

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

这个结果应该更接近论文/截图里的 “LightTAEW 2.1 路径”，因为截图通常统计的是 VAE/autoencoder 部分，而不是 minWM Action2V DMD 的端到端生成。

## 12. 还原 patch

如果同一个 minWM 仓库被 patch 很多次，可以先还原再跑：

```bash
cd /root/autodl-tmp/workspace/minWM

cp Wan21/wan_inference.py.before_4090_overlay Wan21/wan_inference.py 2>/dev/null || true
cp Wan21/pipeline/causal_inference.py.before_4090_overlay Wan21/pipeline/causal_inference.py 2>/dev/null || true
cp Wan21/wan_utils/wan_wrapper.py.before_4090_overlay Wan21/wan_utils/wan_wrapper.py 2>/dev/null || true
```

如果 minWM 是正常 git clone 且 `.git` 还在，用这个更干净：

```bash
git restore Wan21/wan_inference.py Wan21/pipeline/causal_inference.py Wan21/wan_utils/wan_wrapper.py
```

## 13. 上传仓库建议

建议上传以下内容：

```text
minwm_overlay_docs/
├── README.md
├── *.py
├── 20260705_213020_wan/
│   ├── summary.md
│   ├── results.csv
│   └── stage_profile.csv
└── 20260705_214326_combined/
    ├── summary.md
    ├── results.csv
    └── stage_profile.csv
```

不建议上传：

```text
__pycache__/
*.log
*_profile_rank0.jsonl
模型权重
生成视频
```
