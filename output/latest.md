# 🚀 vLLM & SGLang 每日更新 — 2026-11-17

> 自动生成于 2026-11-17 09:42 UTC | AI 解读: ✅ 含代码解读

## vllm
## 💡 今日关键词（放在最前面！）

- **关键词：ROCm / AMD 生态适配**
  - 一句话通俗解释：让 vLLM 这套推理引擎能在 AMD 的 GPU（MI300/MI350 等）上跑得又快又稳。
  - 为什么社区在关注：20 个 commit 里有 5 个带 `[ROCm]` 标签，涉及 MoE 路由、Muse Spark 模型、AITER 依赖、CI 镜像。这说明 AMD 不再是"顺便支持"，而是被当成一等公民在持续打磨。
  - 对新手意味着什么：如果你只会 NVIDIA CUDA，现在补一手 ROCm/HIP 知识，在招人市场上是稀缺技能；AMD 的软件栈还在快速迭代，早期参与容易做出可见贡献。

- **关键词：量化（Quantization）与 CPU Offload**
  - 一句话通俗解释：把模型权重从 16 位浮点压成 4 位/8 位，省显存；把暂时用不到的权重放到 CPU 内存里，进一步省显存。
  - 为什么社区在关注：`NVFP4 在线量化 + CPU offload`（Commit 18）和 `OCP MX 模拟 MoE`（Commit 3）都在解决"大模型塞不进单卡"的现实问题。这是部署侧最刚需的方向。
  - 对新手意味着什么：量化是入门门槛相对低、见效极快的方向——改几行配置就能让模型从"跑不动"变成"跑得动"，非常适合作为第一个贡献点。

- **关键词：稀疏注意力 / Sparse MLA**
  - 一句话通俗解释：长上下文推理时，不是每个 token 都要看前面所有 token，只挑最重要的看，从而省算力省显存。
  - 为什么社区在关注：Commit 2（QSA top-k）、Commit 6（FA3 MLA split-KV OOM）、Commit 12（HiSparse 热缓存）全是这个方向。长上下文（1M token）是当前模型竞赛的主战场。
  - 对新手意味着什么：稀疏注意力涉及 kernel 编程和内存布局，是"高性能推理"里最硬核也最值钱的部分；从读测试用例开始理解它，比直接啃论文友好得多。

- **关键词：安全加固（Security Hardening）**
  - 一句话通俗解释：防止恶意用户用超长输入、超大视频、密钥泄露等方式把服务打挂或偷数据。
  - 为什么社区在关注：Commit 11（限制错误信息里的模型名长度）、Commit 14（视频帧解码前先计入预算）、Commit 8（水印密钥从文件读）三个安全 commit 同天出现，说明 vLLM 开始认真对待"生产环境暴露在公网"的场景。
  - 对新手意味着什么：安全方向改动通常小而清晰（加个截断、加个检查），是理解代码库结构的好切入点。

- **关键词：多硬件后端抽象（Device-Generic）**
  - 一句话通俗解释：同一份代码，不写死 `torch.cuda.xxx`，而是用 `torch.accelerator.xxx`，这样 CUDA/ROCm/XPU/NPU 都能跑。
  - 为什么社区在关注：Commit 13（D2H 拷贝路径改用通用 stream/event）、Commit 7（XPU 冻结 GC 堆）、Commit 19/20（XPU 跳过冗余测试）都指向"一套代码多硬件"。
  - 对新手意味着什么：学会写设备无关代码是跨平台工程师的基本功；这类重构 commit 通常 diff 不大、逻辑清晰，适合精读学习。

---

## 🧪 重要知识点 & 动手实验

- **知识点 1：CUDA Graph 与"图外批次"的内存陷阱（来自 Commit 6）**
  - **是什么**：CUDA Graph 是把一段 GPU 计算录制成"录像带"，之后重放以省去启动开销。但录制的批次大小是固定的，超出这个大小的请求会走"eager 模式"（不用录像带，现场执行）。
  - **为什么重要**：Commit 6 修的就是这个坑——eager 模式下 FA3 的 MLA kernel 会按"最坏情况"（一个请求独占所有 query）分配一个 2.5 GiB 的 fp32 累加缓冲区，直接 OOM。不懂这个机制，你会以为"显存明明够却崩了"是玄学。
  - **动手试试**：
    ```bash
    # 观察不同 cudagraph 模式下显存占用差异（需要一张支持 FA3 MLA 的卡）
    python -c "
    import torch
    from vllm.config import CUDAGraphMode
    print('FULL_AND_PIECEWISE =', CUDAGraphMode.FULL_AND_PIECEWISE)
    print('PIECEWISE =', CUDAGraphMode.PIECEWISE)
    "
    # 然后跑 vLLM 的 MLA backend 测试，看 eager 分支的断言
    pytest tests/v1/attention/test_mla_backends.py -k "eager_decode_batch" -v
    ```
  - **预期结果**：测试会验证 `builder.max_cudagraph_size < num_tokens`，即"这个批次确实超出了图捕获范围"，并断言没有分配 split 累加器。你会理解"图内/图外"这条分界线在哪。

- **知识点 2：在线量化 + CPU Offload 的协作机制（来自 Commit 18）**
  - **是什么**：在线量化 = 加载 FP16/BF16 权重后，在 GPU 上现场压成 NVFP4。CPU Offload = 把权重放 CPU 内存，用的时候再搬到 GPU。
  - **为什么重要**：两者叠加时有个坑——量化后的权重是在 GPU 上"凭空造出来"的（meta device），offloader 不能把它当普通权重搬走。Commit 18 的处理是：全精度专家先放 CPU，量化时"一次搬一个专家到 GPU"。
  - **动手试试**：
    ```bash
    # 对比开/关 CPU offload 时的行为（需要 Blackwell SM100 GPU）
    python -m pytest tests/quantization/test_online.py \
      -k "test_online_nvfp4_quantizes_original_expert_weights" -v
    ```
    然后读这个测试的参数化：`@pytest.mark.parametrize("staging_device", ["cuda", "cpu"])`——它验证了权重从 CPU 搬上来后，量化结果依然在 CUDA 上。
  - **预期结果**：两个参数都通过，且断言 `quantized.is_cuda`。你会明白"staging device"（暂存设备）这个概念。

- **知识点 3：设备无关的 Stream / Event 写法（来自 Commit 13）**
  - **是什么**：GPU 编程里，Stream 是一条"指令流水线"，Event 是流水线上的"打卡点"。以前大家写 `torch.cuda.Stream()`，现在推荐 `torch.accelerator.current_stream()`。
  - **为什么重要**：写死 `torch.cuda` 的代码在 NPU/XPU 上直接崩。Commit 13 把 D2H（device-to-host）拷贝路径全部换成通用 API，还顺手在 CPU 后端（`shm.py`）里给 `torch.Stream` 打了桩。
  - **动手试试**：
    ```python
    import torch
    # 通用写法：不指定 cuda
    acc = torch.accelerator.current_accelerator()
    print("当前加速器:", acc)
    ev = torch.Event()
    ev.record()
    ev.synchronize()
    print("Event 通用 API 可用")
    ```
    然后 grep 一下仓库里还有多少 `torch.cuda.Stream`：
    ```bash
    grep -rn "torch\.cuda\.Stream" vllm/ | head -20
    ```
  - **预期结果**：你能看到哪些文件已经迁移、哪些还在用旧写法——这就是潜在的贡献机会。

- **知识点 4：块大小（Block Size）协商机制（来自 Commit 16）**
  - **是什么**：KV Cache 按"块"管理，块大小（如 16、32、64、128）影响内存碎片和 kernel 效率。不同 attention backend 对块大小有不同偏好。
  - **为什么重要**：Commit 16 的规则是"只有所有 backend 都偏好同一个块大小时才采用它，否则退回默认值"。以前是"某个 backend 偏好就采用"，导致 DeepSeek-V4.1 在 SM90 上 SWA 想要 256、indexer 想要 64，结果被错误地设成 256。
  - **动手试试**：
    ```bash
    python -m pytest tests/v1/worker/test_gpu_model_runner.py \
      -k "preferred_block_size" -v
    ```
    重点看 `test_preferred_block_size_requires_a_unanimous_preference` 这个用例——它用 mock backend 演示了"ROCm 双方都要 16 → 用 16"、"SWA 要 256 但 indexer 要 64 → 用默认 64"。
  - **预期结果**：所有用例通过，你能清楚说出"一致偏好"和"LCM 回退"两条路径的区别。

- **知识点 5：媒体解码预算的"先计费后消费"（来自 Commit 14）**
  - **是什么**：视频解码很吃内存。vLLM 用 `MediaDecodedBudget` 给每个请求设一个"解码字节预算"。旧逻辑是"解完帧再计费"，新逻辑是"抽样后、解码前就按 `宽×高×3` 计费"。
  - **为什么重要**：旧逻辑下，攻击者发一个超大视频，解码瞬间就能吃掉几 GB 内存——这是典型的 DoS 漏洞。新逻辑用 `tracemalloc` 测试证明峰值内存 < 2 帧大小。
  - **动手试试**：
    ```bash
    python -m pytest tests/multimodal/media/test_video.py \
      -k "budget_rejects_before_decoding or charges_decoded_frames_once" -v
    ```
    然后读 `test_video_request_budget_rejects_before_decoding`：它用 `tracemalloc` 测量峰值内存，并断言 `budget.used == 0`（被拒绝的请求不消耗预算）。
  - **预期结果**：测试通过，你会理解"预检（pre-check）"和"事后计费（post-charge）"在安全设计上的本质差别。

---

## 🔥 重要更新

- **Commit 6（FA3 MLA split-KV OOM 修复）**：修复了一个"显存够却 OOM"的隐蔽 bug，影响所有用 DeepSeek 系 MLA 模型 + CUDA Graph + 超图批次的生产部署。这是今天最有实战价值的修复。
- **Commit 18（在线 NVFP4 量化 + CPU Offload）**：让"大 MoE 模型 + 小显存"的组合成为可能，是部署侧的能力扩展，直接降低硬件门槛。
- **Commit 14（视频帧解码前计费）**：一个真实的安全漏洞修复（DoS），任何对外暴露多模态服务的部署都应该升级。
- **Commit 11（错误信息截断模型名）**：防御性安全加固——防止超长模型名撑爆日志和响应体，虽然小但体现了生产级思维。

---

## 📋 逐条解读

### 1. [Misc] Update ML4 recipe (#68126)
- **代码层面**：只改 `recipes/models/mistralai/Mistral-Large-4-1T-A52B.yaml`。把 docker 镜像从 `mistral-large-4` 专用 tag 改成正式发布的 `v0.34.0`，删掉"需要特殊镜像"的 `install` 说明和 MI350/MI355 的 `hardware_overrides`，更新日期。
- **新手概念课堂**：Recipe 就像"菜谱"——告诉你用什么模型、什么镜像、什么参数能跑起来。以前是"内部特供菜谱"，现在 Mistral Large 4 正式进了 v0.34.0 发布版，菜谱就改成"用标准食材"了。
- **对你有什么影响**：如果你要部署 Mistral Large 4，现在直接 `docker pull vllm/vllm-openai:v0.34.0` 就行，不用再找特殊镜像。

### 2. [Perf][Qwen4Exp] QSA top-k: default to FlashInfer gvr_2 on SM100 (#67741)
- **代码层面**：改 `tests/models/qwen4_exp/test_qsa_reference.py` 和 CI 配置，新增 `qsa_indexer.py` 到测试范围。测试参数化扩展为三组 `(query_lens, sequence_lens)`，新增 `large-chunk-wide` 场景（14 万 token 序列），专门覆盖 SM100 上 gvr_2 的流式 prefill 引擎。
- **新手概念课堂**：QSA（Query Sparse Attention）的 top-k 选择是"从长序列里挑最重要的 K 个 token"。FlashInfer 是一个高性能 kernel 库，`gvr_2` 是它的一种实现版本。SM100 是 Blackwell 架构的代号。就像"点外卖时从 100 家店里挑 5 家"，挑法有很多种，gvr_2 在 Blackwell 上最快。
- **对你有什么影响**：Qwen4Exp 系列模型在 Blackwell GPU 上长上下文推理会更快，但普通用户感知不到——这是内核级优化。

### 3. [ROCm][Bugfix] Route OCP MX emulation MoE experts without a host sync (#68050)
- **代码层面**：改 `tests/kernels/moe/test_ocp_mx_moe.py`。测试从"5 个专家被反量化"改为"每个 top-k 槽位一个专家"（8 槽位），并新增 `torch.cuda.set_sync_debug_mode("error")`——强制任何 host 同步都报错。把 `run()` 拆成 `make_kernel()` + `apply()`，支持同一 kernel 重放不同路由（模拟 CUDA graph replay）。
- **新手概念课堂**：MoE（混合专家）模型里，每个 token 只激活少数专家。OCP MX 是一种低精度格式。"host sync"是 GPU 停下来等 CPU 下指令——在 CUDA graph 里这是大忌，会让图重放失效。就像流水线工人每次都要等老板签字才能继续，效率极低。
- **对你有什么影响**：AMD GPU 上跑 MX 格式的 MoE 模型，CUDA graph 能真正生效，吞吐提升。

### 4. [Recipe] Make the DeepSeek-V4.1-Flash and Kimi-K3 AgentX presets opt-in (#68109)
- **代码层面**：改两个 YAML recipe 文件，删掉多处 `default: true`。意思是这些"AgentX 预设"不再默认选中，用户需要显式选择。
- **新手概念课堂**：Preset（预设）= 一套调好的参数组合。"opt-in"= 默认不启用，想要才勾选。就像手机相机里的"专业模式"——默认用自动，想要才切过去。
- **对你有什么影响**：用 recipe 生成启动命令时，不会自动套用这些实验性 AgentX 配置，避免"我什么都没选怎么就用了奇怪参数"。

### 5. [Recipe] Refresh nightly requirements after v0.34.0 (#68110)
- **代码层面**：改多个 recipe YAML，删掉 `nightly_required: true`，更新文档里"currently nightly"的措辞。因为 v0.34.0 正式发布后，之前只在 nightly 里的功能（如 MiniMax-M3、Qwen3.6 NVFP4、Qwen3.8）已经进稳定版。
- **新手概念课堂**：nightly = 每晚构建的开发版，功能新但不稳定。stable = 正式发布版。"nightly_required"标记的意思是"这功能只有 nightly 有，别用 stable 试"。现在毕业了，标记就撤了。
- **对你有什么影响**：这些模型现在可以用稳定版镜像部署，不用追 nightly 了。

### 6. [Bugfix] Fix OOM from unused FA3 MLA split-KV buffer on decode batches above CUDA graph sizes (#67489)
- **代码层面**：改 `vllm/v1/attention/backends/mla/flashattn_mla.py`（diff 里主要展示测试）。新增测试 `test_flash_attn_mla_eager_decode_batch_allocates_no_split_accumulator`，构造 170 个请求 × 48 query 的批次，断言 `max_cudagraph_size < num_tokens`（确认走 eager 路径），并验证不分配 split 累加器。
- **新手概念课堂**：FA3 = FlashAttention 3。MLA = Multi-head Latent Attention（DeepSeek 系的核心注意力）。"split-KV" = 把 KV 序列切块并行算再合并，需要一个 fp32 累加缓冲区。问题在于：eager 模式下 kernel 按"一个请求占满所有 query"的最坏情况算缓冲区大小 → 2.5 GiB。就像订会议室按"全公司都来"订，结果只来 3 个人。
- **对你有什么影响**：如果你的服务有"超出 CUDA graph 捕获范围的大批次"（比如突发流量），以前可能莫名 OOM，现在不会了。

### 7. [XPU][Bugfix] Freeze the startup heap in XPUWorker.init_device (#67639)
- **代码层面**：改 `vllm/v1/worker/xpu_worker.py`。把 `import gc` + `gc.collect()` 替换为 `freeze_gc_heap()`（来自 `vllm.utils.gc_utils`）。注释解释：冻结启动堆后，后续 `gc.collect()`（加载权重、内存分析、KV cache 定容）会跳过它，减少停顿。
- **新手概念课堂**：Python 的 GC（垃圾回收）会周期性扫描所有对象。启动时创建的对象（配置、日志器等）永远活着，扫它们纯属浪费。"冻结堆"= 告诉 GC "这批对象别扫了"。就像图书馆管理员知道"参考书区不用每天盘点"。
- **对你有什么影响**：Intel XPU 用户启动更快、内存分析阶段停顿更短。其他平台无影响（GPU worker 早已这么做）。

### 8. [Misc] Watermarking: read the key from a file (#64550)
- **代码层面**：改 `docs/features/watermarking.md`、`tests/engine/test_arg_utils.py`、`tests/test_config.py`。新增 `key_file` 字段：从文件读十进制整数密钥，忽略空白。`key` 和 `key_file` 互斥。错误信息里不引用密钥值（防止日志泄露）。
- **新手概念课堂**：水印 = 在生成文本里嵌入不可见标记，用于溯源。密钥是水印的核心机密。命令行传参会被 `ps` 看到、被 shell history 记录——所以生产环境应该从挂载的 secret 文件读。就像密码不该写在便利贴上贴显示器。
- **对你有什么影响**：部署水印服务时更安全，符合 K8s/Docker secret 的最佳实践。

### 9. [ROCm][Perf] Muse Spark Open Shard residual stream across TP (#63085)
- **代码层面**：新增 `tests/distributed/test_muse_spark_token_shard.py`，测试 `TokenShard` 类的 `local()` / `gather()` / `reduce_scatter()`。改 CI 配置加入新测试和 `vllm/models/muse_spark/amd/model.py`。
- **新手概念课堂**：TP（Tensor Parallelism）= 把一个大张量切到多张卡上。"residual stream" = Transformer 里那条贯穿所有层的主干数据流。"token shard" = 按 token 切分，每张卡负责一部分 token 的行。就像 4 个人抄一本 1000 页的书，每人抄 250 页，最后拼起来。
- **对你有什么影响**：Muse Spark 模型在 AMD 多卡上跑得更省通信、更快。普通用户无感，但这是 AMD 大模型支持的拼图。

### 10. [Core][Bugfix] Honor prefill_schedule_interval for prefix replay (#67879)
- **代码层面**：改 `vllm/v1/core/sched/scheduler.py` 和测试。当 `throttle_prefills=True`（节流 prefill）时，prefix replay（前缀重放）也要被推迟，不能插队。
- **新手概念课堂**：Prefix caching = 相同前缀的请求复用已算好的 KV。但有些 backend（如 sliding window）需要"重放"一段窗口来保证正确性。"throttle_prefills" = 限制 prefill 的调度频率，给 decode 让路。Bug 是：节流时 replay 偷偷插队了，破坏了节流语义。就像排队时 VIP 通道偷偷放人。
- **对你有什么影响**：开启 prefill 节流 + prefix caching 的组合场景下，调度行为更符合预期，延迟更可预测。

### 11. [Security] Bound invalid model names in error messages (#67765)
- **代码层面**：改 OpenAI 兼容服务器代码和文档。模型不存在时返回 404，错误信息里模型名最多显示前 256 字符 + `...`。测试用 16384 个 `x` 的模型名验证响应体被截断。
- **新手概念课堂**：如果错误信息原样回显用户输入，攻击者可以发一个 10 MB 的模型名，让服务器日志和响应体爆炸（日志注入 / 内存放大）。截断 = 只显示前 256 字符。就像快递单上地址太长会被截断，但分拣系统内部用完整地址。
- **对你有什么影响**：API 错误响应更规范、更安全。正常用户无感。

### 12. [Bugfix][HiSparse] Read the hot cache as flat rows in FlashMLA FP8 decode (#68014)
- **代码层面**：改 `vllm/v1/attention/backends/mla/flashmla_sparse.py`。把 `_fp8_kv_pages(hot_cache, ...)[0]` 换成 `flat_kv_row_view(hot_cache, ...)` 后 `unsqueeze(1)`。测试从检查"shape 是 (4,64,4)"改为检查"每个 topk index 取到的行内容正确"。
- **新手概念课堂**：HiSparse = 分层稀疏缓存，有"热缓存"（GPU 上，高频访问）和"冷缓存"（其他层）。热缓存里不同 layer 的行是交错存放的（stride=100 而非 64），所以不能简单当"64 行一页"来索引。就像停车场的车位不是连续的，你得按真实车位号找车，不能按"第几排第几个"算。
- **对你有什么影响**：HiSparse + FlashMLA FP8 解码在热缓存布局非连续时结果正确（之前可能读错行，输出乱码）。

### 13. [Core][spec_decode] Use device-generic streams and events in the D2H copy path (#67724)
- **代码层面**：改 `vllm/v1/worker/gpu/async_utils.py`、`vllm/v1/worker/cpu/shm.py` 和测试。把 `torch.cuda.Event` / `torch.cuda.Stream` 换成 `torch.Event` / `torch.accelerator.current_stream()`。CPU 后端给 `torch.Stream` 和 `torch.accelerator.set_stream/current_stream` 打桩。
- **新手概念课堂**：D2H = Device to Host，把 GPU 上的数据（比如 draft token）拷回 CPU 做决策。Stream 是异步流水线，Event 是同步点。写死 `torch.cuda` 的代码在 NPU/XPU 上跑不了。就像写"只能用 220V 插座"的电器，到了 110V 国家就废了。
- **对你有什么影响**：投机解码在非 CUDA 设备（NPU/XPU）上也能用 D2H 拷贝路径，不再崩。

### 14. [Bugfix][Security] Charge sampled video frames to the request budget before decoding (#67314)
- **代码层面**：改 `vllm/multimodal/video.py`（diff 主要展示测试和文档）。视频采样出帧数后，按 `宽×高×3 字节` 立即计入 `MediaDecodedBudget`，超预算直接拒绝，不解码。测试用 `tracemalloc` 验证峰值内存 < 2 帧大小，且 `budget.used == 0`。
- **新手概念课堂**：DoS（拒绝服务）攻击 = 让服务器资源耗尽。旧逻辑"解完再算账"，攻击者一个 4K 视频就能吃掉几 GB。新逻辑"先算账再干活"，就像餐厅先看你有多少钱再决定上多少菜。
- **对你有什么影响**：对外提供视频理解服务的部署，安全性显著提升。正常用户无感。

### 15. [Recipe] Add MiniCPM-OCR 1.0 (#68078)
- **代码层面**：新增 `recipes/models/openbmb/MiniCPM-OCR-1.0.yaml`。1.3B OCR 专用 VLM，支持 BF16/AWQ/GPTQ 三种变体，配 DSpark 投机解码（15 token/步），默认 RTX 4090，`--limit-mm-per-prompt` 限制 36 张图。
- **新手概念课堂**：OCR 专用模型 = 只干"图片转文字"这一件事，所以 1.3B 就能打。DSpark = 一种投机解码方法，用小模型猜、大模型验，加速生成。"recipe"= 开箱即用的部署配方。
- **对你有什么影响**：想跑文档解析（表格、公式、印章）的新手，现在有一张 4090 就能跑起来的现成方案。

### 16. [Bugfix][Attention] Use a block size only when every backend prefers it (#60892)
- **代码层面**：改 `vllm/platforms/interface.py` 的 `_preferred_block_size_for_backends`。规则从"任一 backend 偏好即采用"改为"所有 backend 偏好一致才采用，否则走 LCM 或默认值"。测试新增 `test_preferred_block_size_requires_a_unanimous_preference`，覆盖 ROCm/HPU/DeepSeek-V4.1-SM90 三个真实场景。
- **新手概念课堂**：块大小 = KV cache 的分配粒度。不同 attention backend（全注意力、滑窗、indexer）对块大小有不同偏好。以前"谁嗓门大听谁的"，现在"必须全体同意"。就像合租定空调温度，以前一人说了算，现在要全员一致。
- **对你有什么影响**：DeepSeek-V4.1 在 SM90 上不会再被错误地设成 256 块大小（导致 indexer 效率下降）。

### 17. [ROCm][CI] Install the AITER nightly's own requirements in the test image (#67986)
- **代码层面**：改 `docker/Dockerfile.rocm`。安装 AITER nightly wheel 时用 `--no-deps`（保留栈内 torch/triton），然后单独读 wheel 的 `requires` 列表，在栈约束下安装，最后 `pip check` 确认没有未满足依赖，并打印被改动的包。
- **新手概念课堂**：`--no-deps` = "只装这个包，别动我的依赖树"，防止 AITER 把 torch 升级到不兼容版本。但 AITER 可能依赖一个更新的 `flydsl`，不装就 import 失败。解法：装完 AITER 后，按它声明的依赖在约束内补装。就像买家具只买主体，螺丝按说明书单独配。
- **对你有什么影响**：ROCm CI 镜像更稳定，AITER 相关测试不会因依赖缺失而假失败。普通用户无感。

### 18. [Quantization] Support CPU offloading with online NVFP4 MoE quantization (#67136)
- **代码层面**：改 `vllm/model_executor/layers/quantization/online/nvfp4.py` 和 offloader。核心：`nvfp4_per_token` 方案在 CPU offload 生效时，把全精度专家放 CPU，量化时一次搬一个专家到 GPU。`UVAOffloader` 跳过 meta device 上的权重（在线量化会自己创建）。文档补充说明。
- **新手概念课堂**：在线量化 = 加载 FP16 后在 GPU 上现场压成 NVFP4。CPU offload = 权重放 CPU 内存，用时搬 GPU。两者叠加的坑：量化产物是"GPU 上凭空造的"，offloader 不该管它。就像你先在厨房（CPU）备菜，再到灶台（GPU）现炒——备菜阶段不占用灶台。
- **对你有什么影响**：大 MoE 模型（如 Qwen3.6-35B-A3B）可以在小显存卡上跑 NVFP4 量化 + CPU offload，硬件门槛降低。

### 19. [XPU][CI] Skip redundant LoRA TP4 tests on XPU (#67243)
- **代码层面**：改 `tests/lora/test_transformers_model.py`。两个 TP4 测试的 skip 条件从 `is_cuda_alike()` 扩展为 `is_cuda_alike() or is_xpu()`。
- **新手概念课堂**：TP4 = 4 卡张量并行。这些测试在 CUDA 上已经跑过了，XPU 上再跑一遍是重复劳动，浪费 CI 时间。"redundant"= 冗余。
- **对你有什么影响**：XPU CI 更快，开发者等结果时间更短。普通用户无感。

### 20. [XPU] Skip test_one_sided_all2all_default_follows_dispatch_format (#68052)
- **代码层面**：改 `tests/test_config.py`。`_SKIP_FP8_ON_ROCM` 这个 skip 标记的条件从 `is_rocm()` 扩展为 `is_rocm() or is_xpu()`。原因：测试 fixture 假装自己是 CUDA 主机，FP8 量化初始化会调 CUDA op，ROCm/XPU 都没有，会污染后续测试的模块导入状态。
- **新手概念课堂**：测试之间会互相影响——一个测试把 `vllm.models` 导入到一半失败了，后面的测试就全乱。"skip"= 跳过这个测试，避免污染。就像化学实验里一个试管炸了，别让碎片污染其他试管。
- **对你有什么影响**：XPU 上跑 `test_config.py` 不会出现莫名其妙的连锁失败。普通用户无感。

## sglang
## 💡 今日关键词

- **关键词：CUDA Graph 内存管理**
  一句话解释：CUDA Graph 把一系列 GPU 操作录制成"宏"，回放时省去 CPU 调度开销；但录制出来的对象如果没被正确释放，就会一直占着显存。
  为什么社区在关注：今天有 3 个 commit（#50980、#50981、#50982）都在修 GC 和 CUDA Graph 的交互问题，说明大模型推理对显存和延迟的"抠"已经到了非常细的粒度。
  对新手意味着什么：如果你学推理部署，别只盯着模型本身，**运行时资源生命周期**（谁创建、谁释放、何时 GC）才是生产环境真正的坑。

- **关键词：AMD ROCm / gfx1250 适配**
  一句话解释：AMD 新一代 GPU 架构 gfx1250 正在被逐项打通，从 attention kernel 到 GEMM 到 layernorm 融合。
  为什么社区在关注：今天有 4 个 AMD 相关 commit（#48095、#48065、#41683、#48064），覆盖 DSPARK、DeepSeek-V4、Qwen3.5、稀疏 prefill，说明 AMD 已经从"能跑"进入"跑得快"的阶段。
  对新手意味着什么：异构计算不再是 NVIDIA 一言堂，**会写 Triton kernel 且懂多后端差异**的人会越来越吃香。

- **关键词：NPU（昇腾）生态补齐**
  一句话解释：华为昇腾 NPU 正在把 SGLang 的各个模块逐个适配，包括序列并行、HiCache 大页内存、融合算子。
  为什么社区在关注：今天有 5 个 NPU 相关 commit（#41496、#50676、#49353、#50925、#50117），密度很高，说明国产算力落地是刚需。
  对新手意味着什么：如果你在国内做 AI 基础设施，**懂 CANN/torch_npu 的调试**是稀缺技能。

- **关键词：GC 冻结（heap freeze）**
  一句话解释：Python 的 GC 会定期扫描所有对象，对象越多越慢；`gc.freeze()` 把当前存活对象"冻结"起来，后续扫描跳过它们。
  为什么社区在关注：#50883 引入了"空闲时冻结"特性，#50980/#50981 又连修两个相关 bug，说明这是个**刚上线、正在打磨**的新功能。
  对新手意味着什么：这是理解"Python 性能优化"的绝佳案例——很多时候瓶颈不在算法，而在解释器本身。

- **关键词：KV Cache 分片（KVShard）**
  一句话解释：把 KV cache 拆成多份放到不同 rank 上，突破单卡显存上限。
  为什么社区在关注：#44447 把 KVShard 扩展到 Qwen3.8 Flash Next（压缩 QSA 模型），说明长上下文 + 大模型的组合正在倒逼显存架构创新。
  对新手意味着什么：**分布式推理的核心就是"怎么切"**，KV cache 切分是最值得深入的方向之一。

## 🧪 重要知识点 & 动手实验

- **知识点 1：Python GC 的 freeze/unfreeze 机制**
  - **是什么**：Python 垃圾回收器分"代"扫描对象；`gc.freeze()` 把当前所有对象移入一个"永久代"，之后 `gc.collect()` 不再扫描它们，从而大幅降低 GC 停顿。
  - **为什么重要**：大模型服务启动后会加载海量对象（权重、buffer、配置），这些对象几乎不会变成垃圾。如果每次 GC 都扫描它们，会造成周期性卡顿。今天 #50883 就是给调度器加了"空闲时冻结"。
  - **动手试试**：
    ```python
    import gc, time

    # 造大量长期存活对象
    data = [{"i": i, "payload": [0]*10} for i in range(1_000_000)]

    def measure(label):
        t = time.perf_counter()
        gc.collect()
        print(f"{label}: {time.perf_counter()-t:.3f}s, 冻结对象数={gc.get_freeze_count()}")

    measure("冻结前")
    gc.freeze()
    measure("冻结后")
    gc.unfreeze()
    measure("解冻后")
    ```
  - **预期结果**：你会看到"冻结后"的 `gc.collect()` 明显更快（通常快一个数量级），且 `get_freeze_count()` 从 0 变成几十万。这就解释了为什么 #50981 要特别处理"capture 前已冻结、capture 后要重新冻结"的场景。

- **知识点 2：CUDA Graph 与对象生命周期的陷阱**
  - **是什么**：CUDA Graph 捕获时会持有对 tensor、kernel 参数的引用。如果 runner 和 backend 互相引用（循环引用），Python 的引用计数释放不掉，只能靠 GC；而 GC 又被 freeze 禁用了，于是显存泄漏。
  - **为什么重要**：#50982 的注释写得很直白——"the runner and its backend reference each other, so dropping it alone leaves them to the cyclic GC, which never frees them once the heap is frozen"。这是典型的**循环引用 + GC 冻结**双重陷阱。
  - **动手试试**：
    ```python
    import gc

    class Backend:
        def __init__(self): self.runner = None
        def cleanup(self): print("backend cleanup")

    class Runner:
        def __init__(self, backend): self.backend = backend

    b = Backend(); r = Runner(b); b.runner = r  # 循环引用
    del b, r
    gc.freeze()          # 模拟 capture 后冻结
    gc.collect()
    print("冻结时 cleanup 没被调用 → 泄漏")

    gc.unfreeze(); gc.collect()
    print("解冻后才可能回收")
    ```
  - **预期结果**：冻结期间 `cleanup` 不会打印，解冻后仍然不会自动打印（因为 `__del__` 未定义），但对象会被 GC 回收。这让你直观感受到"显式 cleanup"（#50982 的修复）为什么必要。

- **知识点 3：RMSNorm 的融合与平台差异**
  - **是什么**：RMSNorm 是 `x / sqrt(mean(x²) + eps) * weight`。融合版本把归一化、缩放、shift 合并成一个 kernel，减少内存往返。
  - **为什么重要**：#49353 在 NPU 上把 `nn.RMSNorm` 换成自定义 `RMSNorm` 并禁用 CUDA 专用的 `rmsnorm_indexed_scale_shift`，因为 NPU 有自己的融合算子。**同一个数学公式，不同硬件的最优实现完全不同**。
  - **动手试试**：
    ```python
    import torch

    def rmsnorm_naive(x, w, eps=1e-6):
        return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w

    x = torch.randn(4, 8, 128)
    w = torch.randn(128)
    ref = rmsnorm_naive(x, w)

    # PyTorch 内置（会 upcast 到 fp32 做归约）
    out = torch.nn.functional.rms_norm(x, (128,), w, eps=1e-6)
    print("最大误差:", (ref - out).abs().max().item())
    ```
  - **预期结果**：误差在 1e-6 量级，说明数学等价。你可以进一步用 `torch.profiler` 对比两者的 kernel 数量，理解"融合"省了什么。

- **知识点 4：KV Cache 分片的基本思想**
  - **是什么**：把 KV cache 按 head 或按 page 切到多个 rank，每个 rank 只存一部分，decode 时通过 all-gather 或 P2P 拉取需要的部分。
  - **为什么重要**：#44447 的 diff 里出现了 `kv_shard_state_peer_only`、`_is_request_peer` 等逻辑，核心是"KV pages 每个 rank 都发，但 GDN/QSA state 只从对应的一个 rank 发"。**分片后"谁发给谁"是最容易出 bug 的地方**。
  - **动手试试**：
    ```python
    import torch

    # 模拟 4 个 rank，每个 rank 存 2 个 KV head
    num_ranks, heads_per_rank, seq, dim = 4, 2, 8, 16
    shards = [torch.randn(heads_per_rank, seq, dim) for _ in range(num_ranks)]

    # decode 时需要全部 head
    full = torch.cat(shards, dim=0)
    print("拼接后形状:", full.shape)  # [8, 8, 16]

    # 按 page 切分（每 rank 存 1/4 的 page）
    num_pages = 16
    pages = torch.randn(num_pages, dim)
    page_shards = torch.chunk(pages, num_ranks)
    print("每 rank page 数:", [p.shape[0] for p in page_shards])
    ```
  - **预期结果**：你会看到 head 切分和 page 切分的维度差异。结合 commit 里"GDN head shard r 在 prefill TP rank r 上"的描述，就能理解为什么需要 `_is_request_peer` 做 rank 映射。

- **知识点 5：Triton kernel 的多后端条件加载**
  - **是什么**：同一个 Triton kernel 在不同 GPU 架构（gfx950/gfx1250/CUDA）上的性能差异很大，所以代码里常用 `try: import ... except ImportError` 做条件加载，运行时按硬件选路径。
  - **为什么重要**：#48064 和 #41683 都用了这个模式——`_HAS_AITER_GFX1250`、`_is_gfx1250_supported`。**写跨平台 kernel 代码，第一步永远是"能力探测"**。
  - **动手试试**：
    ```python
    # 模拟能力探测模式
    def _probe():
        try:
            import triton  # noqa
            return True
        except ImportError:
            return False

    HAS_TRITON = _probe()
    print("Triton 可用:", HAS_TRITON)

    def dispatch(x):
        if HAS_TRITON:
            return "走 Triton kernel"
        return "走 PyTorch 回退"

    print(dispatch(None))
    ```
  - **预期结果**：输出取决于你环境里是否装了 triton。这个模式在 SGLang 里出现了几十次，是阅读源码的"基本功"。

## 🔥 重要更新

- **#50982 + #50981 + #50980（GC/CUDA Graph 三连修）**：这三个 commit 环环相扣——#50883 引入空闲冻结，#50980 修正"先 collect 再 freeze"的顺序，#50981 保证 capture 后重新冻结，#50982 显式释放旧 graph。**如果你在跑长稳测试时遇到显存缓慢增长，这三个修复可能就是解药**。

- **#44447（KVShard 支持 Qwen3.8 Flash Next）**：这是功能扩展，但 diff 里暴露了"CP-TP group sharing 下 state 只能从 peer rank 拿"的复杂性。**做分布式推理的同学一定要读 `_is_request_peer` 的注释**，它把 head 分片和 rank 映射讲得很清楚。

- **#48064（AMD dsv4 aiter 稀疏 prefill + fp8 wo_a + mxfp8 GEMM）**：一次性上了三个性能优化，说明 AMD 在 DeepSeek-V4 这类稀疏注意力模型上已经能跑出有竞争力的性能。**如果你在评估 AMD 方案，这个 commit 是重要参考**。

- **#50640（Ideogram 4 分辨率对齐）**：看起来是"小修"，但 `_align_resolution` 在 `_adjust_visual_fields` 和 `prepare_synthetic_warmup_request_for_queue` 两处都调用，因为"warmup 会在 _adjust 之后覆盖尺寸"。**这是典型的"生命周期顺序 bug"**，值得学习它的防御式写法。

## 📋 逐条解读

### 1. [Test] kernels: Remove input-rejection cases that only test generic argument checking
- **代码层面**：删掉了 `test_bias_gelu.py`、`test_kimi_k3_kda_decode.py`、`test_qsa_indexer.py` 里若干"传入非法参数应报错"的测试用例。这些用例测的是通用参数校验（比如 rank 不对、dtype 不对），不是 kernel 本身的计算逻辑。
- **新手概念课堂**：就像你测试"微波炉能不能加热食物"，不需要再单独测"把砖头放进去会不会报错"——那是微波炉的通用保护机制，不是加热功能。测试应该聚焦在"功能是否正确"，而不是"错误处理是否齐全"。
- **对你有什么影响**：无感知。但如果你在维护测试套件，可以学到：**测试要测"这个模块独有的逻辑"，通用校验交给框架层测**，否则测试会越来越臃肿。

### 2. [Perf] [AMD] dspark: Speed up the DSPARK draft step
- **代码层面**：`deepseek_v4_dspark.py` 的 `collapse_hc_head` 里加了一个分支：当检测到 gfx1250 GPU 且输入在 CUDA 上时，调用 `fused_hc_head` 融合 kernel，替代原来的 PyTorch 逐算子实现。
- **新手概念课堂**：DSPARK 是推测解码（speculative decoding）的草稿模型。`collapse_hc_head` 是把多个 head 的输出合并。原来要跑好几步 PyTorch 操作，现在一个 kernel 搞定——就像原来要跑 5 个窗口办手续，现在一个窗口全办完。
- **对你有什么影响**：AMD gfx1250 用户用 DeepSeek-V4 推测解码时会更快。其他平台无变化（走原来的 `hc_head_torch`）。

### 3. [Fix] diffusion: Align Ideogram 4 request sizes to the 16-pixel patch grid
- **代码层面**：`ideogram.py` 新增 `_align_resolution`，把用户请求的 width/height 向下取整到 `patch_size * ae_scale_factor` 的倍数，并在 `_adjust_visual_fields` 和 `prepare_synthetic_warmup_request_for_queue` 两处调用。
- **新手概念课堂**：图像生成模型内部把图片切成 16×16 的小块（patch）处理。如果图片尺寸不是 16 的倍数，最后一块就不完整，下游 stage 会直接报错。这个修复就是"把尺寸对齐到网格"。
- **对你有什么影响**：以前请求 1000×1000 会崩，现在会自动调整成 992×992 并打 warning。**注意：这是静默修改用户输入**，如果你对尺寸敏感，要留意日志。

### 4. [Fix] cuda-graph: Free the old decode graphs before a recapture
- **代码层面**：`model_runner.py` 的 `init_decode_cuda_graph` 开头调用新函数 `release_decode_cuda_graph`；`cuda_graph_setup.py` 把原来 `drop_elastic_cuda_graph_state` 里的 cleanup 逻辑抽成独立函数。
- **新手概念课堂**：重新捕获 CUDA Graph 前，必须先释放旧的。但 runner 和 backend 互相引用，`del runner` 释放不掉，只能等 GC——而 GC 又被 freeze 禁用了。所以必须**显式调用 `backend.cleanup()`**。
- **对你有什么影响**：修复了"elastic CUDA graph 重捕获时显存泄漏"的问题。如果你用 elastic graph 做动态 batch，长跑会更稳。

### 5. [Fix] [AMD] dsv4: Fix the mHC weight orientation across layer boundaries
- **代码层面**：`deepseek_v4_boundary.py` 和 `deepseek_v4_platform_stage.py` 里，`fn_transpose` 参数从固定的 `not self.attention` / `False` 改成 `_is_gfx1250_supported or not self.attention` / `_is_gfx1250_supported`。
- **新手概念课堂**：矩阵乘法的权重有"行主序"和"列主序"两种摆法。不同层之间如果对权重的朝向理解不一致，结果就全错。gfx1250 上的融合 kernel 期望的朝向和其他平台不同，所以要按平台切换。
- **对你有什么影响**：AMD gfx1250 用户跑 DeepSeek-V4 时，跨层计算的数值正确性修复。其他平台行为不变。

### 6. [Fix] gc: Keep the heap frozen after a CUDA graph capture
- **代码层面**：`base_cuda_graph_runner.py` 的 `freeze_gc` 里新增 `heap_was_frozen = gc.get_freeze_count() > 0`，并在 capture 结束、解冻收集后，如果之前是冻结状态就再次 `gc.freeze()`。
- **新手概念课堂**：capture 前如果堆已经冻结（比如 warmup 后），capture 过程中为了清理临时对象会 unfreeze，capture 完如果不重新 freeze，后续 GC 又会变慢。这就像"进门脱鞋，出门要记得穿回去"。
- **对你有什么影响**：修复了"warmup 后 heap 冻结状态丢失"导致的性能回退。对长跑服务的 P99 延迟有正面影响。

### 7. [Fix] [NPU] distributed: Avoid NVML queries in multimem initialization
- **代码层面**：`triton_symm_mem_ag.py` 里，原来直接调用 `is_one_nvlink_clique(...)`（内部会查 NVML），现在改成先判断 `is_cuda()` 再调用。NPU 上跳过这个检查。
- **新手概念课堂**：NVML 是 NVIDIA 的管理库，NPU 上根本没有。原来代码在 NPU 上会尝试调用 NVML 然后崩溃或报错。修复就是"先问一句'你是 NVIDIA 吗'，不是就跳过"。
- **对你有什么影响**：NPU 多机多卡场景下，multimem all-gather 初始化不再因为 NVML 缺失而失败。

### 8. [Perf] [AMD] qwen3_5: use strided GDN prefill projection views on HIP
- **代码层面**：`qwen3_5.py` 里 `use_strided_prefill_z` 的判断从 `_is_cuda and ...` 改成 `(_is_cuda or _is_hip) and ...`，让 HIP（AMD）也走 strided view 路径。测试文件加了 `register_amd_ci` 和 `skipIf(not is_cuda())`。
- **新手概念课堂**：GDN（Gated Delta Net）的 prefill 阶段需要把 QKV 投影结果拆开。strided view 是"不复制数据，只改索引方式"，省一次内存拷贝。AMD 的 Triton kernel 现在也支持这种布局了。
- **对你有什么影响**：AMD 用户跑 Qwen3.5 prefill 会更快（少一次 cat 拷贝）。CUDA 用户无变化。

### 9. [Feat] KVShard: Enable KV cache sharding for Qwen3.8 Flash Next
- **代码层面**：`kv_shard_hook.py` 里对 QSA 模型要求必须开 CP-TP group sharing；`conn.py` 新增 `kv_shard_state_peer_only` 标志和 `_is_request_peer` 等三个辅助函数，处理"KV pages 全发、state 只发 peer"的逻辑。
- **新手概念课堂**：QSA 是压缩注意力，它的 state（GDN/QSA/PLE）在每个 rank 上是完整的或按 head 切的，而 DSA 的 indexer pages 需要每个 owner 都发。所以分片后"谁发给谁"要分两类处理。
- **对你有什么影响**：Qwen3.8 Flash Next 用户可以用 KVShard 突破单卡显存限制，但必须配 `--enable-prefill-cp` 且 `--attn-cp-size == --tp-size`。

### 10. [Feat] benchmark: add rerank benchmark client
- **代码层面**：新增 `benchmark/prefill_only/bench_rerank.py`，纯 HTTP 客户端，测 `/v1/rerank` 接口的吞吐和延迟，支持配置 input_len/query_len/documents/concurrency。
- **新手概念课堂**：rerank 是"给候选文档打分排序"的模型（cross-encoder）。这个脚本帮你压测 rerank 服务的性能，输入是合成的 query+document。
- **对你有什么影响**：如果你部署 rerank 服务，现在有官方压测工具了。注意脚本会校验 token 数，确保合成文本长度符合预期。

### 11. [Fix] gc: Collect before freezing the heap
- **代码层面**：`gc_freezer.py` 去掉 `_freeze` 里的 `gc.collect()`；`gc_utils.py` 把 `gc.collect()` 移进 `freeze_gc` 开头，并从 `refreeze_gc` 里删掉。
- **新手概念课堂**：冻结前必须先收集，否则"已经是垃圾的循环引用"会被一起冻结，永远释放不掉。这个 commit 把"collect 然后 freeze"的顺序统一收口到 `freeze_gc` 里，避免调用方忘记。
- **对你有什么影响**：修复了"冻结时把垃圾也冻住"导致的内存泄漏。和 #50981、#50982 是同一系列修复。

### 12. [Perf] [AMD] dsv4: aiter sparse prefill, fp8 wo_a and mxfp8_128 GEMM
- **代码层面**：`paged_prefill.py` 在 gfx1250 上优先调用 aiter 的 `pa_prefill_sparse`；`fp8.py` 新增 `aiter_mxfp8_gfx1250_block_fp8_linear` 和 `gfx1250_mxfp8_prepare_weight` 的调度；`model_hook.py` 的 `_rocm_fp8_wo_a_supported` 扩展到 gfx1250。
- **新手概念课堂**：aiter 是 AMD 的算子库。mxfp8 是"每 32 个元素共享一个指数"的 fp8 格式，比普通 fp8 精度更好。这个 commit 让 DeepSeek-V4 在 AMD 上用上这些优化。
- **对你有什么影响**：AMD gfx1250 用户跑 DeepSeek-V4 的 prefill 和 GEMM 会显著提速。需要装 aiter。

### 13. [Feat] [NPU] layernorm-sp: Support sequence parallel on NPU
- **代码层面**：`layernorm_sp.py` 新增 NPU MC2 融合路径（`npu_mm_reduce_scatter_base` / `npu_all_gather_base_mm`），在 arch35 上禁用（精度问题）；`sp_fused_matmul_eligible` 增加 NPU 的 2-D 限制。
- **新手概念课堂**：MC2 是 CANN 的"矩阵乘 + 集合通信"融合算子，省一次通信。但 arch35（Ascend 950）上 GEMM 精度低，会掉模型准确率，所以只在老架构上开。
- **对你有什么影响**：NPU 用户开序列并行时，unquantized bf16/fp16 的 linear 会走融合路径，通信开销降低。arch35 用户走回退路径。

### 14. [Feat] [NPU] hicache: back hugepage host pools with the mmap allocator
- **代码层面**：`npu_host_memory.py` 的 `AclHostMemoryBackend` 从 `torch.empty` 改成 `alloc_mmap`，支持 `SGLANG_HUGEPAGE_SIZE`（2MB/1GB）；`ensure_capacity` 里设置 `Executor.AICPU` 和 `OFFLOAD_FLAG_GIANT_PAGE`。
- **新手概念课堂**：大页内存（hugepage）让 CPU 的 TLB 缓存命中率更高，大块数据传输更快。原来用 `torch.empty` 只能拿 4KB 页，现在用 mmap 可以申请 2MB/1GB 大页。
- **对你有什么影响**：NPU HiCache 的 host pool 性能提升，尤其是大 KV cache 换入换出场景。失败会回退到 4KB 页并打日志。

### 15. [Perf] [NPU] MiniMax-H3: optimize RMSNorm and SwiGLU
- **代码层面**：`minimax_h3.py` 里 `_norm` 在 NPU 上返回自定义 `RMSNorm` 而非 `nn.RMSNorm`；`_modulate_rmsnorm_scale_shift` 在 NPU 上禁用 CUDA 专用的 fused 路径；引入 `SiluAndMul` 替代手写 silu*mul。
- **新手概念课堂**：NPU 有自己的融合算子库，`nn.RMSNorm` 在 NPU 上可能不如自定义 `RMSNorm` 快。同理 SwiGLU 用 `SiluAndMul` 融合算子比 `silu(x) * y` 两步快。
- **对你有什么影响**：NPU 用户跑 MiniMax-H3 生成视频会更快。文档里的性能数据也更新了（4 NPU 从 103.57s 降到 96.48s）。

### 16. [Test] [NPU] deepseek-v4: Enable DSPARK on PD prefill and add off-CP / off-DSPARK testcases
- **代码层面**：`nightly-test-npu.yml` 里把原来的 DSPARK+CP 测试注释掉，新增 `off_cp` 和 `off_dspark` 两组测试用例。
- **新手概念课堂**：DSPARK、DP、CP 三个特性在 NPU 上有冲突，暂时不能同时开。所以测试拆成"关掉 CP"和"关掉 DSPARK"两组，分别验证。
- **对你有什么影响**：无直接感知。但如果你在 NPU 上同时开这三个特性遇到问题，这个 commit 告诉你"官方已知冲突，暂时别这么配"。

### 17. [Fix] [NPU] diffusion: MiniMax-H3 crashes with mxfp8 quantization or ref2va configuration
- **代码层面**：`minimax_h3.py` 把 `vsa_h3_fold_gate` 的 import 从顶层移到函数内（避免循环导入）；`_accepts_mxfp8_input` 增加 `isinstance(linear.quant_method, LinearMethodBase)` 检查；`reference_encoding.py` 把 `memfd_create` 抽成 `_create_reference_video_memfd` 并处理 seccomp 场景。
- **新手概念课堂**：循环导入是 Python 常见坑——A import B，B 又 import A。延迟导入（函数内 import）是标准解法。`memfd_create` 在容器里可能被 seccomp 禁用，所以要 try/except 回退。
- **对你有什么影响**：NPU 用户跑 MiniMax-H3 的 mxfp8 量化或 ref2va 配置不再崩溃。

### 18. [Perf] [Intel] dp: Routes all_gather_into_tensor and reduce_scatter_tensor to shm kernels
- **代码层面**：`interface.cpp` 和 `shm.cpp` 重构 `reduce_scatter_outer_loop`，把 `data_size` 参数去掉，改用 `chunk_n` 和 `rank_stride_el`；`naive_reduce_scatter` 支持"单 chunk 一次 memcpy"和"多 chunk 逐 rank 拷贝"两种模式。
- **新手概念课堂**：reduce_scatter 是"先 reduce 再 scatter"。原来按字节偏移循环，现在按元素 chunk 处理，更 cache 友好。单 chunk 时还能一次 memcpy 搞定。
- **对你有什么影响**：Intel CPU 上 DP（数据并行）的 all_gather/reduce_scatter 走共享内存 kernel，比走网络快很多。

### 19. [Feat] gc: Add idle freeze and checks for heap growth and frozen garbage
- **代码层面**：新增 `--gc-freeze-on-idle-secs` 参数和 `SGLANG_ENABLE_GC_FROZEN_GARBAGE_CHECK` 环境变量；`gc_freezer.py` 实现"空闲 N 秒后 freeze、每 10 分钟 refreeze 检查垃圾"；`role_switch.py` 在 PD 角色切换时 unfreeze。
- **新手概念课堂**：服务空闲时是"整理内存"的最佳时机——没有请求在跑，freeze 不会影响延迟。但冻结后如果有循环垃圾，就永远释放不掉，所以要定期 unfreeze 检查。
- **对你有什么影响**：长跑服务可以开 `--gc-freeze-on-idle-secs 60` 降低 GC 开销，但要注意"任何请求（包括 /health）都会重置空闲计时器"。

### 20. [Doc] diffusion: update logo
- **代码层面**：`logo.svg` 和 `logo-dark.svg` 的动画参数调整（点阵漂移轨迹、glow 透明度），`README.md` 更新描述。
- **新手概念课堂**：SVG 动画用 CSS keyframes 实现，`prefers-reduced-motion` 下自动停止——这是无障碍设计的基本要求。
- **对你有什么影响**：纯视觉更新，无功能变化。

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)