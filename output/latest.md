# 🚀 vLLM & SGLang 每日更新 — 2026-11-23

> 自动生成于 2026-11-23 09:43 UTC | AI 解读: ✅ 含代码解读

## vllm
## 💡 今日关键词（放在最前面！）

- **关键词：Pipeline Parallel 精细化构建**
  - 一句话解释：在多卡流水线并行里，每张卡只创建自己真正需要用的那部分模型（比如只有最后一张卡才建 LM Head），而不是所有卡都建一遍再丢掉。
  - 为什么社区在关注：Commit 10、11 都围绕 PP（流水线并行）展开，说明 vLLM 正在把大模型分布式部署的显存浪费一点点抠出来。大模型时代显存就是钱，这代表"分布式推理从能跑向跑得省"演进。
  - 对新手意味着什么：如果你学分布式推理，PP 是绕不开的三大并行（TP/PP/DP）之一。现在正是理解"分层构建模型"这个工程技巧的好时机。

- **关键词：量化后端百花齐放（torchao / NVFP4 / MXFP8 / AWQ）**
  - 一句话解释：把模型权重从 16 位浮点压到 4 位、8 位，省显存、提速度，但不同硬件和框架各有各的量化格式。
  - 为什么社区在关注：Commit 1（torchao AWQ bugfix）、12（XPU 加 modelopt_mxfp8）、13（Muse Spark NVFP4 在线量化）全是量化相关，说明量化已经从"实验特性"变成"生产刚需"，而且各家硬件（NVIDIA/AMD/Intel）格式不统一。
  - 对新手意味着什么：量化是当前最容易出成果的方向之一，懂一种量化格式的端到端链路（权重打包→kernel→数值对齐）就很吃香。

- **关键词：异构硬件适配（XPU / ROCm / CUDA 三线并进）**
  - 一句话解释：同一份 vLLM 代码要跑在 Intel GPU（XPU）、AMD GPU（ROCm）、NVIDIA GPU（CUDA）上。
  - 为什么社区在关注：Commit 4（ROCm MiniMax 部署配方）、7/8（Intel CI 扩展）、12/14/17（XPU 量化、安装、block size 修复）密集出现。这说明 vLLM 把 Intel 和 AMD 当一等公民在维护，"CUDA 独占"的时代在松动。
  - 对新手意味着什么：如果你手上只有 AMD 或 Intel 显卡，现在入场 vLLM 的门槛比一年前低很多；懂跨平台适配（platform 抽象层）是稀缺技能。

- **关键词：KV Cache 与连接器（Connector）生态**
  - 一句话解释：KV Cache 是推理时缓存的历史计算结果，连接器负责在 GPU、CPU、远端之间搬运它。
  - 为什么社区在关注：Commit 6（CPU offload 部分命中 bug）、11（PP 各 stage 暴露 KV 元数据给连接器）说明 KV 管理越来越复杂——PD 分离、前缀缓存、CPU 卸载全在动这块。
  - 对新手意味着什么：KV Cache 是 LLM 推理性能的核心，理解它的生命周期（分配→命中→卸载→复用）比背 API 有用得多。

- **关键词：DCP / Sparse MLA 稀疏注意力**
  - 一句话解释：DeepSeek 系模型用的 MLA 注意力 + 稀疏化，配合 Decode Context Parallel（DCP）把长上下文拆到多卡。
  - 为什么社区在关注：Commit 2 是 Kimi K3.1 系列的第 6/6 个补丁，用了 CuTe DSL 写 A2A reduce，说明前沿模型（Kimi、DeepSeek）正在推动注意力层的极致定制。
  - 对新手意味着什么：稀疏注意力和 MLA 是当前最前沿的推理优化方向，门槛高但天花板也高。

---

## 🧪 重要知识点 & 动手实验

### 知识点 1：Pipeline Parallel 下的"按需构建"（模型分层）
- **是什么**：把一个几十层的大模型切成几段，每段放在一张卡上。第一段只需要 embedding，最后一段才需要 LM Head，中间段两者都不需要。
- **为什么重要**：如果每张卡都无脑建完整的 embedding + LM Head，中间卡会白白浪费显存（LM Head 的 vocab×hidden 往往上百 MB 甚至 GB 级）。不懂这个，你部署 PP 时显存会莫名爆掉。
- **动手试试**：
  ```bash
  # 需要至少 2 张 GPU
  python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-0.8B \
    --pipeline-parallel-size 2 \
    --max-model-len 512
  ```
  然后观察两张卡的显存占用（`nvidia-smi` 或 `rocm-smi`）。
- **预期结果**：两张卡的显存占用不完全对称——第一张卡有 embedding、最后一张卡有 LM Head，中间卡（如果 PP≥3）两者都没有。这验证了 Commit 10 的"按需构建"。

### 知识点 2：在线量化（Online Quantization）——加载时顺手把 BF16 变 NVFP4
- **是什么**：checkpoint 里某些层还是 BF16（没量化），vLLM 在加载模型时现场把它们量化成 4 位，不用重新导出模型。
- **为什么重要**：很多开源模型只量化了部分层（比如 MoE 专家），剩下的 BF16 层会拖慢推理。在线量化让你"补刀"，不用等官方出全量化版本。不懂的话，你可能以为必须重新训练/导出才能量化。
- **动手试试**（需要 Blackwell GPU，如 B200/RTX 50 系）：
  ```bash
  vllm serve meta-models/Muse-Spark-Open-816B-A42B-NVFP4 \
    --quantization-config '{"targets":{"*layers.0.feed_forward.experts*":"nvfp4_per_token","*layers.61.feed_forward.experts*":"nvfp4_per_token"}}'
  ```
  如果没有这么大的模型，可以先看 Commit 13 的测试文件 `tests/kernels/moe/muse_spark_utils.py`，理解 `exclude_modules` 参数怎么控制哪些层被量化。
- **预期结果**：服务能启动，且日志里能看到 layer 0 和 61 的 experts 被量化为 nvfp4_per_token；对比不加 `--quantization-config` 的显存占用，会明显下降。

### 知识点 3：KV Cache 的"部分命中"与卸载
- **是什么**：前缀缓存（prefix caching）能命中之前算过的 token。但有时只命中"半块"——比如一个 chunk 是 16 个 token，只命中了前 12 个，剩下的 4 个还得算。算完之后，这 4 个 token 也得存进 CPU 卸载层。
- **为什么重要**：如果代码逻辑是"命中过的 chunk 就整块跳过存储"，那这 4 个新 token 就永远丢了，下次请求还得重算。Commit 6 修的就是这个边界 bug。不懂这块，你会写出"缓存命中率看起来很高但实际很慢"的代码。
- **动手试试**：直接跑 Commit 6 的单元测试，看它怎么构造这个场景：
  ```bash
  pytest -v -s tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py::test_partial_tail_hit_still_stores_the_containing_chunk
  ```
  然后读测试里的 `_make_partial_tail_scheduler` 和 `_make_partial_tail_request` 两个辅助函数。
- **预期结果**：测试通过，并且你能在 `build_connector_meta` 返回的 `store_jobs` 里看到 `block_ids.tolist() == [32]`——即只存储了那个"半块"里新算出来的部分。

### 知识点 4：量化里的"融合层 + 每 shard 独立缩放"
- **是什么**：AWQ 量化里，q_proj / k_proj / v_proj 各有一个 `act_pre_scale`（激活预缩放系数）。vLLM 为了性能把这三个矩阵融合成一个 `qkv_proj`，但融合后如果只用了最后一个 shard 的 scale，数值就全错了。
- **为什么重要**：这是典型的"性能优化引入的数值 bug"。融合（fuse）是推理加速的常见手段，但融合必须保证数学等价。不懂这个，你可能会在"明明量化了、输出却乱码"时抓瞎。
- **动手试试**：看 Commit 1 的测试 `test_awq_fused_layers_keep_each_shard_act_pre_scale`，它用 Qwen3-0.6B 做 AWQ 量化、对比 HF 和 vLLM 的 logprobs：
  ```bash
  pytest -v -s tests/quantization/test_torchao.py::test_awq_fused_layers_keep_each_shard_act_pre_scale
  ```
  （需要 CUDA + torchao 环境）
- **预期结果**：测试断言 `max_shards(model) > 1`（确认确实有多个 shard 被融合），然后 `check_logprobs_close` 通过——说明每个 shard 的 scale 都被正确应用了。

### 知识点 5：跨平台适配——用 `set_current_vllm_config` 修复 block size
- **是什么**：XPU（Intel GPU）上，FlashAttention 支持的 block size 取决于当前配置。但 worker 构建 KV cache 规格时，全局配置可能还没设好，导致用户指定的 block_size 被覆盖。
- **为什么重要**：这是"全局状态 + 多平台"的经典坑。同一个函数在 CUDA 上没问题，在 XPU 上因为 kernel 约束不同就出错。Commit 17 的修法是在调用前用 `set_current_vllm_config` 上下文管理器把配置"钉住"。
- **动手试试**：读 Commit 17 的 diff（`vllm/v1/worker/xpu_worker.py`），然后跑它的测试：
  ```bash
  pytest -v -s tests/v1/e2e/general/test_correctness_sliding_window.py::test_hybrid_sliding_window_honors_user_block_size
  ```
  （需要 XPU 或能跑 Gemma-3-1b 的环境）
- **预期结果**：测试断言 `outputs[0].num_cached_tokens == block_size`（16）——说明用户指定的 block_size 被真正尊重了，前缀缓存能命中整整一个 block。

---

## 🔥 重要更新

1. **Commit 10 + 11：PP 精细化 + KV 元数据暴露给连接器**——这是今天最成体系的一组改动。前者让 PP 各 stage 只建自己需要的层（省显存），后者让 KV 连接器能看到每个 PP stage 的 block 大小（支持更复杂的 PD 分离/卸载场景）。两者叠加，说明 vLLM 在认真做"大规模流水线部署"的工程化。

2. **Commit 2：Sparse MLA DCP 的 CuTe DSL A2A reduce（Kimi K3.1 6/6）**——Kimi 系列补丁的收官之作，用 NVIDIA CuTe DSL 手写 all-to-all reduce，这是把注意力 kernel 压到极致的信号。对做长上下文推理的人是重要参考。

3. **Commit 13：Muse Spark 在线 NVFP4 量化**——把"checkpoint 里残留的 BF16 层在加载时补量化"这个能力产品化了，配了文档和 kernel 测试。这是量化从"离线导出"走向"运行时灵活配置"的标志。

4. **Commit 6：KV Offload 部分命中 bugfix**——看似小众，但影响所有用 CPU 卸载 + 前缀缓存的场景。缓存"半块"不存会导致重复计算，是隐蔽的性能杀手。

---

## 📋 逐条解读

### 1. [Bugfix][Quantization] torchao: apply each fused shard's AWQ act_pre_scale (#69002)
- **代码层面**：改的是 `tests/quantization/test_torchao.py`（新增测试）和 torchao 量化路径的实现代码。核心是：当 q/k/v_proj 被融合成 qkv_proj 时，确保每个原始 shard 的 `act_pre_scale` 都被正确应用，而不是只用最后一个。
- **新手概念课堂**：想象你有三个不同刻度的量杯（q/k/v），倒进一个大桶（融合层）后，如果只记住最后一个杯子的刻度，倒进去的水量就全算错了。AWQ 的 `act_pre_scale` 就是"刻度"，融合后必须每个都保留。
- **对你有什么影响**：如果你用 torchao + AWQ 量化模型，之前融合层可能输出数值错误（表现为生成质量下降或乱码），现在修好了。普通用户感知不到 API 变化，但量化模型的输出会变正确。

### 2. [Attention][DCP] Sparse MLA DCP: CuTe DSL A2A reduce and DCP ShardQ (Kimi K3.1 6/6) (#66027)
- **代码层面**：改的是 `tests/distributed/test_dcp_direct_a2a_lse_reduce.py` 和 DCP 管理器相关代码。新增 CuTe DSL 写的 all-to-all reduce kernel，以及 `query_gather` 的直连路径（查询能塞进 workspace 时走快速通道）。
- **新手概念课堂**：DCP（Decode Context Parallel）是把长上下文的 KV 拆到多张卡上。decode 时每张卡算一部分，最后要把结果"汇总"（reduce）。CuTe DSL 是 NVIDIA 用来手写高性能 kernel 的 Python 工具，比 CUDA C++ 好写但性能接近。
- **对你有什么影响**：跑 Kimi/DeepSeek 这类 MLA 模型 + 长上下文 + 多卡 decode 时，吞吐会提升。普通用户无感，但如果你在做长上下文推理服务，值得关注。

### 3. [Bugfix] call sys._current_frames() to dump callstack (#68924)
- **代码层面**：改的是 `vllm/utils/watch_dog.py` 和对应测试。watchdog 超时时，用 `sys._current_frames()` 抓取所有线程的调用栈，按线程 ID 倒序写入 dump 文件，并支持类级别的 handler 共享。
- **新手概念课堂**：watchdog 是"看门狗"，监控某个操作有没有卡死。卡死时抓调用栈就像"案发现场拍照"——`sys._current_frames()` 能拿到每个线程当前执行到哪一行代码。
- **对你有什么影响**：当 vLLM 卡住（比如 feed 超时），dump 文件里会有更清晰的线程栈，方便你或开发者定位死锁。调试体验变好。

### 4. [Recipe][ROCm] MiniMax-M3.1-Flash MI355X AgentX PD disaggregation over Mooncake (#67857)
- **代码层面**：改的是 `recipes/models/MiniMaxAI/MiniMax-M3.1-Flash.yaml`。把 MI355X 上的 1P2D（1 个 prefill + 2 个 decode）部署从 MoRI-IO 传输换成 Mooncake RDMA，更新了镜像、环境变量（`MC_GID_INDEX`、`MC_IB_TC` 等）和启动参数。
- **新手概念课堂**：PD 分离 = prefill（处理输入）和 decode（逐 token 生成）用不同机器跑，因为两者计算特性不同。Mooncake 是月之暗面开源的 KV Cache 传输框架，用 RDMA 高速搬运 KV。就像"厨房做菜"和"餐厅上菜"分开，中间用传送带（RDMA）连接。
- **对你有什么影响**：如果你在 AMD MI355X 上部署 MiniMax 大模型，现在有了一份可直接抄的 Mooncake 配置。普通用户无感。

### 5. [Engram] Back NVLink peer shards with Grace EGM (#68685)
- **代码层面**：改的是 `docs/features/engram.md` 和 `tests/distributed/test_engram_dp_shard.py` 等。让 Engram（一种跨引擎共享 embedding 表的机制）的 NVLink peer shard 可以放在 Grace CPU 的 EGM（Extended GPU Memory）里，而不只是 HBM。
- **新手概念课堂**：Grace 是 NVIDIA 的 ARM CPU，EGM 是"扩展 GPU 内存"——CPU 内存通过 NVLink 让 GPU 直接访问。就像你的书桌（HBM）放不下所有书，但旁边书架（EGM）能通过快速通道直接取。
- **对你有什么影响**：在 GB200/Grace-Hopper 平台上跑 embedding 共享场景，可以省 HBM、放更大的表。普通用户无感。

### 6. [Bugfix][KV Offload] Store the chunk containing a partial-tail CPU hit (#69048)
- **代码层面**：改的是 `vllm/distributed/kv_transfer/kv_connector/v1/offloading/store_triggers/on_compute.py`。把 `cdiv`（向上取整）改成向下取整的逻辑，确保"部分命中的 chunk"里新算出来的部分仍会被存储。
- **新手概念课堂**：见知识点 3。简单说：缓存命中"半块"时，剩下一半算完必须存回去，否则下次还得重算。
- **对你有什么影响**：用 CPU offload + 前缀缓存时，重复计算减少，吞吐提升。是个隐蔽但实用的修复。

### 7. [XPU][CI] Expand more entrypoints in Intel GPU CI (#68354)
- **代码层面**：改的是 `.buildkite/intel_jobs/entrypoints_intel.yaml`。把 Intel GPU 的 CI 从单机 agent 迁到 K8s（`intel_gpu_k8s`），新增 `entrypoints/common`、`entrypoints/launchers` 等测试入口，加了分片（`parallelism: 2`）。
- **新手概念课堂**：CI（持续集成）就是"每次改代码自动跑测试"。把 CI 迁到 K8s 意味着测试环境更标准化、可弹性扩缩。
- **对你有什么影响**：Intel GPU 上的回归 bug 会更早被发现。普通用户无感，但 XPU 稳定性会提升。

### 8. [XPU][CI] revert skipped cases v1/worker (#68618)
- **代码层面**：改的是 `.buildkite/intel_jobs/misc_intel.yaml`。删掉之前为了绕过问题而加的 `--ignore` 和 `--deselect`，说明那些测试在 XPU 上已经能过了。
- **新手概念课堂**：`--ignore` 是"跳过这个测试文件"，`--deselect` 是"跳过这个具体测试"。删掉它们 = 重新启用测试。
- **对你有什么影响**：XPU 上 `v1/worker` 和 `v1/streaming_input` 的测试覆盖恢复，质量更有保障。

### 9. [Doc] Highlight HTTP-only (no vLLM import) weight updates in training docs (#69333)
- **代码层面**：改的是 `docs/training/README.md` 和 `docs/training/weight_updates/README.md`。新增一节说明"训练器可以不安装/导入 vLLM，直接通过 HTTP 端点同步权重"，并强调 `VLLM_SERVER_DEV_MODE=1` 才能开这些端点。
- **新手概念课堂**：RLHF 训练时，训练器（PyTorch/JAX）要把新权重推给推理引擎（vLLM）。以前必须 import vLLM，版本冲突很烦。现在纯 HTTP + NCCL/IPC 协议，训练器零依赖。
- **对你有什么影响**：如果你用 JAX 或不同版本的 PyTorch 做 RLHF，现在可以完全解耦。文档还给了两个可跑的例子（`rlhf_http_nccl_no_vllm.py` 等）。

### 10. [Model][PP] Build input embeddings, final norms and LM heads only on the pipeline stages that use them (#67349)
- **代码层面**：改的是 `vllm/model_executor/models/` 下多个模型文件（`cosmos3_edge.py`、`gemma3.py`、`nemotron_h.py` 等）。用 `get_pp_group().is_last_rank` 判断，非首/末 stage 用 `PPMissingLayer()` 占位，不真正创建 embedding/LM Head。
- **新手概念课堂**：见知识点 1。就像流水线工厂，第一道工序才需要"进料口"（embedding），最后一道才需要"包装机"（LM Head），中间工位不需要。
- **对你有什么影响**：PP 部署时显存占用下降，能跑更大的模型或更长上下文。测试新增了 Qwen3.5、Gemma3、Granite4 的 PP 覆盖。

### 11. [Core][PP][KV Connector] Expose per-stage KV cache metadata to connectors (#67998)
- **代码层面**：改的是 `vllm/v1/core/kv_cache_planning.py`（新增 `get_pipeline_stages`）、KV connector 相关代码和多个测试。让每个 worker 都知道每个 PP stage 的 `bytes_per_block`（包括最后 stage 的 draft 层）。
- **新手概念课堂**：KV 连接器要搬运 KV Cache，必须知道"一块 KV 有多大"。PP 下每个 stage 的层数不同，块大小也不同。以前连接器只能看到自己 stage 的，现在能看到所有 stage 的。
- **对你有什么影响**：PD 分离 + PP + KV 卸载的组合场景会更稳定。同时删掉了一个"忽略 runtime offload sizing"的配置哈希测试，说明这块语义收紧了。

### 12. [XPU] add modelopt_mxfp8 quantization format (#68858)
- **代码层面**：改的是 `vllm/platforms/xpu.py`，在支持列表里加了 `"modelopt_mxfp8"`。
- **新手概念课堂**：MXFP8 是 OCP 定义的 8 位浮点格式（带共享指数），ModelOpt 是 NVIDIA 的量化工具。加进 XPU 支持列表 = Intel GPU 现在能跑这种量化格式的模型。
- **对你有什么影响**：如果你有 Intel GPU 且模型是 ModelOpt MXFP8 量化的，现在能直接跑。一行改动，但解锁了一类模型。

### 13. [Model] Quantize Muse Spark BF16 routed experts with online nvfp4_per_token (#64552)
- **代码层面**：改的是 `docs/features/quantization/online.md`、`tests/kernels/moe/muse_spark_utils.py` 等。支持用 `--quantization-config` 指定某些层（如 layer 0 和 61 的 experts）在加载时在线量化为 `nvfp4_per_token`，并加了 TRT-LLM MoE kernel 测试。
- **新手概念课堂**：见知识点 2。`nvfp4_per_token` = 4 位浮点，每个 token 单独算缩放系数（更精确）。
- **对你有什么影响**：跑 Muse Spark 这类"部分量化"的 checkpoint 时，可以补量化残留的 BF16 层，省显存、提速度。需要 Blackwell GPU + FlashInfer TRT-LLM 后端。

### 14. [XPU][Doc] Add vllm-xpu-kernels find-links to wheel install commands (#68626)
- **代码层面**：改的是 `docs/getting_started/installation/gpu.xpu.inc.md`。在 `uv pip install` 命令里加了 `--find-links https://vllm-project.github.io/vllm-xpu-kernels/whl/vllm-xpu-kernels/`。
- **新手概念课堂**：`--find-links` 是 pip/uv 的"额外包源"，用于那些不在 PyPI 上的预编译 kernel 包。XPU 的 kernel 单独发布，所以要额外指路。
- **对你有什么影响**：在 Intel GPU 上装 vLLM 时，按文档命令复制粘贴就能装全（包括 kernel），不会出现"缺 kernel 包"的报错。

### 15. [Bugfix][Benchmark] Keep bench serve warmups and ready check out of the prefix cache (#66701)
- **代码层面**：改的是 `vllm/benchmarks/serve.py` 和新增测试 `tests/benchmarks/test_serve_cache_salt.py`。给 ready check 和 warmup 请求加 `cache_salt`（唯一盐值），让它们不污染正式压测的前缀缓存；如果服务器拒绝 salt 才降级去掉。
- **新手概念课堂**：`cache_salt` 是"缓存盐"——加了不同盐的请求，前缀缓存视为不同，不会互相命中。就像给试吃的菜加个标记，不跟正式菜单混。
- **对你有什么影响**：如果你用 `vllm bench serve` 压测，之前 warmup 请求会把前缀缓存"预热"，导致正式测试的 TTFT 偏乐观。现在压测数据更真实。

### 16. [Doc] Fix relative example links in docs/training/weight_updates/engines (#69300)
- **代码层面**：改的是 `docs/training/weight_updates/engines/` 下 `ipc.md`、`m2n.md`、`nccl.md` 等。把 `../../../examples/` 改成 `../../../../examples/`（多一层 `../`）。
- **新手概念课堂**：Markdown 相对链接的路径层级。文档目录深了一层，链接就得多退一级。
- **对你有什么影响**：点文档里的示例链接不再 404。纯文档修复，但很实用。

### 17. [Bugfix][XPU] Honor user block size for sliding-window layers on XPU (#68108)
- **代码层面**：改的是 `vllm/v1/worker/xpu_worker.py`，重写 `get_kv_cache_layer_configs`，用 `set_current_vllm_config(self.vllm_config)` 上下文包住父类调用。新增 e2e 测试验证 Gemma-3 的 sliding-window block size。
- **新手概念课堂**：见知识点 5。sliding-window 注意力只看最近 N 个 token，KV 分块策略和全注意力不同；XPU 的 FlashAttention kernel 对 block size 有约束，配置没设好就会被改。
- **对你有什么影响**：XPU 上跑 Gemma-3 这类 sliding-window 模型 + 前缀缓存时，用户指定的 `--block-size` 会被尊重，缓存命中率正常。

### 18. [Qwen4Exp] Share FP8 PLE tables over NVLink and RDMA (#68684)
- **代码层面**：改的是 `docs/features/engram.md`、`tests/models/qwen4_exp/test_ple.py` 等。让 Qwen4Exp 的 PLE（Per-Layer Embedding）表可以通过 Engram pool 在 NVLink 或 RDMA 上跨引擎共享，支持 `cpu_offload` 和 `peer_group_transport=rdma`。
- **新手概念课堂**：PLE 是 Qwen 系列的一种"每层独立 embedding 表"，非常大。多个推理引擎（比如 prefill 和 decode）可以共享一份，而不是各存一份。NVLink 是同机高速互联，RDMA 是跨机网络。
- **对你有什么影响**：跑 Qwen3.8-Flash-Next-FP8 时，多引擎可以共享 PLE 表，省显存。但注意：不支持 `ipc_cache`、`sharded_state` 加载器和在线权重更新，改权重需重启。

### 19. [BUG] Fix async EPLB worker shutdown ordering (#67589)
- **代码层面**：改的是 `vllm/v1/worker/gpu_worker.py` 和 `tests/distributed/test_eplb_execute.py`。确保 EPLB（Expert Parallel Load Balancer）的异步 worker 在模型资源释放之前先停掉，避免"worker 还在访问已释放的模型"。
- **新手概念课堂**：EPLB 是 MoE 模型的"专家负载均衡器"，异步模式下有个后台线程在搬专家。如果主线程先销毁模型、后台线程还在跑，就会崩。就像先关厨房再赶厨师，顺序反了。
- **对你有什么影响**：用 `enable_eplb=True` + `use_async=True` 的 MoE 部署，shutdown 时不再崩溃或报错。测试验证了 generate 后正常退出。

### 20. [Bugfix] Restore HY4 Flash CUDA platform checks (#69193)
- **代码层面**：改的是 `vllm/models/hy_v4_flash/nvidia/model.py` 和 `vl_model.py`。新增 `validate_platform()` 函数，在 `__init__` 里检查 `current_platform.is_cuda()`，非 CUDA 直接抛 `NotImplementedError`。
- **新手概念课堂**：HY4 Flash 是一个只支持 NVIDIA CUDA 的模型（用了 CUDA 专属 kernel）。之前可能在非 CUDA 平台上静默失败或报奇怪的错，现在明确报"不支持"。
- **对你有什么影响**：如果你在 AMD/Intel 上尝试加载 HY4 Flash，会立刻得到清晰错误，而不是跑到一半崩溃。错误信息更友好。

## sglang
## 💡 今日关键词（放在最前面！）

- **关键词：异构后端适配（AMD ROCm / NPU / CPU）**
  - 一句话通俗解释：同一份推理代码，要在 NVIDIA、AMD、华为昇腾 NPU、Intel CPU 上都能跑对、跑快。
  - 为什么社区在关注：今天 20 个 commit 里至少 4 个（#52234 AMD qknorm、#51747 NPU hicache、#46559 CPU GPTQ、#50968 NPU DSA-TAIL）都是"某硬件上某功能修 bug"。这说明 SGLang 已从"GPU 推理框架"演化为"多硬件推理框架"，硬件厂商在主动贡献补丁。
  - 对新手意味着什么：如果你会 CUDA 之外的 Triton / CANN / oneDNN，或者愿意读硬件厂商的 issue，这是一个门槛不高、竞争不激烈的切入点——很多 bug 都是"某后端没考虑某个 dtype/flag"。

- **关键词：投机解码（Speculative Decoding）与 MTP/Draft 模型**
  - 一句话通俗解释：用小模型"猜"后面几个 token，大模型一次性验证，从而加速生成。
  - 为什么社区在关注：#52234（draft megakernel 不满足要求时优雅降级）、#46784（DSA MTP draft 的 prefill KV cache 只对最后一个 token 做 attention）都是围绕 draft 模型的工程细节。说明投机解码已进入"打磨性能与鲁棒性"阶段，而不是"能不能用"阶段。
  - 对新手意味着什么：投机解码是当前 LLM 推理加速最热的方向之一，理解 draft/target 模型、KV cache、verify 流程，是进入推理框架核心开发的好路径。

- **关键词：Triton kernel 的"请求路径不重编译"**
  - 一句话通俗解释：Triton 会为每种参数形状/常量组合编译一个 kernel 变体；如果形状随请求变化，就会在线上反复编译，造成卡顿。
  - 为什么社区在关注：#52224（VSA-H3 用 `do_not_specialize`）、#52225（warm-up 时预编译 cuDNN attention kernel）、#52220（给 6 个热点 Triton wrapper 加 `replay_launches`）——三个 commit 都在解决"首次请求慢/抖动"。
  - 对新手意味着什么：会写 Triton kernel 只是第一步，会"控制 Triton 的编译行为"才是生产级技能。`tl.constexpr`、`do_not_specialize`、`do_not_specialize_on_alignment` 是必须掌握的关键词。

- **关键词：HiCache / 分层 KV cache 与异步流水线**
  - 一句话通俗解释：把 KV cache 在 GPU、CPU 内存、甚至远端存储之间分层管理，用异步拷贝隐藏延迟。
  - 为什么社区在关注：#50469（用 stream fence 代替阻塞 scheduler）、#51747（NPU 去掉 giant-page flag）——HiCache 的瓶颈从"功能"转向"别阻塞调度器"。
  - 对新手意味着什么：CUDA stream / event 的同步语义是高性能推理的必修课；理解"为什么阻塞 host 会拖慢整个调度循环"是进阶关键。

- **关键词：采样（Sampling）重构与确定性**
  - 一句话通俗解释：从 logits 到最终 token 的采样流程，正在被拆成清晰的 pipeline 阶段，并保证可复现。
  - 为什么社区在关注：#52277（用安全 RNG 生成 filler seed）、#52270（每步构建 seed/mask 行，删掉 filter/merge）、#52192（grammar mask 作为 pipeline 阶段）——连续三个采样重构，说明采样模块在做架构级清理。
  - 对新手意味着什么：采样看似简单，实则涉及随机数、mask、grammar、自定义 logit processor 的交互；这是理解"推理框架数据流"的绝佳样本。

---

## 🧪 重要知识点 & 动手实验

- **知识点 1：Triton 的 `do_not_specialize`——别让形状变化触发重编译**
  - **是什么**：Triton 默认会把标量参数的值/对齐性当作"特化"依据，值不同就编译新 kernel；`do_not_specialize=["N_TILES"]` 告诉它"这个参数变了也别重编"。
  - **为什么重要**：在 diffusion / VLM 场景中，prompt 长度、tile 数随请求变化，若不控制会导致请求路径上反复 JIT 编译，首 token 延迟抖动。不懂这个坑，你会以为"代码没改为什么突然慢"。
  - **动手试试**：
    ```python
    import torch, triton, triton.language as tl

    @triton.jit
    def k_spec(x_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        tl.store(x_ptr + offs, tl.load(x_ptr + offs), mask=offs < n)

    @triton.jit(do_not_specialize=["n"])
    def k_nospec(x_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        tl.store(x_ptr + offs, tl.load(x_ptr + offs), mask=offs < n)

    for f in (k_spec, k_nospec):
        x = torch.zeros(1024, device="cuda")
        for n in (100, 200, 300, 400):
            f[(triton.cdiv(n, 128),)](x, n, BLOCK=128)
        print(f.__name__, "cache size:", len(f.device_caches[0][0]))
    ```
  - **预期结果**：`k_spec` 的 cache size 会随 n 的不同值增长（多个变体），`k_nospec` 只保留一个变体。说明你掌握了"控制 Triton 特化"的方法。

- **知识点 2：CUDA Stream / Event 的 fence 语义——用异步代替阻塞**
  - **是什么**：`stream.wait_event(ev)` 让一个 stream 等待另一个 stream 上的事件完成，而不阻塞 CPU；`torch.cuda.synchronize()` 则阻塞整个 host。
  - **为什么重要**：#50469 把 HiCache 的 write-back 从"阻塞 scheduler"改成"在 forward stream 上 fence"，这样调度器可以继续处理下一个 batch。不懂 stream/event，你写的异步代码要么不生效、要么出现数据竞争。
  - **动手试试**：
    ```python
    import torch, time

    a = torch.zeros(1 << 26, device="cuda")
    b = torch.zeros_like(a)
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()

    # 阻塞版
    t0 = time.perf_counter()
    for _ in range(20):
        b.copy_(a, non_blocking=True)
        torch.cuda.synchronize()
    print("blocking:", time.perf_counter() - t0)

    # event fence 版
    t0 = time.perf_counter()
    for _ in range(20):
        with torch.cuda.stream(s1):
            b.copy_(a, non_blocking=True)
        ev = torch.cuda.Event(); ev.record(s1)
        s2.wait_event(ev)
    torch.cuda.synchronize()
    print("fenced:", time.perf_counter() - t0)
    ```
  - **预期结果**：fenced 版本 CPU 侧几乎不等待，整体耗时更短（尤其在拷贝与计算可重叠时）。你会理解 #50469 为什么能"不阻塞 scheduler"。

- **知识点 3：INT4 量化的对称 vs 非对称——zero point 到底怎么算**
  - **是什么**：对称量化把 0 映射到 0（zero point = 2^(bits-1) = 8）；非对称量化用 checkpoint 里存的 zero point，且常见实现会 +1 偏移（`(stored & 0xF) + 1`）。
  - **为什么重要**：#46559 修复了 Qwen3.5 GPTQ-Int4 在 CPU 上"规则失败 + 精度下降"，根因就是对称/非对称的 zero point 处理不一致。量化搞错 zero point，模型输出会"看起来能跑但答非所问"。
  - **动手试试**：
    ```python
    import torch

    def dequant_sym(qweight, scales, group_size=128):
        # qweight: int8 in [0,15], zero point = 8
        return (qweight.float() - 8) * scales.repeat_interleave(group_size, dim=-1)

    def dequant_asym(qweight, qzeros, scales, group_size=128):
        # qzeros: 存的 zero point，解码时 +1
        zp = (qzeros & 0xF) + 1
        return (qweight.float() - zp.repeat_interleave(group_size, dim=-1)) * scales.repeat_interleave(group_size, dim=-1)

    qw = torch.randint(0, 16, (1, 256), dtype=torch.int8)
    sc = torch.rand(1, 2)
    qz = torch.zeros(1, 2, dtype=torch.int8)
    print("sym  :", dequant_sym(qw, sc)[0, :4])
    print("asym :", dequant_asym(qw, qz, sc)[0, :4])
    ```
  - **预期结果**：对称版以 8 为中心，非对称版以 1 为中心（因为 qz=0 → zp=1）。你会明白为什么"zero point 差 1"会造成精度显著下降。

- **知识点 4：投机解码中的 draft 模型"优雅降级"**
  - **是什么**：当某个加速特性（如 megakernel）不满足硬件/模型条件时，draft 模型应回退到自己的普通 forward，而不是让整个 server 拒绝启动。
  - **为什么重要**：#52234 的核心思想——"能力探测失败 = 降级，不是崩溃"。生产系统中，一个可选优化不适用不应导致服务不可用。这是所有推理框架的通用设计原则。
  - **动手试试**（伪代码，理解模式）：
    ```python
    def try_install_optimization(spec, device):
        reasons = check_requirements(spec, device)
        if reasons:
            if is_draft:
                logger.info("draft keeps its own forward: %s", reasons)
                return False   # 降级，不抛异常
            raise RuntimeError(f"target cannot start: {reasons}")
        install(spec)
        return True
    ```
  - **预期结果**：你能写出一段"target 严格、draft 宽松"的安装逻辑，并解释为什么这样设计能提高系统可用性。

- **知识点 5：安全随机数与可复现采样**
  - **是什么**：`secrets.randbits()` 用操作系统 CSPRNG 生成种子，避免多个请求"撞种子"；同时保留显式 `sampling_seed` 保证用户可复现。
  - **为什么重要**：#52277 用 `secrets` 替代普通 `random` 生成 filler seed，防止请求间随机性相关（尤其在批量采样时）。不懂这个，你可能写出"同一 batch 内多个请求输出高度相似"的 bug。
  - **动手试试**：
    ```python
    import random, secrets, collections

    def collisions(gen, n=10000):
        c = collections.Counter(gen() for _ in range(n))
        return sum(1 for v in c.values() if v > 1)

    print("random :", collisions(lambda: random.getrandbits(32)))
    print("secrets:", collisions(lambda: secrets.randbits(32)))
    ```
  - **预期结果**：`random.getrandbits(32)` 在 1 万次采样中可能出现碰撞（尤其种子未初始化时），`secrets` 几乎不会。你会理解为什么"filler seed"要用 CSPRNG。

---

## 🔥 重要更新

- **#50469 HiCache 用 stream fence 替代阻塞 scheduler**：直接影响高并发下的调度吞吐，是 HiCache 从"能用"走向"高性能"的关键一步，所有开启分层 KV cache 的用户都会受益。
- **#52277 + #52270 + #52192 采样模块三连重构**：采样流程被拆成清晰的 pipeline 阶段，种子生成更安全，是后续支持更多采样特性（grammar、自定义 processor）的架构基础，影响面覆盖所有请求。
- **#52234 draft megakernel 优雅降级**：修复了"一个可选优化不适用导致整个 server 拒绝启动"的可用性问题，对启用投机解码 + NVIDIA megakernel 的部署尤其重要。
- **#46559 CPU GPTQ-Int4 zero point 修复**：直接修复了 Qwen3.5-35B-A3B-GPTQ-Int4 在 CPU 上的精度下降，是量化 + CPU 后端用户的实际收益。

---

## 📋 逐条解读

### 1. [Fix] [AMD] qknorm: Fall back from fused QK norm on ROCm when norm weights differ in dtype (#52234)
- **代码层面**：改 `python/sglang/srt/models/utils.py` 的 `apply_qk_norm`。原来条件是 `_is_cuda or _is_hip` 就走融合 kernel；现在 HIP 分支额外要求 `q_norm.weight.dtype == k_norm.weight.dtype == q.dtype`，否则回退到非融合路径。
- **新手概念课堂**：融合 QK norm 就像一个"打包工具"，把 Q、K 的归一化一次算完。但这个工具只接受"同一种包装材料"（dtype）；AMD 上如果权重是 fp32 而激活是 bf16，工具会算错或崩溃，所以要"换回普通工具"。
- **对你有什么影响**：在 AMD GPU 上跑某些 dtype 混合的模型（如 norm 权重保存为 fp32）时，不再报错或精度异常，而是自动走慢一点但正确的路径。

### 2. [Fix] kernel_plugins: Let a draft megakernel spec this server does not meet decline instead of refusing the server (#52438)
- **代码层面**：`kernel_plugins/nvidia/binding.py` 新增 `_residency_reasons()`，把"SM 数量不够"从 `_model_disable_reasons` 拆出来；`install_megakernel` 在 `is_draft` 且存在未满足条件时，所有 TP rank 一起 `return`（降级），只打日志不抛异常。测试新增 `test_a_draft_spec_this_server_does_not_meet_declines_on_every_rank`。
- **新手概念课堂**：target 模型是"主厨"，draft 模型是"帮厨"。帮厨的专用刀具（megakernel）不合手，就让他用普通刀，而不是把整个厨房关掉。关键是"所有 rank 一起决定"，否则分布式下会死锁。
- **对你有什么影响**：启用投机解码 + megakernel 时，即使 draft 模型不满足 megakernel 条件，server 也能正常启动，只是 draft 走普通 forward。

### 3. [Fix] cookbook: Render AgentX linker options with --unified-cache-external-linker-config (#52373)
- **代码层面**：改 `docs/src/snippets/_agentx.jsx`，新增 `withLinkerConfig`，把旧参数 `--hicache-storage-backend-extra-config` 重写为 `--unified-cache-external-linker-config`，并保持参数排序。
- **新手概念课堂**：这是文档渲染层的"参数迁移适配器"——老教程里写的旧参数，新版本 server 不认了，文档生成时自动翻译成新参数，避免用户复制粘贴报错。
- **对你有什么影响**：看 AgentX cookbook 时，生成的命令能直接在 SGLang 1.0 上跑，不会因为参数名过时报错。

### 4. [Fix] [NPU] hicache: Drop the MemFabric giant-page flag for AICPU (#51747)
- **代码层面**：`npu_host_memory.py` 删掉 `config.flags = offload.OFFLOAD_FLAG_GIANT_PAGE` 一行。
- **新手概念课堂**：巨页（giant page）像"大号集装箱"，能减少 TLB miss；但 AICPU 执行器不支持这个 flag，设了反而出问题。删掉就是"别给不支持的搬运工发大集装箱"。
- **对你有什么影响**：NPU 上开 HiCache 时，host memory 初始化不再因 flag 不兼容而失败。

### 5. [Fix] [CPU] Fix: Qwen3.5-35B-A3B-GPTQ-Int4 quantization rule failure and accuracy degradation (#46559)
- **代码层面**：`gemm.h` / `gemm_int4.cpp` 给 `convert_weight_packed_scale_zp`、`unpack_4bit_to_32bit_signed`、`autogptq_to_int4pack`、`int4pack` 都加了 `is_symmetric` 参数；对称 GPTQ 用 zero point 8，非对称解码时 `(stored & 0xF) + 1`；AWQ 对称不支持，非对称用 checkpoint zero point。
- **新手概念课堂**：INT4 量化像"把 0~15 的刻度尺压缩表示"。对称量化把中点固定在 8；非对称量化每个 group 有自己的中点，且存储时少存 1（所以解码要 +1）。搞错中点，反量化出来的权重全歪。
- **对你有什么影响**：CPU 上跑 Qwen3.5-35B-A3B-GPTQ-Int4 不再精度崩坏，输出恢复正常。

### 6. [Feat] [NPU] DSA-TAIL: Support PD-transfer and decode-retraction (#50968)
- **代码层面**：`memory_pool_npu.py` 把 `get_cpu_copy` 拆成 `_get_kv_cache_cpu_copy` / `_load_kv_cache_cpu_copy` / `_get_index_cache_layer_buffers` 等，按 layer 和 chunk 分块拷贝，支持 DSA（DeepSeek Sparse Attention）的 index cache 与 KV cache 分离；新增 PD 传输和 decode 回退（retraction）支持。
- **新手概念课堂**：PD 分离（Prefill/Decode 分离）像"餐厅前后厨分工"；decode retraction 像"客人临时取消，要把已占的座位（KV cache）收回"。NPU 上要支持这些，就得让 KV cache 的 CPU 拷贝能按层、按块精细搬运。
- **对你有什么影响**：在 NPU 上部署 DeepSeek 类 DSA 模型时，PD 分离和 decode 回退能正常工作。

### 7. [Perf] deterministic: Launch a fixed number of Triton decode blocks for large split budgets (#47410)
- **代码层面**：`decode_attention.py` 新增 `_MAX_SPLIT_PROGRAMS = 32` 和 `_split_programs()`；`_fwd_grouped_kernel_stage1` 增加 `SPLIT_STRIDE` 常量，当 split 数很大时，每个 (request, head tile) 只启动固定数量 block，循环处理多个 split，而不是每个 split 一个 block。
- **新手概念课堂**：想象 1000 个包裹要分拣。原来"每个包裹一个分拣员"（block），分拣员太多反而调度开销大；现在"固定 32 个分拣员，每人处理多个包裹"，总工作量不变但启动开销小。
- **对你有什么影响**：开启确定性推理 + 大 split budget 时，decode 阶段 kernel 启动开销降低，吞吐提升，且结果仍确定。

### 8. [Feat] diffusion: Serve MiniMax-H3 fl2va and ref2va from one server (--model-variant fl2va+ref2va) (#52023)
- **代码层面**：文档 + 代码支持 `--model-variant fl2va+ref2va`，一个 server 同时加载 FL2VA 和 Ref2VA 两个 transformer；共享 text encoder / VAE / tokenizer；按请求 task 自动路由；启动时检查两个 partition 的 sampling contract 一致；提供内存策略（≥181.7 GiB 双常驻、≥120 GiB 组件 offload、<120 GiB 拒绝启动）。
- **新手概念课堂**：像"一家餐厅同时卖两种菜系"，共用厨房设备（encoder/VAE），但两套主厨（transformer）各做各的；客人点什么菜就派哪位主厨，不用开两家店。
- **对你有什么影响**：部署 MiniMax-H3 时只需一个 server 就能服务 t2va/fl2va/ref2va 全部模式，省一套进程和一份共享权重。

### 9. [Perf] diffusion: No request-path recompiles of the VSA-H3 kernels and the exact scaled residual add (#52224)
- **代码层面**：`scale_shift_triton.py` 把 `numel` 从 `tl.constexpr` 改成运行时参数；`vsa_h3_kernels.py` 给 `_pack_tiles_kernel`、`_untile_kernel`、`_topk_lists_kernel` 加 `do_not_specialize`（含 `N_TILES`、`used`、strides、`out_ptr` 对齐）。
- **新手概念课堂**：Triton 的 `constexpr` 是"编译期常量"，值一变就重编。把随请求变化的量从 constexpr 里拿出来，就像"把每天变的菜单从印刷版改成白板"，不用每天重印。
- **对你有什么影响**：MiniMax-H3 的 VSA attention 在请求路径上不再反复编译 kernel，首 token 延迟更稳定。

### 10. [Perf] diffusion: Build MiniMax-H3's vision-tower cuDNN attention kernels in the warm-up (#52225)
- **代码层面**：`minimax_h3_qwen3vl.py` 新增 `_VISION_ATTENTION_KERNEL_GRIDS`（4 组 (t,h,w)）和 `build_vision_attention_kernels()`，在 warm-up 时用空白图像跑一遍 vision tower，预编译 cuDNN SDPA kernel；`text_encoding.py` 在 `batch.is_warmup` 时调用。
- **新手概念课堂**：cuDNN 会根据输入形状在运行时"现场编译"最优 kernel，第一次很慢。warm-up 时故意用几种典型形状"预热"，就像开车前先热车，避免第一个真实请求卡顿。
- **对你有什么影响**：MiniMax-H3 的第一个带图请求不再因为 cuDNN 编译而明显变慢。

### 11. [Perf] hicache: Fence write_back eviction copies on the forward stream instead of blocking the scheduler (#50469)
- **代码层面**：`cache_controller.py` 新增 `forward_on_fence_stream`；`scheduler.py` 设置它等于 `enable_overlap`；`unified_radix_cache.py` 新增 `_can_fence_write_back_on_streams()` 和 `_fence_write_back_on_streams()`，在满足条件时用 `wait_event` 代替 `writing_check(write_back=True)` 的阻塞等待。
- **新手概念课堂**：原来"把 KV cache 写回 CPU"要等写完才能继续调度（阻塞）；现在在 forward stream 上插一个"等这个 event 完成"的标记，调度器立刻返回处理下一个 batch，拷贝在后台完成。
- **对你有什么影响**：开启 HiCache + overlap 时，KV cache 驱逐不再卡住调度器，高并发下吞吐提升。

### 12. [Feat] glm5_next: Support LayerNorm sequence parallelism (#50906)
- **代码层面**：`layernorm_sp_hook.py` 新增 `_validate_glm5_next_layernorm_sp()`，要求 `attn_cp_size == 1`、`moe_a2a_backend == "none"`、不支持 `--enable-attn-tp-input-scattered`；文档和 `parallel.py` 更新支持列表（Qwen3 dense、Qwen4Exp、Kimi-K3、GLM-5.3-Flash、Muse Spark）。
- **新手概念课堂**：LayerNorm SP 把 LayerNorm/residual 沿序列维切分到多个 GPU，用 reduce-scatter + all-gather 替代 all-reduce，通信量不变但计算更并行。GLM-5.3-Flash 的 attention/FFN 在 TP 组上对 gathered rows 操作，所以某些模式（CP、MoE A2A）会冲突，需要显式禁止。
- **对你有什么影响**：GLM-5.3-Flash 在纯 TP + 无 MoE A2A 场景下可开启 `--enable-layernorm-sp` 提速 prefill。

### 13. [Perf] diffusion/ltx-2.3: Reuse the cond pass's prefix blocks for the STG pass (#51999)
- **代码层面**：`ltx_2.py` 的 `forward` 支持 `ltx2_stg_capture_block` / `ltx2_stg_resume_block`：cond pass 在 block K 处 clone 并返回 hidden states；STG pass 从 block K 直接 resume，跳过 [0, K) 的重算；`denoising.py` 新增 `_ltx2_stg_shared_prefix_block()` 判断能否共享。
- **新手概念课堂**：STG（Spatio-Temporal Guidance）pass 和 cond pass 在前 K 个 block 输入完全相同，只有从第 K 个 block 起才跳过 self-attention。就像两条路线前 K 个路口一样，没必要走两遍，直接"接力"。
- **对你有什么影响**：LTX-2.3 开启 STG guidance 时，denoising 计算量减少，生成更快。

### 14. [Perf] diffusion: replay launches for six more hot Triton wrappers (#52220)
- **代码层面**：给 `fused_bias_silu`、`fused_bias_glu`、`_residual_gate_fp32`、`fused_temb_table_slices`、`matrix_rope` 等 6 个热点 Triton wrapper 加 `@replay_launches` 装饰器。
- **新手概念课堂**：`replay_launches` 是一种"启动参数缓存"机制——相同形状的 kernel 启动直接复用上次的 launch 配置，省去 Python 侧计算 grid/参数的开销。就像"常点的外卖直接一键下单"。
- **对你有什么影响**：diffusion 模型（SANA、WAN、LTX 等）的这些小算子调用开销降低，整体生成速度小幅提升。

### 15. [Fix] diffusion/lora: Snapshot the offloaded weight when committing a merged LoRA as the base (#52226)
- **代码层面**：`lora/linear.py` 的 `commit_merged_as_base` 在非 DTensor 分支调用 `self._materialized_weight_src()` 拿到真实权重（layerwise offload 下 `base_layer.weight` 只是 (1,) 占位符），并设 `_base_is_view = False`；测试新增 offload 场景。
- **新手概念课堂**：layerwise offload 时，GPU 上的 `weight` 只是个"占位牌"，真权重在 CPU 的 manager 里。提交 merged LoRA 为 base 时如果拍的是占位牌，快照就是错的。修复就是"去仓库拿真货再拍照"。
- **对你有什么影响**：LTX-2 等模型在 layerwise offload + LoRA commit 场景下，权重快照正确，不再出现精度异常。

### 16. [Fix] kernels: Guard zero-token launch in grouped_gemma_rmsnorm (#52074)
- **代码层面**：`grouped_gemma_rmsnorm.cuh` 在 `num_blocks = num_tokens * num_groups` 后加 `if (num_blocks == 0) return;`；测试新增 `test_empty_dp_rank`。
- **新手概念课堂**：DP（数据并行）下某些 rank 可能分到 0 个 token。启动 0 个 block 的 kernel 在 CUDA 上是未定义行为（可能报错或挂起）。加个"空批次直接返回"的保护。
- **对你有什么影响**：多卡 DP 推理时，空 rank 不再因 kernel 启动失败而崩溃。

### 17. [Perf] speculative: Fill the DSA MTP draft's prefill KV cache without attention for all but the last token (#46784)
- **代码层面**：`dsa_indexer.py` 新增 `forward_k_only()`（只写 indexer K cache，不做 top-k）；`dsa_indexer_kpool.py` 新增 `forward_k_only()` 走 `_forward_cuda_skip_logits`；`forward_mha.py` 等改动让 MTP draft 的 prefill 阶段对除最后一个 token 外的所有 token 跳过 attention，只填 KV cache。
- **新手概念课堂**：MTP draft 在 prefill 时，前面的 token 只需要"把 K/V 存进 cache"供后续 verify 用，不需要真的算 attention 输出（因为只有最后一个 token 的 hidden state 会被用来预测）。就像"抄笔记"时前面几页只抄目录，最后一页才认真写。
- **对你有什么影响**：DeepSeek 类 DSA 模型的 MTP 投机解码 prefill 更快，首 token 延迟降低。

### 18. [Fix] sampling: Draw filler seeds per request with a secure RNG (#52277)
- **代码层面**：`io_struct.py` 的 `TokenizedGenerateReqInput` 新增 `filler_seed`；`request_receiver.py` 在 `_recv_and_broadcast` 中用 `secrets` 为每个请求生成 `filler_seed`（仅 pp_rank 0 生成，广播给其他 rank）；`schedule_batch.py` 把 `filler_sampling_seed` 改名为 `filler_seed`。
- **新手概念课堂**：批量采样时，没有显式 `sampling_seed` 的请求需要一个"填充种子"来保证随机性。用 `random` 可能撞种子（尤其 fork 后），用 `secrets`（OS CSPRNG）几乎不会。就像发牌时用真随机洗牌，而不是用固定算法。
- **对你有什么影响**：同一 batch 内多个无种子请求的输出不再高度相关，随机性更真实。

### 19. [Refactor] sampling: Build seeds and sampling-mask rows per step; drop SamplingBatchInfo filter/merge (#52270)
- **代码层面**：`SamplingBatchInfo.from_schedule_batch` 被移除，改为 `SamplingBatchInfo(vocab_size=..., device=...)` + 每步调用 `set_seeds_and_mask_rows(reqs)`；`prepare_for_extend`、`prepare_for_prebuilt`、`prepare_for_idle` 都改为构造空 `SamplingBatchInfo`；MLX 后端同步调整。
- **新手概念课堂**：原来采样信息在 batch 构建时"一次性算好并 filter/merge"，现在改成"每步按当前 reqs 重新构建"。好处是 retraction、PD 分离、idle batch 等场景下状态更一致，代码更简单。
- **对你有什么影响**：采样行为在复杂调度场景（retract、PD、idle）下更一致，减少"状态不同步"类 bug。

### 20. [Refactor] sampling: Apply the grammar mask as a pipeline stage (#52192)
- **代码层面**：新增 `sampling/grammar_mask.py` 的 `build_grammar_mask()`；`SamplingBatchInfo.update_regex_vocab_mask()` 和 `grammar_mask` 字段被移除；MLX 的 `tp_worker.py` 改为直接调用 `build_grammar_mask`；`schedule_batch.py` 注释更新说明 `grammars` 由 `ForwardBatch` 派生。
- **新手概念课堂**：grammar mask（约束解码）原来混在 `SamplingBatchInfo` 里，现在抽成独立 pipeline 阶段——"先算 grammar mask，再算 logit_bias，最后合并"。就像把"加调料"从炒菜步骤里拆出来，变成独立的调味台。
- **对你有什么影响**：使用 grammar 约束（JSON schema、正则）时，mask 构建更清晰，CUDA 和 MLX 后端行为一致，减少 VRAM 泄漏风险。

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)