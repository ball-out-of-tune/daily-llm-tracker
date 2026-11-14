# 🚀 vLLM & SGLang 每日更新 — 2026-11-14

> 自动生成于 2026-11-14 07:59 UTC | AI 解读: ✅ 含代码解读

## vllm
## 💡 今日关键词（放在最前面！）

- **关键词：AMD/ROCm 全栈适配**
  - 一句话通俗解释：vLLM 正在把原本主要跑在 NVIDIA GPU 上的功能，一条条搬到 AMD 的 MI300/MI355X 上，包括推理、CI 测试、KV 缓存、MoE 算子。
  - 为什么社区在关注：今天 20 个 commit 里有 6 个带 `[ROCm]` 标签，覆盖 CI 队列重排、KV 缓存策略、MoRIIO 分离式推理、稀疏 MLA 测试修复。这说明 AMD 不再是"能跑就行"，而是要"跑得和 NVIDIA 一样好"，异构计算真正进入工程深水区。
  - 对新手意味着什么：如果你只会 CUDA，建议开始了解 ROCm/HIP 和 AITER 这套 AMD 算子库——这是目前人才缺口大、竞争相对小的方向。

- **关键词：Kimi-K3.1-Pro 推理配方（Recipe）**
  - 一句话通俗解释：vLLM 仓库里有一批 YAML 文件，专门记录"某个大模型在某款硬件上怎么配置参数才能跑出最佳性能"，今天连续两个 commit 在打磨 Kimi-K3.1-Pro 在 MI355X 上的配方。
  - 为什么社区在关注：2.8T 参数的 MoE 多模态模型 + 1M 上下文，是当前最前沿的推理场景。配方化（recipe）意味着大模型部署从"手调参数"走向"可复现的工程资产"。
  - 对新手意味着什么：学会读这些 YAML 配方，比读论文更能快速理解真实生产环境的调优逻辑（TP、DCP、KV offload、spec decoding 怎么组合）。

- **关键词：投机解码（Speculative Decoding）与注意力元数据**
  - 一句话通俗解释：用一个"小模型"先猜几个 token，大模型再一次性验证，从而加速生成。今天有 commit 专门优化"当 prefill 阶段只需要写 KV 缓存时，不要构建 draft 模型的注意力元数据"。
  - 为什么社区在关注：投机解码是当前推理加速的头号手段，但它的工程复杂度很高——元数据、CUDA graph、KV 布局都要精细配合，稍有不慎就是性能回退或崩溃。
  - 对新手意味着什么：想进推理优化方向，投机解码是必学模块，建议从"draft model + target model 验证"的基本流程入手。

- **关键词：GC（垃圾回收）与启动性能**
  - 一句话通俗解释：Python 的垃圾回收器会定期扫描所有对象，启动阶段对象多、扫描慢，今天有 commit 通过"先回收再冻结"来加速初始化。
  - 为什么社区在关注：大模型服务启动动辄几分钟，启动时间直接影响弹性扩缩容和故障恢复速度。这类"看起来不性感"的优化，实际收益很大。
  - 对新手意味着什么：性能优化不只有 kernel 和算子，Python 层面的 GC、内存分配、对象生命周期同样是战场。

- **关键词：多模态 CUDA Graph 与 padding 逻辑**
  - 一句话通俗解释：多模态模型的视觉编码器要用 CUDA Graph 加速，但不同 batch 大小要对齐，今天修的是"对齐时 `cu_seqlens` 应该重复最后一个值而不是补零"。
  - 为什么社区在关注：多模态推理正在成为主流负载，CUDA Graph 是降低 CPU 开销的关键手段，但 padding 逻辑错了会导致结果错误，属于"隐蔽 bug"。
  - 对新手意味着什么：CUDA Graph 是高性能推理的必修课，理解"固定形状 + 缓冲区复用"的思想对做任何推理框架都有帮助。

---

## 🧪 重要知识点 & 动手实验

- **知识点 1：投机解码中的"只写 KV 的 prefill"（Commit 6）**
  - **是什么**：在投机解码里，draft 模型有时只需要把 KV 缓存写进去（context-K/V-only prefill），不需要真正算 attention 输出，这时就不该构建 draft 的注意力元数据。
  - **为什么重要**：如果无脑构建元数据，会浪费 workspace 显存、触发不必要的 CUDA graph 捕获限制，甚至导致 kernel 崩溃。真实场景中这类"多余计算"会吃掉吞吐。
  - **动手试试**：
    ```bash
    # 先看测试怎么写的
    git clone https://github.com/vllm-project/vllm.git
    cd vllm
    git show 165156b5 --stat
    # 跑相关单测（需要 GPU）
    pytest tests/v1/worker/test_attn_utils.py -k "no_attn_metadata_layers" -v
    ```
  - **预期结果**：测试通过，你会看到断言"这些层保留了 kernel block 布局但没有 builder"——即缓存布局不变、元数据为空。理解"布局不变但元数据可省"这个设计思想。

- **知识点 2：Python GC 的 freeze 机制（Commit 11）**
  - **是什么**：`gc.freeze()` 把所有当前存活对象移到"永久代"，之后 GC 扫描时直接跳过它们。今天改成"先 `gc.collect()` 再 `gc.collect(0)` 再 `gc.freeze()`"。
  - **为什么重要**：vLLM 启动时会加载大量模型权重和配置对象，如果这些对象还留在年轻代，每次 GC 都要扫描，启动和首次推理都会变慢。
  - **动手试试**：
    ```python
    import gc, time

    class Obj:
        pass

    # 模拟启动时创建大量对象
    keep = [Obj() for _ in range(1_000_000)]

    t0 = time.perf_counter()
    gc.collect()
    print("collect all:", time.perf_counter() - t0)

    gc.freeze()
    print("frozen count:", gc.get_freeze_count())

    t0 = time.perf_counter()
    gc.collect()
    print("collect after freeze:", time.perf_counter() - t0)
    ```
  - **预期结果**：freeze 之后的 `gc.collect()` 明显更快，`get_freeze_count()` 返回约 100 万。你就理解了为什么 vLLM 要在加载权重前 freeze。

- **知识点 3：MoE 中的 padding 行掩码（Commit 16）**
  - **是什么**：数据并行（DP）时，每个 rank 的 batch 大小可能不同，需要 padding 到统一形状。这些 padding 行不应该参与 MoE 路由（否则会分到专家、浪费算力），今天让 grouped top-k 默认把 padding 行的 expert id 设为 -1。
  - **为什么重要**：MoE 模型（如 DeepSeek、Kimi）的专家路由是性能热点，padding 行如果被路由，会造成"假负载"，拖慢整个 batch。
  - **动手试试**：
    ```bash
    # 看测试逻辑
    git show fee96868 -- tests/kernels/moe/test_grouped_topk.py
    # 跑测试（需要 CUDA/ROCm）
    pytest tests/kernels/moe/test_grouped_topk.py -k "masks_padding" -v
    ```
  - **预期结果**：测试断言 padding 行的 `topk_ids == -1`、`topk_weights == 0`。你会理解"用 -1 表示丢弃"这个约定。

- **知识点 4：音频解码的后备链（Commit 4）**
  - **是什么**：vLLM 解码音频默认用 torchcodec（底层 FFmpeg），但某些损坏的 MP3 帧 FFmpeg 会直接报错，而 soundfile 用的 mpg123 能"跳过坏帧继续解"。今天改成"torchcodec 解码失败时，只回退到 soundfile"。
  - **为什么重要**：多模态服务里，用户上传的音频质量参差不齐，一个坏文件不应该让整个请求 500。容错设计是生产系统的基本功。
  - **动手试试**：
    ```python
    from vllm.multimodal.media.audio import load_audio
    import io

    # 正常音频
    with open("test.wav", "rb") as f:
        data = f.read()
    audio = load_audio(io.BytesIO(data), backend="auto")
    print(audio.shape)

    # 截断的音频（模拟损坏）
    broken = data[: len(data) // 2]
    try:
        audio = load_audio(io.BytesIO(broken), backend="auto")
        print("fallback worked:", audio.shape)
    except Exception as e:
        print("failed:", type(e).__name__, e)
    ```
  - **预期结果**：正常文件成功；截断文件可能触发回退到 soundfile 后成功，或抛出 `_CorruptAudioError`。你会理解"分级回退"的错误处理模式。

- **知识点 5：Rust 前端的优雅关闭与请求排空（Commit 5）**
  - **是什么**：vLLM 有一个 Rust 写的前端（替代 Python 的 API server）。关闭时原来的顺序是"先杀引擎，再等 API server"，导致还在处理的请求失败。今天改成"先等 API server 排空请求，再用剩余时间关引擎"。
  - **为什么重要**：滚动升级、K8s 优雅下线都依赖"排空"机制。顺序错了，用户会看到 5xx 错误。
  - **动手试试**：
    ```bash
    # 看变更的核心逻辑
    git show ec94b6a7 -- rust/src/cmd/src/main.rs
    # 跑 Rust 测试（需要 Rust 工具链）
    cd rust && cargo test -p vllm-managed-engine
    ```
  - **预期结果**：测试验证"只 SIGTERM 引擎的 launcher 进程，而不是整个进程组"，这样 in-flight 请求能继续完成。你会理解"进程组信号"和"优雅关闭预算"的概念。

---

## 🔥 重要更新

- **Commit 17（NIXL KV Connector 支持 PP prefill 拉取）**：这是今天最大的功能 commit，让分离式推理（prefill/decode 分离）支持流水线并行的 prefill 阶段，并兼容混合 KV 缓存管理器。对做大规模分布式推理的团队影响巨大。
- **Commit 13（Kimi-K3 图像 metadata-only 输入）**：支持 EPD（编码-预填充-解码分离）场景下只传图像元数据，避免重复传输图像，是多模态分离式推理的关键能力。
- **Commit 6（投机解码不构建多余元数据）**：修的是性能与正确性问题，对开启投机解码的用户直接受益，避免显存浪费和潜在崩溃。
- **Commit 11（启动 GC 优化）**：看似小改动，但影响所有用户的启动时间，属于"普惠型"性能优化。

---

## 📋 逐条解读

### 1. [Recipe] Kimi-K3.1-Pro: MI355X AgentX preset covering all six InferenceX#5043 concurrencies (#67600)
- **代码层面**：修改 `recipes/models/moonshotai/Kimi-K3.1-Pro.yaml`。把原来只覆盖 concurrency 48 的 `mi355x_agentx` preset，扩展成覆盖 concurrency 1/4/16/24/32/48 六个点的完整配方，每个点有不同的 draft token 数（7/5/4/4/3/3）和是否开启 low-latency decode、KV offload。
- **新手概念课堂**：Recipe 就像"菜谱"——告诉你用什么锅（硬件）、放多少料（参数）、火候多大（并发数）。Draft token 数就像"猜题数量"，猜得多省时间但猜错浪费，猜得少稳但慢。
- **对你有什么影响**：如果你在 MI355X 上部署 Kimi-K3.1-Pro，直接照抄这个 YAML 就能复现 InferenceX 基准测试的成绩，不用自己摸参数。

### 2. [Bugfix][Multimodal] Size the MOSS-Audio profiling dummy to its audio budget (#67474)
- **代码层面**：修改 `vllm/model_executor/models/moss_audio.py` 的 `get_dummy_mm_data`，把 dummy 音频长度从硬编码的 16000 改成 `mel_sr * DEFAULT_MAX_AUDIO_SECONDS`，并新增测试验证 dummy 音频占满预算。
- **新手概念课堂**：vLLM 启动时会用"假数据"跑一遍推理来估算显存（profiling）。假数据太小，估算的显存就偏小，实际推理时就会 OOM。就像用小车试停车位，结果来了大货车停不进去。
- **对你有什么影响**：用 MOSS-Audio 模型时，显存估算更准确，减少"启动正常、一跑就 OOM"的问题。

### 3. [Bugfix][Rust Frontend] Return 400 instead of 500 when tool parsing is disabled (#63580)
- **代码层面**：修改 `rust/src/chat/src/error.rs`，把 `ParserDisabled` 错误归类为客户端错误（400），并新增 gRPC 和 HTTP 两处测试。
- **新手概念课堂**：400 是"你请求错了"，500 是"服务器错了"。用户传了 tools 但服务端没开 tool parser，这是用户的配置问题，应该返回 400 让用户自己改，而不是 500 让用户以为服务器挂了。
- **对你有什么影响**：用 Rust 前端时，错误信息更准确，排障更快。

### 4. [Bugfix][Audio] Fall back to soundfile when torchcodec fails to decode (#67105)
- **代码层面**：修改 `vllm/multimodal/media/audio.py`，新增 `_CorruptAudioError` 异常类，`load_audio(backend="auto")` 在 torchcodec 解码失败时只回退到 soundfile（不再回退到 PyAV）。
- **新手概念课堂**：不同解码器对损坏文件的容忍度不同。FFmpeg 严格，遇到坏帧就报错；mpg123（soundfile 用的）宽松，能跳过坏帧。就像两个翻译，一个遇到生词就卡住，一个能猜着往下翻。
- **对你有什么影响**：上传质量不佳的音频时，请求成功率更高。

### 5. [Bugfix][Rust Frontend] SIGTERM only the managed engine's launcher so in-flight requests drain (#67282)
- **代码层面**：修改 `rust/src/cmd/src/main.rs` 和 `rust/src/managed-engine/src/process.rs`。关闭顺序改为"先等 API server 排空，再用剩余时间关引擎"，并且只向引擎的 launcher 进程发 SIGTERM，不向整个进程组发。
- **新手概念课堂**：进程组就像一个班级，向进程组发信号等于"全班放学"，正在考试的学生也被赶走。只向 launcher 发信号等于"只通知班主任，让学生考完再走"。
- **对你有什么影响**：滚动升级或重启服务时，正在处理的请求不会被中断，用户体验更平滑。

### 6. [Spec Decode][MRV2] Build no draft attention metadata on a context-K/V-only prefill (#67291)
- **代码层面**：修改注意力元数据构建逻辑，新增 `metadata_builders` 列表，在 context-K/V-only prefill 时不构建 draft 的元数据；同时更新多个测试。
- **新手概念课堂**：投机解码里 draft 模型有时只负责"写 KV 缓存"，不负责"算输出"。就像秘书只负责整理资料，不负责做决策，那就不需要给秘书准备决策会议室。
- **对你有什么影响**：开启投机解码时，显存占用更低，prefill 阶段更快。

### 7. [ROCm] Keep the extensible KV cache for fd-exporting KV connectors (#67577)
- **代码层面**：修改 `vllm/config/vllm.py` 的 `_resolve_extensible_kv_cache`，把判断条件从 `needs_single_chunk()` 改成 `KVCacheAccess.REGISTERS in kv_cache_access`，让通过文件描述符（fd）导出 KV 缓存的连接器在 ROCm 上仍能使用可扩展 KV 缓存。
- **新手概念课堂**：KV 缓存是"草稿纸"，可扩展缓存意味着草稿纸可以按需加页。ROCm 上原来因为 UCX/RDMA 注册限制不敢用，现在发现 fd 导出不涉及注册，就可以放心用了。
- **对你有什么影响**：在 AMD GPU 上做 KV 缓存跨进程传输时，显存利用更灵活。

### 8. [BugFix] Fix Mistral3 multimodal offline example (#43638)
- **代码层面**：修改 `examples/generate/multimodal/vision_language_offline.py`，给 Mistral3 加上 `tokenizer_mode="mistral"`、`config_format="mistral"`、`load_format="mistral"`，并改用 `llm.chat()` 接口而非手拼 prompt。
- **新手概念课堂**：Mistral 有自己的模型格式（不是标准 HF 格式），必须显式告诉 vLLM "用 Mistral 的方式加载"。就像打开一个特殊压缩包，要用对应的解压工具。
- **对你有什么影响**：跑 Mistral3 多模态示例时不再报错，且用标准 chat 接口更符合实际用法。

### 9. [ROCm][CI] Retarget 36 MI300 1 GPU jobs to the mi300_dpx queue (#67308)
- **代码层面**：修改 `.buildkite/test-amd.yaml`，把 36 个 MI300 单 GPU 测试任务的 `agent_pool` 从 `mi300_1` 改为 `mi300_dpx`，标签也相应更新。
- **新手概念课堂**：CI 队列就像食堂窗口，`mi300_1` 窗口排队太长，把一部分人分流到 `mi300_dpx` 窗口，整体吞吐就上去了。
- **对你有什么影响**：普通用户无感知，但 AMD CI 反馈更快，间接让 ROCm 功能迭代更稳。

### 10. [ROCm][CI] Stub dcp_world_size in the merged-cache sparse MLA test (#67555)
- **代码层面**：修改 `tests/kernels/attention/test_rocm_triton_attn_dsv4.py` 补上 `impl.dcp_world_size = 1`，并在 `tests/v1/test_outputs.py` 里加了 `torch.accelerator.synchronize()` 再读 CPU 张量。
- **新手概念课堂**：`dcp_world_size` 是 DCP（解码上下文并行）的并行度，测试里没设置就默认不存在，代码一读就崩。同步则是因为 GPU 是异步的，不等它算完就读 CPU 数据会读到旧值。
- **对你有什么影响**：ROCm CI 更稳定，减少"莫名其妙失败"的测试。

### 11. [Perf] Optimize Garbage Collection at Init Time (#64542)
- **代码层面**：修改 `vllm/utils/gc_utils.py` 的 `freeze_gc_heap()`，把"分三代 collect 再 freeze"改成"先 `gc.collect()` 再 `gc.collect(0)` 再 freeze"；并在 `vllm/v1/worker/gpu_worker.py` 的 `init_device` 里提前调用 freeze。
- **新手概念课堂**：GC 就像定期打扫房间，启动时房间堆满东西（模型对象），打扫很慢。`gc.freeze()` 相当于把"确定不会再动"的东西贴上"勿扫"标签，之后打扫就快了。
- **对你有什么影响**：服务启动更快，首次推理延迟更低。

### 12. [ROCm][Disagg][CI] Give Wide-EP headless nodes the KV config (#65098)
- **代码层面**：修改 `.buildkite/amd-disagg/vllm_disagg.sh`，让 headless（无 API server）节点也拿到 `--kv-transfer-config`，并补上 `remote_hosts`、`remote_dp_size_local` 等字段；同时改了几个 MoRIIO 测试的初始化。
- **新手概念课堂**：分离式推理里，prefill 和 decode 是两拨人，headless 节点是"只干活不接客"的工人。工人也需要知道"KV 往哪送"，否则干完活不知道给谁。
- **对你有什么影响**：AMD 上的分离式推理（Wide-EP）配置更完整，部署更省心。

### 13. [Feature][EPD] Support Kimi-K3 image metadata-only inputs (#66925)
- **代码层面**：新增 EC（Encoder Cache）传输的元数据收集与占位符解析逻辑，支持只传图像尺寸等元数据而不传图像本身；新增大量测试。
- **新手概念课堂**：EPD 是"编码-预填充-解码"三段分离。图像编码很贵，如果预填充节点只需要知道"图像多大、占几个 token"，就不用把整张图传过去，像寄快递只填单号不寄实物。
- **对你有什么影响**：用 Kimi-K3 做多模态分离式推理时，网络传输量大幅降低。

### 14. [Bugfix][Multimodal] Repeat-last default padding for `cu_seqlens` in multimodal CG (#65453)
- **代码层面**：修改 `vllm/v1/cudagraph/encoder_cudagraph.py`（推测），新增 `padding_logics` 机制，`cu_seqlens`/`cu_window_seqlens` 默认用"重复最后一个值"填充，而非补零；更新设计文档和测试。
- **新手概念课堂**：`cu_seqlens` 是"累积序列长度"，比如 `[0, 5, 9]` 表示两段长度 5 和 4。补零会变成"多了一段长度 0 的序列"，语义错乱；重复最后值 `[0, 5, 9, 9]` 表示"多了一段空序列"，语义正确。
- **对你有什么影响**：多模态模型开 CUDA Graph 时，结果更正确，避免"偶发输出乱码"。

### 15. [Recipe] Kimi-K3.1-Pro: MI355X kimi-k31-6d74a15 image, concurrency-48 with 3 draft tokens, align with InferenceX#5043 (#67487)
- **代码层面**：修改同一个 Kimi-K3.1-Pro YAML，更新 AMD 镜像 tag 到 `kimi-k31-6d74a15`，新增 `VLLM_ROCM_DCP_VUP_RS`、`VLLM_ROCM_DSA_MULTISTREAM`、`VLLM_ROCM_MEM_HEADROOM_GIB` 等环境变量，并重写 runbook 说明。
- **新手概念课堂**：这是 Commit 1 的"前传"——先把 concurrency 48 这个点调好（Commit 15），再扩展成六个点（Commit 1）。`VUP_RS` 是"先算 v_up 再做 reduce-scatter"，`MULTISTREAM` 是"用第二条流水线并行跑 indexer"。
- **对你有什么影响**：MI355X 上跑 Kimi-K3.1-Pro 的性能进一步提升，且配方注释详细，适合学习调优思路。

### 16. [Perf][MoE] Mask DP padding rows in grouped top-k routing by default (#66799)
- **代码层面**：修改 `vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py`（推测），让 `GroupedTopKRouter` 默认对 padding 行输出 expert id = -1；新增测试覆盖 fused/non-fused、CUDA/ROCm 组合。
- **新手概念课堂**：MoE 路由就像"把学生分到不同班级"，padding 行是"空座位"，不应该被分班。用 -1 标记"这个座位没人"，后续算子就会跳过它。
- **对你有什么影响**：MoE 模型在数据并行场景下吞吐更高，尤其是 batch 里有 padding 时。

### 17. [KV Connector][NIXL] Pull KV from pipeline-parallel prefill stages, with the hybrid KV cache manager (#65336)
- **代码层面**：大幅修改 NIXL KV 连接器（`vllm/distributed/kv_transfer/kv_connector/v1/nixl/`），新增 `_add_remote_stage`、`_remote_stages` 等机制，支持从 PP prefill 各阶段拉取 KV；更新设计文档 `docs/design/nixl_kv_push_connector.md` 和兼容性文档。
- **新手概念课堂**：PP（流水线并行）把模型切成几段放在不同 GPU 上，prefill 时每段只算一部分层的 KV。decode 节点要拉 KV 时，得知道"第 3 层的 KV 在第 2 段 GPU 上"，不能一股脑拉。这个 commit 就是建立"分层索引"。
- **对你有什么影响**：做大规模分离式推理（尤其 PP + 混合 KV 缓存）时，NIXL 连接器终于能用了。

### 18. [DeepSeek-V4] Warm the DeepGEMM kernels DeepSeek-V4/V4.1 JIT-compile at runtime (#64827)
- **代码层面**：新增 `vllm/model_executor/warmup/deep_gemm_warmup.py` 中的 FP8 einsum 预热逻辑，提前 JIT 编译 DeepGEMM 的 `o_proj` 相关 kernel；新增大量测试。
- **新手概念课堂**：DeepGEMM 是 DeepSeek 自研的矩阵乘库，首次调用时要 JIT 编译（现场编译），慢且可能卡住。预热就是"提前把菜做好"，用户点单时直接上菜。
- **对你有什么影响**：跑 DeepSeek-V4/V4.1 时，首次推理不再卡顿，延迟更稳定。

### 19. [Transformers backend] Default decoder pooling to LAST and honour `is_causal=False` (#67455)
- **代码层面**：修改 `vllm/config/model.py` 和 `vllm/model_executor/models/transformers/base.py`、`utils.py`。对 Transformers 后端的 pooling 模型，默认用 LAST token pooling；并识别 `is_causal=False` 的配置（如 gte-Qwen2），改用 `EncoderOnlyAttention`。
- **新手概念课堂**：Pooling 是"把一串向量压成一个向量"用于分类/检索。LAST 是"取最后一个 token"，MEAN 是"取平均"。gte-Qwen2 虽然叫 Qwen（decoder 架构），但配置里说"我是双向注意力"，所以要按 encoder 处理。
- **对你有什么影响**：用 Transformers 后端跑 embedding 模型时，结果更符合预期，不再需要手动指定 pooling 方式。

### 20. [ROCm][CI] Cover test_areas gaps on AMD mirrors (#67317)
- **代码层面**：修改 `.buildkite/test_areas/distributed.yaml` 和 `models_basic.yaml`，在 AMD 镜像上补充 `test_cpu_all_reduce.py`、`test_pipeline_partition.py`、`test_deepseek_v41_pcp.py` 等测试。
- **新手概念课堂**：CI 就像"体检项目清单"，之前 AMD 体检漏了几项，现在补上。`test_areas` 是按功能域划分的测试集合。
- **对你有什么影响**：AMD 上的分布式和 DeepSeek-V4.1 功能更有保障，减少"在 NVIDIA 上好好的，到 AMD 就崩"的情况。

## sglang
## 💡 今日关键词

- **关键词：HiCache / KV 缓存分层存储**
  - 一句话通俗解释：把显存里放不下的 KV Cache 挪到 CPU 内存甚至磁盘/远端存储，需要时再搬回来，像给 GPU 加了个"外挂内存"。
  - 为什么社区在关注：今天 20 个 commit 里有 6 个以上直接改 HiCache、buffer 模式、storage key、prefetch 调度，说明**长上下文 + 大模型推理的内存墙**是当前最痛的瓶颈，社区在拼命打磨"分层缓存"的每个细节（key 命名、所有权校验、异步拷贝）。
  - 对新手意味着什么：如果你在学推理系统，**KV Cache 管理**是含金量极高的方向，面试常考、工程上真能省几倍成本。建议从理解 PagedAttention + 前缀缓存开始。

- **关键词：多模态处理器重构（MM refactor 系列）**
  - 一句话通俗解释：把"图片/视频/音频怎么变成模型输入"这套代码拆成可复用的模块（Mixin、transport、preprocessing）。
  - 为什么社区在关注：这是 9 步重构的第 3、4 步，说明 SGLang 的多模态代码已经从"能跑"进入"要维护、要扩展"的阶段，**工程债偿还**正在进行。
  - 对新手意味着什么：学会"看大型重构 PR"是进阶技能——不要逐行看 diff，而是看**新文件边界**（mixin 抽了什么、transport 抽了什么），这比读教程更能学到架构设计。

- **关键词：Rust 前端 / Rust tree core**
  - 一句话通俗解释：SGLang 把 HTTP 服务层和基数树（radix tree）核心用 Rust 重写，追求更低延迟和更高吞吐。
  - 为什么社区在关注：今天有 4 个 commit 是 rust-server（HTTP/2、请求 ID、多模态参数）和 rust tree core（联合驱逐），说明**Python 写业务、Rust 写热路径**的混合架构正在成为推理框架标配。
  - 对新手意味着什么：Rust 在 AI infra 里的岗位需求在涨。哪怕不写 Rust，也要能读懂"为什么这里要用 Rust"——通常是**延迟敏感 + 并发密集**。

- **关键词：调度策略精细化（shortest-prefill-first + aging、health check）**
  - 一句话通俗解释：请求排队时"谁先上 GPU"的规则，现在细到"短 prefill 优先 + 防饿死老化 + 健康探针插队"。
  - 为什么社区在关注：说明推理服务已从"能推理"进入"**SLA 可承诺**"阶段——TTFT、排队公平性、探针不卡死都是生产环境刚需。
  - 对新手意味着什么：调度是**算法 + 系统工程**的交叉点，门槛不高但很能体现功力。建议动手写个简单的优先队列调度器练手。

- **关键词：性能微优化（查表 SiLU、FlashInfer FA2、异步 D2H）**
  - 一句话通俗解释：把 exp() 换成查表、把注意力换成更快后端、把同步拷贝改成异步。
  - 为什么社区在关注：这些是"最后 10% 性能"的争夺，说明主流路径已优化得差不多，社区开始啃**ALU 瓶颈、kernel launch、同步点**这类硬骨头。
  - 对新手意味着什么：这类 commit 是学习 **CUDA/GPU 性能分析**的最佳教材——每个都对应一个具体的 profiler 现象（ALU-bound、stall on sync）。

---

## 🧪 重要知识点 & 动手实验

- **知识点 1：KV Cache 分层存储与 storage key 命名（对应 Commit 7）**
  - **是什么**：HiCache 把 KV 存到远端时，需要一个"文件名"来区分不同模型、不同数据格式。这次修复把 **逻辑 KV dtype**（比如 FP8 E4M3 vs E5M2）加进了 key，因为两者底层都是 uint8，不加就会读错数据。
  - **为什么重要**：如果 key 不区分 dtype，A 机器用 FP8-E4M3 写入、B 机器用 FP8-E5M2 读取，**不会报错但结果全错**——这是最恐怖的一类 bug（静默数据损坏）。生产环境换量化格式时必须靠 key 隔离。
  - **动手试试**：
    ```python
    import torch
    from sglang.srt.utils import normalize_dtype  # 若可导入
    # 模拟 key 拼接逻辑
    def storage_key(model_name, tag, dtype):
        parts = [model_name, tag, f"dtype_{dtype}"]
        return "-".join(p for p in parts if p)
    print(storage_key("llama3", None, "float8_e4m3fn"))
    print(storage_key("llama3", None, "float8_e5m2"))
    ```
  - **预期结果**：两个 key 不同（`llama3-dtype_float8_e4m3fn` vs `llama3-dtype_float8_e5m2`），说明即使底层 buffer 一样，key 也能区分。

- **知识点 2：调度策略中的"老化"（aging）防饿死（对应 Commit 18）**
  - **是什么**：shortest-prefill-first 会让短请求一直插队，长请求可能永远排不上（饿死）。"老化"就是**每等一会儿，请求的优先级就提升一点**，保证最终能被执行。
  - **为什么重要**：任何"贪心优先"策略都有饿死风险。操作系统调度、数据库查询调度、推理请求调度都靠 aging 兜底。不懂这个，线上就会出现"某些请求永远超时"的诡异现象。
  - **动手试试**：
    ```python
    import heapq
    # 模拟：短 prefill 优先 + aging_factor
    aging = 0.5
    processed = 0
    reqs = [("long", 100), ("short", 5), ("long2", 80)]
    heap = []
    for rid, work in reqs:
        heapq.heappush(heap, (work, rid))
    # 每处理一个 token，等待请求的 rank 降低
    for _ in range(10):
        processed += 1
        heap = [(max(0, w - aging), rid) for w, rid in heap]
        heapq.heapify(heap)
    print(heap[:3])
    ```
  - **预期结果**：长请求的 work 值逐渐下降，最终能排到前面——这就是 aging 的效果。

- **知识点 3：CUDA 异步拷贝与"同步点"消除（对应 Commit 17）**
  - **是什么**：GPU 和 CPU 之间搬数据（D2H/H2D）如果同步等，CPU 会空转。异步拷贝用 `cudaMemcpyAsync` + `Event`，让 CPU 继续干活，等真正需要结果时再 `event.synchronize()`。
  - **为什么重要**：推理 pipeline 里任何一次隐式 `.cpu()` 或 `torch.equal()` 都可能触发全设备同步，**把并发彻底打回串行**。这是新手最容易踩的性能坑。
  - **动手试试**：
    ```python
    import torch, time
    x = torch.randn(1000, 1000, device="cuda")
    # 同步方式
    t0 = time.time()
    for _ in range(100):
        _ = x.cpu()
    torch.cuda.synchronize()
    print("sync:", time.time() - t0)
    # 异步方式（用 pinned memory + non_blocking）
    buf = torch.empty_like(x, device="cpu", pin_memory=True)
    t0 = time.time()
    for _ in range(100):
        buf.copy_(x, non_blocking=True)
    torch.cuda.synchronize()
    print("async:", time.time() - t0)
    ```
  - **预期结果**：async 版本通常明显更快（尤其在大 tensor 上），因为 CPU 不用每次等 GPU。

- **知识点 4：用查表替代超越函数（对应 Commit 4）**
  - **是什么**：SiLU 激活需要算 `x * sigmoid(x)`，含 `exp()` 和除法，很耗 ALU。由于 bf16/fp16 只有 16 位，输入总共 65536 种可能，**直接预计算一张 64K 的表**，运行时查表即可。
  - **为什么重要**：这是"用空间换时间"的经典手法。在 power-capped（功耗受限）的 GPU 上，ALU 是瓶颈，查表能显著提速。理解这个思路，你就懂了为什么很多 kernel 会预计算常量。
  - **动手试试**：
    ```python
    import torch, time
    x = torch.randn(1<<20, dtype=torch.float16, device="cuda")
    # 直接算
    t0 = time.time()
    for _ in range(100):
        y = x * torch.sigmoid(x)
    torch.cuda.synchronize(); print("compute:", time.time() - t0)
    # 查表（用 fp16 的 bit pattern 做索引）
    table = torch.nn.functional.silu(torch.arange(65536, dtype=torch.float16, device="cuda").view(torch.int16).view(torch.float16))
    bits = x.view(torch.int16).to(torch.int64) & 0xFFFF
    t0 = time.time()
    for _ in range(100):
        y = table[bits]
    torch.cuda.synchronize(); print("lookup:", time.time() - t0)
    ```
  - **预期结果**：lookup 版本在 ALU 受限时更快；注意查表结果与直接算在 fp16 精度下应完全一致（因为 SiLU 是确定性函数）。

- **知识点 5：FastAPI 嵌套路由的鉴权解析（对应 Commit 5）**
  - **是什么**：FastAPI 的 `include_router` 会把子路由"嵌套"进 `app.routes`，但 `route.matches()` 对嵌套路由不直接工作。这个 commit 写了个递归 `_iter_leaf_routes`，剥掉 prefix 再匹配，确保每个 endpoint 的鉴权等级被正确识别。
  - **为什么重要**：鉴权等级搞错 = **安全漏洞**（本应 admin-only 的接口变成公开）。新手常以为"装饰器加了就行"，但路由嵌套会让匹配失败，静默降级为 NORMAL。
  - **动手试试**：
    ```python
    from fastapi import FastAPI, APIRouter
    app = FastAPI()
    sub = APIRouter(prefix="/admin")
    @sub.get("/secret")
    def secret(): return "ok"
    app.include_router(sub)
    # 观察 app.routes 的结构
    for r in app.routes:
        print(type(r).__name__, getattr(r, "path", None),
              getattr(r, "original_router", None) is not None)
    ```
  - **预期结果**：你会看到 `Mount` 或带 `original_router` 的嵌套结构——这正是为什么需要递归解析。

---

## 🔥 重要更新

1. **Commit 7（HiCache storage key 加 dtype）**：修复了一个**静默数据损坏**级 bug——FP8 E4M3 和 E5M2 共享 uint8 buffer，不加 dtype 到 key 会读错 KV。任何用 HiCache + 量化的用户都必须关注。
2. **Commit 15（health check 插队响应）**：之前健康探针排在长 prefill 后面会被"饿死"，导致 K8s/LB 误判实例挂了。这是**生产可用性**的关键修复。
3. **Commit 4（SiLU 查表）**：diffusion 模型在 sm_10x（Blackwell）上 ALU-bound，查表是显著的性能提升，且**输出字节完全一致**（不是近似），非常干净的优化。
4. **Commit 14（Rust tree core 联合 Full/SWA 驱逐）**：`joint_lru` 策略之前在 Rust tree core 上被降级为普通 `lru`，现在原生支持了——混合注意力模型（Full + SWA）的内存效率直接受益。

---

## 📋 逐条解读

### 1. [Test] ep: Temporarily disable test_mooncake_ep_small on the Mooncake EP JIT nvcc spawn flake
- **代码层面**：`test/registered/ep/test_mooncake_ep_small.py` 里给测试注册加了个 `disabled="..."` 字段，附带原因和 issue 链接。
- **新手概念课堂**：CI 里的"flake"指**偶发失败的测试**（flaky test）。就像考试时偶尔抽到一道有歧义的题，不是你不会，是题有问题。临时禁用是标准做法，但**必须写明原因和恢复条件**，否则就变成永久掩盖。
- **对你有什么影响**：普通用户无感。但如果你在跑 Mooncake EP 相关测试，会看到它被 skip。

### 2. [Refactor] multimodal: [MM refactor 4/9] Split feature transport and preprocessing placement out of the base processor
- **代码层面**：`base_processor.py` 里把 CUDA IPC 传输逻辑、memory pool 逻辑、NPU 图像预处理 patch 等抽到新文件 `feature_transport.py` 和 `serving_preprocessing.py`，base processor 只保留编排。
- **新手概念课堂**：**Mixin / 组合优于继承**。原来 base processor 是个"上帝类"，什么都干；现在拆成"负责传输的"和"负责预处理的"两个独立模块，就像把瑞士军刀拆成几把专用刀，每把更好维护。
- **对你有什么影响**：行为不变，但如果你写过自定义多模态 processor，未来要按新接口组织代码。

### 3. [Perf] diffusion/cosmos: optimize bf16 fa
- **代码层面**：新增 `flashinfer_fa2.py` 后端，用 FlashInfer 的 `single_prefill_with_kv_cache` 处理 dense BSHD 注意力（含 GQA），避免 stack/copy 输出。
- **新手概念课堂**：**注意力后端可插拔**。就像手机相机可以选不同 App，模型也能选 FA2/FA3/FlashInfer 等不同 kernel。GQA（分组查询注意力）是"多个 Q 头共享一个 KV 头"，省显存但 kernel 要特殊处理。
- **对你有什么影响**：Cosmos 等 diffusion 模型在 bf16 下更快，可通过 `--attention-backend flashinfer_fa2` 启用。

### 4. [Perf] diffusion: Look up the rounded SiLU of the gated activation in a table on sm_10x
- **代码层面**：`activation.cuh` 新增 `silu_and_mul_rounded_table_kernel`，把 64K 个 16-bit 输入的 SiLU 结果预计算到 shared memory，运行时查表。仅在 sm_10x（Blackwell）且非 ROCm 时启用。
- **新手概念课堂**：**Shared memory 查表**。GPU 的 shared memory 比 global memory 快几十倍，把表拷进去后每个线程查表几乎零延迟。代价是每个 block 要先花时间填表（但摊薄到大量元素上很划算）。
- **对你有什么影响**：Blackwell 上跑 diffusion 的 gated activation 更快，输出与原来逐位一致。

### 5. [Fix] entrypoints: Resolve auth levels of included routers and pin a level for every route
- **代码层面**：`utils/auth.py` 新增 `_iter_leaf_routes` 递归展开嵌套路由，`_route_auth_level` 统一取 endpoint 的 `_auth_level`；测试里加了完整的路由→鉴权等级映射表。
- **新手概念课堂**：**FastAPI 路由嵌套**。`include_router` 不是把路由"拍平"，而是保留嵌套结构。鉴权中间件如果只扫顶层，就会漏掉子路由——这是**安全边界**问题。
- **对你有什么影响**：安全修复。之前某些嵌套路由的鉴权可能被错误降级为 NORMAL，现在每个路由都有明确等级。

### 6. [Fix] [NPU] modelslime: Fix NEXTN startup and mamba scatter UB overflow
- **代码层面**：`ascend_hybrid_linear_attn_backend.py` 新增 `_mamba_scatter_h_block_size`，根据 V/K 维度动态计算 H_BLOCK_SIZE，避免 Ascend UB（Unified Buffer）溢出；`modelslim.py` 支持 MoE 层 skip。
- **新手概念课堂**：**UB 是 NPU 的"片上暂存区"**，类似 GPU 的 shared memory，但容量固定（192KB）。kernel 的 tile 太大就编译失败报 "ub overflow"。解法是**按硬件容量反推 tile 大小**。
- **对你有什么影响**：Ascend NPU 上跑 ModelSlim 量化的 Mamba 模型不再启动失败。

### 7. [Fix] hicache: Include logical KV dtype in storage keys
- **代码层面**：`storage_model_name` 签名加 `kv_cache_dtype` 参数，key 变成 `{model}-{tag}-dtype_{dtype}`；`cache_controller.py` 从 anchor tensor 取 dtype 传入；README 更新说明。
- **新手概念课堂**：**逻辑 dtype vs 物理 dtype**。FP8 E4M3 和 E5M2 底层都是 8 位，物理上都是 uint8，但**解释方式不同**（指数/尾数位数不同）。key 必须用逻辑 dtype 区分。
- **对你有什么影响**：升级后旧 key 变"冷"（cache miss 重新填充），但不再有跨 dtype 读错数据的风险。

### 8. [Perf] sampling: Re-derive the top-k flags when a running batch drops rows
- **代码层面**：`SamplingBatchInfo` 新增 `refresh_top_k_flags`，`filter_batch` 后调用；把 `from_schedule_batch` 里的 flag 计算抽成 `_top_k_flags` 复用。
- **新手概念课堂**：**粘性状态 bug**。batch 里一个 greedy 请求（top_k=1）结束后，`need_top_k_sampling=True` 这个标志位**不会自动重置**，导致剩余请求白跑 top-k kernel。就像房间里最后一个人走了但灯没关。
- **对你有什么影响**：混合采样参数的 batch 在请求陆续结束后，采样 kernel 开销下降。

### 9. [Feat] rust-server: Accept client request IDs up to 256 bytes
- **代码层面**：Rust 测试重构为 `assert_duplicate_abort_and_reuse(public_id)` 参数化，新增 200 字节长 ID 的测试用例。
- **新手概念课堂**：**参数化测试**。把重复的测试逻辑抽成函数，用不同输入跑多遍——比复制粘贴测试更易维护，且能覆盖边界值（长 ID）。
- **对你有什么影响**：Rust 前端支持更长的客户端请求 ID（如 UUID+前缀），不会因 ID 过长被拒。

### 10. [Feat] rust-server: Pass the video item frame and token caps to multimodal processors
- **代码层面**：`MediaHints` 加 `max_frames`、`max_tokens_per_frame` 字段，贯穿 wire → request_log → multimodal 各层。
- **新手概念课堂**：**视频采样的两个旋钮**。`max_frames` 控制抽多少帧，`max_tokens_per_frame` 控制每帧多少 token——两者共同决定视频的**计算预算**。不设上限，一个长视频可能撑爆 context window。
- **对你有什么影响**：Rust 前端调用视频模型时，可以精确控制视频的帧数和 token 数。

### 11. [Feat] rust-server: Serve HTTP/2 (h2c) beside HTTP/1.1 with --enable-http2
- **代码层面**：Rust 前端原生支持 h2c（明文 HTTP/2），Python 前端仍走 Granian；`serving_hook.py` 里 granian 检查加 `if not envs.SGLANG_RUST_SERVER.get()` 条件；移除 Rust 前端的 `enable_http2` 不兼容标记。
- **新手概念课堂**：**h2c vs h2**。h2 需要 TLS（HTTPS），h2c 是明文版 HTTP/2，适合内网服务间通信。HTTP/2 的多路复用能避免 HTTP/1.1 的队头阻塞。
- **对你有什么影响**：Rust 前端开 `--enable-http2` 不再需要装 granian，客户端可用 h2c 连接。

### 12. [Feat] kv-shard: Let an out-of-tree HiCache storage backend join logical-page KV sharding
- **代码层面**：`kv_shard_hook.py` 新增 `register_kv_shard_storage_backend` 注册机制；`cache_controller.py` 的 `alloc_prefetch_host_buffers` 加 `start_owner` 参数支持分片；`pipeline.py` 传 shard 信息。
- **新手概念课堂**：**插件注册表模式**。核心代码不硬编码支持哪些后端，而是维护一个 `set`，第三方通过 `register_xxx(name)` 注册。这样**核心不用改**就能扩展，是开源项目常见的扩展点设计。
- **对你有什么影响**：自研 HiCache storage 后端可以通过注册加入 KV sharding，不用 fork 主仓库。

### 13. [Refactor] multimodal: [MM refactor 3/9] Extract a shared multimodal processor mixin
- **代码层面**：新建 `multimodal_processor_mixin.py`（411 行），把媒体加载（load_image/video/audio）、URL 安全配置、tokenizer 解析等共享逻辑抽成 Mixin。
- **新手概念课堂**：**Mixin 模式**。Python 多继承的一种"轻量"用法：Mixin 只提供方法，不定义完整类，通过 `class MyProcessor(MultimodalProcessorMixin, BaseProcessor)` 组合进来。好处是**训练和服务代码共享同一套媒体加载**。
- **对你有什么影响**：行为不变，但自定义 processor 未来应继承 Mixin 而非复制代码。

### 14. [Feat] mem_cache: Joint Full/SWA eviction walk on the Rust tree core
- **代码层面**：Rust tree core 的 `TreeCoreInitParamsBinding` 加 `joint_lru` 字段；`adapter.py` 移除"joint_lru 降级为 lru"的逻辑，新增 `joint_window_type`；文档更新。
- **新手概念课堂**：**Full attention vs SWA（滑动窗口注意力）**。Full 看全部历史，SWA 只看最近 N 个 token。混合模型两者都有，驱逐时要**统一排序**（joint）而不是各管各的，否则会驱逐掉还在被 SWA 引用的 Full KV。
- **对你有什么影响**：混合注意力模型 + Rust tree core + `--radix-eviction-policy joint_lru` 现在真正生效，缓存命中率提升。

### 15. [Fix] scheduler: Answer health probes that are waiting behind other requests
- **代码层面**：`scheduler.py` 新增 `_ack_waiting_health_checks`，在 `_get_new_batch_prefill_raw` 早退分支之后调用；把排队中的 health check 请求直接 ack 并释放，不等它跑完。
- **新手概念课堂**：**健康探针的语义**。K8s 的 liveness probe 只关心"进程活着吗"，不需要真的跑一次推理。如果探针排在长 prefill 后面，会被误判为"挂了"然后被 kill——这是**级联故障**的经典来源。
- **对你有什么影响**：高负载时实例不再被误杀，稳定性提升。

### 16. [Refactor] mem_cache: Tag full-only KV frees that release a tree node's value, and commit load-backs through one method
- **代码层面**：`FreeDeviceKVFullOnly` 加 `tree_owned: bool` 字段；`swa.py` 里标记 `tree_owned=True`；`unified_radix_cache.py` 抽出 `_commit_load_back` 统一入口。
- **新手概念课堂**：**所有权标记**。同样是"释放 KV slot"，但"树节点的值"和"请求临时占用的"释放语义不同——前者要通知树，后者不用。加个布尔标记让下游能区分，避免**双重释放**。
- **对你有什么影响**：内部重构，修复潜在的双重释放 bug。

### 17. [Perf] hicache: Keep the buffer load-back ownership check from waiting on the GPU at admission
- **代码层面**：新增 `_OwnershipCheck` 结构，把 `torch.equal` 的同步比较改成**异步 GPU 比较 + Event**，在 load-back 的 ack 时才读结果；用 `async_d2h` 做异步拷贝。
- **新手概念课堂**：**同步点 = 性能杀手**。`torch.equal(a, b)` 会强制 GPU 算完并把 bool 传回 CPU，整个 pipeline 卡住。改成"先在 GPU 上算，Event 记录，稍后再读"就能重叠计算。
- **对你有什么影响**：HiCache buffer load-back 的 admission 路径不再阻塞，吞吐提升。

### 18. [Feat] scheduler: Add opt-in aging to the shortest-prefill-first policy
- **代码层面**：新增 `--shortest-prefill-aging-factor` 参数（默认 0 关闭），`schedule_policy.py` 里排序 key 加入 `processed_tokens * factor` 的补偿项；校验 factor 必须有限且非负、且仅在 shortest-prefill-first 下可用。
- **新手概念课堂**：**老化（aging）**。操作系统调度里经典的反饿死机制：等待越久，优先级越高。这里用"每处理一个 token，等待请求的 rank 降 factor"实现，等待上限约为 `uncached_len / factor`。
- **对你有什么影响**：启用后长 prefill 请求不会被短请求无限插队，尾延迟更可控。

### 19. [Fix] scheduler: Observe finished storage prefetches before shortest-prefill-first ranks, and count staged prefixes as cached
- **代码层面**：`poll_storage_prefetches` 在排序前轮询 L3 prefetch 进度；`_shortest_prefill_work` 用 `staged_storage_prefix_end` 把"已 prefetch 但未入树"的 prefix 也算作 cached。
- **新手概念课堂**：**"已加载"和"已入树"的区别**。prefetch 完成的数据可能还在 buffer 里没插进 radix tree，如果排序时不算它，请求会被误判为"很长"而排到后面，形成**自我实现的慢**。
- **对你有什么影响**：HiCache + shortest-prefill-first 组合下，请求排序更准确，TTFT 更稳。

### 20. [Perf] hicache: End each buffer-only backup chain where the insert walk's previous one ended
- **代码层面**：`_InsertWalkState` 加 `backup_chain_tail` 字段，`_insert_backup_action` 记录上次 chain 的终点，`_build_backup_kv_action` 加 `stop_at` 参数提前终止向上遍历。
- **新手概念课堂**：**O(d²) → O(d) 的遍历优化**。原来每次 backup 都从节点往上遍历所有未 backup 的祖先，深度 d 的路径总共遍历 d(d+1)/2 次。记录"上次停在哪"，下次从那停，就变成线性。
- **对你有什么影响**：深路径（长对话）的 HiCache backup 开销显著下降。

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)