# 🚀 vLLM & SGLang 每日更新 — 2026-10-28

> 自动生成于 2026-10-28 07:21 UTC | AI 解读: ✅ 含代码解读

## vllm
## 💡 今日关键词（放在最前面！）

- **关键词：Rust 前端对齐 Python 语义**
  - 一句话解释：vLLM 正在用 Rust 重写前端（HTTP 服务层），今天一口气有 5 个 commit 都在"让 Rust 的行为和 Python 版一模一样"（404 错误码、reasoning token 计数、chat template 列表格式、max_tokens 校验、metrics 导出）。
  - 为什么社区在关注：这说明 Rust 前端已经从"能跑"进入"逐字节对齐"的成熟阶段——性能敏感的服务层正在被 Rust 替换，这是推理框架的大趋势（SGLang、TensorRT-LLM 也在做类似的事）。
  - 对新手意味着什么：如果你会 Rust，现在切入 LLM 推理基础设施是黄金窗口期；如果你只会 Python，至少要能读懂 Rust 代码，否则未来看不懂服务层。

- **关键词：安全加固（Security Hardening）**
  - 一句话解释：今天有 3 个 commit 是安全修复——ReDoS 正则拒绝服务、音频时长边界限制、错误信息脱敏。
  - 为什么社区在关注：vLLM 已经是大规模生产部署的组件，安全漏洞（尤其是能让服务卡死的 ReDoS）影响面极大，说明项目从"研究工具"正式转向"生产级基础设施"。
  - 对新手意味着什么：写正则表达式时一定要想"最坏输入是什么"，这是每个后端工程师的必修课，比学任何框架都重要。

- **关键词：异构硬件适配（XPU / ROCm / SM120）**
  - 一句话解释：Intel XPU 升级 kernel 版本、AMD ROCm 修 Kimi-K3 权重折叠、NVIDIA SM120 优化 MLA FP8 KV cache。
  - 为什么社区在关注：推理框架不再是"NVIDIA 专属"，Intel、AMD 都在投入，硬件碎片化是必然趋势。
  - 对新手意味着什么：不要只盯着 CUDA，理解"同一个算法在不同硬件上的实现差异"是稀缺能力。

- **关键词：KV Cache 布局与流水线并行**
  - 一句话解释：PP（流水线并行）不同 stage 持有不同层，支持的 KV cache 布局可能不同，今天从"断言必须一致"改成"求交集合并"。
  - 为什么社区在关注：大模型必须多卡并行，PP 是最常用的方案之一，这类"分布式配置协商"问题会越来越多。
  - 对新手意味着什么：分布式系统里"假设所有节点配置相同"是经典坑，学会用"协商/求交集"思维设计接口。

- **关键词：可观测性（Metrics / Observability）**
  - 一句话解释：Rust 前端补齐了 Python 版的 Prometheus 指标导出（KV fetch 阶段、sleep 状态）。
  - 为什么社区在关注：生产部署离不开监控，指标不齐就没法做容量规划和故障定位。
  - 对新手意味着什么：写服务时从一开始就设计指标，比事后补容易一百倍。

---

## 🧪 重要知识点 & 动手实验

- **知识点 1：ReDoS（正则表达式拒绝服务攻击）与原子分组**
  - **是什么**：某些正则表达式在特定输入下会指数级回溯。比如 `(?:/[\w\-]+)+/[\w\-]+\.\w+` 匹配 `"/a" * 50000` 时，引擎会疯狂尝试各种切分方式，CPU 直接拉满，服务卡死。
  - **为什么重要**：vLLM 用这个正则做错误信息脱敏（把文件路径替换成 `<path>`）。攻击者只要发一个恶意请求触发错误，就能让整个服务不可用。修复方法有两个：① 先判断 `"." in message` 快速短路；② 用 `(?>)` 原子分组禁止回溯。
  - **动手试试**：
    ```python
    import re, time

    # 危险版本：嵌套量词 + 无原子分组
    bad = re.compile(r"(?:/[\w\-]+)+/[\w\-]+\.\w+")
    # 安全版本：原子分组 (?>...) 禁止回溯
    good = re.compile(r"(?>/[\w\-]+)+/[\w\-]+\.\w+")

    payload = "/a" * 50000

    t0 = time.perf_counter()
    # bad.search(payload)  # 取消注释会卡住几十秒甚至更久
    t1 = time.perf_counter()
    print(f"bad: {t1-t0:.3f}s")

    t0 = time.perf_counter()
    good.search(payload)
    t1 = time.perf_counter()
    print(f"good: {t1-t0:.3f}s")
    ```
  - **预期结果**：`good` 版本在毫秒级完成；如果你真的取消注释 `bad`，会看到它跑几十秒甚至更久（这就是 ReDoS）。掌握标志：你能解释"为什么嵌套量词会导致指数回溯"。

- **知识点 2：KV Cache 布局（Layout）与分布式协商**
  - **是什么**：KV cache 在显存里怎么排布——是 `[Layer, Block, Head, Num, Dim]`（LBHND）还是 `[Block, Layer, Head, Num, Dim]`（BLHND）。不同 attention backend 支持的布局不同。
  - **为什么重要**：流水线并行时，Stage 0 可能是普通 attention 层（支持 LBHND），Stage 1 可能是 FlashInfer 层（只支持 BLHND）。旧代码直接断言"所有 stage 必须一致"→ 崩溃。新代码改成"求所有 stage 支持布局的交集，取投票最多的"。
  - **动手试试**：
    ```python
    # 模拟 vLLM 的合并逻辑
    from collections import Counter

    def merge_layouts(stage_prefs):
        # stage_prefs: 每个 stage 的偏好列表，按优先级排序
        # 空列表表示该 stage 不投票
        non_empty = [p for p in stage_prefs if p]
        if not non_empty:
            return None
        # 求交集
        common = set(non_empty[0])
        for p in non_empty[1:]:
            common &= set(p)
        if not common:
            raise ValueError("No layout satisfies all stages")
        # 每个 stage 的"第一选择"里出现次数最多的
        votes = Counter(p[0] for p in non_empty if p[0] in common)
        return votes.most_common(1)[0][0]

    print(merge_layouts([["LBHNC", "BLHNC"], ["LBHNC", "BLHNC", "LBNHC"]]))
    # 输出: LBHNC
    print(merge_layouts([["LBHNC"], ["BLHNC"]]))
    # 抛出 ValueError
    ```
  - **预期结果**：第一个输出 `LBHNC`（两个 stage 都支持且都排第一），第二个抛异常。掌握标志：你能解释"为什么空列表不能投票"（否则一个未声明的 stage 会污染统计）。

- **知识点 3：FP8 KV Cache 的"紧凑行"优化（NoPE 模型）**
  - **是什么**：DeepSeek 系 MLA 模型的 KV cache 用 FP8 存储，每行 656 字节 = 512 字节 NoPE（8-bit）+ 128 字节 RoPE + 16 字节 scale。但如果模型没有 RoPE（NoPE 模型），那 128 字节就是浪费，可以压缩到 528 字节。
  - **为什么重要**：KV cache 是推理显存瓶颈。省 20% 的 KV cache 意味着可以多跑 20% 的并发请求，直接降低成本。
  - **动手试试**：
    ```python
    # 计算 KV cache 节省比例
    full_row = 656  # 有 RoPE tail
    compact_row = 528  # 无 RoPE tail
    saved = 1 - compact_row / full_row
    print(f"每行节省 {saved:.1%}")

    # 假设 70B 模型，80 层，每层 128 头，每头 576 维，8K 上下文
    layers, heads, dim, seq = 80, 128, 576, 8192
    tokens = seq
    full_bytes = tokens * layers * heads * dim * 1  # FP8 = 1 byte
    print(f"理论节省: {full_bytes * saved / 1e9:.1f} GB")
    ```
  - **预期结果**：看到约 19.5% 的节省。掌握标志：你能说清"为什么 NoPE 模型可以省掉 RoPE tail"（因为不需要存旋转位置编码）。

- **知识点 4：CUDA Checkpoint（进程级 GPU 状态快照）**
  - **是什么**：NVIDIA 驱动提供的 API，可以把一个进程的 CUDA 状态（显存、CUDA graph、context）冻结到磁盘，之后恢复。vLLM 用它实现"深度睡眠"——sleep 时连 CUDA graph 都不销毁，wake 时不用重新 capture（capture 很慢）。
  - **为什么重要**：RL 训练场景需要频繁在"训练"和"推理"之间切换，每次重新 capture CUDA graph 要几十秒。Checkpoint 能把切换降到亚秒级。
  - **动手试试**（概念验证，不需要真 GPU）：
    ```python
    # 模拟 sleep/wake 状态机
    class EngineState:
        def __init__(self):
            self.state = "awake"
            self.graph_captured = True

        def sleep(self, level):
            if level == 1:
                # 普通 sleep：释放 KV cache，保留 graph
                self.state = "asleep_l1"
                print("释放 KV cache，CUDA graph 保留")
            elif level == 2:
                # 深度 sleep：销毁 graph
                self.state = "asleep_l2"
                self.graph_captured = False
                print("释放 KV cache + 销毁 CUDA graph")

        def wake(self):
            if not self.graph_captured:
                print("重新 capture CUDA graph（慢！）")
                self.graph_captured = True
            self.state = "awake"
            print("唤醒完成")

    e = EngineState()
    e.sleep(2); e.wake()  # 会重新 capture
    e.sleep(1); e.wake()  # 不会重新 capture
    ```
  - **预期结果**：level=2 唤醒时打印"重新 capture"，level=1 不打印。掌握标志：你能解释"为什么 CUDA graph 的 capture 是昂贵的"（需要实际跑一遍 kernel 来记录）。

- **知识点 5：OpenAI 兼容层的"null 字段"陷阱**
  - **是什么**：OpenAI API 规范里 `annotations` 是 `array` 类型（不可为 null），`usage.prompt_tokens_details` 是 `object`（不可为 null）。但 Python 的 Pydantic 默认会把 `None` 序列化成 `null`，导致严格客户端（比如 OpenAI 官方 SDK）解析失败。
  - **为什么重要**：这是"规范兼容性"的经典坑。很多客户端用严格 schema 校验，多一个 null 字段就报错。vLLM 用 Pydantic 的 `exclude_if` 在序列化时剔除 null 字段。
  - **动手试试**：
    ```python
    from pydantic import BaseModel, Field

    class Message(BaseModel):
        role: str
        content: str | None = None
        # 关键：exclude_if 让 None 不出现
        annotations: list[dict] | None = Field(
            default=None, exclude_if=lambda v: v is None
        )

    m = Message(role="assistant", content="hi")
    print(m.model_dump())  # {'role': 'assistant', 'content': 'hi'}
    # 没有 annotations 键！

    m2 = Message(role="assistant", content="hi", annotations=[{"type": "url"}])
    print(m2.model_dump())  # annotations 出现了
    ```
  - **预期结果**：第一个 dump 里没有 `annotations` 键，第二个有。掌握标志：你能解释"为什么 `exclude_none=True` 不够用"（因为它会剔除所有 None，包括用户显式传的 None）。

---

## 🔥 重要更新

1. **ReDoS 安全修复（#52120）**：一个恶意请求就能让 vLLM 服务卡死几十秒，这是生产环境的严重风险，所有部署 vLLM 的用户都应该升级。
2. **PP 多 stage KV cache 布局协商（#61294）**：修复了流水线并行 + 混合 attention backend 场景下的启动崩溃，是大模型多卡部署的关键修复。
3. **NoPE 模型 FP8 KV cache 紧凑行（#58305）**：KV cache 省 20% 显存，直接提升并发能力，对 DeepSeek 系模型用户影响显著。
4. **Rust 前端 5 连发对齐 Python**：标志着 Rust 前端进入"可替换 Python"阶段，是 vLLM 架构演进的重要里程碑。

---

## 📋 逐条解读

### 1. [XPU] bump xpu-kernels to v0.1.16.2 (#61956)

- **代码层面**：`requirements/xpu.txt` 里 `vllm_xpu_kernels` 从 `0.1.15.4` 升到 `0.1.16.2`；同时 CI 配置文件里给一个 LoRA 测试加了 `--deselect`（暂时跳过 `test_fused_moe_lora_kernel_large_rank_fallback` 的某个参数化用例）。
- **新手概念课堂**：XPU 是 Intel 的通用 GPU（类似 NVIDIA 的 GPU 但架构不同）。`vllm_xpu_kernels` 是专门为 Intel GPU 编译的底层算子库，类似 CUDA 的 `cuda_kernel`。升级版本 = 换一套更快的"零件"。
- **对你有什么影响**：如果你用 Intel GPU 跑 vLLM，升级后可能获得性能提升；用 NVIDIA/AMD 的用户无感知。CI 里 deselect 说明新版本有个用例暂时不通过，属于"已知问题"。

### 2. [XPU][CI] Fix test_tp1_fp8_fusions rms_quant_fusion on XPU (#63975)

- **代码层面**：`tests/compile/fusions_e2e/test_tp1_quant.py` 里，针对 XPU 平台加了两处特殊处理：① 强制 `linear_backend="xpu"` 走 W8A8 量化路径；② 把 `attn_quant_fusion` 的期望匹配数改成 0（XPU 不支持 attention+quant 融合）。
- **新手概念课堂**：**算子融合（fusion）** 是把多个小算子（如 RMSNorm + 量化）合并成一个大 kernel，减少显存读写。但不同硬件的融合能力不同——NVIDIA 能融合 attention+quant，XPU 不能，所以测试期望值要按平台调整。
- **对你有什么影响**：纯 CI 修复，普通用户无感知。但它揭示了一个重要事实：**同一份代码在不同硬件上跑出的"优化图"是不一样的**，写性能测试时必须按平台分支。

### 3. [Bugfix][Pooling] Reject prompt-wide truncation for Jina ranking (#55200)

- **代码层面**：`docs/` 里加了文档说明；测试文件里删掉了一个"转发 truncate_prompt_tokens 到 proxy"的旧测试（因为现在直接拒绝这个参数）；核心逻辑改为：Jina reranker-v3 收到 `truncate_prompt_tokens` 参数时直接返回 400 错误，并提示用户改用 `max_tokens_per_query` / `max_tokens_per_doc`。
- **新手概念课堂**：**truncate_prompt_tokens** 是"把整个 prompt 截断到 N 个 token"。但对 reranker（重排序模型）来说，query 和 document 是分开的，整体截断会破坏语义（可能把 query 截掉一半）。所以正确做法是分别限制 query 和 doc 的长度。
- **对你有什么影响**：如果你用 Jina reranker 的 `/rerank` 接口，传 `truncate_prompt_tokens` 会报 400，需要改用新参数。这是一个"API 语义更严格"的破坏性变更。

### 4. [Security] Bound FunAudioChat runtime audio to the profiled duration (#57411)

- **代码层面**：新增测试文件 `test_funaudiochat.py`；`funaudiochat.py` 模型代码里，对运行时音频长度做了上限检查——不能超过 profile 阶段测得的时长。
- **新手概念课堂**：**profile 阶段**是 vLLM 启动时"试跑一遍"来确定显存分配。如果运行时音频比 profile 时长得多，特征提取器会申请超预期显存 → OOM。这是一个"资源边界"安全修复。
- **对你有什么影响**：用 FunAudioChat 模型时，超长音频会被拒绝而不是 OOM 崩溃。普通用户感知是"更稳定"，但如果你依赖超长音频，需要调整 profile 配置。

### 5. [Security] Fix ReDoS in validation error path sanitizer (#52120)

- **代码层面**：`vllm/entrypoints/serve/exception_handling/utils.py` 的 `sanitize_message` 函数：① 加 `if "." in message` 快速短路；② 正则从 `(?:/[\w\-]+)+` 改成 `(?>/[\w\-]+)+`（原子分组，禁止回溯）。测试文件加了 4 个 ReDoS 回归测试，包括 50k 字符输入必须在 0.1 秒内完成。
- **新手概念课堂**：**ReDoS** 是"正则表达式拒绝服务"。攻击者构造特殊输入，让正则引擎的回溯次数爆炸。原子分组 `(?>...)` 告诉引擎"匹配完就别回头"，从根本上杜绝回溯。
- **对你有什么影响**：这是一个**高危安全修复**。之前攻击者发一个恶意请求就能让服务卡死几十秒，现在不会了。强烈建议所有生产部署升级。

### 6. [Bugfix] Omit null `annotations` and `usage.prompt_tokens_details` in chat completions (#64177)

- **代码层面**：`chat_completion/protocol.py` 里 `ChatMessage.annotations` 的类型从 `OpenAIAnnotation | None` 改成 `list[OpenAIAnnotation] | None`，并加了 `Field(exclude_if=lambda v: v is None)`；`usage.prompt_tokens_details` 同样处理。测试验证：没设置时序列化结果里没有这两个键。
- **新手概念课堂**：**OpenAI API 规范**里 `annotations` 是数组、`prompt_tokens_details` 是对象，都不能是 null。Pydantic 默认会把 None 序列化成 null，严格客户端（如 OpenAI 官方 SDK）会解析失败。`exclude_if` 让 None 字段"消失"而不是变成 null。
- **对你有什么影响**：用 OpenAI 官方 SDK 或严格 schema 校验的客户端调 vLLM 时，不会再因为多出的 null 字段报错。这是一个"兼容性修复"。

### 7. [Bugfix] Merge supported KV cache layouts across pipeline stages instead of asserting they match (#61294)

- **代码层面**：`vllm/v1/attention/backends/utils.py` 里抽出 `_merge_supported_layouts` 函数；`resolve_kv_cache_layout` 从"断言所有 worker 布局一致"改成"求交集 + 投票"。测试新增 `test_pipeline_stages_resolve_a_layout_every_stage_supports`。
- **新手概念课堂**：**流水线并行（PP）** 把模型按层切到不同 GPU。Stage 0 可能有普通 attention 层（支持 LBHND），Stage 1 可能有 FlashInfer 层（只支持 BLHND）。旧代码假设"所有 stage 配置相同"，PP 场景直接崩溃。新代码用"协商"思维：求所有 stage 支持布局的交集，取投票最多的。
- **对你有什么影响**：用 PP + 混合 attention backend 部署大模型时，不会再启动崩溃。这是多卡部署的关键修复。

### 8. [Perf][SM120][MLA] Use compact FP8 KV rows for NoPE models (#58305)

- **代码层面**：`csrc/cache_kernels.cu` 里 `concat_and_cache_ds_mla_kernel` 新增 `has_rope_tail` 参数；行字节数从固定 656 改成"656（有 RoPE）或 528（无 RoPE）"；kernel 里 `threadIdx.x >= 64` 的 warp 在无 RoPE 时直接 `return`，省掉 32 个线程的工作。
- **新手概念课堂**：**MLA（Multi-head Latent Attention）** 是 DeepSeek 提出的 attention 变体，KV cache 结构特殊：512 字节 NoPE（内容）+ 128 字节 RoPE（位置）+ 16 字节 scale = 656 字节。**NoPE 模型**（不用 RoPE）可以省掉那 128 字节，变成 528 字节，省 20% 显存。
- **对你有什么影响**：用 DeepSeek 系 NoPE 模型时，KV cache 显存省 20%，意味着并发能力提升约 20%。SM120 是 NVIDIA 新架构（Blackwell 系列），老 GPU 无感知。

### 9. [Feature][Core] Add opt-in multi-worker CUDA checkpoint sleep (#64186)

- **代码层面**：新增 `CUDACheckpointExecutor`（`vllm/v1/executor/cuda_checkpoint_executor.py`）；文档 `sleep_mode.md` 加了 130 行说明；核心是调用 NVIDIA 驱动的 `cuCheckpointProcessLock/Checkpoint/Restore/Unlock` API，把 worker 进程的 CUDA 状态冻结到磁盘。
- **新手概念课堂**：**CUDA Checkpoint** 类似"进程的休眠"——把 GPU 上的所有状态（显存、CUDA graph、context）存到磁盘，之后恢复。普通 sleep 只释放 KV cache，CUDA graph 要重新 capture（几十秒）；Checkpoint 连 graph 都保留，唤醒是亚秒级。
- **对你有什么影响**：**实验性功能，默认关闭**。RL 训练场景（频繁在训练/推理间切换）能大幅提速。但要求 Linux + 特定 NVIDIA 驱动 + 单节点 TP，限制很多，不建议生产使用。

### 10. [Doc] Add autotrust/JEV-27B-VL multimodal recipe (vLLM >= 0.32.0) (#64180)

- **代码层面**：新增 `recipes/models/autotrust/JEV-27B-VL.yaml`（200 行），包含模型元信息、启动参数、依赖下载命令、可选功能（text_only、encoder_parallel）。
- **新手概念课堂**：**Recipe** 是 vLLM 的"部署配方"——一份 YAML 描述某个模型怎么跑（启动参数、环境变量、注意事项）。类似 Docker Compose 的 `docker-compose.yml`，但是给 LLM 用的。
- **对你有什么影响**：想部署 JEV-27B-VL 的用户可以直接抄配置，不用自己摸索参数。这是社区生态建设的一部分。

### 11. [Bugfix][Frontend] fix Anthropic inline-system probe crashing startup (#64020)

- **代码层面**：`vllm/entrypoints/anthropic/serving.py` 的 `_detect_merge_inline_system` 函数，`except jinja2.TemplateError` 改成 `except Exception`。测试加了 3 个用例：`tools is none` 检查、`for tool in tools` 循环、非 TemplateError 异常。
- **新手概念课堂**：**Jinja2 模板探测**是启动时"试渲染一遍 chat template"来判断是否需要合并 inline system message。但有些模板在 `tools` 未定义时会抛 `TypeError`（不是 `TemplateError`），旧代码没捕获 → 启动崩溃。
- **对你有什么影响**：用 OLMo 3 或 Hermes 风格 chat template 时，服务不会再启动崩溃。这是一个"启动健壮性"修复。

### 12. [Rust Frontend] Export vllm:num_requests_kv_fetch_by_stage like Python (#63962)

- **代码层面**：`rust/src/engine-core-client/src/metrics.rs` 新增 `scheduler_kv_fetch_by_stage: [U64Gauge; 3]`（3 个 stage 的 gauge），从 `SchedulerStats.num_kv_fetch_reqs_by_stage` 读取并 set 到 Prometheus。
- **新手概念课堂**：**KV fetch by stage** 是统计"每个流水线 stage 有多少请求在等 KV cache 传输"。这是 PP/PD 分离场景的关键指标——哪个 stage 是瓶颈一目了然。**Prometheus Gauge** 是"可增可减的数值"，适合表示当前排队数。
- **对你有什么影响**：用 Rust 前端 + PP 部署时，Prometheus 里能看到和 Python 版一样的指标，监控面板不用改。

### 13. [ROCm][Bugfix][Kimi-K3] Build the folded q_b_proj / W_UK weight on the root model (#64158)

- **代码层面**：`vllm/models/kimi_k3/amd/model.py` 里 `KimiK3ForConditionalGeneration` 新增 `process_weights_after_loading` 方法，转发给 `self.language_model.process_weights_after_loading()`。
- **新手概念课堂**：**权重折叠（weight folding）** 是把两个矩阵乘（`q_b_proj` 和 `W_UK`）提前合并成一个大矩阵，推理时少算一次 GEMM。但 vLLM 的 loader 只调用**根模型**的 `process_weights_after_loading` 钩子，子模型（language_model）的钩子不会被调用 → 折叠没做 → 推理时 assert 失败。
- **对你有什么影响**：AMD ROCm 上跑 Kimi-K3 时，DCP=1（数据并行=1）场景不会再崩溃。这是 AMD 适配的关键修复。

### 14. [RL] Support per-update source ranks for NCCL weight transfer (#62557)

- **代码层面**：`docs/training/weight_transfer/nccl.md` 更新文档；核心是 NCCL 权重传输支持"多个广播源 rank"——之前只有 trainer rank 0 广播，现在可以多个 rank 各广播一部分参数（适合 PP 训练场景）。`rank_offset` 从固定 1 改成"广播源数量"。
- **新手概念课堂**：**权重传输**是 RL 训练中"训练进程把新权重发给推理进程"的过程。**NCCL** 是 NVIDIA 的集合通信库。之前设计假设"只有 rank 0 有全部权重"，但 PP 训练时每个 stage 只有部分权重，需要多个 rank 同时广播。
- **对你有什么影响**：用 PP 做 RL 训练（如 GRPO + PP）时，权重同步能正常工作。单卡训练用户无感知。

### 15. [Bench] Default to Zipf routing for forward benchmarks with dummy weights (#64167)

- **代码层面**：`vllm/benchmarks/forward/` 里 `--routing` 新增 `zipf` 选项；`auto` 的语义从"dummy 权重时用 balanced"改成"dummy 权重时用 zipf"；文档更新。Zipf 分布：每个 token 按 `rank^-1.25` 的概率选专家，每层随机打乱专家排名。
- **新手概念课堂**：**MoE 路由**决定每个 token 发给哪些专家。**Zipf 分布**是"少数专家被大量使用"的长尾分布，更接近真实负载。之前用 **balanced**（均匀分布）测出的性能是"理想情况"，不真实。
- **对你有什么影响**：跑 MoE 模型 benchmark 时，默认结果更接近真实场景（不会虚高）。想测理想情况可以显式 `--routing balanced`。

### 16. [Rust Frontend] Export vllm:engine_sleep_state like Python (#63959)

- **代码层面**：`rust/src/engine-core-client/src/client.rs` 里 `sleep` 方法签名从 `level: u32` 改成 `level: SleepLevel`（类型安全）；sleep/release_kv_cache/wake_up 后调用 `record_sleep_state` 更新指标；新增 `EngineSleepState` 枚举。
- **新手概念课堂**：**sleep 状态机**有 3 个状态：Awake / Asleep(level) / KvCacheReleased。Prometheus 指标 `engine_sleep_state` 用 gauge 表示当前状态，监控面板可以画"引擎是否在睡眠"。
- **对你有什么影响**：用 Rust 前端时，监控面板能看到 sleep 状态，和 Python 版一致。`SleepLevel` 类型化后，传非法值会在编译期报错而不是运行时。

### 17. [Rust Frontend] Return 404 for a missing LoRA adapter path like Python (#63961)

- **代码层面**：`rust/src/server/src/error.rs` 新增 `ApiError::NotFound` 变体（404）；`grpc/control.rs` 里 `LoadLoraError::PathAccess(NotFound)` 映射到 gRPC `Code::NotFound`；`lora.rs` 里路径检查失败时返回 NotFound 而非 InvalidPath。
- **新手概念课堂**：**HTTP 状态码语义**：404 = 资源不存在，400 = 请求格式错。LoRA 路径不存在是"资源不存在"，应该 404；路径格式非法（如包含 `..`）才是 400。旧代码统一返回 400，和 Python 版不一致。
- **对你有什么影响**：用 Rust 前端加载不存在的 LoRA 时，客户端能正确区分"路径错"和"资源不存在"，便于错误处理。

### 18. [Rust Frontend] Count gpt-oss reasoning tokens like Python (#63960)

- **代码层面**：`rust/src/chat/src/output/harmony/mod.rs` 里 `HarmonyState` 新增 `encoding`、`reasoning_tokens`、`unattributed_reasoning_tokens` 字段；新增 `count_reasoning_token` 方法，判断当前 token 是否属于 `analysis` channel 或带 recipient 的 `commentary` channel，且不是特殊 token。
- **新手概念课堂**：**gpt-oss 的 Harmony 格式**用特殊 token 标记"推理内容"（`<|channel|>analysis`）和"最终回答"（`<|channel|>final`）。**reasoning token 计数**是 usage 里的 `reasoning_tokens` 字段，OpenAI 规范要求统计。Rust 版之前没统计，和 Python 不一致。
- **对你有什么影响**：用 Rust 前端跑 gpt-oss 模型时，`usage.reasoning_tokens` 字段有值了，和 Python 版一致。计费/统计系统能正常工作。

### 19. [Rust Frontend] Accept list-form chat templates like Transformers (#63963)

- **代码层面**：`rust/src/chat/src/backend/hf.rs` 里 chat template 解析支持 list 格式——`{"chat_template": [{"name": "default", "template": "..."}, {"name": "tool_use", "template": "..."}]}`。根据请求是否带 tools 选择对应模板；显式配置的 template 或 `chat_template.jinja` 文件会覆盖所有命名模板。
- **新手概念课堂**：**Transformers 的 chat template 格式**支持两种：① 单个字符串（所有场景用同一个）；② 列表（按场景命名，如 `default`、`tool_use`）。vLLM Python 版支持列表，Rust 版之前只支持字符串。
- **对你有什么影响**：用 Rust 前端 + 带命名模板的模型（如某些 Qwen 变体）时，tool calling 场景会用正确的模板，不会渲染错。

### 20. [Rust Frontend] Reject max_tokens above max_model_len like Python (#63964)

- **代码层面**：`rust/src/server/src/routes/openai/utils/mod.rs` 新增 `validate_max_output_tokens` 函数；`chat_completions.rs` 和 `completions.rs` 在请求预处理后调用它——如果 `max_tokens > max_model_len` 返回 400 错误。
- **新手概念课堂**：**max_model_len** 是模型的最大上下文长度（如 8192）。如果用户请求 `max_tokens=10000`，物理上不可能生成那么多 token（模型没有那么多位置编码）。Python 版会提前拒绝，Rust 版之前会"静默 clamp"（截断到 8192），行为不一致。
- **对你有什么影响**：用 Rust 前端时，请求超过上下文长度的 `max_tokens` 会得到明确的 400 错误（和 Python 一致），而不是静默截断。便于客户端发现配置错误。

## sglang
## 💡 今日关键词（放在最前面！）

- **关键词：AMD ROCm / Mono-kernel 生态**
  - 一句话通俗解释：SGLang 正在为 AMD 的 MI355X 等 GPU 打造“单核融合”高性能推理配方，并持续升级 `sgl-amd-mono` 依赖。
  - 为什么社区在关注：DeepSeek-V4、Qwen3.5 等大模型都开始有专门的 MI355X 调优方案，说明 AMD GPU 不再只是“能跑”，而是进入“精调性能”阶段。异构计算（NVIDIA + AMD + NPU）正在成为推理框架的标配。
  - 对新手意味着什么：如果你只会写 CUDA，现在是补 ROCm/HIP 和 Triton 的好时机；跨平台 kernel 能力会越来越值钱。

- **关键词：Triton 3.9 兼容性修复**
  - 一句话通俗解释：Triton 3.9 删除了 `tl.make_block_ptr` / `tl.advance`，SGLang 用兼容层把 FLA 等 kernel 救回来了。
  - 为什么社区在关注：Triton 是 SGLang kernel 层的主力 DSL，上游 API 一变动，下游框架就得打补丁。今天有两个 commit 都在处理 Triton 3.9 的行为差异（block pointer 移除、swap_ab 下的 scale 广播）。
  - 对新手意味着什么：学 Triton 不能只学“写 kernel”，还要学“读 changelog + 写兼容层”。这是真实工程能力。

- **关键词：MoE 通信与量化优化**
  - 一句话通俗解释：MoE（混合专家）的 all-to-all 通信、block-FP8 scale 广播、Ascend NPU 的 MoE 后端都在今天被优化或修复。
  - 为什么社区在关注：MoE 是大模型降本增效的核心结构，但它的通信开销和量化精度是两大痛点。今天既有 NVIDIA 的 custom-all-reduce 调优，也有 Ascend NPU 的新后端，说明 MoE 优化是跨硬件的主战场。
  - 对新手意味着什么：想进大模型推理方向，MoE 的 dispatch/combine、量化 scale 布局是必学内容。

- **关键词：零拷贝 / 内存复用**
  - 一句话通俗解释：多个 commit 在消除不必要的内存拷贝（page-index 对齐跳过 copy、aiter MoE 复用输出 buffer、Engram 用 Triton 重排替代 transpose+reshape）。
  - 为什么社区在关注：推理性能瓶颈往往不在算力，而在 HBM 带宽和 kernel launch 开销。零拷贝是“免费性能”的来源。
  - 对新手意味着什么：写高性能代码时，先问“这次拷贝能不能省”，比盲目优化 GEMM 更有效。

- **关键词：多硬件后端（NPU / Rust server / Diffusion）**
  - 一句话通俗解释：Ascend NPU 新增 MoE 后端，Rust server 支持每节点 sidecar，Diffusion 模型（MOSS-Image、Ming）持续优化。
  - 为什么社区在关注：SGLang 正在从“LLM 推理框架”扩展为“多模态 + 多硬件 + 多语言运行时”的统一平台。
  - 对新手意味着什么：不要只盯着 CUDA + LLM，NPU 适配、Diffusion 推理、Rust 服务端都是新增长点。

## 🧪 重要知识点 & 动手实验

- **知识点 1：Triton 兼容层——当上游 API 被删除怎么办**
  - **是什么**：Triton 3.9 移除了 `tl.make_block_ptr` 和 `tl.advance`，SGLang 写了一个 `triton_block_ptr_compat.py`，在检测到 API 缺失时用普通指针 + 手动偏移重建 block pointer 语义。
  - **为什么重要**：真实工程中，你依赖的框架随时可能 break。不会写兼容层，就只能等上游修；会写，就能当天恢复 CI。这个模式（try import → fallback 实现）在 Python 生态里非常通用。
  - **动手试试**：
    ```python
    # 模拟兼容层思路
    try:
        from triton.language import advance  # 新版本有
        HAS_BLOCK_PTR = True
    except ImportError:
        HAS_BLOCK_PTR = False

    if not HAS_BLOCK_PTR:
        # 用普通指针 + 手动计算偏移替代
        def manual_advance(base_ptr, offsets, strides):
            return base_ptr + sum(o * s for o, s in zip(offsets, strides))
    print("HAS_BLOCK_PTR =", HAS_BLOCK_PTR)
    ```
  - **预期结果**：在你的 Triton 版本上打印出 True/False，理解“检测能力 → 降级实现”的兼容模式。

- **知识点 2：MoE block-FP8 的 scale 广播——swap_ab 后维度会翻转**
  - **是什么**：在 fused MoE kernel 里，为了性能会把 A、B 矩阵交换（swap_ab）。交换后 `a_scale` 和 `b_scale` 的语义也交换了：原来按 token 的 scale 变成了按输出通道的 scale，广播方向必须跟着改。
  - **为什么重要**：这是典型的“量化 + 布局优化”交叉坑。scale 广播错了不会报错，只会静默算错，精度掉点很难排查。今天 commit 10 专门加了测试覆盖 `swap_ab=True` 且 `BLOCK_SIZE_N <= group_n` 的分支。
  - **动手试试**：
    ```python
    import torch
    # 模拟：a_scale 是 per-token (m,)，b_scale 是 per-channel (n,)
    m, n = 4, 8
    a_scale = torch.rand(m)
    b_scale = torch.rand(n)
    # 正常：a_scale[:, None] * b_scale[None, :]
    normal = a_scale[:, None] * b_scale[None, :]
    # swap_ab 后：a_scale 变标量/通道，b_scale 变 per-token
    swapped = a_scale * b_scale[None, :]  # 注意这里 a_scale 被当作标量广播
    print(normal.shape, swapped.shape)
    ```
  - **预期结果**：看到两种广播形状不同，理解“swap 不只是换矩阵，scale 语义也要换”。

- **知识点 3：零拷贝——`F.pad` 即使不需要 pad 也会 copy**
  - **是什么**：Commit 7 在 `_pad_last_dim` 里加了一行 `if target_size == curr_size: return x`。因为 `F.pad` 即使 pad 长度为 0，也会返回一个新 tensor（触发拷贝）。
  - **为什么重要**：这是 PyTorch 里非常隐蔽的性能坑。很多“看起来无害”的 pad/reshape/contiguous 调用，在 hot path 上就是白白的 HBM 带宽浪费。
  - **动手试试**：
    ```python
    import torch
    import torch.nn.functional as F
    x = torch.randn(1024, 128, device="cuda")
    y = F.pad(x, (0, 0))          # pad 0，但仍然 copy
    z = x if x.shape[-1] == 128 else F.pad(x, (0, 128 - x.shape[-1]))
    print("F.pad data_ptr 相同?", x.data_ptr() == y.data_ptr())  # False
    print("跳过 pad 后相同?", x.data_ptr() == z.data_ptr())       # True
    ```
  - **预期结果**：看到 `F.pad(x, (0,0))` 返回新内存，理解“零长度 pad 也要判等跳过”。

- **知识点 4：MoE 的 all-to-all vs all-reduce——通信量差 tp_size 倍**
  - **是什么**：Engram 的 DP lookup 支持 `all_reduce` / `reduce_scatter` / `all_to_all`。`all_to_all` 只发送每个 rank 自己那部分 hash 列，通信量是前两者的 `1/tp_size`。
  - **为什么重要**：MoE/专家并行的核心成本就是通信。理解“谁需要什么数据”决定了用哪种 collective。文档里说 8192 token 以上 `all_to_all` 快 8-62%，这就是选对 collective 的价值。
  - **动手试试**：
    ```python
    # 用 torch.distributed 概念模拟（单机多卡需 torchrun）
    # 伪代码：all_reduce 每个 rank 发全量；all_to_all 每个 rank 只发 1/world
    world = 4
    per_rank_bytes = 1024
    print("all_reduce 总通信量:", world * per_rank_bytes)
    print("all_to_all 总通信量:", world * (per_rank_bytes // world))
    ```
  - **预期结果**：看到 all_to_all 通信量是 all_reduce 的 1/world，理解为什么大 batch 下 all_to_all 更快。

- **知识点 5：Rust server 的 sidecar 生命周期——每节点一次**
  - **是什么**：Commit 15 让 `--sidecar` 在 Rust server 模式下每个节点只跑一个，而不是每个 scheduler 进程都跑。通过 `rust_server_sidecar(...)` 上下文管理器包住 `block_until_scheduler_exits()`。
  - **为什么重要**：这是“进程生命周期管理”的经典问题。多进程架构里，谁负责启动/回收子进程，很容易出现重复启动或泄漏。用 context manager 是 Python 里最清晰的写法。
  - **动手试试**：
    ```python
    from contextlib import contextmanager

    @contextmanager
    def sidecar(enabled: bool):
        if enabled:
            print("start sidecar")
        try:
            yield
        finally:
            if enabled:
                print("stop sidecar")

    with sidecar(True):
        print("scheduler running")
    ```
  - **预期结果**：看到 start → running → stop 的顺序，理解 context manager 如何保证清理逻辑一定执行。

## 🔥 重要更新

- **Commit 12 `[Fix] attention: Restore block pointers on Triton 3.9 for FLA kernels`**：Triton 3.9 删了 block pointer API，这个兼容层直接决定 FLA 系列 kernel 能否在新 Triton 上跑，影响面很大。
- **Commit 14 `[Feat] [NPU] moe: Add Ascend Ops backend`**：Ascend NPU 新增 `ascend_ops` MoE 后端，支持 prefill `allgather`/`alltoallv` 和 decode `low_latency`/`fused_op`，是国产硬件适配的重要一步。
- **Commit 8 `[Perf] [AMD] kimi-k3: Remove per-layer copies from the aiter MoE path`**：通过 `zero_copy_context` 复用 MoE 输出 buffer，消除每层拷贝，AMD 上 Kimi-K3 推理的直接性能收益。
- **Commit 13 `[Perf] dsv41: Pick the Engram DP lookup collective automatically`**：`auto` 模式根据 batch 形态自动选 all_to_all / reduce_scatter / all_reduce，文档给出 134/144 形状最优的实测数据。

## 📋 逐条解读

### 1. [Doc] [AMD] deepseek-v4: Update MI355X C1 mono-kernel recipe (#46966)
- **代码层面**：只改 `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`。把 c1 配方的 decode 从 `--tp 4` 改成 `--tp 8`，并新增一句：c1 要用 `lmsysorg/sglang-rocm:v0.5.22-rocm10-mi35x-20261028` 镜像，decode 命令加 `--enable-amd-monokernel`。
- **新手概念课堂**：Mono-kernel 就像“把多道小菜合成一个大锅菜”——把多个小 kernel 融合成一个，减少启动开销和内存往返。AMD 的 MI355X 上，这个融合对低并发（c1）延迟特别敏感的场景收益明显。
- **对你有什么影响**：如果你在 MI355X 上跑 DeepSeek-V4 的低延迟服务，照新配方改 `--tp 8` + 开 monokernel，延迟会更好；普通 NVIDIA 用户无感知。

### 2. [Doc] cookbook: Refresh SemiAnalysis AgentX launches to InferenceX main 67de4b34 (#46948)
- **代码层面**：改 `docs/src/snippets/agentx/...` 下的 JSX 生成文件，同步 SemiAnalysis InferenceX 仓库的 main commit 和 PR 列表，更新各模型的 launch 命令来源和验证状态。
- **新手概念课堂**：这是“文档即代码”的实践——cookbook 里的启动命令由脚本从上游 benchmark 仓库自动生成，避免手抄出错。JSX 文件里 `GENERATED ... do not edit by hand` 就是提示。
- **对你有什么影响**：看 cookbook 里的 AgentX 配方时，命令来源更可信、更新更及时；普通用户只是文档变准了。

### 3. [Chore] [AMD] mono-kernel: Bump sgl-amd-mono for MoRI PD decode (#46951)
- **代码层面**：改 `python/pyproject_other.toml`，把 `sgl-amd-mono` 的 git pin 从 `db7adb72...` 升到 `90c23551...`，用于 MoRI PD（Prefill-Decode 分离）decode 路径。
- **新手概念课堂**：PD 分离就像“后厨做菜”和“前台传菜”分开——prefill 算 prompt，decode 逐 token 生成，两者资源特征不同，分开调度能提高吞吐。MoRI 是 AMD 上的通信库。
- **对你有什么影响**：用 AMD + PD 分离部署的用户，升级依赖后 decode 路径的 monokernel 行为会更新；不装 `monokernels_hip` extra 的用户无影响。

### 4. [Chore] codeowners: Add paged experts maintainers (#46440)
- **代码层面**：改 `.github/CODEOWNERS`，给 `paged_experts.*` 和 `layers/moe/paged_experts/` 加上 `@N3mezis @BBuf` 作为 owner。
- **新手概念课堂**：CODEOWNERS 就像“图书馆分区管理员”——谁负责哪块代码，PR 会自动请求他们 review。Paged experts 是把专家权重分页管理，类似操作系统的虚拟内存分页。
- **对你有什么影响**：普通用户无感知；给 paged experts 提 PR 的同学会被自动 @ 到正确的维护者。

### 5. [Test] HiCache: Don't require a storage hit in the random-failure test (#46821)
- **代码层面**：改 `test/registered/hicache/test_hicache_storage_file_backend.py`，在子类里把继承来的 `test_basic_backup_and_prefetch = None` 禁用掉，因为注入故障后 prefetch 必然 miss，原测试的“要求 storage hit”前提不成立。
- **新手概念课堂**：HiCache 是“分层缓存”——GPU 显存 → 主机内存 → 存储，像书架（GPU）、仓库（主机内存）、远处图书馆（存储）。故障注入测试是故意弄坏一层，看系统能否优雅降级成 cache miss。
- **对你有什么影响**：只是测试更合理，不会误报；普通用户无感知。

### 6. [Perf] dsv41: Reorder the Engram all-to-all output with a Triton kernel (#46925)
- **代码层面**：新增 `engram_rows_to_token_major` Triton kernel 和注册项，替换原来的 `transpose + reshape` 重排；文档更新 all_to_all 的实测优势数据（134/144 形状最优）。
- **新手概念课堂**：all-to-all 的输出是“按源 rank 分块”的，但后续计算需要“按 token 连续”。`transpose + reshape` 会触发额外拷贝，Triton kernel 直接按目标布局读写，省一次 HBM 往返。
- **对你有什么影响**：DeepSeek-V4.1 用 Engram + DP attention 的用户，lookup 后的重排更快；普通模型无感知。

### 7. [Perf] dsv4: Skip the copy when the page-index table is already aligned (#46835)
- **代码层面**：`deepseek_v4_backend.py` 的 `_pad_last_dim` 加了一行：如果 `target_size == curr_size` 直接 `return x`，因为 `F.pad` 即使 pad 0 也会拷贝。
- **新手概念课堂**：page-index table 是 KV cache 的“页表”，对齐是为了 kernel 向量化读取。已经对齐还 pad 就是白拷贝——就像书已经按 32 本一箱装好了，你还重新装箱一遍。
- **对你有什么影响**：DeepSeek-V4 在 page-index 已对齐时省一次拷贝，decode 路径微小提速；普通用户无感知。

### 8. [Perf] [AMD] kimi-k3: Remove per-layer copies from the aiter MoE path (#33916)
- **代码层面**：`moe_runner/aiter.py` 里，当 `runner_input.output is None` 时，通过 `zero_copy_context.get_moe_output_spec(...)` 拿一个匹配的已发布 buffer 作为 `extra["output"]`，避免每层新建输出；`mxfp4.py` 新增 `_aiter_situ_uses_a8w4()` 判断；`kimi_k3.py` 调整 dense bf16 契约判断。
- **新手概念课堂**：MoE 每层都新建输出 tensor 就像“每道菜都换一个新盘子”——盘子（显存）分配和拷贝都是开销。复用 buffer 就是“洗完盘子接着用”。
- **对你有什么影响**：AMD 上跑 Kimi-K3 + aiter MoE 的用户，每层少一次拷贝，吞吐/延迟有直接收益。

### 9. [Doc] [AMD] cookbook: Add the Qwen3.5 MI355X FP8 agentic recipe (#46757)
- **代码层面**：`Qwen3.5.mdx` 新增 3.3.1 节，给出 MI355X 上 FP8 + MTP 的 agentic 长上下文配方，含并发 1-80 的 max running / graph batch / HiCache GB 表格和完整启动命令；更新 FP8 KV cache 警告措辞。
- **新手概念课堂**：Agentic 负载是“多轮对话 + 长上下文”，KV cache 很容易爆显存。HiCache 把冷 prefix 放主机内存，像“书桌放常用的书，书架放不常用的”。并发 32 以上用 `--hicache-size` 固定主机 tier 大小。
- **对你有什么影响**：在 MI355X 上部署 Qwen3.5-397B-FP8 做 agent 服务的用户，可以直接抄表里的并发-参数对应关系。

### 10. [Fix] moe: Broadcast scalar block scales correctly with swap_ab on Triton 3.9 (#45889)
- **代码层面**：`fused_moe_triton_kernels.py` 在 `BLOCK_SIZE_N <= group_n` 且 `swap_ab` 时，新增分支 `accumulator += tl.dot(a, b) * (a_scale * b_scale[None, :])`；新增测试文件覆盖 swap_ab × block_n 组合。
- **新手概念课堂**：swap_ab 是“把 A、B 矩阵换个位置算”的性能技巧，但 scale 的广播轴会跟着翻转。原来 `a_scale[:, None]`（按 token）变成标量，`b_scale`（按通道）变成 `[None, :]`。广播错了不报错，只算错。
- **对你有什么影响**：用 block-FP8 MoE + Triton 3.9 + swap_ab 的用户，精度 bug 被修复；不涉及该路径的用户无感知。

### 11. [Perf] [NVIDIA] custom-all-reduce: Enable 2shot_push in SM107 TP4/TP8/TP16 heuristic (#46197)
- **代码层面**：`custom_all_reduce_v2.py` 的 `_sm107_configs` 给 TP4/8/16 的 graph/eager 配置加 `two_shot_push_threshold=16MB`，并下调 one-shot 阈值（如 TP4 从 6.938MB 降到 1MB）。
- **新手概念课堂**：all-reduce 有 one-shot（一次发完）和 two-shot（分两轮）策略。小消息 one-shot 快，大消息 two-shot 省带宽。阈值就是“多大消息切换策略”的分界线。
- **对你有什么影响**：SM107（Rubin/VR）上 TP4/8/16 的 NCCL 替代通信更快；其他架构无感知。

### 12. [Fix] attention: Restore block pointers on Triton 3.9 for FLA kernels (#45470)
- **代码层面**：新增 `triton_block_ptr_compat.py`，检测 `tl.advance` 是否存在，不存在时用 `_BlockPtr` 聚合类和 `_materialize` 函数在普通指针上重建 block pointer 语义；`fused_norm_gate.py` 和 `fla/utils.py` 导入该兼容模块。
- **新手概念课堂**：block pointer 是 Triton 里“带边界检查的多维指针”，像“带围栏的田地”。Triton 3.9 把围栏拆了，兼容层用普通指针 + 手动算坐标 + mask 把围栏重新搭起来。
- **对你有什么影响**：升级到 Triton 3.9 的用户，FLA 系列 kernel 不再 import 失败；老 Triton 用户走原路径无影响。

### 13. [Perf] dsv41: Pick the Engram DP lookup collective automatically (#46917)
- **代码层面**：`engram` 的 `lookup` 默认从 `collective` 改为 `auto`，新增 `cols_per_rank` 判断；`deepseek_v4_hook.py` 里根据 DP/CP 配置和 batch 形态选 all_to_all / reduce_scatter / all_reduce；文档和 server_arguments 同步更新。
- **新手概念课堂**：三种 collective 像“三种送货方式”——all_reduce 每家都送全量再汇总，reduce_scatter 汇总后只留自己那份，all_to_all 只送对方要的那部分。数据量越大，all_to_all 越省。
- **对你有什么影响**：DeepSeek-V4.1 + Engram + DP attention 的用户，默认配置自动变快；想手动控制仍可显式指定。

### 14. [Feat] [NPU] moe: Add Ascend Ops backend for MoE prefill and decode (#46282)
- **代码层面**：新增 `ascend_ops` 作为 `--moe-a2a-backend` 和 `--speculative-moe-a2a-backend` 的选项，新增 `--moe-ascend-ops-config`（prefill: `allgather`/`alltoallv`，decode: `low_latency`/`fused_op`）；`deepseek_v4_hook.py` 的 supported backends 加入 `ascend_ops`；文档更新。
- **新手概念课堂**：Ascend Ops 是华为昇腾 NPU 上的融合算子后端。prefill 阶段 token 多、适合 allgather/alltoallv；decode 阶段逐 token、适合 low_latency/fused_op——就像“大批量走物流，小包裹走快递”。
- **对你有什么影响**：昇腾 NPU 用户多了一个 MoE 后端选择，可按 prefill/decode 分别调优；NVIDIA 用户无感知。

### 15. [Feat] rust_server: Run --sidecar once per node (#46635)
- **代码层面**：移除 `serving_hook.py` 里“Rust server 不支持 sidecar”的报错；`engine.py` 和 `http_server.py` 用 `with rust_server_sidecar(rust_server_owns_base_port):` 包住 scheduler 阻塞逻辑；`serving.py` 文档说明每节点一个 sidecar、`SGLANG_GRPC_ENDPOINT` 指向该节点第一个 listener。
- **新手概念课堂**：sidecar 是“挂在主服务旁边的辅助进程”，像“主厨旁边的配菜师”。多节点部署时，每个节点配一个就够，不需要每个 scheduler 进程都配。
- **对你有什么影响**：用 Rust server + `--sidecar` 的多节点用户，不再被报错拦住，且不会重复启动 sidecar。

### 16. [Perf] diffusion/ming: Reuse ordered RMSNorm in the DiT (#46705)
- **代码层面**：`rmsnorm_cast_before_weight_bitexact.py` 把 kernel 泛化：支持 `HEADS`/`TOKEN_STRIDE`/`EPS` 参数，宽度支持 128/3072/3840，eps 支持 1e-6/1e-5；Ming-Image 的 DiT 复用这个 bitexact RMSNorm。
- **新手概念课堂**：RMSNorm 是 Transformer 里的归一化层。“bitexact”意味着和参考实现逐 bit 一致——对 diffusion 生成质量很关键，差一点 rounding 图像就变了。复用 kernel 就是“同一把尺子量所有东西”。
- **对你有什么影响**：跑 Ming-Image diffusion 的用户，DiT 里 RMSNorm 更快且保持 bit 一致；普通 LLM 用户无感知。

### 17. [Fix] diffusion/moss: Preserve full latents for sequence parallelism (#46664)
- **代码层面**：`moss_image.py` 新增 `shard_latents_for_sp` 返回 `(latents, False)`，表示不在 pipeline 层切分 latents；测试改为在 stage 层调 `_preprocess_sp_latents` / `_postprocess_sp_latents`，验证 DiT 内部自己切分。
- **新手概念课堂**：序列并行（SP）是把长序列切到多卡。切分时机很关键——如果 RoPE 需要全图信息，就必须先建完整 RoPE 再切 token，否则位置编码错乱。
- **对你有什么影响**：MOSS-Image 用 SP 多卡生成的用户，图像质量修复；单卡用户无感知。

### 18. [Fix] dsv4: Refuse the draft metadata glue when the backend needs host lengths (#46777)
- **代码层面**：`deepseek_v4_backend.py` 的 `glue_draft_metadata` 在拒绝列表里加 `(self.needs_cpu_seq_lens, "host sequence lengths")`；同时删掉 `seq_lens_sum` 的冗余计算；测试新增对应 case。
- **新手概念课堂**：draft metadata glue 是“草稿模型和主模型共享元数据”的优化，但前提是后端不需要 host 侧 seq_lens（否则要 GPU→CPU 同步，glue 就不安全）。这就像“拼车可以，但有人要中途下车就不行”。
- **对你有什么影响**：DeepSeek-V4 draft + 需要 host seq_lens 的后端组合，会明确拒绝 glue 而不是静默出错；其他组合无感知。

### 19. [Fix] MegaMoE: stop double-dividing the DP prefill budget (#46267)
- **代码层面**：`mega_moe_hook.py` 里，attn DP 开启时 `local_chunked_prefill_size` 不再除以 `token_partition_size`（因为 `chunked_prefill_size` 已经是 per-rank 解析后的值），只改 `token_partition_name` 标注；测试新增 `test_megamoe_token_budget_uses_resolved_dp_chunked_prefill`。
- **新手概念课堂**：token budget 是“每个 rank 最多处理多少 token”。如果 `chunked_prefill_size` 已经是 per-rank 值，再除一次 DP size 就是“除两次”，预算被低估一半，导致误报配置错误。
- **对你有什么影响**：MegaMoE + attn DP 的用户，token budget 校验不再误报；其他用户无感知。

### 20. [Perf] MOSS-Image: Fuse joint-block QKV projections in high quality (#46641)
- **代码层面**：`rmsnorm_cast_before_weight_bitexact.py` 的 `_ordered_rstd` / `_norm_rope_kernel` 支持 `HEADS`/`TOKEN_STRIDE` 处理非连续布局；MOSS-Image 在 `quality=high` 时把 joint block 的 Q/K/V 三个投影合成一个 GEMM；文档说明 `exact`/`lossless` 仍走分开投影。
- **新手概念课堂**：QKV 融合是“三次小搬运合成一次大搬运”——三个 GEMM 合成一个，减少 kernel launch 和内存往返。但融合会改变 BF16 rounding 顺序，所以只在 `high` 质量档开启，`exact`/`lossless` 保持逐 bit 一致。
- **对你有什么影响**：MOSS-Image 用 `quality=high` 的用户生成更快，但图像可能有微小差异；追求 bit 一致的用户用 `exact`/`lossless` 不受影响。

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)