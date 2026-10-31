# 🚀 vLLM & SGLang 每日更新 — 2026-10-31

> 自动生成于 2026-10-31 07:28 UTC | AI 解读: ✅ 含代码解读

## vllm
## 💡 今日关键词

- **关键词：Direct Weight Reload（直接权重热重载）**
  - 一句话解释：训练完新权重后，不用重启推理服务，直接写进正在运行的模型参数里。
  - 为什么社区在关注：RLHF / 在线学习场景越来越普遍，重启服务意味着几秒钟到几分钟的中断，热重载能把"换脑"变成无感操作。这说明 vLLM 正在从"纯推理服务器"往"训练-推理闭环"方向走。
  - 对新手意味着什么：如果你学强化学习或在线微调，理解"权重张量在内存中的生命周期"会成为刚需技能，而不是只懂 `load_model()` 就完事。

- **关键词：ROCm / AMD 适配（gfx950、MI355）**
  - 一句话解释：同一套 vLLM 代码要能在 AMD 的 GPU 上跑出接近 NVIDIA 的效果。
  - 为什么社区在关注：今天 20 个 commit 里有 4 个和 ROCm 相关（AITER MLA、KDA gluon、SWA replay、Step-5 支持），异构计算已经从"能跑"进入"逐个 kernel 抠性能"阶段。
  - 对新手意味着什么：只懂 CUDA 已经不够了，Triton 这种跨后端 kernel 语言的价值在上升，是差异化竞争点。

- **关键词：KV Cache 跨节点传输（NIXL / PP / HMA）**
  - 一句话解释：把一台机器算好的 KV 缓存搬到另一台机器上继续用，省掉重复计算。
  - 为什么社区在关注：NIXL 连发两个 commit（按 layer 配对 + 版本升级到 1.5.2），说明 PD 分离（Prefill-Decode disaggregation）架构正在成为大规模部署标配。
  - 对新手意味着什么：分布式推理不再只是"多卡切模型"，而是"多机流水线 + 缓存搬运"，这是后端工程师的高价值方向。

- **关键词：流式解析器性能（Parser Streaming）**
  - 一句话解释：模型一边吐 token，前端一边解析工具调用参数，不能让解析耗时随输出长度线性增长。
  - 为什么社区在关注：连续 3 个 commit（Qwen3、GLM4.7、Responses API）都在修同一个 O(n²) 问题，说明 Agent 场景下长输出 + 工具调用已成主流负载。
  - 对新手意味着什么：写前端/网关代码时，"增量解析"和"幂等重放"是必须掌握的工程直觉。

- **关键词：模型适配潮（Canary、Step-5、DeepSeek-V4.1）**
  - 一句话解释：新模型层出不穷，vLLM 要持续跟进架构支持。
  - 为什么社区在关注：一天之内新增语音模型、多模态模型、MoE 大模型支持，说明 vLLM 的"模型注册表"已经成为事实标准。
  - 对新手意味着什么：读一个 `vllm/model_executor/models/xxx.py` 就是最好的"Transformer 变体教科书"。

---

## 🧪 重要知识点 & 动手实验

- **知识点 1：权重热重载的"地址不变"不变量（Commit 1）**
  - **是什么**：模型加载后，某些参数会被后处理改布局、派生新张量、甚至被"退休"（retire）。热重载时必须保证：重载后等价于冷启动，且原有内存地址不变（否则 CUDA Graph 会指向旧数据）。
  - **为什么重要**：如果你直接 `param.data.copy_(new_weight)`，遇到"后处理会转置权重"的模型（如 MLA 的 W_UK/W_UV），新权重会被再转置一次，结果全错。不懂这个坑，热更新后模型输出会变成乱码。
  - **动手试试**：
    ```python
    # 模拟"后处理会改布局"的陷阱
    import torch
    w = torch.arange(6.).reshape(2, 3)      # 原始权重
    w_processed = w.t().contiguous()         # 后处理：转置
    new_w = torch.ones(2, 3)                 # 新 checkpoint 权重
    # 错误做法：直接覆盖 processed 存储
    w_processed.copy_(new_w)                 # 形状对不上会报错，或语义错误
    # 正确做法：先按 checkpoint 布局写入，再重跑后处理
    w.copy_(new_w)
    print(w.t().contiguous())                # 等价于冷启动
    ```
  - **预期结果**：你会看到"直接写处理后张量"和"写原始张量再重跑后处理"结果不同，理解为什么 vLLM 要引入 `begin_direct_reload / finish_direct_reload` 这套生命周期。

- **知识点 2：Triton kernel 里的 int32 溢出（Commit 11）**
  - **是什么**：Triton 的 `tl.program_id(0)` 默认是 int32，当行数 × 列偏移超过 2^31 时，`row * stride` 会溢出成负数，导致写错内存或越界。
  - **为什么重要**：大模型激活张量动辄几十 GB，`num_tokens × hidden_dim` 很容易越过 21 亿。这是"小模型测不出、大模型线上崩"的经典 bug，而且往往表现为随机乱码而非报错。
  - **动手试试**：
    ```python
    import torch, triton, triton.language as tl

    @triton.jit
    def bad_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)          # int32
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask), mask=mask)

    @triton.jit
    def good_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0).to(tl.int64)   # 关键一行
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask), mask=mask)
    # 用超大 N 测试（需足够显存），观察 bad 版本是否写错位置
    ```
  - **预期结果**：在超大张量上，`bad_kernel` 会出现部分数据未写入或写错位置；`good_kernel` 正常。你会明白为什么 Commit 11 只是加了 `.to(tl.int64)` 就是一个"Kernel"级别的修复。

- **知识点 3：流式解析的 O(n²) 陷阱（Commit 14/15/17）**
  - **是什么**：每来一个新 token 就把"到目前为止的全部文本"重新解析一遍，复杂度是 O(n²)。正确做法是只解析新增部分，并缓存已关闭参数的结果。
  - **为什么重要**：Agent 场景下工具调用参数可能几千 token，O(n²) 会让首 token 延迟之外的"流式输出"越来越卡，用户感知为"越到后面越慢"。
  - **动手试试**：
    ```python
    import re, time
    text = "<parameter=content>" + "x" * 20000 + "</parameter>"
    pat = re.compile(r"<parameter=(\w+)>(.*?)</parameter>", re.S)

    # 朴素做法：每次切片都全量 finditer
    t0 = time.time()
    for i in range(0, len(text), 100):
        list(pat.finditer(text[:i]))
    print("naive:", time.time() - t0)

    # 增量做法：只对新增窗口做匹配
    t0 = time.time()
    for i in range(0, len(text), 100):
        pat.finditer(text[max(0, i-200):i])
    print("incremental:", time.time() - t0)
    ```
  - **预期结果**：朴素做法耗时随长度平方增长，增量做法近似线性。这就是 Commit 15 里 `slot.open_value_streams = False` 想表达的"已关闭的值不再重算"。

- **知识点 4：EPLB 冗余专家的自动推导（Commit 18）**
  - **是什么**：MoE 模型里专家数不一定能被 EP（专家并行）组大小整除，EPLB（专家负载均衡）需要补一些"冗余专家"让总数整除，否则负载均衡算法会退化。
  - **为什么重要**：不懂这个，手动配 `num_redundant_experts` 容易配成负收益（多占显存但不均衡）。Commit 18 把它默认成"最小合法值"，公式是 `(ep_size - num_experts % ep_size) % ep_size`。
  - **动手试试**：
    ```python
    def min_redundant(num_experts, tp, pcp, dp):
        ep = tp * pcp * dp
        return (-num_experts) % ep   # Python 取模天然给出最小非负补数

    for args in [(8,8,1,1), (8,8,1,2), (8,2,1,3), (10,4,1,2), (8,2,3,1)]:
        print(args, "->", min_redundant(*args))
    ```
  - **预期结果**：输出与 Commit 18 测试用例的 `expected` 列完全一致（0, 8, 4, 6, 4）。你会理解为什么 `ep_size` 必须把 TP × PCP × DP 都算进去。

- **知识点 5：KV 缓存按 layer 配对而非按 index 配对（Commit 3）**
  - **是什么**：在 PP（流水线并行）下，P 节点只持有部分层，D 节点持有全部层，两边 KV region 的数量和顺序都对不上。必须用"层名"做键来配对，而不是数组下标。
  - **为什么重要**：如果按 index 配对，PP 场景下会把第 3 层的 KV 写到第 5 层的位置，模型输出会"看起来能跑但语义全错"，是最难排查的一类 bug。
  - **动手试试**：
    ```python
    # 模拟 P 和 D 的 region 元数据
    D = {0: ["L0","L0.swa"], 1: ["L1"], 2: ["L2"]}
    P = {0: ["L2"]}
    # 按 index 配对（错误）：P[0] 对 D[0] -> 把 L2 写到 L0 位置
    # 按 layer 配对（正确）：
    d_layer_to_region = {l: r for r, ls in D.items() for l in ls}
    pairs = [(r, d_layer_to_region[l]) for r, ls in P.items() for l in ls]
    print(pairs)   # [(0, 2)] -> P 的 region 0 对 D 的 region 2
    ```
  - **预期结果**：输出 `[(0, 2)]`，直观理解 Commit 3 里 `_pair_regions` 的核心逻辑。

---

## 🔥 重要更新

- **Commit 1（Direct weight reload）**：RL 场景下"不重启换权重"的里程碑，涉及内存地址、后处理重放、退休参数等一整套不变量，是理解 vLLM 模型生命周期的最佳入口。
- **Commit 3（NIXL 按 layer 配对）**：修的是 PD 分离 + PP + HMA 三重叠加下的正确性 bug，这类"能跑但结果错"的问题对生产环境影响最大。
- **Commit 18（EPLB 冗余专家默认值）**：一个配置默认值的改动，但直接影响 MoE 大模型在多卡上的负载均衡效果，属于"改一行、收益大"的典型。
- **Commit 14/15/17（流式解析三连修）**：同一天三个 commit 修同一类 O(n²) 问题，说明 Agent 长输出场景已经成为性能瓶颈重灾区。

---

## 📋 逐条解读

### 1. [RL][Model Loader] Direct weight reload: write checkpoint tensors into the live model

- **代码层面**：新增 `vllm/model_executor/model_loader/reload/direct.py`，提供 `begin_direct_reload / finish_direct_reload` 两个入口；新增 578 行测试 `test_direct_reload.py`，用玩具模型模拟 6 种真实模型的"后处理陷阱"（派生张量、转置、无 weight_loader 写入、退休参数、打包、padding）。
- **新手概念课堂**：把模型想象成一台正在运转的机器。普通加载是"停机换零件"，热重载是"边转边换"。难点在于有些零件换上去后还要经过一道"校准工序"（后处理），你换的时候必须按"未校准"的形态放进去，不能直接放"已校准"的成品。
- **对你有什么影响**：如果你在做 RLHF 或在线微调，未来可以在不中断服务的情况下更新策略模型；普通用户感知不到，但服务可用性会提升。

### 2. [Model] Add NVIDIA Canary (nvidia/canary-1b-v2) speech-to-text support

- **代码层面**：新增 `vllm/model_executor/models/canary.py`（652 行），实现 FastConformer 编码器 + Transformer 解码器；在 `docs/models/supported_models.md` 和 `tests/models/registry.py` 注册；在转录正确性测试里加入 WER 基线 12.34。
- **新手概念课堂**：语音识别模型 = "耳朵"（编码器，把声音波形变成特征）+ "嘴巴"（解码器，把特征变成文字）。Canary 是 NVIDIA 的英德法西多语种模型，支持翻译式转录。
- **对你有什么影响**：可以直接 `vllm serve nvidia/canary-1b-v2` 起一个语音转文字服务，支持多语种，不用再单独部署 Whisper。

### 3. [KV Connector][NIXL] Pair KV regions by layer for every transfer

- **代码层面**：重写 `NixlPushConnector` 的 `_pair_regions` 逻辑，从"按 region index 对齐"改为"按 layer name 查表配对"，并在握手元数据里用 `region_members` 广告每层属于哪个 region；文档 `nixl_kv_push_connector.md` 同步更新配对示意图。
- **新手概念课堂**：两台机器搬家具，如果两边房间编号规则不同（P 只有 3 个房间、D 有 10 个），按编号搬就会放错。正确做法是每件家具上贴"我是第几层的"，按标签找位置。
- **对你有什么影响**：用 PD 分离 + 流水线并行部署大模型时，KV 传输不再错位，长上下文场景的输出正确性有保障。

### 4. [Bugfix][gRPC] Drain in-flight RPCs and engine work within --shutdown-timeout

- **代码层面**：新增 `tests/entrypoints/launchers/test_grpc_shutdown.py`（219 行），实现 `_drain_and_stop` 的优雅关闭：先停止接收新 RPC，再等待 in-flight 请求完成，超时才强杀，全部在 `--shutdown-timeout` 内完成。
- **新手概念课堂**：餐厅打烊不是立刻关灯赶人，而是"停止接客 → 等现有客人吃完 → 到点还没吃完才清场"。gRPC 服务的优雅关闭同理。
- **对你有什么影响**：滚动更新或缩容时，正在生成的请求不会被硬切断，用户不会看到"连接被重置"。

### 5. [Feature][DSv4.1] Enable PCP with PP or DP

- **代码层面**：`vllm/models/deepseek_v41/nvidia/model.py` 里删掉 `pipeline_parallel_size == 1` 的限制条件，让 PCP（Prefill Context Parallel）可以和 PP 或 DP 组合；新增 8 卡集成测试 `test_pcp_generates_with_pipeline_or_data_parallelism`。
- **新手概念课堂**：PCP 是把一个超长 prompt 切给多张卡并行做 prefill。之前它只能单独用，现在可以和"流水线并行"（PP，按层切）或"数据并行"（DP，按请求切）叠加，像乐高积木一样组合。
- **对你有什么影响**：部署 DeepSeek-V4.1 长上下文服务时，并行策略的排列组合更灵活，可以按硬件拓扑选最优方案。

### 6. [Perf][GLM-5.3-Flash] Update FlashKDA pin, consume row-strided QKV directly

- **代码层面**：升级 FlashKDA 的 git pin；`vllm/models/glm5next/common/kda.py` 里去掉 `q.contiguous() / k.contiguous() / v.contiguous()` 三次拷贝，直接把 row-strided 视图传给 kernel。
- **新手概念课堂**：`contiguous()` 是"把数据重新排成连续内存"，代价是一次显存拷贝。如果 kernel 本身支持带 stride 读取，这次拷贝就是纯浪费。
- **对你有什么影响**：GLM-5.3-Flash 的 prefill 阶段少三次全量拷贝，长 prompt 场景延迟下降。

### 7. [NIXL] Bump NIXL version to 1.5.2

- **代码层面**：`requirements/kv_connectors*.txt` 和 `docker/Dockerfile.xpu` 里把 `nixl` 从 1.5.0 升到 1.5.2。
- **新手概念课堂**：NIXL 是 NVIDIA 的"跨节点内存搬运库"，相当于分布式系统里的"快递公司"。升级版本通常是修 bug 或加新传输后端。
- **对你有什么影响**：使用 KV Connector 的部署会拿到 1.5.2 的稳定性修复，无 API 变化。

### 8. [Bugfix][ROCm] Size AITER MLA persistent metadata for every decode query length

- **代码层面**：`rocm_aiter_mla` 的 metadata builder 从"按 `reorder_batch_threshold` 一次性分配"改为"遍历 1..threshold 每个 qlen 分别算需求，取最大值"；新增测试 `test_builder_sizes_metadata_for_every_decode_qlen`，模拟 gfx950 上 bf16 小块的 64 行 tile 行为。
- **新手概念课堂**：预分配缓冲区时，如果只按"最大 qlen"算，遇到"小 qlen 反而需要更多 tile"的硬件特性（比如 7 行要 2 个 tile、8 行只要 1 个），就会缓冲区不足。
- **对你有什么影响**：ROCm + MLA 模型 + 投机解码（DSpark）组合下不再出现 metadata 越界导致的崩溃或乱码。

### 9. [ROCm][Kimi-K3] Keep KDA prefill checkpoints with the gluon backend

- **代码层面**：`vllm/models/kimi_k3/amd/kda.py` 里让 gluon 后端在"带 checkpoint 的 prefill 步骤"自动切到 fused kernel（因为 AITER 的 `chunk_kda` 不支持导出 checkpoint），其余步骤仍用 gluon；新增两个测试验证跨后端续算的一致性。
- **新手概念课堂**：chunked prefill 像"分段读一本书"，每段结束要存书签（checkpoint）。如果某个后端不支持存书签，就在需要存书签的那一段换一个支持的后端，读完再换回来。
- **对你有什么影响**：Kimi-K3 在 AMD 上开 prefix caching 时，prefill checkpoint 不再丢失，长文档问答的首 token 延迟更稳定。

### 10. [ROCm][DSv4.1][Perf] Enable prefix-cache SWA replay on gfx950

- **代码层面**：`vllm/models/deepseek_v41/attention.py` 里让 SWA（滑动窗口注意力）在 CDNA4（gfx950）上发布"bounded replay" spec，使 prefix cache 命中时只重算窗口部分而非整段；新增 106 行测试验证 CDNA3 上仍不启用。
- **新手概念课堂**：SWA 只看最近 N 个 token。prefix cache 命中后，如果直接把缓存全用上，窗口外的旧 token 会污染结果；bounded replay 就是"只重算窗口内那一段"。
- **对你有什么影响**：DeepSeek-V4.1 在 MI355 上开 prefix caching 的长对话场景，TTFT 明显下降。

### 11. [Kernel] Qwen4Exp: compute HyperConnection and PLE gate row offsets in int64

- **代码层面**：`vllm/models/qwen4_exp/{amd,nvidia}/ops/hc.py` 和 `common/ops/ple.py` 里所有 `tl.program_id(0)` 后面加 `.to(tl.int64)`，共 6 处。
- **新手概念课堂**：见知识点 2。int32 最大约 21 亿，`row * hidden_dim` 很容易溢出。
- **对你有什么影响**：Qwen4Exp 在超长序列或超大 batch 下不再出现随机乱码（此前是"小规模测不出、大规模偶发"的隐蔽 bug）。

### 12. [Pooling] Speed up ColBERT reranking with batched token heads

- **代码层面**：`TokenPooler` 把多个请求的 token embedding 投影（`nn.Linear`）合并成一次 batched 调用，只在 activation 参数不一致时才分组；新增测试验证"混合参数下投影只调 1 次、activation 按组调"。
- **新手概念课堂**：ColBERT 重排序要把 query 和 doc 的每个 token 都过一遍投影层。逐个请求调用 GPU 利用率低，合并成一个大 batch 才能喂饱 GPU。
- **对你有什么影响**：RAG 场景下的 ColBERT rerank 吞吐提升，检索延迟下降。

### 13. [Perf][LoRA][MoE] Use the shrink/expand kernels for MoE-LoRA batches of up to 64 tokens

- **代码层面**：`vllm/lora/ops/triton_ops/` 里新增常量 `FUSED_MOE_LORA_TWO_KERNEL_MAX_TOKENS = 64`，小 batch（≤64 token）走"shrink + expand"两段式 kernel 而非 one-shot；测试从 `num_tokens=[16,64]` 改为 `[288,320]` 以覆盖 NPID>1 分支。
- **新手概念课堂**：MoE-LoRA 的计算是"先降维（shrink）到 rank，再升维（expand）回 hidden"。小 batch 下 one-shot kernel 启动开销占比高，拆成两个小 kernel 反而更快。
- **对你有什么影响**：多 LoRA 多专家的在线服务，短请求（如分类、意图识别）延迟下降。

### 14. [Bugfix][Parser] Parse closed Qwen3 tool parameters once while streaming

- **代码层面**：`vllm/parser/qwen3.py` 里对已闭合的 `<parameter>` 做缓存，避免每次新 token 到达都重新 `finditer` 整个已解析文本；新增 `test_closed_params_are_parsed_once` 用计数器验证匹配次数不随输出长度增长。
- **新手概念课堂**：见知识点 3。模型陷入循环重复输出同一参数时，O(n²) 会变成 O(n³) 级别的卡顿。
- **对你有什么影响**：Qwen3 工具调用流式输出不再"越到后面越慢"，长参数（如写文件内容）场景尤其明显。

### 15. [Perf][Parser] Skip tool-argument conversions after a replaced value

- **代码层面**：`vllm/parser/engine/parser_engine.py` 的 `_compute_arg_delta` 里，当检测到"重复 key 替换了已流式的值"时，设置 `slot.open_value_streams = False`，后续只扩展新值的 delta 跳过 arg_converter 调用。
- **新手概念课堂**：模型有时会先输出 `content="abc"`，再输出 `content="abcdef"`。第二次是"替换"而非"追加"，如果每次都重新跑 JSON 转换器，成本会翻倍。
- **对你有什么影响**：GLM4.7-MoE 等模型的工具调用流式解析 CPU 占用下降，高并发 Agent 服务受益。

### 16. [Frontend] Reject Anthropic-defined tools by type in /v1/messages

- **代码层面**：`vllm/entrypoints/anthropic/` 的请求校验里，按 `tool.type` 前缀（`web_search_*`、`bash_*` 等）识别 Anthropic 服务端工具并返回 400；文档 `claude_code.md` 补充说明"vLLM 不运行服务端工具，请用 MCP"。
- **新手概念课堂**：Anthropic 的 `web_search` 工具是"在 Anthropic 服务器上跑的"，vLLM 只是推理引擎，没有联网搜索能力。之前静默忽略会导致 Claude Code 用户困惑。
- **对你有什么影响**：用 Claude Code 连 vLLM 时，WebSearch 会明确报错并提示用 MCP 替代，不再"看起来成功但没结果"。

### 17. [Perf][Frontend] Decode streamed custom-tool input incrementally

- **代码层面**：`vllm/entrypoints/openai/responses/utils.py` 新增 `decode_custom_tool_input_prefix`，`streaming_events.py` 里对 custom tool 的 `input` 字段做增量解码而非每次全量 `partial_json_parser.loads`；新增 200 轮随机 fuzz 测试保证与全量解码等价。
- **新手概念课堂**：Responses API 的 custom tool 参数是一个 JSON 字符串，流式输出时不能等全部到齐再 `json.loads`，要"边到边解"，但必须保证和最终全量解析结果一致。
- **对你有什么影响**：OpenAI Responses API 的 custom tool 流式场景，CPU 开销从 O(n²) 降到 O(n)。

### 18. [Feature] Default EPLB num_redundant_experts to minimum valid value

- **代码层面**：`vllm/config/parallel.py` 里 `EPLBConfig.num_redundant_experts` 默认改为 `None`，在 `ParallelConfig` 构造时按 `(-num_experts) % (tp*pcp*dp)` 自动算最小合法值；关闭 EPLB 时强制为 0 并校验。
- **新手概念课堂**：见知识点 4。EP 组大小不能整除专家数时，负载均衡算法会退化，补几个"冗余专家"让总数整除即可。
- **对你有什么影响**：开 EPLB 不用再手算 `num_redundant_experts`，默认就是最优值；配错（如 EPLB 关闭却设非零）会直接报错而非静默异常。

### 19. [Config] Move V2-only checks into V1 model runner unsupported features

- **代码层面**：`vllm/config/vllm.py` 里把 HiSparse、watermarking 等"仅 V2 支持"的检查从分散的 property 里统一收拢到 `_get_v1_model_runner_unsupported_features` 列表；测试从"检查 property 抛错"改为"检查列表内容"。
- **新手概念课堂**：V1 和 V2 是两代 model runner。某些新特性只在 V2 有，之前是"用到才报错"，现在改成"启动时统一检查并列出所有不兼容项"，报错更友好。
- **对你有什么影响**：误用 V1 + V2-only 特性时，错误信息会一次性列出所有冲突项，而不是修一个报一个。

### 20. [ROCm][Model] Step-5-Preview support on ROCm

- **代码层面**：新增 `.buildkite/test-amd.yaml` 里的 MI355 Step5 测试步骤；`vllm/models/step5/` 相关代码适配 ROCm 的 `rocm_aiter_unified_attn` 后端；warmup 脚本 `step5_dsa_warmup.py` 加入依赖列表。
- **新手概念课堂**：Step-5 是阶跃星辰的多模态模型，DSA（Dynamic Sparse Attention）需要在 ROCm 上找到对应的 attention backend。适配工作主要是"把 NVIDIA 专用 kernel 换成 AITER 的等价实现"。
- **对你有什么影响**：AMD MI355 用户可以直接跑 Step-5-Preview，不用再切回 NVIDIA 卡。

## sglang
## 💡 今日关键词

- **关键词：HiCache / 分层 KV 缓存**
  - 一句话通俗解释：把 KV cache 分层放在 GPU 显存、主机内存甚至磁盘上，用“命中率”换取更大的上下文和更高的吞吐。
  - 为什么社区在关注：今天有 3 个 commit（#43110、#47543、#44718）都在修 HiCache 的正确性和性能，说明分层缓存已经从“能跑”进入“跑得稳、跑得快”的阶段。
  - 对新手意味着什么：如果你学推理系统，KV cache 管理是最值得深入的方向之一——它同时涉及显存管理、I/O、并发调度和正确性验证。

- **关键词：异构硬件适配（AMD / NPU / GB300）**
  - 一句话通俗解释：同一套推理框架要跑在 NVIDIA、AMD、昇腾 NPU 等不同芯片上，需要针对每家的算子、量化格式和内存模型做适配。
  - 为什么社区在关注：今天有 AMD（#44718、#47408）、NPU（#44442、#45302、#40890）、GB300（#45876）多个方向的 commit，异构适配是当前最密集的工作流。
  - 对新手意味着什么：只懂 CUDA 已经不够了，理解“同一算法在不同硬件上的实现差异”是稀缺能力。

- **关键词：量化推理（MXFP4 / W4A16 / W8A8S / UE8M0）**
  - 一句话通俗解释：把模型权重和激活从 16 位浮点压到 4 位或 8 位，省显存、提吞吐，但需要处理 scale 和反量化。
  - 为什么社区在关注：#47337 修 dummy 权重的 scale、#47408 修 AMD 上 MXFP4 的 W4A16 回退路径、#44442 加 NPU W8A8S 支持——量化格式越来越多，边界情况也越来越多。
  - 对新手意味着什么：量化不只是“调个参数”，你需要理解 block scale、per-channel scale、反量化 kernel 的实现细节。

- **关键词：调度器不变量与可观测性**
  - 一句话通俗解释：调度器内部维护“哪些请求正在跑”这类状态，用不变量检查（invariant check）在 CI 里自动发现状态不一致的 bug。
  - 为什么社区在关注：#43899 引入 `RunningReqs` 所有者列表并加了 CI 默认开启的一致性检查，说明调度器复杂度已经高到需要“运行时断言”来兜底。
  - 对新手意味着什么：写并发系统时，“状态一致性检查”比“事后 debug”便宜得多，这是工程成熟度的标志。

- **关键词：确定性（Determinism）**
  - 一句话通俗解释：同样的输入必须得到同样的输出，哪怕 GPU 上有多个线程竞争。
  - 为什么社区在关注：#45174 修了 top-k 在分数相同时的 tie-breaking，让结果可复现——这对评测和调试至关重要。
  - 对新手意味着什么：并行算法里“相等值怎么排序”是最容易被忽略的坑，学会用 block id / lane id 做确定性排序是必备技能。

---

## 🧪 重要知识点 & 动手实验

- **知识点 1：KV Cache 分层与“派生 sidecar”**
  - **是什么**：HiCache 把 KV cache 分成多个 pool（比如 KV pool、Mamba 状态 pool），某些 pool 的索引是从另一个 pool“派生”出来的（sidecar）。当源 pool 被拆成多次传输时，派生 pool 必须覆盖所有源传输。
  - **为什么重要**：如果只取第一个源传输，派生 pool 的索引就会漏掉一部分，导致缓存命中错误或数据错位——这是典型的“看起来能跑但结果不对”的 bug。
  - **动手试试**：读 `python/sglang/srt/mem_cache/hicache_storage.py` 里新增的 `merge_source_transfers`，然后写一个小脚本模拟：
    ```python
    from dataclasses import dataclass
    from typing import Optional

    @dataclass
    class PoolTransfer:
        name: str
        host_indices: Optional[list] = None
        indices_from_pool: Optional[str] = None

    def merge(transfers, pool):
        sources = [t for t in transfers if t.indices_from_pool is None and t.name == pool]
        if len(sources) <= 1:
            return sources[0] if sources else None
        merged = []
        for t in sources:
            merged.extend(t.host_indices)
        return PoolTransfer(name=pool, host_indices=merged)

    ts = [
        PoolTransfer("KV", [1, 2]),
        PoolTransfer("KV", [3, 4]),
        PoolTransfer("MAMBA", None, "KV"),
    ]
    print(merge(ts, "KV"))
    ```
  - **预期结果**：输出 `PoolTransfer(name='KV', host_indices=[1, 2, 3, 4])`，你会理解“合并所有源传输”的含义。

- **知识点 2：量化中的“中性 scale”**
  - **是什么**：UE8M0 格式的 block scale 是 uint8，值 127 表示 2^0 = 1.0（中性）。如果初始化成 0，权重会被缩放成 2^-127，几乎归零。
  - **为什么重要**：dummy weight 初始化用于 profiling 和显存估算，如果 scale 是 0，测出来的数值全是 0，会掩盖真实的内存/计算行为。
  - **动手试试**：
    ```python
    import torch

    def init_dummy(param):
        if param.dtype == torch.uint8:
            param.fill_(127)
        else:
            param.uniform_(0.25, 0.75)

    scale = torch.zeros(4, 2, dtype=torch.uint8)
    init_dummy(scale)
    print(scale)
    # 模拟反量化：2^(127-127) = 1.0
    print(torch.pow(2.0, scale.float() - 127))
    ```
  - **预期结果**：scale 全是 127，反量化后全是 1.0，说明“中性”意味着不改变数值。

- **知识点 3：调度器不变量检查（Invariant Check）**
  - **是什么**：在每次调度后断言“running_reqs 集合 == 所有 batch 里请求的并集”，一旦不一致就报错。
  - **为什么重要**：调度器有 running batch、waiting queue、chunked prefill 等多个状态，请求在它们之间流转，很容易出现“请求丢了”或“请求重复”的 bug。CI 默认开启检查能在合入前抓住问题。
  - **动手试试**：写一个最小化的“集合不变量”演示：
    ```python
    class RunningReqs:
        def __init__(self):
            self._set = set()
        def join(self, reqs):
            self._set.update(reqs)
        def leave(self, reqs):
            self._set.difference_update(reqs)
        def snapshot(self):
            return set(self._set)

    rr = RunningReqs()
    batch_a = {"r1", "r2"}
    batch_b = {"r3"}
    rr.join(batch_a)
    rr.join(batch_b)
    assert rr.snapshot() == batch_a | batch_b, "invariant violated"
    print("OK:", rr.snapshot())
    ```
  - **预期结果**：打印 `OK: {'r1', 'r2', 'r3'}`。把 `join(batch_b)` 注释掉再跑，断言会失败——这就是 CI 里能抓到的 bug。

- **知识点 4：确定性 top-k 的 tie-breaking**
  - **是什么**：当多个候选分数完全相同时，用 block id（或 lane id）作为第二排序键，保证“低 id 优先”，从而每次运行结果一致。
  - **为什么重要**：GPU 上原子操作和并行归约的顺序不确定，如果不加 tie-break，同样的输入可能选出不同的 top-k，导致评测结果抖动、复现困难。
  - **动手试试**：用 Python 模拟“按 (score, id) 排序”：
    ```python
    candidates = [(0.9, 5), (0.9, 2), (0.9, 7), (0.8, 1)]
    # 确定性：分数降序，id 升序
    top2 = sorted(candidates, key=lambda x: (-x[0], x[1]))[:2]
    print(top2)  # [(0.9, 2), (0.9, 5)]
    ```
  - **预期结果**：输出 `[(0.9, 2), (0.9, 5)]`，而不是随机顺序的 `(0.9, 5), (0.9, 7)`。这就是 commit #45174 在 CUDA kernel 里做的事。

- **知识点 5：mlock 与页老化（Page Aging）**
  - **是什么**：`mlock` 把内存页锁定在物理内存中，不让操作系统换出；被锁定的页位于“不可回收 LRU”，不会被页老化扫描清除 accessed 标志。
  - **为什么重要**：GPU 通过 ATS 直接读主机内存时，如果页的 accessed 标志被清除，下一次访问会触发缺页中断，拖慢拷贝。锁定页能避免这个问题。
  - **动手试试**（Linux）：
    ```bash
    # 查看当前进程的 memlock 限制
    ulimit -l
    # 查看已锁定内存
    grep VmLck /proc/self/status
    ```
    然后在 Python 里用 `ctypes` 调 `mlock`（需要权限）：
    ```python
    import ctypes, mmap
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    buf = mmap.mmap(-1, 4096)
    addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
    ret = libc.mlock(ctypes.c_void_p(addr), ctypes.c_size_t(4096))
    print("mlock ret:", ret, "errno:", ctypes.get_errno())
    ```
  - **预期结果**：`ret == 0` 表示成功；如果返回 -1 且 errno 是 `ENOMEM`，说明 `RLIMIT_MEMLOCK` 太小——这正是 commit #47543 里 `_memlock_state()` 要打印的诊断信息。

---

## 🔥 重要更新

- **#43899 调度器引入 RunningReqs 所有者列表 + CI 默认不变量检查**：这是架构级重构，把“哪些请求在跑”从多个 batch 的隐式并集变成显式维护的集合，并在 CI 里自动校验，能系统性防止请求丢失/重复类 bug。
- **#47543 HiCache 用 mlock 锁定 pinned host buffer**：解决了 ATS 场景下页老化导致的性能抖动，是主机内存与 GPU 直连场景的重要优化。
- **#47408 AMD Quark MXFP4 在 gfx1151 上回退到 W4A16 并修复 EP>1 崩溃**：说明量化格式与硬件能力的匹配需要显式回退路径，且回退路径也要做 SwiGLU 变体的防御性检查。
- **#45174 Minimax-M3 top-k 确定性 tie-breaking**：虽然只改了一个 CUDA kernel，但“可复现性”对评测和生产调试价值极高。

---

## 📋 逐条解读

### 1. [Fix] hicache: Resolve derived sidecars over every source transfer (#43110)
- **代码层面**：在 `hicache_storage.py` 新增 `merge_source_transfers()`，把同一 pool 的多次独立传输合并成一个 span（`torch.cat` 拼接 host/device indices 和 keys）；`hybrid_cache_controller.py` 里把 `source_len[t.name] = ...` 改成累加，并用 `merge_source_transfers` 替代原来只取第一个源传输的 `next(...)`。
- **新手概念课堂**：就像你往仓库分三批送货，派生清单（sidecar）必须把三批货都登记上，不能只登记第一批。否则盘点时对不上账。
- **对你有什么影响**：使用 HiCache + 混合缓存（如 Mamba 状态派生自 KV）时，长上下文或分块传输场景下不会再出现缓存索引缺失导致的错误命中。

### 2. [Fix] megamoe: Allocate the symmetric buffers before KV cache sizing (#47577)
- **代码层面**：`mega_moe.py` 新增 `allocate_mega_moe_buffers(model)`，在 KV cache 计算显存预算之前就把 MegaMoE 的对称缓冲区分配好并打日志；同时删掉了文档里“可能额外分配导致 OOM，需要调低 mem-fraction”的警告。
- **新手概念课堂**：就像装修前先把大件家具搬进屋，再量剩余空间买沙发；否则沙发买回来发现放不下。
- **对你有什么影响**：使用 MXFP4 MegaMoE + 投机解码时，KV cache 预算会把 MegaMoE 缓冲区算进去，CUDA graph capture 不再因为“事后多分配”而 OOM。

### 3. [Refactor] scheduler: Track running requests in an owner list (#43899)
- **代码层面**：新增 `scheduler_components/running_reqs.py` 的 `RunningReqs` 类，在 `scheduler.py` 初始化 `self.running_reqs`，在各 batch 合并/启动处调用 `join()`，并新增 `SGLANG_ENABLE_RUNNING_REQS_CHECK` 环境变量（CI 默认开启）做不变量校验；`environ.py` 把 `_default_tree_cache_sanity_check` 重命名为通用的 `_default_on_in_ci`。
- **新手概念课堂**：以前“谁在跑”是靠翻遍所有 batch 推断出来的，现在有一本花名册，每次点名都和实际在场的人核对。
- **对你有什么影响**：普通用户感知不到，但调度器 bug（请求卡死、重复调度）会更早被 CI 发现，长期稳定性提升。

### 4. [Feat] [AMD] mamba: Store hicache checkpoints without a GPU slot (#44718)
- **代码层面**：`fused_kda_conv_recurrent_verify.py` 和 `fused_sigmoid_gating_recurrent.py` 的 Triton kernel 新增 `ring_indices` 参数和 `SEPARATE_RING_INDEX` 编译期开关，允许 KDA 的 ring buffer 索引与 mamba slot 索引分离；这样 checkpoint 可以存在主机侧而不占 GPU slot。
- **新手概念课堂**：就像酒店房间（GPU slot）不够时，把行李寄存到前台（host），但取行李的凭证号（ring index）要和房间号分开管理。
- **对你有什么影响**：AMD 上跑 Mamba 类模型 + HiCache 时，可以缓存更多 checkpoint，不被 GPU slot 数量限制。

### 5. [Perf] disaggregation: Stop device admission in a decode transfer pass at the first refusal (#46898)
- **代码层面**：`decode.py` 的 `pop_transferred_with_outcomes` 新增 `device_admission_refused` 标志，一旦某个请求 `allocate_host_staged` 失败，后续请求本轮直接跳过，等下一轮。
- **新手概念课堂**：排队进停车场，第一辆车发现没车位，后面的车就不用再挨个试了，等下一轮统一放行。
- **对你有什么影响**：PD 分离解码时，设备准入失败后的无效尝试减少，调度轮次更快。

### 6. [Perf] DFLASH: route draft bf16 GEMMs through tuned dispatch on GB300 (#45876)
- **代码层面**：`cutedsl_bf16_gemm.py` 和 `unquant.py` 的调优表新增多组 GB300 上 DFLASH draft 的 GEMM shape（如 `(14336, 8192)`、`(50512, 8192)`）；`dflash.py` 的 `_project_candidate_logits` 改用 `bf16_gemm_dispatch` 而非直接 `torch.matmul`。
- **新手概念课堂**：不同尺寸的矩阵乘法就像不同体型的货物，需要不同型号的卡车（tile/stage 配置）才跑得快；调优表就是“货物-卡车”对照表。
- **对你有什么影响**：在 GB300 上跑 DFLASH 投机解码 draft 模型时，候选头投影和 gate_up GEMM 更快。

### 7. [Perf] [NVIDIA] dsv41: Balance the SM90 paged decode indexer's tiles (#47387)
- **代码层面**：`fp4_indexer.py` 新增 `_fp4_index_logits_paged_balanced` 路径（由 `SGLANG_OPT_DSV41_FAST_DECODE_INDEXER` 控制），把 paged decode indexer 的 K tile 按 GPU 负载均衡划分，避免单个长上下文请求独占一个 SM 拖慢整个 step。
- **新手概念课堂**：就像打扫房间，不是让一个人打扫整个大房间，而是把大房间切成小块分给所有人，大家同时完工。
- **对你有什么影响**：DeepSeek-V4.1 类模型在 SM90 上做长上下文 decode 时，step time 不再被单个长请求主导。

### 8. [Fix] model_loader: Neutral UE8M0 block scales for dummy weights (#47337)
- **代码层面**：`weight_utils.py` 的 `initialize_dummy_weights` 增加分支：`weight_scale_inv` 且 dtype 为 `uint8` 时填 127（即 2^0），而不是走浮点随机初始化；新增对应单测。
- **新手概念课堂**：UE8M0 的 scale 是“指数”，127 对应指数 0，也就是乘以 1；填 0 相当于乘以 2^-127，权重全变 0。
- **对你有什么影响**：用 dummy weight 做 profiling/显存估算时，MXFP8 模型的数值行为更接近真实权重，测量更准。

### 9. [Perf] unified-memory: Bind the SWA tail's new pages without a host sync (#47542)
- **代码层面**：`chain.py` 的 `alloc_extend_swa_tail` 里，`tail_pages` 从“用 `torch.unique` + `torch.isin` 在设备上匹配”改成“直接取 `new_pages[-num_swa_pages:]`”，避免设备端操作阻塞调度器。
- **新手概念课堂**：以前要等 GPU 算完“哪些页是新的”才能继续，现在根据“第 j 个新页对应第 first_new_page+j 个序列页”的规则直接切片，不用等 GPU。
- **对你有什么影响**：unified-memory 场景下 SWA（滑动窗口注意力）的 extend 分配不再有 host sync，调度延迟降低。

### 10. [Fix] hicache: mlock pinned host buffers so page aging never scans them (#47543)
- **代码层面**：`pool_host/common.py` 新增 `_mlock_host_buffer` / `_munlock_host_buffer` / `_memlock_state`，在 CUDA host register 成功后调用 `mlock`，unregister 完所有 range 后 `munlock`；日志打印 `RLIMIT_MEMLOCK`、`CAP_IPC_LOCK`、`VmLck`。
- **新手概念课堂**：pinned 内存只是“不被换出”，但仍在可回收 LRU 上，页老化会清 accessed 标志；`mlock` 把它移到“不可回收”区，页老化根本不看它。
- **对你有什么影响**：HiCache 从主机内存拷贝到 GPU 时（尤其 ATS 场景）不再因缺页中断变慢；日志能帮你诊断 memlock 权限不足。

### 11. [Fix] [AMD] quark: Fix the EP>1 crash in the Quark MXFP4 W4A16 MoE path (#47408)
- **代码层面**：`environ.py` 新增 `SGLANG_ENABLE_QUARK_MXFP4_MOE_W4A16`；`quark_w4a4_mxfp4_moe.py` 把回退条件从“所有非 gfx95/gfx1250 的 HIP”收窄为“gfx1151”，并新增对 `gemm1_alpha/beta/clamp_limit/swiglu_limit` 的 `NotImplementedError` 防御；修复 EP>1 相关逻辑。
- **新手概念课堂**：W4A16 是“权重 4 位、激活 16 位”，在 triton_kernels 里反量化；但如果是带 clamp/alpha 的 SwiGLU 变体，这个简化实现会算错，所以要显式拒绝。
- **对你有什么影响**：gfx1151（Strix Halo）上跑 Quark MXFP4 MoE 不再崩溃；其他 AMD GPU 不再被误判进 W4A16 路径。

### 12. [Feat] [NPU] GLM-4.1V: Add ModelSlim W8A8S support (#44442)
- **代码层面**：`linear_method_npu.py` 对 `weight_scale` 做 `hasattr` 保护（W8A8S 导出可能没有该字段）；`vision.py` 的 `VisionAttentionMetadata` 新增 `actual_seq_lengths_cpu`，避免每个 block 都把 `cu_seqlens` 拷回 CPU；`modelslim.py` 增加量化前缀候选。
- **新手概念课堂**：W8A8S 是“权重 8 位、激活 8 位、带 scale”；视觉注意力每个 block 都要知道序列边界，以前每次都从 NPU 拷回 CPU，现在一次算好复用。
- **对你有什么影响**：NPU 上部署 GLM-4.1V 多模态模型时，W8A8S 量化可用，且视觉编码器每 block 少一次 NPU→CPU 拷贝。

### 13. [Fix] [NPU] DSA: Correct DSA Models Sparse offload index-cache budgeting (#45302)
- **代码层面**：`sparsity_driven_kv_offload/config.py` 的 `get_sparsity_driven_kv_offload_cell_size` 新增 `start_layer` / `is_draft_worker` 参数，用 `dsa_layer_skips_topk` 统计“真正有 indexer 的层数”，而不是直接用总层数；`pool_configurator.py` 传入这两个参数；新增单测。
- **新手概念课堂**：DSA 模型里有些层跳过 top-k（没有 indexer），如果按总层数算缓存，会多分配；就像宿舍里有人不住，还按满员订餐。
- **对你有什么影响**：NPU 上 DSA 模型 + 稀疏 KV offload 的显存预算更准，不会因高估而浪费或 OOM。

### 14. [Doc] cookbook: Add Qwen3.8-Flash-Next B300 NVFP4 AgentX recipes (#47560)
- **代码层面**：新增 `qwen3.8-flash-next-b300.jsx`（431 行配置数据），在 mdx 里合并进 `agentx.cells`；文档补充 B300 HiCache 需要 298 GB 主机内存、`--mamba-track-interval 262144` 等调参说明。
- **新手概念课堂**：这是“菜谱”，告诉你 B300 上跑 Qwen3.8-Flash-Next NVFP4 用哪些参数、什么并发度、怎么配 HiCache。
- **对你有什么影响**：想在 B300 上复现 AgentX 基准测试，可以直接抄配置，注意主机内存和 shared memory 要预留够。

### 15. [Fix] diffusion: Check cancellation at stage and sampling boundaries (#47240)
- **代码层面**：`gpu_worker.py` 在 `_execute_forward_common` 里、保存或交给后台线程之前调用 `raise_if_cancelled`；`progress_reporter.py` 移除 `ReportingTqdm` 里“进度回调顺便检查取消”的逻辑，改为独立的取消检查点；文档更新取消语义。
- **新手概念课堂**：以前取消检查搭在“进度条更新”的便车上，进度条更新频率一变，取消行为就变；现在取消检查是独立站点，进度条只负责显示。
- **对你有什么影响**：diffusion 服务里 DELETE 一个视频任务，取消更及时、行为更可预测；自定义采样循环需要自己在 step 之间调 `check_cancelled`。

### 16. [Fix] Minimax-M3: Deterministic tie-breaking in the CUDA radix decode top-k (#45174)
- **代码层面**：`minimax_decode_topk.cuh` 删除 `counter_final`，新增 `block_rank` lambda（用 `__ballot_sync` + `__popc` 算“比我 id 小且 flag 置位的线程数”）；最后一轮 tie 时按 block id 排序，低 id 优先；小路径（`num_blocks <= kSmallThreshold`）行为不变。
- **新手概念课堂**：分数相同的候选，按“座位号小的先选”规则排队，保证每次选出来的人一样。
- **对你有什么影响**：Minimax-M3 的 top-k 输出可复现，评测分数不再因 GPU 调度抖动。

### 17. [Perf] diffusion: Hold a Qwen-Image-2.1 Q/K norm + RoPE row with a half-warp (#47465)
- **代码层面**：`qknorm_complex_rope.cuh` 把每行 128 维从“32 lane 各 4 元素（8 字节访问）”改成“16 lane 各 8 元素（16 字节访问）”，线程数 256→128，`kVecAlignBytes` 8→16；归约树从 32 lane 的 16/8/4/2/1 改为 16 lane 的 8/4/2/1 + 线程内一次加法，保持与 aten 的 fp32 求和顺序一致。
- **新手概念课堂**：以前 32 个人搬一行货，每人搬 4 件；现在 16 个人搬，每人搬 8 件，单次搬运量翻倍（16 字节向量化），总人数减半。
- **对你有什么影响**：Qwen-Image-2.1 的 Q/K norm + RoPE 融合 kernel 更快，且数值与 PyTorch eager 逐位一致。

### 18. [Perf] diffusion: Replay Qwen-Image-2.1's text-only encoder from CUDA graphs (#47467)
- **代码层面**：`qwen_image21.py` 新增 `TextEncoderGraphs` 类，按 prompt 长度缓存 CUDA graph（最多 8 个），第二次遇到同一长度时 capture，之后 replay；权重地址变化（offload/reload）时清空所有 graph；不满足条件（batch>1、有 pixel_values、TP>1、正在 capture）时回退 eager。
- **新手概念课堂**：文本编码器 eager 前向要发 ~2600 个小 kernel，CPU 发命令比 GPU 算还慢；CUDA graph 把整串命令录下来一次 replay，CPU 开销几乎归零。
- **对你有什么影响**：Qwen-Image-2.1 纯文本 prompt 的编码阶段显著加速，且输出与 eager 逐位一致（因为 replay 的是同样的 kernel）。

### 19. [Perf] [NPU] mimo: fold attention_value_scale into o_proj weights (#40890)
- **代码层面**：`mimo_v2.py` 新增 `v_scale_folded` 标志和 `_fold_attention_value_scale_into_o_proj()`，在 `load_weights` 末尾把 `attention_value_scale` 乘进浮点 o_proj 权重，之后 forward 不再做 `v = v * v_scale`；`deepep.py` 在 NPU 非 zbal 路径跳过 `topk_ids.to(torch.int64)` 的往返转换。
- **新手概念课堂**：注意力对 V 是线性的，`softmax(qk) @ (s*V) == s * (softmax(qk) @ V)`，所以把 s 提前乘进 o_proj 权重，数学上完全等价，但运行时少一次逐元素乘法 kernel。
- **对你有什么影响**：NPU 上 mimo 模型 decode 每层少一个 Muls kernel；量化版 o_proj 不走折叠（保持运行时乘法），避免破坏量化 scale。

### 20. [Doc] [NPU] docs: Rename NPU hardware to A2/A3 Series products (#47380)
- **代码层面**：文档里 `Ascend A3 Series` → `A3 Series`、`Ascend 950PR/DT Series` → `950PR/DT Series`、`Ascend 910B/910C` → `A2/A3 series`；顺手修了 `realses` → `releases` 的拼写错误。
- **新手概念课堂**：统一产品命名，避免“Ascend A3”和“A3”混用导致搜索/引用混乱。
- **对你有什么影响**：查 NPU 安装文档时命名更一致，按 A2/A3/950PR 搜索更容易找到对应版本。

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)