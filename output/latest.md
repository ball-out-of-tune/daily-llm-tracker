# 🚀 vLLM & SGLang 每日更新 — 2026-11-27

> 自动生成于 2026-11-27 10:05 UTC | AI 解读: ✅ 含代码解读

## vllm
## 💡 今日关键词（放在最前面！）

- **关键词：EPLB 异步策略进程化（Async EPLB Subprocess）**
  - 一句话通俗解释：把"专家负载均衡"这种重计算挪到一个常驻的独立 CPU 子进程里跑，不再占用推理主进程的 GIL，让 GPU 推理和 CPU 规划真正并行。
  - 为什么社区在关注：MoE 大模型（Kimi K3、DeepSeek V4.1、Qwen4Exp）越来越多，EPLB 是刚需；但之前异步 EPLB 用线程实现，被 Python GIL 拖累。改成子进程是"真正解耦"的信号，说明 vLLM 在往"控制面/数据面分离"演进。
  - 对新手意味着什么：如果你在学分布式推理，这是一个绝佳的"进程间通信 + 序列化 + 超时/回收"实战案例；也提醒你：自定义 policy 必须可 pickle，否则默认配置会挂。

- **关键词：DeepSeek V4.1 特殊 token 编码（Special-token spelling → plain text）**
  - 一句话通俗解释：用户输入里如果写了 `<|begin_of_sentence|>` 这种"看起来像特殊 token"的字符串，前端要把它当普通文本编码，而不是当成真正的特殊 token。
  - 为什么社区在关注：这是典型的"prompt injection / 模板注入"安全问题——用户不该能伪造模板控制 token。Rust 前端专门引入 tokenizer 做精确编码，说明前端正在从"字符串拼接"走向"token 级正确性"。
  - 对新手意味着什么：做 LLM 应用时，永远不要用字符串拼接构造 prompt；理解 tokenizer 的 special token 边界是基本功。

- **关键词：多模态 token 计数对齐（MM token count ↔ processor）**
  - 一句话通俗解释：vLLM 预估"一张图会占多少 token"必须和真实 processor 算出来的一致，否则调度会算错显存、截断出错。
  - 为什么社区在关注：OCR 类模型（DeepSeek-OCR、Unlimited-OCR、MiniCPM-V）有复杂的裁剪/瓦片逻辑，token 数是动态的。这次给 encoder budget 加了覆盖所有瓦片布局的测试，是"预算系统"走向严谨的标志。
  - 对新手意味着什么：做多模态服务时，"token 预算"是最容易踩坑的地方；学会用 `get_mm_max_toks_per_item` 这类工具验证你的 processor。

- **关键词：ROCm / XPU 异构 CI 与镜像缓存**
  - 一句话通俗解释：AMD（ROCm）和 Intel（XPU）的 CI 开始做基础镜像复用、按 job 状态精确触发缓存机器人。
  - 为什么社区在关注：连续多个 commit 都是 ROCm/XPU 的 CI 修复，说明 vLLM 的异构硬件支持已从"能跑"进入"工程化打磨"阶段。
  - 对新手意味着什么：如果你有 AMD/Intel GPU，现在参与 vLLM 贡献的门槛在降低；CI 基础设施代码也是很好的学习材料。

- **关键词：Kimi K3 / DSA Indexer 精度控制（indexer_logits_dtype）**
  - 一句话通俗解释：稀疏注意力（DSA）的 indexer 计算 logits 时用 fp32 还是 bf16，现在可以由模型配置精确控制，并被真正传到 kernel。
  - 为什么社区在关注：长上下文 + 稀疏注意力是 2025-2026 的主战场，logits 精度直接影响 top-k 选择质量。这是"配置项真的生效了"类 bugfix，价值很高。
  - 对新手意味着什么：学习"配置 → 模型 hook → kernel 参数"这条链路怎么打通，是理解 vLLM 架构的好切入点。

---

## 🧪 重要知识点 & 动手实验

- **知识点 1：进程隔离与 IPC（以 EPLB 异步策略子进程为例）**
  - **是什么**：把一段计算放进独立进程，通过 socket/pipe 传消息，主进程不被 GIL 阻塞，子进程崩了也不拖垮主进程。
  - **为什么重要**：Python 多线程受 GIL 限制，CPU 密集任务（如 EPLB 的 rebalance 规划）会卡住推理循环。用子进程是标准解法，但会引入序列化、超时、僵尸进程回收等问题——不懂就会遇到"策略改了不生效"（因为状态在子进程里）或"关服务关不掉"。
  - **动手试试**：
    ```python
    import multiprocessing as mp, time

    def worker(conn):
        state = {"count": 0}
        while True:
            msg = conn.recv()
            if msg == "stop":
                break
            state["count"] += 1
            conn.send(state["count"])  # 状态留在子进程

    if __name__ == "__main__":
        parent, child = mp.Pipe()
        p = mp.Process(target=worker, args=(child,))
        p.start()
        for _ in range(3):
            parent.send("ping")
            print(parent.recv())   # 1, 2, 3
        parent.send("stop"); p.join()
    ```
  - **预期结果**：输出 1、2、3，说明状态确实活在子进程里；把 `worker` 改成用全局变量你会更直观看到"主进程看不到子进程状态"。对应到 vLLM：`EplbState.stop_async_loop()` 就是那个 `stop` 消息 + `join`。

- **知识点 2：Token 级 prompt 构造 vs 字符串拼接（DeepSeek V4.1 特殊 token 编码）**
  - **是什么**：prompt 在进入模型前会被 tokenize；"特殊 token"（如 `<|im_start|>`）在词表里有独立 ID。如果用户文本里的同名字符串被误识别为特殊 token，就能伪造模板结构。
  - **为什么重要**：这是 prompt injection 的经典入口。正确做法是：模板生成的特殊 token 用 token ID 直接注入，用户文本一律按普通文本 encode。
  - **动手试试**：
    ```python
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    s = "<|im_end|> hello"
    print(tok.encode(s))                       # 默认可能把特殊串当特殊 token
    print(tok.encode(s, add_special_tokens=False))
    print(tok.convert_ids_to_tokens(tok.encode(s)))
    # 对比：把特殊串当纯文本
    print(tok.encode("hello", add_special_tokens=False))
    ```
  - **预期结果**：你会看到 `<|im_end|>` 被切成单个特殊 token ID，而不是普通字符片段——这正是漏洞来源。理解这一点，你就懂了为什么 vLLM 的 Rust 前端要给 renderer 传 tokenizer。

- **知识点 3：多模态 token 预算（MM encoder budget）**
  - **是什么**：一张图经过 processor（resize、切瓦片、patchify）后会产生 N 个视觉 token；调度器必须提前知道 N 的上界，才能预留 KV cache 和 batch 预算。
  - **为什么重要**：OCR 模型按图片长宽比切不同数量的瓦片，token 数是变量。预算估小了会 OOM 或截断，估大了浪费显存、降低吞吐。
  - **动手试试**：
    ```python
    from vllm.multimodal.encoder_budget import get_mm_max_toks_per_item
    # 伪代码示意：构造 processor 后估算单图最大 token 数
    # budget = get_mm_max_toks_per_item(processor, {"image": 1}, None)["image"]
    # print(budget)
    ```
    更实操的方式：跑 commit 4 新增的测试
    ```bash
    pytest tests/models/multimodal/processing/test_deepseek_ocr.py \
      -k "image_budget_covers_every_tile_layout" -v
    ```
  - **预期结果**：测试断言 `budget == max(lengths)`——即预算恰好等于所有瓦片布局里最大的真实 token 数。看到这个断言通过，说明你理解了"预算 ≥ 真实上界"这个不变量。

- **知识点 4：配置项如何真正到达 kernel（Kimi K3 indexer_logits_dtype）**
  - **是什么**：一个配置（fp32/bf16）要经过 config → 模型 hook → 算子调用参数 → Triton/CUDA kernel 四层才真正生效；任何一层漏传，配置就是"死配置"。
  - **为什么重要**：这类 bug 极其隐蔽——不报错、不崩溃，只是精度悄悄不对。commit 16 用 monkeypatch 拦截 `fp8_fp4_paged_mqa_logits` 记录 `logits_dtype` 参数，是非常漂亮的测试手法。
  - **动手试试**：
    ```python
    import torch
    # 模拟"记录参数是否传到底层"
    seen = []
    def fake_kernel(x, logits_dtype=torch.float32):
        seen.append(logits_dtype)
        return x
    def wrapper(x, logits_dtype):
        return fake_kernel(x, logits_dtype=logits_dtype)
    wrapper(torch.zeros(1), torch.bfloat16)
    assert seen == [torch.bfloat16]
    print("配置成功到达 kernel 层")
    ```
  - **预期结果**：打印成功。把 `wrapper` 里的 `logits_dtype=logits_dtype` 删掉，断言就会失败——这就是 commit 16 想防的回归。

- **知识点 5：CI 镜像缓存与状态触发（ROCm/XPU）**
  - **是什么**：CI 里"构建一次镜像，多次复用"；用 GitHub status 事件（`ci/step: xxx`）精确唤醒缓存机器人，而不是笼统地监听整个 pipeline。
  - **为什么重要**：GPU 镜像动辄几十 GB，每次重建浪费大量时间和带宽。精确触发能减少无意义的重建，也让缓存逻辑可测试。
  - **动手试试**：
    ```bash
    # 本地验证 shell 里的"镜像是否存在"探测逻辑（commit 10 的 image_digest 思路）
    docker buildx imagetools inspect nginx:latest 2>&1 | awk '$1=="Digest:"{print $2; exit}'
    docker buildx imagetools inspect nginx:this-tag-does-not-exist 2>&1 | head -3
    ```
  - **预期结果**：第一条输出 `sha256:...`；第二条输出 manifest unknown 类错误。理解这个区分，你就懂了为什么 commit 10 要写"重试 registry 错误，但不重试 missing image"。

---

## 🔥 重要更新

1. **#68606 EPLB 异步策略改为常驻子进程**：默认 `use_async=true` 的语义变了，自定义 policy 必须可序列化，插件作者需要适配——影响面最大的架构级变更。
2. **#69064 投机解码支持 VLM hidden states 提取**：之前 chunked prefill 不兼容的限制被解除，并新增 Nemotron 3.5 Parse 的提取示例，多模态 + 投机解码组合能力增强。
3. **#64435 DeepSeek V4.1 支持 PCP 图像输入**：多模态 + prefill context parallel 打通，长图/多图场景的分块 prefill 有了真实验证。
4. **#70187 Qwen4Exp Blackwell FP8 QSA 性能优化**：Split-K、PDL、`cvt.rn.bf16x2.e4m3x2` 指令，属于底层 kernel 级性能提升，对 Blackwell 用户吞吐有直接收益。

---

## 📋 逐条解读

### 1. [Feature][EPLB] Run async policy computation in a persistent subprocess (#68606)
- **代码层面**：新增 `vllm/distributed/eplb/policy_process.py`，实现 `EplbPolicyProcess`（常驻 CPU 子进程 + 60s IPC 超时 + 错误上报）。`EplbState` 的异步路径从"后台线程调 policy"改为"把 policy 序列化后发给子进程"。文档同步更新了 `plugin_system.md` 和 `expert_parallel_deployment.md`，明确"policy 必须可序列化、进程本地资源要懒初始化或实现 `__getstate__`/`__setstate__`"，并新增 `EplbState.stop_async_loop()` 用于关停回收。
- **新手概念课堂**：把 EPLB 想象成"仓库管理员定期重新安排货架"。以前管理员和收银员（推理）在同一个柜台，管理员算账时收银员就得等（GIL）。现在给管理员单独开一个办公室（子进程），两人通过传纸条（IPC）沟通。但纸条只能写文字，所以管理员的所有"记忆"（policy 状态）必须先能写下来（序列化）。
- **对你有什么影响**：如果你写了自定义 EPLB policy 且持有锁、进程组、设备上下文，升级后可能直接报序列化错误；需要按文档改造，或在 shutdown 时调用 `stop_async_loop()`。

### 2. [Rust Frontend][DeepSeek V4.1] Encode special-token spellings in client text as plain text (#70271)
- **代码层面**：`rust/src/chat/src/renderer/deepseek.rs` 新增 `writer.rs` 模块（`DsPromptWriter`、`SpecialTokenizer`），renderer 现在接收 tokenizer，返回 `Prompt`（可能是 token IDs 而非字符串）。`hf.rs` 里构造 `DeepSeekV41ChatRenderer` 时传入 `tokenizer.clone()`。删除了旧的 `reject_image_placeholder_text` 拒绝逻辑，改为"把用户文本里的特殊 token 拼写当普通文本编码"。
- **新手概念课堂**：想象你在填一张官方表格，"签名"栏必须手写。如果有人在"备注"栏写了"签名"两个字，工作人员不能把它当成真的签名。特殊 token 就是"签名栏"，用户文本是"备注栏"，两者必须严格区分。
- **对你有什么影响**：用 DeepSeek V4.1 时，用户输入里包含 `<|...|>` 样式的字符串不再报错，而是安全地当普通文本处理；渲染结果从字符串变成 token 序列，下游若有依赖字符串的代码需注意。

### 3. [Bug][Spec Decode] Enable VLM hidden states extraction (#69064)
- **代码层面**：`tests/v1/kv_connector/extract_hidden_states_integration/test_extraction.py` 新增 `_VLMExtractionCase` 和 `_run_vlm_extraction`，支持带图像的多模态提取，并覆盖 chunked prefill（`enable_chunked_prefill=True` + `disable_chunked_mm_input=True`）。文档删除了"chunked prefill 不兼容"的 note，新增 Nemotron 3.5 Parse 的说明。
- **新手概念课堂**：hidden states 提取就像"把模型思考过程中的草稿纸留下来"。以前多模态模型（看图说话）的草稿纸不让留，现在可以了，而且长 prompt 分块处理时也能留。
- **对你有什么影响**：想做 EAGLE 类投机解码、需要 VLM 中间层特征的场景，现在可以直接用 `extract_hidden_states` 方法，且不必关闭 chunked prefill。

### 4. [Bugfix][Multimodal] Match multimodal token counts to the processor (#68623)
- **代码层面**：涉及 `deepseek_ocr2`、`minicpmv`、`unlimited_ocr` 等 processor 的 token 计数逻辑；新增 `test_image_budget_covers_every_tile_layout` 用 `calculate_aspect_ratios(MIN_CROPS, hf.max_crops)` 枚举所有瓦片布局，断言 `get_mm_max_toks_per_item` 的预算等于真实最大长度。GLM5Next 的 encoder profile 也加了 dummy 构建测试。
- **新手概念课堂**：就像快递打包，你得先知道"这个包裹最多占几个格子"才能安排货车。以前估算和实际不一致，货车装不下或空跑；现在用"所有可能形状都试一遍"来校准。
- **对你有什么影响**：OCR 类多模态模型在高分辨率图上不再出现 token 预算不足导致的截断或 OOM，吞吐估算更准。

### 5. [Bugfix][Frontend] Honor reasoning_effort and documents in /tokenize (#69917)
- **代码层面**：`rust/src/server/src/routes/tokenize/types.rs` 的 `TokenizeChatRequest` 新增 `reasoning_effort: Option<ReasoningEffort>` 和 `documents: Option<Vec<Value>>` 字段，并映射到 `ChatOptions`。新增测试 `test_tokenize_matches_chat.py` 并加入 CI 列表，验证 `/tokenize` 与 `/chat/completions` 渲染一致。
- **新手概念课堂**：`/tokenize` 是"预览接口"——只算 token 不生成。以前预览时忽略了 `reasoning_effort` 和 RAG 文档，导致"预览和实际不一致"，就像试衣间镜子和真实灯光色差。
- **对你有什么影响**：用 `/tokenize` 做成本预估或上下文长度检查时，结果现在和实际 chat 请求一致，不会低估 token 数。

### 6. [Bugfix][ROCm][MoRIIO] WRITE mode free prefill blocks (#67218)
- **代码层面**：MoRIIO KV connector 的 scheduler 新增 `_unfinished_write_sends` 集合，worker 新增 `_completed_notification_owners` 映射，用于在 prefill 期间或 transfer id 复用时正确释放 block，避免提前 free 或错配 ack。多个单测 fixture 同步补上这些字段。
- **新手概念课堂**：KV 传输像"图书馆借书"。prefill 还没写完就把书还回书架（free block），别人借走就会读到半本书。新逻辑是"借出登记本"：记录谁还没还、谁已经还了，防止张冠李戴。
- **对你有什么影响**：ROCm + MoRIIO + WRITE 模式下的 PD 分离部署，长 prefill 或高并发复用 transfer id 时不再出现数据损坏或崩溃。

### 7. [ROCm][CI] Fix HY4 sequence parallel test (#70305)
- **代码层面**：`tests/models/hy_v4/test_sequence_parallel.py` 中用 `getattr(layer, "ihc_attn_boundary", None)` 替代直接属性访问，因为 ROCm decoder layer 没有 fused boundaries，直接访问会 AttributeError。
- **新手概念课堂**：CUDA 上有"融合算子"（把两步合成一步），ROCm 上没有，所以属性不存在。测试要写成"有就用，没有就走备选路径"。
- **对你有什么影响**：纯 CI 修复，普通用户无感；但说明 vLLM 在不同硬件上走不同代码路径，写测试要兼容。

### 8. [Bugfix][UMBP] Restore Mamba hybrid prefixes at boundaries inside attention blocks (#67942)
- **代码层面**：UMBP store connector 文档里 `enable_partial_hash_hits` 默认值改为"Mamba hybrid 模型默认 true"；测试 fixture `umbp_hybrid_llm` 支持 standalone server、`prefix_match_unit` 参数化，新增 sub-block 命中恢复的测试。
- **新手概念课堂**：Mamba hybrid 模型的"状态检查点"落在 prompt 尾部更细的哈希边界上，而普通 block 粒度太粗，几乎永远命中不了。所以要允许"块内部分命中"。
- **对你有什么影响**：用 Mamba hybrid 模型 + UMBP KV 池时，前缀缓存命中率显著提升，首 token 延迟下降。

### 9. [Recipe] Kimi-K3.1-Pro: B300 PD preset points 64, 264 and 640 (#70188)
- **代码层面**：`recipes/models/moonshotai/Kimi-K3.1-Pro.yaml` 把 B300 PD 预设从 c96/c660 改为 c64/c264/c640，新增 EPLB 参数、`--prefix-match-unit 704`、FlashInfer autotune、`VLLM_SWA_FINE_GRAINED_HITS` 等环境变量，并更新 InferenceX 引用 revision。
- **新手概念课堂**：Recipe 是"调好的菜谱"——别人验证过的启动参数组合。这次是重新对齐上游 benchmark 的并发点，并加入 EPLB 与细粒度前缀命中。
- **对你有什么影响**：直接抄 recipe 部署 Kimi K3.1-Pro 的用户，能获得与 InferenceX 对齐的性能数字；c64 预设还开启了 prefill EPLB。

### 10. [XPU][CI] Build and reuse XPU base image across CI runs (#67428)
- **代码层面**：`image_build_xpu.sh` 重写：新增 `image_digest()` 函数（3 次重试 registry 错误，但不重试 missing image）、`publish_base` 判断（仅 main 分支非 PR 时发布基础镜像）、基于 base key 的镜像复用逻辑。`ci_config_intel.yaml` 增加对 `docker-bake-xpu.hcl` 等的触发。
- **新手概念课堂**：CI 镜像像"预装好环境的虚拟机模板"。每次都从零装太慢，所以先做一个"基础层"（base image），业务层在其上叠加，基础层不变就复用。
- **对你有什么影响**：Intel XPU 的 PR CI 会更快；普通用户无感，但可以学习"镜像分层 + 缓存 key"的工程实践。

### 11. [DSv4.1][PCP] Support image input (#64435)
- **代码层面**：新增 `tests/distributed/test_deepseek_v41_multimodal_pcp.py`，用 4 GPU（TP2 + PCP2）跑真实权重图像输入，验证跨 PCP chunk 的 draft 和混合 batch 缓存。新增 `vllm/v1/worker/gpu/mm/`、`model_states/default.py`、`vllm/models/deepseek_v41/` 等目录纳入 CI 触发。
- **新手概念课堂**：PCP（Prefill Context Parallel）是把超长 prompt 切成几段并行 prefill。图像 token 也要跟着切，切错了 draft 模型就看不到完整图像上下文。
- **对你有什么影响**：DeepSeek V4.1 多模态 + 长上下文 + PCP 组合可用，适合长文档/多图 RAG 场景。

### 12. [Bugfix][CPU] Skip unused embedding packs and refresh tied LM heads (#67782)
- **代码层面**：`vocab_parallel_embedding.py` 中 CPU 路径不再在 `process_weights_after_loading` 里立即打包 GEMM，而是设 `cpu_linear = None`，延迟到首次 `apply` 时 `dispatch_cpu_unquantized_gemm`；新增 `_finalize_cpu_embedding`，在所有权重加载后（尤其 tied LM head 可能没有独立 checkpoint）再 finalize。
- **新手概念课堂**：tied embedding 是"输入词表和输出词表共用一份权重"。如果加载时先打包输入侧，输出侧还没加载完就打包，会用到旧权重。正确做法是"全部加载完再统一打包"。
- **对你有什么影响**：CPU 后端 + tied LM head 模型（很多小模型）在权重重载（如 LoRA 切换、热更新）后输出正确，不再出现"词表错乱"。

### 13. [Recipe] Kimi-K3.1-Pro: GB200 AgentX presets (#70285)
- **代码层面**：YAML 新增 `gb200: verified` 硬件、`strategy_min_gpus.gb200: 16`、三个 GB200 AgentX preset（TP16+DCP16、TP8+DCP8 等），配置 DSpark draft、FP8 KV、FP32 KDA state、`--use-replayssm` 等。
- **新手概念课堂**：GB200 是 NVIDIA 新一代机架级 GPU，显存大但单 tray 装不下 2.8T 模型，所以最少 4 tray。Recipe 把这些约束写进 `strategy_min_gpus`。
- **对你有什么影响**：有 GB200 的用户可以直接用验证过的 AgentX 配置，省去调参；也说明 vLLM 在追最新硬件。

### 14. [Model][Hardware] Inkling: relative attention on SM8x via Triton kernel (#69960)
- **代码层面**：`vllm/models/inkling/nvidia/attention.py` 在 SM8x 上跳过 FA4 kernel 的 JIT warmup（因为根本不会跑）；`fa4_rel_attention.py` 新增 Triton 实现的 relative attention 路径供 SM8x 使用；测试移除 Hopper+ 的 skip 条件。
- **新手概念课堂**：FA4（FlashAttention 4）的 score-mod 需要 Hopper 及以上。SM8x（Ampere/Ada）没有，所以要写一个 Triton 版本作为 fallback，同时别在 SM8x 上白白 warmup 一个永远不会用的 kernel。
- **对你有什么影响**：在 A100/4090 等 SM8x 卡上跑 Inkling 模型不再报错，可以用 Triton 相对注意力。

### 15. [Bugfix][Pooling] Apply --default-chat-template-kwargs to pooling chat requests (#68664)
- **代码层面**：pooling/embedding/scoring 的 chat 请求路径现在会合并服务端 `--default-chat-template-kwargs`，请求级 `chat_template_kwargs` 仍优先。新增 `server_with_default_chat_template_kwargs` fixture 和 token 数断言测试。
- **新手概念课堂**：`--default-chat-template-kwargs` 是"服务端默认模板参数"，比如统一加一句 "Represent this conversation for retrieval:"。以前只在生成接口生效，embedding/rerank 接口被漏掉了。
- **对你有什么影响**：用 embedding/rerank 服务时，服务端默认模板参数现在一致生效，检索质量更可控。

### 16. [Kimi K3][Bugfix] Honor indexer_logits_dtype in the DSA indexer (#69838)
- **代码层面**：`KimiK3DSAIndexer.score` 把 `logits_dtype` 真正传给 `fp8_fp4_paged_mqa_logits`；新增测试用 monkeypatch 记录 kernel 收到的 `logits_dtype` 并断言等于配置值；`test_indexer_logits_dtype_config.py` 新增 Kimi K3 的默认值（float32）并给 hf_config 加 `dsa_config={}` 以便 hook 生效。
- **新手概念课堂**：DSA（DeepSeek Sparse Attention）的 indexer 负责"先粗筛哪些 token 值得算注意力"。筛选用 fp32 还是 bf16 会影响 top-k 质量。配置写了 bf16 但 kernel 收到 fp32，就是"配置没生效"。
- **对你有什么影响**：Kimi K3 长上下文场景下，indexer 精度按预期工作，top-k 选择更准，长文本质量更稳定。

### 17. [ROCm][CI] Wake the ROCm image cache bot on its two jobs' statuses (#70086)
- **代码层面**：GitHub workflow 的触发条件从 `buildkite/ci/pr`、`buildkite/ci` 改为精确的 `ci/step: ROCm image cache` 和 `ci/step: Cache a PR's ROCm images for main`；`run_ci_command.py` 用新的 `rocm_promote_result()` 解析 promote 结果。
- **新手概念课堂**：以前机器人监听"整个 pipeline 结束"，现在监听"特定 job 结束"。信号更精确，避免无关 job 触发缓存逻辑。
- **对你有什么影响**：ROCm CI 缓存更可靠；普通用户无感，但可学习 GitHub Actions 的 status 事件过滤。

### 18. [Bugfix][EPD] Preserve per-video processor kwargs during encoder fan-out (#70167)
- **代码层面**：`disagg_epd_proxy.py` 新增 `PER_VIDEO_MM_KWARGS = ("fps", "num_frames")` 和 `_loads_audio_from_video()`，在把请求 fan-out 到多个 encoder 时按视频顺序切片这些列表参数；README 提示音频从视频提取时要把 per-video kwargs 放进 `videos_kwargs`。
- **新手概念课堂**：一个请求有 3 个视频，每个视频 fps 不同（`fps=[1,2,4]`）。fan-out 时每个子请求只带 1 个视频，必须带对应的那一个 fps，否则都用第一个值，缓存 key 对不上。
- **对你有什么影响**：EPD（Encoder-Prefill-Decode 分离）多视频场景下，视频采样参数正确，缓存命中，不再出现"同一视频不同帧率"的错乱。

### 19. [Perf][Qwen4Exp] Split-K, fused tiny-M slots, PDL, and cvt.rn.bf16x2.e4m3x2 for Blackwell FP8 QSA (#70187)
- **代码层面**：`vllm/models/qwen4_exp/nvidia/ops/qsa.py` 的 Split-K kernel 和 merge kernel 增加 `launch_pdl` 参数（`gdc_wait`/`gdc_launch_dependents`），索引计算改用 `int64` 防溢出，新增 fused tiny-M slot 和 Blackwell FP8 转换指令；测试新增 `num_requests=[1,4]` 小 batch 用例。
- **新手概念课堂**：Split-K 是"把一个大的矩阵乘拆成几段并行算再合并"；PDL（Programmatic Dependent Launch）是"让下一个 kernel 提前启动、等数据就绪"，减少 kernel 间空泡；`cvt.rn.bf16x2.e4m3x2` 是一条 Blackwell 指令一次转两个 FP8。
- **对你有什么影响**：Blackwell 上 Qwen4Exp 的 FP8 注意力 decode 更快，尤其小 batch（1、4 请求）延迟降低。

### 20. [Bugfix][MM] Bypass receiver MM cache for pre-rendered features items (#69392)
- **代码层面**：scale-out 的 token-in-token-out 路径中，pre-rendered features 不再走 receiver 的 MM processor cache，避免把已有图像缓存条目挤掉；`serve_tokens` 对非法 SHM handle 改为抛 `ValueError`（之前返回 ErrorResponse）；新增 `test_mm_cache_state_drift.py` 端到端回归。
- **新手概念课堂**：MM processor cache 存"图片 → 处理结果"。如果 pre-rendered features（已经处理好的）还去写这个 cache，就会触发 LRU 淘汰，把真正需要的原始图片缓存挤出去，导致后续请求重复处理。
- **对你有什么影响**：scale-out 多模态服务在高并发下缓存更稳定，不会因为"重复写入已渲染特征"导致缓存抖动、延迟上升。

## sglang
## 💡 今日关键词

- **PD 分离（Prefill/Decode Disaggregation）首 token 流式策略**：把"预填充算完的第一个 token 什么时候发给用户"做成可调参数，在 TTFT 和 TPOT 之间做取舍。社区在关注：说明大模型推理服务正在从"能跑"走向"能精细调优 SLA"，首 token 延迟和每 token 延迟的权衡成了产品级问题。对新手意味着：理解"延迟指标拆解"（TTFT/TPOT/E2E）是进入推理服务优化的第一课。
- **共享内存 / 多进程通信生命周期管理**：CPU 后端 all-reduce 通过 `/dev/shm` 共享内存做进程间通信，今天修了"映射后及时 unlink"和"尺寸校验"的问题。社区在关注：说明异构 CPU 推理、多卡多进程通信的健壮性正被认真打磨。对新手意味着：`shm_open`/`mmap`/`shm_unlink` 这套 POSIX 共享内存 API 是后端工程师的必修课。
- **量化格式的激活函数适配（W4A16 + GELU）**：Marlin 量化 MoE 内核原来只支持 SiLU/SiTU，现在支持 GELU/GELU-tanh，以适配 Gemma 4 这类模型。社区在关注：说明量化不是"一刀切"，不同模型架构需要内核级别的适配。对新手意味着：想深入推理优化，必须懂"量化格式 × 激活函数 × 内核实现"这三者的耦合。
- **多硬件后端齐头并进（Ascend NPU / AMD ROCm / NVIDIA SM107）**：今天有 NPU DFLASH 投机解码、AMD MI355X 调优配方、NVIDIA SM107 融合 all-reduce 阈值提升。社区在关注：SGLang 正在成为"全硬件栈"推理框架，异构计算是明确趋势。对新手意味着：只学 CUDA 不够了，理解"平台抽象层"和"后端分发"是稀缺技能。
- **HiCache / 分层 KV 缓存的一致性**：修了 buffer 模式下"父节点被 split 时子节点写入丢失"的并发 bug。社区在关注：KV 缓存分层（GPU→CPU→磁盘）是长上下文和高并发的关键，但并发正确性极难。对新手意味着：缓存一致性问题（in-flight 状态、分裂、覆盖判定）是分布式系统的经典难点。

## 🧪 重要知识点 & 动手实验

- **知识点 1：POSIX 共享内存的创建、映射与 unlink**
  - **是什么**：多个进程通过 `/dev/shm` 下的命名共享内存段交换数据，`shm_open` 创建/打开、`mmap` 映射到地址空间、`shm_unlink` 删除名字（但已映射的进程仍可用）。
  - **为什么重要**：Commit 2 修的就是"映射完成后不 unlink 导致 `/dev/shm` 残留文件堆积"。如果忘记 unlink，共享内存会一直占着 tmpfs（内存！），机器跑久了就 OOM。另一个坑是：`shm_open` 成功不代表尺寸够，必须 `fstat` 校验 `st_size`，否则 `mmap` 后越界访问直接段错误。
  - **动手试试**：
    ```bash
    # 终端 1：创建并映射一个 1MB 共享内存，写入后 unlink
    cat > /tmp/shm_demo.c <<'EOF'
    #include <stdio.h>
    #include <fcntl.h>
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <unistd.h>
    #include <string.h>
    int main() {
        const char *name = "/demo_shm";
        int fd = shm_open(name, O_CREAT | O_RDWR, 0600);
        ftruncate(fd, 1 << 20);
        struct stat st; fstat(fd, &st);
        printf("size = %ld\n", (long)st.st_size);
        char *p = mmap(NULL, 1 << 20, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        strcpy(p, "hello shm");
        shm_unlink(name);   // 名字删掉，映射仍有效
        printf("read back: %s\n", p);
        munmap(p, 1 << 20); close(fd);
        return 0;
    }
    EOF
    gcc /tmp/shm_demo.c -o /tmp/shm_demo -lrt && /tmp/shm_demo
    ls /dev/shm | grep demo_shm || echo "no leftover file (good)"
    ```
  - **预期结果**：打印 `size = 1048576` 和 `read back: hello shm`，最后 `ls /dev/shm` 找不到 `demo_shm` 残留——说明你理解了"unlink 只删名字不删映射"的语义。

- **知识点 2：TTFT / TPOT / E2E 延迟指标拆解**
  - **是什么**：TTFT = 首 token 到达时间；TPOT = 后续每个 token 的平均间隔，通常算作 `(E2E - TTFT) / (输出token数 - 1)`；E2E = 端到端总时间。
  - **为什么重要**：Commit 1 的核心就是"第一个 token 早发还是晚发"的选择——早发降低 TTFT 但会让"第二个 token 的等待"算进 TPOT；晚发则相反。不懂这个拆解，你调参就是瞎调，用户抱怨"卡"你也不知道是首 token 慢还是后续慢。
  - **动手试试**：
    ```python
    # 模拟两种策略下的指标差异
    def metrics(t_first, t_second, n_out, e2e):
        ttft = t_first
        tpot = (e2e - ttft) / (n_out - 1)
        return ttft, tpot
    # 策略 A：admission，首 token 早发 0.10s，第二个 token 0.15s
    # 策略 B：first_decode，首 token 晚发 0.15s，第二个 token 0.15s
    for name, t1 in [("admission", 0.10), ("first_decode", 0.15)]:
        ttft, tpot = metrics(t1, 0.15, 11, 1.60)
        print(f"{name:14s} TTFT={ttft:.3f}s TPOT={tpot:.3f}s")
    ```
  - **预期结果**：`admission` 的 TTFT 更低（0.100 vs 0.150），但 TPOT 更高（0.150 vs 0.145）——直观看到"总时间不变，只是把等待记到哪个指标上"。

- **知识点 3：Marlin 量化 MoE 的激活函数门控**
  - **是什么**：MoE 的专家 FFN 通常是"门控线性层"：`act(gate(x)) * up(x)`，其中 `act` 可以是 SiLU、GELU、GELU-tanh 等。量化内核（Marlin）必须为每种激活提供对应的融合算子。
  - **为什么重要**：Commit 15 说明原来只支持 SiLU，Gemma 4 用 GELU 就跑不了或跑错。这是一个"看起来只是换个激活函数，实际要改内核分发"的典型例子。新手常以为"激活函数随便换"，但在量化+融合内核场景下，每个组合都要单独实现和验证。
  - **动手试试**：
    ```python
    import torch, torch.nn.functional as F
    x = torch.randn(4, 64)
    gate, up = x.chunk(2, dim=-1)
    silu_out = F.silu(gate) * up
    gelu_out = F.gelu(gate) * up
    gelu_tanh_out = F.gelu(gate, approximate="tanh") * up
    print("silu vs gelu max diff:", (silu_out - gelu_out).abs().max().item())
    print("gelu vs gelu_tanh max diff:", (gelu_out - gelu_tanh_out).abs().max().item())
    ```
  - **预期结果**：两个 diff 都明显非零——说明"激活函数不同，输出就不同"，量化内核必须精确匹配模型训练时用的激活，否则精度崩。

- **知识点 4：Prometheus 直方图的 label 缓存**
  - **是什么**：`Histogram.labels(**kwargs)` 每次调用都会重建 label 元组并做字符串化，高频调用（每个请求每个阶段）会产生可观开销。缓存 `labels()` 返回的 child 对象即可复用。
  - **为什么重要**：Commit 14 就是一个"微小但真实"的性能优化——观测系统本身不能成为瓶颈。这是"可观测性开销"的经典问题：metrics 埋点太贵会拖慢主路径。
  - **动手试试**：
    ```python
    import time
    from prometheus_client import Histogram
    h = Histogram("demo_latency", "demo", ["model", "stage"])
    N = 200_000
    # 不缓存
    t0 = time.perf_counter()
    for _ in range(N):
        h.labels(model="m", stage="prefill").observe(0.01)
    t1 = time.perf_counter()
    # 缓存 child
    child = h.labels(model="m", stage="prefill")
    t2 = time.perf_counter()
    for _ in range(N):
        child.observe(0.01)
    t3 = time.perf_counter()
    print(f"no-cache: {t1-t0:.3f}s  cached: {t3-t2:.3f}s")
    ```
  - **预期结果**：cached 版本明显更快（通常 2-5 倍），说明"缓存 label child"是低成本高收益的优化。

- **知识点 5：CUDA Graph 捕获与侧流事件（NPU 修复的通用教训）**
  - **是什么**：CUDA Graph 把一串 kernel 录制成图后重放，但图捕获期间对"事件等待/记录"有严格约束——未消费的事件会导致 `EndCapture` 报错。Commit 7 的修复是把 `record_stream` 在 NPU 上无条件启用，绕开侧流捕获问题。
  - **为什么重要**：图捕获是推理性能的关键手段，但"捕获期间的副作用操作"是常见踩坑点。新手写自定义算子时，一旦进了 capture 模式，很多"平时没问题"的写法（动态分配、host 同步、事件操作）都会炸。
  - **动手试试**：
    ```python
    import torch
    if torch.cuda.is_available():
        g = torch.cuda.CUDAGraph()
        x = torch.randn(1024, device="cuda")
        # 先在侧流上做 warmup（必须）
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            y = x * 2
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(g):
            y = x * 2
        x.copy_(torch.ones_like(x))
        g.replay()
        print("graph replay result[0]:", y[0].item())  # 应为 2.0
    else:
        print("no CUDA device, skip")
    ```
  - **预期结果**：打印 `2.0`，说明你成功捕获并重放了一个最简单的 CUDA Graph——理解"warmup → capture → replay"三步是掌握图优化的起点。

## 🔥 重要更新

- **Commit 1（PD 分离首 token 流式策略）**：新增 `--disaggregation-decode-first-token-stream` 参数，让部署方在 TTFT 和 TPOT 之间显式取舍。这是 PD 分离从"能用"到"可调 SLA"的重要一步，直接影响线上服务的延迟指标口径。
- **Commit 11（HiCache buffer 模式并发正确性修复）**：修了一个真实的竞态——父节点在存储写入 in-flight 时被 split，新子节点的页会丢失写入。这类 bug 极难复现，修复对长上下文 + 分层缓存场景的可靠性意义重大。
- **Commit 15（Marlin W4A16 MoE 支持 GELU）**：让 Gemma 4 这类用 GELU 的 MoE 模型能在量化路径上正确运行，同时在 ROCm 路径上加了"非对称量化不支持"的显式报错，避免静默错误结果。
- **Commit 6（NVIDIA SM107 融合 all-reduce 阈值提升到 2MB）**：针对新架构把 1shot push 的适用消息大小从 512KB 提到 2MB，减少小消息走 2shot pull 的开销，是典型的"架构感知调优"。

## 📋 逐条解读

### 1. [Feat] disaggregation: Choose when a PD decode server streams the first token
- **代码层面**：新增参数 `disaggregation_decode_first_token_stream`（`fields/disagg.py`），默认 `first_decode`；在 `pd_disaggregation_hook.py` 里做默认值解析；在 `batch_result_processor.py` 里根据该值决定首 token 是"admission 时立刻发"还是"跟第一个 decode 输出一起发"。
- **新手概念课堂**：PD 分离就像餐厅分工——厨房（prefill）先把菜做好，服务员（decode）负责一道道端上桌。"第一个 token 什么时候端"有两种策略：菜一到就端（TTFT 低，但客人等第二道菜的时间会被算进"上菜间隔"），或者等第二道菜一起端（TTFT 高一点，但上菜间隔更好看）。总时间一样，只是记账方式不同。
- **对你有什么影响**：普通用户感知是"首字延迟"和"打字速度"的观感差异；运维/调优人员可以用这个开关让监控指标更符合业务 SLA 的口径。

### 2. [Fix] CPU: Unlink shared memory after mapping
- **代码层面**：`shm.cpp` 中 `shared_open` 增加 `fstat` 尺寸校验（不够就返回 `ENOENT`）；`allreduce_workspace` 增加 `mapped` 标志；`shm_initialize` 在所有 rank 映射完成后用原子操作同步，然后 `shm_unlink` 删除共享内存名字。
- **新手概念课堂**：共享内存像一块公共白板，`shm_open` 是"申请一块白板并起名"，`mmap` 是"把白板挂到墙上用"，`shm_unlink` 是"把名字从登记簿划掉"——已挂墙上的白板还能继续用，但新进程不能再按名字找到它。不划名字，白板永远占着房间（内存）。
- **对你有什么影响**：长时间运行的多进程 CPU 推理服务不再因 `/dev/shm` 残留文件而逐渐耗尽内存；尺寸不匹配时会安全失败而不是段错误。

### 3. [Feat] metrics: Export total_kv_bytes_usage
- **代码层面**：`pool_stats_observer.py` 的 `PoolStats` 新增 `total_kv_bytes_usage` 字段，在 `_get_window_token_info` 中计算（仅当"一个 full band + 恰好一个 swa band + 无 state band"时有效）；`storage.py` 新增 `kv_bytes_usage()` 计算已用页字节占 arena 总字节的比例；`static_kv_storage.py` 返回 `None`。
- **新手概念课堂**：KV 缓存像一个停车场的不同区域（full 区、滑动窗口区），以前只报"停了多少辆车"（token 数），现在报"占了多少面积"（字节占比）。因为不同区域的"车位大小"不同，token 数不能直接反映内存压力，字节占比才是真实的内存水位。
- **对你有什么影响**：监控面板上多了一个更直观的 KV 内存使用率指标，容量规划更准确。

### 4. [Test] hicache: Retain checkpoints for branching KL tests
- **代码层面**：测试文件里给启动参数加了 `--hybrid-cache-checkpoints-level 3`。
- **新手概念课堂**：这就像测试"分支剧情"时要求游戏保留存档点——KL 测试要在不同分支上对比，如果没有 checkpoint 保留，分支一改前面的状态就没了，测试无法复现。
- **对你有什么影响**：普通用户无感知，属于测试基础设施维护。

### 5. [Doc] [NPU] cookbook: add Kimi K3.1 Pro A3 Series deployment
- **代码层面**：纯文档，新增 Ascend A3 系列的 Docker 镜像、权重链接、四节点 64 die 部署说明，以及"该配方不支持 mamba ratio、只支持 Parsers 卡片"等限制说明。
- **新手概念课堂**：这是"菜谱"更新——告诉你在昇腾 A3 这套"厨房设备"上怎么做 Kimi K3.1 Pro 这道"大菜"，包括用什么"锅"（镜像）、"食材"在哪（权重）、几个人配合（四节点）。
- **对你有什么影响**：如果你有昇腾 A3 集群，现在有官方配方可抄；没有的话也能了解国产硬件生态的进展。

### 6. [Perf] [NVIDIA] kimi-k3: Push the fused all-reduce up to 2 MB on SM107
- **代码层面**：`k3_ar_fusion.py` 新增 `_PUSH_MAX_BYTES_SM107 = 2MB` 和 `_push_max_bytes()`，在 SM107 架构上把 1shot push all-reduce 的适用上限从 512KB 提到 2MB；两处调用点改用该函数。
- **新手概念课堂**：all-reduce 有两种打法——"push"（每个 rank 主动把数据推给别人）和"pull"（从别人那里拉）。小消息 push 更快，大消息 pull 更快，分界线取决于硬件。新硬件 SM107 上分界线右移了，所以调大 push 的适用范围。
- **对你有什么影响**：在 SM107 硬件上跑 Kimi-K3 类模型的 all-reduce 会更快，尤其是中等消息大小（512KB~2MB）的场景。

### 7. [Fix] [NPU] kimi-k3: Fix graph capture with shared-expert overlap
- **代码层面**：`kimi_k3.py` 中 `wait_and_finalize_shared_experts` 把条件从 `fine_grained_overlap or (_is_npu and get_is_capture_mode())` 简化为 `if _is_npu:`，统一用 `record_stream` 而不是 `shared_alt_output` 路径，并留 TODO 说明等 PTA 修复侧流图捕获。
- **新手概念课堂**：图捕获像"录像"，录像期间不能有"临时起意"的动作（未消费的事件），否则录像结束时报错。NPU 上原来的条件判断在捕获模式下走了一条会留下未消费事件的路，现在统一走"安全写法"。
- **对你有什么影响**：NPU 上开启共享专家重叠 + 图捕获不再报 `EndCapture events not joined` 错误。

### 8. [Doc] [AMD] GLM-5.3: Add the MI355X FP8 DCP4 recipe
- **代码层面**：纯文档，新增 MI355X 上 GLM-5.3 FP8 + DCP4 的高并发配方，解释为什么并发 16 以上要用 DCP4（KV 池从 3.3M 提到 10.5M token，命中率 93%→97%），以及 prefill 要禁用 breakable CUDA graph 等细节。
- **新手概念课堂**：DCP（Decode Context Parallelism）是把 KV 缓存切到多张卡上，像把一本书拆成几册分给几个人保管——单卡放不下整本书（长上下文），拆开就能放下，但每次查资料要多一次"跨人协调"的开销。
- **对你有什么影响**：AMD MI355X 用户有了经过实测的高并发配方，直接抄参数即可。

### 9. [Feat] [NPU] kimi-k3.1: Enable DFLASH speculative decoding on Ascend
- **代码层面**：`kimi_k3_hook.py` 把 `linear_attn_verify_backend` 的默认值从"非 gfx95 一律 nv_cutedsl"改为"仅 sm100 用 nv_cutedsl，其他用 triton"（因为 nv_cutedsl 是 CUDA-only）；`ascend_backend.py` 的 `forward_mtp` 新增 ENCODER_ONLY 层的滑动窗口 mask 处理（sparse_mode=4 + mtp_mask）。
- **新手概念课堂**：投机解码像"实习生先猜答案，正式员工再核对"。DFLASH 的草稿层是"非因果 + 滑动窗口"的注意力，昇腾内核需要用"带状 mask"来表达这个窗口，就像给实习生划定"只能看最近 N 个字"的范围。
- **对你有什么影响**：昇腾 NPU 上 Kimi K3.1 可以用 DFLASH 投机解码加速；同时修了一个隐患——非 sm100 平台不再误用 CUDA-only 内核导致首次 verify 崩溃。

### 10. [Test] [NPU] acc: detect the PR pipeline by workflow ref
- **代码层面**：测试里判断"是否 PR 流水线"的条件从 `GITHUB_EVENT_NAME == "pull_request"` 改为 `"pr-test-npu.yml" in GITHUB_WORKFLOW_REF`，因为 PR 流水线现在也会被每日 schedule 触发。
- **新手概念课堂**：原来靠"事件名"判断是不是 PR 跑，但 PR 流水线多了个定时触发，事件名变成 `schedule` 就漏判了。改成看"工作流文件名"更可靠——就像认人不能只看穿什么衣服（事件名），要看身份证（工作流文件）。
- **对你有什么影响**：无直接感知，属于 CI 正确性修复。

### 11. [Fix] hicache: Admit a buffer-mode child whose parent was split while the parent's storage write is in flight
- **代码层面**：`buffer_mode/pipeline.py` 的 `_backup_parent_covered` 中，原来直接查 `storage_existence_cache.contains(parent_last_hash)`，改为调用 `self._pool_covered(PoolName.KV, [parent_last_hash])`——因为"从 in-flight 节点分裂出来的新头节点"自己没有 in-flight 条目，它的页是搭原节点的"顺风车"写入的。
- **新手概念课堂**：这像快递分拣——一个大包裹（父节点）正在打包发往仓库（in-flight 写入），这时从里面拆出一个小包裹（子节点）。小包裹自己没有单独的快递单号（in-flight 条目），但它确实在车上。原来的系统查单号查不到就以为"没发货"，导致重复发货或漏发。修复是"查它搭的那趟车"。
- **对你有什么影响**：HiCache buffer 模式下，高并发 + 前缀共享场景不再出现"子节点数据丢失"的诡异 bug，长上下文缓存更可靠。

### 12. [Test] tests: Fix tests broken by recent refactors
- **代码层面**：多处修复——`_calc_rel_diff` 在两向量全零时用 `torch.where` 避免 0/0=NaN（且无 host sync，图捕获安全）；`qwen2_moe.py` 补上 `get_embed_and_head`/`set_embed_and_head` 方法；测试里给 mock backend 补 `runner_ids` 等字段。
- **新手概念课堂**：`torch.where(cond, a, b)` 是"逐元素选择"，关键优势是不触发 GPU→CPU 同步（不像 Python `if` 要读回标量），所以在图捕获里安全。这是写高性能 PyTorch 代码的常用技巧。
- **对你有什么影响**：调试工具在"两边都是全零"时不再输出 NaN 误导人；Qwen2-MoE 支持 embedding/head 替换（用于权重加载/量化等场景）。

### 13. [Fix] minimax: Cap MSA-v1 decode plan pages at the indexer limit
- **代码层面**：`msa_v1_backend.py` 的 `_prepare_msa_v1_decode_plan` 中，`max_pages` 计算加 `min(..., 8192)` 上限，因为 MSA-v1 indexer 内核最多寻址 8192 页，而 1M 上下文 + 投机解码余量会让页数超限导致启动失败。
- **新手概念课堂**：这像"书架最多放 8192 本书，但你按 1M 字算出来要 8200 本的位置"——不是书放不下，是索引卡（内核的页表寻址位宽）只有 8192 个格子。修复是"按索引卡上限截断"。
- **对你有什么影响**：MiniMax M3.1 在 1M 上下文 + 投机解码配置下不再启动失败。

### 14. [Perf] observability: Reuse the per-stage request latency histogram child per stage
- **代码层面**：`metrics_collector.py` 新增 `_per_stage_req_latency_children` 字典缓存每个 stage 的 histogram child，`observe_per_stage_req_latency` 先查缓存再 `labels()`；新增单测验证复用行为。
- **新手概念课堂**：Prometheus 的 `histogram.labels(stage="prefill")` 每次调用都要"拼字符串 + 查表"，像每次点咖啡都重新报一遍会员号。缓存 child 就是"记住你的会员号"，下次直接用。
- **对你有什么影响**：高 QPS 下观测系统自身开销降低，主路径更快；指标数值不变。

### 15. [Fix] compressed-tensors: Run W4A16 MoE experts with a GELU activation (Gemma 4)
- **代码层面**：`fused_marlin_moe.py` 新增 `gelu_and_mul` / `gelu_tanh_and_mul` 分支；`marlin.py` 的激活断言扩展支持 gelu/gelu_tanh；`compressed_tensors.py` 在 ROCm Triton 路径对非对称量化显式报错（避免静默错误）；`compressed_tensors_wNa16_moe.py` 断言改为"门控 + (silu|gelu|gelu_tanh)"。
- **新手概念课堂**：量化 MoE 的专家计算是"门控线性 → 激活 → 逐元素乘 → 下投影"，激活函数是中间一环。Marlin 内核把这几步融合成一个 kernel，所以每支持一种新激活就要加一条分支。另外"对称量化"是权重以 0 为中心，"非对称"有零点偏移——Triton 路径没传零点，遇到非对称 checkpoint 会算错，所以显式拒绝比静默错误好。
- **对你有什么影响**：Gemma 4 等用 GELU 的 W4A16 MoE 模型可以正确跑量化推理；ROCm 用户遇到不支持的 checkpoint 会看到清晰报错而不是错误输出。

### 16. [Perf] diffusion: Leave components LTX-2.5 does not load out of the all-resident auto tier
- **代码层面**：`base.py` 新增 `unloaded_checkpoint_components()` 钩子（默认空）；`ltx_2_5.py` 实现它，返回 `("prompt_enhancer", "diffusion_decoder")`（除非显式加载 decoder）；`generate.py` 在构建 `ServerArgs` 前就解析 config 里的 `use_diffusion_decoder` 并强制 `load_diffusion_decoder=True`（因为内存规划在构建时就要知道）；`auto_tune.py` 的 `_fits_all_components_resident` 排除未加载组件。
- **新手概念课堂**：checkpoint 里可能带了一些"这个部署根本不用"的组件（比如提示词增强 LLM）。自动内存策略如果把它们也算进去，就会误判"放不下"而走更保守的路径。修复是"先问配置：哪些组件你根本不加载？排除它们再算内存"。
- **对你有什么影响**：LTX-2.5 视频生成在显存够的机器上更容易走"全驻留"快路径，生成更快；显式请求 decoder 时仍会正确加载。

### 17. [Perf] diffusion/minimax_h3: Preserve live packed bounds and capture single-GPU varlen attention in BCG
- **代码层面**：`minimax_h3.py` 引入 `H3PackedPrompt` dataclass 和 `ContextVar` 保存"活"的 packed 边界信息，重放时只拷贝 step 变化的 kwargs；新增 `_single_sequence` 判断单序列场景；大量重构 padding/mask 逻辑以支持单 GPU varlen attention 进 breakable CUDA graph。
- **新手概念课堂**：BCG（Breakable CUDA Graph）是"可打断的图"——大部分计算录制重放，但中间允许插入动态逻辑。难点是"哪些参数每次请求都变，哪些可以静态化"。这里把"边界（bounds）"静态化保存，只让"step 相关 kwargs"在重放时更新，就像录视频时背景固定、只换前景。
- **对你有什么影响**：MiniMax H3 视频生成在单 GPU 上 varlen attention 也能享受图加速，生成更快。

### 18. [Perf] diffusion/qwen-image: Batch the two CFG branches into one forward when the DiT streams
- **代码层面**：`base.py` 新增 `batch_cfg_branch_kwargs()` 钩子（默认 None）；`qwen_image.py` 实现它：检查 cond/uncond kwargs 结构一致、文本长度可右填充 + mask、img_shapes 相同等条件，满足则把两个 CFG 分支拼成 batch=2 一次前向。
- **新手概念课堂**：CFG（Classifier-Free Guidance）要跑两次前向——一次带条件、一次不带，然后混合。如果模型支持 batch，可以把两次拼成一次 batch=2 的前向，GPU 利用率更高。就像去食堂打饭，两份菜单不同但可以一趟端回来，不用跑两趟。代价是要处理"两份文本长度不同"的 padding 和 mask。
- **对你有什么影响**：Qwen-Image 在 DiT 流式加载（显存受限）场景下生成速度提升，因为少了一次前向的启动开销。

### 19. [Fix] diffusion: Exit the worker after a fatal error in a grouped sequential request
- **代码层面**：`pipeline_executor.py` 新增 `_forward_error_ends_the_replica()`：OOM（多卡时）或不可恢复设备错误（`torch.AcceleratorError`、`RuntimeError` 含特定标记）时让 worker 退出而不是返回错误继续服务；`gpu_worker.py` 改用统一的 `is_out_of_memory_error()`。
- **新手概念课堂**：多卡推理时所有卡要同步做集合通信。如果一张卡出错但"假装没事"继续接下一个请求，其他卡还在等它上一轮的通信，就会死锁或数据错乱。正确做法是"出错就退出，让调度器重启整个副本"——像合唱团一个人唱错了，不能让他继续跟着唱，得整队重来。
- **对你有什么影响**：扩散模型服务在遇到 OOM/设备错误时不再"半死不活"，而是快速失败重启，减少诡异 hang 住的情况。

### 20. [Fix] test: Give the bare Scheduler server_args in test_scheduler_recv_skipper
- **代码层面**：测试里用 `Scheduler.__new__` 绕过 `__init__` 构造对象后，补上 `scheduler.server_args = SimpleNamespace(elastic_ep_backend=None)`，因为新代码 `init_watch_dog_memory_saver_input_blocker` 会读这个字段。
- **新手概念课堂**：`__new__` 只创建对象不初始化，像"盖了房子没通水电"。测试为了轻量常这么干，但一旦被测方法开始读某个属性，就得手动"补水电"。这是单元测试里 mock 不完整导致的典型断裂。
- **对你有什么影响**：无直接感知，属于测试随重构同步更新。

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)