# 🚀 vLLM & SGLang 每日更新 — 2026-10-13

> 自动生成于 2026-10-13 08:24 UTC | AI 解读: ✅ 含代码解读

## vllm
## 💡 今日关键词（放在最前面！）

- **关键词：ROCm / AMD GPU 适配**
  - **一句话通俗解释**：ROCm 是 AMD 版的 CUDA，今天有 5 个 commit 都在为 AMD 显卡做性能优化和功能适配。
  - **为什么社区在关注**：vLLM 正在大力支持非 NVIDIA 硬件，说明异构计算（同一套代码跑不同厂商 GPU）是行业大趋势，不再被 NVIDIA 垄断。
  - **对新手意味着什么**：如果你只会写 CUDA，可以开始学 Triton 或 HIP，这类"跨厂商算子"人才非常稀缺，是差异化竞争力。

- **关键词：混合模型（Mamba + Attention）**
  - **一句话通俗解释**：Mamba 是一种比 Transformer 更省显存的序列模型，把它和传统 Attention 混在一起用是当前热点。
  - **为什么社区在关注**：长上下文场景下，纯 Attention 太贵，混合架构能在保持效果的同时大幅省显存。
  - **对新手意味着什么**：理解"前缀缓存"和"状态缓存"的区别，是理解下一代 LLM 推理架构的关键。

- **关键词：量化（FP8 / MXFP8 / NVFP4）**
  - **一句话通俗解释**：把模型权重和中间结果从 16 位压缩到 8 位甚至 4 位，让显存和带宽都省一半以上。
  - **为什么社区在关注**：大模型越来越大，量化是"跑得动"的唯一出路，特别是 FP8 已成主流。
  - **对新手意味着什么**：学会看懂 `scale`（缩放因子）、`dequant`（反量化）这些概念，是入门推理优化的第一课。

- **关键词：KV Cache 规划与缓存管理**
  - **一句话通俗解释**：KV Cache 是大模型推理时"记住上下文"的显存池，怎么分配、复用、淘汰它决定了吞吐量。
  - **为什么社区在关注**：今天有重构、有前缀缓存、有 Mamba 状态缓存，说明这块是性能瓶颈的核心战场。
  - **对新手意味着什么**：想搞推理优化，KV Cache 是绕不过去的一课，建议从 PagedAttention 论文读起。

- **关键词：LoRA 与专家并行（MoE）**
  - **一句话通俗解释**：LoRA 是给大模型"打补丁"的低成本微调方法，MoE 是"多个专家分工"的稀疏模型结构。
  - **为什么社区在关注**：当两者组合（MoE + LoRA + 张量并行）时，切片对齐问题非常容易出错，今天的 bugfix 就是典型。
  - **对新手意味着什么**：想玩微调部署，一定要理解"张量并行后权重被切分"这件事，否则会踩到维度不匹配的坑。

---

## 🧪 重要知识点 & 动手实验

- **知识点 1：KV Cache 前缀缓存与 Mamba 状态缓存**
  - **是什么**：Transformer 缓存每一层的 Key/Value，Mamba 缓存的是一个"状态向量"（相当于 RNN 的 hidden state）。前缀缓存就是"相同开头不重算"。
  - **为什么重要**：多轮对话、系统提示词共享场景下，前缀缓存能省 50% 以上算力。但 Mamba 的状态只在"块边界"有效，所以必须对齐块大小，否则会命中错误状态。不懂这一点，你会遇到"回答突然变傻"的诡异 bug。
  - **动手试试**：
    ```bash
    # 启动一个带前缀缓存的 Qwen 模型
    vllm serve Qwen/Qwen2.5-1.5B-Instruct \
      --enable-prefix-caching \
      --max-model-len 4096
    # 连续发送两条相同系统提示的请求，观察日志里的 prefix cache hit rate
    ```
  - **预期结果**：第二次请求日志会出现 `prefix cache hit rate: 0.XX`，延迟明显下降。如果命中率是 0，说明提示词没对齐，你就理解为什么需要"块边界对齐"了。

- **知识点 2：FP8 量化与反量化（Dequant）**
  - **是什么**：FP8 用 8 位表示浮点数，范围小、精度低，但显存减半。用之前要用 `scale`（缩放因子）把数值"拉回"到 FP16 再算。
  - **为什么重要**：Commit 4 里 Ampere（A100）不支持原生 FP8 指针，只能靠"位运算解码"手动还原。不懂反量化，你看到 FP8 模型输出乱码会一头雾水。
  - **动手试试**：
    ```python
    import torch
    # 模拟 FP8 e4m3 编码：把一个 fp16 值量化再反量化
    x = torch.tensor([1.5, -0.25, 3.0], dtype=torch.float16)
    scale = x.abs().max() / 448.0  # e4m3 最大值 448
    x_q = (x / scale).to(torch.float8_e4m3fn)
    x_deq = x_q.to(torch.float16) * scale
    print("原始:", x, "反量化:", x_deq)
    ```
  - **预期结果**：`x_deq` 和 `x` 非常接近但不完全相等，差值就是量化误差。你会直观感受到"为什么需要 scale"。

- **知识点 3：张量并行下的权重切片（LoRA + MoE）**
  - **是什么**：张量并行（TP）把一个大矩阵按列或按行切成 N 份，每张 GPU 只存一份。LoRA 微调的权重也必须按同样规则切片，否则维度对不上。
  - **为什么重要**：Commit 5 修的就是"切片用了 padding 后的大小，导致错位"。这是分布式训练/推理最经典的坑：**所有并行维度必须严格一致**。
  - **动手试试**：
    ```python
    import torch
    # 模拟 TP=2 下切分一个 expert 的 LoRA B 矩阵
    W = torch.arange(16).reshape(4, 4)  # (output, rank)
    tp_size, tp_rank = 2, 0
    shard = W.shape[0] // tp_size
    sliced = W[tp_rank*shard : (tp_rank+1)*shard]
    print("rank0 拿到:", sliced)
    # 如果 base 权重被 padding 到 6 行，这里就会错位！
    ```
  - **预期结果**：你会看到 rank0 只拿到前 2 行。如果换成 padding 后的尺寸去切，切出来的行数就对不上了——这就是 bug 的本质。

- **知识点 4：JSON Schema 类型强制转换（工具调用）**
  - **是什么**：LLM 生成工具调用参数时，模型可能输出 `"3.0"` 而不是 `3`，但 JSON Schema 声明是 `integer`。解析器需要智能转换。
  - **为什么重要**：Commit 14/16 同时修了 Python 和 Rust 两个前端，说明这种"边界情况"在真实工具调用中非常常见。不处理用户会看到 `type error`。
  - **动手试试**：
    ```python
    def coerce(raw, types):
        for t in types:
            try:
                if t == "integer":
                    try:
                        return int(raw)
                    except ValueError:
                        n = float(raw)
                        if n.is_integer():
                            return int(n)
                if t == "number":
                    return float(raw)
            except ValueError:
                continue
        return raw
    print(coerce("3.0", ("integer",)))   # 3
    print(coerce("3.5", ("integer",)))   # "3.5"（不合法，原样返回）
    print(coerce("3.5", ("integer","number")))  # 3.5
    ```
  - **预期结果**：输出 `3`、`"3.5"`、`3.5`。你就理解了"为什么 `3.0` 算整数而 `3.5` 不算"——JSON Schema 规定整数是"没有小数部分"。

- **知识点 5：多进程 spawn 与 monkeypatch 的坑**
  - **是什么**：Linux 默认 `fork` 创建子进程会继承父进程内存；但 ROCm/XPU 强制用 `spawn`，子进程会重新 import 模块，父进程的 monkeypatch 全部失效。
  - **为什么重要**：Commit 12 就是解决"测试里改了函数，子进程却看不到"的问题。写分布式测试时这是必踩的坑。
  - **动手试试**：
    ```bash
    # 写一个 sitecustomize.py 放到 PYTHONPATH 里
    echo 'import builtins; builtins.MY_FLAG = "hello"' > /tmp/sitecustomize.py
    PYTHONPATH=/tmp python -c "import multiprocessing as mp; \
      mp.set_start_method('spawn'); \
      p = mp.Process(target=lambda: print(getattr(__import__('builtins'),'MY_FLAG','MISSING'))); \
      p.start(); p.join()"
    ```
  - **预期结果**：输出 `hello`。如果你用 `monkeypatch.setattr` 而不是 `sitecustomize`，spawn 子进程会打印 `MISSING`——这就是 Commit 12 要解决的问题。

---

## 🔥 重要更新

- **Commit 3：移除 `mamba_cache_mode="all"` 残留** — Mamba 前缀缓存正式进入"默认开启"路线，配置简化，是架构成熟的重要标志。
- **Commit 6：DeepGEMM 升级到 26/09/30 版本 + MXFP8 支持** — 引入 MX 格式缩放，为 SM100 上的稀疏 MQA 铺路，是硬件新特性落地的关键一步。
- **Commit 11：ROCm AiterExperts 按量化类型拆分** — AMD 上 MoE 内核从"一个万能类"变成"按量化方案分家的多个类"，可维护性和性能双提升。
- **Commit 18：KV Cache 规划逻辑独立成文件** — 大规模重构，把规划逻辑从 `kv_cache_utils.py` 拆出来，说明这块逻辑复杂度已经高到需要单独模块。

---

## 📋 逐条解读

### 1. [Tests][Mamba] Wait for accepted counts in fake MTP proposer
- **代码层面**：改的是 `tests/v1/e2e/general/test_mamba_prefix_cache.py`，在读取 `num_accepted_tokens_cpu` 之前加了一句 `synchronize()`，等待 GPU 到 CPU 的异步拷贝完成。
- **新手概念课堂**：GPU 和 CPU 之间拷贝数据默认是"异步"的，就像你下单后快递还没到就去开箱，会拿到旧数据。`synchronize()` 就是"等快递到了再拆"。
- **对你有什么影响**：普通用户无感知，但如果你是测试维护者，不加这个同步会看到"偶发失败"的诡异测试结果。

### 2. [Bugfix][EC Connector] Don't create hash directories on cache-miss lookups
- **代码层面**：改的是 `example_connector.py`，给 `_generate_filename_debug` 加了 `create_folder=False` 参数，查缓存未命中时不再创建空目录。
- **新手概念课堂**：就像你去图书馆查书，如果书不存在，管理员不应该凭空给你建个空书架。原来的代码每次查询都会建目录，磁盘很快被空文件夹塞满。
- **对你有什么影响**：使用 EC Connector 做多模态编码缓存时，磁盘不会被无意义的空目录污染。

### 3. [Core][Mamba] Remove leftovers of mamba_cache_mode="all"
- **代码层面**：删掉了很多地方对 `--mamba-cache-mode align` 的引用，包括文档、测试、CI 配置。Mamba 前缀缓存现在默认生效。
- **新手概念课堂**：以前要手动"打开开关"才能用 Mamba 前缀缓存，现在这个开关被焊死在"开"的位置了。就像手机从"需要手动开 WiFi"变成"默认连 WiFi"。
- **对你有什么影响**：启动 Mamba 模型（如 Kimi-Linear）时不用再加 `--mamba-cache-mode align` 参数，配置更简洁。

### 4. [Model][Qwen4Exp] FP8 QSA main KV cache on Ampere
- **代码层面**：改的是 `qsa.py` 里的 Triton 内核，新增 `_maybe_dequant` 函数，用位运算手动把 FP8 e4m3 字节解码成 FP16，让 A100（sm_80）也能跑 FP8 KV Cache。
- **新手概念课堂**：老显卡不支持新格式的"原生解码器"，只能靠软件模拟。就像老式收音机收不到数字广播，但可以用转接器解调。
- **对你有什么影响**：如果你有 A100/A30，现在可以跑 Qwen4Exp 的 FP8 KV Cache，显存省一半，速度也不慢。

### 5. [Bugfix][LoRA] Slice expert LoRA by the unpadded per-rank size under TP
- **代码层面**：改的是 `fused_moe.py`，新增 `_tp_shard_size` 属性，用"未 padding 的 per-rank 大小"来切片 LoRA 权重，并在拼接 w1/w3 时补 padding。
- **新手概念课堂**：就像把一块蛋糕分给 4 个人，但盘子是圆的（有 padding），你得按"蛋糕实际大小"切，不能按"盘子大小"切。
- **对你有什么影响**：使用 MoE + LoRA + TP 组合时，不再出现维度错位的崩溃或输出乱码。

### 6. [Kernel][Build] Bump DeepGEMM to the 26/09/30 release
- **代码层面**：改了 CMake 里的 DeepGEMM 版本哈希，新增 `locality_domain.py` 安装；`cache_kernels.cu` 里新增 `use_mx_scale` 参数，支持把 scale 打包成 MX 字节。
- **新手概念课堂**：MX 格式是一种"每 32 个元素共享一个缩放因子"的量化方案，比每个元素独立缩放更省空间，是新一代硬件的标配。
- **对你有什么影响**：在 Blackwell（SM100）等新卡上跑 FP8 稀疏注意力会更高效，但需要重新编译 vLLM。

### 7. [Attention][DCP] Support a2a DCP comm backend for FlashMLA sparse
- **代码层面**：删掉了 `flashmla_sparse.py` 里"只允许 ag_rs 后端"的限制，新增测试覆盖 a2a（all-to-all）通信后端。
- **新手概念课堂**：DCP（Decode Context Parallel）是把长上下文切给多张卡算。通信方式有"先聚合再规约"（ag_rs）和"直接点对点"（a2a），后者在特定拓扑下更快。
- **对你有什么影响**：部署 FlashMLA 稀疏注意力时，可以选择更快的 a2a 通信后端，长上下文解码吞吐更高。

### 8. [Model] Add Clef-Flash support
- **代码层面**：新增 `ClefFlashForJointSchema` 架构识别逻辑，在 `structured_decisions.md` 里加了文档，支持 Cloudflare 的 Clef-Flash 模型。
- **新手概念课堂**：这是一种"一次前向给所有问题打分"的决策模型，不是生成式模型，而是 pooling 模型（输出分数而非 token）。
- **对你有什么影响**：可以直接 `vllm serve Cloudflare/clef-flash` 跑结构化决策任务，支持图片和视频输入。

### 9. [ROCm][Perf] Skip redundant sparse index remap on non-indexer layers
- **代码层面**：新增 `_INDEX_EPOCHS` 计数器，用 epoch 号判断索引是否已经重映射过，避免重复计算。
- **新手概念课堂**：就像 Excel 里改了数据后只重算受影响的单元格，而不是整张表重算。epoch 号就是"数据版本号"。
- **对你有什么影响**：AMD 上跑稀疏 MLA 注意力时，非 indexer 层不再做无用功，解码速度提升。

### 10. [ROCm][Perf] Skip cleaning sparse prefill MQA logits
- **代码层面**：在 `rocm_fp8_mqa_logits` 里传入 `clean_logits=False`，跳过对 logits 的清理步骤。
- **新手概念课堂**：清理 logits 是把无效位置设为 -inf，防止 softmax 溢出。但在稀疏 prefill 场景下这些位置本来就不会被选中，清理是多余的。
- **对你有什么影响**：AMD 上稀疏 prefill 更快，尤其是长 prompt 场景。

### 11. [ROCm][MoE] Split AiterExperts by weight-quantization key
- **代码层面**：把原来的 `AiterExperts` 拆成 `AiterUnquantizedExperts`、`AiterFp8BlockExperts`、`AiterFp8TensorExperts`、`AiterMxfp4Experts` 等多个类。
- **新手概念课堂**：以前是一个"万能工具箱"，现在按量化类型分成多个专用工具箱，每个只装对应工具，找起来更快、维护更清晰。
- **对你有什么影响**：AMD 上跑 MoE 模型时，不同量化方案走各自的优化路径，性能和稳定性都更好。

### 12. [XPU] [Test] Extend spawn-safe sitecustomize monkeypatch to XPU
- **代码层面**：把 ROCm 专用的 `rocm_sitecustomize_factory` 改名为通用的 `spawn_sitecustomize_factory`，同时支持 XPU（Intel GPU）。
- **新手概念课堂**：Intel GPU 和 AMD GPU 一样用 `spawn` 启动子进程，所以父进程的 monkeypatch 会失效，必须用 `sitecustomize.py` 注入。
- **对你有什么影响**：Intel GPU 用户跑分布式测试时不再有"改了函数但子进程看不到"的困惑。

### 13. [Bugfix][Model] Fix EXAONE-4.5 image inputs: ViT received un-normalised pixels
- **代码层面**：在 `exaone4_5.py` 里给视觉编码器加了 `input_norm` 模块，对图像像素做归一化。
- **新手概念课堂**：图像像素原始范围是 0-255，模型训练时用的是归一化到 [-1,1] 或 [0,1] 的输入。不归一化就像给模型喂"生米"，它没法消化。
- **对你有什么影响**：EXAONE-4.5 的图片理解能力恢复正常，之前可能输出胡言乱语。

### 14. [Bugfix] Accept whole-number floats for integer tool arguments
- **代码层面**：改的是 `response_parser.py` 的 `_coerce` 函数，当 `int(raw)` 失败时，尝试 `float(raw)` 并检查是否为整数。
- **新手概念课堂**：JSON Schema 规定 `3.0` 是合法的整数（因为小数部分为 0），但 Python 的 `int("3.0")` 会报错，需要先转 float 再转 int。
- **对你有什么影响**：模型调用工具时输出 `"count": 3.0` 不再报错，工具调用成功率提升。

### 15. [Bugfix] Preserve independent tokenizer code_revision in snapshot resolution
- **代码层面**：改的是 `resolve_tokenizer_args`，当 `code_revision` 和 `revision` 不同时，保留 Hub ID 而不是解析成本地快照路径。
- **新手概念课堂**：`revision` 是数据版本，`code_revision` 是代码版本。如果强行把两者都指向数据快照，动态加载的 tokenizer 代码会被"数据版本"覆盖，导致代码找不到。
- **对你有什么影响**：使用带远程代码的 tokenizer 且指定了独立代码版本时，不再报"tokenization_fast.py 找不到"。

### 16. [Rust Frontend] Accept whole-number floats for integer tool arguments
- **代码层面**：Rust 前端同步修复，新增 `convert_whole_number_text` 函数，逻辑与 Python 版一致。
- **新手概念课堂**：vLLM 有两个前端（Python 和 Rust），修 bug 要"两边都改"，否则用 Rust 前端的用户还是会踩坑。
- **对你有什么影响**：使用 Rust 前端时，工具调用的整数参数兼容性一致。

### 17. [Test][Renderer] Cover the reasoning -> reasoning_content mirror
- **代码层面**：新增 3 个测试，验证 assistant 消息的 `reasoning` 字段会被镜像到 `reasoning_content`，其他角色不会。
- **新手概念课堂**：`reasoning` 是新的思维链字段，`reasoning_content` 是旧字段。为了兼容老模板，新字段要"抄一份"给旧字段。
- **对你有什么影响**：普通用户无感知，但使用旧版 chat template 时思维链内容能正确渲染。

### 18. [Core][Refactor] Move KV cache planning logic to kv_cache_planning.py
- **代码层面**：把 `kv_cache_utils.py` 里的规划逻辑（如内存估算、块数计算）拆到新文件 `kv_cache_planning.py`。
- **新手概念课堂**：就像把"厨房"和"仓库"分开，做饭的代码和存菜的代码不再混在一起，各自职责清晰。
- **对你有什么影响**：无功能变化，但如果你在改 KV Cache 相关代码，需要去新文件里找。

### 19. [ROCm][Perf][DSv4.1] Enable decoder SWA bounded replay
- **代码层面**：新增 `test_deepseek_v41_amd_replay_forward.py`，验证 AMD 上 DeepSeek V4.1 的滑动窗口注意力（SWA）在 prefill 超长时的"有界重放"行为。
- **新手概念课堂**：SWA 只保留最近 N 个 token 的 KV，超出的部分被"丢掉"。但如果 prompt 太长，需要"重放"来重建状态，有界重放就是控制重放范围。
- **对你有什么影响**：AMD 上跑 DeepSeek V4.1 长 prompt 时，显存占用可控，不会 OOM。

### 20. [ROCm][Perf][DSv4.1] Mono decode on every backbone layer
- **代码层面**：新增 `test_dsv41_mono_decode.py` 测试，验证每一层 backbone 都走"单体解码"（mono decode）路径，包括第一层从 embedding 直接进入。
- **新手概念课堂**："单体解码"是把多个小算子融合成一个大 kernel，减少启动开销。让每一层都走这条路，等于"全程高速不换挡"。
- **对你有什么影响**：AMD 上 DeepSeek V4.1 解码速度进一步提升，尤其是首 token 延迟。

## sglang
## 💡 今日关键词

- **AMD ROCm 生态适配**：让 SGLang 在 AMD GPU 上也能跑得又稳又快。
  - **为什么社区在关注**：今天 20 个 commit 里有 6 个以上是 AMD/ROCm 相关（MORI 传输、KDA 内核、DSV4 量化、CI 测试），说明 AMD 正在成为 NVIDIA 之外真实可用的推理硬件，社区在补齐生态短板。
  - **对新手意味着什么**：如果你只会 CUDA，现在补 ROCm/HIP 技能是差异化优势；很多 AMD 适配工作还在早期，容易做出贡献。

- **PD 分离（Prefill/Decode Disaggregation）**：把"读题"和"答题"两个阶段拆到不同机器上跑。
  - **为什么社区在关注**：今天有 3 个 PD 相关 commit（NIXL、MORI、异构 TP），这是大模型推理降本增效的核心架构。
  - **对新手意味着什么**：理解 PD 分离是进入推理系统工程师岗位的敲门砖，涉及 KV 传输、负载均衡、异构拓扑等真实工程问题。

- **量化（FP8/FP4/MXFP8）**：用更少的比特表示模型权重和激活，省显存、加速计算。
  - **为什么社区在关注**：DSV4.1 的 FP4 indexer 融合、MXFP8 KV cache 限制，说明量化已经深入到 attention 内部。
  - **对新手意味着什么**：量化不再是"调个参数"，而是需要理解数值格式、舍入策略、硬件指令。

- **Triton/JIT 内核**：用 Python 写 GPU 内核，运行时编译。
  - **为什么社区在关注**：fast-math、NGRAM 树重建、LTX-2 视频加速都走 JIT 路线，说明"写一次、多硬件编译"是趋势。
  - **对新手意味着什么**：Triton 比 CUDA 更容易上手，是新手进入 GPU 编程的最佳切入点。

- **路由器负载均衡**：把请求分给最空闲的 worker。
  - **为什么社区在关注**：两个 power_of_two 相关 commit，说明网关层的调度策略直接影响端到端延迟。
  - **对新手意味着什么**：这是分布式系统的经典问题（一致性哈希、最少连接、P2C），面试常考。

## 🧪 重要知识点 & 动手实验

- **知识点 1：Power-of-Two Choices 负载均衡**
  - **是什么**：随机挑两个 worker，把请求给负载更低的那个。看似简单，效果接近全局最优。
  - **为什么重要**：在大规模集群里，全局轮询开销大且容易惊群；P2C 只用两次采样就能做出好决策。不懂的话，你会以为"随机分配"就够了，结果长尾延迟爆炸。
  - **动手试试**：
    ```python
    import random
    # 模拟 10 个 worker，负载随机
    workers = [{"id": i, "load": random.randint(0, 100)} for i in range(10)]
    # 随机两个，选负载低的
    a, b = random.sample(workers, 2)
    chosen = a if a["load"] < b["load"] else b
    print(f"选中 worker {chosen['id']}, 负载 {chosen['load']}")
    # 对比：纯随机
    print(f"纯随机选中负载: {random.choice(workers)['load']}")
    ```
  - **预期结果**：P2C 选中的负载通常明显低于随机选择。多跑几次，你会看到 P2C 的平均负载更接近全局最小值。

- **知识点 2：Fast-Math 编译选项**
  - **是什么**：让编译器用"更快但略不精确"的数学指令（如快速 exp、近似除法）。
  - **为什么重要**：激活函数（SiLU/GELU）里 exp 调用极频繁，fast-math 能提速 10-20%。但精度敏感场景（如 Blackwell 上的某些模型）会出错，所以需要按架构开关。
  - **动手试试**：
    ```bash
    # 用 nvcc 对比编译
    cat > test.cu << 'EOF'
    #include <cuda_runtime.h>
    #include <math.h>
    __global__ void k(float* x) { x[threadIdx.x] = expf(x[threadIdx.x]); }
    EOF
    nvcc -arch=sm_90 -c test.cu -o test_precise.o
    nvcc -arch=sm_90 --use_fast_math -c test.cu -o test_fast.o
    cuobjdump -sass test_precise.o | grep -i "MUFU\|EX2" | head
    cuobjdump -sass test_fast.o | grep -i "MUFU\|EX2" | head
    ```
  - **预期结果**：fast-math 版本会出现 `MUFU.EX2`（硬件近似指数），精确版本会调用更长的软件实现。这就是速度差异的来源。

- **知识点 3：KV Cache 与 PD 分离**
  - **是什么**：KV cache 是模型生成时缓存的注意力键值对；PD 分离把 prefill（算 KV）和 decode（用 KV）分到不同 GPU。
  - **为什么重要**：PD 分离能让 prefill 用大 batch 高吞吐、decode 用低延迟小 batch，各取所长。但 KV 要在网络上传，异构 TP（不同并行度）时传输逻辑极其复杂。
  - **动手试试**：
    ```python
    import torch
    # 模拟一个 KV cache 分片
    batch, heads, seq, dim = 1, 8, 128, 64
    k = torch.randn(batch, heads, seq, dim)
    # 假设 prefill 用 4 卡 TP，每卡 2 个 head
    prefill_shards = k.chunk(4, dim=1)
    print(f"prefill 每卡 shape: {prefill_shards[0].shape}")
    # decode 用 2 卡 TP，每卡需要 4 个 head → 需要合并两个 prefill 分片
    decode_shard_0 = torch.cat([prefill_shards[0], prefill_shards[1]], dim=1)
    print(f"decode 卡0 shape: {decode_shard_0.shape}")
    ```
  - **预期结果**：你会看到 head 维度的重新切分。这正是 commit 7 里 `_get_mha_head_slice_blocks` 做的事情——把 prefill 的 head 分片重新映射到 decode 的 head 分片。

- **知识点 4：FP4 量化与舍入到最近偶数（RNE）**
  - **是什么**：FP4 只有 4 位，能表示 16 个值。把 FP32 压到 FP4 时，需要决定"舍入到哪个值"。
  - **为什么重要**：舍入策略直接影响模型精度。RNE（round-to-nearest-even）是 IEEE 标准，但硬件实现各有差异，手写内核必须精确复现。
  - **动手试试**：
    ```python
    import numpy as np
    # E2M1 FP4 可表示的正值
    fp4_values = np.array([0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    def quantize_rne(x):
        # 找最近的两个值，平局时选偶数索引
        idx = np.searchsorted(fp4_values, x)
        if idx == 0: return fp4_values[0]
        if idx >= len(fp4_values): return fp4_values[-1]
        lo, hi = fp4_values[idx-1], fp4_values[idx]
        if x - lo < hi - x: return lo
        if hi - x < x - lo: return hi
        # 平局：选偶数索引
        return lo if (idx-1) % 2 == 0 else hi
    for v in [0.75, 1.25, 2.5, 3.5]:
        print(f"{v} -> {quantize_rne(v)}")
    ```
  - **预期结果**：0.75→0.5 或 1.0（看平局规则），2.5→2.0（平局选偶数索引），3.5→4.0。这就是 commit 11 里 `_fp4_e2m1_code_rne_lean` 用 `rint` 实现的逻辑。

- **知识点 5：异步事件循环中的阻塞操作**
  - **是什么**：Python asyncio 是单线程的，任何同步阻塞（如 tokenizer）都会卡住整个服务。
  - **为什么重要**：长 prompt 的 tokenization 可能耗时几十毫秒，期间所有请求都排队。用 `run_in_executor` 把阻塞操作扔到线程池，事件循环就能继续处理其他请求。
  - **动手试试**：
    ```python
    import asyncio, time
    def blocking_tokenize(text):
        time.sleep(0.1)  # 模拟 tokenizer
        return len(text.split())
    async def bad_handler(texts):
        return [blocking_tokenize(t) for t in texts]
    async def good_handler(texts):
        loop = asyncio.get_running_loop()
        return await asyncio.gather(*[
            loop.run_in_executor(None, blocking_tokenize, t) for t in texts
        ])
    async def main():
        texts = ["hello world"] * 10
        t0 = time.time(); await bad_handler(texts); print(f"阻塞: {time.time()-t0:.2f}s")
        t0 = time.time(); await good_handler(texts); print(f"线程池: {time.time()-t0:.2f}s")
    asyncio.run(main())
    ```
  - **预期结果**：阻塞版本约 1.0 秒（串行），线程池版本约 0.1-0.2 秒（并行）。这就是 commit 15 的优化效果。

## 🔥 重要更新

- **Commit 15（tokenizer 性能优化）**：把 tokenization 从事件循环移到线程池，长 prompt 场景下服务吞吐可能翻倍。这是所有 OpenAI 兼容 API 用户都能感知的改进。
- **Commit 7（异构 TP 的 SWA KV 传输）**：让 prefill 和 decode 可以用不同的 TP 大小，混合滑动窗口模型（如 gpt-oss-120b）的 PD 部署更灵活，直接降低硬件成本。
- **Commit 9（修复 power_of_two + dp-aware）**：修复了路由器在 DP-aware 模式下的 worker 数量校验 bug，否则网关会拒绝启动。
- **Commit 20（复用 HF LFM2-MoE 配置）**：删掉 100 多行自定义配置代码，直接继承 HF，减少维护负担，也意味着 LFM2-MoE 模型在 SGLang 上的支持更稳定。

## 📋 逐条解读

### 1. [Perf] Enable fast-math for JIT activations on Blackwell
- **代码层面**：`arch.py` 里的 `get_activation_cuda_cflags()` 从"HIP 或 Blackwell 返回空列表"改成"只要是 CUDA 就返回 `--use_fast_math`"。逻辑简化了，但行为上 Blackwell 现在也启用 fast-math。
- **新手概念课堂**：fast-math 就像让计算器"四舍五入到小数点后两位"——快，但不够精确。之前 Blackwell 因为精度问题被排除，现在通过其他方式（可能是内核改写）解决了。
- **对你有什么影响**：在 Blackwell GPU 上跑激活函数会更快的，但如果你做数值敏感的科学计算，要留意精度变化。

### 2. [router] power_of_two: in-flight request load for --dp-aware decode
- **代码层面**：`power_of_two.rs` 新增 `LoadMetric` 枚举，支持按 token 数、按 in-flight 请求数、或全局最少请求数来选 worker。通过环境变量 `SGLANG_ROUTER_P2C_LOAD_METRIC` 控制。
- **新手概念课堂**：就像餐厅领班安排座位，"看哪桌菜快上完了"（token 负载）还是"看哪桌客人少"（请求数）。不同场景适合不同指标。
- **对你有什么影响**：DP-aware 部署下，你可以通过环境变量调优路由策略，减少长尾延迟。

### 3. [PD][NIXL] Fix staging buffer startup by accepting slot_layer_ids
- **代码层面**：`nixl/conn.py` 加了一段注释，说明 `slot_layer_ids` 的顺序和 `kv_args.kv_layer_ids` 不同（有 draft 模型时）。
- **新手概念课堂**：staging buffer 像"中转仓库"，prefill 的 KV 先放这里，再传给 decode。如果仓库货架编号和实际货物对不上，就会拿错东西。
- **对你有什么影响**：修复了带 draft 模型的 PD 部署启动失败问题，普通用户无感知但更稳定。

### 4. [Docs] Cookbook: refresh SemiAnalysis AgentX launches
- **代码层面**：更新了 docs 里的 JSX 代码片段，同步 SemiAnalysis InferenceX 的最新 commit 和 PR 状态。
- **新手概念课堂**：这是"文档即代码"的实践——启动命令从上游仓库自动生成，避免手工维护出错。
- **对你有什么影响**：如果你参考 cookbook 部署 DeepSeek-V4.1 或 GLM-5.2，会看到最新的推荐配置。

### 5. [AMD][PD] enable mori transfer for PD decode DCP
- **代码层面**：`pd_disaggregation_hook.py` 把 `mori` 加入允许的后端列表；`mori/conn.py` 新增 DCP（decode context parallel）相关的字段和传输计划构建。
- **新手概念课堂**：MORI 是 AMD 的 KV 传输库，类似 NVIDIA 的 NIXL。DCP 是把 decode 阶段的上下文切到多卡上，需要跨卡传输 KV。
- **对你有什么影响**：AMD GPU 用户现在可以用 MORI 做 PD 分离 + DCP，部署更灵活。

### 6. [Diffusion] Speed up LTX-2 guidance and video output
- **代码层面**：`residual_gate_add.cuh` 新增 `kBatchedRow` 模式，支持每个 batch 一行 gate；kernel 里 gate 索引计算从简单取模改成按 batch 偏移。
- **新手概念课堂**：视频生成里，每个 batch 的引导强度可能不同。以前只能"全 batch 共享一行"或"每 token 一行"，现在可以"每 batch 一行"，更灵活。
- **对你有什么影响**：LTX-2 视频生成更快，且支持更细粒度的引导控制。

### 7. [PD] Support heterogeneous TP for hybrid SWA KV transfer
- **代码层面**：`base/conn.py` 的 `KVArgs` 新增 `swa_kv_cache_layout` 字段；`common/conn.py` 新增 `_get_mha_head_slice_blocks` 方法，处理不同 TP 大小下的 head 分片重映射；对 MXFP8 KV cache 直接报错。
- **新手概念课堂**：TP（张量并行）是把一个矩阵切成多份放到多卡。prefill 用 4 卡、decode 用 2 卡时，KV 的 head 分片方式不同，需要"翻译"。
- **对你有什么影响**：混合滑动窗口模型（如 gpt-oss-120b）现在可以用不等 TP 做 PD 分离，硬件利用率更高。

### 8. [ROCm][Spec] Route NGRAM tree reconstruction to the JIT kernel on HIP
- **代码层面**：`ngram_worker.py` 的导入条件从 `is_cuda()` 改成 `is_cuda() or is_hip()`；测试文件注册 AMD CI。
- **新手概念课堂**：NGRAM 推测解码用一棵"候选树"来并行验证多个 token。树重建是热点，JIT 内核比 Python 快得多。
- **对你有什么影响**：AMD GPU 上 NGRAM 推测解码性能提升，且有了 CI 保障。

### 9. [router] Fix power_of_two with --dp-aware
- **代码层面**：`validation.rs` 在 `dp_aware` 模式下跳过"至少 2 个 worker"的校验（因为 DP-aware 的 URL 会在运行时展开）；`worker_manager.rs` 重构了负载查询逻辑。
- **新手概念课堂**：DP-aware 模式下，一个 URL 代表多个 DP rank，启动时看不到全部 worker，所以不能提前校验数量。
- **对你有什么影响**：修复了 DP-aware + power_of_two 组合下网关启动失败的问题。

### 10. [AMD][DI][CI] dsv4pro-fp8 2p1d-ep16-mtp: raise prefill mem fraction
- **代码层面**：YAML 配置文件里 `prefill_mem_fraction_static` 从 0.85 提到 0.88。
- **新手概念课堂**：mem fraction 是"给 KV cache 留多少显存"。MORI EP v2 本身占内存，所以要提高 prefill 的显存占比。
- **对你有什么影响**：AMD CI 用户跑这个 recipe 时不会 OOM。

### 11. [AMD] Fuse prefill index-Q RoPE fake-quant and fp4 quantize kernels in DSV4.1
- **代码层面**：`fp4_indexer_hip.py` 新增 `_fp4_e2m1_code_rne_lean` 和 `_index_q_quant_groups` 两个 Triton 内核，把 RoPE 后的 fake-quant 和 FP4 量化融合成一步。
- **新手概念课堂**：RoPE 是旋转位置编码，fake-quant 是"假装量化"（保留 FP32 但模拟量化误差）。融合后少了一次内存读写。
- **对你有什么影响**：DSV4.1 在 AMD 上 prefill 更快。

### 12. [AMD] Fix Flux2 kv-gather abort on non-contiguous K/V
- **代码层面**：`group_coordinator.py` 的 `all_gather` 加 `input_.contiguous()`；`attention/layer.py` 的 `_gather_sharded_sequence` 对 tensor 调 `.contiguous()`；新增单元测试。
- **新手概念课堂**：`chunk` 出来的 tensor 是"视图"，底层内存不连续。集合通信要求连续内存，否则报错。
- **对你有什么影响**：Flux2 图像生成在 AMD 上不再崩溃。

### 13. [AMD][Kimi-K3] Use public FlyDSL APIs in KDA decode kernels
- **代码层面**：`kimi_k3_kda_decode.py` 从直接用 MLIR 内部 API 改成用 FlyDSL 公开 API（如 `fx.Vector(...).reduce("add")`），分支用闭包代替 `scf.IfOp`。
- **新手概念课堂**：内部 API 像"直接改发动机零件"，公开 API 像"用方向盘"。后者更稳定、可移植。
- **对你有什么影响**：Kimi-K3 的 KDA 解码内核在 AMD 上更易维护，未来升级 FlyDSL 不会破坏代码。

### 14. [diffusion] perf: implement paint_use_remesh for Hunyuan3D
- **代码层面**：`hunyuan3d.py` 配置注释；`paint.py` 的 `_unwrap_mesh` 在 UV 展开前先 `mesh_simplify`；`mesh3d_utils.py` 新增 QEM 网格简化算法（约 130 行）。
- **新手概念课堂**：UV 展开像"把 3D 球面剪开铺平贴图"。面数越多越慢，先简化到 4 万面再展开，速度快很多。
- **对你有什么影响**：Hunyuan3D 贴图生成更快，默认开启网格简化。

### 15. tokenizer perf optimization
- **代码层面**：`serving_base.py` 的 `handle_request` 用 `loop.run_in_executor` 把 `_convert_to_internal_request`（含 jinja 渲染和 tokenization）扔到线程池；测试文件加 `encoding_started` 事件同步。
- **新手概念课堂**：asyncio 事件循环像"单线程服务员"，tokenizer 像"后厨切菜"。服务员亲自切菜时没人接待新客人，扔给后厨就能继续接待。
- **对你有什么影响**：长 prompt 场景下，服务不再被 tokenization 阻塞，并发能力提升。

### 16. [diffusion] fix: avoid redundant LoRA scaling while preserving exact numerics
- **代码层面**：`linear.py` 新增 `_scale_lora_delta` 方法，在 lossless 模式下把 `strength * runtime_scale` 合并成一次乘法；测试参数化 device 和更多 scale 组合。
- **新手概念课堂**：LoRA 是"小补丁"，缩放系数是"补丁强度"。`x * 1.0` 是精确的，但 `x * 0.7 * 0.6` 和 `x * 0.42` 可能有舍入差异。
- **对你有什么影响**：LoRA 推理在保持数值精确的前提下更快。

### 17. [diffusion] fix: keep MiniMax-H3 VAE fused norms off ROCm
- **代码层面**：`attention.py` 和 `vae_vit.py` 加 `current_platform.is_cuda()` 检查；测试从 `torch.cuda.is_available()` 改成 `current_platform.is_cuda()`。
- **新手概念课堂**：ROCm 上 `torch.cuda.is_available()` 也返回 True（因为 PyTorch 把 HIP 映射成 CUDA API），但 Triton 内核在 ROCm 上会输出 NaN。
- **对你有什么影响**：MiniMax-H3 VAE 在 AMD GPU 上不再输出 NaN，回退到未融合路径。

### 18. [diffusion] comfyui: run MiniMax H3, FastH3 and VDN in SGLD integrated mode
- **代码层面**：`generator.py` 新增 `_spawn_without_launcher_main` 上下文管理器（解决 ComfyUI spawn 时 `__main__` 冲突）、`_check_h3_single_file`（校验 H3 checkpoint 选项）、`_H3_MODEL_IDS` 映射。
- **新手概念课堂**：多进程 spawn 会重新导入 `__main__`，ComfyUI 的 `main.py` 会干扰。临时删掉 `__file__` 和 `__spec__` 就能避免。
- **对你有什么影响**：ComfyUI 用户现在可以在 SGLD 集成模式下跑 MiniMax H3、FastH3、VDN 三种模型。

### 19. fix(openai): honor tool_choice=none in DeepSeek encoders
- **代码层面**：`serving_chat.py` 的 `_apply_jinja_template` 里，`request.tools` 改成 `request_tools = request.tools if tools else None`，其中 `tools` 已经考虑了 `tool_choice=none`。
- **新手概念课堂**：`tool_choice=none` 表示"不要调用工具"，但之前的代码仍然把工具定义塞进 prompt，浪费 token 还可能误导模型。
- **对你有什么影响**：DeepSeek 系列模型在 `tool_choice=none` 时不再把工具定义写进 prompt，省 token 且行为更符合 OpenAI 规范。

### 20. Reuse Hugging Face LFM2-MoE configuration
- **代码层面**：`lfm2_moe.py` 从自定义 `PretrainedConfig` 子类改成继承 `transformers.Lfm2MoeConfig`，删掉约 100 行重复定义。
- **新手概念课堂**：HF transformers 是模型配置的"标准库"。以前 HF 没发布 LFM2-MoE 配置，SGLang 只能自己写；现在 HF 发布了，直接继承即可。
- **对你有什么影响**：LFM2-MoE 模型在 SGLang 上的配置更标准，减少因配置不一致导致的加载失败。

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)