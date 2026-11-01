# 🚀 vLLM & SGLang 每日更新 — 2026-11-01

> 自动生成于 2026-11-01 08:05 UTC | AI 解读: ✅ 含代码解读

## vllm
## 💡 今日关键词

- **关键词：KV Offload（KV 缓存卸载）**
  - **一句话解释**：把 GPU 显存里放不下的历史对话缓存（KV cache）搬到 CPU 内存或硬盘上，需要时再搬回来。
  - **为什么社区在关注**：今天 3 个 commit 都与 KV offload 相关（#63900、#63695、#64607 的 recipe），说明"长上下文 + 多轮 agent"场景下显存不够用是普遍痛点，社区正在把 offload 从"能跑"打磨成"好用、可观测、可自定义策略"。
  - **对新手意味着什么**：如果你在做推理服务，KV cache 管理是必须掌握的一课；理解 offload 机制相当于理解了"内存分级"这一套通用工程思想。

- **关键词：MoE 路由与 padding 跳过融合（Fused MoE routing）**
  - **一句话解释**：MoE 模型每个 token 只激活少数专家，但 batch 里 padding 的空 token 也会被送进路由计算，今天把它在 kernel 里直接跳过。
  - **为什么社区在关注**：MoE 是当前大模型主流架构（DeepSeek、Qwen3.5 等都是 MoE），kernel 级别的每一分性能都被认真对待，说明性能优化已经深入到 CUDA 内核的指令级。
  - **对新手意味着什么**：想进推理优化方向，CUDA kernel 阅读和改写能力是硬通货，今天这个 commit 是很好的学习材料。

- **关键词：Pipeline Parallel 跨阶段状态传递（PP）**
  - **一句话解释**：模型被切到多张卡上按层流水线执行，某些中间结果（如共享 top-k 索引）需要跨卡传递，今天修了 DeepSeek-V3.2 的这个 bug。
  - **为什么社区在关注**：大模型单卡放不下，PP 是必备分布式策略；这类"跨 stage 状态"bug 隐蔽且影响正确性，说明分布式推理的复杂度在上升。
  - **对新手意味着什么**：分布式推理调试是稀缺技能，理解 PP 的数据流能让你在部署大模型时少踩坑。

- **关键词：Batch Invariance（批不变性 / 确定性推理）**
  - **一句话解释**：同一个请求，不管它和谁拼在一个 batch 里，输出都应该完全一致——今天通过"钉住"特定 kernel 实现。
  - **为什么社区在关注**：agent、评测、RL 训练都要求可复现性，batch invariance 正从"nice to have"变成硬需求。
  - **对新手意味着什么**：如果你做评测或训练，理解"为什么同样的输入输出会变"是基本功。

- **关键词：AMD ROCm recipe 大量补充**
  - **一句话解释**：今天有 3 个 commit（#64792、#64805、#64804）在给 MI355X/MI350X 补性能配方（recipe）。
  - **为什么社区在关注**：AMD GPU 正在成为 NVIDIA 之外的真实选项，recipe 是"开箱即用"的关键，说明异构计算生态在快速成熟。
  - **对新手意味着什么**：如果你只有 AMD 卡或者想降低成本，现在正是入局 ROCm 的好时机，资料比以前丰富多了。

## 🧪 重要知识点 & 动手实验

- **知识点 1：KV Cache 卸载与事件通知（KV Offload）**
  - **是什么**：GPU 显存有限，把不常用的 KV cache 块挪到 CPU 内存（甚至磁盘），需要时再 load 回来；offload manager 会发出 "stored/evicted" 事件供上层观察。
  - **为什么重要**：多轮对话、长文档问答时，prefix 复用能极大降低 TTFT；不懂 offload 就会遇到"显存 OOM"或"prefix cache 命中率低"的问题。
  - **动手试试**：
    ```bash
    # 启动一个带 KV offload 的服务（需要 vLLM 支持 native backend 的版本）
    vllm serve Qwen/Qwen2.5-7B-Instruct \
      --kv-offloading-backend native \
      --kv-offloading-size 16 \
      --enable-prefix-caching
    ```
    然后连续发两次相同前缀的请求，观察第二次的 `usage.prompt_tokens_details.cached_tokens` 是否 > 0。
  - **预期结果**：第二次请求的 cached_tokens 明显大于 0，TTFT 显著下降；如果 offload 生效，日志里能看到 store/load 事件。

- **知识点 2：MoE 路由中的 padding 跳过（Fused Grouped TopK）**
  - **是什么**：MoE 的 grouped_topk kernel 原本对 padding token 也计算 top-k，今天直接在 kernel 里判断 `is_padding`，把 padding 的 topk_indices 写成 -1、topk_values 写成 0。
  - **为什么重要**：batch 里 padding 比例可能很高（尤其变长请求），跳过它们能省掉大量无用计算；同时避免 padding token 被错误路由到专家。
  - **动手试试**：阅读 `csrc/moe/grouped_topk_kernels.cu` 中 `grouped_topk_fused_kernel` 的 diff，重点看这段：
    ```cpp
    topk_indices[i] = is_padding && is_padding[token_id] ? -1 : static_cast<IdxT>(i);
    topk_values[i]  = is_padding && is_padding[token_id] ? 0.0f : 1.0f / topk_i32;
    ```
    然后自己写一个 Python 小实验模拟：给定 `is_padding=[True, False, True]`，验证 padding 位置的输出被置零。
  - **预期结果**：你能解释"为什么 padding token 的 topk_values 要置 0"——因为后续 expert 加权求和时 0 权重等于跳过。

- **知识点 3：Pipeline Parallel 下的跨 stage 状态传递**
  - **是什么**：DeepSeek-V3.2 的稀疏 indexer 有一个"共享 top-k"结果，某些层复用前面层的 top-k；当模型被 PP 切分后，这个状态必须从上一个 stage 传到下一个 stage。
  - **为什么重要**：不传的话，下游 stage 会用自己的 top-k，导致结果错误（不是崩溃，而是静默错误——最难查的那种）。
  - **动手试试**：
    ```bash
    # 跑新增的测试，看它如何验证跨 stage 行为
    pytest tests/models/deepseek_v32/test_pp_topk.py -v
    ```
    阅读 `pp_splits_layer_sharing` 的测试用例表，理解"split-on-producer" vs "split-on-consumer" 的区别。
  - **预期结果**：测试通过，你能说清楚"为什么 PP 切分点落在 producer 和 consumer 之间时行为不同"。

- **知识点 4：Batch Invariance（批不变性）**
  - **是什么**：同一个 prompt，无论 batch size 是 1 还是 64，输出 token 应该逐位相同。但 GPU kernel 的 reduction 顺序随 batch 变化，会导致浮点结果微小差异。
  - **为什么重要**：评测复现、RL 训练、agent 调试都要求确定性；今天通过 `VLLM_BATCH_INVARIANT=1` 钉住特定 kernel 来实现。
  - **动手试试**：
    ```bash
    # 开启 batch invariant 跑一个确定性测试
    VLLM_BATCH_INVARIANT=1 pytest tests/v1/determinism/test_cutlass_batch_invariance.py -v
    ```
    然后关掉环境变量再跑一次，对比结果。
  - **预期结果**：开启时同一输入多次运行输出完全一致；关闭时可能出现微小差异（尤其在 fp8 路径上）。

- **知识点 5：Rust 前端计算 prompt block hash**
  - **是什么**：vLLM 有 Rust 写的前端（用于高性能路由），今天把"prompt 分块哈希"（用于 prefix cache 查找）搬到 Rust 里算，并加了跨语言一致性测试向量。
  - **为什么重要**：prefix cache 的 key 必须 Python 和 Rust 两端算出来完全一样，否则缓存永远不命中；用共享 JSON 向量测试是保证一致性的经典做法。
  - **动手试试**：
    ```bash
    # 查看测试向量文件结构
    cat rust/src/engine-core-client/src/tests/prompt_block_hash_vectors.json | head -50
    # 跑 Python 侧的向量测试
    pytest tests/v1/core/test_prompt_block_hash_vectors.py -v
    ```
  - **预期结果**：Python 和 Rust 用同一份向量算出相同 hash；你能理解"为什么跨语言一致性要用 golden vector 测试"。

## 🔥 重要更新

- **#64955 [Perf][MoE] Fuse padding skipping in grouped and Muse prepared routing**：MoE 路由 kernel 里直接跳过 padding token，减少无效计算，对所有 MoE 模型的吞吐都有正面影响。
- **#63842 [Perf][Frontend] Compute prompt block hashes in the Rust frontend**：把 prefix cache 的哈希计算搬到 Rust 前端，降低 Python 侧 CPU 开销，对高 QPS 服务影响明显。
- **#62192 [Kernel] Batch invariance: pin FlashInfer FP8 bmm and W4A16 NVFP4 kernels**：让 `VLLM_BATCH_INVARIANT=1` 真正覆盖 FP8/NVFP4 路径，是确定性推理的重要补全。
- **#64338 [Bugfix] Flashinfer attention warmup deadlock under PP>1**：修复 PP>1 时 attention warmup 的死锁，影响所有用 FlashInfer + PP 的部署。

## 📋 逐条解读

### 1. [KV Offload] Remove unused touch from CachePolicy and OffloadingManager (#63900)
- **代码层面**：删掉了 `CachePolicy` 接口里的 `touch` 方法，改为在 `on_request_finished` 钩子里统一更新 recency；同步清理了测试里所有 `touch.assert_not_called()` 断言，并更新了文档。
- **新手概念课堂**：想象一个图书馆书架管理策略，以前每本书被翻一次就要"摸一下"记录热度（touch），现在改成"读者离开时统一结算"，接口更简洁、职责更清晰。
- **对你有什么影响**：如果你写过自定义 KV offload 策略，需要把 `touch` 逻辑迁移到 `on_request_finished`；普通用户无感知。

### 2. [KV Offload] Carry request context on raw stored events (#63695)
- **代码层面**：`StoredEvent` 等事件对象新增 `req_context` 字段，CPU/FS/Obj tier 在 complete_store 时把请求上下文带上；测试里 `_store_job_keys` 重命名为 `_store_jobs`。
- **新手概念课堂**：以前快递员只告诉你"包裹已入库"，现在还会告诉你"是哪个客户下的单"，方便追踪和审计。
- **对你有什么影响**：监听 KV 事件的监控/调度系统现在能知道事件属于哪个请求，可以做更精细的缓存策略。

### 3. [Perf][MoE] Fuse padding skipping in grouped and Muse prepared routing (#64955)
- **代码层面**：`grouped_topk_fused_kernel` 和 `grouped_topk_fused_small_expert_count_kernel` 新增 `is_padding` 参数；padding token 的 topk_indices 置 -1、topk_values 置 0，避免无效路由。CMake 测试范围也加入了 `muse_spark` 相关目录。
- **新手概念课堂**：考试时空白试卷不用批改，直接打 0 分——省下批改时间。
- **对你有什么影响**：变长请求多的场景（agent、chat）吞吐会有提升，尤其是 padding 比例高的 batch。

### 4. [Bugfix][DSv3.2] Carry the shared indexer top-k across pipeline stages (#59961)
- **代码层面**：新增 `pp_splits_layer_sharing` 工具函数判断 PP 切分点是否落在 top-k 共享链上；模型 forward 时把共享 top-k 通过 `IntermediateTensors` 跨 stage 传递；新增 `tests/models/deepseek_v32/test_pp_topk.py`。
- **新手概念课堂**：流水线上游工位算出的"零件清单"必须传给下游，否则下游会自己瞎猜一份，装出来的产品就不对。
- **对你有什么影响**：用 DeepSeek-V3.2 + PP>1 部署时，输出正确性得到修复（之前是静默错误）。

### 5. [Recipe] Qwen3.5-397B-A17B: add the B200 NVFP4 TP2 agentic preset (#64607)
- **代码层面**：在 `recipes/models/Qwen/Qwen3.5-397B-A17B.yaml` 新增 `b200_tp2_kv_offload` preset，包含 TP2 + EP、MTP 3 draft tokens、FP8 KV cache、160 GiB host KV offload、`--max-num-seqs 48` 等参数和性能数据。
- **新手概念课堂**：像菜谱一样，告诉你"这道菜（这个模型）在 B200 上怎么做最好吃"，直接抄参数就能跑出好性能。
- **对你有什么影响**：有 B200 的用户可以直接复用这套配置，省去调参时间。

### 6. [Doc] MiniMax-M3.1-Flash: update the GB300 AgentX presets (#64910)
- **代码层面**：更新 GB300 聚合/分离式 AgentX preset 的 InferenceX 引用版本、性能数据表（新增 E2E latency p90 列）、`DYN_TOKENIZER=fastokens` 环境变量说明、Mooncake pool size 从 160GB 调到 180GB 等。
- **新手概念课堂**：recipe 就是"官方调好的性能作业"，每次更新都是别人替你踩过坑后的最优解。
- **对你有什么影响**：用 GB300 部署 MiniMax-M3.1-Flash 的用户能拿到更新的性能基线和更稳的配置。

### 7. [Doc] MiniMax-M3.1-Flash: update the GB200 AgentX presets (#64909)
- **代码层面**：类似 #64910，针对 GB200 更新 InferenceX 引用、性能表、`--gpu-memory-utilization` 从 0.88 调到 0.92、删除 `VLLM_MAX_ASYNC_KV_ADMITS_PER_STEP` 等。
- **新手概念课堂**：同款菜在不同厨房（GB200 vs GB300）参数不同，recipe 帮你分别调好。
- **对你有什么影响**：GB200 用户获得更新后的性能基线和更优配置。

### 8. [Docs][ROCm] Mistral Large 4 recipe: adds TP4 and lowest-latency presets (#64792)
- **代码层面**：新增 `mi350x_mxfp4_tp4`、`mi350x_mxfp4_tp4_specialized`、`mi355x_tp4` 等 preset；给 ROCm variant 加 `docker_serve_command: "vllm serve"`；新增 `decode_backend: latency_specialized` 配置项。
- **新手概念课堂**：AMD 卡上也有"一卡多副本"和"低延迟特化 kernel"两种玩法，recipe 帮你选。
- **对你有什么影响**：用 MI350X/MI355X 跑 Mistral Large 4 的用户多了 TP4 和低延迟选项。

### 9. [Docs][ROCm] Add MiniMax-M3.1-Flash MI355X TP4 AgentX preset (#64805)
- **代码层面**：新增 `mi355x_agentx` preset，按 concurrency 分 c1/c2/c4/c8（mono decode）、c16/c32（DBO）、c64（DBO + KV offload）多档配置，每档有独立的 `--max-num-seqs`、KV cache 类型、speculative tokens 数。
- **新手概念课堂**：像游戏画质预设——低并发用高质量 kernel，高并发切到吞吐优先模式。
- **对你有什么影响**：MI355X 用户按并发档位直接抄配置即可。

### 10. [Docs][ROCm] Add DeepSeek-V4.1-Flash MI355X TP8 concurrency-1 AgentX preset (#64804)
- **代码层面**：新增 `mi355x_agentx_tp8` preset，TP8 + mono decode + Engram 常驻表，`--max-num-seqs 1`、`--max-num-batched-tokens 32768`；runbook 说明 mono decode 在 TP8 上只适合 concurrency 1。
- **新手概念课堂**：TP8 单请求延迟最低，但并发一高就亏——recipe 明确告诉你"这是 c1 专用配置"。
- **对你有什么影响**：单请求低延迟场景（如交互式代码补全）可以用这套配置。

### 11. [Bugfix][Tool Parser] Preserve whitespace in HYV4 argument names (#61542)
- **代码层面**：HYV4 tool parser 在解析 `<arg_key>` 时不再 strip 空白；新增 `tests/tool_parsers/test_hy_v4_tool_parser.py` 覆盖带/不带 suffix 的 tokenizer 以及流式 chunking。
- **新手概念课堂**：函数参数名里可能有空格（比如 `"user name"`），以前被误删导致调用失败，现在保留原样。
- **对你有什么影响**：用 HYV4 模型做 function calling 且参数名含空格时不再出错。

### 12. [Bugfix][Spec Decode] Honor draft moe/attention backend overrides for MRV2 standalone draft model (#64751)
- **代码层面**：`StandaloneARSpeculator` 创建 draft model 时调用 `SpeculativeConfig.apply_draft_overrides`，把 `moe_backend`、`attention_backend` 等 override 应用到 draft 模型的 `kernel_config`/`attention_config`。
- **新手概念课堂**：草稿模型（draft）和主模型可以有不同的"引擎调校"，以前草稿模型忽略了这些设置，现在会正确继承。
- **对你有什么影响**：用 MRV2 + 独立 draft model 且指定了 backend override 的用户，配置现在真正生效。

### 13. [Perf][Frontend] Compute prompt block hashes in the Rust frontend (#63842)
- **代码层面**：Rust `engine-core-client` 新增 `prompt_block_hasher` 模块（引入 `sha2` 依赖），在客户端侧计算 prompt block hash；新增 `prompt_block_hash_vectors.json` 共享测试向量，Python 侧 `test_prompt_block_hash_vectors.py` 用它验证一致性；Dockerfile.cpu 加入该向量文件。
- **新手概念课堂**：以前"查字典"（prefix cache 查找）前要先在 Python 里算好页码，现在 Rust 前端直接算好页码再发给引擎，省一次往返。
- **对你有什么影响**：高 QPS 服务的 CPU 开销降低，prefix cache 命中路径更快。

### 14. [Perf][Kernel] Single-launch fused scoring+top-k kernel for sparse-indexer prefill, without materializing logits (#60618)
- **代码层面**：新增 `csrc/fused_mqa_topk/` 目录（`fused_mqa_topk.cu`、`candidates.cuh` 等），针对 SM100 家族实现单次 launch 完成 scoring + top-k，不物化 logits；CMake 增加 `FUSED_MQA_TOPK_ARCHS` 编译分支，依赖 DeepGEMM headers。
- **新手概念课堂**：以前要先把所有候选分数写进显存再排序，现在边算边选，省掉一大块显存和一次 kernel launch。
- **对你有什么影响**：DeepSeek 类稀疏 attention 模型的 prefill 阶段更快、显存占用更低（仅 SM100 + CUDA 12.9+）。

### 15. [ModelRunner V2] Reduce peak memory of chunked prompt logprobs (#64841)
- **代码层面**：`compute_prompt_logprobs_with_chunking` 从"每 chunk append 到 list 再 cat"改为预分配输出张量并原地写入；单 chunk 时直接返回 int32 的 token_ids；新增测试验证 chunked 与 unchunked 结果一致。
- **新手概念课堂**：以前是"每收集一批就复制到新箱子"，现在是"一开始就准备好大箱子往里填"，峰值内存减半。
- **对你有什么影响**：请求 prompt logprobs 且 prompt 很长时，显存峰值下降，不容易 OOM。

### 16. [Bugfix][Frontend] Report prompt_tokens_details in batched chat completions (#64757)
- **代码层面**：`OpenAIServingChatBatch` 新增 `enable_prompt_tokens_details` 属性，batch 响应里对 `cached_tokens`、`created_cache_tokens`、`multimodal_tokens` 求和；新增集成测试验证 batch 值是单条的两倍；文档补充说明。
- **新手概念课堂**：批量接口以前不告诉你"省了多少缓存"，现在会汇总告诉你。
- **对你有什么影响**：用 `/v1/chat/completions/batch` 且开启 `--enable-prompt-tokens-details` 时，能看到 batch 级别的缓存命中统计。

### 17. Skip redundant weight tensor clones for loaders that yield owned tensors (#63174)
- **代码层面**：升级 `runai-model-streamer` 到 0.17.1（该版本 yield 的 tensor 是 owned 的），权重加载路径跳过不必要的 `.clone()`；同步更新 Dockerfile、`versions.json`、各 requirements 文件。
- **新手概念课堂**：以前快递到手还要再复印一份才敢用，现在快递直接给你原件，省一次复印。
- **对你有什么影响**：模型加载更快、峰值内存更低（尤其大模型从 S3/GCS 加载时）。

### 18. [Bugfix][Model] Fix CohereASR FP16 attention bias dtype (#54399)
- **代码层面**：删除 `cohere_asr.py` 里 `load_weights` 中把 `pos_bias` buffer dtype 转成 loaded_weight dtype 的逻辑（这会把 fp16 模型转成 fp32）；新增测试验证 fp16 下加载后 dtype 保持 fp16 且 attention 输出 fp16。
- **新手概念课堂**：以前加载权重时"好心"把 buffer 转成权重类型，结果把 fp16 模型偷偷变成 fp32，显存翻倍；现在保持运行时 dtype。
- **对你有什么影响**：CohereASR 用 fp16 部署时显存不再异常翻倍，attention 计算 dtype 正确。

### 19. [Kernel] Batch invariance: pin FlashInfer FP8 bmm and W4A16 NVFP4 kernels (#62192)
- **代码层面**：`init_nvfp4_linear_kernel` 在 `VLLM_BATCH_INVARIANT=1` 且 `use_a16=True` 时选择 `HummingNvFp4LinearKernel`（batch-invariant 的 CUTLASS kernel 只支持 W4A4）；FlashInfer FP8 bmm 在 batch invariant 模式下固定 backend；新增对应测试。
- **新手概念课堂**：以前"批不变"模式只管一部分 kernel，FP8/NVFP4 路径仍会随 batch 变化；现在把这些路径也钉死。
- **对你有什么影响**：开启 `VLLM_BATCH_INVARIANT=1` 后，FP8/NVFP4 推理的输出可复现性得到保证。

### 20. [Bugfix] Flashinfer attention warmup deadlock under PP>1 (#64338)
- **代码层面**：`attention_warmup` 在 PP>1 时通过 `cpu_all_reduce_` 协调各 rank 的 warmup 顺序，避免不同 stage 的 warmup 互相等待；新增 `test_attention_warmup.py` 用多线程模拟 PP group 验证不死锁。
- **新手概念课堂**：流水线上多个工位同时"预热机器"，如果大家都等对方先动就会卡死；现在约定好顺序。
- **对你有什么影响**：用 FlashInfer + PP>1 部署时启动不再卡死。

## sglang
## 💡 今日关键词

- **关键词：PD 分离（Prefill/Decode Disaggregation）**
  - 一句话解释：把"读 prompt 算 KV"和"逐 token 生成"拆到不同机器上跑，像餐厅把备菜和后厨出菜分开。
  - 为什么社区在关注：今天 20 个 commit 里有 3 个（#47725、#47704、#46373）都在修 PD 分离的时序和连接管理 bug，说明这套架构已经从"能跑"进入"跑稳"的深水区。
  - 对新手意味着什么：分布式推理的调试能力（race condition、连接池、生命周期管理）是稀缺技能，比只会调模型参数值钱得多。

- **关键词：KV Cache 内存池与 torch.compile**
  - 一句话解释：KV 缓存是推理时存"历史对话记忆"的地方，它的内存布局直接影响显存占用和编译优化效果。
  - 为什么社区在关注：#47670、#47705、#40916 三个 commit 连续改动 memory_pool，说明大家在为"编译模式 + 新注意力结构"重新设计内存布局。
  - 对新手意味着什么：理解 tensor 的 stride/alias（是否共享同一块内存）是性能调优的基本功。

- **关键词：扩散模型（Diffusion）推理加速**
  - 一句话解释：文生图/视频模型生成慢，社区在用 FP8 量化、cuDNN SDPA、跳过无效计算等手段提速。
  - 为什么社区在关注：#47121（1.7x 加速）、#47586、#47566、#46871 全是 diffusion 性能/正确性修复，说明多模态生成正在成为 SGLang 的第二战场。
  - 对新手意味着什么：扩散模型推理优化是蓝海，比 LLM 推理卷得少。

- **关键词：数值溢出与 int64 安全**
  - 一句话解释：int32 最大约 21 亿，当"槽位号 × 步长"超过它就会翻转成负数，读到错误内存。
  - 为什么社区在关注：#47029 专门修了这个，说明长上下文/大 KV 池场景下，32 位索引已经不够用了。
  - 对新手意味着什么：写 kernel 时永远问自己"这个乘法会不会溢出"。

- **关键词：AMD/ROCm 异构适配**
  - 一句话解释：同一份代码要在 NVIDIA 和 AMD GPU 上都跑得快。
  - 为什么社区在关注：#40708 给 AMD 接了 aiter 的 topk_gating 内核，说明 AMD 不再是"能跑就行"，而是开始做专项优化。
  - 对新手意味着什么：懂 ROCm/aiter 的人少，是差异化竞争力。

---

## 🧪 重要知识点 & 动手实验

- **知识点 1：线程启动前的初始化顺序（Commit 1）**
  - **是什么**：多线程程序里，如果子线程可能先跑起来并访问还没赋值的成员变量，就会崩。正确做法是"先把东西准备好，再启动线程"。
  - **为什么重要**：真实场景中这类 bug 是间歇性的（race condition），测试环境跑一万次不挂，生产环境挂一次。不懂就会写出"偶发崩溃"的代码。
  - **动手试试**：
    ```python
    import threading, time

    class Bad:
        def __init__(self):
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()   # 先启动
            self.data = "ready"   # 后赋值 —— 可能来不及

        def run(self):
            time.sleep(0.001)
            print("Bad:", self.data)   # AttributeError!

    class Good:
        def __init__(self):
            self.data = "ready"   # 先赋值
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()   # 后启动

        def run(self):
            print("Good:", self.data)

    for _ in range(3):
        try: Bad()
        except AttributeError as e: print("崩了:", e)
        Good()
    ```
  - **预期结果**：`Bad` 类大概率抛 `AttributeError`，`Good` 类永远正常。你就理解了 commit 1 为什么要把 `self._loop = None` 提前。

- **知识点 2：字符串前缀匹配的陷阱（Commit 2）**
  - **是什么**：用 `startswith("10.0.0.1:8998")` 匹配连接 key 时，会把 `10.0.0.1:89981` 也误删。
  - **为什么重要**：这是经典的"边界匹配"bug。任何用字符串做 ID 的地方（缓存 key、文件路径、URL 路由）都可能踩。
  - **动手试试**：
    ```python
    pool = {
        "10.0.0.1:8998_0_0_0": "conn-a",
        "10.0.0.1:89981_0_0_0": "conn-b",   # 不同端口！
        "10.0.0.2:8998_0_0_0": "conn-c",
    }
    dead = "10.0.0.1:8998"

    bad  = [k for k in pool if k.startswith(dead)]
    good = [k for k in pool if k.startswith(f"{dead}_")]
    print("错误匹配:", bad)    # 把 89981 也删了
    print("正确匹配:", good)
    ```
  - **预期结果**：`bad` 包含 2 个 key（误伤 `89981`），`good` 只有 1 个。加分隔符 `_` 就解决了。

- **知识点 3：FP8 量化与 per-token/per-head scale（Commit 3）**
  - **是什么**：把 FP16 权重压成 8 位浮点（FP8），显存减半、算得更快；但范围小，需要给每个 token/每个 head 配一个"缩放系数"再还原。
  - **为什么重要**：这是当前推理加速最主流的手段（Wan-Animate 提速 1.7x 就靠它）。不懂 scale 的概念就看不懂任何量化代码。
  - **动手试试**（需要 PyTorch，CPU 即可跑通逻辑）：
    ```python
    import torch

    x = torch.randn(2, 4) * 100   # 模拟大范围数据
    scale = x.abs().amax(dim=-1, keepdim=True) / 448.0  # e4m3 最大 448
    x_q = (x / scale).to(torch.float8_e4m3fn)           # 量化
    x_deq = x_q.to(torch.float32) * scale               # 反量化
    print("原始:", x[0])
    print("还原:", x_deq[0])
    print("误差:", (x - x_deq).abs().max().item())
    ```
  - **预期结果**：误差在千分之几量级，说明"先除 scale、存 FP8、再乘 scale"能近似还原。这就是 commit 3 里 `_quantize_fp8_per_token_head` 做的事。

- **知识点 4：int32 溢出与 int64 索引（Commit 12）**
  - **是什么**：int32 上限 2,147,483,647。当 `slot * stride` 超过它，结果变成负数，kernel 读到错误地址。
  - **为什么重要**：长上下文、大 KV 池、大 batch 时这是真实事故。而且它不报错，只出错结果——最难查的 bug。
  - **动手试试**：
    ```python
    import numpy as np

    slot = np.int32(2**28)      # 2.68 亿
    stride = np.int32(8)        # 8 个头
    print("int32 结果:", slot * stride)          # 溢出成负数！
    print("int64 结果:", np.int64(slot) * stride) # 正确
    ```
  - **预期结果**：第一行输出负数（约 -2147483648 附近），第二行是 2147483648。commit 12 就是在 Triton kernel 里把 `offs_kv_loc` 强制 `.to(tl.int64)`。

- **知识点 5：torch.compile 下的内存别名问题（Commit 8）**
  - **是什么**：如果 K 和 V 共用一块显存（别名 alias），`torch.compile` 为了保证正确性，会在每步把整块显存复制一遍——性能暴跌。
  - **为什么重要**：这是"手动优化"和"编译器优化"打架的典型案例。很多人开了 compile 反而变慢，原因就在这里。
  - **动手试试**：
    ```python
    import torch

    # 别名：v 是 k 的后半段
    k = torch.zeros(1024)
    v = k[512:]
    print("共享存储?", k.data_ptr() == v.data_ptr() - 512*4)

    # 独立分配
    k2 = torch.zeros(512)
    v2 = torch.zeros(512)
    print("独立?", k2.data_ptr() != v2.data_ptr())
    ```
  - **预期结果**：第一组打印 `True`（共享），第二组打印 `True`（独立）。commit 8 就是在 `enable_torch_compile` 时强制让 K/V 走独立分配。

---

## 🔥 重要更新

1. **#46373（PD 分离时序修复）**：decode 端在"上一轮 forward 还没写完"时就把传输目标发给 prefill，prefill 用 RDMA 直接往里写，可能覆盖正在用的数据——这是会导致输出乱码的严重并发 bug，本次用 CUDA Event 等待解决。
2. **#47121（Wan-Animate-2 提速 1.7x）**：扩散视频生成里，把参考视频的 K/V 缓存用 FP8 存储并精确定位 FA3 注意力，是今天性能收益最大的改动。
3. **#47029（int32 溢出）**：修复 200 万 token 以上场景下注意力读错 KV 的问题，长上下文用户必看。
4. **#47670（torch.compile 下 K/V 分离分配）**：修复"开了 compile 反而更慢/更吃显存"的问题，对使用编译加速的用户影响直接。

---

## 📋 逐条解读

### 1. [Fix] disaggregation: Initialize the bootstrap server loop and runner before the server thread starts
- **代码层面**：改 `disaggregation/common/conn.py` 的 `CommonKVBootstrapServer.__init__`。原来线程 `self.thread.start()` 在 `self._loop`/`self._runner` 赋值之前就可能跑起来，`_run_server` 的 `finally` 里访问 `self._runner.cleanup()` 会 `AttributeError`。现在先把两个字段初始化为 `None`，清理时判空。测试新增了 `TestBootstrapServerClose`，用 mock 掉线程 start 来复现。
- **新手概念课堂**：像装修房子——你得先把水电接好（初始化成员），再让工人进场（启动线程）。顺序反了，工人一开灯就跳闸。
- **对你有什么影响**：普通用户感知不到，但服务启动/关闭时的偶发崩溃会消失。

### 2. [Fix] disaggregation: Match the whole bootstrap address when evicting a dead prefill's connections
- **代码层面**：`_handle_node_failure` 里把 `k.startswith(failed_addr)` 改成 `k.startswith(f"{failed_addr}_")`。因为连接池 key 格式是 `"ip:port_rank..."`，前缀匹配会把 `:89981` 误认成 `:8998`。
- **新手概念课堂**：像门牌号匹配——说"找 8 号楼"不能把"80 号楼"也算进去，得加个分隔符。
- **对你有什么影响**：一个 prefill 节点挂掉时，不会误伤其他节点的连接，PD 集群更稳定。

### 3. [Perf] diffusion: Wan-Animate-2 1.7x faster denoising via exact FA3 in-context attention, opt-in FP8 reference K/V
- **代码层面**：`wan_animate_2.py` 的 `build_reference_kv` 现在会把参考视频的 K/V 按 `_InContextLayout` 对齐 padding，并可选地用 `_quantize_fp8_per_token_head` 量化成 FP8 存起来（需要 sm89+）。新增环境变量 `SGLANG_DIFFUSION_WAN_ANIMATE_2_FP8_REFERENCE_KV`。
- **新手概念课堂**：参考视频的 K/V 就像"剧本参考页"，每生成一帧都要翻。以前用大字号（FP16）存，占地方；现在用小字号（FP8）+ 每页一个换算表，翻得更快。
- **对你有什么影响**：跑 Wan-Animate-2 视频生成时开这个开关，去噪阶段快 1.7 倍，显存也更省。

### 4. [Doc] skills: Update the CI skills for the tier layout and current pipeline
- **代码层面**：只改 `.agents/skills/` 下的 markdown 文档，补充"大规模审计 CI 测试"的方法论：先脚本盘点、再分发给便宜模型读、主 agent 复核、优先降级而非删除。
- **新手概念课堂**：像给新人写"如何审代码"的 SOP 手册——不是代码，但决定了团队效率。
- **对你有什么影响**：无直接影响，但如果你参与 SGLang 贡献，CI 测试的增删标准更清晰了。

### 5. [Fix] diffusion: Exit the worker after an unrecoverable device error
- **代码层面**：`gpu_worker.py` 新增 `_FatalDeviceError` 异常和 `_is_unrecoverable_device_error()` 判定函数。遇到 CUDA driver error、allocator 断言等非 OOM 错误时，不再尝试请求级恢复，直接退出 worker 让上层重启。OOM 仍按可恢复处理。
- **新手概念课堂**：像电脑蓝屏——有些错误（内存不够）关个程序就好，有些错误（驱动崩了）必须重启，硬撑只会让后面所有请求都出垃圾结果。
- **对你有什么影响**：GPU 出硬件级错误时，服务会明确挂掉并报错，而不是默默返回错误结果。

### 6. [Fix] multimodal: Keep shared-memory feature transport for multi-node pipeline parallelism
- **代码层面**：`transport/__init__.py` 的 `determine_tensor_transport_mode` 原来"多节点就退回 default"，现在新增 `_tokenizer_consumers_on_first_node()`：只要 tokenizer 的消费者（第一个 PP stage 的 TP 组）都在 node 0，且不用 Ray、dp_size=1、nnodes ≤ pp_size，就仍用 `cuda_ipc`。
- **新手概念课堂**：像快递——只要收件人都在同一栋楼，就能用楼内直送（共享内存）；以前一看"跨楼"就改走邮政，其实只是楼号多，人没散。
- **对你有什么影响**：多节点 PP 部署多模态模型时，特征传输走共享内存，延迟更低。

### 7. [Fix] diffusion: keep tied vocab tables on device under layerwise offload
- **代码层面**：`layerwise_offload.py` 的 `_host_resident_tables` 增加判断：如果某个权重被别的模块（如输出 head）共享引用，就不把它挪到 host。否则 tied embedding 被搬到 CPU，输出 head 在 GPU 上做 matmul 就崩。
- **新手概念课堂**：像两个人共用一把钥匙——你不能因为 A 暂时不用就把钥匙收走，B 还要开门呢。
- **对你有什么影响**：开了 layerwise offload 的扩散模型，tied 词表模型不再报错。

### 8. [Fix] mem_cache: Keep K/V in separate allocations under torch.compile
- **代码层面**：`memory_pool.py` 里原来"k_shape == v_shape 就合并成一块大 allocation"的优化，现在加了条件 `not get_flags().capture.enable_torch_compile`。因为 compile 下 KV store 会 mutate 两个张量，别名会导致 AOTAutograd 每步克隆整块 KV。
- **新手概念课堂**：像两个人共用一本笔记本——不编译时没问题；一旦让"自动整理员"（compile）介入，他怕两人写冲突，每次都要复印整本，反而更慢。
- **对你有什么影响**：开启 torch.compile 后不再出现"越编译越慢/越吃显存"。

### 9. [Refactor] memory-pool: Rename index key format to layout
- **代码层面**：`IndexKeyFormat` 枚举改名为 `IndexKeyLayout`，`DSA_FP8` 改为更描述性的 `PAGED_FP8_KEYS_WITH_SCALES`，字段 `key_format` → `key_layout`。纯重命名，无行为变化。
- **新手概念课堂**：像把变量名 `a` 改成 `user_count`——代码没变，但读起来不费劲了。
- **对你有什么影响**：无，但如果你写插件依赖旧字段名，需要跟着改。

### 10. [Feat] hicache: Add the fast_file local storage backend
- **代码层面**：新增 `fast_file` 分层缓存后端，保持和 `file` 相同的磁盘页格式，但加了直接 host-buffer I/O、并行读、原子发布、后台 LRU 淘汰。benchmark 脚本同步更新，去掉了内部 `meta_cache_store` 相关逻辑。
- **新手概念课堂**：`file` 后端像"单车道乡道"，`fast_file` 像"多车道高速 + 智能红绿灯"——路还是那条路（格式不变），但通行效率高得多。
- **对你有什么影响**：本地 NVMe 做 L3 缓存时吞吐明显提升，配置换成 `--hicache-storage-backend fast_file` 即可。

### 11. [Refactor] memory-pool: Build QSA indexer host pools from buffer info
- **代码层面**：`device_pool_info.py` 新增 `IndexKeyFormat.KEY_ROWS`（每行一个 key 的布局）和 `page_views()` 方法，`hybrid_pool_assembler.py` 改为从 buffer info 构建 host pool。为 QSA（量化稀疏注意力）索引器统一了 host 侧内存描述。
- **新手概念课堂**：像给仓库建"货架图纸"——以前每个仓管自己记货在哪，现在统一按图纸摆放，谁来取都一样。
- **对你有什么影响**：使用 QSA 索引器 + 分层缓存时，host 内存分配更规范，减少 OOM。

### 12. [Fix] attention: Compute extend attention KV offsets in int64
- **代码层面**：`extend_attention.py` 两个 Triton kernel（`_fwd_kernel`、`_fwd_kernel_unified`）里，`offs_kv_loc` 计算后加 `.to(tl.int64)`。因为调用方可能传 int32 的 `kv_indices`，`slot * stride` 在 200 万 token 以上会溢出。新增测试用 8.6GB 显存构造 2^32/128 个槽位复现。
- **新手概念课堂**：像计数器只有 3 位——数到 999 再加 1 就变 000。int32 数到 21 亿就"归零变负"，读到的 KV 全是错的。
- **对你有什么影响**：超长上下文（>200 万 token 等效 KV）推理不再输出乱码。

### 13. [Perf] [AMD] qwen3.5: Route 512-expert softmax MoE routing through aiter topk_gating
- **代码层面**：`topk.py` 新增 `_use_aiter_topk_gating_softmax()` 判定，在 gfx95、hidden=2560/8192、512 专家、topk=10、bf16、行数 ≤ 4096 等条件下，用 aiter 的 `topk_gating` 替代 `fused_topk`。新增环境变量 `SGLANG_ROCM_DISABLE_AITER_TOPK_GATING` 可回退。
- **新手概念课堂**：MoE 路由像"分诊台"——512 个科室里选 10 个。以前用通用分诊流程，现在 AMD 有专用分诊台（ASM 内核），人多时更快。
- **对你有什么影响**：在 AMD MI300/gfx95 上跑 Qwen3.5 系列，MoE 路由更快；遇到问题可设环境变量回退。

### 14. [Refactor] diffusion: normalize diffusers output layout in one place
- **代码层面**：`diffusers_pipeline.py` 新增 `_to_output_layout()`，把"通道优先"的张量统一转成 `(B,C,H,W)` 或 `(B,C,T,H,W)`，从 `_fix_output_shape` 里移除重复的 permute 逻辑。修复了单视频 `(T,C,H,W)` 被误当图像处理的 bug。
- **新手概念课堂**：像快递分拣——以前好几个工位各自判断"这是文件还是包裹"，现在统一在入口按规则分好，后面不用再猜。
- **对你有什么影响**：diffusers 后端输出视频/图像形状更一致，下游处理不再形状错乱。

### 15. [Feat] model: support Unlimited-OCR v2
- **代码层面**：新增 `unlimited_ocr_v2` 注意力后端、DASH 稀疏注意力、按页切窗 + KV 复用、`--uocr-doc-split` 参数、`unlimited-ocr` chat template；新增 cookbook 文档 `Unlimited-OCR-2.mdx`。
- **新手概念课堂**：像读一本 500 页的书——以前要一口气背下来（上下文爆炸），现在按页读、记住前页要点（KV 复用），一本一本处理。
- **对你有什么影响**：可以用 SGLang 部署 Unlimited-OCR-2 做整本书/多页文档 OCR，单请求处理整份文档。

### 16. [Perf] diffusion/minimax-h3: Skip attention for the packing's alignment rows in the Ulysses pipeline
- **代码层面**：`minimax_h3.py` 新增 `_minimax_h3_group_attend`，检测到 `cu_seqlens` 是 `[0, used, total]` 且尾部是纯 padding 时，只对前 `used` 行做 cuDNN SDPA varlen，尾部直接 `zero_()`，跳过无意义的注意力计算。
- **新手概念课堂**：像排队时多摆了几把空椅子——以前每把空椅子都要"服务"一遍，现在直接跳过，只服务真人。
- **对你有什么影响**：MiniMax-H3 视频生成在 Ulysses 并行下更快，尤其 padding 比例高时。

### 17. [Doc] cookbook: Add MiniMax-M3.1-Flash GB300 PD AgentX points
- **代码层面**：纯文档/配置更新，`minimax-m3.1.jsx` 新增 GB300 上的 disagg 6P1D 配置、HiCache 参数、DSpark 调度策略等 AgentX 基准点。
- **新手概念课堂**：像餐厅菜单加新菜——告诉你怎么点（启动命令）、用什么料（配置）。
- **对你有什么影响**：想在 GB300 上跑 MiniMax-M3.1-Flash PD 分离，有现成配方可抄。

### 18. [Perf] diffusion: keep GQA and MOSS-Image attention on cuDNN SDPA on Blackwell
- **代码层面**：`sdpa.py` 的 `_use_cudnn_sdpa` 放宽条件：sm_10x 上 Q/KV head 数不等（GQA）也允许走 cuDNN（因为 `CudnnSDPAImpl` 支持 `enable_gqa`），只要整除即可。`moss_image.py` 去掉 `default_attention_backend=FA` 的硬绑定，让 Blackwell 自动选 cuDNN SDPA。
- **新手概念课堂**：像导航——以前"不是单行道就不走高速"，现在"多车道也能上高速（只要车道数能整除）"，Blackwell 上走得更顺。
- **对你有什么影响**：Blackwell 上跑 GQA 模型和 MOSS-Image，注意力自动走更快的 cuDNN SDPA。

### 19. [Fix] rust-server: Validate gathered listener ranks and address families
- **代码层面**：`rust_server/server.py` 的 `gather_server_endpoints` 现在：① 记录 `bind_host`，如果绑 `0.0.0.0` 但 advertise 了 IPv6 地址，写入 `address_error`；② 收集完后统一校验所有 rank 的 `address_error` 并抛错；③ 校验 `dp_rank` 集合是否等于 `range(attn_dp_size)`，防止漏 rank 导致 gather 卡死。
- **新手概念课堂**：像开会点名——以前各报各的，有人报错地址也没人管；现在主持人统一核对名单和地址，错一个全员一起停，不会有人傻等。
- **对你有什么影响**：Rust server 多 DP rank 部署时，配置错误会立刻报清晰错误，而不是挂起或部分失败。

### 20. [Fix] disaggregation: Send decode transfer destinations to prefill only after the in-flight forward
- **代码层面**：`decode.py` 的 `_send_kv_metadata` 在 `enable_overlap` 且非 deferred publish 且目标不是 HOST 时，不再立刻 `send_metadata`，而是记录 `(ready_event, args, kwargs)` 到 `decode_req.pending_metadata`；新增 `publish_pending_metadata()` 在 forward 完成后通过 `all_reduce_flags(MIN)` 保证所有 rank 同步发布。`conn.py` 里 `chunk_staging_infos` 提前初始化为空列表。
- **新手概念课堂**：像交接班——上一班的人还在写黑板（forward 没写完），你不能让下一班的人（prefill）直接往黑板上写，否则字叠在一起。必须等上一班擦完（Event 触发）再交接。
- **对你有什么影响**：PD 分离 + overlap 模式下，不再出现 KV 被覆盖导致的输出错乱，是今天最关键的稳定性修复。

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)