# 🚀 vLLM & SGLang 每日更新 — 2026-11-04

> 自动生成于 2026-11-04 07:58 UTC | AI 解读: ✅ 含代码解读

## vllm
# vLLM 今日 Commit 深度解读

## 💡 今日关键词

- **关键词：Rust 前端（Rust Frontend）**
  - **一句话解释**：vLLM 正在把 Python 写的 HTTP 前端（路由、请求校验、响应序列化）逐步用 Rust 重写，追求更低延迟和更高吞吐。
  - **为什么社区在关注**：今天 20 个 commit 里有 **8 个** 都是 `[Rust Frontend]` 开头，集中在"和 Python 行为对齐"——布尔参数拼写、空 tools 数组、省略 null 字段、501 状态码等。这说明 Rust 前端已经从"能跑"进入"逐字节对齐 Python 语义"的打磨期，是 vLLM 走向生产级的关键一步。
  - **对新手意味着什么**：如果你会 Rust，这是一个非常好的切入点——大量小而清晰、带完整测试的 PR，非常适合第一次贡献开源。如果你只学 Python，也要理解"前端/引擎分离"的架构，未来调优会经常碰到。

- **关键词：CUDA Graph 精细化（Prefill/Decode 分离）**
  - **一句话解释**：CUDA Graph 是把一串 GPU 操作"录下来"再一次性重放，省掉 CPU 逐个下发 kernel 的开销。今天把它拆成了"prefill 按 token 数"和"decode 按请求数"两套独立配置。
  - **为什么社区在关注**：这是核心性能路径的重构，直接关系到启动时间、显存占用和 decode 吞吐。分离后可以只为 prefill 实例或 decode 实例捕获各自需要的图，避免浪费。
  - **对新手意味着什么**：想深入 LLM 推理性能优化，CUDA Graph 是必修课。理解"为什么按 batch size 捕获固定形状"是理解 vLLM 性能调优的钥匙。

- **关键词：AMD/ROCm 与 XPU 异构适配**
  - **一句话解释**：vLLM 在 NVIDIA 之外，持续为 AMD（ROCm）和 Intel（XPU）补测试、修 kernel、加平台特定优化。
  - **为什么社区在关注**：Commit 5、6、12 都是平台适配。特别是 MiniMax-M3、Step5 这些新模型在 AMD 上的 CP（context parallel）门控优化，说明国产/新兴模型 + 非 NVIDIA 硬件正在成为真实部署场景。
  - **对新手意味着什么**：异构计算是确定趋势。学 Triton kernel 和平台差异（如 `BLOCK_DMODEL` 对齐、FP8 支持）会让你在硬件厂商和云厂商那里非常抢手。

- **关键词：KV Cache 卸载与分层（KV Offload / Tiering）**
  - **一句话解释**：把 KV cache 从 GPU 显存挪到 CPU/DRAM/磁盘，腾出显存服务更多请求，是长上下文和高并发场景的关键技术。
  - **为什么社区在关注**：Commit 2、10 都在完善 offload 的事件描述和驱逐（eviction）测试，说明这块从"能用"走向"正确性可验证"。
  - **对新手意味着什么**：KV cache 管理是推理系统的核心难点，理解"块哈希、前缀缓存、驱逐水位线"能让你读懂大量 vLLM 内部逻辑。

- **关键词：工具调用（Tool Calling）一致性**
  - **一句话解释**：让模型输出结构化的函数调用（JSON），前端负责解析、校验、渲染回对话历史。
  - **为什么社区在关注**：Commit 7、9、15、16 全在修 tool calling 的边界情况——空数组、未完成的标记、非对象参数、`tool_choice=none`。这是 Agent 应用的地基。
  - **对新手意味着什么**：想做 AI Agent 方向，tool calling 的解析/渲染链路必须烂熟于心，这些 commit 就是最好的教材。

---

## 🧪 重要知识点 & 动手实验

### 知识点 1：CUDA Graph 的"形状捕获"与 prefill/decode 分离（性能优化）

- **是什么**：GPU 执行 kernel 有 CPU 下发开销。CUDA Graph 把固定形状的一批操作录制成图，重放时零下发开销。但图是"形状固定"的，所以 vLLM 要预先捕获一批 batch size 的图。今天把它拆成两套：prefill 按 **token 数**捕获，decode 按 **请求数**捕获。
- **为什么重要**：decode 阶段每个请求只生成 1 个 token（或 spec decode 的 q 个），batch 里请求数决定形状；prefill 阶段一次处理很多 token，token 数决定形状。混在一起配置会导致要么图太大浪费显存，要么图不够覆盖实际 batch 而回退到 eager 模式变慢。不懂这个，你调 `max_num_seqs` 和 `cudagraph_capture_sizes` 时会一头雾水。
- **动手试试**：
  ```bash
  # 只捕获 decode 图，prefill 走 eager（适合纯 decode 场景调试）
  vllm serve meta-llama/Llama-3.2-1B \
    --compilation-config '{"decode_cudagraph_capture_request_sizes": [1,2,4,8,16]}'

  # 完全关闭 decode 图（disaggregated 部署里的 prefill 实例）
  vllm serve meta-llama/Llama-3.2-1B \
    --compilation-config '{"decode_cudagraph_capture_request_sizes": []}'
  ```
  然后看启动日志里 `Capturing cudagraphs for ...` 的行数变化。
- **预期结果**：设置 `[]` 后启动不再捕获 decode 图，启动更快、显存占用更低；设置具体列表后，日志显示按请求数捕获。你会理解"图的数量 × 形状 = 显存与启动时间的权衡"。

### 知识点 2：OpenAI API 兼容性——null 与"省略字段"的区别（API 设计）

- **是什么**：JSON 里 `"field": null` 和"字段根本不存在"是两回事。Python 的 vLLM 对某些字段（如 `prompt_tokens_details`、`annotations`）是**省略**而非置 null，因为 OpenAI 规范里这些字段是"非 nullable 的对象/数组"。
- **为什么重要**：客户端 SDK（尤其是严格类型校验的）遇到 `null` 会报错。Rust 前端要和 Python 逐字节对齐，否则同一个模型两个前端返回不同 JSON，用户迁移就炸。这是"兼容性"最容易被忽视的坑。
- **动手试试**：
  ```python
  import openai, json
  client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="x")
  resp = client.chat.completions.create(
      model="meta-llama/Llama-3.2-1B",
      messages=[{"role": "user", "content": "hi"}],
      max_tokens=5,
  )
  raw = resp.model_dump()
  print("prompt_tokens_details in usage:", "prompt_tokens_details" in raw["usage"])
  print("annotations in message:", "annotations" in raw["choices"][0]["message"])
  ```
- **预期结果**：两个都是 `False`（字段被省略）。如果你把后端换成旧版本或 Rust 前端未对齐的版本，可能变成 `True` 且值为 `None`——这就是 commit 20 要修的问题。

### 知识点 3：工具调用的解析状态机——"未完成标记"如何丢弃（Agent 基础）

- **是什么**：模型输出工具调用时是一段特殊标记（如 `<|tool_call>...`）。如果生成因为 `max_tokens` 或 stop string 中途截断，标记是不完整的。Python 的解析引擎会**丢弃**这段残骸，只保留之前的正常文本；Rust 前端今天补上了同样行为。
- **为什么重要**：如果不丢弃，用户会在聊天界面看到半截 `<|tool_call>call:write_file{...` 的乱码。这是 Agent 产品体验的底线。同时 `tool_choice=none` 时也要"解析但丢弃"工具调用，而不是让它漏进正文。
- **动手试试**：用 Python 侧复现（Rust 侧逻辑一致）：
  ```python
  from vllm.entrypoints.openai.tool_parsers import get_tool_parser
  # 模拟一段被截断的 gemma4 工具调用
  text = "Writing it. <|tool_call>call:write_file{content:<|\"|>hello world"
  # 用对应 parser 的 extract_tool_calls / 流式解析，观察 content 只保留 "Writing it. "
  ```
  或者更简单：起一个服务，`max_tokens` 设得很小，让模型输出工具调用被截断，观察返回的 `content` 是否干净。
- **预期结果**：`content` 里只有截断前的正常文本，没有半截标记；`tool_calls` 为空。你会理解"流式解析器必须处理 EOF 时的中间状态"。

### 知识点 4：KV Cache 前缀缓存与块哈希（分布式/存储）

- **是什么**：vLLM 把 KV cache 切成固定大小的 block，每个 block 用其内容的哈希标识。相同前缀的请求可以复用已算好的 block（prefix caching）。offload 时要把"存了哪些 block、token 是什么、lookahead 是什么"描述清楚，才能正确驱逐和恢复。
- **为什么重要**：Commit 10 修的是"store 事件用请求上下文描述完整信息"，Commit 2 测的是"DRAM 池驱逐时恢复仍然正确"。搞错会导致缓存命中错误、输出不一致——这是最难 debug 的一类 bug。
- **动手试试**：
  ```bash
  # 开启前缀缓存 + CPU offload，观察命中
  vllm serve Qwen/Qwen3-0.6B --enable-prefix-caching \
    --kv-transfer-config '{"kv_connector":"CPUOffloadingConnector","kv_role":"kv_both"}'
  ```
  然后连续发两个共享长前缀的请求，看日志里 `prefix cache hit` 的 token 数。
- **预期结果**：第二个请求的 `num_cached_tokens` 大于 0。把 offload 池容量调小（模拟驱逐），再发老请求，验证输出和首次一致——这就是 commit 2 的测试思路。

### 知识点 5：平台特定 Kernel 对齐——为什么 XPU 要 `BLOCK_DMODEL = max(16, ...)`（异构计算）

- **是什么**：Triton kernel 里 `BLOCK_DMODEL` 是 head dimension 的 tile 大小，取 2 的幂。Intel XPU 上要求最小 16，否则某些指令或内存对齐会出问题（commit 12）。
- **为什么重要**：同一个 Triton kernel 在不同硬件上约束不同。写跨平台 kernel 时，`next_power_of_2` 之后还要考虑硬件最小对齐。不懂这个，你在 XPU/AMD 上跑会得到错误结果或崩溃，而且极难定位。
- **动手试试**：
  ```python
  import triton
  # 观察不同 head_dim 的 tile 选择
  for d in [8, 16, 32, 64, 128, 192]:
      block = triton.next_power_of_2(d)
      print(f"head_dim={d:4d} -> BLOCK_DMODEL={block:4d}",
            "| XPU min16:", max(16, block))
  ```
- **预期结果**：`head_dim=8` 时原本是 8，XPU 上被抬到 16。你会直观看到"平台差异如何影响 kernel 参数"，并理解为什么这类改动要单独提 PR 加测试。

---

## 🔥 重要更新

1. **#63406 CUDA Graph 拆分 prefill/decode 捕获尺寸**：核心性能路径重构，`cudagraph_capture_sizes` 在 Model Runner V2 下被废弃，改为两套独立配置。影响所有追求极致吞吐的部署，值得仔细读文档。
2. **#65498 MoE combine 跳过 staging copy**：FlashInfer NVLink one-sided 路径去掉一次显存拷贝，MoE 推理的直接性能收益，大模型推理用户可感知。
3. **#65547 空 tools 数组按 OpenAI 语义接受**：`tools: []` 不再返回 400，而是等同没传 tools。大量客户端会传空数组，这个修复能减少真实报错。
4. **#64697 Rust 前端支持 `--skip-tokenizer-init`**：让 Rust 前端能跑"无 tokenizer"模式（如纯 token 输入场景），是 Rust 前端功能对齐的重要里程碑。

---

## 📋 逐条解读

### 1. [Perf][MoE] Skip the combine staging copy in flashinfer_nvlink_one_sided (#65498)
- **代码层面**：改的是 `flashinfer_nvlink_one_sided` 的 prepare/finalize 路径和对应测试。核心是让 expert 的输出**直接写进 combine workspace**，finalize 时传 `payload_in_workspace=True`，省掉一次中间 staging buffer 的拷贝。测试里用 `spy_combine` 断言这个标志确实为 True。
- **新手概念课堂**：MoE（混合专家）里，token 要"分发"到各专家算，再"合并"回来。合并时如果先把结果拷到临时 buffer 再拷到目标，就是两次搬运。就像快递：本来可以直接送到你家，非要先放中转站再送，多一趟。这个 PR 就是取消中转站。
- **对你有什么影响**：跑 MoE 大模型（如 DeepSeek、Mixtral 类）时吞吐略升、显存带宽压力略降。普通用户感知不明显，但大规模部署省下的带宽很可观。

### 2. [Test][UMBP] Cover end-to-end restore while the embedded DRAM pool is evicting (#64658)
- **代码层面**：在 `test_mori_token_e2e.py` 里把 fixture 参数化，新增一个 64MiB 小池子 + 高低水位线（0.75/0.5）的用例 `test_restore_under_pool_eviction`，验证"最新 prompt 精确恢复、最老 prompt 已被驱逐"。
- **新手概念课堂**：KV offload 池像冰箱，容量有限，满了要按水位线扔东西（LRU 之类）。测试就是验证：冰箱满了扔东西时，你刚放进去的还能拿出来，最早放的可能没了——而且拿出来内容不能错。
- **对你有什么影响**：使用 KV offload / prefix caching 时，驱逐场景下的正确性更有保障，不会出现"缓存命中但输出乱掉"的诡异 bug。

### 3. [Bugfix] Honor add_generation_prompt and continue_final_message in the DeepSeek V4.1 tokenizer (#62729)
- **代码层面**：改 `tests/parser/engine/test_deepseek_v41.py` 和 tokenizer 测试，新增"当前轮已闭合 reasoning 应被限定在本轮"的用例。核心是让 tokenizer 正确处理 `add_generation_prompt`（是否在末尾加"该助手说话了"的提示）和 `continue_final_message`（续写最后一条消息而非新开一轮）。
- **新手概念课堂**：聊天模板像"填空题模板"。`add_generation_prompt` 决定要不要在末尾留一个"助手："的空位让模型填；`continue_final_message` 决定是接着上一条助手消息写，还是新起一轮。搞错会导致模型"自言自语"或格式错乱。
- **对你有什么影响**：用 DeepSeek V4.1 做多轮对话或续写时，reasoning（思考）块的边界正确，不会把上一轮的思考误当本轮。

### 4. [Core][CUDA Graph] Separate prefill (token) and decode (request) CUDA graph capture sizes (#63406)
- **代码层面**：大改 `vllm/config/compilation.py`，新增 `prefill_cudagraph_capture_grid`、`decode_cudagraph_capture_grid`、`decode_cudagraph_capture_shapes` 等函数；`cudagraph_capture_sizes` / `max_cudagraph_capture_size` 在 Model Runner V2 下废弃。文档 `torch_compile.md` 详细说明新参数。测试新增几百行覆盖各种 spec decode 组合。
- **新手概念课堂**：见知识点 1。简单说：prefill 是"一次读很多字"，decode 是"一次吐一个字"。给它们配同一套"衣服尺码"（图尺寸）不合理，现在分开量体裁衣。
- **对你有什么影响**：启动更快、显存利用更合理；speculative decoding 用户要注意新参数名，旧的 `cudagraph_capture_sizes` 在 V2 下不再生效。

### 5. [ROCm][Perf][MiniMax-M3.1] Score DSpark 7 decode locally below 3 requests (#65432)
- **代码层面**：在 `indexer_nvfp4.py` 加 `CP_MIN_REQS = 3`，`_decode_cp` 里当 `query_len == 2*CP_QUERY_LEN`（DSpark 7）且 `num_decodes < 3` 时返回 None，走本地单头打分而非跨 rank 的 context parallel。新增 `test_decode_cp_gate` 参数化测试。
- **新手概念课堂**：CP（context parallel）把长上下文切到多卡算，但跨卡通信有固定开销（~13us/层）。请求很少时，通信开销比本地算还贵，不如本地算。就像只有 2 个包裹，叫跨城快递不如自己骑车送。
- **对你有什么影响**：AMD 上跑 MiniMax-M3 小批量 decode 更快；这是典型的"按 batch size 动态选路径"优化，值得学习。

### 6. [ROCm][Step5] Exercise the real CSA decode and engine paths in tests (#64836)
- **代码层面**：在 `test_amd_csa_paged.py` 新增 `test_decode_selects_and_attends_over_summaries`，覆盖 TP1 decode（bf16 KV）和 TP8 verify（fp8 KV）两种布局，验证 summary 打分、top-k 选择、gated attention 的完整链路。
- **新手概念课堂**：CSA（压缩稀疏注意力）先把长序列压成"摘要区域"，decode 时先给摘要打分选 top-k，再只对选中的做注意力。测试就是验证"选得对、算得对"。
- **对你有什么影响**：Step5 模型在 AMD 上的长上下文推理更可信；测试覆盖真实 engine 路径而非 mock，回归风险降低。

### 7. [Bugfix][Frontend] Accept an empty tools array like OpenAI (#65547)
- **代码层面**：改 `ChatCompletionRequest` 校验逻辑，`tools: []` 不再报 "must not be an empty array"，而是归一化为 `tools=None`、`tool_choice="none"`。但 `tool_choice` 指定了具体函数名时仍报错（因为没这个工具）。测试参数化覆盖 `None/none/auto/required` 四种 tool_choice。
- **新手概念课堂**：OpenAI 把 `tools: []` 当"没传 tools"。很多客户端（尤其自动生成的 SDK）会无条件带上空数组，旧 vLLM 直接 400，用户体验很差。
- **对你有什么影响**：用某些 SDK 或前端框架调 vLLM 时不再莫名 400；注意 Rust 前端暂未对齐（测试里 skip 了）。

### 8. [Rust Frontend] Support --skip-tokenizer-init (#64697)
- **代码层面**：`hf.rs` 里根据 `text_backend.has_tokenizer()` 决定用 `HfChatBackend` 还是新增的 `TokenizerlessChatBackend`。后者对所有 chat 请求返回错误（因为渲染/解析 chat 必须有 tokenizer）。新增测试验证 `skip_tokenizer_init=true` 时 chat 被拒、text 后端无 tokenizer。
- **新手概念课堂**：`--skip-tokenizer-init` 用于"输入已经是 token id"的场景（省去加载 tokenizer 的开销）。但 chat 模板渲染需要 tokenizer 把文本转 token，所以 chat 接口必须拒绝。
- **对你有什么影响**：Rust 前端现在能跑纯 token 输入的离线/基准场景；chat 用户不受影响。

### 9. [Rust Frontend] Drop incomplete tool-call markup at the end of output like Python (#65375)
- **代码层面**：`unified.rs` 的 `finish()` 里，遇到 `is_incomplete_tool_call()` 错误时不再 warn 并返回错误，而是清空 `open_call_index`、reset parser、保留之前的事件。测试从"recovers"改名为"drops"，并验证有前缀文本时前缀保留。
- **新手概念课堂**：见知识点 3。流式输出到一半被 `max_tokens` 掐断，解析器处于"半个工具调用"的中间态，必须优雅收尾。
- **对你有什么影响**：Rust 前端流式输出遇到截断时，不会把半截标记当正文吐给用户，和 Python 行为一致。

### 10. [KV Offload] Describe complete stores from request context (#63696)
- **代码层面**：`tracker.record_store` 现在可以从 `ReqContext`（请求上下文快照）重建完整的 `BlockStored` 事件（block_hashes、token_ids、lookahead_token_ids），即使原 `Request` 对象已被 GC。测试用 `weakref` 验证 request 释放后仍能正确产出事件。
- **新手概念课堂**：offload 事件要告诉下游"我存了哪些块、内容是什么"。如果只存了引用，请求结束对象没了就描述不全。改成从上下文快照取，像"出门前把要买的东西列清单"，而不是"到超市再回忆"。
- **对你有什么影响**：KV offload + 事件上报（如给外部缓存系统）时，驱逐/恢复描述更完整，减少缓存不一致。

### 11. [Model] Canary: require language and test translation (#65445)
- **代码层面**：`canary.py` 里强制要求 `language` 参数（Canary 无法自动检测语言），translation 还要求 `to_language` 且只能英译/译英。文档和测试同步更新，新增 `test_canary_translation` 验证合法请求成功、非法请求 400 并给出明确错误。
- **新手概念课堂**：Whisper 能自动检测语言，Canary 不能——就像有人能听懂多国语言，有人必须先告诉他说哪国话。API 要在入口就拦住，而不是返回错误翻译。
- **对你有什么影响**：用 `nvidia/canary-1b-v2` 做语音转写/翻译时，必须传 `language`；错误提示更清晰，不会静默返回乱码。

### 12. [XPU] fix case test_p2p_lifecycle_e2e.py and test_mm_prefix_lm.py (#65416)
- **代码层面**：测试里给 P2P 用例加 `block_size=16`（XPU 要求）；`triton_prefill_attention.py` 里 XPU 平台把 `BLOCK_DMODEL` 抬到至少 16。
- **新手概念课堂**：见知识点 5。不同硬件对 tile 大小、block size 有最小对齐要求，跨平台 kernel 要加平台分支。
- **对你有什么影响**：Intel XPU 上跑 P2P KV 传输和 prefill attention 不再出错。

### 13. [Rust Frontend] Stop binding --data-parallel-rank to VLLM_DP_RANK (#65369)
- **代码层面**：`data_parallel.rs` 里去掉 `#[arg(long, env = "VLLM_DP_RANK")]` 的 `env` 绑定，只保留 `--data-parallel-rank`。新增测试断言该参数不再读环境变量。
- **新手概念课堂**：原来 `vllm serve --data-parallel-rank` 会同时读环境变量，导致"我只想传 flag 却被环境变量覆盖"的困惑。Python 只在 `ParallelConfig` 的 fallback 里读环境变量（如离线 SPMD），语义不同。
- **对你有什么影响**：用 Rust 前端做 DP 部署时，flag 和环境变量不再互相干扰，行为更可预测。

### 14. [Rust Frontend] Accept Python underscore and dotted flag forms under vllm-rs serve (#65370)
- **代码层面**：CLI 解析层支持 `--served_model_name`（下划线）、`--kv_transfer_config.kv_role`（点号嵌套）、`--hf-overrides.rope_scaling.factor=4` 等 Python 风格写法，并正确合并到 JSON 参数。新增大量测试。
- **新手概念课堂**：Python 的 argparse 习惯用下划线和点号传嵌套配置，Rust 的 clap 默认用连字符。为了"用户从 Python 迁过来零改动"，Rust 前端要兼容两种写法。就像新键盘支持旧键位。
- **对你有什么影响**：从 Python vLLM 迁到 Rust 前端时，命令行脚本不用改。

### 15. [Rust Frontend] Coerce replayed tool-call arguments that are not an object (#65371)
- **代码层面**：新增 `tool_call_arguments_or_empty()`，把非对象 JSON（数组、数字、字符串、null、非法 JSON）统一转成空对象 `{}`。DeepSeek V3.2、GLM5 等 renderer 改用它。测试覆盖 7 种非法输入都渲染成空参数。
- **新手概念课堂**：回放历史对话时，用户可能手改过 tool call 参数成非法 JSON。与其报错，不如当空参数处理——就像填表时某栏写错，先当没填，别整张表作废。
- **对你有什么影响**：多轮 Agent 对话里，历史 tool call 参数不规范也不会让整个请求 500。

### 16. [Rust Frontend] Drop tool calls under tool_choice=none for split tool parsers (#65377)
- **代码层面**：`resolve_tool_parser` 改为：即使 `tool_parsing_enabled=false`（`tool_choice=none`），只要配置了 parser 就仍解析，但解析出的调用被 `max_tool_calls` 丢弃，避免标记漏进 content。引入 `ResolvedToolContext`。无 parser 且解析关闭时返回 None 而非报错。
- **新手概念课堂**：`tool_choice=none` 意思是"别调工具"。但如果模型硬输出了工具标记，不解析就会当正文显示。正确做法是"解析出来然后扔掉"，而不是"不解析"。
- **对你有什么影响**：设 `tool_choice=none` 时，输出更干净，不会出现工具标记乱码。

### 17. [Rust Frontend] Accept FastAPI boolean spellings on dev route query parameters (#65372)
- **代码层面**：新增 `deserialize_query_bool`，让 query 参数接受 `True/False/1/0/true/false` 等多种拼写（FastAPI/Pydantic 的行为）。应用到 `/reset_prefix_cache`、`/pause` 等 dev 路由。
- **新手概念课堂**：FastAPI 的 bool 解析很宽松，`?reset_running_requests=True` 能工作。Rust 的 serde 默认只认 `true/false`。为了兼容，要自定义反序列化器。
- **对你有什么影响**：用 curl/脚本调 dev 路由时，`True` 大小写不再报错。

### 18. [Rust][Benchmark] Map --output-len to the sharegpt and hf datasets (#64946)
- **代码层面**：`config.rs` 里 `sharegpt_output_len: args.output_len.or(args.sharegpt_output_len)`，hf 同理。新增测试验证 `--output-len` 能映射到两种数据集，且 0 值报错。
- **新手概念课堂**：benchmark 有通用 `--output-len` 和数据集专用 `--sharegpt-output-len`。用户只想设一个通用值，不该被迫记每个数据集的专用 flag。
- **对你有什么影响**：跑 Rust benchmark 时 `--output-len 128` 对 sharegpt/hf 数据集直接生效。

### 19. [Rust Frontend] Return 501 for /generative_scoring and the opt-in Cohere v2 chat routes (#65376)
- **代码层面**：重构路由构建，引入 `RouteGates` 结构体统一管理 dev_mode / lora / scale_out / cohere_api 四个开关。未实现的 `/generative_scoring` 和 Cohere v2 chat 路由返回 501（Not Implemented）而非 404，语义更准确。
- **新手概念课堂**：404 是"没这个接口"，501 是"接口存在但没实现"。客户端能据此区分"我拼错了"还是"服务端还不支持"。同时把散落的 bool 参数收进结构体，代码更清晰。
- **对你有什么影响**：用 Cohere SDK 调 Rust 前端时得到明确的 501，而不是困惑的 404。

### 20. [Rust Frontend] Omit unset usage prompt_tokens_details and message annotations like Python (#65374)
- **代码层面**：给 `Usage.prompt_tokens_details` 和 `ChatCompletionMessage.annotations` 加 `#[serde(skip_serializing_if = "Option::is_none")]`，未设置时不序列化（而非输出 null）。测试从"断言为 null"改为"断言字段不存在"。
- **新手概念课堂**：见知识点 2。OpenAI 规范里这两个字段是"对象/数组，非 nullable"，输出 null 会让严格客户端报错。Python 端省略，Rust 对齐。
- **对你有什么影响**：Rust 前端返回的 JSON 和 Python 逐字节一致，客户端无需为两个前端写不同解析逻辑。

## sglang
## 💡 今日关键词

- **FP8 / MXFP8 量化内核**：用 8 位浮点数存储权重和激活，省显存、提吞吐；MXFP8 是带块级共享指数（UE8M0）的 FP8 变体。
  - **为什么社区在关注**：今天有 3 个 commit（#45828、#47853、#43786）都在打磨 FP8/MXFP8 内核，说明低精度推理已经从“能跑”进入“抠细节”阶段——小 batch 的 GEMV、融合量化、多后端切换。
  - **对新手意味着什么**：如果你在学 CUDA/Triton 或量化，现在是切入“量化内核优化”的好窗口，这类岗位需求正在爆发。

- **异构硬件适配（XPU / NPU / Blackwell / Arm64）**：同一套推理框架要跑在 Intel GPU、昇腾 NPU、NVIDIA 新卡、Arm CPU 上。
  - **为什么社区在关注**：今天有 XPU 依赖管理、NPU 命名规范、HALOps Blackwell 后端、Arm64 测试修复等多条 commit，说明 sglang 正在从“NVIDIA 专用”变成“全平台”。
  - **对新手意味着什么**：懂一种非 NVIDIA 平台（尤其昇腾 NPU 或 Intel XPU）会是差异化优势。

- **投机解码（Speculative Decoding）**：用小模型“猜”几个 token，大模型一次验证，加速生成。
  - **为什么社区在关注**：今天有 4 条 spec 相关 commit（TP 同步默认值、模拟接受长度、DSpark 修复、测试注册），说明投机解码是当前推理加速的主战场之一。
  - **对新手意味着什么**：这是算法+工程结合最紧的方向，适合想深入推理系统的同学。

- **PD 分离与 KV 传输**：把 Prefill 和 Decode 拆到不同机器，中间传 KV Cache。
  - **为什么社区在关注**：今天有 NIXL 线程绑卡、KV checksum 拒绝、HiSparse 资源释放等多条修复，说明 PD 分离已进入“稳定性打磨”阶段。
  - **对新手意味着什么**：分布式推理的资源生命周期管理是高频 bug 来源，值得专门学习。

- **调度器统一化重构**：把散落各处的“中止请求”逻辑收拢到一个入口。
  - **为什么社区在关注**：`RequestRetirer` 的引入说明代码库在长大，需要治理“重复逻辑”这种技术债。
  - **对新手意味着什么**：学会读重构类 commit，是理解大型项目架构演进的最佳方式。

---

## 🧪 重要知识点 & 动手实验

- **知识点 1：FP8 量化的“块级缩放”（Blockwise FP8 Quantization）**
  - **是什么**：不整行共用一个缩放因子，而是每 128 个元素（一个 block）算一个缩放因子，精度损失更小。
  - **为什么重要**：整行缩放会被一个极大值“带偏”，导致其他小数全变成 0；块级缩放是 FP8 能用于生产的核心技巧。不懂这个，你调 FP8 模型时精度掉了都不知道为什么。
  - **动手试试**：
    ```python
    import torch

    def blockwise_quant_fp8(x, block=128):
        # x: [M, K] fp32
        M, K = x.shape
        xb = x.view(M, K // block, block)
        amax = xb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        scale = amax / 448.0          # FP8 E4M3 最大可表示值
        q = (xb / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        return q, scale

    x = torch.randn(4, 512)
    x[0, 0] = 1000.0                  # 制造一个极端值
    q, s = blockwise_quant_fp8(x)
    deq = q.float() * s
    print("整行缩放 max 误差:", (x - deq).abs().max().item())
    print("非极端块误差:", (x[1:] - deq[1:]).abs().max().item())
    ```
  - **预期结果**：极端值所在块的误差较大，但其他块误差很小——这就是块级缩放的价值。对比“整行一个 scale”的版本，你会发现后者误差爆炸。

- **知识点 2：投机解码的“接受长度模拟”**
  - **是什么**：用一个可配置的分布（比如固定值、几何分布）模拟“草稿模型猜对几个 token”，用于压测和调优，而不需要真跑草稿模型。
  - **为什么重要**：调投机解码参数时，如果每次都跑真实草稿模型，实验成本极高；模拟器让你能在几秒内扫遍参数空间。不懂这个，你会浪费大量 GPU 时间。
  - **动手试试**：
    ```python
    import random

    def sample_acc_len(mean_len: float, block_size: int, method="geometric"):
        if method == "fixed":
            return min(int(mean_len), block_size)
        # 几何分布：P(k) = (1-p)^(k-1) * p，期望 = 1/p
        p = 1.0 / mean_len
        k = 1
        while random.random() > p and k < block_size:
            k += 1
        return k

    for mean in [1, 2, 4]:
        lens = [sample_acc_len(mean, 8) for _ in range(10000)]
        print(f"目标均值 {mean}, 实测均值 {sum(lens)/len(lens):.2f}")
    ```
  - **预期结果**：实测均值接近目标均值，且被 block_size 截断。commit #48298 修的正是“每次从服务器种子抽样”而非用全局 `random`，保证可复现。

- **知识点 3：CUDA 线程与设备的绑定（Device Binding）**
  - **是什么**：多卡机器上，每个线程默认可能操作 device 0；必须显式 `torch.cuda.set_device(gpu_id)` 才能让该线程的 CUDA 调用落在正确卡上。
  - **为什么重要**：commit #47255 修的就是“NIXL 传输线程没绑卡，结果跑到 device 0”——在多卡 PD 分离场景下会造成数据错乱或性能塌方。这是分布式推理最经典的坑之一。
  - **动手试试**：
    ```python
    import threading, torch

    def worker(gpu_id, results):
        # 错误做法：不 set_device，默认 device 0
        # 正确做法：
        torch.cuda.set_device(gpu_id)
        results[gpu_id] = torch.cuda.current_device()

    results = {}
    threads = [threading.Thread(target=worker, args=(i, results)) for i in range(torch.cuda.device_count())]
    [t.start() for t in threads]
    [t.join() for t in threads]
    print(results)   # 期望 {0:0, 1:1, ...}；若全为 0 说明没绑卡
    ```
  - **预期结果**：每个线程报告自己的 device id。把 `set_device` 注释掉再跑，你会看到所有线程都返回 0——这就是 bug 现场。

- **知识点 4：上下文管理器的“状态恢复”（try/finally）**
  - **是什么**：用 `@contextlib.contextmanager` 写的上下文管理器，必须在 `finally` 里恢复被修改的全局状态，否则异常退出时会“污染”进程。
  - **为什么重要**：commit #48028 修的就是 `set_batch_invariant_mode` 退出时没恢复 `torch.bmm`，导致后续所有 bmm 都变慢。这类 bug 极其隐蔽——只在异常路径触发，测试很难覆盖。
  - **动手试试**：
    ```python
    import contextlib, torch

    _original_bmm = torch.bmm

    @contextlib.contextmanager
    def bad_mode():
        torch.bmm = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("patched"))
        yield
        # 缺少 finally：异常时不会恢复

    @contextlib.contextmanager
    def good_mode():
        global _original_bmm
        old = torch.bmm
        torch.bmm = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("patched"))
        try:
            yield
        finally:
            torch.bmm = old

    try:
        with good_mode():
            raise ValueError("boom")
    except ValueError:
        pass
    print("good_mode 后 bmm 是否恢复:", torch.bmm is _original_bmm)  # True
    ```
  - **预期结果**：`good_mode` 恢复成功；把 `good_mode` 换成 `bad_mode` 再测，`torch.bmm` 就永久被污染了。

- **知识点 5：Triton 内核中手写指针 vs `make_block_ptr`**
  - **是什么**：`tl.make_block_ptr` 是 Triton 的便捷 API，但某些版本/后端有性能或兼容性问题；手写 `base + offset` 加 `mask` 更可控。
  - **为什么重要**：commit #47944 把 FLA 注意力内核从 `make_block_ptr` 迁到手动指针，说明“便捷 API 不一定最好”。理解底层寻址是写高性能内核的必修课。
  - **动手试试**：
    ```python
    import torch, triton, triton.language as tl

    @triton.jit
    def copy_manual(src, dst, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        v = tl.load(src + offs, mask=mask, other=0.0)
        tl.store(dst + offs, v, mask=mask)

    x = torch.arange(1000, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    copy_manual[(triton.cdiv(1000, 256),)](x, y, 1000, BLOCK=256)
    print("拷贝正确:", torch.equal(x, y))
    ```
  - **预期结果**：输出 `True`。把 `mask` 去掉再跑（N 不是 256 的整数倍），你会看到越界写或结果错误——这就是为什么边界 mask 必须写。

---

## 🔥 重要更新

1. **#45828 FP8 Hopper 小 M GEMV + 融合 KDA 输出量化**：decode 阶段 batch 很小，cuBLAS 会走 split-K 再 reduce，开销大；这个 commit 手写专用 GEMV 内核，是 decode 延迟优化的典型手段，对线上 P99 延迟有直接影响。
2. **#47853 HALOps 作为 Blackwell MXFP8 GEMM 可选后端**：新增一个第三方 GEMM 后端，且用 Docker 多阶段构建“下载失败就跳过”，说明框架开始为“多后端可插拔”做基础设施，影响未来所有 MXFP8 用户。
3. **#48283 调度器统一 `RequestRetirer` 入口**：把所有“中止排队请求”的路径收拢到一个类，是典型的架构治理重构，减少未来 bug 面，值得读 diff 学重构手法。
4. **#48028 修复 `set_batch_invariant_mode` 退出不恢复 `torch.bmm`**：一个 `finally` 缺失导致进程级性能污染，是“上下文管理器必须 try/finally”的教科书案例。

---

## 📋 逐条解读

### 1. [Perf] fp8: Hopper small-M GEMV, fused KDA output quant, CUDA grouped top-k, partial-block tails (#45828)
- **代码层面**：新增 `hopper_bf16_tiny_gemv.cuh` 等内核文件；新增两个环境变量 `SGLANG_ENABLE_FP8_BLOCK_GEMV` 和 `SGLANG_ENABLE_FUSED_KDA_OUTPUT_QUANT`，默认关闭。核心是：当 M 很小（decode 阶段）时，绕开 cuBLAS，走手写的“一行一个 block”的 GEMV 内核。
- **新手概念课堂**：cuBLAS 像“通用快递”，大件小件都送，但小件也要走完整分拣流程（split-K + reduce）。手写 GEMV 像“同城闪送”，专为小包裹优化，省掉分拣。
- **对你有什么影响**：开启这两个环境变量后，decode 阶段延迟可能下降；默认关闭意味着你需要自己压测验证收益。

### 2. [Fix] test: Import sgl_kernel in the Arm64 MoE test (#48223)
- **代码层面**：在 `test/registered/cpu/arm64/test_moe.py` 顶部加一行 `import sgl_kernel`。
- **新手概念课堂**：`sgl_kernel` 是一个 C++/CUDA 扩展包，import 它才会把自定义算子注册到 `torch.ops`。不 import，测试里 `torch.ops.sgl_kernel.xxx` 就找不到——像“没插电源就开灯”。
- **对你有什么影响**：纯测试修复，普通用户无感知；但如果你写测试，记住“用自定义算子前必须先 import 注册包”。

### 3. [Feat] halops: Add HALOps as an opt-in MXFP8 GEMM backend for Blackwell and support mxfp8 megamoe (#47853)
- **代码层面**：`docker/Dockerfile` 新增 `halops_wheel` 构建阶段，从 PaddlePaddle 的索引下载预编译 wheel，装失败就跳过；新增 `--fp8-gemm-backend=halops` 选项。
- **新手概念课堂**：GEMM 后端像“发动机”，同一辆车可以换不同发动机。HALOps 是第三方提供的 MXFP8 发动机，专为 Blackwell 调优。
- **对你有什么影响**：Blackwell 用户可尝试 `--fp8-gemm-backend=halops` 对比性能；Docker 构建不会因缺 wheel 而失败，更健壮。

### 4. [Chore] [XPU] deps: Pin sglang-kernel-xpu by version using uv (#46519)
- **代码层面**：CI 和 `xpu.Dockerfile` 从 `pip install` 改为 `uv pip install`；`pyproject_xpu.toml` 里 `sglang-kernel-xpu` 从“git 直接引用”改为“版本号 + git 覆盖”双模式。
- **新手概念课堂**：`uv` 是 Rust 写的 pip 替代品，快 10-100 倍。把依赖从“git 分支”改成“版本号”，像从“跟着 master 走”变成“锁版本”，更可复现。
- **对你有什么影响**：XPU 用户构建更快、更稳定；如果你自己写 Dockerfile，可以学这种“版本锁 + 可选 git 覆盖”的写法。

### 5. [Chore] [NPU] common: Update references from A5 to arch35 or Ascend 950 across multiple modules (#41504)
- **代码层面**：把代码和文档里所有 “A5” 字样改成 “arch35” 或 “Ascend 950”，涉及注意力后端、调度参数、环境变量注释等多处。
- **新手概念课堂**：这是“命名规范化”——内部代号 A5 对用户不友好，改成产品名 Ascend 950 更清晰。像把“项目代号凤凰”改成“产品名 XX 手机”。
- **对你有什么影响**：昇腾用户看日志/文档时不会再被“A5”搞晕；纯命名变更，无功能影响。

### 6. [Perf] DeepSeek-V4: Fuse inverse RoPE into wo_a FP8 quantization (#43786)
- **代码层面**：`fp8_wo_a_group_major_quant.cuh` 里抽出 `QuantizeWoAGroup` 模板函数，把“逆 RoPE”和“FP8 量化”两个步骤融合进一个内核，减少一次显存读写。
- **新手概念课堂**：融合内核像“洗切炒一体锅”——本来要洗、切、炒三个锅，现在一个锅搞定，省了两次装盘（显存读写）。显存带宽是推理瓶颈，融合是核心优化手段。
- **对你有什么影响**：DeepSeek-V4 用户推理更快；学习者可研究“如何把多个 elementwise 操作融合成一个 kernel”。

### 7. [Fix] moe: Match padded-row indices to token-count dtype (#48109)
- **代码层面**：新增 `_padded_row_mask` 辅助函数，确保 `torch.arange` 生成的索引 dtype 与 `num_token_non_padded` 一致，替换多处重复代码。
- **新手概念课堂**：dtype 不匹配像“拿米尺量英寸”——数值对但单位错，比较结果就错。GPU 上 int32 vs int64 比较可能静默出错。
- **对你有什么影响**：MoE 在 padding 场景下更稳定；写代码时注意“比较两个 tensor 前先对齐 dtype”。

### 8. [Refactor] scheduler: Retire every queued abort through one RequestRetirer entry (#48283)
- **代码层面**：新增 `RequestRetirer` 类，把 `decode.py`、`dllm/mixin/scheduler.py` 等多处“中止请求”逻辑统一到 `request_retirer.abort_queued(req)` 一个入口。
- **新手概念课堂**：像把散落在各房间的“总开关”集中到配电箱——以前每个房间自己拉闸，现在统一管理，减少漏拉、错拉。
- **对你有什么影响**：中止请求的行为更一致，减少“某些路径忘了释放资源”的 bug；重构类 commit 是学习架构设计的好材料。

### 9. [Fix] disaggregation: Reject the KV checksum with DeepSeek-V4 and on a HiSparse decode (#48292)
- **代码层面**：在 `hisparse_hook.py` 和 `pd_disaggregation_hook.py` 里新增校验：HiSparse decode + KV checksum、DeepSeek-V4 + KV checksum 这两种组合直接报错拒绝启动。
- **新手概念课堂**：这是“快速失败”（fail-fast）——与其运行时校验失败产生脏数据，不如启动时就拒绝。像“安全带没系就不让发车”。
- **对你有什么影响**：如果你同时开这几个选项，会看到明确的报错信息而不是诡异 bug；这是配置校验的好范式。

### 10. [Perf] spec: Default `SGLANG_SPEC_TP_SYNC` to `init` (#48111)
- **代码层面**：环境变量默认值从 `"all"` 改为 `"init"`，即只在启动时的内存探测阶段做 TP 同步，每步推理不再广播；日志从 warning 降为 info。
- **新手概念课堂**：TP（张量并行）多卡之间每步都同步像“每走一步都开会对齐”——安全但慢。改成只在启动时对齐，假设各卡计算一致，性能提升明显。
- **对你有什么影响**：投机解码默认更快；如果遇到 TP 卡间结果不一致或 hang，按注释设 `SGLANG_SPEC_TP_SYNC=all` 并报 bug。

### 11. [Fix] spec: Draw `SGLANG_SIMULATE_ACC_LEN` once per step from the server seed (#48298)
- **代码层面**：删除 `simulated_bonus_sample` 里用全局 `random.getrandbits` 的版本，改为传入 `generator`（来自服务器种子），保证可复现；同时清理了重复代码。
- **新手概念课堂**：全局 `random` 像“公共骰子”，谁都能摇，结果不可复现；传入 `generator` 像“每人一个带编号的骰子”，可追溯、可复现。
- **对你有什么影响**：压测结果可复现；写涉及随机的代码时，优先用显式 `torch.Generator` 而非全局 `random`。

### 12. [Fix] test: Import sgl_kernel in the Xeon thread-binding test (#48282)
- **代码层面**：`test/registered/cpu/test_binding.py` 加一行 `import sgl_kernel`。
- **新手概念课堂**：同 commit 2，自定义算子需先 import 注册包才能用。
- **对你有什么影响**：测试修复，无功能影响。

### 13. [Fix] batch-invariant: restore torch.bmm when set_batch_invariant_mode exits (#48028)
- **代码层面**：`set_batch_invariant_mode` 上下文管理器改为 `try/finally`，保存并恢复 `torch.bmm` 和 `_original_torch_bmm`，确保异常退出也能还原。
- **新手概念课堂**：`finally` 像“出门必关灯”——不管你是正常出门还是被赶出门（异常），灯都会关。没有它，异常路径就会“灯一直亮着”（全局状态被污染）。
- **对你有什么影响**：用 batch-invariant 模式后不会再拖慢后续所有 `bmm`；这是写上下文管理器的必学范式。

### 14. [Test] [NPU] deepseek_v4: Replace 8K perf case with 128K 1P1D case and add L2 host hit-rate checks (#48160)
- **代码层面**：nightly 测试用例从 8K 输入换成 128K 输入（90% 前缀共享），新增 `min_hit_rate_pcts = {"host": 0}` 断言 L2 主机缓存命中率 > 0；新增 `run_bench_serving` 调用。
- **新手概念课堂**：L2 host hit rate 是“请求的 KV 有多少能从主机内存缓存命中”——长上下文场景下这是关键指标。像图书馆“有多少书在分馆就能借到，不用跑总馆”。
- **对你有什么影响**：昇腾用户的长上下文性能有更严格的回归测试保障；写性能测试时可学“用共享前缀构造缓存命中场景”的技巧。

### 15. [Perf] dsv41: Keep MXFP4 Engram tables packed and decode E2M1 in the gather (#47935)
- **代码层面**：`engram_gather.py` 新增 `_dequant_rows` Triton 函数，直接在 gather 时解码 MXFP4（E2M1）的打包字节（一个字节两个值），不再预先解包成 fp8；新增 `unpack_mxfp4_rows` 作为 torch 参考实现。
- **新手概念课堂**：MXFP4 一个字节塞两个 4-bit 数，像“压缩饼干”——吃的时候再泡开（gather 时解码），省存储和带宽。E2M1 是 4-bit 浮点：1 符号 + 2 指数 + 1 尾数。
- **对你有什么影响**：DeepSeek-V4.1 的 engram 表显存占用减半；学习“位运算解码 + 融合 gather”是低精度推理的核心技能。

### 16. [Fix] disaggregation: Bind NIXL transfer workers to the rank's device (#47255)
- **代码层面**：NIXL 连接器创建传输线程时，用 `run_on_device(self.kv_args.gpu_id, self.transfer_worker, ...)` 包装，确保线程在该 rank 的 GPU 上执行 CUDA 调用。
- **新手概念课堂**：多卡机器上，线程默认操作 device 0；不绑卡就像“让 8 号窗口的柜员去 1 号窗口办事”——数据错乱。`run_on_device` 是“给线程发一张指定窗口的工牌”。
- **对你有什么影响**：多卡 PD 分离传输更稳定；这是分布式推理必知的“线程-设备绑定”坑。

### 17. [Fix] dspark: Expand [N, H] staged embeds in the DeepSeek-V4 DSpark draft forward (#48244)
- **代码层面**：`deepseek_v4_dspark.py` 抽出 `_expand_hc` 方法，并在 `_forward_impl` 里判断：如果传入的 `input_embeds` 是 2 维 `[N, H]`，先扩展成 `[N, hc_mult, H]` 再继续。
- **新手概念课堂**：`hc_mult` 是“每个 token 复制几份”的维度（类似多头）。2 维输入像“单人照”，3 维是“连拍”——下游代码期望连拍，收到单人照就崩。
- **对你有什么影响**：DSpark 草稿模型在外部传入 embedding 时不再报 shape 错误；写模型 forward 时注意“兼容多种输入维度”。

### 18. [Fix] attention: Migrate FLA Triton kernels off tl.make_block_ptr (#47944)
- **代码层面**：`chunk_delta_h.py` 等多个 FLA 内核，把 `tl.make_block_ptr` + `boundary_check` 改成手动计算 `base + offset` + 显式 `mask`，并统一用 int64 偏移防溢出。
- **新手概念课堂**：`make_block_ptr` 像“自动挡”，方便但某些路况（Triton 版本/后端）会出问题；手动指针是“手动挡”，繁琐但完全可控。这次是“为了兼容性和性能换手动挡”。
- **对你有什么影响**：FLA 注意力在更多 Triton 版本上能跑；学习 Triton 时，理解手动寻址是进阶必修。

### 19. [Test] dspark: Register the simulated-bonus test for CUDA CI (#48278)
- **代码层面**：`test_dspark_simulated_bonus.py` 新增 `register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")`。
- **新手概念课堂**：CI 注册像“给测试报名参加体检”——不注册就不会在 CI 里跑，回归 bug 就漏网。
- **对你有什么影响**：DSpark 模拟采样的回归有 CI 保障；写新测试记得注册到 CI。

### 20. [Fix] hisparse: Release HiSparse rows on the KV checksum mismatch and chunked abort exits (#43911)
- **代码层面**：`decode.py` 的 `_handle_kv_checksum_mismatch` 和 `scheduler.py` 的 `process_pending_chunked_abort` 里，在释放 KV 之前先调 `hisparse_coordinator.request_finished(req)` 释放 HiSparse 的 host 页；新增单元测试验证调用顺序。
- **新手概念课堂**：HiSparse 把 KV 放在 host 内存池，释放顺序错了就像“先退房再还钥匙”——房卡系统里还挂着记录，资源泄漏。必须先通知 HiSparse 释放，再释放 KV。
- **对你有什么影响**：HiSparse 用户长时间运行不再泄漏 host 内存；这是“资源释放顺序”的经典案例，值得学习。

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)