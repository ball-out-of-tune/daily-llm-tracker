# 🚀 vLLM & SGLang 每日更新 — 2026-10-29

> 自动生成于 2026-10-29 07:38 UTC | AI 解读: ✅ 含代码解读

## vllm
## 💡 今日关键词

- **关键词：MXFP8 / 微缩放量化（Microscaling）**
  - 一句话解释：把权重和激活按 32 个元素一组共享一个指数缩放因子，压到 FP8 存储，矩阵乘时硬件直接"边乘边还原"。
  - 为什么社区在关注：今天有 3 个 commit（#63179、#63915、#64111 附带基准）都在做 MXFP8 的"融合量化"，说明低精度推理已从"能跑"进入"抠每一微秒"的阶段，而且 AMD CDNA4 的 `tl.dot_scaled` 成为主战场。
  - 对新手意味着什么：量化不再是"调个参数"，而是要和 kernel 的 tile 形状、K 维对齐、split-K 策略一起设计。学 Triton + 量化格式（E4M3/E8M0）是当下性价比最高的组合。

- **关键词：KV Cache 卸载 / 分层（Offloading & Tiering）**
  - 一句话解释：把 GPU 显存放不下的 KV 缓存挪到 CPU 内存、甚至远端 peer 的显存里，需要时再搬回来。
  - 为什么社区在关注：#52838 和 #64256 都是"卸载链路出错时别把整个引擎搞挂"的 bugfix，说明 KV 卸载已经从实验特性走向生产，故障处理成为刚需。
  - 对新手意味着什么：分布式存储/传输的"失败语义"设计（超时、abort ack、延迟释放）是稀缺技能，比单纯调 kernel 更少见。

- **关键词：异构后端适配（AMD ROCm / Intel XPU / Ascend）**
  - 今天至少 8 个 commit 与 AMD 相关（#63179、#63915、#64064、#61529、#62990、#52838…），还有 XPU（#64228）和 Ascend（#64013）。
  - 为什么社区在关注：vLLM 正在从"NVIDIA 优先"变成"多硬件一等公民"，每个后端都有自己的 tile 配置、attention 后端、量化路径。
  - 对新手意味着什么：会写"平台无关的 kernel 抽象层"（如 `_select_cfg`、`is_device_capability`）比只会 CUDA 更值钱。

- **关键词：投机解码（Speculative Decoding）的工程化**
  - #64356 把 draft logits 重构成 `DraftLogitsCache` 支持 batch 分片，#62990 让 DSpark 自适应验证在 ROCm DCP 上跑通。
  - 为什么社区在关注：投机解码从"论文 demo"进入"多卡、多后端、多 draft 策略"的复杂工程阶段。
  - 对新手意味着什么：理解 draft/target 模型的对齐、logits 缓存、验证策略，是进入推理加速核心圈的门票。

- **关键词：融合算子（Fusion）**
  - #61107（Qwen MoE 共享 MLP 的 act+quant 融合）、#63382（NVFP4 融合 scale 初始化 bug）、#63915（quant 融进 GEMM）。
  - 为什么社区在关注：单算子优化见顶，跨算子融合（activation → quant → GEMM）是剩余性能空间。
  - 对新手意味着什么：学会读 kernel 的"内存访问模式"和"launch 开销"，才能判断哪里值得融合。

---

## 🧪 重要知识点 & 动手实验

### 知识点 1：KV Cache 卸载的"失败语义"——超时后为什么不能立刻释放内存

- **是什么**：当 P2P（peer-to-peer）KV 加载超时时，本机不能马上把这块 CPU 内存还给空闲池，因为远端 peer 可能还在往里写。必须等对方发来 `AbortAckMsg` 确认后才释放（#64256）。
- **为什么重要**：如果提前释放，这块内存可能被下一个请求复用，而远端还在写旧数据 → 数据竞争 → 输出乱码。这是典型的"分布式系统里失败不是终点，是另一个状态"的坑。
- **动手试试**：
  ```bash
  # 跑这个 commit 新增的单元测试，观察 deferred release 计数
  cd vllm
  pytest -v -s tests/v1/kv_offload/tiering/p2p/test_manager.py::test_timed_out_load_holds_chunks_until_abort_ack
  ```
  然后打开 `vllm/v1/kv_offload/` 下相关文件，搜索 `num_deferred_release_chunks`，看它在超时路径和 abort ack 路径分别怎么增减。
- **预期结果**：测试通过，并且你能在代码里指出"超时 → 标记 deferred → 收到 ack → 真正 free"这条状态链。理解后你就能回答："为什么不能用一个简单的 `finally: free()`？"

### 知识点 2：MXFP8 的"融合量化"——把量化塞进 GEMM 的 K 循环

- **是什么**：小 batch（M ≤ 8 或 ≤ 16）时，单独跑一次 activation 量化 kernel 的 launch 开销比计算本身还大。于是把量化融进 GEMM kernel 内部（#63179、#63915），要求结果与"先量化再 GEMM"**逐位相同**。
- **为什么重要**：融合最容易引入数值差异，测试里专门写了 `torch.equal(out, expected)`（bit-exact），而不是 `assert_close`。不懂这个区别，你会写出"看起来对但线上偶发 NaN"的 kernel。
- **动手试试**：
  ```bash
  # 需要 CDNA4 (gfx95x) GPU；没有就只看代码
  pytest -v -s tests/kernels/quantization/test_rocm_mxfp8_linear.py -k "fused_act_quant_bitwise_eq_unfused"
  ```
  阅读 `vllm/model_executor/kernels/linear/mxfp8/rocm_native.py` 里的 `_select_fused_quant_cfg`：它按 `(M, N, K)` 决定是否走融合路径。试着列出"哪些形状会被拒绝融合"。
- **预期结果**：你能解释为什么"融合路径必须 bit-exact"，以及为什么 `x[0, :32] = 0.0`（全零 block）和 `x[1] *= 3e6`（超 fp16 范围）是两个必测边界。

### 知识点 3：NVFP4 融合 kernel 的"未初始化内存"陷阱

- **是什么**：#63382 发现 `silu_mul_cvt_fp16_to_fp4` 只遍历有效行列，但 scale factor 张量是用 `torch.empty` 分配的，padding 区域的 scale 是垃圾值 → 下游 GEMM 读到 NaN scale → NaN logits。
- **为什么重要**：`torch.empty` 不等于零！这是新手最常踩的坑之一。修复方式是让 kernel 遍历到 `round_up(numRows, 128)` 和 padding 列，显式写 0。
- **动手试试**：
  ```python
  import torch
  # 复现"empty 不是零"
  t = torch.empty(1024, device="cuda")
  print(t.isnan().any(), t.abs().max())  # 很可能非零
  ```
  然后读 `csrc/quantization/fp4/activation_nvfp4_quant_fusion_kernels.cu` 的 diff，找到 `sf_m = round_up<int>(numRows, 128)` 和 `*sf_out = 0;` 两处。
- **预期结果**：你能说清"为什么 padding 区域也要初始化"，以及为什么修复放在 kernel 里而不是 host 端 `zero_()`（省一次全量 memset）。

### 知识点 4：流水线并行（PP）里"部分和"必须先 all-reduce

- **是什么**：#60477 修复 DeepSeek V3.2 的 PP：中间 stage 的 MLP 输出是 TP 各 rank 的**部分和**，但 PP send 要求张量是 TP 复制的完整值。所以发送前必须先 `tensor_model_parallel_all_reduce`，并且只传 `hidden_states`（residual 已在本地累加）。
- **为什么重要**：PP 和 TP 组合时，"哪个张量是部分和、哪个是完整值"是最高频的 bug 来源。测试里用 `_PartialSumLayer` 返回 `hidden + 1`，all-reduce 后乘 2，期望值 8.0，非常直观。
- **动手试试**：
  ```bash
  pytest -v -s tests/models/deepseek_v32/test_sequence_parallel.py::test_pp_stage_sends_tp_reduced_residual_stream
  ```
  然后手算：输入 5.0，layer 返回 `(ones, 5+1=6)`，all-reduce 把 partial 1 变成 2，residual 6 + 2 = 8。对照 `vllm/models/deepseek_v32/amd/model.py` 的 forward。
- **预期结果**：你能画出"PP stage 输入 → 本地 residual → TP 部分和 → all-reduce → PP send"的数据流图，并解释为什么 `make_empty_intermediate_tensors` 从 `["hidden_states", "residual"]` 改成 `["hidden_states"]`。

### 知识点 5：Triton 的语法兼容性——链式 `or` 在 Ascend 后端会挂

- **是什么**：#64013 把 `a or b or c` 改成 `(a or b) or c`，因为某些 Triton 后端（Ascend）的解析器不接受链式布尔 or。
- **为什么重要**：这是"跨后端可移植性"的微观体现。一个在 CUDA 上跑得好好的 kernel，换个后端可能因为 AST 解析差异直接编译失败。加括号是最小侵入的修复。
- **动手试试**：
  ```python
  # 在本地 Triton 里验证两种写法语义相同
  import triton, triton.language as tl
  @triton.jit
  def k(a, b, c):
      x = (a or b) or c   # 试试改成 a or b or c
      return x
  # 用 triton.compile 或直接跑一个最小 kernel 看是否报错
  ```
  再读 `vllm/v1/worker/gpu/sample/penalties.py` 的 diff，注意注释"Parenthesize for Triton backends that reject chained `A or B or C`"。
- **预期结果**：你能解释"为什么加括号不改变语义却能修 bug"，并意识到写 kernel 时要考虑的不只是 GPU 架构，还有编译器前端。

---

## 🔥 重要更新

- **#64111 新增 MiniMax-M3.1-Flash 模型和 kernel**：新模型 + MegaMoE + B200 专用测试，是今天体量最大的 commit（3 万+ 行 diff），标志着又一个国产大模型被 vLLM 一等公民支持。
- **#63382 修复 NVFP4 NaN logits**：影响所有用 NVFP4 融合 act+quant 的模型，属于"线上会出垃圾输出"的严重 bug，值得所有量化用户升级。
- **#60477 修复 DSA PP all-reduce**：DeepSeek V3.2 在 PP+TP 组合下数值错误的根因修复，影响多机部署正确性。
- **#64256 / #52838 KV 卸载故障处理**：两个 commit 一起把"卸载链路出错"从"引擎崩溃"降级为"缓存降级"，是生产可用性的关键补丁。

---

## 📋 逐条解读

### 1. [Bugfix][AMD] A GPU-CPU KV transfer fault in OffloadingConnector takes the engine down instead of degrading the cache. (#52838)

- **代码层面**：改的是 `tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py` 及调度器逻辑。核心是：当某个 worker 报告 `failed_jobs` 时，不能把这个 store job 标记为"可加载"，否则会把回收块的旧 KV（可能是别的请求的）当成新数据返回。修复后失败信息会跨 step 保留，直到 job 真正完成时才以 `success=False` 调用 `complete_store`。
- **新手概念课堂**：就像快递柜——包裹没送到（store 失败），你不能把柜子标记为"有你的快递"，否则下一个人取到的是别人的东西。失败状态必须"记住"，不能因为下一步没再报错就忘掉。
- **对你有什么影响**：用 KV 卸载（offloading connector）时，偶发的传输错误不再导致整个引擎崩溃或返回错乱输出，而是该缓存块被丢弃、重新计算。

### 2. [ROCm] Fuse tiny-M MXFP8 quant into the block32 GEMM; guard lm_head skinny-GEMM by weight size (#63179)

- **代码层面**：`vllm/model_executor/kernels/linear/mxfp8/rocm_block32_gemm.py` 新增 `rocm_mxfp8_block32_gemm_quant`，在 M ≤ 8 时把 activation 量化融进 GEMM 的 K 循环；同时给 `lm_head` 的 skinny-GEMM 加权重尺寸门槛，避免小权重走错路径。测试覆盖 V4.1 attention 在 TP=4 的 5 种形状 × M ∈ {1,5,8}，要求与两段式路径 bit-exact。
- **新手概念课堂**：量化就像把一箱书按"每 32 本一个箱子、箱子贴一个缩放标签"打包。融合量化 = 打包工人直接在传送带上打包，而不是先全部打包再搬运——省一次往返，但必须保证每本书的位置一个不差。
- **对你有什么影响**：AMD MI300/MI355 上跑 DeepSeek 类模型 decode 阶段会更快，且数值与未融合路径完全一致，不用担心精度回归。

### 3. [New Model] Add MiniMax-M3.1-Flash Model and Kernels (#64111)

- **代码层面**：新增 `vllm/models/minimax_m3/` 目录（NVIDIA/AMD 两套实现）、`tests/models/test_minimax_m3_mega_moe.py`、benchmark `benchmark_mxfp8_tp4.py`，并在 `.buildkite` 里加了 B200 专用测试 job。包含 MegaMoE（大规模专家并行 MoE）kernel 和 MXFP8 TP4 线性层。
- **新手概念课堂**：新模型接入 = 翻译官 + 专用引擎。翻译官把 HuggingFace 权重映射到 vLLM 的层结构，专用引擎（MegaMoE）针对这个模型的稀疏专家模式做定制计算。
- **对你有什么影响**：可以直接 `vllm serve` MiniMax-M3.1-Flash，B200 上有专门的 CI 保障。

### 4. [Bugfix][KV Offload][P2P] Fail a timed-out load at once, defer releasing its slots (#64256)

- **代码层面**：`vllm/v1/kv_offload/tiering/p2p/session/client.py` 引入 `_PD_LOAD_TIMEOUT_S` 超时判定；`TieringOffloadingManager` 在超时后立即把 job 标记失败，但把 chunk 放进 `num_deferred_release_chunks`，直到收到 `AbortAckMsg` 才真正 free。测试 `test_timed_out_load_holds_chunks_until_abort_ack` 验证 free 数量不变。
- **新手概念课堂**：像图书馆借书——你发现对方超时没还，可以立刻标记"这本书状态异常"，但不能马上放回书架，因为对方可能还在读。必须等对方说"我确认不读了"（AbortAck）才能上架。
- **对你有什么影响**：P2P KV 卸载在远端故障时不会静默损坏数据，而是安全降级。

### 5. [Bugfix] DSA: all-reduce partial hidden_states before PP send (#60477)

- **代码层面**：`vllm/models/deepseek_v32/amd/model.py` 和 `nvidia/model.py` 的 `forward` 里，PP 中间 stage 在 send 前调用 `tensor_model_parallel_all_reduce`；`make_empty_intermediate_tensors` 从 `["hidden_states", "residual"]` 改为只传 `["hidden_states"]`（residual 已在本地累加进 hidden_states）。
- **新手概念课堂**：TP 把一层拆到多卡，每卡算一部分和（partial sum）；PP 把不同层放不同机器，机器间传的必须是"完整值"。所以中间 stage 必须先把部分和加起来再传，否则下一 stage 拿到的是残缺数据。
- **对你有什么影响**：DeepSeek V3.2 在 PP+TP 多机部署下输出正确，不再出现"能跑但结果不对"的隐蔽 bug。

### 6. [BugFix]/minicpm ocr downsample mode (#64105)

- **代码层面**：`MiniCPM-OCR` 的 `find_best_resize` / `get_refine_size` 增加 `downsample_mode` 参数（"4x"/"16x"），不再从 `self` 读；测试同步更新，验证 4x 模式对齐到 28px、16x 对齐到 56px。
- **新手概念课堂**：多模态预处理里，"下采样倍率"决定图像切成多少 patch。4x 和 16x 的 patch 网格不同，token 数差 4 倍。参数从实例属性改成显式传参，避免同一 processor 实例被两种模式复用时串味。
- **对你有什么影响**：MiniCPM-OCR 在不同 downsample 模式下 token 数正确，不再出现图像被错误切分导致的识别异常。

### 7. [Kernel] Update DeepGEMM commit (#64411)

- **代码层面**：`cmake/external_projects/deepgemm.cmake` 里 `_DEEPGEMM_UPSTREAM_TAG` 从 `0820030...` 更新到 `5cdb145...`。
- **新手概念课堂**：第三方 kernel 库（DeepGEMM）以 git submodule/commit 方式锁定版本。升级 commit = 拿到上游的 bugfix 和新 kernel，但也要重新跑 CI 验证 ABI 兼容。
- **对你有什么影响**：使用 DeepGEMM 的 MoE/GEMM 路径会拿到上游修复（注释提到 MX MegaMoE alignment fix）。

### 8. [Bugfix][DSv4.1] Budget prefill memory the startup profile misses (#63047)

- **代码层面**：`vllm/model_executor/layers/sparse_attn_indexer.py` 里 expanded block table 的尺寸从"按 token budget 分配"改为"按 decode token 数（每请求 1 个，无 draft）分配"；`tests/distributed/test_engram_dp_shard.py` 里 metadata-less replica 改为执行真实 lookup（DEAD_ID），让内存 profile 能看到 Engram 激活。
- **新手概念课堂**：启动时的显存 profile 是"抽样估算"，会漏掉某些只在 prefill 出现的临时张量。修复思路是：要么让 profile 覆盖到（Engram 测试），要么让分配不随 token budget 膨胀（block table）。
- **对你有什么影响**：DeepSeek V4.1 长 prefill 时不容易 OOM。

### 9. [Spec Decode] Batch-sharded draft logits refactor (#64356)

- **代码层面**：新增 `vllm/v1/worker/gpu/spec_decode/draft_logits.py::DraftLogitsCache`，替代裸 `torch.Tensor` + `draft_logits_num_shards` / `draft_logits_shard_idx` 两个散落字段。测试里 `spec.draft_logits.values` 访问底层张量。
- **新手概念课堂**：把"一个张量 + 两个描述它分片方式的整数"封装成类，就像把"文件句柄 + 偏移 + 长度"封装成 `FileView`。分片逻辑集中，调用方不用记 shard 语义。
- **对你有什么影响**：投机解码在多卡 batch 分片时逻辑更清晰，为后续支持更多 draft 策略铺路。

### 10. [Fusion] Fuse Qwen MoE shared MLP activation quantization (#61107)

- **代码层面**：`vllm/model_executor/models/qwen2_moe.py` 的 shared MLP `forward` 里，`self.act_fn(gate_up)` 改为 `maybe_fused_act_quant(self.act_fn, gate_up, self.down_proj)`，当 down_proj 是量化层时把 activation 量化融进激活函数。
- **新手概念课堂**：原本是"激活 → 写回显存 → 量化 kernel 读回 → 写回"，融合后"激活 + 量化"在一次 kernel 里完成，省两次显存往返。
- **对你有什么影响**：Qwen MoE 系列在量化（FP8/NVFP4）下 decode 更快。

### 11. [Kernel][MiniMax-M3] Enable SM90 split-K in the Triton indexer (#64037)

- **代码层面**：`vllm/models/minimax_m3/common/ops/index_topk.py` 里 `score_target_grid` 从固定 48 改为按架构选择：SM12.0 → 48，SM90 → 1056，其他 → 1（即不 split-K）。测试参数化 `device_capability ∈ {80, 90, 120}`。
- **新手概念课堂**：split-K = 把一个 K 维很长的点积拆给多个 block 并行算再求和。SM90（H100）的 SM 数量多、L2 大，适合更大的 split 目标；SM80 反而不划算。target grid 就是"我想让 grid 大约有多少个 block"。
- **对你有什么影响**：MiniMax-M3 在 H100 上 prefill 的 indexer 更快。

### 12. [KV Connector][NIXL] Add a ucx_backend_params key for UCX backend init params (#62905)

- **代码层面**：NIXL connector 的 `kv_connector_extra_config` 新增 `ucx_backend_params`，透传给 NIXL 的 `create_backend("UCX", init_params)`。文档示例：`{"ucx_error_handling_mode": "none"}` 可启用 `rocm_ipc` 等节点内 GPU 传输。
- **新手概念课堂**：UCX 是高性能网络/共享内存传输库。默认 `error_handling_mode=peer` 只选能检测 peer 故障的 transport，代价是排除了一些快的节点内通道。这个配置让你在"容错"和"速度"之间选。
- **对你有什么影响**：AMD 多卡节点内 KV 传输可以更快，但需要自己权衡故障检测。

### 13. [ROCm][Perf] Fuse MXFP8 activation quant into the 1x32 dot_scaled linear for M <= 16 (#63915)

- **代码层面**：`vllm/model_executor/kernels/linear/mxfp8/rocm_native.py` 新增 `_select_fused_quant_cfg`，M ≤ 16 时 `_launch` 收到 `x_scale=None` 表示量化在 kernel 内做。测试用 `_spy_launch` 断言 `fused == [True]` 且输出 bit-exact。
- **新手概念课堂**：和 #63179 是同一思路的不同 tile 配置（1x32 vs block32）。`dot_scaled` 是 Triton 的 microscaling 矩阵乘原语，硬件直接吃 FP8 + E8M0 scale。
- **对你有什么影响**：AMD 上小 batch 的 MXFP8 线性层少一次 kernel launch，decode 延迟降低。

### 14. [Models] Deprecate `torch.compile` for Qwen3.5 models (#51012)

- **代码层面**：`vllm/models/qwen3_5/` 新目录（模型代码从 `vllm/model_executor/models/qwen3_5.py` 迁移/重构），并在多处 `.buildkite` 配置里把 `vllm/models/qwen3_5/` 加入依赖；Qwen3.5 不再走 `torch.compile` 路径（改用自定义 kernel/attention 后端）。
- **新手概念课堂**：`torch.compile` 是通用图编译器，但对有特殊 attention（如 GDN/Mamba 混合）的模型，手写 kernel 往往更快更可控。deprecate 意味着维护者判断"通用编译器在这条路上收益不如专用实现"。
- **对你有什么影响**：Qwen3.5 启动更快（省编译时间），但如果你依赖 compile 的某些行为需要调整。

### 15. [ROCm][Model] Enable ROCm AITER unified attention for MiMo-V2.6 (#64064)

- **代码层面**：`benchmarks/kernels/benchmark_reshape_and_cache_flash.py` 新增 `_make_packed_kv_caches` 支持 K/V 打包在一个 buffer（asymmetric head_size）；`.buildkite/test-amd.yaml` 加 `test_mimo_v2_attention_backend.py`。
- **新手概念课堂**：AITER 是 AMD 的算子库。unified attention 要求 K 和 V 在内存里连续打包（共享 head stride），这样一次 DMA 就能搬 K+V。benchmark 里 `view.split([head_size, v_head_size], dim=-1)` 就是构造这种布局。
- **对你有什么影响**：MiMo-V2.6 在 AMD 上走 AITER 统一 attention，吞吐提升。

### 16. [Bugfix][NVFP4] Fix NaN logits and garbage output from uninitialized fused act+quant scales (#63382)

- **代码层面**：`csrc/quantization/fp4/activation_nvfp4_quant_fusion_kernels.cu` 里循环上界从 `numRows` 改为 `sf_m = round_up(numRows, 128)`，列上界从 `num_packed_cols` 改为 `num_padded_cols`；padding 区域显式 `*sf_out = 0`，并且只在 `valid` 时调用 `cvt_warp_fp16_to_fp4`。
- **新手概念课堂**：NVFP4 的 scale factor 按 128 行分块，张量分配时 padding 到 128 的倍数。如果 kernel 不写 padding 区域，`torch.empty` 的垃圾值会被下游 GEMM 当成合法 scale → NaN。修复 = 让 kernel 也负责"擦干净"padding。
- **对你有什么影响**：用 NVFP4 量化跑推理不再偶发 NaN logits / 乱码输出，强烈建议升级。

### 17. [ROCm][Qwen4Exp] Enable fp8 KV and indexer caches on the AMD QSA path (#61529)

- **代码层面**：`tests/models/qwen4_exp/test_qsa_amd.py` 参数化 `use_fp8`，测试里把 KV cache 量化到 `current_platform.fp8_dtype()` 后用 upcast 到 float 的参考值对比；`.buildkite` 加 MI300 Qwen4 Exp job。
- **新手概念课堂**：FP8 KV cache = 把 K/V 存成 8 位浮点，显存减半。但 fp8 是有损的，所以参考实现必须用"量化后的值"而不是原始 bf16 值来比，否则测试会假失败。
- **对你有什么影响**：Qwen4 Exp 在 AMD 上可以用 fp8 KV cache，长上下文显存压力减半。

### 18. [Bugfix] Parenthesize penalties Triton chained `or` for Ascend backends (#64013)

- **代码层面**：`vllm/v1/worker/gpu/sample/penalties.py` 里 `use_penalty = a or b or c` 改为 `(a or b) or c`，加注释说明 Ascend 的 Triton 前端不接受链式 or。
- **新手概念课堂**：Python 语义里 `a or b or c` 是左结合，加括号不改变结果；但某些 Triton 编译器前端在 AST 解析时不支持 N 元 BoolOp，加括号强制成二元嵌套。
- **对你有什么影响**：Ascend 后端上 penalties 采样不再编译失败。

### 19. [ROCm][Kimi-K3] Allow DSpark adaptive verification with DCP on ROCM_AITER_MLA (#62990)

- **代码层面**：`vllm/v1/attention/backends/mla/rocm_aiter_mla.py` 里 DCP（decode context parallel）路由在 adaptive verification 开启时可以选择 CPRR（custom partial reduction）或 SEGMENTED；测试 `test_adaptive_varlen_shards_match_reference` 用变长序列对比参考实现。
- **新手概念课堂**：DCP = 把长上下文的 decode 拆到多卡，每卡算一段 attention 再合并（类似 split-K）。adaptive verification = 投机解码时根据 draft 质量动态决定验证多少 token。两者组合时，合并逻辑要匹配。
- **对你有什么影响**：Kimi-K3 在 AMD 多卡 + 投机解码下能开 adaptive verification，长上下文 decode 更快。

### 20. [XPU][UT] reuse test_phi3v.py::test_models_text on XPU (#64228)

- **代码层面**：`tests/models/multimodal/pooling/test_phi3v.py` 删除了 `@pytest.mark.skipif(current_platform.is_xpu(), ...)`，因为 `vllm-xpu-kernels` 已补上 `paged_decode_default.conf` 里缺的 `8,96,64,false,true,false` tuple。
- **新手概念课堂**：XPU 的 paged attention 需要预编译的 kernel 配置表（按 block_size/head_size 等 tuple 索引）。缺一个 tuple 就跑不了对应形状。这个 commit 是"上游依赖修好了，把 skip 摘掉"。
- **对你有什么影响**：Intel XPU 上 Phi-3V 的文本测试重新纳入 CI，回归能被及时发现。

## sglang
## 💡 今日关键词

- **关键词：AMD GPU 生态适配（gfx1250 / MI355X / aiter）**
  - 一句话解释：SGLang 正在把大量算子（MLA absorb BMM、lm_head、fused QK-norm+RoPE、MoE 调优）从通用 torch 实现替换成 AMD 自家 aiter 库的专用 kernel，并针对新硬件 gfx1250 做专门优化。
  - 为什么社区在关注：今天 20 个 commit 里有 6 个是 `[AMD]` 标签，说明 AMD 正在成为 SGLang 的一等公民平台，异构计算（NVIDIA + AMD + NPU）已经是现实需求而不是口号。
  - 对新手意味着什么：如果你只会写 CUDA，建议尽快补 ROCm/HIP 和 Triton 的知识；AMD 生态的 kernel 工程师目前非常稀缺，这是明显的蓝海方向。

- **关键词：投机解码（Speculative Decoding / DSpark / MTP）**
  - 一句话解释：用小模型“草稿”多个 token，再让大模型一次性验证，从而加速推理。今天涉及“分片接受（sharded accept）”、“模拟接受（simulated acceptance）”等细节优化。
  - 为什么社区在关注：投机解码是当下推理加速最有效的杠杆之一，但正确性（接受概率分布必须严格等于目标分布）和分布式（TP 分片后 logits 不在一个 rank 上）是两大难点。
  - 对新手意味着什么：想进入推理优化领域，投机解码是必学内容，涉及概率论 + 分布式通信 + kernel 实现，是很好的综合练习场。

- **关键词：流水线并行下的 LoRA 显存池（PP + Multi-LoRA）**
  - 一句话解释：多 LoRA 服务时，每个 GPU 只为自己负责的那几层分配 LoRA buffer，PP 各 stage 共享一个“适配器 ID 池”。
  - 为什么社区在关注：多租户 LoRA 服务是实际生产需求，但显存是瓶颈；PP 场景下的显存复用此前是个坑。
  - 对新手意味着什么：分布式推理 + 参数高效微调是两个热门方向的交汇点，理解“池化”和“按需分配”思想对做系统优化很有帮助。

- **关键词：CUDA Graph 与设备无关化（device-agnostic）**
  - 一句话解释：把硬编码的 `torch.cuda.Event` / `torch.cuda.current_stream()` 替换成 `device_module` 抽象，让同一份代码能在 CUDA、ROCm、NPU 上跑。
  - 为什么社区在关注：这是“一套代码多硬件”的必经之路，也是很多隐蔽 bug（比如非 CUDA 设备上 event 挂起）的根源。
  - 对新手意味着什么：写跨平台代码时，养成“不直接调 torch.cuda.*”的习惯，是进阶工程师的基本素养。

- **关键词：分层 KV 缓存（HiCache / NIXL / L3 存储）**
  - 一句话解释：把 KV cache 从 GPU 显存下沉到 CPU 内存甚至远端存储，用 RDMA 传输，以支持超长上下文。
  - 为什么社区在关注：长上下文（800k+）是当前大模型的标配，显存根本放不下，分层缓存是唯一出路。
  - 对新手意味着什么：存储层次结构（GPU→CPU→远端）是系统设计的经典问题，理解它等于掌握了“内存墙”的应对思路。

---

## 🧪 重要知识点 & 动手实验

- **知识点 1：Triton 写自定义 kernel —— 以 scale-shift 为例**
  - **是什么**：Triton 是 OpenAI 出的类 Python DSL，让你用接近 NumPy 的写法生成 GPU kernel，不用写 CUDA C++。Commit 2 用 Triton 重写了 `x * (scale_constant + scale) + shift` 这个逐元素运算，并保证与 torch 逐位一致（bit-exact）。
  - **为什么重要**：很多“看起来简单”的逐元素操作（加法、缩放、归一化）在 torch 里会启动多个 kernel、产生多次显存读写；融合成一个 Triton kernel 能省带宽。但“bit-exact”很难——要避免 FMA 合并、控制舍入顺序，否则数值对不上，模型输出漂移。
  - **动手试试**：
    ```python
    import torch, triton, triton.language as tl

    @triton.jit
    def add_one_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + 1.0, mask=mask)

    n = 1024
    x = torch.randn(n, device="cuda")
    out = torch.empty_like(x)
    grid = (triton.cdiv(n, 256),)
    add_one_kernel[grid](x, out, n, BLOCK=256)
    print(torch.allclose(out, x + 1.0))  # True
    ```
    然后尝试把 `x + 1.0` 改成 `x * 2.0 + 1.0`，对比 `torch.addcmul` 的结果，观察是否 bit-exact。
  - **预期结果**：你会看到 Triton 输出与 torch 结果一致；如果改成涉及乘加，可能因 FMA 产生 1 ulp 差异，这正是 Commit 2 里“without FMA contraction”注释要解决的问题。

- **知识点 2：投机解码的接受-拒绝采样（Acceptance Sampling）**
  - **是什么**：草稿模型给出分布 q，目标模型给出分布 p。对草稿 token x，以 `min(1, p(x)/q(x))` 的概率接受；拒绝时从 `max(p-q, 0)` 归一化后的分布重采样一个“bonus token”。这样最终输出的分布严格等于 p。Commit 1 和 Commit 6 都在处理这个算法的分布式/模拟版本。
  - **为什么重要**：如果实现错了，输出分布会偏离目标模型，导致质量下降但很难察觉（因为看起来“还挺通顺”）。这是投机解码最容易踩的坑。
  - **动手试试**：
    ```python
    import torch

    def speculative_sample(p, q, draft_token):
        # p, q: [vocab] 概率分布；draft_token: int
        u = torch.rand(())
        if u * q[draft_token] < p[draft_token]:
            return draft_token, True
        # 拒绝：从 max(p - q, 0) 重采样
        residual = torch.clamp(p - q, min=0)
        residual = residual / residual.sum()
        return torch.multinomial(residual, 1).item(), False

    p = torch.softmax(torch.randn(10), 0)
    q = torch.softmax(torch.randn(10), 0)
    print(speculative_sample(p, q, 3))
    ```
    进一步：循环 10 万次，统计接受样本的分布，验证它是否等于 p。
  - **预期结果**：接受样本的经验分布应逼近 p（而非 q），说明算法正确。

- **知识点 3：流水线并行下的显存池化（LoRA PP）**
  - **是什么**：PP（Pipeline Parallelism）把模型按层切到不同 GPU。多 LoRA 服务需要为每个 adapter 存 LoRA 权重；Commit 4 让每个 stage 只为自己负责的层分配 buffer，并且所有 in-flight micro-batch 共享一个 adapter ID 池。
  - **为什么重要**：如果不共享，micro-batch 一多，显存会随 batch 数线性膨胀；如果不按层分配，每个 stage 都会存全模型的 LoRA 权重，浪费 pp_size 倍显存。
  - **动手试试**（概念模拟，无需多卡）：
    ```python
    # 模拟：4 层模型，2 个 stage，每个 adapter 每层占 1 单位显存
    num_layers, pp_size, num_adapters = 4, 2, 3
    layers_per_stage = num_layers // pp_size
    # 旧方案：每个 stage 存全部层
    old = pp_size * num_layers * num_adapters
    # 新方案：每个 stage 只存自己的层
    new = pp_size * layers_per_stage * num_adapters
    print(f"old={old}, new={new}, ratio={old/new}")  # old=24, new=12, ratio=2.0
    ```
    把 `pp_size` 改成 4、8，观察比例变化。
  - **预期结果**：比例约等于 `pp_size`，与 Commit 4 文档中“约 1/pp_size”的描述吻合。

- **知识点 4：设备无关的 CUDA Graph / Event 编程**
  - **是什么**：`torch.cuda.Event` 是 CUDA 专用 API。Commit 9 把它换成 `self.device_module.Event()`，其中 `device_module` 在 CUDA 上是 `torch.cuda`，在 ROCm 上是 `torch.cuda`（HIP 兼容层），在 NPU 上是 `torch_npu`。同时修复了 NIXL worker 线程未绑定设备导致 event 挂错 GPU 的问题。
  - **为什么重要**：跨设备代码里，任何硬编码的 `torch.cuda.*` 都是定时炸弹。此外，多 GPU 场景下 worker 线程必须显式 `set_device`，否则默认用 device 0，会导致跨卡同步死锁或数据错乱。
  - **动手试试**：
    ```python
    import torch
    # 不好的写法
    # ev = torch.cuda.Event()
    # 好的写法（SGLang 风格）
    device_module = torch.cuda  # 或 torch_npu / torch.xpu
    ev = device_module.Event()
    ev.record(device_module.current_stream())
    ev.synchronize()
    print("event ok, device:", torch.cuda.current_device())
    ```
    如果你有多卡，试试在 `threading.Thread` 里不调用 `set_device` 直接 record event，观察行为。
  - **预期结果**：显式 `set_device` 后 event 绑定正确；不设置时在多卡机器上可能落到 device 0，导致同步语义错误。

- **知识点 5：量化权重加载中的 TP 分片与 act-order**
  - **是什么**：GPTQ/Marlin 量化的 MoE 模型里，w2（down projection）在 TP 下要按输入维度切分；act-order 量化还会重排 scale 的顺序。Commit 5 修复了 BF16 dtype 和 TP 下 w2 scale 加载错误，并新增 `load_full_w2` 参数以支持某些需要整块加载的场景。
  - **为什么重要**：量化模型的权重加载一旦切错维度，不会报错，只会让输出变成乱码或精度暴跌——这是最隐蔽的 bug 类型之一。
  - **动手试试**：
    ```python
    import torch
    # 模拟 w2 的 TP 切分：expert_data 形状 [hidden, intermediate]
    # TP=2 时每个 rank 只拿 intermediate 的一半
    full_w2 = torch.arange(24).reshape(4, 6)  # [4, 6]
    tp_size, tp_rank = 2, 0
    shard = full_w2.chunk(tp_size, dim=1)[tp_rank]
    print("full:", full_w2.shape, "shard:", shard.shape)
    # 验证拼接能还原
    shards = [full_w2.chunk(tp_size, dim=1)[r] for r in range(tp_size)]
    print(torch.equal(torch.cat(shards, dim=1), full_w2))  # True
    ```
    再尝试把 `dim=1` 改成 `dim=0`，观察结果是否还能还原（不能），理解“切错维度”的后果。
  - **预期结果**：正确维度下拼接可还原；错误维度下形状不匹配或数值错乱，体会 Commit 5 修复的重要性。

---

## 🔥 重要更新

- **Commit 1（DSpark 分片接受）**：AMD 上投机解码不再每步 gather 整个 vocab 的 draft logits，改为在 TP 分片上直接做接受-拒绝采样，显著降低通信量。这是投机解码 + 分布式的前沿优化，值得关注其正确性保证。
- **Commit 4（PP 下多 LoRA 显存优化）**：多 LoRA + 流水线并行的显存占用从“每 stage 存全模型”降到“约 1/pp_size”，直接提升可服务的 adapter 数量，是生产环境的实打实收益。
- **Commit 9（disaggregation event 修复）**：修复非 CUDA 设备和 NIXL 下 early-send wait event 的挂起/错绑问题，影响 PD 分离部署的正确性，属于“不修就偶发卡死”的隐蔽 bug。
- **Commit 16（hybrid SWA 从 layer_types 自动检测）**：模型配置里带 `layer_types` 时自动识别混合滑动窗口注意力，减少手动配置错误，影响长上下文模型的显存和性能表现。

---

## 📋 逐条解读

### 1. [Perf] [AMD] dsv41: Drop the per-step whole-vocab gather of draft logits via sharded accept (#44465)
- **代码层面**：修改 `markov_sample.py` 的 `MarkovSampleStep`，把条件判断从 `gather is None` 改为 `gather_logits is None`，修正了逻辑错误（原来用错变量）；新增 `sharded_accept_hip.py`（287 行），实现“draft logits 留在 TP 分片上”的接受-拒绝采样 Triton kernel。
- **新手概念课堂**：想象一个班级投票，原来每个小组都要把完整选票汇总到班长那里统计（whole-vocab gather）；现在每个小组只统计自己负责的那部分，最后合并结果（sharded accept）。通信量从“全量”降到“分片”，但算法必须保证合并后结果与集中统计完全等价。
- **对你有什么影响**：在 AMD 多卡上跑投机解码时，每步的通信开销降低，吞吐提升；普通单卡用户感知不明显。

### 2. [Perf] [AMD] diffusion: Replace the gfx1250 bf16 scale-shift torch fallback with a bit-exact Triton kernel (#46723)
- **代码层面**：在 `scale_shift_triton.py` 新增 `_scale_shift_fp32_rows_kernel` 和 `_fuse_scale_shift_fp32_gfx1250`，处理 `x * (scale_constant + scale) + shift`，要求 bf16 输入、fp32 scale、shift 支持 [B,1,C] 或 [B,L,C]，并做严格的形状/dtype 校验，不满足则返回 None 回退 torch。
- **新手概念课堂**：这是“融合算子”的典型例子——把三个逐元素操作合成一个 kernel，减少显存往返。难点在于“bit-exact”：torch 的 fallback 每步都舍入到 fp32，Triton 版本必须避免编译器把乘加合并成 FMA（会改变舍入），才能逐位一致。
- **对你有什么影响**：gfx1250 上 diffusion 模型的 modulate 层变快，且数值与原来完全一致，不会出现“升级后出图风格变了”的问题。

### 3. [Fix] hicache: Isolate NIXL storage tests from background eviction (#47197)
- **代码层面**：测试文件 `test_hicache_nixl_storage.py` 把固定目录 `/tmp/test_nixl_unified` 改为 `tempfile.mkdtemp()`，用 `addCleanup` 替代 `tearDown`，并新增 `l3_cleaner_enabled: False` 配置，避免后台清理线程删掉测试刚写入的文件。
- **新手概念课堂**：这是“测试隔离”的经典问题——多个测试共享固定路径会互相污染；后台清理线程（L3 cleaner）会在测试断言前把文件删掉，导致偶发失败（flaky test）。用临时目录 + 关闭清理器，让测试可控。
- **对你有什么影响**：普通用户无感；但对贡献者来说，CI 更稳定，不会因为随机失败重跑。

### 4. [Perf] LoRA: improve throughput and GPU memory usage for multi-LoRA serving under PP (#45951)
- **代码层面**：`lora_manager.py` 的 `fetch_new_loras` 新增 `pp_loras` 参数，把 PP 其他 stage 的 adapter 也纳入池容量校验；`init_memory_pool` 传入 `lora_layer_ids`、`has_embed_tokens_lora`、`has_lm_head_lora`，让 mem_pool 只为本地层分配 buffer；文档和参数说明同步更新。
- **新手概念课堂**：把 LoRA 显存池想象成酒店房间：以前每个楼层（stage）都建了整栋楼的房间（全模型 buffer），现在每层只建自己那几间；同时所有在住的客人（in-flight micro-batch）共享一个房号池（adapter ID 池），避免重复开房。
- **对你有什么影响**：PP + 多 LoRA 部署时，同样显存能装更多 adapter，或同样 adapter 数下显存更省；`--max-loras-per-batch` 的语义变成“池容量”，需要重新理解。

### 5. [Fix] gptq_marlin: Fix BF16 dtype and TP act-order w2 scales in MoE (#35955)
- **代码层面**：`fused_moe_triton/layer.py` 的 `_load_w2` 新增 `load_full` 参数，`_load_model_weight_or_group_weight_scale` 新增 `load_full_w2` 并透传；把 TP 切分逻辑包进 `if not load_full:`，支持整块加载 w2；同时修正 BF16 dtype 处理。
- **新手概念课堂**：量化模型的权重加载像拼图：TP 下每张卡只拿拼图的一块（按 intermediate 维度切），act-order 量化还会打乱编号（重排 scale）。如果切错维度或没处理重排，拼出来的图就是乱的——但程序不报错，只是输出质量崩了。
- **对你有什么影响**：使用 GPTQ/Marlin 量化的 MoE 模型（尤其 BF16 + TP + act-order）时，输出正确性和精度恢复；之前可能遇到“模型能跑但答非所问”的问题。

### 6. [Fix] dspark: Keep the real verify accept cost under simulated acceptance (#46840)
- **代码层面**：`simulated_bonus.py` 删除了旧的 `_simulated_bonus_partial_kernel` 和 `_simulated_bonus_select_kernel`（两阶段 Gumbel-max 采样），改为单 kernel 实现；调整 `simulated_bonus_sample` 的调用逻辑。
- **新手概念课堂**：“模拟接受”是测试/基准里用的技巧——不真正跑目标模型验证，而是用模拟的接受率来估算性能。但如果模拟路径比真实路径快太多，测出的数字就是假的。这个 commit 保证模拟接受时仍付出真实的 verify 成本，让 benchmark 可信。
- **对你有什么影响**：如果你看 DSpark 的性能数据，数字更可信；普通推理用户无感。

### 7. [Perf] [AMD] mla: Use aiter bf16 batched GEMM for the absorb BMMs on gfx1250 decode (#44456)
- **代码层面**：`forward_mla_rocm.py` 新增 `_use_gfx1250_absorb_bmm` 开关和 `_gfx1250_absorb_bmm_ok` 门控（要求 bf16、w_scale==1.0、decode 行数 ≤ 阈值、非 piecewise CUDA graph），在 `rocm_absorb_q_bmm` / `rocm_absorb_v_bmm` 里用 `batched_gemm_bf16` 替代 `torch.bmm`，小 batch 用 “gluon” backend，大 batch 用 “triton”。
- **新手概念课堂**：MLA（Multi-head Latent Attention）的“absorb”操作是把 Q/V 与权重矩阵做批量矩阵乘（BMM）。torch.bmm 是通用实现，aiter 的专用 GEMM 针对 AMD 硬件调优，decode 阶段（行数少）收益最大；prefill 行数多，torch.bmm 反而更好，所以加了门控。
- **对你有什么影响**：gfx1250 上 DeepSeek 类 MLA 模型 decode 变快；CUDA graph 场景自动回退，不会因 dynamo 无法 trace aiter 而崩。

### 8. [Perf] [AMD] lm_head: Use aiter tgemm for the bf16 lm_head on gfx1250 (#44455)
- **代码层面**：`logits_processor.py` 新增 `_use_aiter_lm_head` 开关和 `_aiter_lm_head_ok` 门控（2D、bf16），在 `_compute_lm_head` 里用 `tgemm.mm` 替代 `torch.matmul`；新增单元测试 `test_logits_processor_aiter_lm_head.py` 验证门控逻辑。
- **新手概念课堂**：lm_head 是把 hidden state 投影到 vocab 的大矩阵乘（vocab 通常 10 万+），是 decode 阶段的计算大头之一。aiter 的 tuned GEMM 会根据形状选最优 kernel，比通用 matmul 快。
- **对你有什么影响**：gfx1250 上 bf16 模型的最后一步变快，首 token 延迟和吞吐都有改善。

### 9. [Fix] disaggregation: Fix the early-send wait event on non-CUDA devices and in nixl (#43299)
- **代码层面**：`prefill.py` 把 `torch.cuda.Event()` / `torch.cuda.current_stream()` 换成 `self.device_module.*`；`nixl/conn.py` 的 transfer worker 新增 `device_bound` 标志，首次 wait 前显式 `set_device(gpu_id)`，并在 `add_transfer_request` 里把 `wait_event` 传给 `TransferKVChunk`；`common/conn.py` 更新注释。
- **新手概念课堂**：PD 分离（prefill/decode 分开部署）时，prefill 写完 KV 要通知 decode 端“可以读了”，这个通知就是 event。在非 CUDA 设备上，硬编码 `torch.cuda.Event` 会失败；在多卡机器上，worker 线程不绑定设备会默认用 device 0，导致 event 记在错误的卡上，同步永远等不到。
- **对你有什么影响**：在 AMD/NPU 上做 PD 分离部署不再偶发卡死；NIXL 传输路径的 early-send 正确性修复。

### 10. [Fix] [AMD] deepseek-v2/v3: Fix PPMissingLayer AttributeError in aiter_gfx95 code path (#21194)
- **代码层面**：`deepseek_v2.py` 里把 `self.embed_tokens.embedding_dim` 改为 `getattr(self.embed_tokens, "embedding_dim", None)`，把 `self.layers[i].mlp` 改为 `getattr(self.layers[i], "mlp", None)`，避免 PP 下某些 stage 的层是 `PPMissingLayer`（占位符，没有这些属性）时抛 AttributeError。
- **新手概念课堂**：PP 切分后，每个 stage 只保留自己负责的层，其他层用 `PPMissingLayer` 占位（不占显存）。但代码如果直接访问 `.mlp` 或 `.embedding_dim`，占位符没有这些属性就会崩。`getattr(..., None)` 是防御性写法。
- **对你有什么影响**：AMD gfx95 上跑 DeepSeek V2/V3 的 PP 部署不再启动即崩。

### 11. [Fix] [AMD] docker: Remove implicit INT8 QuickReduce default as it causes accuracy issues in 800k+ long context evals (#47163)
- **代码层面**：`docker/rocm.Dockerfile` 删掉一行 `ENV ROCM_QUICK_REDUCE_QUANTIZATION=INT8`。
- **新手概念课堂**：QuickReduce 是 ROCm 上的 all-reduce 优化，INT8 量化能省带宽但引入误差。短上下文看不出来，800k+ 长上下文时误差累积导致精度明显下降。默认开启这种“有损优化”是危险的。
- **对你有什么影响**：用官方 ROCm Docker 跑超长上下文评测时，精度恢复正常；需要 INT8 加速的用户可以手动设置环境变量。

### 12. [Fix] kernel-plugins: Serve the MSA plugin behind --enable-nvidia-megakernel (#46625)
- **代码层面**：`argparse_actions.py` 新增 `DeprecatedChoiceToFlagAction`，把旧的 `--monokernel-backend msa` 映射到新的 `--enable-nvidia-megakernel`；文档更新 `--enable-nvidia-megakernel` 说明（支持 MiniMax-M3.1 的 MSA 插件）和 `--monokernel-backend` 的 deprecated 说明。
- **新手概念课堂**：这是“参数迁移”的工程实践——旧参数有多个取值，每个取值变成一个独立布尔 flag。`DeprecatedChoiceToFlagAction` 在解析时自动翻译并打警告，保证老用户的启动脚本不坏。
- **对你有什么影响**：用旧参数 `--monokernel-backend msa` 仍能工作（有警告），新用户直接用 `--enable-nvidia-megakernel`。

### 13. [Fix] sgl-kernel: bump sgl-attn to pick up FP8 two-level accumulation (#46770)
- **代码层面**：`CMakeLists.txt` 把 flash-attention 依赖的 commit hash 从 `f89bc23` 换成 `141fa06`，更新 SHA256。
- **新手概念课堂**：FP8 注意力的累加如果只用一级（直接累加到 fp32），长序列下误差大；“两级累加”先用低精度累加局部块，再高精度合并，平衡速度和精度。这是 kernel 库的内部优化，通过 bump 依赖引入。
- **对你有什么影响**：使用 FP8 注意力时精度更好，尤其长序列；无需改代码，重新编译/安装即可。

### 14. [Fix] mem_cache: Default hicache_write_back_staging on MLAView (#47168)
- **代码层面**：`mla.py` 的 `MLAView` 类新增字段 `hicache_write_back_staging: Optional[Tensor | Tuple[Tensor, Tensor]] = None`，由 HiCache 在需要时设置。
- **新手概念课堂**：HiCache 把 KV 从 GPU 写回 CPU 时，可能需要一块“暂存区”（staging buffer）来做异步拷贝。MLA 的 KV 布局特殊（压缩成 latent），之前没有这个字段导致访问时报 AttributeError。
- **对你有什么影响**：MLA 模型（DeepSeek 系列）开启 HiCache 写回时不再崩。

### 15. [Fix] diffusion: LoRA adapter-state and merge-transition fixes (#45837)
- **代码层面**：`pipeline.py` 新增 `_lora_module` 方法处理 “critic” → “fake_score_transformer” 的别名；`_apply_lora_to_layers` 里把 `clear_existing and idx == 0` 改为 `clear_existing and adapted_count == adapted_before`（只在第一个覆盖该层的 adapter 上清空）；`_reactivate_cached_dynamic_lora_layers` 调整检查顺序，先查 `layer.merged` 再查 `len(lora_weights_list)`。
- **新手概念课堂**：多 LoRA 叠加时，“清空旧权重”只能在第一个真正作用到该层的 adapter 上做。如果按 adapter 序号判断（idx==0），但第 0 个 adapter 不覆盖这一层，就会漏清或错清，导致权重叠加错误。
- **对你有什么影响**：diffusion 模型多 LoRA 切换/合并时结果更正确，不再出现“换 adapter 后画风错乱”。

### 16. [Perf] hybrid-swa: Detect hybrid SWA models from layer_types (#45295)
- **代码层面**：`attention_hook.py` 新增 `_hybrid_swa_memory_backend_disable` 后处理 pass；`model_override_base.py` 的 `model_config_of` 缓存 key 从 `model_path` 改为 `(model_path, disable_hybrid_swa_memory)`，因为 override 会改变 `is_hybrid_swa`；相关注释更新。
- **新手概念课堂**：混合滑动窗口注意力（hybrid SWA）指模型里一部分层用全注意力、一部分用滑动窗口。以前靠模型名硬编码判断，现在从 config 的 `layer_types` 自动识别。缓存 key 要包含影响结果的 flag，否则 override 改了 flag 但缓存返回旧配置。
- **对你有什么影响**：新模型只要 config 里有 `layer_types` 就能自动启用 hybrid SWA 优化，减少手动配置；显存和长上下文性能改善。

### 17. [Doc] docs: Fix documentation normative issues (#47041)
- **代码层面**：`installation.mdx` 里把裸 `System Settings` 改为链接 `[System Settings](#system-settings)`；把多个 `link` 文本改为有意义的描述（“HDK driver download”、“TorchNPU releases”）；`Some other dependencies` 改为 `Additional dependencies`。注意 diff 里有个拼写错误 “realses”（应为 releases），可能是待修的。
- **新手概念课堂**：文档规范（normative）关注可访问性——链接文本要有描述性，不能全是 “link”（屏幕阅读器用户无法区分）；章节引用要可点击。
- **对你有什么影响**：阅读 Ascend NPU 安装文档时导航更方便；对代码无影响。

### 18. [Perf] [AMD] qwen3: Use the fused QK-norm + RoPE + KV write in decode with an fp8 KV cache (#46832)
- **代码层面**：`qwen3.py` 新增 `_host_kv_scale` 辅助函数（从 `k_scale`/`k_scale_float` 取 host 端 float，避免 device-to-host 同步破坏 graph capture），把 `_fused_k_scale`/`_fused_v_scale` 改为惰性构建的 `_fused_kv_scales`（因为 scale 在 `__init__` 之后才设置）；扩展 `aiter_fused_qk_norm_rope_set_kv` 支持 fp8 KV cache。
- **新手概念课堂**：decode 时每个 token 都要做 QK-norm、RoPE、写 KV cache，三个操作融合成一个 kernel 能省大量显存往返。难点是 KV scale：kernel 在 host 端读 scale，如果传 device tensor 会触发 D2H 同步，CUDA graph 捕获时直接报错。所以必须提前把 scale 拷到 CPU 并缓存。
- **对你有什么影响**：gfx1250 上 Qwen3 用 fp8 KV cache 时 decode 变快，且 CUDA graph 能正常捕获。

### 19. [Perf] [AMD] moe: Add tuned Triton fused MoE config for gemma-4-26B-A4B on MI355X (#46731)
- **代码层面**：新增 JSON 配置文件 `E=128,N=704,device_name=AMD_Instinct_MI355X.json`，为不同 batch size（1, 2, 4, 8, ..., 256）指定 `BLOCK_SIZE_M/N/K`、`GROUP_SIZE_M`、`num_warps`、`num_stages`、`waves_per_eu` 等调优参数。
- **新手概念课堂**：Triton fused MoE kernel 的性能高度依赖 tile 大小等参数，不同 batch size 最优参数不同。这是“自动调优（autotuning）结果固化”——离线跑一遍搜索，把最优配置存成 JSON，运行时按形状查表。
- **对你有什么影响**：MI355X 上跑 Gemma-4-26B-A4B 的 MoE 推理变快，无需手动调参。

### 20. [Perf] observability: Send scheduler timing to the tokenizer as typed reports (#47166)
- **代码层面**：`io_struct.py` 里 `BatchTokenIDOutput`、`BatchStrOutput` 的 `time_stats` 从 `Optional[PickleWrapper]` 改为 `Optional[List[Optional[SchedulerOutputTimeStats]]]`（强类型，不再 pickle），`BatchEmbeddingOutput` 直接删掉该字段；`multi_tokenizer_mixin.py` 里不再对 `time_stats` 做 pickle unwrap。
- **新手概念课堂**：进程间传数据用 pickle 虽然方便，但有安全风险（反序列化可执行任意代码）、版本兼容性差、调试困难。改成强类型 dataclass/list 后，序列化更快、更安全、IDE 能补全。
- **对你有什么影响**：开启 metrics 时，调度器到 tokenizer 的计时数据传输开销降低；如果你写自定义 observability 插件，需要适配新的类型。

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)