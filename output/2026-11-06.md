# 🚀 vLLM & SGLang 每日更新 — 2026-11-06

> 自动生成于 2026-11-06 08:28 UTC | AI 解读: ✅ 含代码解读

## vllm
## 💡 今日关键词

- **关键词：Pipeline Parallel（流水线并行）下的权重共享**
  - 一句话通俗解释：把大模型切成几段放在不同 GPU 上，段与段之间要"交接棒"，但有些权重（如 embedding）必须每段都能访问，否则会算错。
  - 为什么社区在关注：今天有 2 个 commit（#65816、#65989）专门修 PP 下的 embedding 传递问题，说明 vLLM 正在大规模支持多卡切分大模型，PP 场景的边角 bug 成为高频修复对象。
  - 对新手意味着什么：如果你想学大模型推理，分布式并行（TP/PP/DP）是必修课，这类 bug 是最好的"源码阅读入口"。

- **关键词：Model Loader 重载 / RL 训练推理协同**
  - 一句话通俗解释：RLHF 训练时需要把新权重"热更新"到推理引擎里，不能重启进程，这叫 weight reload。
  - 为什么社区在关注：今天有 3 个 commit（#64352、#65235、#65914）围绕 reload 的 padding 重置、load plan、finish 流程，说明 vLLM 正在把"训练-推理一体化"做成核心能力。
  - 对新手意味着什么：RL + 推理是当下最热的方向之一（GRPO/PPO 训练框架都依赖它），值得投入。

- **关键词：多模态（Multimodal）输入处理**
  - 一句话通俗解释：模型能同时吃文字、图片、音频、视频，但要先把它们"翻译"成 token。
  - 为什么社区在关注：今天 5 个 commit（#53369、#66017、#60314、#64831、#66029）都在修多模态相关 bug，说明多模态是 vLLM 最活跃的功能线。
  - 对新手意味着什么：多模态是 LLM 应用落地的主战场（OCR、视频理解、语音助手），学它不愁没项目做。

- **关键词：异构硬件适配（XPU / ROCm / SM120）**
  - 一句话通俗解释：让同一份代码在 Intel GPU、AMD GPU、NVIDIA 新架构上都能跑。
  - 为什么社区在关注：今天有 5 个 commit 是 XPU/ROCm/SM120 相关（#66039、#66046、#66029、#65971、#65937），说明 vLLM 正从"NVIDIA 专属"走向"全硬件支持"。
  - 对新手意味着什么：如果你手头只有 AMD 卡或 Intel 卡，现在正是参与开源贡献的好时机，CI 修复类 PR 门槛低。

- **关键词：结构化输出 + 投机解码的交互**
  - 一句话通俗解释：强制模型输出 JSON/正则格式，同时又用"投机解码"加速，两者叠加时容易出 bug。
  - 为什么社区在关注：#65710 修复了 EOS 在投机批次中的处理，说明 vLLM 正在打磨"性能优化 × 功能正确性"的交叉地带。
  - 对新手意味着什么：这类"两个 feature 叠加出 bug"的场景是理解系统设计的最佳教材。

---

## 🧪 重要知识点 & 动手实验

### 知识点 1：Pipeline Parallel 下的 embedding 共享（来自 #65816、#65989）

- **是什么**：大模型切成多段放多卡时，第一段算出的 hidden_states 要传给第二段。但有些模型（如 Gemma4 的 per-layer embeddings、DeepSeek-V4 的 MTP drafter）需要额外的 embedding 张量，这些张量也得跟着"交接棒"传下去。
- **为什么重要**：如果不传，后面的 stage 会拿到 `None` 或旧值，导致输出错乱但不报错（silent bug），非常难排查。今天两个 commit 都是这个坑：一个用 `is not None` 判断（错，因为传过来的是空 dict），一个忘了声明 `has_own_embed_tokens = False`。
- **动手试试**：
  ```bash
  # 用 2 卡跑 Gemma4 的 PP 测试（需要 2 张 GPU）
  pytest tests/models/language/generation/test_gemma.py::test_gemma4_per_layer_embeddings_pipeline_parallel -v
  ```
  或者直接读代码理解修复逻辑：
  ```python
  # vllm/model_executor/models/gemma4.py 修复前
  if per_layer_inputs is not None:          # ❌ 空 dict 也非 None
      per_layer_inputs = intermediate_tensors["per_layer_inputs"]
  # 修复后
  if "per_layer_inputs" in intermediate_tensors.tensors:  # ✅ 检查 key 是否存在
      per_layer_inputs = intermediate_tensors["per_layer_inputs"]
  ```
- **预期结果**：PP=2 和 PP=1 的 logprobs 应该几乎一致（`check_logprobs_close` 通过）。你会理解"分布式传参时，判断存在性要用 key 而不是 is not None"。

### 知识点 2：CUTLASS GEMM 的 epilogue bias 融合（来自 #61821）

- **是什么**：矩阵乘法 `D = A × B + bias`，传统做法是先算 `A×B` 再单独加 bias（多一次内存读写）。CUTLASS 的 epilogue 允许在输出阶段"顺手"把 bias 加上，省一次 kernel launch 和显存往返。
- **为什么重要**：在 NVFP4（4-bit 浮点）量化推理里，GEMM 是绝对热点，epilogue 融合能省 5-15% 的延迟。今天这个 PR 把 bias 支持加进了 SM120（Blackwell 消费级架构）的 NVFP4 kernel。
- **动手试试**：
  ```python
  # 伪代码：对比有无 bias 的调用签名
  # 修复前
  cutlass_scaled_fp4_mm(D, A, B, A_sf, B_sf, alpha)
  # 修复后（多一个可选 bias 参数）
  cutlass_scaled_fp4_mm(D, A, B, A_sf, B_sf, alpha, bias=None)
  # SM100 上会显式报错：bias 不支持
  STD_TORCH_CHECK(!bias.has_value(), "bias epilogue not supported on SM ", sm)
  ```
  实际跑测试（需要 SM120 GPU）：
  ```bash
  pytest tests/kernels/quantization/test_nvfp4_scaled_mm.py -v -k sm120
  ```
- **预期结果**：带 bias 和不带 bias 的 GEMM 结果数值一致，但带 bias 版本少一次 kernel。你会理解"epilogue fusion"是量化推理优化的核心手段之一。

### 知识点 3：结构化输出 + 投机解码的 EOS 处理（来自 #65710）

- **是什么**：结构化输出（如正则 `a+`）要求模型只能生成合法 token。投机解码会一次"猜"多个 token 再验证。当猜出的批次里出现 EOS（结束符）时，语法校验器必须"只验到 EOS 为止"，后面的 token 不能算数。
- **为什么重要**：如果校验器把 EOS 后面的 token 也验了，会误判为非法，导致本应结束的序列被强行继续，输出乱码。这是"两个优化叠加出 bug"的经典案例。
- **动手试试**：
  ```python
  # 直接跑单元测试看行为
  pytest tests/v1/structured_output/test_backend_guidance.py::test_backend_guidance_eos_in_token_batch -v
  ```
  核心断言：
  ```python
  tokens = valid + invalid + eos + invalid   # 合法 + 非法 + EOS + 非法
  assert grammar.accept_tokens("", tokens) is False  # 前面有非法，整体拒绝
  # 但 EOS 之后的非法 token 不应影响"已终止"判断
  assert grammar.accept_tokens("", invalid + eos + invalid)  # EOS 后不再校验
  ```
- **预期结果**：测试通过，你会理解"语法校验器在遇到 EOS 后必须短路"这一设计原则。

### 知识点 4：vLLM 的 weight reload 生命周期（来自 #64352、#65914）

- **是什么**：RL 训练时，训练框架不断把新权重推给推理引擎。vLLM 的 reload 分三步：`start_reload`（准备）→ 加载权重 → `finish_reload`（收尾，跑 post-load hook 重建派生状态）。
- **为什么重要**：MoE 层为了对齐硬件会把 expert 的 intermediate_size 向上取整（如 256→384），多出来的 padding 必须在每次 reload 开始时清零，否则上一轮的脏数据会污染下一轮。今天 #64352 就是修这个。
- **动手试试**：
  ```bash
  pytest tests/kernels/moe/test_fp8_tp_loading.py::test_direct_reload_after_dummy_start_resets_padding -v
  ```
  关键逻辑：
  ```python
  # 第一次用 dummy 权重（全 7.0）启动
  layer.e_score_correction_bias.fill_(7)
  record_checkpoint_layout(model)
  begin_direct_reload(model)   # ← 这里会清零 padding
  # 然后加载真实 checkpoint，结果应与"直接加载真实权重"完全一致
  ```
- **预期结果**：dummy 启动后 reload 的结果 == 直接加载的结果。你会理解"热更新时必须重置临时状态"这一通用工程原则。

### 知识点 5：Anthropic count_tokens 的 max_model_len 语义（来自 #65697）

- **是什么**：Anthropic API 有个 `/v1/messages/count_tokens` 端点，客户端用它判断 prompt 是否超长。vLLM 之前会直接拒绝超长 prompt，但 count_tokens 的语义应该是"照常计数，不拒绝"。
- **为什么重要**：这是 API 兼容性的细节——客户端靠 count_tokens 做前置检查，如果它报错，客户端就没法优雅降级。修复方式是给 count_tokens 单独构造一个 `TokenizeParams(max_total_tokens=None)`，绕过长度校验。
- **动手试试**：
  ```python
  # 启动 vLLM 的 Anthropic 兼容服务
  vllm serve facebook/opt-125m --api-key test
  # 用 curl 测超长 prompt
  curl -X POST http://localhost:8000/v1/messages/count_tokens \
    -H "x-api-key: test" -H "content-type: application/json" \
    -d '{"model":"claude-3-7-sonnet-latest","messages":[{"role":"user","content":"word word word ...(3000次)"}]}'
  ```
- **预期结果**：count_tokens 返回一个大于 `max_model_len` 的数字（不报错），但 `/v1/messages` 创建请求时会返回 400。你会理解"计数接口"和"生成接口"的语义差异。

---

## 🔥 重要更新

1. **#65816 + #65989：Pipeline Parallel 下的 embedding 传递修复** — 影响所有用 PP 跑 DeepSeek-V4 MTP 和 Gemma4 的用户，之前会静默算错，现在 PP=2 与 PP=1 结果对齐。
2. **#64352 + #65914：Model Loader reload 生命周期统一** — RL 训练场景的核心修复，MoE padding 清零 + 统一 finish 流程，避免热更新后模型输出漂移。
3. **#61821：SM120 NVFP4 GEMM 支持 bias epilogue** — Blackwell 消费级 GPU 上量化推理性能提升，是硬件适配 + kernel 优化的交叉成果。
4. **#65826：新增 LFM3-Text 模型支持** — LiquidAI 的新模型家族，同时扩展了 causal_conv1d kernel 支持 width 5/6/7，说明 vLLM 对非 Transformer 架构（Mamba 系）的支持在加深。

---

## 📋 逐条解读

### 1. [XPU][CI] Correct deselected file path in entrypoints/scale_out test (#66039)
- **代码层面**：改的是 `.buildkite/intel_jobs/entrypoints_intel.yaml`，把 pytest 的 `--deselect` 路径从 `entrypoints/scale_out/...` 改成 `tests/entrypoints/scale_out/...`。因为 pytest 在 `cd tests` 之后运行，deselect 需要相对于仓库根目录的完整路径。
- **新手概念课堂**：就像你告诉快递员"别送 3 号楼 201"，但你只说了"3 号楼 201"，快递员在小区里找不到——必须说"阳光小区 3 号楼 201"。pytest 的 deselect 需要"从仓库根算起"的完整路径。
- **对你有什么影响**：Intel XPU 的 CI 不再误跑那两个已知失败的测试，CI 更稳定。普通用户无感知。

### 2. [Bugfix][Frontend] Match default MM LoRA for OpenAI input_audio parts (#53369)
- **代码层面**：改的是 `vllm/entrypoints/serve/engine/serving.py` 的 `_get_message_types`。原来对 `input_audio` 类型会 `split("_")[0]` 得到 `"input"`，导致默认多模态 LoRA（按 `"audio"` 匹配）匹配不上。修复后对 `input_*` 前缀做 `removeprefix("input_")`。
- **新手概念课堂**：就像你的通讯录按"姓"分组，来了个复姓"欧阳"的人，程序却按第一个字分到了"欧"组。修复是识别"复姓"规则。
- **对你有什么影响**：用 OpenAI 格式的 `input_audio` 传音频 + 配置了默认多模态 LoRA 时，LoRA 现在能正确生效了。

### 3. [ROCm][CI] Skip the unlimited OCR tests until the ROCm CK OCR2 DASH updates land (#66046)
- **代码层面**：在两个测试文件里加 `pytestmark = pytest.mark.skipif(current_platform.is_rocm(), ...)`，跳过 ROCm 上的 Unlimited-OCR-2 测试，因为 DASH attention 目前是 CUDA-only。
- **新手概念课堂**：就像某款游戏只在 Windows 上有，Mac 用户点开就闪退。与其让测试红着，不如明确标记"暂不支持"。
- **对你有什么影响**：AMD 用户暂时不能用 Unlimited-OCR-2 模型，等 #66031 落地后恢复。

### 4. [Bugfix][Qwen3-Omni] Size the video encoder budget from the processor's video limits (#66017)
- **代码层面**：改的是 Qwen3-Omni 的多模态 processor，让视频 token 预算从 processor 的 `videos_kwargs`（如 `size.shortest_edge`）推导，而不是写死。新增了多个测试验证 budget 能撑满 `max_model_len`。
- **新手概念课堂**：就像你去自助餐厅，盘子大小（budget）要按餐厅规定（processor 配置）来，不能自己随便拿个大盆。
- **对你有什么影响**：Qwen3-Omni 处理长视频时不再因为预算算错而 OOM 或截断。

### 5. [Bugfix][DeepSeek-V4] Share the target embedding with the MTP drafter under PP (#65816)
- **代码层面**：在 `DeepSeekV4MTP` 类里加 `has_own_embed_tokens = False`，告诉框架"我没有自己的 embedding，用 target 的"。PP 场景下最后一个 rank 会保留一份 target embedding 给 drafter 用。
- **新手概念课堂**：就像接力赛，最后一棒选手（drafter）没有自己的接力棒（embedding），必须从前一棒（target）那里借一根。
- **对你有什么影响**：用 PP 跑 DeepSeek-V4 + MTP 投机解码时，输出正确了（之前会因缺 embedding 而算错）。

### 6. [Bugfix][Gemma4] Read per-layer embeddings on later pipeline stages (#65989)
- **代码层面**：`gemma4.py` 里把 `if per_layer_inputs is not None` 改成 `if "per_layer_inputs" in intermediate_tensors.tensors`。因为 PP 传过来的可能是空 dict（非 None），原判断会误以为有值。
- **新手概念课堂**：你收到一个快递盒，判断"有没有东西"不能看盒子是不是空的（盒子本身存在），要看里面有没有物品（key 是否存在）。
- **对你有什么影响**：Gemma4 用 PP=2 跑时，第二段能正确拿到第一段传来的 per-layer embeddings，输出与单卡一致。

### 7. [Bugfix][Frontend] Count prompts over max_model_len in Anthropic count_tokens (#65697)
- **代码层面**：新增 `_CountTokensChatRequest` 类，覆写 `build_tok_params` 返回 `max_total_tokens=None`，让 count_tokens 端点不校验长度。同时补了测试验证超长 prompt 能计数但生成会 400。
- **新手概念课堂**：就像体重秤不会因为"你太重"而拒绝称重，它只负责报数；但电梯会因为超重而不关门。
- **对你有什么影响**：用 Anthropic SDK 的 `client.messages.count_tokens` 时，超长 prompt 能正常返回 token 数，方便客户端做前置判断。

### 8. [Bugfix][Structured Output] Handle EOS in speculative Guidance batches (#65710)
- **代码层面**：改的是 Guidance 后端的语法校验逻辑，当批次中出现 EOS 时，只校验到 EOS 为止，后面的 token 不再参与合法性判断。新增了两个测试覆盖 valid/invalid 前缀 + EOS 的组合。
- **新手概念课堂**：就像考试时老师说"写到'完'字就停笔"，你后面多写的草稿不算分，也不该被判作弊。
- **对你有什么影响**：用 guidance 后端 + ngram 投机解码时，`ignore_eos=True` 的强制长度生成不再因为 EOS 后的 token 被误判而崩溃。

### 9. [Perf][Kernel] Add the bias in the SM120 NVFP4 CUTLASS GEMM epilogue (#61821)
- **代码层面**：给 `cutlass_scaled_fp4_mm` 加 `std::optional<Tensor> bias` 参数，SM120 的 kernel 模板加 `HasBias` 布尔参数，在 epilogue 里融合 bias 加法。SM100 上显式 `TORCH_CHECK` 拒绝 bias（不支持）。
- **新手概念课堂**：就像做菜时"炒菜和撒盐"分开做（两次下锅）vs 出锅前顺手撒盐（一次完成），后者省时省力。
- **对你有什么影响**：Blackwell 消费级 GPU（SM120）上跑 NVFP4 量化模型时，带 bias 的线性层延迟降低。

### 10. [Doc][Bugfix] Fix strict docs build broken by unresolved autorefs (#66044)
- **代码层面**：给几个类/函数加 docstring（`TrtLlmMxfp4ExpertsModular`、`store_partial` 等），修复 Sphinx 严格模式下的 autoref 解析失败。
- **新手概念课堂**：就像图书馆索引里引用了"某本书第 3 章"，但那本书没写章节标题，索引就断了。补上标题即可。
- **对你有什么影响**：文档构建不再失败，CI 绿灯。

### 11. [Model Loader] Direct reload: reset MoE expert padding at begin (#64352)
- **代码层面**：在 `begin_direct_reload` 里增加逻辑：根据 `record_checkpoint_layout` 记录的原始 padding 信息，把 MoE expert 的 padding 区域清零。新增测试验证 dummy 启动后 reload 的结果与直接加载一致。
- **新手概念课堂**：就像你租了个带储物间的房子，上个租客留了一堆垃圾（padding 脏数据），你搬进去第一件事是清空储物间。
- **对你有什么影响**：RL 训练中反复 reload 权重时，MoE 模型不再因为 padding 残留而输出漂移。

### 12. [Bugfix][Benchmark] Clean up sweep workers after the server exits (#60010)
- **代码层面**：改的是 `vllm/benchmarks/sweep/server.py` 的 `ServerProcess`，停止时不仅杀 server 主进程，还要杀整个进程组（包括 server 已退出后残留的 worker）。新增测试用 fork + socket 验证。
- **新手概念课堂**：就像关灯时不仅要关主灯，还要关掉主灯开关控制的所有射灯——只关主灯，射灯还亮着。
- **对你有什么影响**：跑 benchmark sweep 时不再残留僵尸进程占 GPU 显存。

### 13. [Model] Support data-parallel multimodal encoder for Gemma-4 (#60314)
- **代码层面**：给 Gemma4 的多模态 encoder 加 DP（data parallel）支持：新增 `_process_image_input_local` / `_process_video_input_local` 方法，用 `tensor_model_parallel_all_gather` 和 `all_reduce` 在 TP 组间同步，并用 `get_load_balance_assignment` 做负载均衡。
- **新手概念课堂**：就像处理 100 张图片，一个人处理太慢，分给 4 个人并行处理（DP），最后汇总结果。但每个人处理前要知道自己分到哪几张（load balance）。
- **对你有什么影响**：Gemma4 多模态模型在多卡上推理时，图像/视频 encoder 能并行加速，吞吐提升。

### 14. [RL][Model Loader] Load plan: plan_loads and TraceTensor (#65235)
- **代码层面**：新增 `plan_loads` 函数和 `TraceTensor` 机制：在第一次 weight update 时"预演"一遍加载流程，生成加载计划（哪些参数怎么切分、怎么传输），后续 update 直接按计划执行。失败时回退到 `load_weights`。
- **新手概念课堂**：就像搬家前先画一张"家具摆放图"（plan），之后每次搬新家具都按图放，不用每次重新规划。图画错了就退回"现场随便放"（fallback）。
- **对你有什么影响**：RL 训练中频繁 weight update 时，加载速度更快（避免重复计算切分方案），且失败有兜底。

### 15. [Bugfix][Benchmark] Fix bench mm-processor crash with HF datasets (#64831)
- **代码层面**：在 `_to_serve_args` 里加 `d.setdefault("hf_name", None)`，因为 `get_samples` 读 `args.hf_name` 时 bench mm-processor 没这个属性，会 AttributeError。
- **新手概念课堂**：就像你调用一个函数，它内部读了 `config.debug`，但你的 config 对象没这个字段——补个默认值 `None` 就好。
- **对你有什么影响**：用 `vllm bench mm-processor --dataset-name hf` 时不再崩溃。

### 16. [Model Loader] One finish for every update path; drop refresh_derived_weights (#65914)
- **代码层面**：删除 `refresh_derived_weights` 函数，所有 reload 路径统一走 `finalize_model_weights`（跑 post-load hook 重建派生状态）。`inplace.py` 里用 `weights_already_processed()` 上下文避免重复处理。
- **新手概念课堂**：以前有两条"收尾流程"（A 和 B），容易漏掉一条；现在合并成一条，所有路径都走它，不会遗漏。
- **对你有什么影响**：不同 reload 方式（inplace / direct / sparse NCCL）的行为一致，减少"某种方式下模型输出不对"的玄学 bug。

### 17. [Model] Add LFM3-Text (#65826)
- **代码层面**：新增 `Lfm3ForCausalLM` 模型实现，注册到 `supported_models.md`。同时扩展 `causal_conv1d` kernel 支持 width 5/6/7（LFM3 用到了非 4 的卷积核宽度），并加了 width 1/8 的显式报错。
- **新手概念课堂**：就像游戏新增一个角色，同时要把它的特殊技能（width 5/6/7 卷积）加进引擎，否则技能放不出来。
- **对你有什么影响**：可以直接用 vLLM 跑 LiquidAI 的 LFM3 系列模型了。

### 18. [ROCm][Kernel] Follow-ups to the gfx942 MLA Triton verify kernel (#65971)
- **代码层面**：简化 `verify_num_kv_splits` 的调用（去掉 `select_verify_config` 的重复计算），修复 int64 page offset 测试在 16/128 heads 下的覆盖，并给 AITER 的 FP8 V-scale bug 加注释（ROCm/aiter#6732）。
- **新手概念课堂**：就像修水管时发现"这里算了两遍水压"，删掉重复计算；同时给已知的漏水点贴个标签"等厂商修"。
- **对你有什么影响**：AMD gfx942 上 MLA 投机解码的 verify kernel 更稳定，大 cache（>2GB）不再 int32 溢出。

### 19. [ROCm] Bump AITER to v0.1.28.post1 (#65937)
- **代码层面**：`Dockerfile.rocm_base` 和 `Dockerfile.rocm_base_gfx1250` 里 `AITER_BRANCH` 从 `v0.1.27` / `f85b24b` 升到 `v0.1.28.post1`。
- **新手概念课堂**：就像手机系统从 iOS 17.1 升到 17.2，修了 bug 加了功能。
- **对你有什么影响**：AMD 用户用最新 Docker 镜像时，AITER 相关 kernel 的 bug 修复生效。

### 20. [XPU] increase gpu_memory_utilization for test_granite4_vision.py (#66029)
- **代码层面**：把 `test_granite4_vision.py` 里的 `gpu_memory_utilization` 从 0.4 提到 0.9，因为同一进程里跑两个 engine，0.4 不够用。
- **新手概念课堂**：就像一台电脑同时开两个大型游戏，每个只分配 40% 显存不够，得给到 90%。
- **对你有什么影响**：Intel XPU 上 Granite4-Vision 的 CI 不再 OOM 失败。普通用户无感知。

## sglang
## 💡 今日关键词

- **关键词：FlashAttention 依赖"去第三方化"**
  - 一句话解释：SGLang 把原来依赖外部 `flash-attn` / `kernels-hub` 下载的 FA3/FA4 内核，改成仓库内自带的 vendored 版本，并从默认依赖里删掉 `flash-attn-4`。
  - 为什么社区在关注：这说明推理框架正在把"关键性能内核"从外部包收回自维护，避免上游发版节奏、ABI 兼容、aarch64 无预编译包等问题拖累。这是"框架自研内核 + 多硬件适配"的大趋势。
  - 对新手意味着什么：如果你在学 CUDA/Triton 内核开发，这类"框架自带内核"的岗位需求会越来越多；同时也要习惯读 `sglang/kernels/ops/...` 这种内部路径，而不是只认 PyPI 包。

- **关键词：新模型快速接入（ERNIE 5.5 Flash）**
  - 一句话解释：一个 PR 同时加上了模型结构、reasoning parser、tool-call parser、专用 attention backend（`ernie55`）、FP4 indexer 开关。
  - 为什么社区在关注：模型厂商发布节奏越来越快，推理框架的"接入速度"成为竞争力。ERNIE 5.5 Flash 用了 DSA 稀疏注意力 + MLA，说明国产模型也在走 DeepSeek 那条稀疏化路线。
  - 对新手意味着什么：学会"读 config → 找 attention 结构 → 加 parser → 加 backend"这套流水线，是模型适配岗的核心技能。

- **关键词：量化 KV Cache 的一致性**
  - 一句话解释：FP4 MLA 的 KV 数据搬移时，必须把 block scale 一起搬，否则反量化结果全错；PD 分离时还要校验两端 KV page stride 是否一致。
  - 为什么社区在关注：量化越激进（FP8→FP4），"数据 + 缩放因子"成对搬移的正确性越容易出 bug，且这类 bug 往往表现为"输出乱码"而非崩溃，极难排查。
  - 对新手意味着什么：做量化相关工作时，"scale 和 data 必须同生命周期"要刻进肌肉记忆。

- **关键词：平台插件化（weight update / AMD monokernel）**
  - 一句话解释：新增 `/plugin_weight_update` 接口，让不同硬件平台用自己的传输方式（而不是只能走 torch.distributed）来更新权重；AMD 侧则通过 monokernel 把 GLM-5.3 的低延迟推理跑起来。
  - 为什么社区在关注：异构硬件（NVIDIA / AMD / Intel / NPU）已是常态，框架必须提供"平台可插拔"的抽象层，而不是到处写 `if is_amd`。
  - 对新手意味着什么：理解"抽象接口 + 平台实现"的设计模式，比背某个具体 API 更有长期价值。

- **关键词：启动速度与依赖瘦身**
  - 一句话解释：把 `LazyImport` 从 `sglang.utils` 抽到 `sglang._lazy_import`，并拆分 `utils/common.py`，让 `import sglang` 不再拖上一堆重模块。
  - 为什么社区在关注：推理框架的 import 时间直接影响容器冷启动、CI 时长、CLI 响应。这是"大型 Python 项目工程化"的典型动作。
  - 对新手意味着什么：学会用 `python -X importtime` 分析 import 热点，是很实用的工程技能。


## 🧪 重要知识点 & 动手实验

- **知识点 1：量化 KV Cache 中"数据 + scale 必须成对搬移"**
  - **是什么**：FP4/FP8 量化把每个数据块压缩成"4bit/8bit 数据 + 一个 scale 因子"。读取时要用 scale 还原。所以搬数据时必须把 scale 一起搬。
  - **为什么重要**：如果只搬数据不搬 scale，目标位置的旧 scale 会去反量化新数据，结果数值完全错误但不报错——表现为模型输出乱码、精度暴跌，排查成本极高。
  - **动手试试**（模拟 Commit 15 的核心逻辑）：
    ```python
    import torch

    # 模拟：2 个 layer 的 FP4 KV buffer 和对应 scale buffer
    num_slots, block = 8, 4
    kv    = [torch.arange(num_slots * block).reshape(num_slots, block) + 100 * i for i in range(2)]
    scale = [torch.full((num_slots,), 1.0 + i) for i in range(2)]

    src = torch.tensor([5, 6])
    dst = torch.tensor([1, 2])

    # 正确做法：data 和 scale 一起搬
    for i in range(2):
        kv[i][dst]    = kv[i][src].clone()
        scale[i][dst] = scale[i][src].clone()

    print("data :", kv[0][dst])
    print("scale:", scale[0][dst])
    ```
  - **预期结果**：`data` 和 `scale` 都来自 src 位置。如果你注释掉 `scale[i][dst] = ...` 那行，你会发现 data 变了但 scale 还是 dst 原来的值——这正是 Commit 15 修的 bug。

- **知识点 2：PD 分离中"两端内存布局必须一致"**
  - **是什么**：PD 分离（Prefill/Decode 分离）时，prefill 节点把 KV cache 通过网络写到 decode 节点的内存里，寻址方式是 `dst_ptr + page * item_len`，其中 `item_len` 用的是 prefill 自己算的。
  - **为什么重要**：如果两端 `item_len`（页步长）不同，每一页都会写到错误偏移，最后几页甚至越界写坏内存，表现为"偶发崩溃"或"解码乱码"，且只在特定配置下复现。
  - **动手试试**（模拟 Commit 16 的校验思路）：
    ```python
    def check_stride(prefill_item_len, decode_item_lens):
        # 简化版：每层 item_len 必须一致
        for layer_id, d in enumerate(decode_item_lens):
            if d != prefill_item_len:
                return f"layer {layer_id}: prefill={prefill_item_len} decode={d} MISMATCH"
        return "OK"

    print(check_stride(128, [128, 128, 128]))   # OK
    print(check_stride(128, [128, 256, 128]))   # 第 1 层不一致
    ```
  - **预期结果**：第二种情况会明确报出哪一层不匹配。真实代码里这个检查发生在 Mooncake 注册阶段，早失败早报错，比等到写坏内存好得多。

- **知识点 3：`contextmanager` 必须正确恢复全局状态**
  - **是什么**：Python 的 `@contextlib.contextmanager` 用来"进入时改状态、退出时恢复"。但如果恢复逻辑写错（比如只恢复了一半变量），退出后全局状态就"脏"了。
  - **为什么重要**：batch-invariant 模式会替换 `torch.bmm` 等全局算子。如果退出时没恢复，进程里后续所有 `torch.bmm` 都变成慢速/非标准实现，性能悄悄下降且没人发现。
  - **动手试试**（模拟 Commit 6 的 bug 与修复）：
    ```python
    import contextlib

    STATE = {"mode": False, "bmm": "native"}

    @contextlib.contextmanager
    def buggy(enabled=True):
        old = dict(STATE)
        STATE["mode"] = enabled
        STATE["bmm"] = "invariant" if enabled else "native"
        try:
            yield
        finally:
            STATE["mode"] = old["mode"]   # BUG: 忘了恢复 bmm
            # STATE["bmm"] = old["bmm"]

    @contextlib.contextmanager
    def fixed(enabled=True):
        old = dict(STATE)
        STATE["mode"] = enabled
        STATE["bmm"] = "invariant" if enabled else "native"
        try:
            yield
        finally:
            STATE.update(old)             # 一次性恢复

    with buggy(True):
        pass
    print("after buggy:", STATE)          # bmm 没恢复！

    STATE.update({"mode": False, "bmm": "native"})
    with fixed(True):
        pass
    print("after fixed:", STATE)          # 全部恢复
    ```
  - **预期结果**：`after buggy` 里 `bmm` 仍是 `"invariant"`，`after fixed` 里是 `"native"`。这就是 Commit 6 的核心修复思路：用"记录快照 + 整体恢复"代替逐字段手工恢复。

- **知识点 4：用 `LazyImport` 加速 `import`**
  - **是什么**：把 `from sglang.utils import LazyImport` 换成仓库内的 `sglang._lazy_import.LazyImport`，让 `import sglang` 时不加载重型后端模块，只在第一次真正调用时才 import。
  - **为什么重要**：大型框架的 `import` 时间可能到几秒，直接影响容器冷启动、CI、CLI 响应。而且顶层 import 重型模块会拖慢所有下游使用者。
  - **动手试试**（模拟 Commit 7 的类）：
    ```python
    # lazy_demo.py
    import importlib, time

    class LazyImport:
        def __init__(self, module_name, class_name):
            self.module_name = module_name
            self.class_name = class_name
            self._module = None
        def _load(self):
            if self._module is None:
                m = importlib.import_module(self.module_name)
                self._module = getattr(m, self.class_name)
            return self._module
        def __getattr__(self, name):
            return getattr(self._load(), name)
        def __call__(self, *a, **kw):
            return self._load()(*a, **kw)

    t0 = time.time()
    Heavy = LazyImport("json.decoder", "JSONDecoder")  # 假装是重模块
    print(f"定义耗时: {time.time()-t0:.4f}s")           # 几乎为 0
    t0 = time.time()
    _ = Heavy()                                         # 第一次调用才真正 import
    print(f"首次调用耗时: {time.time()-t0:.4f}s")
    ```
  - **预期结果**：定义耗时接近 0，首次调用才付出 import 成本。可以用 `python -X importtime -c "import sglang"` 对比优化前后。

- **知识点 5：性能调优 = 查表 + 按硬件/形状分派**
  - **是什么**：Commit 12/14 都是"为特定 GPU（SM107 / H20）、特定形状（M, N, K）预先测好最优 GEMM 参数，运行时查表分派"。
  - **为什么重要**：GPU 上同一个矩阵乘，不同的 block size / split-K / warp 数组合性能差几倍。自动调优（autotune）在启动时做太慢，因此社区常用"离线调优 + 在线查表"。
  - **动手试试**（模拟 Commit 12 的查表逻辑）：
    ```python
    TUNED = {
        (1, 6144, 4096): (32, 8, 256, 11, 2),
        (4, 6144, 4096): (32, 8, 256, 11, 2),
        (20, 6144, 4096): (32, 32, 256, 6, 2),
    }
    DEFAULT = (32, 16, 256, 9, 2)

    def pick_tactic(m, n, k, device="SM107"):
        if device == "SM107":
            return TUNED.get((m, n, k), DEFAULT)
        return DEFAULT

    for m in (1, 4, 20, 7):
        print(m, "->", pick_tactic(m, 6144, 4096))
    ```
  - **预期结果**：1/4/20 命中调优表，7 落到默认值。真实代码里还会把查表结果 JIT 预编译，避免 CUDA graph capture 时冷启动。


## 🔥 重要更新

- **#48547 支持 ERNIE 5.5 Flash**：一个 PR 打通模型结构、reasoning parser、tool-call parser、`ernie55` attention backend 和 FP4 indexer 开关，是今天最完整的"新模型接入"范例，值得对照学习。
- **#48853 + #48829 移除 flash-attn 外部依赖**：FA3/FA4 改为仓库自带内核，Docker 构建和默认依赖都瘦身。这会影响所有依赖 FA 的用户，尤其是 aarch64 用户（以前要运行时 JIT）。
- **#48433 新增 `/plugin_weight_update`**：权重更新不再绑定 torch.distributed，平台可以自定义传输方式，是异构硬件支持的关键抽象。
- **#44437 + #48231 两个量化 KV 正确性修复**：FP4 MLA scale 搬移 + PD 分离 page stride 校验，都是"不修就出不报错的错结果"的高危 bug，值得所有做量化和 PD 分离的人读。


## 📋 逐条解读

### 1. [Chore] flash-attn: Remove the kernels-hub FA3 loader and the kernels dependency (#48853)
- **代码层面**：改 `docker/Dockerfile` 和 `Dockerfile.cu134`，删掉 `kernels lock python`、`kernels download python`、aarch64 跳过分支、`kernels.lock` 搬迁等一大段 shell；同步更新 `environment_variables.mdx` 里相关环境变量说明。
- **新手概念课堂**：以前 Docker 构建时会从 kernels-hub 下载预编译好的 FA3 cubin（GPU 二进制），就像"去应用商店下载现成的 App"。现在改成"源码自带内核，构建时自己编译"，好处是不依赖外部商店、版本可控，代价是构建时间可能变长。
- **对你有什么影响**：如果你自己 build SGLang 镜像，构建脚本变短了，但不再有"预编译 cubin 兜底"，aarch64 上会走运行时 JIT（首次推理慢一点）。

### 2. [Chore] deps: Drop flash-attn-4 from default dependencies (#48829)
- **代码层面**：从 `python/pyproject.toml` 的 `dependencies` 删掉 `flash-attn-4>=4.0.0b18`；修改 `flash_attention_v4.py` / `flash_attention_v4_sm120.py` 的 ImportError 提示；`multiview_attention.py` 里把 `import flash_attn.cute` 改成 `import sglang.kernels.ops.attention.flash_attn.cute`。
- **新手概念课堂**：`pip install sglang` 不再自动装 `flash-attn-4`，因为 SGLang 现在自带一份 vendored 的 FA4 CuTe 内核。就像你以前要单独买一个工具箱，现在工具箱已经焊在车里了。
- **对你有什么影响**：升级后如果代码里还 `import flash_attn`，可能报 ImportError；需要改成从 `sglang.kernels...` 导入。装包体积变小，但 FA4 功能仍可用（前提是 `nvidia-cutlass-dsl` 等底层依赖在）。

### 3. [Feat] model: Support ERNIE 5.5 Flash (#48547)
- **代码层面**：`server_arguments.mdx` 的 `--reasoning-parser` / `--tool-call-parser` / `--attention-backend` 列表都加了 `ernie55`；新增 `--enable-fp4-indexer` 参数说明；`indexer_k.cuh` 里把 DSA indexer 的 kernel 泛化，加了 `kHadamard` / `kNeoxStyle` 模板参数，从"DeepSeek-V3.2 专用"变成"通用 DSA indexer"。
- **新手概念课堂**：DSA（DeepSeek Sparse Attention）是"只挑 top-k 个关键 token 做注意力"的稀疏注意力。ERNIE 5.5 Flash 也用了这套结构，所以 SGLang 把原来写死给 DeepSeek 的 kernel 泛化，让两个模型共用。就像把"只能插一种插头的插座"改成"万能插座"。
- **对你有什么影响**：可以直接 `--model-path` 加载 ERNIE 5.5 Flash，并用 `--reasoning-parser ernie55`、`--tool-call-parser ernie55` 解析输出；SM100 上还能开 `--enable-fp4-indexer` 省显存。

### 4. [Fix] rust-server: Publish 127.0.0.1 for an IPv6 loopback SGLANG_HOST_IP on an IPv4 wildcard bind (#48752)
- **代码层面**：`rust_server/server.py` 的 `_advertised_host` 里，当 bind 是 `0.0.0.0`、检测到的 IP 是 IPv6 loopback（`::1`）时，对外发布 `127.0.0.1` 而不是 `::1`；测试同步更新。
- **新手概念课堂**：`0.0.0.0` 是"监听所有 IPv4 地址"，`::1` 是"IPv6 的本机回环地址"。如果服务只监听 IPv4，却告诉别人"我在 `::1`"，别人就连不上。就像你说"我在 3 号楼"，但实际你只在 3 号楼的 1 层——地址对，但具体位置不对。
- **对你有什么影响**：多节点部署时，如果 `SGLANG_HOST_IP=::1` 且 bind 是 `0.0.0.0`，其他节点现在能正确拿到 `127.0.0.1` 而不是连不上的 `::1`。

### 5. [Fix] hicache: Do not count staged-prefetch deferrals while KV awaits a PD transfer (#48693)
- **代码层面**：`schedule_policy.py` 的 `PrefillAdder.__init__` 新增 `kv_awaiting_transfer` 参数；`_defer_staged_for_capacity_if_idle` 里把这个条件加进"不该强制重算"的判断；`scheduler.py` 传 `bool(self.disagg_prefill_transfer_queue)`。
- **新手概念课堂**：PD 分离的 prefill 节点算完 KV 后，要等 decode 节点来拉走。这段时间 KV 还占着显存。HiCache 的"staged prefetch"如果因为显存不够而延迟，会被误判为"空闲失败"，从而触发不必要的重算。修复就是告诉调度器："还有 KV 在等传输呢，别急着重算。"
- **对你有什么影响**：PD 分离 + HiCache 场景下，prefill 阶段不再因为"看起来空闲"而重复计算，吞吐和延迟都会改善。

### 6. [Fix] batch-invariant: restore the dispatcher state when a scope exits (#48061)
- **代码层面**：`batch_invariant_ops.py` 的 `set_batch_invariant_mode` 从"逐字段手工恢复"改成"记录 `previous_mode` / `previous_bmm_enabled`，退出时对比并按需 disable/enable"；新增测试文件 `test_batch_invariant_context.py`。
- **新手概念课堂**：batch-invariant 模式会替换 `torch.bmm` 等全局算子，保证不同 batch size 下结果一致。`contextmanager` 退出时必须恢复。以前的恢复逻辑漏掉了某些状态，导致退出后进程"中毒"。就像你借了别人的车，还车时忘了把座椅调回去。
- **对你有什么影响**：使用 batch-invariant 模式做推理后，退出该模式不会污染后续普通推理的性能和结果。

### 7. [Refactor] utils: Keep sglang.srt and sglang.kernels off sglang.utils (#48827)
- **代码层面**：新增 `python/sglang/_lazy_import.py` 存放 `LazyImport`；`sglang/__init__.py` 改为从这里导入；pre-commit 的 `check-srt-utils-imports` 规则扩展到 `python/sglang/(srt|kernels)/`；大量文件把 `from sglang.utils import ...` 改成从具体模块导入。
- **新手概念课堂**：`sglang.utils` 以前是个"万能工具箱"，谁都能 import，导致 `import sglang` 时被迫加载一堆重模块（torch、CUDA 等）。这次把工具拆到具体归属模块，并加 lint 规则防止回退。就像把"公共仓库"拆成"各部门自己的仓库"，减少无关搬运。
- **对你有什么影响**：`import sglang` 更快；但如果你以前 `from sglang.utils import LazyImport`，需要改成 `from sglang._lazy_import import LazyImport`。

### 8. [Feat] weight_update: Let platforms serve weight transports through /plugin_weight_update (#48433)
- **代码层面**：`http_server.py` 新增 `/plugin_weight_update` POST 接口；`io_struct.py` 新增 `PluginWeightUpdateReqInput` / `PluginWeightUpdateReqOutput`；通过 `SRTPlatform.get_weight_update_handlers()` 查找平台注册的 handler。
- **新手概念课堂**：以前更新权重只能走 `/update_weights_from_distributed`（基于 torch.distributed）。现在平台可以注册自己的传输方式（比如 RDMA、共享内存、自定义协议），HTTP 接口只负责转发。就像快递公司可以自选运输方式，收件口统一。
- **对你有什么影响**：如果你在非 NVIDIA 平台上做权重热更新，现在可以通过插件机制接入，不用改 SGLang 核心代码。

### 9. [Doc] [AMD] GLM-5.3: Enable MTP on MI355X and add the agentic recipe with the AMD monokernel (#46662)
- **代码层面**：`GLM-5.3.mdx` 新增 "MI355X FP8 deployment with the AMD monokernel" 章节，给出 TP8·EP1 的完整 `sglang serve` 命令；更新 AMD MTP 支持说明；`glm-5.3.jsx` 同步配置。
- **新手概念课堂**：MTP（Multi-Token Prediction）是"一次预测多个 token"的投机解码，能加速生成。monokernel 是 AMD 侧把多个小 kernel 融合成一个大 kernel 的优化。以前文档说 AMD 上 MTP 未验证，现在 MI355X 上验证通过了。
- **对你有什么影响**：用 MI355X 跑 GLM-5.3 时，可以照抄文档命令开 EAGLE MTP，低并发下延迟明显降低。

### 10. [Fix] nemotron: Accept transformers 5 layer types in NemotronHConfig (#47458)
- **代码层面**：`nemotron_h.py` 新增 `TRANSFORMERS_V5_LAYER_TYPE_ALIASES = {"linear_attention": "mamba", "full_attention": "attention"}` 和 `_normalize_layer_type_names`；`nano_nemotron_vl.py` 删掉重复的 `_nemotron_h_compatible_config`。
- **新手概念课堂**：transformers 5.13 把 config.json 里的层类型名从 `mamba` / `attention` 改成了 `linear_attention` / `full_attention`。SGLang 内部仍用旧名，所以要在读取时做别名映射。就像 API 改版后，老客户端要加一层兼容层。
- **对你有什么影响**：用新版 transformers 加载 Nemotron-H 系列模型不再报 "invalid layer type"。

### 11. [Fix] overlap: Skip deferred-relay flush when future_map is None (#48809)
- **代码层面**：`overlap_utils.py` 的 `resolve_forward_inputs` 签名从 `future_map: FutureMap` 改成 `Optional[FutureMap]`，并加 `if future_map is not None and ...` 判断。
- **新手概念课堂**：`future_map` 是 overlap 调度里管理"延迟 relay"的对象。独立 prefill 调用（如 LLaDA-Image 的条件阶段）没有 future_map，以前会 `AttributeError`，现在直接跳过 flush。
- **对你有什么影响**：多模态模型 LLaDA-Image 的独立 prefill 路径不再崩溃。

### 12. [Perf] [NVIDIA] glm5: Route small-M BF16 GEMMs to FlashInfer warp split-K with PDL on SM107 (#47354)
- **代码层面**：`unquant.py` 新增 `_BF16_SM107_WARP_SPLITK_TUNED_TACTICS` 调优表、`_BF16_SM107_FUSED_A_SPLITK_KEYS` 集合、`_use_sm107_splitk_tactics` 开关；`precompile_splitk_tactics` 支持只预编译指定 keys；新增 `prefer_bf16_splitk_over_fused_a`。
- **新手概念课堂**：decode 阶段 M（batch 里的 token 数）很小，普通 GEMM 效率低。warp split-K 把 K 维度切给多个 warp 并行算，再 reduce。PDL（Programmatic Dependent Launch）让相邻 kernel 重叠启动。SM107 是 Vera Rubin 架构。
- **对你有什么影响**：GLM-5 在 SM107（Vera Rubin NVL72）上 decode 更快，尤其是小 batch 场景；其他硬件不受影响。

### 13. [Test] [NPU] kimi_k3: run the decode perf case via the aisbench prefix mode (#48290)
- **代码层面**：测试文件里 `dataset_type` 从 `"gsm8k-shared-prefix"` 改成 `AISBENCHMARK_DATASET_GSM8K`；新增 `repeat_rate = 1`、`dp = 4`；去掉 `warmup_dp` / `warmup_shape` / `api_type`。
- **新手概念课堂**：这是测试用例的写法调整，用 aisbench 官方的 prefix 模式（`repeat_rate=1` 表示所有请求共享同一前缀，命中率接近 100%）来测 decode 延迟。以前是自定义的 shared-prefix 模式。
- **对你有什么影响**：普通用户无感知；做 NPU 性能测试的人可以参照这个用例写自己的 benchmark。

### 14. [Perf] [NVIDIA] quantization: Add tuned W8A8 block-FP8 GEMM configs for H20 (#41301)
- **代码层面**：新增多个 JSON 配置文件，如 `N=1280,K=5120,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[32, 32].json`，里面按 M 值（1, 8, 64, ...）记录最优的 `BLOCK_SIZE_M/N/K`、`GROUP_SIZE_M`、`num_warps`、`num_stages`、`SPLIT_K`。
- **新手概念课堂**：W8A8 block-FP8 是"权重 8bit、激活 8bit、按 32×32 块量化"的方案。不同 M 下最优 tile 大小不同，所以离线调优后存成表，运行时按 M 查。
- **对你有什么影响**：H20 上跑 W8A8 block-FP8 量化模型时，GEMM 自动走调优配置，吞吐提升。

### 15. [Fix] mem_cache: Move FP4 MLA block scales with their KV in `move_kv_cache` (#44437)
- **代码层面**：`memory_pool.py` 的 `MLATokenToKVPoolFP4` 新增 `move_kv_cache` 重写，在 `super().move_kv_cache` 之后，对每个 `kv_scale_buffer` 执行 `kv_scale[tgt] = kv_scale[src]`；新增测试 `test_mla_fp4_move_kv_cache_moves_block_scales_with_data`。
- **新手概念课堂**：FP4 MLA 的 KV 是"4bit 数据 + block scale"。`move_kv_cache` 用于把 KV 从一个 slot 搬到另一个（比如 prefix cache 命中后复用）。父类只搬数据，不搬 scale，导致反量化用错 scale。
- **对你有什么影响**：FP4 MLA 模型（如 DeepSeek 系列 FP4 版）在 prefix cache 复用、KV 搬移场景下输出正确，不再出现"复用后乱码"。

### 16. [Fix] disaggregation: Reject a decode peer whose KV page stride differs at Mooncake registration (#48231)
- **代码层面**：`mooncake/conn.py` 的 `_publish_peer_registration` 新增 `_kv_layout_mismatch` 检查，比较 prefill 和 decode 的 `kv_item_lens`；`_state_stride_mismatch` 里 `state_type=None` 时用 `"KV"` 作为 label。
- **新手概念课堂**：PD 分离时，prefill 按自己的 `item_len` 算目标地址 `dst_ptr + page * item_len`。如果 decode 的 `item_len` 不同，每页都写错位置。这个检查在注册阶段就拒绝不匹配的 peer，避免运行时写坏内存。
- **对你有什么影响**：PD 分离部署时，如果两端 attention backend / kv-cache-dtype 不一致导致布局不同，会立刻报错而不是静默出错结果。

### 17. [Refactor] utils: Split `common.py` into topic modules and owner packages (#48781)
- **代码层面**：把 `sglang/srt/utils/common.py` 拆成 `device.py`、`log_utils.py`、`triton_utils.py`、`multimodal/media_io.py` 等多个模块；大量 import 路径更新（如 `from sglang.srt.utils.common import is_host_cpu_arm64` → `from sglang.srt.utils.device import is_host_cpu_arm64`）；CI 脚本、文档同步更新。
- **新手概念课堂**：`common.py` 变成了"什么都往里塞"的垃圾抽屉，任何模块 import 它都会拖上一堆无关依赖。拆成按主题归属的模块后，依赖关系清晰，import 更快。就像把"杂物间"按厨房、工具房、书房重新分类。
- **对你有什么影响**：如果你写代码 import 了 `sglang.srt.utils.common` 里的函数，升级后可能报 ImportError，需要按新路径改。

### 18. [Perf] mhc: halve the DeepGEMM prenorm slices when the grid overflows one wave (#48761)
- **代码层面**：`layernorm/mhc.py` 新增 `_HC_PRENORM_BLOCK_M = 64` 和 `_hc_mix_deepgemm_slices(m)`：当 `cdiv(m, 64) * 16 > num_sms` 时，把 slices 从 16 减到 8；`hc_mix_stats_sinkhorn_deepgemm` 里用这个函数替换常量。
- **新手概念课堂**：DeepGEMM 把 K 维切成 slices 累加来补偿 fp32 精度。如果 grid（M 方向块数 × slices）超过 SM 数量，就要多波执行，效率低。减半 slices 让 grid 落在一波内，代价是精度略降（但注释说误差随 K 线性增长，不低于一半即可）。
- **对你有什么影响**：MHC（多头压缩？）相关模型在大 M 场景下 prenorm 更快，精度影响可忽略。

### 19. [Fix] [Intel] cpu-kernel: Fix rope for VLA model (#40139)
- **代码层面**：`rope.cpp` 的 `RopeParams` 从"假设所有 tensor 是 `[B, S, H, D]` 连续布局"改成"支持任意 stride 的输入/输出"：新增 `q_out_stride*` / `k_out_stride*` / `row_sizes` / `row_strides`，并新增一个接受 `query_out` / `key_out` 的构造函数。
- **新手概念课堂**：RoPE（旋转位置编码）是给 Q/K 加位置信息。VLA（Vision-Language-Action）模型的 Q/K 可能是非连续布局（比如 view 出来的），旧代码按连续布局算偏移就会读错数据。新代码按实际 stride 算。
- **对你有什么影响**：Intel CPU 上跑 VLA 模型（如 OpenVLA 类）不再因 RoPE 算错而输出乱码。

### 20. [Chore] flashmla: bump sgl-flash-mla to 1.0.2 (#48525)
- **代码层面**：`python/pyproject.toml` 里 `sgl-flash-mla==1.0.0` → `==1.0.2`。
- **新手概念课堂**：`sgl-flash-mla` 是 SGLang 团队维护的 MLA（Multi-head Latent Attention）Flash 内核包。版本 bump 通常带 bug fix 或新硬件支持。
- **对你有什么影响**：`pip install` 后自动用 1.0.2，MLA 模型（DeepSeek 系列等）的 attention 性能和稳定性可能改善；无需改代码。

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)