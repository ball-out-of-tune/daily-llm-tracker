# 🚀 vLLM & SGLang 每日更新 — 2026-11-09

> 自动生成于 2026-11-09 08:38 UTC | AI 解读: ✅ 含代码解读

## vllm
## 💡 今日关键词

- **关键词：投机解码（Speculative Decoding）生态化**
  - 一句话：先用小模型"猜"几个 token，再让大模型一次性验证，从而提速。
  - 为什么社区在关注：今天有 4 个 commit（#66483、#63096、#66177、#58094）都在打磨投机解码——基准测试、ROCm 支持、流水线并行、配置透传。说明投机解码已从"实验特性"变成"生产必备"，社区在补齐各种边角场景。
  - 对新手意味着什么：这是当前 LLM 推理加速最热门的方向之一，学懂 EAGLE/MTP 的验证逻辑，能直接参与高性能推理框架开发。

- **关键词：异构硬件适配（ROCm / XPU / CPU）**
  - 一句话：同一套代码要跑在 NVIDIA、AMD、Intel GPU 和 CPU 上。
  - 为什么社区在关注：#63096（ROCm）、#60609（XPU）、#66238（Xeon CPU）、#66350（ROCm AITER 回退）都在这条线上。说明 vLLM 正在从"NVIDIA 优先"走向"全硬件支持"。
  - 对新手意味着什么：如果你手里只有 AMD 卡或 Intel 卡，现在正是参与贡献的好时机——大量适配工作缺人手。

- **关键词：MoE 可插拔化（Pluggable Routers）**
  - 一句话：把 MoE 的"路由器"（决定 token 走哪个专家）做成可替换的插件。
  - 为什么社区在关注：#51006 让研究者可以自定义路由策略而不改核心代码，这是 MoE 研究走向工程化的信号。
  - 对新手意味着什么：做 MoE 算法研究的人，现在可以把新路由算法直接插进 vLLM 验证效果，不用 fork 整个仓库。

- **关键词：多模态视频预算（Video Budget）**
  - 一句话：视频输入会切成一堆帧，每帧占 token，怎么在上下文长度内分配是个难题。
  - 为什么社区在关注：#66432 修复了 Qwen2-VL 视频预算没填满上下文的问题——多模态长视频是 2025 年的主战场。
  - 对新手意味着什么：多模态推理的显存/上下文管理是新兴蓝海，懂视频 token 计算的人很稀缺。

- **关键词：Rust 前端**
  - 一句话：vLLM 正在用 Rust 重写 API 前端，追求更低延迟和更强类型安全。
  - 为什么社区在关注：#66477、#60274 都是 Rust 前端的对齐工作，说明 Rust 重写已进入"功能对齐"阶段。
  - 对新手意味着什么：Rust + LLM 推理是稀缺技能组合，早入场有红利。

---

## 🧪 重要知识点 & 动手实验

### 知识点 1：投机解码中的"验证采样"（Commit 1）

- **是什么**：大模型验证小模型猜的 token 时，不是简单接受/拒绝，而是按概率做"拒绝采样"——保证最终分布和大模型自己采样一致。
- **为什么重要**：如果不做正确的拒绝采样，投机解码会悄悄改变输出分布，生成质量下降但你还以为是模型问题。这是投机解码最容易被忽略的正确性细节。
- **动手试试**：
  ```bash
  # 用 vllm bench forward 对比开启/关闭 sampler 的耗时
  vllm bench forward \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --case decode32x1k=32@1024 \
    --speculative-config '{"method":"eagle","model":"...","num_speculative_tokens":3}'
  # 再加 --exclude-sampler 对比
  ```
- **预期结果**：你会看到 `timing_scope` 从 `target_only` 变成 `target_and_sampler`，且采样确实占了可观的 GPU 时间——这解释了为什么投机解码的"验证"阶段不能省。

### 知识点 2：权重名映射（WeightsMapper）（Commit 2、15）

- **是什么**：HuggingFace 检查点里的参数名和 vLLM 内部层名往往不一样（比如 GPT-J 的 `attn` 对应 vLLM 的 `attn.attn`），需要一张"翻译表"。
- **为什么重要**：名字对不上，权重就加载失败或静默错位——模型能跑但输出是乱码。这是模型适配最常见的坑。
- **动手试试**：
  ```python
  from vllm.model_executor.models.gpt_j import GPTJForCausalLM
  from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig
  # 观察 mapper 如何翻译名字
  mapper = GPTJForCausalLM.hf_to_vllm_mapper
  print(mapper._map_name("transformer.h.0.attn.k_scale"))
  # 期望: transformer.h.0.attn.attn.k_scale
  ```
- **预期结果**：看到名字被正确加上一层 `.attn`，理解"正则负向前瞻 `(?<!\.attn)`"如何避免重复映射。

### 知识点 3：KV Cache 块大小统一（Commit 13）

- **是什么**：不同层（注意力层、卷积状态层）可能想要不同的 cache block size，planner 会把它们统一放大到能整除的最小公倍数。
- **为什么重要**：如果某层用原始块大小索引而实际 cache 是按放大后的块分配的，就会读到错误数据甚至越界崩溃（illegal memory access）。
- **动手试试**：
  ```python
  from vllm.v1.core.kv_cache_planning import unify_kv_cache_spec_page_size
  from vllm.v1.kv_cache_interface import FullAttentionSpec
  import torch
  attn = FullAttentionSpec(block_size=64, num_kv_heads=4, head_size=128, dtype=torch.bfloat16)
  # 假设 conv 层 block_size=4
  # unified 会把 conv 放大到 4*kernel_size 使两者对齐
  ```
- **预期结果**：理解"统一页大小"是 KV cache 管理的基本约束，任何自定义层都要遵守。

### 知识点 4：可插拔层的 OOT 注册机制（Commit 18）

- **是什么**：`PluggableLayer.register_oot()` 让你在不改源码的情况下，用自己写的子类替换框架里的类。
- **为什么重要**：做算法研究时，你不想维护一个 fork；OOT（out-of-tree）注册让你把实验代码放在独立包里。
- **动手试试**：
  ```python
  from vllm.model_executor.layers.fused_moe.router.base_router import FusedTopKRouter
  from vllm.model_executor.custom_op import op_registry_oot

  class MyRouter(FusedTopKRouter):
      pass

  FusedTopKRouter.register_oot(MyRouter)
  assert op_registry_oot["FusedTopKRouter"] is MyRouter
  ```
- **预期结果**：注册后，模型构造时会自动用你的 `MyRouter`——注意必须保持构造函数签名和路由语义一致。

### 知识点 5：环境变量优先级与回退策略（Commit 19）

- **是什么**：`VLLM_ROCM_USE_AITER=1` 只是"偏好"用 AITER 内核，如果该层不支持就回退到 Triton；但显式 `VLLM_ROCM_USE_AITER_MOE=1` 则表示"必须用，不支持就报错"。
- **为什么重要**：这是典型的"软偏好 vs 硬要求"设计模式。搞混了会导致要么莫名崩溃，要么性能不达预期。
- **动手试试**：
  ```bash
  # 软偏好：不支持时回退
  VLLM_ROCM_USE_AITER=1 python -c "from vllm... import select_fp8_moe_backend; ..."
  # 硬要求：不支持时报错
  VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1 python ...
  ```
- **预期结果**：前者返回 TRITON 后端，后者抛 `ValueError: AITER`。理解"显式配置覆盖隐式默认"的优先级链。

---

## 🔥 重要更新

1. **#66177 [Spec Decode][PP] 集中化辅助隐藏状态中继**：EAGLE3/DFlash/DSpark 现在支持流水线并行，意味着投机解码可以在多卡 PP 部署下使用，这是大模型推理的重要能力扩展。
2. **#51006 [MoE] 让 fused MoE 路由器可插拔**：MoE 路由策略从"写死"变成"插件"，研究者和框架开发者可以解耦，长期看会催生大量自定义路由实验。
3. **#66432 [Bugfix][Multimodal] Qwen2-VL 视频预算填满上下文**：修复了长视频输入时 token 预算没用满的问题，直接影响多模态长视频推理的可用性。
4. **#4088c283 [Core][Kimi K3.1] KV cache 与注意力改动支持 TP8**：为 Kimi K3.1 在 8 卡张量并行下服务铺路，涉及 DSA 稀疏注意力和 DEP 解码，是超大 MoE 模型部署的关键基础设施。

---

## 📋 逐条解读

### 1. [Bench] Time target sampling in vllm bench forward (#66483)

- **代码层面**：改了 `vllm/benchmarks/forward` 相关代码和文档。原来 bench forward 只测"目标模型前向"，现在默认把"目标采样 + 拒绝采样"也纳入计时，新增 `--exclude-sampler` 开关可以关掉。
- **新手概念课堂**：就像测一辆车的加速性能——以前只测发动机（前向），现在把变速箱（采样）也算进去，因为实际开车时变速箱也要花时间。`--exclude-sampler` 就是"只测发动机"的开关。
- **对你有什么影响**：如果你用 `vllm bench forward` 做性能对比，数字会变大（因为多测了采样），跨版本对比时要注意 `timing_scope` 字段。

### 2. [Bugfix] Map compressed-tensors KV-cache scale names for GPT-J (#65949)

- **代码层面**：在 `vllm/model_executor/models/gpt_j.py` 的 `hf_to_vllm_mapper` 里加了一条正则映射：`(?<!\.attn)\.attn\.([qkv]_(?:scale|zero_point))$` → `.attn.attn.\1`。因为 GPT-J 的 HF 注意力模块本身叫 `attn`，而 vLLM 的 Attention 层是它的 `attn` 子模块。
- **新手概念课堂**：想象你搬家，快递单上写"3 楼 302"，但新楼的门牌规则是"3 楼 302 室的门"——需要多加一层地址。权重名映射就是给检查点参数"重新贴地址"。
- **对你有什么影响**：用 GPT-J + compressed-tensors 量化 KV cache 的用户，之前可能加载失败或输出乱码，现在正常了。

### 3. [Rust Frontend] Keep special tokens' spelling in logprob tokens (#66477)

- **代码层面**：`rust/src/text/src/output/decoded.rs` 里，`decode_logprobs` 和 `decode_prompt_logprobs` 不再接收 `skip_special_tokens` 参数。logprob 里的 token 一律保留特殊 token 原样（如 `<s>`），只有输出文本才受 `skip_special_tokens` 影响。
- **新手概念课堂**：就像字幕翻译——正片字幕可以省略"（掌声）"这种提示，但如果你要分析"掌声出现在第几秒"，就必须保留原始标记。logprob 分析需要原始 token，不能被过滤。
- **对你有什么影响**：用 Rust 前端做 logprob 分析的用户，现在能正确看到 `<s>`、`</s>` 等特殊 token 的 logprob，与 Python 前端行为一致。

### 4. [Bugfix][Multimodal] Fill the Qwen2-VL video budget up to the context length (#66432)

- **代码层面**：修改了 Qwen2-VL 系列的视频 token 预算计算逻辑，让视频帧数能填满 `max_model_len`，而不是提前截断。新增测试验证 60 帧视频能生成 >17280 token。
- **新手概念课堂**：视频就像一叠照片，每张照片要占"内存格子"（token）。以前算法保守，明明还有空格子却不放照片；现在会把格子填满，视频看得更完整。
- **对你有什么影响**：Qwen2-VL/Qwen2.5-VL 用户处理长视频时，模型能看到更多帧，理解更完整。但注意：`max_model_len` 调大会增加 profiling 的视频尺寸，可能挤压 KV cache 空间。

### 5. [ROCm][Spec Decode] Enable fused multi-step draft decode for ROCM_AITER_UNIFIED_ATTN (#63096)

- **代码层面**：在 `rocm_aiter_unified_attn.py` 里给 `RocmAiterUnifiedAttentionMetadataBuilder` 加了 `supports_draft_decode_metadata_update = True` 和一个空的 `update_draft_decode_metadata` 方法。意思是这个后端支持"原地更新元数据"，不用重建。
- **新手概念课堂**：CUDA graph 捕获后，每次重放不能改结构。如果元数据能"原地刷新"，就像给录像带换字幕而不重录——省下重建开销。
- **对你有什么影响**：AMD ROCm 用户跑投机解码时，多步 draft decode 现在能走融合路径，性能提升。

### 6. [Frontend] Mark disabled scale-out routes in the startup route log (#66481)

- **代码层面**：在 `launcher.py` 的启动日志里，给被禁用的路由加上 `(disabled)` 后缀；新增常量 `DISABLED_ROUTE_TAG = "disabled"`；`factories.py` 给占位路由打上这个 tag。
- **新手概念课堂**：就像商场导览图上，关闭的店铺标注"暂停营业"——用户一眼就知道哪些功能没开。
- **对你有什么影响**：启动日志更清晰，排查"为什么某个 API 404"时能直接看到它被禁用了。

### 7. [Bugfix][LoRA] Use 3D MoE LoRA wrapper for Gemma4 (#64437)

- **代码层面**：新增 `tests/lora/test_gemma4_moe_lora.py`，验证 Gemma4 的 MoE 专家 LoRA 使用 3D 融合布局（`experts.base_layer.lora_{A,B}`）时，服务结果与"把 LoRA 合并进权重"一致。
- **新手概念课堂**：LoRA 像给模型贴"便利贴"，MoE 模型有多个专家，便利贴要贴对专家。3D 布局就是把所有专家的便利贴打包成一个三维张量，省内存。
- **对你有什么影响**：用 PEFT 格式的 Gemma4 MoE LoRA 适配器，之前可能加载错乱，现在能正确服务。

### 8. [XPU] Bump up kernel 0.1.17, change MoE and MLA interfaces (#60609)

- **代码层面**：`requirements/xpu.txt` 把 `vllm_xpu_kernels` 从 0.1.16.3 升到 0.1.17；`_custom_ops.py` 里 `concat_and_cache_mla` 去掉 XPU 特判，统一传 `skip_scale`；`xpu_moe.py` 删掉 `prepare_fp8_moe_layer_for_xpu` 函数，改由内核内部重排。
- **新手概念课堂**：就像换了个新版本的"外挂工具箱"，以前需要手动把工具摆好（转置权重），现在工具箱自己会整理。
- **对你有什么影响**：Intel XPU 用户升级后 MLA 和 MoE 接口更统一，代码更简洁，但需要配套升级内核包。

### 9. [Spec Decode][PP] Centralize auxiliary hidden-state relay (#66177)

- **代码层面**：文档 `docs/features/speculative_decoding/eagle.md` 新增"Pipeline Parallelism"章节；测试 `test_eagle3_pp.py` 扩展支持多种模型和 chat prompt。核心是把"辅助隐藏状态跨 PP 阶段传递"的逻辑集中管理。
- **新手概念课堂**：流水线并行像工厂流水线，第 1 站做完传给第 2 站。EAGLE 草稿模型需要目标模型的"中间产物"（隐藏状态），必须一站站传到最后一站。
- **对你有什么影响**：用 PP 部署 + EAGLE3 的用户，现在可以正常工作了；不支持的架构会在启动时报明确错误。

### 10. [CPU][recipes] Add Qwen3-VL-4B-Instruct support for Intel Xeon 6 (#66238)

- **代码层面**：新增 `recipes/models/Qwen/Qwen3-VL-4B-Instruct.yaml`，声明在 Intel Xeon 6 CPU 上用 BF16 推理，包含 `--language-model-only`（跳过视觉编码器省显存）和 `--mm-encoder-tp-mode data` 等特性。
- **新手概念课堂**：recipe 就像"菜谱"，告诉你用什么硬件、什么参数、什么版本能跑通某个模型。
- **对你有什么影响**：只有 CPU 的用户现在有官方验证的 Qwen3-VL 配置可参考，不用自己摸索参数。

### 11. [Bugfix][Spec Decode] Propagate rope hf_overrides to MTP draft (#58094)

- **代码层面**：`SpeculativeConfig.compose_draft_hf_overrides` 现在会把 `text_config.rope_parameters` 这类嵌套的 rope 覆盖传给草稿模型，但 `max_position_embeddings` 等非 rope 覆盖仍然只作用于目标模型。
- **新手概念课堂**：rope 是位置编码的"尺子"，目标模型把尺子调长了（YaRN 扩展），草稿模型也必须用同一把尺子，否则位置对不上。
- **对你有什么影响**：用 MTP 草稿 + YaRN 长上下文扩展的用户，之前草稿模型位置编码不匹配导致接受率低，现在修复了。

### 12. [Rust Frontend] support --enable-force-include-usage (#60274)

- **代码层面**：`rust/src/cmd/src/cli.rs` 新增 `enable_force_include_usage` 字段，映射到配置的 `force_include_usage`；测试从"不支持该参数"改为"支持并透传"。
- **新手概念课堂**：就像快递单上"是否强制显示运费"——有些客户端需要每次响应都带 token 用量，即使流式输出中间块也要带。
- **对你有什么影响**：用 Rust 前端的用户现在可以用 `--enable-force-include-usage`，与 Python 前端行为对齐。

### 13. [Bugfix][Model] Fix Inkling NVIDIA sconv cache block-size mismatch (#51951)

- **代码层面**：测试 `test_qkvr_prep.py` 新增验证：conv 层的 cache 索引必须用 planner 放大后的 block size，而不是原始窗口大小 W。用 `unify_kv_cache_spec_page_size` 把 conv 块从 4 放大到 16（4×kernel_size）。
- **新手概念课堂**：就像停车位编号——如果停车场把 4 个小车位合并成 1 个大车位，你还按小号找车就会停错地方。conv cache 的索引必须和分配时的块大小一致。
- **对你有什么影响**：Inkling 模型在 NVIDIA GPU 上，之前可能输出乱码或崩溃（越界），现在修复。

### 14. [XPU][CI] Migrate several B70 jobs from intel_gpu to intel_gpu_k8s (#66034)

- **代码层面**：`.buildkite/intel_jobs/*.yaml` 把 `device: intel_gpu` 改成 `device: intel_gpu_k8s`，加 `num_devices` 和 `resource_claim_template`，工作目录改成 `/workspace/vllm/`。
- **新手概念课堂**：就像测试机房从"固定工位"搬到"云桌面"，资源按需申请，更灵活。
- **对你有什么影响**：普通用户无感，但 Intel GPU 的 CI 更稳定，回归问题能更早发现。

### 15. [Bugfix][Qwen4Exp] Exclude the HC padding shard from the loaded-shard check (#66461)

- **代码层面**：`hyperconnection.py` 里给 `input_mix_weight_down_block_inject.shard_ids` 设为 `(0, 1)`，告诉加载器"只期望这两个 shard，padding shard 不用从检查点加载"。
- **新手概念课堂**：就像拼图有 3 块，但检查点只给了 2 块，第 3 块是填充用的空白块——加载器不该抱怨"缺了一块"。
- **对你有什么影响**：Qwen4Exp 模型加载时不再误报"缺少权重"。

### 16. [Core][Kimi K3.1] KV cache and attention changes to serve Kimi K3.1 at TP8 (#66037)

- **代码层面**：大规模改动，涉及 `tests/models/kimi_k3/nvidia/specialized_decode/` 下的测试和对应的 attention/KV cache 实现。核心是让 Kimi K3.1 的 DSA（稀疏注意力）和 DEP（专用解码）在 TP8 下正确工作。
- **新手概念课堂**：TP8 就是把模型切成 8 份放在 8 张卡上。DSA 是"只关注最重要的几个 token"，DEP 是"专门优化的解码路径"。让它们在 8 卡下协同，就像 8 个人分工读一本书还要保持页码一致。
- **对你有什么影响**：部署 Kimi K3.1 的用户可以在 8 卡上获得更好的吞吐；这是超大 MoE 模型服务的基础设施。

### 17. [Bugfix][CPU] Name the cause when a DP rank's assigned NUMA nodes don't exist (#64499)

- **代码层面**：`cpu_resource_utils.py` 新增 `get_worker_memory_nodes`，当分配的 NUMA 节点不存在时抛 `ValueError` 并提示"DP×TP×PP 不能超过 NUMA 节点数"；`omp` 模块在 `VLLM_CPU_SIM_MULTI_NUMA` 与显式绑定冲突时给出明确断言消息。
- **新手概念课堂**：NUMA 是 CPU 的内存分区，就像一栋楼有多个仓库。如果给某个工人分配了"3 号仓库"，但楼里只有 1、2 号，要直接说"3 号不存在"，而不是让他去撞墙（IndexError）。
- **对你有什么影响**：CPU 多卡用户配置错误时，会看到清晰的错误提示，而不是莫名其妙的崩溃。

### 18. [MoE] Make fused MoE routers pluggable (#51006)

- **代码层面**：`docs/design/plugin_system.md` 新增说明；测试 `test_routing.py` 验证 `FusedTopKRouter.register_oot(MyRouter)` 后输出与原生一致，且 DeepseekV2MoE 的 correction bias 参数能正确共享。
- **新手概念课堂**：路由器是 MoE 的"分诊台"，决定每个 token 去哪个专家。可插拔就像分诊台可以换成"按症状分"或"按年龄分"，医院主体不用改。
- **对你有什么影响**：做 MoE 路由算法研究的用户，可以把新算法作为插件注册进 vLLM 验证，不用 fork 仓库。

### 19. [Bugfix][ROCm] Let the FP8 MoE oracle fall back when the global AITER flag can't serve the layer (#66350)

- **代码层面**：`oracle/fp8.py` 里把 `else:` 改成 `elif envs.is_set("VLLM_ROCM_USE_AITER_MOE"):`。含义：只有显式设置了 `VLLM_ROCM_USE_AITER_MOE` 才强制要求 AITER，否则 `VLLM_ROCM_USE_AITER=1` 只是软偏好，不支持时回退 Triton。
- **新手概念课堂**：就像点菜说"有牛排就吃牛排"（软偏好）vs"必须吃牛排，没有就不吃"（硬要求）。前者餐厅没牛排会给你换鸡排，后者直接拒绝。
- **对你有什么影响**：AMD 用户开 `VLLM_ROCM_USE_AITER=1` 时，某些层不支持 AITER 也不会崩了，自动回退到 Triton。

### 20. [Sleep Mode] Remove level-2 workarounds that #66010 makes unnecessary (#66287)

- **代码层面**：删除了 `test_lm_head_reset_sharded_to_full_mapping` 测试和 `reset_lora_state` 相关逻辑，改成 `on_target_weights_replaced` 钩子。因为 #66010 已经从根本上解决了权重替换问题，不再需要 LoRA 层的特殊重置。
- **新手概念课堂**：就像以前桥断了要绕路（workaround），现在新桥修好了（#66010），绕路的临时路牌可以拆了。清理技术债是成熟项目的标志。
- **对你有什么影响**：普通用户无感，但代码更简洁，未来 sleep mode + 权重更新的组合更可靠。

## sglang
## 💡 今日关键词（放在最前面！）

- **关键词：HiSparse / 稀疏 KV 交换（shared KV sources）**
  - 一句话通俗解释：把"主机内存里的大块 KV 缓存"和"GPU 显存里的工作区"之间做搬运时，允许同一个 KV 源被多个请求共享，而不是每个请求各拷一份。
  - 为什么社区在关注：大模型推理的瓶颈越来越从"算"转向"显存/带宽"，HiCache、HiSparse 这类"分层 KV 缓存"是当前最热的方向之一。这次改动还引入了 `LateBoundHostBase`（运行时才绑定主机地址和步长），说明大家在追求"地址可以晚绑定、布局可以灵活"的极致灵活性。
  - 对新手意味着什么：如果你在学 GPU 编程，`__restrict__`、`__grid_constant__`、`cudaMemcpyAsync`、页对齐、host/device 指针语义这些"底层功夫"会越来越值钱。

- **关键词：JIT 内核统一化（activation.cuh 统一 FP32 激活函数）**
  - 一句话通俗解释：把 sigmoid/silu/gelu/softplus 这些激活函数收拢到一个头文件里，提供"精确版"和"快速近似版"两套实现。
  - 为什么社区在关注：SGLang 的 JIT kernel 体系正在从"每个 kernel 自己写一份"走向"公共库 + 规范"，这是工程成熟度的标志。
  - 对新手意味着什么：写 CUDA 不只要会写 kernel，还要会"设计 API"——同一函数两种精度模式、`kFastMath` 编译期开关，这是很实用的设计模式。

- **关键词：HugePage 预检与分配策略（SGLANG_HUGEPAGE_MODE）**
  - 一句话通俗解释：给主机端 KV 缓存分配内存时，可以选择"普通页 / 尽量用大页 / 必须用大页"三种策略，并在启动时预检可用大页数量。
  - 为什么社区在关注：HiCache 把 KV 放到 CPU 内存后，TLB miss 和页表开销会显著拖慢 PCIe 传输；大页是标准解法。新增 `required` 模式说明生产环境开始要求"确定性"。
  - 对新手意味着什么：Linux 内存管理（`mmap`、`MAP_HUGETLB`、`/proc/sys/vm/nr_hugepages`、`madvise`）是 AI Infra 的必备知识。

- **关键词：CUDA Graph padding 正确性（多个 commit 都在修）**
  - 一句话通俗解释：CUDA Graph 为了复用，会把 batch 补齐到固定大小，但补齐出来的"假 token"不能写入 KV、不能参与计算，否则会污染结果。
  - 为什么社区在关注：Commit 5（inkling）、6（gemma4）、7（dp-attention）全都在修 padding 相关问题，说明"图捕获 + 动态 batch"是当前 bug 高发区。
  - 对新手意味着什么：理解 CUDA Graph 的"静态形状"约束，是性能优化的入门必修课。

- **关键词：异构硬件适配（AMD / Intel XPU / SM120 / CPU）**
  - 一句话通俗解释：同一份代码要跑在 NVIDIA、AMD、Intel、CPU 上，需要大量架构判断和降级路径。
  - 为什么社区在关注：Commit 9（AMD fp8 融合）、10/12（SM120）、15（CPU Intern-S2）、16（Intel XPU 0.6.0）都在做这件事，说明 SGLang 正在成为"多硬件后端"的统一推理框架。
  - 对新手意味着什么：`get_platform().is_sm100 / is_sm120 / is_cpu` 这类平台抽象层，是跨硬件开发的通用范式。

---

## 🧪 重要知识点 & 动手实验

### 知识点 1：CUDA Graph 的 padding 陷阱（来自 Commit 5 / 6 / 7）

- **是什么**：CUDA Graph 在捕获时要求所有张量形状固定，所以推理框架会把 batch 补齐到固定大小；多出来的行是"假 token"，必须被显式跳过。
- **为什么重要**：如果不跳过，假 token 会写入 KV cache（污染其他请求）、参与 attention（NaN 传播）、进入 logits（影响采样）。这类 bug 往往只在特定 batch 大小下复现，极难排查。
- **动手试试**：

```python
import torch

# 模拟：真实 token 5 个，padding 到 8 个
real, bucket = 5, 8
hidden = torch.randn(bucket, 8)
hidden[real:] = float("nan")  # 模拟 padding 行里的脏数据

# 错误做法：直接用全部行做下一层计算
bad = hidden @ torch.randn(8, 8)
print("含 NaN 的输出:", bad.isnan().any().item())

# 正确做法：先用 cu_seqlens 截断
cu = torch.tensor([0, 5])
real_hidden = hidden[: cu[-1]]
good = real_hidden @ torch.randn(8, 8)
print("截断后输出正常:", not good.isnan().any().item())
```

- **预期结果**：第一段输出 `True`（NaN 污染），第二段输出 `True`（安全）。你会理解为什么 Commit 5 要在 kernel 里加 `t < cu[B]` 判断，Commit 7 要在 `post_forward_mlp_sync_batch` 里截断 `hidden_states`。

---

### 知识点 2：Triton 中把量化融进写缓存（来自 Commit 9）

- **是什么**：KV cache 是 fp8 的，但 attention 算出来是 bf16/fp16，写缓存时需要做 `clamp(x / scale, -fp8_max, fp8_max)` 再 cast。把这一步融进写入 kernel，省掉一次中间张量和一次 kernel launch。
- **为什么重要**：KV cache 写入是 decode 阶段的每步必经操作，融合能减少内存带宽和 launch 开销；同时 `k_scale` 支持"传指针"（device 上读）避免 host sync，这对 CUDA Graph 捕获至关重要。
- **动手试试**：

```python
import torch

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max

def quantize(x, scale):
    x = x.to(torch.float32) / scale
    x = torch.clamp(x, -FP8_MAX, FP8_MAX)
    return x.to(torch.float8_e4m3fn)

x = torch.randn(4, 8, dtype=torch.bfloat16) * 100
scale = torch.tensor(2.0)
q = quantize(x, scale)
dq = q.to(torch.float32) * scale
print("量化误差:", (x.float() - dq).abs().max().item())
```

- **预期结果**：误差应在 `scale * 0.5` 量级（fp8 的量化步长）。你会明白为什么 `scale` 一般用 `amax / FP8_MAX` 计算，以及为什么 `clamp` 不能省。

---

### 知识点 3：HugePage 分配与 TLB（来自 Commit 4）

- **是什么**：Linux 默认页大小 4KB，大页（2MB / 1GB）能显著减少页表项数量和 TLB miss。`mmap` 时通过 `MAP_HUGETLB | MAP_HUGE_2MB` 指定。
- **为什么重要**：HiCache 把 KV 放到主机内存后，PCIe DMA 传输会频繁访问这些内存；大页能减少 TLB miss，提升带宽。`required` 模式则保证"要么大页，要么启动失败"，避免生产环境悄悄降级。
- **动手试试**：

```bash
# 1. 查看当前大页配置
cat /proc/meminfo | grep -i huge
cat /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages

# 2. 申请 512 个 2MB 大页（需要 root）
echo 512 | sudo tee /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages

# 3. 用 Python 验证 mmap 大页
python3 -c "
import mmap, os
size = 2 * 1024 * 1024
fd = os.open('/dev/zero', os.O_RDONLY)
m = mmap.mmap(fd, size, flags=mmap.MAP_PRIVATE | mmap.MAP_HUGETLB | (21 << 26))
print('大页 mmap 成功, 地址:', hex(id(m)))
m.close(); os.close(fd)
"
```

- **预期结果**：如果大页不足会报 `OSError: [Errno 12] Cannot allocate memory`。这就是 SGLang 新增 `preflight` 预检的原因——提前告诉你"大页不够"，而不是跑到一半才失败。

---

### 知识点 4：流式响应的资源清理（来自 Commit 8）

- **是什么**：客户端断开连接时，ASGI 服务器会停止 `send`，但 Python 的 async generator 不会自动关闭，导致请求一直挂在 scheduler 里跑。解决办法是在 `finally` 里显式 `await generator.aclose()`。
- **为什么重要**：这是"客户端断开 → 后端继续烧 GPU"的经典浪费场景。修复后，客户端一断开，请求立即被 abort，释放 KV 和算力。
- **动手试试**：

```python
import asyncio

async def gen():
    try:
        for i in range(100):
            yield i
            await asyncio.sleep(0.1)
    finally:
        print("generator 被清理了")

async def main():
    g = gen()
    async for x in g:
        if x == 3:
            break  # 模拟客户端断开
    await g.aclose()  # 关键：显式关闭

asyncio.run(main())
```

- **预期结果**：输出 `generator 被清理了`。如果去掉 `await g.aclose()`，`finally` 要到 GC 时才执行（甚至不执行）。这就是 Commit 8 加 `_ClosingStreamingResponse` 的原因。

---

### 知识点 5：跨硬件平台的架构门控（来自 Commit 10 / 12 / 15 / 16）

- **是什么**：同一份 Python 代码要跑在 SM90（Hopper）、SM100（Blackwell）、SM120（消费级 Blackwell）、AMD、Intel XPU、CPU 上，需要用 `get_platform().is_sm100` 这类判断选择不同 kernel 路径。
- **为什么重要**：不同架构的 kernel 支持情况不同（比如 DeepGEMM 的 sparse kernel 只支持 SM100），如果门控写错，轻则性能下降，重则启动报错。Commit 12 就是修"SM120 被误判"的问题。
- **动手试试**：

```python
import torch

def get_arch():
    if not torch.cuda.is_available():
        return "cpu"
    major, minor = torch.cuda.get_device_capability()
    return f"sm{major}{minor}"

arch = get_arch()
print("当前设备架构:", arch)

# 模拟门控逻辑
def can_use_deepgemm_dense(arch):
    return arch in ("sm100", "sm120", "sm121")

def can_use_deepgemm_sparse(arch):
    return arch == "sm100"  # 严格只支持 SM100

print("可用 dense DeepGEMM:", can_use_deepgemm_dense(arch))
print("可用 sparse DeepGEMM:", can_use_deepgemm_sparse(arch))
```

- **预期结果**：在 4090（sm89）上两者都是 `False`；在 B200（sm100）上两者都是 `True`；在 5090（sm120）上 dense 为 `True`、sparse 为 `False`。你会理解 Commit 12 为什么要区分 `is_sm100` 和 `is_sm100_or_newer` 两个函数。

---

## 🔥 重要更新

1. **Commit 1（HiSparse 支持共享 KV 源）**：引入 `LateBoundHostBase` 和 `PassThroughDeviceLocs`，让主机地址和步长在运行时才绑定。这是 HiCache/HiSparse 走向"灵活布局 + 多请求共享"的关键一步，直接影响长上下文场景的显存效率。

2. **Commit 8（客户端断开立即 abort 请求）**：修复了"客户端走了、GPU 还在烧"的经典浪费。通过 `_ClosingStreamingResponse` 和 `finally: await stream.aclose()` 保证请求被及时清理，对线上服务的资源利用率有直接收益。

3. **Commit 4（HugePage 预检 + 三种分配模式）**：新增 `SGLANG_HUGEPAGE_MODE=off/prefer/required`，并在启动时预检大页可用性。生产环境可以用 `required` 保证确定性，避免"悄悄降级到 4KB 页"导致的性能抖动。

4. **Commit 12（恢复 SM120/SM121 的 DeepGEMM indexer 门控）**：拆分 `is_sm100`（严格 SM100，用于 sparse kernel）和 `is_sm100_or_newer`（SM100+，用于 dense kernel），修复了 SM120 用户启动报错的问题。这是"架构门控要精确"的典型案例。

---

## 📋 逐条解读

### 1. [Feat] hisparse: Support shared KV sources in HiSparse MLA swap kernels
- **代码层面**：改 `hisparse.cuh` 里的 `copy_miss_item` 和 `load_cache_to_device_buffer_kernel`。核心是新增 `host_row_bytes` 参数（主机行步长可以和 `item_size_bytes` 不同），以及 `PassThroughDeviceLocs`、`LateBoundHostBase` 两个编译期开关。`host_binding[0]` 是主机基地址，`host_binding[1]` 是行步长，都在 kernel 运行时读取。
- **新手概念课堂**：想象一个图书馆，以前每本书厚度一样（`item_size_bytes`），书架间距等于书厚；现在书架间距可以自定义（`host_row_bytes`），而且书架位置可以"晚点再告诉你"（late bound）。这样同一份数据可以被多个读者共享，不用每人复印一份。
- **对你有什么影响**：长上下文推理时，KV cache 的显存/内存占用会下降，多请求共享前缀的场景（如 system prompt 复用）受益明显。

### 2. [Refactor] jit: Unify FP32 activation functions in sgl_kernel/activation.cuh
- **代码层面**：新建 `activation.cuh`，把 `sigmoid/silu/gelu/gelu_tanh/softplus/sqrt_softplus/relu2` 统一实现，每个函数带 `kFastMath` 模板参数。同时把 `kda_fused_decode.cuh` 里的 `device::math::silu_fast` 替换为 `device::activation::silu`。还更新了开发者文档 `SKILL.md`。
- **新手概念课堂**：就像把散落在各处的"螺丝刀"收进一个工具箱，并且每把螺丝刀都分"精密版"和"快速版"两种。`kFastMath=true` 用 `rcp.approx` 等近似指令，误差约 1e-6；`kFastMath=false` 严格复现 PyTorch 的 eager 结果。
- **对你有什么影响**：写新 kernel 时不用再自己实现激活函数，直接 include 即可；同时测试时要注意选对精度模式，否则 bit-level 对比会失败。

### 3. [Doc] cookbook: Refresh SemiAnalysis AgentX launches to InferenceX main 0d7f94e7
- **代码层面**：更新 `docs/src/snippets/agentx/` 下的 JSX 文件，同步 SemiAnalysis InferenceX 仓库的最新 commit 和 PR 列表（`generatedOn` 从 11-05 改为 11-08）。
- **新手概念课堂**：这是"自动生成的文档"，由同步脚本从上游仓库拉取数据后生成。文件头明确写着 `GENERATED ... do not edit by hand`。
- **对你有什么影响**：普通用户无感知；但如果你要贡献文档，记住别手改这些文件，改脚本。

### 4. [Feat] mem_cache: Add hugepage preflight and allocation modes for host pools
- **代码层面**：新增环境变量 `SGLANG_HUGEPAGE_MODE`（`off/prefer/required`），新增 `host_memory_budget_plain_hugetlb` 和 `host_memory_budget_from_plain_hugetlb` 两个函数用于预算计算。`hicache_auto_size.py` 里根据 allocator 类型和设备决定是否把 hugetlb 计入预算。文档同步更新。
- **新手概念课堂**：大页就像"集装箱"——装同样多的货，用集装箱（2MB/1GB 页）比用小纸箱（4KB 页）搬运次数少得多。`required` 模式就是"没有集装箱就不装船"，`prefer` 是"有就用、没有就用纸箱"。
- **对你有什么影响**：如果你在跑 HiCache 或 PD 分离，可以用 `SGLANG_HUGEPAGE_MODE=required` 保证性能确定性；但记得先按文档预留大页，否则启动会直接失败。

### 5. [Fix] inkling: Skip KV stores for CUDA-graph padding tokens in the fused attn prologue
- **代码层面**：在 `inkling_attn_prologue_fused.cuh` 的 `AttnPrologueExtendParams` 里新增 `B` 字段（真实序列数），kernel 里加判断 `t < cu[B]` 才写 KV。新增测试 `test_inkling_attn_prologue_extend_pad.py`，用 NaN 填充 padding 行来验证不被写入。
- **新手概念课堂**：CUDA Graph 为了复用，会把 batch 补齐到固定大小；补齐出来的 token 是"群众演员"，不能让他们上台抢戏（写 KV）。`cu[B]` 就是"真实演员名单的末尾"。
- **对你有什么影响**：修复了 CUDA Graph + inkling 模型下 KV cache 被 padding token 污染的问题，输出正确性提升。

### 6. [Fix] gemma4: Fix Bolt E4B and Gemma4 prefill CUDA graphs
- **代码层面**：`RadixAttention` 新增 `graph_layer_id` 属性（默认等于 `layer_id`，但可显式指定），`cuda_graph_setup.py` 里用 `_graph_layer_id()` 替代直接读 `layer_id`。原因是"共享 KV cache 不代表两个 decoder 层是同一个 graph 层"。
- **新手概念课堂**：想象两个演员共用同一个化妆间（共享 KV cache），但他们仍是两个不同的人（不同的 graph 层）。以前代码把"共用化妆间"误判为"同一个人"，导致 CUDA Graph 索引错位。
- **对你有什么影响**：Gemma4 / Bolt E4B 的 prefill CUDA Graph 不再报错或输出错误，推理正确性提升。

### 7. [Fix] dp-attention: Trim MLP-sync pad rows from extend outputs before the draft extend
- **代码层面**：`forward_batch_info.py` 的 `post_forward_mlp_sync_batch` 里，extend 模式下用 `_original_num_tokens`（真实 token 数）而不是 `seq_lens_sum`（含 cached prefix）来截断 `next_token_logits`、`hidden_states`、`mm_input_embeds`。新增测试验证 chunked prefill 场景。
- **新手概念课堂**：DP attention 下各 rank 要同步 MLP 结果，会把 batch 补齐到 DP 宽度；补齐出来的行是"凑数的"。`seq_lens_sum` 把 prefix 也算进去了，所以不能直接用它截断。
- **对你有什么影响**：EAGLE 投机解码在 DP attention + chunked prefill 场景下不再因为 hidden_states 行数不对而崩溃。

### 8. [Fix] srt: Promptly abort running requests on client disconnect
- **代码层面**：`serving_chat.py` 新增 `_ClosingStreamingResponse` 类，在 `stream_response` 的 `finally` 里 `await self.body_iterator.aclose()`；`prepend_first_chunk` 的 `finally` 里 `await generator.aclose()`；`_generate_chat_stream` 的 `finally` 里 `await stream.aclose()`。`tokenizer_manager.py` 也有配套改动。
- **新手概念课堂**：客户端断开就像顾客中途离开餐厅，但厨房还在做菜（GPU 还在算）。以前要等菜做完才发现没人吃；现在服务员一看顾客走了就通知厨房停火。
- **对你有什么影响**：线上服务中客户端断开后，GPU 资源立即释放，不再"空转"，吞吐和成本都改善。

### 9. [Perf] [AMD] kvcache: Fuse fp8 quantization into the vectorized_5d KV-cache write
- **代码层面**：`shuffle_5d_cache.py` 的 Triton kernel 新增 `QUANT`、`K_SCALE_IS_PTR`、`V_SCALE_IS_PTR` 三个 constexpr 参数，在写缓存前做 `clamp(x / scale, -fp8_max, fp8_max)` 再 cast。新增 `_scale_arg` 辅助函数处理"scale 是 device tensor / host tensor / float / None"四种情况。
- **新手概念课堂**：以前是"先算好 fp8 结果放临时张量，再写缓存"（两步）；现在是"边算边写"（一步）。`scale` 在 device 上读可以避免 host sync，这对 CUDA Graph 捕获很关键。
- **对你有什么影响**：AMD 平台 fp8 KV cache 的 decode 性能提升，同时减少了中间张量的显存占用。

### 10. [Perf] SM120: honour SGLANG_OPT_DEEPGEMM_HC_PRENORM for the DeepSeek-V4.1 compensated mhc projection
- **代码层面**：`deepseek_v4.py` 和 `deepseek_v4_mhc.py` 里，把 `is_sm100 or is_sm90` 的判断扩展为 `is_sm100 or is_sm90 or is_sm120`。SM120 走纯 Triton 的 bf16x3 路径（不用 DeepGEMM 的 tf32 分支，因为精度不够）。新增测试 `test_mhc_bf16x3_sm120.py`。
- **新手概念课堂**：SM120 是消费级 Blackwell，它的 tf32 精度不如数据中心卡，所以用 bf16x3（三次 bf16 乘法模拟更高精度）来补偿。就像用三把小刀拼出一把大砍刀。
- **对你有什么影响**：5090 等 SM120 显卡跑 DeepSeek-V4.1 时，可以通过 `SGLANG_OPT_DEEPGEMM_HC_PRENORM=1` 启用优化路径，精度和性能都更好。

### 11. [Doc] diffusion: Fix documented snippets that cannot load a model
- **代码层面**：修正文档里的 import 路径（`from sglang import DiffGenerator` → `from sglang.multimodal_gen import DiffGenerator`），`from_pretrained` 改为关键字参数 `model_path=`，`ImageSamplingParams` 改为 `SamplingParams`。
- **新手概念课堂**：文档里的代码示例如果跑不通，比没有文档更糟糕——新手会以为是自己的环境问题。这个 commit 就是"让文档能跑"。
- **对你有什么影响**：照着文档跑 diffusion 示例不会再报 ImportError。

### 12. [Fix] [NVIDIA] dsv4: Restore SM120/SM121 to the DeepGEMM indexer gates
- **代码层面**：`v41_indexer/__init__.py` 拆分出 `is_sm100()`（严格 SM100，用于 sparse kernel）和 `is_sm100_or_newer()`（SM100+，用于 dense kernel）。`_use_deep_gemm_prefill` 和 `use_deep_gemm_decode` 改用后者，`make_candidate_indexer` 保持前者。新增测试 `test_dsv41_indexer_arch_gates.py`。
- **新手概念课堂**：DeepGEMM 有两种 kernel——dense（支持 SM100+）和 sparse（只支持 SM100）。以前用一个 `is_sm100()` 门控两者，导致 SM120 用户被误判为"不支持"，启动报错。现在拆成两个门控，各管各的。
- **对你有什么影响**：SM120/SM121 用户跑 DeepSeek-V4.1 的 dense indexer 不再报错，sparse 路径仍然安全降级。

### 13. [Feat] cli: Export Cookbook commands as per-checkpoint JSON files
- **代码层面**：新增 `docs/scripts/export_commands.mjs`，从 `docs/src/snippets/configs/` 读取配置，生成 `python/sglang/cli/_commands/` 下的 JSON 文件。`.pre-commit-config.yaml` 新增 `export-cookbook-commands` hook，CI 里也加了 Node 环境。
- **新手概念课堂**：以前 `sglang serve <model>` 的命令是硬编码在 Python 里的；现在从文档的配置源头自动导出，保证"文档里写的"和"CLI 实际执行的"永远一致。
- **对你有什么影响**：`sglang serve qwen3.8-flash-next` 这类命令的默认参数会随文档更新自动同步，不会出现"文档说 A、实际跑 B"的情况。

### 14. [Fix] weight_checker: Compare weights on the live weight's device
- **代码层面**：`weight_checker_comparator.py` 里给 `Comparable` 基类新增 `device` 属性和 `iter_chunks(device)` 参数，把硬编码的 `.cuda()` 改为 `.to(device)`。`compare_weights` 里用 `actual.device` 作为目标设备。
- **新手概念课堂**：以前不管三七二十一都往 `cuda:0` 拷，如果模型在 `cuda:1` 或 CPU 上，就会出错或浪费带宽。现在"跟着活权重走"。
- **对你有什么影响**：多卡或 CPU offload 场景下，权重校验不再因设备不匹配而失败。

### 15. [Feat] [CPU] model: Support Intern-S2-Mobius models on CPU
- **代码层面**：`topk.cpp` 里 `topk_softmax_cpu` 新增 `case 2560`（Intern-S2-Mobius 有 2560 个 expert）。`interns2_mobius.py` 里，CPU 平台不再强制 `moe_runner_backend="triton_kernel"`（因为 CPU FP8 MoE 方法不兼容转置权重）。新增测试。
- **新手概念课堂**：MoE 模型的 expert 数量是编译期常量，kernel 里用 `switch-case` 分发。新增一个模型就要新增一个 case。CPU 平台的 FP8 MoE 实现和 GPU 不同，所以不能盲目套用 GPU 的优化配置。
- **对你有什么影响**：可以在纯 CPU 环境跑 Intern-S2-Mobius 模型了。

### 16. [Feat] [Intel] xpu_kernel: update sglang-kernel-xpu to 0.6.0
- **代码层面**：`pyproject_xpu.toml` 里 `sglang-kernel-xpu==0.5.0` 改为 `0.6.0`。
- **新手概念课堂**：这是"依赖版本升级"，把 Intel XPU 的 kernel 包从 0.5.0 升到 0.6.0，通常带来性能改进和 bug 修复。
- **对你有什么影响**：Intel GPU 用户升级后可能获得更好的性能；无 Intel 设备的用户无感知。

### 17. [Fix] disaggregation: Unregister a failed staging room before releasing its KV
- **代码层面**：`decode.py` 里抽出 `_unregister_staging_room(room)` 方法，在 `pop_transferred_with_outcomes` 的失败路径和 `_do_release` 里都先调用它，再 `release` KV。新增测试验证"先 unregister 再 release"的顺序。
- **新手概念课堂**：staging room 就像"中转仓库"，KV 释放后这块内存会被别的请求复用。如果中转仓库还挂着这个 room 的登记，decode 线程可能往已释放的内存里写数据，污染其他请求的 KV。所以必须"先注销登记，再释放内存"。
- **对你有什么影响**：PD 分离场景下，失败的请求不会污染其他请求的 KV cache，正确性提升。

### 18. [Fix] flashinfer: Apply the sliding window in ragged prefill for hybrid-SWA models
- **代码层面**：`flashinfer_backend.py` 里把单个 `prefill_wrapper_ragged` 改为列表 `prefill_wrappers_ragged`（按 wrapper_id 索引），SLIDING_WINDOW 场景下 wrapper 0 用 `fa2` backend（因为 FlashInfer 的 fmha_varlen 硬编码 `use_sliding_window=False`），并在 `forward` 调用里传 `window_left=layer.sliding_window_size`。
- **新手概念课堂**：Sliding Window Attention（SWA）只看最近 N 个 token，但 FlashInfer 的某个 backend 不支持这个参数。所以对 SWA 层改用另一个 backend（fa2），并显式传 `window_left`。
- **对你有什么影响**：混合 SWA 模型（如 Gemma、Mistral）的 prefill 结果正确了，不再"看到了不该看的 token"。

### 19. [Fix] [NVIDIA] dsv4: Import get_eager_max_batch_size from dp_attention in the mega backend
- **代码层面**：`deepseek_v4_mega_backend.py` 里把 `from sglang.srt.utils.common import get_eager_max_batch_size` 改为 `from sglang.srt.layers.dp_attention import get_eager_max_batch_size`。
- **新手概念课堂**：这是一个"import 路径修正"。函数从 `utils.common` 搬到了 `dp_attention` 模块，旧路径的 import 会报 `ImportError`。
- **对你有什么影响**：修复了 DeepSeek-V4 mega backend 的启动报错。

### 20. [Fix] qwen3.5: Keep CPU-offloaded GDN input projections separate
- **代码层面**：`qwen3_5.py` 的 `finalize_fused_in_proj` 里新增判断：如果 `qkvz.weight` 或 `ba.weight` 在 CPU 上，直接 return（不融合）。新增测试 `test_gdn_input_projections_after_cpu_offload`。
- **新手概念课堂**：CPU offload 会把权重暂存到主机内存，需要时再拷回 GPU。融合操作会创建一个新的 GPU 张量，但这个"派生缓存"不会跟着 offload 走，导致设备不一致。所以 CPU offload 场景下干脆不融合，保持两个独立投影。
- **对你有什么影响**：Qwen3.5 在 CPU offload 场景下不再因设备不匹配而崩溃，正确性提升。

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)