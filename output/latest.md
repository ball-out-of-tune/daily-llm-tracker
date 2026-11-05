# 🚀 vLLM & SGLang 每日更新 — 2026-11-05

> 自动生成于 2026-11-05 08:28 UTC | AI 解读: ✅ 含代码解读

## vllm
## 💡 今日关键词（放在最前面！）

- **关键词：多模板处理器（Multi-template processors）**
  - **一句话通俗解释**：一个模型仓库里可能同时带了多个"对话模板"（比如一个给聊天用、一个给重排序用），vLLM 现在会自动挑"default"那个来用，而不是报错。
  - **为什么社区在关注**：HuggingFace 上越来越多模型（如 Qwen3-VL-Reranker）打包多套模板，说明模型分发正在"一模型多用途"化，推理框架必须更聪明地做默认选择。
  - **对新手意味着什么**：如果你在做模型接入或 tokenizer 相关工作，理解"模板选择"是必修课——选错模板，模型输出会胡言乱语。

- **关键词：在线量化（Online quantization）**
  - **一句话通俗解释**：不用下载已经量化好的模型，而是在加载时现场把权重压成低精度（如 MXFP4/MXFP8），还能单独指定"激活"用哪种精度。
  - **为什么社区在关注**：显存越来越贵，用户想在消费级卡上跑大模型；在线量化 + 激活覆盖让"权重 4bit、激活 8bit"这种混搭成为可能。
  - **对新手意味着什么**：量化是当前推理优化的最大红利方向，学会 `--quantization-config` 就能显著省显存。

- **关键词：ROCm / AMD 适配**
  - **一句话通俗解释**：让 vLLM 在 AMD GPU（MI300/MI350/gfx950）上跑得又快又对。
  - **为什么社区在关注**：今天 20 个 commit 里有 5 个是 ROCm 相关（KV cache dtype、Kimi-K3、DSv4.1 采样、top-k 优化），说明 AMD 已成为一等公民。
  - **对新手意味着什么**：异构计算是明确趋势，懂 CUDA 再学点 HIP/ROCm，就业面直接翻倍。

- **关键词：融合内核（Fused kernels）**
  - **一句话通俗解释**：把原本要跑好几趟 GPU 的小算子（投影 + 门控 + 混合）合并成一个 kernel，减少显存读写。
  - **为什么社区在关注**：Qwen4Exp 的 HyperConnection 融合、DSv4.1 的 fused Markov 采样，都是"prefill 长上下文"场景下的关键提速。
  - **对新手意味着什么**：想进推理框架团队，CUDA/Triton 内核优化是最硬的敲门砖。

- **关键词：投机解码（Speculative decoding / MTP）**
  - **一句话通俗解释**：让小模型先"猜"几个 token，大模型再一次性验证，猜对了就白赚速度。
  - **为什么社区在关注**：ERNIE-5.5-Flash 的 recipe 把 MTP 作为默认开启项，说明投机解码已从"实验特性"变成"生产标配"。
  - **对新手意味着什么**：理解 draft/verify 流程是读懂现代推理引擎的钥匙。

---

## 🧪 重要知识点 & 动手实验

### 知识点 1：对话模板（Chat Template）的选择机制
- **是什么**：Chat template 是一段 Jinja 模板，把 `[{"role":"user","content":"Hello"}]` 渲染成模型认识的字符串（如 `<|im_start|>user\nHello<|im_end|>`）。
- **为什么重要**：模板错了，模型不是"答得差"，而是"完全不会答"。当一个模型带多个模板（chat / reranker）时，选错就等于用重排序模板去做聊天，输出乱码。
- **动手试试**：
```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-Reranker-2B")
print(type(tok.chat_template))          # 可能是 dict
print(tok.chat_template.keys())          # 看有哪些模板
print(tok.apply_chat_template(
    [{"role": "user", "content": "Hello"}],
    tokenize=False, add_generation_prompt=True,
))
```
- **预期结果**：你会看到 `chat_template` 是个 dict（含 `default` 和 `reranker` 等 key），渲染结果是 `<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n`。如果直接 `tok.chat_template` 当字符串用会报错——这正是 Commit 1 修的问题。

---

### 知识点 2：在线量化与"激活精度覆盖"
- **是什么**：权重（weight）和激活（activation）可以分别选精度。比如权重 MXFP4（省显存），激活 MXFP8（保精度）。
- **为什么重要**：纯 4bit 权重 + 4bit 激活精度损失大；纯 8bit 又不够省。混搭是甜点区。显式写 `"activation": null` 表示"激活不量化，用 BF16"。
- **动手试试**：
```bash
# 先看帮助，确认你的 vLLM 版本支持
vllm serve --help | grep -A2 quantization-config

# 尝试在线量化（需要支持的 GPU，如 H100/B200）
vllm serve Qwen/Qwen3.5-35B-A3B \
    --quantization online \
    --quantization-config '{"moe":{"weight":"mxfp4","activation":"mxfp8"}}' \
    --max-model-len 2048
```
- **预期结果**：启动日志里会打印每个 linear/MoE 层实际用的 `(weight, activation)` 组合；对比不加 `--quantization-config` 时的显存占用，应明显下降。若你的卡不支持 MXFP4，会看到明确的报错而不是静默降级。

---

### 知识点 3：投机解码中的 `n > 1` 与有状态输出解析
- **是什么**：`n=2` 表示一次请求采样两条候选回答。输出解析器（tool call 解析、reasoning 解析）是**有状态**的，所以每条候选都需要独立的解析器实例。
- **为什么重要**：如果共享一个解析器，两条回答的解析状态会串台，导致 tool call 被错误拼接。Commit 9 在 Rust 前端里给每个 sample 建了一个 processor。
- **动手试试**：
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Say hi in one word."}],
    n=3,
    temperature=0.8,
)
for i, choice in enumerate(resp.choices):
    print(i, choice.index, repr(choice.message.content))
```
- **预期结果**：你会看到 3 个 `choice`，`index` 分别 0/1/2，内容各不相同。如果开启 tool call 解析，每条都应该有独立的 `tool_calls`，不会互相污染。

---

### 知识点 4：MoE 激活函数中的 clamp（截断）
- **是什么**：SwiGLU 系列激活在 MoE 里可以带一个 `clamp_limit`，把输入截断在 `[-limit, limit]` 防止数值爆炸。Ling3 用 7.0，但配置里可以覆盖。
- **为什么重要**：如果代码把 limit 写死成 7.0，而模型配置要求别的值（比如 10.0），推理结果会悄悄偏掉——不报错，只是"答得不对"，最难查。
- **动手试试**：
```python
import torch
from vllm.model_executor.layers.activation import swiglustep_and_mul_triton

x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16) * 20  # 故意放大
out_default = swiglustep_and_mul_triton(torch.empty(4, 64, device="cuda", dtype=torch.bfloat16), x)
out_clamped = swiglustep_and_mul_triton(torch.empty(4, 64, device="cuda", dtype=torch.bfloat16), x, limit=3.0)
print((out_default - out_clamped).abs().max())
```
- **预期结果**：两者差异明显非零，说明 `limit` 参数真的生效了。这就是 Commit 11 修复的核心——把配置里的 `clamp_limit` 传进 kernel，而不是硬编码 7.0。

---

### 知识点 5：KV cache dtype 的"分组覆盖"
- **是什么**：全局设 `--kv-cache-dtype fp8`，但某些层（如 MLA 的 skip-layer、或投机 draft 模型）可能需要 BF16。分组 spec 里的 dtype 优先级高于全局。
- **为什么重要**：MLA（Multi-head Latent Attention）对 KV 精度敏感，盲目全局 FP8 会让长上下文质量掉点；而投机 draft 模型和 target 模型 dtype 不一致时，metadata 尺寸会算错，直接崩。
- **动手试试**：
```bash
# 全局 fp8，但让某层保持 auto(bf16)
vllm serve deepseek-ai/DeepSeek-V3 \
    --kv-cache-dtype fp8 \
    --kv-cache-dtype-skip-layers '["layers.0.self_attn"]' \
    --max-model-len 4096
```
- **预期结果**：启动日志里能看到第 0 层的 KV cache dtype 是 `auto`/`bf16`，其余是 `fp8`。Commit 10 修的就是"分组 spec 没被读、错用了全局值"的 bug。

---

## 🔥 重要更新

1. **Commit 4（在线量化激活覆盖）**：这是功能级更新，让 MXFP4 权重可以配 MXFP8 激活，并支持 per-layer 的 `{weight, activation}` spec。对显存敏感的用户影响最大。
2. **Commit 9（Rust 前端 n>1）**：Rust 前端补齐了 OpenAI 兼容的 `n` 参数，且正确处理有状态解析器。这是 Rust 前端走向生产可用的一大步。
3. **Commit 3 + 18（SM90 小头 FlashMLA / Qwen4Exp 融合 kernel）**：两个性能向的硬核优化，分别针对 sparse MLA 和长 prefill 的 HyperConnection，属于"跑分直接涨"的类型。
4. **Commit 5（Qwen4Exp FP8 KV scale）**：checkpoint 里存的 FP8 KV scale 之前被忽略，导致精度悄悄劣化。这类"静默错误"修复价值极高。

---

## 📋 逐条解读

### 1. [Bugfix][Frontend] Use the default template of multi-template processors (#63606)
- **代码层面**：改 `vllm/renderers/hf.py` 的 `_try_get_processor_chat_template`。原来直接把 `processor.chat_template` 当字符串用，现在先判断是不是 dict，若是则取 `["default"]`，否则抛 `TypeError`。同时加了两个测试（render 服务和 renderer 单测）。
- **新手概念课堂**：就像你买了一个多功能瑞士军刀，包装上写着"主刀在中间层"。以前 vLLM 拿到整把刀不知道用哪片，现在它知道先找标着 "default" 的那片。
- **对你有什么影响**：用 Qwen3-VL-Reranker 这类多模板模型时，聊天接口不再报错或输出乱码。

### 2. [Docs] Fix missing MLA verify parameter types in API docs (#65834)
- **代码层面**：只改 `vllm/v1/attention/ops/triton_decode_attention.py` 的 docstring，把 `q:` 改成 `q (torch.Tensor):` 这种 Google 风格，让 Sphinx 能正确渲染类型。
- **新手概念课堂**：好比菜谱里写"盐：适量" vs "盐（克）：5g"。后者机器才能自动生成漂亮的文档表格。
- **对你有什么影响**：看 API 文档时能看到参数类型了，写调用代码不用猜。

### 3. [Perf] Enable native small-head kernels in SM90 sparse FlashMLA (#64510)
- **代码层面**：在 `tests/v1/attention/test_sparse_mla_backends.py` 加了大量参数化测试，验证在 SM90 上、head 数小于 64 时（4/8/16/32），native kernel 只写实际 head 的输出，不污染 padding 的 head。
- **新手概念课堂**：想象你有一个 128 格的收纳盒，但只放了 8 个杯子。以前 kernel 会把 128 格全写一遍（包括空的），现在只写有杯子的 8 格——省了 15/16 的写带宽。
- **对你有什么影响**：在 H100/H200 上跑稀疏 MLA 模型（如 DeepSeek 系列）时，小 batch 场景吞吐提升。

### 4. [Online quantization] Support activation quant key override for online MXFP4 linear/MOE quant methods (#51808)
- **代码层面**：改动横跨 `docs/features/quantization/online.md`、`vllm/config/quantization.py`、MXFP4 oracle 等。核心是允许 `--quantization-config` 里显式写 `activation`（包括 `null` 表示 BF16），并让它覆盖 checkpoint 自带的激活格式；同时 `targets` 支持 `{weight, activation}` 形式。
- **新手概念课堂**：以前是"套餐制"——点了 MXFP4 就全套 4bit；现在是"自助餐"——权重可以 4bit，激活可以 8bit 或 BF16，还能按层点。
- **对你有什么影响**：显存不够时可以用 `{"moe":{"weight":"mxfp4","activation":null}}` 保住 MoE 精度；文档里也给了 Qwen3.5-35B-A3B 的现成命令。

### 5. [Bugfix] Honor checkpoint FP8 KV scales in Qwen4Exp QSA (#65812)
- **代码层面**：在 `vllm/models/qwen4_exp/{amd,nvidia}/qsa.py` 的 attention 类里新增 `process_weights_after_loading`，把 `_k_scale` / `_v_scale` 转成 Python float 缓存。之前没转，导致 FP8 KV 的 scale 没被真正用上。
- **新手概念课堂**：FP8 存 KV 时，每个数都被压缩过，解码时要乘一个"还原系数"（scale）。checkpoint 里存了正确的系数，但代码没读，等于用错系数还原——数值全偏。
- **对你有什么影响**：Qwen4Exp 系列模型开 FP8 KV cache 后，输出质量恢复正常（之前是静默劣化）。

### 6. [Bugfix][LoRA] Apply DeepSeek-V4 compressor and wo_a adapters (#64880)
- **代码层面**：主要是测试文件改动，把 import 从 `deepseek_v41` 改成 `deepseek_v4`，并新增 `test_dsv4_compressor_projections_apply_lora_with_fp32_output`，验证 compressor 的 FP32 输出路径上 LoRA adapter 真的被应用。
- **新手概念课堂**：LoRA 像给模型戴"滤镜"。如果某个模块的代码路径绕过了滤镜（比如先算 FP32 再转），滤镜就白戴了。这个 commit 保证 compressor 模块不会绕过。
- **对你有什么影响**：用 DeepSeek-V4 + LoRA 微调时，compressor 和 wo_a 部分的微调效果真正生效。

### 7. [Build] Update DeepGEMM pin to latest main (#65494)
- **代码层面**：`cmake/external_projects/deepgemm.cmake` 里把 `_DEEPGEMM_UPSTREAM_TAG` 从 `ae4fd25d...` 换成 `82ca617e...`，注释说明现在需要 PyTorch 2.14+ 头文件。
- **新手概念课堂**：就像你手机里的某个 App 依赖某个系统版本。DeepGEMM 升级了，要求更新的 PyTorch 头文件，构建时得对齐。
- **对你有什么影响**：从源码编译 vLLM 时，如果 PyTorch 版本太老会编译失败——需要升到 2.14+。

### 8. [Recipe] Add ERNIE-5.5-Flash recipe (#65817)
- **代码层面**：新增 `recipes/models/baidu/ERNIE-5.5-Flash.yaml`，680 行，包含 280B/14B-active MoE、HCA+MLA-DSA 混合注意力、ECHO(MTP) 投机解码、1M 上下文、B300/GB300/B200/GB200/H100 的 FP8/BF16 配置。
- **新手概念课堂**：Recipe 就像"菜谱"——告诉你用什么硬件、装什么插件、加什么启动参数，照着抄就能跑起来。ERNIE 5.5 需要单独的 `vLLM-ERNIE` 插件。
- **对你有什么影响**：想部署百度 ERNIE-5.5-Flash 的话，直接照 YAML 抄命令，不用自己摸索参数。

### 9. [Rust Frontend] Support n > 1 choices (#65625)
- **代码层面**：`rust/src/chat/src/lib.rs` 的 `ChatRequestProcessor::prepare` 新增 `n: NonZeroU32` 参数，为每个 sample 创建一个独立的 output processor（因为解析有状态），grammar 只从第一个取（它们相同）。`PreparedChatRequest` 从单个 processor 改成 `Vec`。
- **新手概念课堂**：`n=3` 就像让三个学生同时答题。每个学生需要自己的草稿纸（解析器状态），不能共用一张，否则答案会串。
- **对你有什么影响**：用 Rust 前端（`vllm-rs`）时，`n>1` 请求不再报错或串台，与 Python 前端行为一致。

### 10. [Bugfix][ROCm] Take the AITER MLA KV cache dtype from the group's spec (#63117)
- **代码层面**：测试文件 `test_rocm_aiter_mla_mtp_split.py` 大幅扩展参数化，覆盖"draft 模型 BF16 KV + target FP8"、"skip-layer"、"group dtype 覆盖全局"等场景，验证 builder 按分组 spec 而非全局配置初始化。
- **新手概念课堂**：全局说"大家都穿 FP8 衣服"，但某个分组说"我们这组穿 BF16"。以前代码只听全局的，现在听分组的。
- **对你有什么影响**：在 AMD GPU 上用 MLA + 投机解码 + 混合 KV dtype 时，不再因 metadata 尺寸算错而崩溃。

### 11. [Bugfix][Model] Fix Ling3 SwiGLU clamping (#65088)
- **代码层面**：`vllm/model_executor/layers/fused_moe/activation.py` 里把 `masked_clamp_limit = 7.0` 改成 `7.0 if config.clamp_limit is None else config.clamp_limit`；`apply_moe_activation` 里也把 `limit` 传给 `swiglustep_and_mul_triton`。`bailing_moe_v3.py` 里根据 `expert_swiglu_limit` 是否为 None 选择 `swiglustep` 或 `silu`。
- **新手概念课堂**：clamp 就像给音量加个上限，防止爆音。Ling3 的配置说"上限是 X"，但代码写死 7.0，等于无视配置。
- **对你有什么影响**：跑 Ling3 / Bailing-MoE-V3 时，激活值按模型作者设定的上限截断，输出质量恢复预期。

### 12. [Bugfix][ROCm][Kimi-K3] Refresh derived weight copies after in-place weight writes (#64886)
- **代码层面**：`vllm/distributed/weight_transfer/sparse_nccl_engine.py` 在 `finish_weight_update` 里调用新增的 `refresh_derived_weights`；测试里验证 `reload_weights` 后 `decode_conv1d_weight` 等派生 buffer 的地址不变但内容刷新。
- **新手概念课堂**：你有一份"原始文件"和一份"整理好的副本"。原地改原始文件后，副本不会自动更新。这个 commit 保证改完原始文件后，副本也刷新（且地址不变，因为 CUDA graph 抓的是地址）。
- **对你有什么影响**：Kimi-K3 在 AMD 上做权重热更新（如 RL 训练后 reload）时，decode 不会读到旧数据或 NaN。

### 13. [ROCm][Perf][DSv4.1] Run DSpark's fused Markov sampling on gfx950 with --monokernel (#63891)
- **代码层面**：`vllm/models/deepseek_v41/amd/ops.py` 里加了 `monokernel_cdna3_or_4` 判断，让 gfx950（CDNA4）也能走 fused Markov 采样；测试覆盖 cdna3/cdna4、monokernel 开关、不同 rows 数的组合。
- **新手概念课堂**：Markov 采样是"根据上一个 token 决定下一个"的链式采样。fused kernel 把多步合成一步，省掉中间显存往返。gfx950 是新硬件，之前没开。
- **对你有什么影响**：在 MI350（gfx950）上跑 DeepSeek-V4.1 的投机解码，采样环节更快。

### 14. [ROCm][Perf][DSv4.1] Skip the top-k -1 fill on the rows vLLM's decode top-k writes (#63888)
- **代码层面**：`vllm/v1/attention/ops/rocm_paged_mxfp4_indexer.py`（及相关测试）里，top-k kernel 不再对"会被覆盖的行"预填 -1，只在 padding 行填 -1。测试新增 `pad` / `pool_blocks` 参数验证尾部行为。
- **新手概念课堂**：就像你先在白板上写满 -1，再擦掉重写有效值。既然有效值一定会覆盖，何必先写 -1？省掉这次写就是省带宽。
- **对你有什么影响**：AMD 上跑 MXFP4 稀疏注意力 indexer 时，top-k 环节更快，且 padding 行仍是 -1（语义不变）。

### 15. [Bugfix][MiniMax-M3] Initialize the shard record in the indexer QKV layer (#65806)
- **代码层面**：`vllm/model_executor/layers/linear.py` 里 `MinimaxM3QKVParallelLinearWithIndexer` 的父类从 `ColumnParallelLinear.__init__` 改成 `StackedColumnParallelLinear.__init__`，这样 shard 追踪记录才会被初始化。
- **新手概念课堂**：Stacked 层像"多层抽屉"，每层要登记"我装了哪些东西"。之前这个层跳过了登记步骤，导致加载时以为"什么都没装"，报 missing shards。
- **对你有什么影响**：MiniMax-M3 加载时不再报"权重未加载"的误报，indexer 的 Q/K 分片正确归位。

### 16. [CI Failure] Move Multimodal Models (Extended Pooling) to h200_35gb (#65776)
- **代码层面**：`.buildkite/test_areas/models_multimodal.yaml` 里把 `device: h200_18gb` 改成 `h200_35gb`，label 同步更新。
- **新手概念课堂**：CI 机器就像测试实验室。18GB 显存的 MIG 切片跑不动扩展 pooling 的多模态测试了，换 35GB 的。
- **对你有什么影响**：普通用户无感；贡献者提交 PR 时这个测试不再因 OOM 随机挂掉。

### 17. [XPU] skipp unsupported test cases in CI (#65805)
- **代码层面**：`.buildkite/intel_jobs/entrypoints_intel.yaml` 加 `--deselect` 跳过两个 derender parity 测试；`models_multimodal_intel.yaml` 加 `--ignore models/multimodal/generation/test_unlimited_ocr.py`。
- **新手概念课堂**：Intel GPU（XPU）上有些测试暂时跑不了，与其让 CI 红一片，不如明确跳过并记录。
- **对你有什么影响**：Intel 平台 CI 更稳定；这些被跳过的功能在 XPU 上暂时不可用。

### 18. [Perf][Qwen4Exp] Fuse HyperConnection up projection and gate mix for long prefills (#64898)
- **代码层面**：新增 `csrc/qwen4_exp/hc_up_gate_mix.cu`（573 行 CUDA），在 SM100（B200）上把 `up projection` 和 `gate mix` 融合成一个 persistent kernel；`CMakeLists.txt` 加编译开关；新增 B200 CI job 和 `test_hc_ops.py`。
- **新手概念课堂**：HyperConnection 的 gate 计算原本要写一个 `[M, 4*2560]` 的中间矩阵到显存再读回来。融合后这个矩阵只在 TMEM（片上）里，省掉两次大块显存往返。
- **对你有什么影响**：Qwen4Exp 在 B200 上跑长 prefill（长 prompt）时更快，尤其是 batch 大、序列长的场景。

### 19. [Bugfix] Granite4Vision: strip padding tiles for list pixel_values (#65412)
- **代码层面**：新增测试 `test_granite4_vision.py`，验证当一批请求里的图片 tile 数不同（5/10/7）时，`pixel_values` 会变成"长度不一的 tensor 列表"，模型输出必须与单独跑一致。修复逻辑在模型侧（strip padding tiles）。
- **新手概念课堂**：HF processor 会把 5 个 tile 的图 pad 到 10 个 tile 凑齐 batch。但如果同批还有 7 个 tile 的图，就没法拼成一个规整 tensor，只能传 list。模型得自己把 pad 的部分去掉。
- **对你有什么影响**：用 Granite-4.1-Vision 批量处理不同尺寸图片时，输出不再因 padding tile 而偏移。

### 20. [Perf][Qwen4Exp] Route the PLE shard copy through copy_weight_ (#65692)
- **代码层面**：`vllm/models/qwen4_exp/common/ple.py` 里把 `target.copy_(source.to(device=..., dtype=...))` 换成 `copy_weight_(target, source)`。
- **新手概念课堂**：`copy_weight_` 是 vLLM 封装的"智能拷贝"——自动处理 device/dtype 转换、可能走更快路径。手写 `.to().copy_()` 会多一次临时分配。
- **对你有什么影响**：Qwen4Exp 的 PLE（Parallel Lookup Embedding）分片加载更快，模型启动时间略降。

## sglang
## 💡 今日关键词（放在最前面！）

- **关键词：AMD/ROCm 生态适配**
  - 一句话通俗解释：让 SGLang 能在 AMD 的 GPU（如 MI300、gfx950）上跑得更快、更稳。
  - 为什么社区在关注：今天 20 个 commit 里有 5 个和 AMD 相关（RCCL 固定版本、Mooncake 升级、AITER 确定性推理、Kimi-K3 monokernel 融合），说明 SGLang 正在把"支持 NVIDIA + AMD 双栈"当作一等公民，异构计算不再是副业。
  - 对新手意味着什么：如果你只会 CUDA，建议补一下 HIP/ROCm 的基础（`hipcc`、`rocm-smi`），AMD 生态的人才缺口正在扩大。

- **关键词：确定性推理（Deterministic Inference）**
  - 一句话通俗解释：同样的输入，无论怎么切分、怎么并行，输出永远逐 bit 一致。
  - 为什么社区在关注：`_deterministic_aiter_prefill_page_size` 这个 commit 显示团队在认真做"可复现推理"，这对金融、医疗、审计类场景是硬需求。
  - 对新手意味着什么：理解"batch invariance"（批次不变性）是进阶推理优化的关键概念，值得专门研究。

- **关键词：Triton 融合算子自动调优（Autotuning）**
  - 一句话通俗解释：为每种 GPU 型号、每种 batch size 预先找到最快的 kernel 参数组合，存成 JSON 配置。
  - 为什么社区在关注：L4 显卡的 Qwen3-30B-A3B MoE 配置被单独提交，说明"按设备 + 按模型"细粒度调优正在成为常态。
  - 对新手意味着什么：学会读/写 Triton autotune 配置是进入推理性能优化领域的敲门砖。

- **关键词：多模态调度（Multimodal Scheduling）**
  - 一句话通俗解释：把多个请求里的图片/视频打包成一批，一起送进视觉编码器（ViT）推理。
  - 为什么社区在关注：NPU 后端这次"对齐 CUDA 的跨请求 ViT 批处理"，说明多模态推理的吞吐优化正在从 GPU 扩展到 NPU。
  - 对新手意味着什么：多模态 + 调度是当前最热的交叉方向，懂 vLLM/SGLang 调度器的人很吃香。

- **关键词：FIPS 合规与安全**
  - 一句话通俗解释：让路由器的 TLS 加密使用美国政府认证的加密模块（aws-lc-rs FIPS）。
  - 为什么社区在关注：企业级部署（金融、政府）必须过 FIPS 140-3，SGLang 开始补齐这块短板。
  - 对新手意味着什么：Rust + 密码学合规是冷门但高薪的方向，值得关注 `sgl-router` 的演进。

---

## 🧪 重要知识点 & 动手实验

- **知识点 1：CUDA Graph 与"活 KV 长度"（Live KV Lengths）**
  - **是什么**：CUDA Graph 是"把一串 GPU 操作录下来，之后一键重放"的技术。但录制时如果按最大序列长度分配 workspace，重放短请求时就会浪费算力。这个 commit 让 FA4 在重放时改用"当前真实的 KV 长度"来切分 split。
  - **为什么重要**：不懂这个会踩的坑——你以为开了 CUDA Graph 就快了，结果短请求反而更慢，因为 split 粒度按最大长度固定，导致大部分 split 空转。
  - **动手试试**：
    ```bash
    # 观察 CUDA Graph 是否生效
    python -m sglang.launch_server --model Qwen/Qwen2.5-0.5B-Instruct \
      --enable-cuda-graph --max-running-requests 8
    # 在另一个终端发短请求，看日志里 "cuda graph" 相关行
    curl http://localhost:30000/generate -d '{"text":"Hi","max_new_tokens":4}'
    ```
  - **预期结果**：日志显示 graph capture 成功，短请求延迟明显低于未开 graph 的基线。若你手动改 `select_paged_decode_split_kv_blocks_per_cta` 让它返回固定值，短请求会变慢——这就是这个 commit 修复的问题。

- **知识点 2：MoE 的 Triton 融合 kernel 与 autotune 配置**
  - **是什么**：MoE（混合专家）把 FFN 拆成多个"专家"，每个 token 只走其中几个。融合 kernel 把"路由 + 分组 GEMM"合成一次 GPU 调用，参数（BLOCK_SIZE_M/N/K、num_warps 等）需要针对每张卡调优。
  - **为什么重要**：不懂会踩的坑——直接用默认参数，在 L4 这种小卡上可能慢 2-3 倍；而且不同 batch size 最优参数完全不同。
  - **动手试试**：
    ```bash
    # 查看已有哪些设备的调优配置
    ls python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_8_0/
    # 跑一个 MoE 模型，观察是否命中 L4 配置
    python -m sglang.launch_server --model Qwen/Qwen1.5-MoE-A2.7B --device cuda
    ```
  - **预期结果**：在 L4 上跑 Qwen3-30B-A3B 时，日志会显示加载了 `E=128,N=192,device_name=NVIDIA_L4.json`，吞吐比不加载时提升。

- **知识点 3：确定性推理与 batch invariance**
  - **是什么**：同一个 prompt，无论你单独跑还是和别的请求拼批跑，无论 chunked prefill 怎么切，输出 token 必须完全一致。AITER 的 prefill kernel 只有在 query 被切成 128 的倍数时才 batch-invariant，所以这个 commit 强制 `page_size=128` 且 `chunked_prefill_size` 必须是 128 的倍数。
  - **为什么重要**：不懂会踩的坑——做 A/B 测试时，两次跑同一个 prompt 结果不同，你会以为是模型 bug，其实是 kernel 的 batch 依赖。
  - **动手试试**：
    ```bash
    # 开启确定性推理（需要 gfx950）
    python -m sglang.launch_server --model meta-llama/Llama-3.1-8B \
      --enable-deterministic-inference --prefill-attention-backend aiter \
      --chunked-prefill-size 1024
    # 故意传一个不是 128 倍数的 chunk size，观察报错
    python -m sglang.launch_server --model meta-llama/Llama-3.1-8B \
      --enable-deterministic-inference --prefill-attention-backend aiter \
      --chunked-prefill-size 1000
    ```
  - **预期结果**：第二个命令启动时报 `ValueError`，提示 chunked_prefill_size 必须是 128 的倍数；第一个命令能正常启动且两次请求输出完全一致。

- **知识点 4：CUDA Graph 下的 buffer 所有权（NPU 版本）**
  - **是什么**：NPU 的 ViT graph runner 在 capture 时如果直接把调用者的 tensor 当输入，capture 过程会原地修改它；重放时返回的 tensor 又会被下一次重放覆盖。修复方法是 capture 时 clone 输入、replay 时 clone 输出。
  - **为什么重要**：不懂会踩的坑——你以为 graph 跑完拿到了结果，实际上那个 tensor 在下一次 graph 重放时被悄悄改掉了，导致偶发的、极难复现的数值错误。
  - **动手试试**：
    ```python
    # 概念验证：模拟 graph 重放覆盖输出
    import torch
    pool = torch.zeros(4, 4)
    def fake_replay(out_view, x):
        out_view.copy_(x * 2)   # 原地写
        return out_view          # 返回的是 pool 的视图
    a = fake_replay(pool[0:1], torch.ones(1, 4))
    b = fake_replay(pool[0:1], torch.full((1, 4), 3.0))
    print(a, b)  # a 和 b 是同一块内存！
    ```
  - **预期结果**：`a` 打印出来是 `[[6,6,6,6]]` 而不是 `[[2,2,2,2]]`——这就是"输出被覆盖"的 bug。修复方法是在 replay 里 `return out.clone()`。

- **知识点 5：RoPE 表的设备一致性（CPU offload 数值一致性）**
  - **是什么**：Cosmos 3 视频模型的 RoPE（旋转位置编码）表原来在 meta device 上构建，再 cast 到 GPU；现在改成直接在 `get_local_torch_device()` 上算。CPU 和 GPU 的 `logspace`/`pow` 在 FP32 上可能有 1 ulp 差异，导致 CPU offload 和全 GPU 跑出来的视频不一样。
  - **为什么重要**：不懂会踩的坑——你在 A100 上跑出满意的视频，换到 CPU offload 模式复现，画面细节变了，你会怀疑是采样随机性，其实是 RoPE 表数值不同。
  - **动手试试**：
    ```python
    import torch
    dim, theta = 128, 10000.0
    # CPU 路径
    cpu = (1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)))
    # CUDA 路径（logspace + reciprocal）
    gpu = torch.logspace(0, (dim-2)/dim, dim//2, base=theta,
                         dtype=torch.float32, device='cuda').reciprocal().cpu()
    print((cpu - gpu).abs().max())  # 看差异
    ```
  - **预期结果**：差异通常为 0 或 1 ulp；若你的模型对这点差异敏感（如视频生成），就会出现"CPU offload 结果不同"的现象。这个 commit 的做法是统一在 runtime device 上算，再 `.to(device)`。

---

## 🔥 重要更新

1. **#48325 AWS EFA 并入主镜像**：删除了独立的 `efa.Dockerfile` 和 287 行的 release workflow，EFA 用户不再需要特殊镜像标签，部署复杂度大幅下降。
2. **#48501 DCP-sharded indexer 的 kernel 前置条件**：为 DeepSeek-V4 的 FlashMLA 加了并行 walk 调度器（pointer doubling 找 partition start），这是 DSA（DeepSeek Sparse Attention）走向生产的关键一步。
3. **#48066 sgl-router 支持 FIPS**：路由器可以编译成 FIPS 140-3 合规版本，企业/政府客户终于能过安全审计。
4. **#44435 NPU 跨请求 ViT 批处理对齐 CUDA**：NPU 上多模态推理吞吐有望追平 CUDA，同时修复了 graph buffer 所有权 bug（这类 bug 极难排查）。

---

## 📋 逐条解读

### 1. [Feat] docker: Install AWS EFA in the main image; drop the EFA image variant (#48325)
- **代码层面**：删除了 `.github/workflows/release-docker-efa.yml`（287 行）和 `docker/efa.Dockerfile`，把 EFA 驱动直接装进主镜像 `docker/Dockerfile`。
- **新手概念课堂**：EFA（Elastic Fabric Adapter）是 AWS 的"高速网卡"，类似 NVIDIA 的 InfiniBand。以前要单独维护一个"EFA 特供镜像"，现在合并进主镜像，就像把"川菜专用锅"并进"万能锅"——少维护一套。
- **对你有什么影响**：在 AWS 上用 SGLang 做多机推理，直接 `docker pull lmsysorg/sglang:vX.Y.Z` 就行，不用再找 `-efa` 后缀的 tag。

### 2. [Perf] diffusion: Avoid upscaling native H3 references by default (#47907)
- **代码层面**：`minimax_h3.py` 里把 `reference_resize_mode` 的默认值从 `"diffusers"` 改成 `"max"`，并新增常量 `MINIMAX_H3_DEFAULT_REFERENCE_RESIZE_MODE`；文档和前端配置同步更新。FastH3 Omni 保留旧默认。
- **新手概念课堂**：参考图放大（upscaling）就像把 480p 照片硬拉到 2048p——细节是"编"出来的，反而拖慢速度、污染条件。新默认"只缩不放"，更尊重原图。
- **对你有什么影响**：用 MiniMax-H3 做 Ref2VA 时，小图不再被无谓放大，速度更快、效果更接近原图；想复现旧行为需显式传 `reference_resize_mode: "diffusers"`。

### 3. [Perf] [NVIDIA] moe: Add L4 Triton fused MoE configs for Qwen3-30B-A3B (#48166)
- **代码层面**：新增 `E=128,N=192,device_name=NVIDIA_L4.json`，为 128 专家、192 中间维度的 MoE 在 L4 上按 batch size 提供调优参数（BLOCK_SIZE_M/N/K、GROUP_SIZE_M、num_warps、num_stages）。
- **新手概念课堂**：autotune 配置就像"菜谱"——不同食材（batch size）配不同火候（block size）。L4 是小卡，默认参数是为大卡设计的，直接套用会"糊锅"。
- **对你有什么影响**：在 L4 上跑 Qwen3-30B-A3B 的 MoE 推理，吞吐会有可感知提升，无需手动调参。

### 4. [Perf] attention: Use live KV lengths for SM12x FA4 graph replay (#47815)
- **代码层面**：`fa4_sm120/runtime.py` 的 `select_paged_decode_split_kv_blocks_per_cta` 新增 `is_stream_capturing` 参数；capture 时返回 0（让 kernel 用活长度），eager 时保留旧逻辑。新增测试 `test_sm120_paged_decode_graph_splits_follow_replay_lengths`。
- **新手概念课堂**：CUDA Graph 录制时按"最大可能长度"分配 workspace，但重放时如果请求很短，固定 split 会让大部分 SM 空转。改成"按活长度切"就像餐厅按实际客人数摆桌，而不是永远摆满。
- **对你有什么影响**：在 SM12x（如 RTX 50 系 / B 系列）上，开 CUDA Graph 后短请求的解码速度提升，不再被长请求的 workspace 拖累。

### 5. [Fix] diffusion: Build the Cosmos 3 RoPE table on GPU so CPU offload gives the same output (#47442)
- **代码层面**：`cosmos3video.py` 的 `post_load_weights` 重写 RoPE `inv_freq` 构建逻辑：在 `get_local_torch_device()` 上用 `logspace`（2 的幂维度）或 `arange` 计算，再 `.to(device)`；不再依赖 meta device 上的 `is_meta` 判断。
- **新手概念课堂**：RoPE 是"位置编码的旋转"，表里的每个数都必须和训练时一致。CPU 和 GPU 算 `pow` 可能有 1 ulp 差异，就像两个厨师用不同秤称盐——菜谱一样，味道微差。
- **对你有什么影响**：Cosmos 3 视频模型在 CPU offload 模式下，输出与全 GPU 模式数值一致，可复现性提升。

### 6. [Chore] [AMD] rccl: pin rccl to d2e0de35 (#47940)
- **代码层面**：`docker/rocm.Dockerfile` 里 `RCCL_COMMIT_DEFAULT` 从 PR 分支 commit `1c0ab425` 改为 develop 主线 commit `d2e0de35`，因为 gfx1250 首次 all-reduce 挂起的修复已合入主线。
- **新手概念课堂**：RCCL 是 AMD 版的 NCCL（集合通信库）。之前为了一个 bug 修复临时钉在 PR 分支，现在修复进主线了，就"回归正轨"。
- **对你有什么影响**：AMD GPU 用户用官方 ROCm 镜像时，通信更稳定，且不再依赖未合并的 PR 分支。

### 7. [Fix] mamba: Make the lazy full-pool KL test hold across GPUs (#48571)
- **代码层面**：`test_mamba_cache_kl.py` 把 CI runner 从 `1-gpu-large` 改成 `1-gpu-small`；重构测试逻辑，抽出 `_resume_at_192` 辅助函数，用 `random` 生成不同前缀，避免测试对特定 GPU 的数值敏感。
- **新手概念课堂**：KL 散度衡量两个概率分布的差异。测试"缓存复用后分布是否一致"，但不同 GPU 的浮点误差会让严格的 KL=0 断言偶尔失败——所以要把测试写得"鲁棒"。
- **对你有什么影响**：Mamba 缓存复用逻辑的回归测试更稳定，CI 不再因 GPU 型号不同而 flaky。

### 8. [Feat] [NPU] scheduler: align cross-request ViT batching with CUDA and fix NPU graph buffer ownership (#44435)
- **代码层面**：`vit_npu_graph_runner.py` 中 capture 时 `block_input = x_3d.clone()`（避免原地修改调用者 tensor），replay 时 `return out.clone() if output_indices is None`（避免输出被下次重放覆盖）；`mm_schedule.py` 把 NPU 从"逐请求编码"分支移除，改为与 CUDA 一致的跨请求批处理。
- **新手概念课堂**：graph capture 就像"录像"，如果录像过程中演员（输入 tensor）被换了衣服，录出来的就是错的；replay 返回的 tensor 如果指向共享内存池，下次录像会覆盖它。clone 就是"拷贝一份再录/再给"。
- **对你有什么影响**：NPU 上多模态推理吞吐提升（跨请求批处理），且不再出现偶发的数值错误。

### 9. [Feat] dsa: Kernel prerequisites for a DCP-sharded indexer (#48501)
- **代码层面**：`flashmla_sched_meta.cuh` 新增 `Schedule` 枚举（serial/parallel/windowed）、三种 schedule 的 shared memory 计算函数、`block_exclusive_scan`、`last_le_index`（二分查找）、pointer doubling 找 partition start；`Params` 增加 `max_states` 字段。
- **新手概念课堂**：DSA（DeepSeek Sparse Attention）需要"索引器"决定哪些 KV 块值得看。DCP-sharded 意味着索引器本身被切分到多卡，调度器要在分片间协调。pointer doubling 是并行算法经典技巧：一次跳 1 步、2 步、4 步……O(log n) 找到所有 partition 起点。
- **对你有什么影响**：DeepSeek-V4 长上下文推理的调度开销降低，为后续 DCP 分片索引器铺路（普通用户暂时感知不到，但会体现在长文本吞吐上）。

### 10. [Fix] qwen4: Release loader parameter snapshot before quantization post-processing (#38900)
- **代码层面**：`qwen4_exp.py` 里把 `load_qwen4_exp_ple_shard._warned_downcast` 这个函数属性改成闭包变量 `ple_warned_downcast`（用 `nonlocal`），避免函数属性在多次 `load_weights()` 调用间残留状态。
- **新手概念课堂**：函数属性是"挂在函数对象上的全局变量"，多次调用会共享——就像在公共黑板上写"已警告"，第二次调用看到就不再警告了，即使换了新一批权重。闭包变量每次调用重新初始化，更干净。
- **对你有什么影响**：Qwen4 加载 PLE 权重时，fp8 降精度警告每次 load 都会正确打印，不会被上一次的残留状态吞掉。

### 11. [Fix] [NPU] diffusion: Auto-select FA for MXFP8 attention (#45550)
- **代码层面**：`attention/layer.py` 三处 `get_attn_backend` 调用新增 `is_cross_attention` 和 `quant_config` 参数；`selector.py` 在 `selected_backend is None` 且 `quant_config` 存在且非 cross-attention 时，调用 `resolve_mx_fa_scheme` 判断是否自动选 `FA` 后端。新增测试 `test_mxfp8_attention_selection.py`。
- **新手概念课堂**：MXFP8 是微缩放 FP8 量化格式，Ascend NPU 上的 FA（Flash Attention）实现支持它。以前用户要手动指定 `--attention-backend FA`，现在检测到 MXFP8 量化配置就自动切。
- **对你有什么影响**：在 Ascend NPU 上用 MXFP8 量化跑扩散模型，不用手动选后端，开箱即用。

### 12. [Perf] [AMD] Kimi-K3: prepare the attention interface for mono fusion (#45202)
- **代码层面**：`pyproject_other.toml` 升级 `sgl-amd-mono` pin；`kimi_k3.py` 把 `forward` 拆成 `forward_pre_o_proj`（返回 pre-o_proj 结果）和新的 `forward`（调用前者再走 o_proj）；新增 `_pre_o_proj_out`、`_return_pre_o_proj` 状态，让 `_gated_o_proj_forward` 可以跳过 o_proj 并返回 pre 结果。
- **新手概念课堂**：monokernel 是把多个小 kernel 融合成一个大 kernel（减少 launch 开销）。融合需要"接口预留"——先把 o_proj 前的中间结果暴露出来，后续 monokernel 才能接管这一步。
- **对你有什么影响**：AMD gfx950 上跑 Kimi-K3 时，开启 `--enable-amd-monokernel` 后 attention 部分可能更快（需配合新版 `sgl-amd-mono`）。

### 13. [Doc] cookbook: Add ERNIE-5.5-Flash deployment recipes (#48546)
- **代码层面**：新增 `docs/cards/logos/ernie.svg` 和 `docs/cookbook/autoregressive/Baidu/ERNIE-5.5-Flash.mdx`（232 行），包含 B300/GB300/B200/GB200/H100 的 FP8/BF16 部署矩阵、ECHO 投机解码、HiCache、PD 分离等配置。
- **新手概念课堂**：Cookbook 就是"菜谱"——告诉你什么硬件、什么精度、几张卡、用什么参数。ERNIE-5.5-Flash 是 280B 总参 / 14B 激活的 MoE，混合 HCA + MLA-DSA 注意力。
- **对你有什么影响**：想在百度 ERNIE-5.5-Flash 上部署 SGLang，直接照抄 cookbook 的命令即可，不用自己试错显存预算。

### 14. [Doc] [XPU] Cookbook: Update the XPU Docker version to 0.5.23 (#48104)
- **代码层面**：`docs/docs/hardware-platforms/xpu.mdx` 里 `docker pull lmsysorg/sglang:v0.5.22-xpu` 改成 `v0.5.23-xpu`。
- **新手概念课堂**：XPU 是 Intel 的 GPU 架构。Docker tag 里的版本号对应 SGLang 发布版本，升级 tag 就是"用新版镜像"。
- **对你有什么影响**：Intel GPU 用户拉新版镜像即可获得 0.5.23 的修复和特性。

### 15. [Perf] [AMD] deterministic: Allow AITER prefill attention on gfx950 (#45601)
- **代码层面**：`attention_hook.py` 注册新 post-process `_deterministic_aiter_prefill_page_size`；`overrides.py` 里 `_deterministic_attention_backend` 为 prefill 增加 `"aiter"` 选项（仅 gfx950 + 非 MLA + 无投机 + bf16 KV cache），并强制 `chunked_prefill_size` 为 128 的倍数、`page_size=128`。
- **新手概念课堂**：AITER 是 AMD 的推理算子库。batch invariance 要求"同一 prompt 无论怎么切 chunk，结果一致"——只有当 chunk 边界落在 kernel 的 128 行 query tile 上才成立，所以强制 page_size=128。
- **对你有什么影响**：AMD gfx950 用户开启 `--enable-deterministic-inference` 时，prefill 可以用 AITER 加速，同时保持输出可复现。

### 16. [Doc] cookbook: Mirror InferenceX #4811 Kimi-K3.1-Pro GB300 AgentX launches (#48551)
- **代码层面**：`kimi-k3.1-pro.jsx` 更新为从 SemiAnalysisAI/InferenceX 镜像的启动配置，新增 `source.repo/main/prs` 字段，调整 cell（如 Agg TP16 4 节点、DFlash 8）、镜像 sha、disclosures 等。
- **新手概念课堂**：AgentX 是 SemiAnalysis 的推理基准测试。cookbook 镜像它的配置，让用户能复现基准测试的启动参数。
- **对你有什么影响**：想在 GB300 上复现 Kimi-K3.1-Pro 的 AgentX 成绩，直接照 cookbook 抄命令即可。

### 17. [Fix] [AMD] mooncake: Bump Mooncake to pick up the HIP IPC sub-allocation fix (#48485)
- **代码层面**：`docker/rocm.Dockerfile` 里 `MOONCAKE_COMMIT` 从 `78566d45` 升到 `748af1a5`，引入 HIP IPC 子分配修复。
- **新手概念课堂**：Mooncake 是 KV cache 的分布式存储/传输层。HIP IPC 是 AMD 的进程间通信机制，子分配（sub-allocation）出 bug 会导致 KV cache 传输失败或数据错乱。
- **对你有什么影响**：AMD 上用 Mooncake 做 PD 分离或 HiCache 时，KV 传输更可靠。

### 18. [Feat] sgl-router: Add an opt-in fips feature backed by aws-lc-rs (#48066)
- **代码层面**：`pr-test-sgl-router.yml` 新增 `cargo clippy --features fips` 步骤；`sgl-router.Dockerfile` 新增 `CARGO_FEATURES` 和 `RUNTIME_BASE_IMAGE` build arg，fips 时安装 cmake/golang/clang；`Cargo.toml` 新增 `fips` feature 指向 `aws-lc-rs` 的 FIPS 模块。
- **新手概念课堂**：FIPS 140-3 是美国政府的加密模块认证标准。`aws-lc-rs` 是 AWS 的 Rust 密码学库，它的 FIPS 模块通过认证。开启后，TLS 加密走认证实现。
- **对你有什么影响**：企业/政府客户可以构建 FIPS 合规的 `sgl-router` 镜像：`docker build --build-arg CARGO_FEATURES=fips -f docker/sgl-router.Dockerfile -t sgl-router:fips .`。

### 19. [Perf] [Intel] diffusion: Adding AMX optimizations on CPU platform (#30719)
- **代码层面**：`flash_attn.cpp` 的 `flash_attn_varlen_func` 新增 `sm_scale` 可选参数（不再硬编码 `1/sqrt(head_size)`）；`torch_extension_cpu.cpp` 注册签名同步；`adapter/base.py`、`encoders/base.py` 加入 `AMX_ATTN` 后端枚举；新增 AMX attention 实现（diff 中被截断，共 909 行）。
- **新手概念课堂**：AMX（Advanced Matrix Extensions）是 Intel CPU 的矩阵加速指令集（Sapphire Rapids 起）。用它跑 attention 的 GEMM 部分，比 AVX-512 快数倍。
- **对你有什么影响**：在 Intel 服务器 CPU 上跑扩散模型（multimodal_gen），attention 部分会走 AMX 加速，CPU 推理速度显著提升。

### 20. [Feat] [CPU] glm_ocr: Add model support (#37597)
- **代码层面**：`conv3d.cpp` 把内联的 `pack_vnni` 抽到 `vec_pack.h`（供其他 kernel 复用），并重写 conv3d 的 weight packing 注释和布局（`[OC, K]` → `[OC/BLOCK_N, K/2, BLOCK_N, 2]`）；diff 后续（被截断，共 774 行）新增 glm_ocr 模型支持。
- **新手概念课堂**：VNNI（Vector Neural Network Instructions）是 CPU 上把低精度数据打包成向量做矩阵乘的格式。conv3d 的 weight 重排是为了让 GEMM kernel 能高效读取。
- **对你有什么影响**：在 Intel CPU 上可以跑 GLM-OCR 模型（OCR 场景），conv3d 算子也因重构而更易维护/复用。

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)