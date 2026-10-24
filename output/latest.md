# 🚀 vLLM & SGLang 每日更新 — 2026-10-24

> 自动生成于 2026-10-24 06:25 UTC | AI 解读: ✅ 含代码解读

## vllm
## 💡 今日关键词

- **关键词：Encoder Cache 引用计数**
  一句话通俗解释：多模态请求里同一张图片可能被引用多次，缓存管理器要精确记录"每个 hash 被几个 input_id 引用"，最后一个引用释放时才真正回收。
  为什么社区在关注：多模态推理正在成为 vLLM 的主流负载，缓存管理从"整块释放"进化到"细粒度引用计数"，说明推理引擎已经开始像操作系统内存管理一样精细。
  对新手意味着什么：如果你在学推理系统，引用计数、LRU、内存池是绕不开的三大基本功，值得专门花时间啃。

- **关键词：MoE 路由的非法 ID 处理**
  一句话通俗解释：MoE 模型里 padding token 的路由 ID 可能是 -1，必须映射到一个"哨兵专家"让排序保持有序，否则会污染后续的量化 scale 计算。
  为什么社区在关注：MoE 是当前大模型的主流架构（DeepSeek、Qwen3-MoE、Kimi），但 padding 和 batch 不齐是真实场景的常态，这类"边界条件 bug"极难发现。
  对新手意味着什么：写 GPU kernel 时，"非法输入"比"正常输入"更值得你花时间思考。

- **关键词：AMD ROCm / gfx942 / gfx950 适配**
  一句话通俗解释：同一天有 3 个 commit 专门修 AMD GPU 上的问题（int4 重量化、Triton paged decode、RDNA W4A16），说明 AMD 已经是 vLLM 的一等公民。
  为什么社区在关注：异构计算不再是"支持一下"，而是"每个 arch 都要单独验证"。gfx942（MI300）和 gfx950（MI350）的差异需要专门的测试矩阵。
  对新手意味着什么：如果你在学 GPU 编程，AMD 生态（HIP、Triton on ROCm）的岗位需求正在快速增长。

- **关键词：投机解码（Speculative Decoding）与 LoRA 解耦**
  一句话通俗解释：draft model（草稿模型）不跑 LoRA adapter，所以配置校验不应该因为 draft model 不支持 LoRA 就拒绝整个请求。
  为什么社区在关注：投机解码 + LoRA 是生产环境的常见组合，配置校验的"误杀"会让用户莫名其妙启动失败。
  对新手意味着什么：配置校验逻辑要考虑"谁是真正执行者"，而不是"谁出现在配置里"。

- **关键词：CUDA Graph 内存核算**
  一句话通俗解释：自适应验证的 profiling 会留下临时激活值在 allocator cache 里，如果不 `empty_cache()` 就统计 CUDA graph 内存，会误把 profiling 内存算成 graph 内存，导致 KV cache 被少分配。
  为什么社区在关注：CUDA graph 是 vLLM 性能的核心，内存核算差 2GB 就可能让长上下文请求 OOM。
  对新手意味着什么：`torch.cuda.empty_cache()` 不是"性能优化"，而是"正确性保证"——在某些场景下不调用就是 bug。

---

## 🧪 重要知识点 & 动手实验

### 知识点 1：引用计数缓存管理（Encoder Cache）

- **是什么**：缓存里的每个对象（这里是多模态 hash）记录"被多少个请求的 input_id 引用"，计数归零才真正释放。
- **为什么重要**：多模态请求里同一张图片可能在 prompt 里出现多次（比如"比较图 A 和图 A 的差异"），如果第一次释放就回收，后续引用会读到已释放内存。不懂这个会踩 use-after-free 的坑。
- **动手试试**：
```python
# 模拟 vLLM 的 EncoderCacheManager 引用计数逻辑
from collections import OrderedDict

class MiniCache:
    def __init__(self, size):
        self.size = size
        self.freeable = OrderedDict()  # hash -> num_embeds
        self.refs = {}                 # request_id -> {hash: count}

    def add_ref(self, req_id, mm_hash, num_embeds):
        self.refs.setdefault(req_id, {})
        self.refs[req_id][mm_hash] = self.refs[req_id].get(mm_hash, 0) + 1
        if mm_hash not in self.freeable:
            self.freeable[mm_hash] = num_embeds

    def release(self, req_id, mm_hash):
        self.refs[req_id][mm_hash] -= 1
        if self.refs[req_id][mm_hash] == 0:
            del self.refs[req_id][mm_hash]
            # 只有该 hash 在所有请求里都没引用了才 freeable
            if not any(mm_hash in r for r in self.refs.values()):
                self.freeable[mm_hash] = self.freeable.get(mm_hash, 0)

cache = MiniCache(64)
cache.add_ref("r1", "img0", 4)   # 第一次引用
cache.add_ref("r1", "img0", 4)   # 同一请求第二次引用
cache.release("r1", "img0")
assert "img0" in cache.refs["r1"]  # 还有 1 次引用，不能释放
cache.release("r1", "img0")
assert "img0" not in cache.refs["r1"]  # 引用归零，可释放
print("引用计数正确！")
```
- **预期结果**：第一次 release 后 `img0` 仍在 `refs["r1"]` 里，第二次 release 后才消失。这就是 commit 1 里 `request_hash_refs` 的核心思想。

### 知识点 2：MoE 路由的哨兵专家（Sentinel Expert）

- **是什么**：MoE 里每个 token 被路由到 top-k 个专家，但 padding token 的 topk_id 可能是 -1。排序前把 -1 映射到 `n_expert`（一个不存在的专家 ID），保证排序后非法路由都排在最后。
- **为什么重要**：如果 -1 参与排序，`sortAndScanExpert` 的 offset 扫描会看到乱序的 key，导致某些专家的 offset 计算错位，后续 grouped GEMM 会把 padding 行混进有效行的量化 scale 里。
- **动手试试**：
```python
import torch

def preprocess_topk_ids(topk_ids, n_expert):
    """把 -1 映射到 n_expert 哨兵，保证排序有序。"""
    return torch.where(topk_ids < 0, torch.full_like(topk_ids, n_expert), topk_ids)

# 模拟：4 个 token，每个 top-2，其中 token 2、3 是 padding
topk_ids = torch.tensor([[0, 1], [2, -1], [-1, -1], [3, 0]])
print("原始:", topk_ids.flatten().tolist())
mapped = preprocess_topk_ids(topk_ids, n_expert=4)
print("映射后:", mapped.flatten().tolist())

# 排序后应该所有 4（哨兵）都在最后
sorted_ids, _ = torch.sort(mapped.flatten())
print("排序后:", sorted_ids.tolist())
assert sorted_ids[-1].item() == 4  # 哨兵在最后
```
- **预期结果**：映射后所有 -1 变成 4，排序后 4 全部在尾部。commit 7 还加了 `if (size == 0) return;` 防止空 batch 时 kernel 启动崩溃。

### 知识点 3：CUDA Graph 内存核算的正确姿势

- **是什么**：`capture_model` 期间会跑 profiling 的 dummy run，这些 run 的激活值会留在 PyTorch allocator 的缓存里。统计 "graph 占了多少内存" 前必须 `empty_cache()`，否则会多算。
- **为什么重要**：多算的内存会从 KV cache 预算里扣掉，长上下文请求直接 OOM。这是典型的"看起来只是性能问题，实际是正确性问题"。
- **动手试试**：
```python
import torch

def measure_graph_memory(do_empty_cache: bool):
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()

    # 模拟 profiling 的 dummy run：分配后释放
    dummy = torch.randn(1024, 1024, device="cuda")
    del dummy

    if do_empty_cache:
        torch.cuda.empty_cache()

    # 模拟 graph capture 分配
    graph_buf = torch.randn(512, 512, device="cuda")
    peak = torch.cuda.memory_reserved() - base
    del graph_buf
    return peak

no_empty = measure_graph_memory(do_empty_cache=False)
with_empty = measure_graph_memory(do_empty_cache=True)
print(f"不 empty_cache 统计: {no_empty / 1e6:.2f} MB")
print(f"empty_cache 后统计: {with_empty / 1e6:.2f} MB")
print(f"多算了 {(no_empty - with_empty) / 1e6:.2f} MB")
```
- **预期结果**：不调用 `empty_cache` 时统计值明显偏大（dummy 的 4MB 缓存被算进去了）。这就是 commit 13 修的问题。

### 知识点 4：Triton kernel 的 32-bit vs 64-bit 索引

- **是什么**：Triton kernel 里指针偏移如果写成 `i64 * stride`，在某些 AMD GPU（gfx942）上会被编译器错误地 spill 到 SGPR，导致 stride 被替换成无关值，输出某些 head 没被写入。
- **为什么重要**：这类"部分输出未写入"的 bug 极难发现——不 crash、不报错，只是结果错了。commit 17 的测试用 `torch.full(..., float("nan"))` 初始化输出，未写入的行会变成 NaN 从而暴露。
- **动手试试**：
```python
import torch

def test_output_fully_written(kernel_fn, num_tokens, num_heads, head_size):
    """用 NaN 初始化输出，检查 kernel 是否写满了所有位置。"""
    out = torch.full((num_tokens, num_heads, head_size), float("nan"))
    kernel_fn(out)
    if torch.isnan(out).any():
        raise AssertionError(f"有 {torch.isnan(out).sum().item()} 个位置未写入！")
    print("所有位置都已写入 ✓")

# 模拟一个"漏写最后一个 head"的 buggy kernel
def buggy_kernel(out):
    n, h, d = out.shape
    for i in range(n):
        for j in range(h - 1):  # 故意漏掉最后一个 head
            out[i, j] = torch.randn(d)

try:
    test_output_fully_written(buggy_kernel, 4, 8, 64)
except AssertionError as e:
    print(f"捕获到 bug: {e}")
```
- **预期结果**：输出 "捕获到 bug: 有 256 个位置未写入！"。这个测试模式（NaN 初始化 + 检查）是 GPU kernel 测试的黄金实践。

### 知识点 5：投机解码与 LoRA 的配置解耦

- **是什么**：`verify_with_model_config` 会校验 LoRA 是否被模型支持。但投机解码会用 draft model 的 config 再校验一次，draft model 不跑 LoRA，所以 `runner_type == "draft"` 时应该跳过校验。
- **为什么重要**：配置校验的"误杀"是生产环境最常见的启动失败原因之一。用户明明配了合法的 target model + LoRA，却因为 draft model 不支持 LoRA 而无法启动。
- **动手试试**：
```python
from types import SimpleNamespace

def verify_lora(model_config):
    # 关键：draft 不跑 LoRA，跳过校验
    if model_config.runner_type != "draft" and not model_config.is_lora_supported:
        raise ValueError(f"LoRA 不支持 {model_config.architecture}")

# 场景 1：target model 支持 LoRA → 通过
verify_lora(SimpleNamespace(runner_type="generate", is_lora_supported=True,
                            architecture="LlamaForCausalLM"))
print("target model 校验通过 ✓")

# 场景 2：draft model 不支持 LoRA → 应该通过（不校验）
verify_lora(SimpleNamespace(runner_type="draft", is_lora_supported=False,
                            architecture="DeepSeekMTPModel"))
print("draft model 校验跳过 ✓")

# 场景 3：target model 不支持 LoRA → 应该报错
try:
    verify_lora(SimpleNamespace(runner_type="generate", is_lora_supported=False,
                                architecture="Zamba2ForCausalLM"))
except ValueError as e:
    print(f"正确拒绝: {e}")
```
- **预期结果**：场景 1 和 2 通过，场景 3 报错。这就是 commit 9 加的那一行 `model_config.runner_type != "draft"` 的作用。

---

## 🔥 重要更新

- **Commit 1（Encoder Cache 引用计数）**：多模态缓存从"整块释放"升级为"按 hash 引用计数"，直接影响多模态请求的内存利用率和正确性，是今天最重要的性能 + 正确性双修 commit。
- **Commit 7（MoE 非法 topk_id 重映射）**：修复了无 expert_map 时 padding token 污染量化 scale 的 bug，影响所有 MoE 模型在 batch 不齐时的数值正确性，属于"静默错误"级别的修复。
- **Commit 13（CUDA Graph 内存核算）**：`empty_cache()` 一行修复，避免了 KV cache 被少分配的 OOM 风险，影响所有启用自适应验证的部署。
- **Commit 20（vLLM IR 弃用）**：vLLM IR 将在 v0.34.0 移除，`--ir-op-priority` 开始打 warning。如果你有自定义 op 走 IR 路径，现在就该规划迁移了。

---

## 📋 逐条解读

### 1. [Perf][Multimodal] Make encoder cache free linear in items per request (#62850)
- **代码层面**：`vllm/v1/core/encoder_cache_manager.py` 新增 `request_hash_refs` 字典，记录"每个请求的每个 mm_hash 被多少个 input_id 引用"。`_add_cached_input_id` 统一处理引用计数，`free_encoder_input` 时只有计数归零才把 hash 加入 `freeable`。
- **新手概念课堂**：就像图书馆借书——同一本书可以被同一个人借多次（引用计数），只有所有借阅记录都还了，书才能上架（freeable）。以前是"还一次就上架"，会导致别人还在看时书被收走。
- **对你有什么影响**：多模态请求里重复图片不再导致缓存过早回收，长 prompt 多图场景下内存利用率更高、更不容易 OOM。

### 2. [Security] Refresh vulnerable runtime and test dependencies (#62835)
- **代码层面**：`requirements/common.txt` 升级 aiohttp、pillow、mcp、multidict 等；`kv_connectors.txt` 把 lmcache 从 0.3.9 升到 0.5.6；CUDA 12 的 lmcache 改用单独的 wheel 源。
- **新手概念课堂**：依赖升级就像给房子换锁——旧锁有已知漏洞（CVE），新锁修了漏洞但可能不兼容旧钥匙（API 变更）。`--find-links` 是指定额外的 wheel 下载源。
- **对你有什么影响**：升级 vLLM 时如果遇到 lmcache 相关报错，检查你的 CUDA 版本是否匹配新的 wheel 源。

### 3. [Bugfix][Model] Fix Gemma4 encoder CG with `max_soft_tokens` set (#60687)
- **代码层面**：`gemma4_mm.py` 抽出 `_get_max_soft_tokens()` 方法，统一从 `multimodal_config.mm_processor_kwargs` 读取用户设置的 `max_soft_tokens`，并在 CUDA graph 配置里使用它而不是硬编码的 `default_output_length`。
- **新手概念课堂**：CUDA graph 需要预先知道所有可能的输入形状（capture axes）。如果用户把 `max_soft_tokens` 设成 560，但 graph 只 capture 到 280，就会 fallback 到 eager 模式，性能暴跌。
- **对你有什么影响**：用 Gemma4 多模态 + 自定义 `max_soft_tokens` 时，CUDA graph 能正确覆盖你的配置，不再静默降级。

### 4. [Doc][Recipe] MiniMax-M3.1-Flash: DEP2 + DEP4 PD on GB200/GB300 (#63428)
- **代码层面**：`recipes/models/MiniMaxAI/MiniMax-M3.1-Flash.yaml` 新增 GB200/GB300 的 DEP2（prefill）+ DEP4（decode）分离部署配置，NVIDIA 侧投机 token 从 3 提到 7 并改用 FLASHINFER_TS。
- **新手概念课堂**：PD 分离 = Prefill（算 prompt）和 Decode（逐 token 生成）跑在不同节点上，因为两者的计算特征完全不同（prefill 是 compute-bound，decode 是 memory-bound）。DEP = Data Expert Parallel，是 MoE 专用的并行方式。
- **对你有什么影响**：如果你在 GB200/GB300 上部署 MiniMax-M3.1-Flash，可以直接抄这份 recipe，不用自己调参。

### 5. [Kimi K3] KDA RecoverSSM: CuTe verify and commit kernels (#63045)
- **代码层面**：`recoverssm` 的 verify records 从 KV cache 里移出来，改成独立的 `KDARecoverSSMRecords` 对象；新增 CuTe 实现的 verify 和 commit kernel；`kda_state_shape` 统一计算 state 形状。
- **新手概念课堂**：RecoverSSM 是 Kimi K3 的 Mamba 状态恢复机制，投机解码时需要"验证"草稿 token 并"提交"接受的状态。把 records 移出 KV cache 是为了避免 KV cache 布局被投机解码的特殊需求污染。
- **对你有什么影响**：Kimi K3 投机解码的显存布局更干净，KV cache 可以按标准形状分配，减少碎片。

### 6. [Bugfix][LoRA][MoE] Process reloaded weights of LoRA-wrapped MoE layers (#63319)
- **代码层面**：`test_reload.py` 新增测试，模拟 LoRA 包装后的 expert 权重名（插入 `base_layer`），验证 `reload_weights` 能正确处理。修复了 reload 时 LoRA-wrapped MoE 层权重名不匹配的问题。
- **新手概念课堂**：LoRA 包装后，权重名会从 `model.layers.0.mlp.experts.0.w1` 变成 `model.layers.0.mlp.experts.0.base_layer.w1`。reload 时如果按原名查找就会找不到。
- **对你有什么影响**：用 LoRA + 量化 MoE 模型时，热更新权重（reload）不再静默失败。

### 7. [Bugfix][MoE] Remap invalid topk_ids in moe_permute without expert_map (#59717)
- **代码层面**：`moe_permute_unpermute_op.cu` 里 `preprocessTopkIdLauncher` 从"仅在有 expert_map 时调用"改为"总是调用"；kernel 里加 `if (size == 0) return;` 防止空 batch。
- **新手概念课堂**：见知识点 2。核心是 -1 必须映射到哨兵专家，否则排序后 offset 扫描会错位，污染 per-tensor 量化 scale。
- **对你有什么影响**：MoE 模型在 batch 里有 padding token 时，输出数值更准确（之前可能静默偏差）。

### 8. [Bugfix][ROCm][Kimi-K3] Validate gfx942's int4 expert requant as int4 (#63131)
- **代码层面**：`rocm_aiter_moe.py` 的 `_supports_quant_scheme` 按 arch 区分：gfx942 支持 int4，gfx950 支持 mxfp4，gfx1250 都不支持。新增测试矩阵验证每个 arch 的部署组合。
- **新手概念课堂**：gfx942（MI300）没有 MXFP4 硬件支持，所以 Kimi-K3 的 MXFP4 experts 在加载时会被 requant 成 int4。如果代码不区分 arch，会错误地声称支持 MXFP4 然后运行时报错。
- **对你有什么影响**：在 MI300 上跑 Kimi-K3 不再因为 quant scheme 校验失败而启动不了。

### 9. [Bugfix][LoRA] Don't reject LoRA for speculative decoding drafters (#63186)
- **代码层面**：`lora.py` 的 `verify_with_model_config` 加条件 `model_config.runner_type != "draft"`；`test_config.py` 新增测试验证 draft config 不触发拒绝。
- **新手概念课堂**：见知识点 5。配置校验要区分"谁真正执行 LoRA"，draft model 只是草稿生成器，不跑 adapter。
- **对你有什么影响**：投机解码 + LoRA 的组合不再因为 draft model 配置被误杀而启动失败。

### 10. [ROCm][RDNA][Perf] W4A16: packed fp16 prefill dequant and tiles for gfx1151 (#60644)
- **代码层面**：`benchmark_rdna_hybrid_w4a16_gemm.py` 增加 `--dtype` 参数（float16/bfloat16），`prepare_hybrid_weights` 支持 dtype 参数；kernel 侧新增 packed fp16 prefill dequant 和 gfx1151 专用 tile 配置。
- **新手概念课堂**：W4A16 = 权重 4-bit 量化、激活 16-bit。gfx1151 是 RDNA 架构的 APU，prefill 阶段需要把 4-bit 权重解包成 fp16 再算 GEMM，"packed" 指一次解多个值提高吞吐。
- **对你有什么影响**：在 RDNA APU 上跑 W4A16 模型，prefill 速度提升，且支持 bf16 精度。

### 11. [Bugfix][Kernel] Fix cooperative_topk illegal memory access on NaN scores (#63118)
- **代码层面**：`persistent_topk.cuh` 删除本地 `convert_to_uint32_v2`，统一用 `topk_histogram_4096::convert_to_uint32_v2`（正确处理 NaN）；`radix_topk` 里加 NaN 处理逻辑。
- **新手概念课堂**：NaN 的 bit pattern 在 float 转 uint32 排序键时是特殊的——普通转换会把 NaN 排到错误位置，导致后续索引越界。统一用处理过 NaN 的转换函数是关键。
- **对你有什么影响**：logits 出现 NaN（数值不稳定时常见）时，top-k kernel 不再 crash 或非法访存。

### 12. [Bugfix][AuxOutput] Retain routed experts while a KV cache tier holds their blocks (#62591)
- **代码层面**：`aux_output_connector/store.py` 的 `put` 支持 `retain_keys` 参数，被 KV cache tier 持有的 block 不会被 LRU 淘汰；`store` 在全部对象都被引用时改为"扩容 + warning"而不是直接报错。
- **新手概念课堂**：aux output（比如 MoE 路由的专家 ID）需要和 KV cache block 同生命周期。如果 KV block 还在（被 prefix caching 持有），对应的专家记录就不能被 LRU 踢掉。
- **对你有什么影响**：启用 prefix caching + aux output 时，不再因为 store 满了而丢专家记录导致输出错误。

### 13. [Bugfix][MRV2] Don't count adaptive verification profiling cache as CUDA graph memory (#61509)
- **代码层面**：`model_runner.py` 的 `capture_model` 里，adaptive verification profiling 跑完后加 `torch.accelerator.empty_cache()`，再统计 graph 内存。
- **新手概念课堂**：见知识点 3。profiling 的 dummy run 留下的激活值在 allocator cache 里，不清理就会被算进 graph 内存，导致 KV cache 预算被少分配。
- **对你有什么影响**：启用自适应验证时，KV cache 分配更准确，长上下文请求不容易 OOM。

### 14. [Feature][PCP] Support decoder-side SWA bounded replay with APC for DeepSeek-V4.1-Flash (#62384)
- **代码层面**：`deepseek_v41/attention.py` 的 `forward_kv` 支持 PCP（Parallel Context Parallelism）下 all-gather 所有 rank 的 KV；`decoder_replay_layers.py` 新增 `DecoderReplayLayers` 和 `ReplayBatch`，支持 decoder 侧 SWA（Sliding Window Attention）有界重放 + APC（Automatic Prefix Caching）。
- **新手概念课堂**：SWA 只看最近 N 个 token，APC 缓存前缀。两者结合时，replay 需要知道"哪些 KV 在缓存里、哪些需要重算"。PCP 把上下文切到多个 rank 上，replay 时要 gather 回来。
- **对你有什么影响**：DeepSeek-V4.1-Flash 在长上下文 + prefix caching 场景下，decode 阶段能正确复用缓存，不用全量重算。

### 15. [Refactor] Remove dead code models (#63400)
- **代码层面**：删除 `llava_onevision.py` 的 `LlavaOnevisionMultiInputs` 类型别名、`phi4mm_utils.py` 的 `BlockBase` 和 `AttModule` 类、`radio.py` 的部分死代码。
- **新手概念课堂**：死代码就像衣柜里不穿的衣服——占地方、增加阅读负担、可能误导新人以为它有用。定期清理是维护大型项目的必要操作。
- **对你有什么影响**：读源码时少了一些干扰项，IDE 的 "find usages" 更干净。

### 16. [Refactor] [AMD] flydsl: Move UltraQuant/TurboQuant decode to public APIs (#61649)
- **代码层面**：`flydsl_turboquant_decode.py` 移除 `is_flydsl_gqa6_available` 等内部 API，改用公开 API；测试里 `_build_cache` 支持 `lead_blocks` 参数（把引用块推到高字节偏移，测试大偏移场景）。
- **新手概念课堂**：内部 API 转公开 API 意味着"这个接口我们承诺维护了"。`lead_blocks` 是测试技巧——不增加数据量但改变内存布局，专门测指针偏移的边界。
- **对你有什么影响**：如果你在 AMD 上自定义 TurboQuant 解码，可以依赖稳定的公开 API 而不是随时会变的内部函数。

### 17. [ROCm][Bugfix] Fix Triton paged decode leaving output heads unwritten on gfx942 (#63268)
- **代码层面**：`chunked_prefill_paged_decode.py` 把 `output_offset` 的 i64 计算拆成两步：`output_ptr += start_index * stride_0` 和 `output_offset = head_idx * stride_1`，避免 gfx942 上 i64 形式被编译器错误 spill。测试用 NaN 初始化输出。
- **新手概念课堂**：见知识点 4。gfx942 的寄存器分配器在"高 spill"场景下会把 i64 的 stride 替换成无关的 SGPR 值，导致某些 head 的写入地址错误。
- **对你有什么影响**：在 MI300 上用 Triton paged decode 时，输出不再有部分 head 是垃圾值。

### 18. [Doc] MiniMax-M3: update the B300 NVFP4 InferenceX AgentX profile (#63407)
- **代码层面**：`MiniMax-M3.yaml` 更新 B300 的 AgentX 配置，新增 TP2 点位（24/28 clients + host KV offload），TP4 点位从 36/40 clients 改为 20/24 clients，host KV offload 从 1499GB 降到 749GB。
- **新手概念课堂**：TP2 意味着每个 GPU 只切 2 份模型，单 GPU 显存压力大，所以需要 host KV offload 把 KV cache 放到 CPU 内存。offload 容量减半说明 TP2 下每 GPU 的 KV 需求更小（因为只服务更少请求）。
- **对你有什么影响**：在 B300 上部署 MiniMax-M3 有更多并行度选择，TP2 + offload 适合显存紧张但想要高并发的场景。

### 19. [Core] Let VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS set the EngineCore cleanup grace (#63260)
- **代码层面**：`utils.py` 的 `get_engine_process_shutdown_timeout` 优先读 `VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS` 环境变量，未设置时才用 ROCm 的默认值。
- **新手概念课堂**：EngineCore 收到 SIGTERM 后需要时间释放 GPU 资源（尤其是 ROCm，teardown 慢）。父进程管理器需要知道"等多久才强杀"。环境变量让用户覆盖默认值，适配不同的 teardown 时长。
- **对你有什么影响**：如果你的部署环境 teardown 特别慢（比如大模型 + 多卡），可以设这个环境变量避免被过早强杀导致 VRAM 泄漏。

### 20. [vLLM IR] Deprecate ir_op_priority ahead of vLLM IR removal (#63371)
- **代码层面**：`kernel.py` 的 `ir_op_priority` 字段加 Deprecated docstring；`arg_utils.py` 加 `_IR_OP_PRIORITY_DEPRECATION_MESSAGE`，用户显式设置时打 warning；文档加弃用警告。
- **新手概念课堂**：vLLM IR 是一个"函数式中间表示"，用于 kernel 分发和 lowering。现在要移除，意味着 kernel 选择回归"平台自动决定"的简单模式。deprecation warning 是给用户的迁移窗口。
- **对你有什么影响**：如果你用 `--ir-op-priority` 调优，会看到 warning，需要在 v0.34.0 前迁移到平台默认的 kernel 选择。

## sglang
## 💡 今日关键词（放在最前面！）

- **关键词：PD 分离（Prefill/Decode Disaggregation）的 KV 传输可观测性**
  - 一句话通俗解释：把"读题"（Prefill）和"写答案"（Decode）拆到不同机器上跑，中间靠网络搬运 KV 缓存；今天在给这段"搬运过程"加监控仪表盘。
  - 为什么社区在关注：Commit 1、2 都在改 disagg 相关代码，说明大模型推理的"分离式架构"已经进入精细化运维阶段——不再是"能跑就行"，而是要能看见 KV 传输占了多少显存。
  - 对新手意味着什么：如果你学推理系统，PD 分离是当前最热的生产级部署范式，值得专门花时间啃。

- **关键词：DeepSeek V4.1 的图捕获（CUDA Graph）极致优化**
  - 一句话通俗解释：把一整段计算"录制"下来反复回放，省掉每次重新调度的开销；今天连续 5 个 commit 都在抠这块的性能。
  - 为什么社区在关注：Commit 6/7/8/15/16/17 全部围绕 dsv41 的 prefill graph，说明新模型上线后性能调优是主战场，而且已经细到"每个 layer 少算一次坐标变换"。
  - 对新手意味着什么：CUDA Graph + 自定义 attention backend 是高性能推理的硬核技能，门槛高但稀缺。

- **关键词：异构平台插件化（Out-of-tree Platform Plugin）**
  - 一句话通俗解释：让 AMD、NPU、国产芯片等"非 NVIDIA 阵营"能自己写插件接入 SGLang，而不用改主仓库代码。
  - 为什么社区在关注：Commit 9、10、16 都在做平台扩展接口，说明 SGLang 正在从"CUDA 优先"走向"多硬件生态"。
  - 对新手意味着什么：如果你在国产芯片/AMD 生态工作，这是参与开源的最佳切入点。

- **关键词：Diffusion（扩散模型）服务的工程健壮性**
  - 一句话通俗解释：文生图/视频服务除了"生成得快"，还要处理取消、上传、断点、空闲省电这些"脏活"。
  - 为什么社区在关注：Commit 11/12/13/19/20 全是 diffusion 的 fix/perf，说明多模态生成服务正在从 demo 走向生产。
  - 对新手意味着什么：异步任务、资源清理、幂等性是后端工程师的通用能力，这里是最好的实战教材。

- **关键词：投机解码（Speculative Decoding）的可扩展性**
  - 一句话通俗解释：用小模型"猜"几个 token，大模型一次性验证，加速生成；今天在让第三方硬件也能自定义这套流程。
  - 为什么社区在关注：Commit 9、10 给 EAGLE 的 draft worker 和 graph runner 开了插件口子。
  - 对新手意味着什么：投机解码是当前 LLM 推理加速最主流的技术之一，理解它的工程接口比理解算法更重要。

---

## 🧪 重要知识点 & 动手实验

- **知识点 1：异步任务取消时的资源清理（asyncio.shield + gather return_exceptions）**
  - **是什么**：当一个 HTTP 请求被客户端取消时，后台还在跑的上传任务不能"半路被杀"，否则它正在读的本地文件可能被清理逻辑删掉，导致上传读到一半崩掉。
  - **为什么重要**：真实生产里，用户关页面、超时、服务优雅关闭都会触发取消。不懂 `asyncio.shield` 就会踩"文件被删了但线程还在读"的坑，表现为随机 500 错误或日志里莫名其妙的 `FileNotFoundError`。
  - **动手试试**：
    ```python
    import asyncio, os, tempfile

    async def slow_upload(path):
        await asyncio.sleep(1)          # 模拟 SDK 读文件
        return f"uploaded {os.path.basename(path)}"

    async def handler():
        fd, path = tempfile.mkstemp()
        os.write(fd, b"data"); os.close(fd)
        task = asyncio.create_task(slow_upload(path))
        try:
            await asyncio.shield(task)   # 关键：取消不打断底层任务
        except asyncio.CancelledError:
            print("收到取消，但等上传完成再清理")
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    pass
            os.remove(path)              # 现在删才安全
            raise

    async def main():
        t = asyncio.create_task(handler())
        await asyncio.sleep(0.1)
        t.cancel()
        try: await t
        except asyncio.CancelledError: print("handler 已取消，文件安全删除")

    asyncio.run(main())
    ```
  - **预期结果**：输出"收到取消，但等上传完成再清理"→"handler 已取消，文件安全删除"，说明上传没被打断、文件在正确时机删除。

- **知识点 2：CUDA Graph 中"元数据原地刷新"（maybe_copy_inplace）**
  - **是什么**：CUDA Graph 把一段 kernel 序列录制成固定图，回放时所有 tensor 地址必须不变。所以新请求的元数据（位置、偏移）不能新建 tensor，而要 `copy_` 进预分配的 buffer。
  - **为什么重要**：这是 Commit 7 的核心——把 SWA 位置翻译从"每层都算"改成"每步算一次写进 buffer"。不懂这个模式，写出来的 graph 回放会读到旧数据，表现为"输出乱码但 eager 模式正常"这种极难调的 bug。
  - **动手试试**：
    ```python
    import torch
    # 模拟：静态 buffer 在 capture 时分配，replay 时原地更新
    static_offsets = torch.zeros(2, 8, dtype=torch.int32, device="cuda")
    static_offsets[0, :3] = torch.tensor([0, 5, 9])
    static_offsets[1, :3] = torch.tensor([5, 4, 7])

    # 新一批请求到来，不能 new tensor，必须 copy_ 进去
    new_offsets = torch.tensor([[0, 3, 6], [3, 3, 3]], dtype=torch.int32, device="cuda")
    static_offsets[:, :3].copy_(new_offsets)   # 原地刷新
    print(static_offsets)
    ```
  - **预期结果**：打印出 `[[0,3,6,0,...],[3,3,3,0,...]]`，地址不变、内容更新——这就是 graph replay 能读到新数据的原因。

- **知识点 3：Safetensors 文件头校验与"只读不删"原则**
  - **是什么**：safetensors 文件开头有 JSON header 描述每个 tensor 的偏移。读取前先校验 header，失败就说明文件坏了——但 Commit 13 把"删除坏文件"改成了"只记录不删"。
  - **为什么重要**：多副本共享同一份 checkpoint 缓存时，一个副本删了文件，其他副本就崩了。而且读失败可能是权限/网络 I/O 问题，不是文件真坏。这是分布式存储的经典陷阱。
  - **动手试试**：
    ```python
    import json, struct, tempfile, os
    from safetensors import safe_open

    # 造一个坏文件：header 长度写对，但 JSON 是乱码
    bad = tempfile.mktemp(suffix=".safetensors")
    with open(bad, "wb") as f:
        f.write(struct.pack("<Q", 10))   # header 长度
        f.write(b"not a json")           # 非法 header

    try:
        with safe_open(bad, framework="pt") as f:
            pass
    except Exception as e:
        print(f"读取失败: {type(e).__name__}")   # 注意：文件仍在！
    print("文件还在吗:", os.path.exists(bad))
    ```
  - **预期结果**：打印"读取失败: SafetensorError"和"文件还在吗: True"——校验失败不等于要删文件。

- **知识点 4：MegaGate/DeepGEMM 的 token 上限门控**
  - **是什么**：DeepGEMM 的融合 gate kernel 一次 launch 只能处理固定大小的 token block（2048 行）。超过就得回退到普通 router，否则会越界。
  - **为什么重要**：Commit 15 和 17 是一次"revert 再 revert"的来回——先加了 2048 硬上限，发现 prefill graph capture 时 `is_extend_in_batch` 被清空导致误判，又改回用 `mode.is_extend()` + 环境变量上限。这类"模式标志在 graph capture 下不可靠"是性能优化的经典陷阱。
  - **动手试试**：
    ```python
    # 模拟模式判断的陷阱
    class Mode:
        def __init__(self, is_extend, is_extend_in_batch):
            self._e, self._eb = is_extend, is_extend_in_batch
        def is_extend(self): return self._e
        def is_extend_in_batch(self): return self._eb

    # graph capture 时 is_extend_in_batch 被清空
    capture_mode = Mode(is_extend=True, is_extend_in_batch=False)
    print("用 is_extend():", capture_mode.is_extend())            # True ✓
    print("用 is_extend_in_batch():", capture_mode.is_extend_in_batch())  # False ✗ 误判
    ```
  - **预期结果**：前者 True 后者 False，直观看到为什么必须用 `mode.is_extend()`。

- **知识点 5：空闲时等待 socket 事件（epoll 替代 busy-poll）**
  - **是什么**：调度器空闲时如果死循环 `sleep(0.001)` 轮询，会白白烧掉一个 CPU 核。改成 `poller.poll(timeout=1000)` 后，没消息就阻塞在 socket 上，来消息立刻唤醒。
  - **为什么重要**：Commit 19 的动机很实际——低 QPS 部署时 CPU 占用高、电费贵。但代价是 idle→active 延迟增加，所以默认关闭。这是"吞吐 vs 延迟"权衡的教科书案例。
  - **动手试试**：
    ```python
    import socket, select, time

    srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen()
    port = srv.getsockname()[1]

    # busy-poll 1 秒
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 1:
        try:
            srv.setblocking(False); srv.accept()
        except BlockingIOError:
            pass
    print(f"busy-poll 1s 内空转次数很多，CPU 100%")

    # epoll 等待 1 秒
    ep = select.epoll(); ep.register(srv.fileno(), select.EPOLLIN)
    t0 = time.perf_counter()
    events = ep.poll(timeout=1.0)   # 阻塞，CPU 接近 0
    print(f"epoll 等待 {(time.perf_counter()-t0):.2f}s，事件数 {len(events)}")
    ```
  - **预期结果**：busy-poll 版本 CPU 打满、epoll 版本安静等 1 秒返回空事件——这就是 `--sleep-on-idle` 的原理。

---

## 🔥 重要更新

- **Commit 2（rust-server KV bootstrap 修复）**：Rust server 模式下 prefill 端会启动自己的 KV bootstrap 服务，否则 PD 请求全部无法路由。这是生产可用性的关键修复。
- **Commit 13（checkpoint 读失败不再删除文件）**：之前读失败会删 symlink 和 blob，多副本共享缓存时是灾难性的；现在只报错不删，是数据安全的重要修复。
- **Commit 11（diffusion 上传取消时先 join 再清理）**：修复"取消请求导致上传读到已删文件"的竞态，影响所有开启云存储的 diffusion 部署。
- **Commit 15/17（MegaGate prefill 路由 revert 来回）**：一次快速回滚+重新落地，反映 dsv41 性能优化在 prefill graph 场景下的脆弱性，值得关注后续稳定性。

---

## 📋 逐条解读

### 1. [Feat] metrics: Export prealloc_{full,swa}_token_usage for decode requests with a pending KV transfer (#45970)
- **代码层面**：在 `decode.py` 新增 `pre_allocated_token_counts()`，遍历 transfer 队列统计每个请求已预留的 full/SWA KV slot；在 `metrics_reporter.py` 里除以池容量，导出为两个新指标；`pool_stats_observer.py` 补上 `full_capacity` 字段。
- **新手概念课堂**：想象餐厅提前给"等位的客人"留好桌子（prealloc KV），但桌子还没坐人。这个 commit 就是给"预留了多少桌子"装了个计数器，方便运维看资源水位。
- **对你有什么影响**：运维 dashboard 上会多两个指标，能提前发现"KV 被预留光了但还没真正用起来"的瓶颈。

### 2. [Fix] rust-server: Start a transfer backend's own KV bootstrap server on prefill (#46074)
- **代码层面**：`engine.py` 在 Rust server 模式下调用新函数 `start_disagg_service_for_rust_server()`；`disagg_service.py` 判断当前后端是否需要独立 bootstrap server（Rust 内嵌服务覆盖不了的才启动）；`scheduler.py` 相应简化。
- **新手概念课堂**：PD 分离时 prefill 机器要对外广播"我这儿有 KV 可以来拿"，这个广播服务叫 bootstrap server。Rust 版 server 自带一个，但不是所有传输后端都能被它代理，所以要按后端类型决定是否额外起一个。
- **对你有什么影响**：用 Rust server + 某些传输后端做 PD 分离的用户，之前请求会"发出去没人接"，现在能正常路由了。

### 3. [Feat] rust-server: Let a multimodal processor check and fetch media during admission (#46206)
- **代码层面**：Rust 侧 `CoreHandle` 新增 `mm_processor: OnceLock<Arc<dyn MmProcessor>>` 和 `set_mm_processor()`；`prepare_generations` 把 processor 传给 `prefetch_all`，并把 `MediaError::Client/Server` 分别映射为 400/500。
- **新手概念课堂**：多模态请求（带图片/视频）在"准入"阶段就要检查 URL 能不能访问、下载会不会失败。以前 Rust server 只能做简单 prefetch，现在能调用真正的 MM processor，并且错误能正确分类。
- **对你有什么影响**：带图请求如果图片 URL 挂了，会立刻返回 400 而不是排队很久才失败；服务端错误返回 500，客户端能正确重试。

### 4. [Fix] scheduler: Keep long prefill steps in the busy counter and exclude engine pauses (#46127)
- **代码层面**：新增 `_pause_intervals` 双端队列记录暂停窗口；`_record_step_counters` 计算 prefill 耗时时扣掉与暂停窗口的重叠部分；`pause_generation/continue_generation` 维护暂停起止时间；长 prefill（>2s）不再被 `STEP_MAX_US` 截断丢弃。
- **新手概念课堂**：以前统计"GPU 忙了多久"有两个 bug：一是超过 2 秒的 prefill 直接被扔掉不计（低估负载），二是引擎暂停时间被算成忙（高估负载）。现在两边都修了。
- **对你有什么影响**：负载均衡器的调度决策会更准，长 prefill 场景下不再出现"明明很忙却显示空闲"导致雪崩。

### 5. [Fix] bench: Add missing HiCache RPCs to gRPC mock (#46078)
- **代码层面**：`rust/sglang-bench/tests/grpc.rs` 的 Mock 实现补上 `get/attach/detach/clear_hi_cache_storage` 四个 RPC，都返回 `unimplemented`。
- **新手概念课堂**：gRPC 的 trait 新增方法后，所有实现（包括测试用的 Mock）都必须补上，否则编译不过。这是 Rust 强类型带来的"强制同步"。
- **对你有什么影响**：普通用户无感，但跑 bench 测试的开发者不再编译失败。

### 6. [Perf] dsv41: Use the fused ratio-2 compressor in the prefill graph (#46230)
- **代码层面**：`deepseek_v4_backend.py` 中 compressor 有 `use_fused_gate` 时改调 `project_fused()` 一次输出 kv+score；新增 `low_ratio_extend_offsets` 元数据 tensor 供 graph 内使用；`_source_projection_buffers` 去掉 ratio-2 的独立 kv buffer。
- **新手概念课堂**：原本"算 kv"和"算 score"是两次 kernel launch，融合成一个 kernel 少一次 launch 开销。在 prefill graph 里尤其值钱，因为 graph 内 launch 开销被放大。
- **对你有什么影响**：用 DeepSeek V4.1 + prefill graph 的用户，prefill 吞吐会有小幅提升。

### 7. [Perf] dsv41: Translate SWA locations once per prefill graph step (#46229)
- **代码层面**：`init_forward_metadata_for_breakable_cuda_graph_capture` 里把 `translate_loc_from_full_to_swa` 的结果一次性写进 `swa_out_cache_loc` buffer；replay 时 `copy_` 刷新；`deepseek_v4_replay_tail.py` 去掉 bridge buffer 的尾部清零（因为下游会裁掉）。
- **新手概念课堂**：SWA（滑动窗口注意力）需要把"全局位置"翻译成"窗口内位置"，以前每层都翻译一次，现在每步翻译一次写 buffer，所有层共享。
- **对你有什么影响**：长上下文 prefill 场景下每层省一次小 kernel，累积起来是可观的加速。

### 8. [Perf] dsv41: Feed wo_b the fused MXFP8 attention output in the prefill graph (#46231)
- **代码层面**：新增 `deepseek_v4_mega_attention_with_mxfp8_output`，attention 直接输出 MXFP8 格式的 y 和 scale，喂给下游 `wo_b`（DeepGEMM MXFP8 输入）；处理 TMA 对齐 stride 不一致时的 restride 逻辑。
- **新手概念课堂**：MXFP8 是一种 8 位浮点量化格式，attention 输出直接量化好给下一个 GEMM 用，省掉"反量化再量化"的往返。stride 对齐是 GPU 内存访问的基本功。
- **对你有什么影响**：dsv41 prefill 的 attention→FFN 链路更快，端到端延迟降低。

### 9. [Feat] spec: Let out-of-tree platforms provide EAGLE draft graph runners (#45145)
- **代码层面**：`platforms/interface.py` 新增 `get_speculative_graph_runner_cls(algorithm, phase)`；`eagle_worker_v2.py` 新增 `_select_spec_graph_runner_cls` 优先用平台提供的类；文档同步更新。
- **新手概念课堂**：EAGLE 投机解码的 draft 模型也要跑 CUDA Graph 才快，但不同硬件的 graph runner 实现不同。这个接口让 AMD/NPU 厂商自己提供实现，不用改主仓库。
- **对你有什么影响**：非 NVIDIA 硬件上跑 EAGLE 投机解码会更快、更稳定。

### 10. [Feat] spec: Let out-of-tree platforms provide their own speculative worker (#45144)
- **代码层面**：新增 `get_speculative_worker_cls(algorithm)` 和 `supports_speculative_algorithm(algorithm)`；`speculative_hook.py` 加 `_check_oot_speculative_algorithm` 校验；`spec_tree.py` 对不支持的平台调用 `reject_out_of_tree_path`。
- **新手概念课堂**：比上一个 commit 更彻底——不只 graph runner，连整个 speculative worker 都可以替换。同时加了"能力声明"机制，平台必须明确说"我支持 EAGLE"才能用。
- **对你有什么影响**：插件化平台如果没声明支持某算法，启动时会明确报错而不是运行时神秘崩溃。

### 11. [Fix] diffusion/storage: Join uploads before request cleanup (#46068)
- **代码层面**：`storage.py` 用 `asyncio.shield` 保护上传任务，取消时循环等待其完成再抛出；`image_api.py`/`video_api.py` 的 `gather` 加 `return_exceptions=True` 并逐个检查异常。
- **新手概念课堂**：见知识点 1。核心是"取消信号不能打断正在读文件的 I/O"。
- **对你有什么影响**：开启云存储的 diffusion 服务，用户取消请求不会再偶发上传失败或文件损坏。

### 12. [Fix] diffusion/mesh: Guard asynchronous job publication and defaults (#46081)
- **代码层面**：`mesh_api.py` 用 `model_dump(include=..., exclude_none=True)` 替代手写 if 判断；`os.path.getsize` 不再容忍文件不存在（明确失败）；文档补充 mesh 端点说明。
- **新手概念课堂**：手写 `if req.seed is not None: kwargs["seed"] = req.seed` 容易漏字段，用 Pydantic 的 `model_dump` 声明式过滤更可靠。
- **对你有什么影响**：mesh 生成 API 的行为更可预测，参数传递不再有"某个字段没生效"的坑。

### 13. [Fix] diffusion/loader: Preserve checkpoints on read failures (#46095)
- **代码层面**：`weight_utils.py` 把 `corrupted_files` 改名为 `unreadable_files`，删除所有 `os.remove` 逻辑；`rank_local_checkpoint.py` 同步改名；文档加"Checkpoint read failures"章节。
- **新手概念课堂**：见知识点 3。读失败≠文件坏，可能是权限/网络问题；多副本共享缓存时删文件是灾难。
- **对你有什么影响**：多副本部署时一个副本的读失败不会再"连累"其他副本。

### 14. [Perf] diffusion: Project LTX-2 prompt K/V once per denoising loop (#46105)
- **代码层面**：新增 `LTX2PromptKV` dataclass 和 `project_context_kv()` 方法；`forward` 接受可选的 `context_kv` 参数，传了就跳过 `to_k/to_v/k_norm`；denoising 循环里预计算一次复用。
- **新手概念课堂**：diffusion 去噪要跑几十步，每步 prompt 的 K/V 都一样。以前每步重算，现在算一次缓存起来。典型的"循环不变量外提"优化。
- **对你有什么影响**：LTX-2 视频生成速度提升，步数越多收益越大。

### 15. [Fix] dsv41: Revert MegaGate routing in prefill (#46242)
- **代码层面**：`mega_gate.py` 把 `mode.is_extend()` 改回 `mode.is_target_verify()`，加 `_MEGA_GATE_MAX_ROWS = 2048` 硬上限，去掉对 `SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK` 的依赖。
- **新手概念课堂**：见知识点 4。这是一次"发现问题→回滚"的操作，说明 prefill 场景下 MegaGate 有正确性问题（可能是 graph capture 时模式判断失效）。
- **对你有什么影响**：dsv41 prefill 暂时回退到非 MegaGate 路由，性能略降但正确性优先。

### 16. [Fix] [AMD] dsv4.1: Fix decode graph capture with quantized low-ratio activations (#46151)
- **代码层面**：新增 `_first_rows()` 辅助函数，对 `msgspec.Struct` 类型的量化激活（`Mxfp8Activation`、`Fp8GridActivation`）逐字段切片，而不是直接 `act[:n]`（HIP 上会 TypeError）；只在需要裁剪时才切。
- **新手概念课堂**：量化激活是个 struct（包含 q 和 scale 两个 tensor），不能像普通 tensor 那样切片。这是 AMD ROCm 平台特有的坑。
- **对你有什么影响**：AMD 上跑 dsv4.1 + 量化 + decode graph 的用户，之前会崩溃，现在能正常跑。

### 17. [Perf] dsv41: Run MegaGate routing in prefill (#46232)
- **代码层面**：这是 Commit 15 的反向操作——把 `is_target_verify` 改回 `is_extend`，去掉 2048 硬上限，恢复用 `SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK` 环境变量。
- **新手概念课堂**：说明 Commit 15 的回滚是临时的，根因（环境变量上限没配好）修复后重新启用。这种"revert→fix→re-apply"是大型项目的常态。
- **对你有什么影响**：dsv41 prefill 重新获得 MegaGate 加速，前提是环境变量配置正确。

### 18. [Feat] openai: Parse DeepSeek text for the native endpoints through deepseek-recipe (#46219)
- **代码层面**：`http_server.py` 的 `parse_function_call_request` 和 `separate_reasoning_request` 先尝试 `native_text_parser()`（走 deepseek-recipe 的解析器），失败再回退到旧的 `FunctionCallParser`/`ReasoningParser`。
- **新手概念课堂**：DeepSeek 的 tool call 和 reasoning 有自己特殊的文本格式（比如 `各族人民` 标签），通用解析器解析不准。这个 commit 让原生端点用专用解析器。
- **对你有什么影响**：用 DeepSeek 模型 + 原生 `/parse_function_call` 端点的用户，工具调用解析准确率提升。

### 19. [Perf] diffusion: Wait on the socket when the scheduler is idle (#46007)
- **代码层面**：`gpu_worker.py` 的 `release_cache_if_idle` 返回"距离下次缓存释放还有几秒"；`scheduler.py` 空闲时若满足条件（单 rank、无等待队列、开启 `--sleep-on-idle`）则 `poller.poll(timeout=...)` 阻塞等待；新增 CLI 参数 `--sleep-on-idle`（默认关）。
- **新手概念课堂**：见知识点 5。核心是"空闲时别烧 CPU"，但代价是唤醒延迟，所以默认关、需要用户自己权衡。
- **对你有什么影响**：低 QPS 部署开启后 CPU 占用大降；延迟敏感场景建议先压测再开。

### 20. [Perf] diffusion: Build one LTX-2.3 AdaLN row per sample when every token shares the sigma (#46109)
- **代码层面**：新增 `_ltx23_token_timesteps()`，在 `quality_allows(batch.quality, "lossless")` 时返回 `[B, 1]` 而非 `[B, num_tokens]`，让 DiT 每样本只建一行 AdaLN 然后 broadcast；新增单测。
- **新手概念课堂**：AdaLN（自适应层归一化）需要根据 timestep 生成调制参数。如果所有 token 共享同一个 sigma（无 denoise mask 时），就没必要每个 token 算一行——broadcast 省显存省算力。
- **对你有什么影响**：LTX-2.3 lossless 模式下显存占用和计算量都降低，可以跑更大分辨率。

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)