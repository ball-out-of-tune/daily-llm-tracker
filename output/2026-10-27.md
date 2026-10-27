# 🚀 vLLM & SGLang 每日更新 — 2026-10-27

> 自动生成于 2026-10-27 06:47 UTC | AI 解读: ✅ 含代码解读

## vllm
## 💡 今日关键词

- **关键词：AMD/ROCm 生态适配**
  - 一句话解释：vLLM 正在积极让大模型推理在 AMD 的 GPU（MI300/MI350）上跑起来，包括 DeepSeek-V4、MiniMax-M3、GLM 等大模型的 PD 分离、KV 传输、量化 kernel。
  - 为什么社区在关注：今天 20 个 commit 里至少有 5 个直接跟 ROCm/AMD 相关（#48989、#63833、#63689、#63828、#63868），说明 vLLM 正在从"NVIDIA 优先"走向"多硬件平等"，异构计算已经是刚需。
  - 对新手意味着什么：如果你学的是 CUDA，现在补一点 HIP/ROCm 知识（API 几乎一样，只是 `torch.cuda` 在 AMD 上也叫 `torch.cuda`）会让你在求职时更有差异化。AMD 的软件栈正在快速扩张，人才缺口大。

- **关键词：投机解码（Speculative Decoding）优化**
  - 一句话解释：用一个"小模型"快速猜几个 token，再用大模型批量验证，从而加速生成。今天有多个 commit 在优化它的采样、预算分配、logits 分片。
  - 为什么社区在关注：这是当前 LLM 推理加速最主流的手段之一（EAGLE3、DSpark、GQA draft）。今天的 #63554、#63421 都在改投机解码的采样和预算逻辑，说明这条路线已经进入"抠细节"阶段。
  - 对新手意味着什么：投机解码是推理加速的"必修课"，建议先理解"draft-verify"两阶段模型，再去看 rejection sampling 的数学原理。

- **关键词：CUDA Graph 与内存管理**
  - 一句话解释：CUDA Graph 是把一串 GPU 操作录下来一次性重放，减少 CPU 调度开销。今天有两个 commit 在修它的内存泄漏和弱引用问题。
  - 为什么社区在关注：#63427 修的是嵌套输出的弱引用，#63154 修的是 warmup 内存没释放。这些都是"生产环境 OOM"级别的 bug，说明 CUDA Graph 已经进入大规模使用阶段，细节 bug 开始暴露。
  - 对新手意味着什么：如果你打算做推理引擎开发，CUDA Graph 是必学项。理解"捕获-重放"模型和 Python 引用计数如何交互，是进阶的关键。

- **关键词：量化（FP8/MXFP8/MXFP4）**
  - 一句话解释：把模型权重和激活从 16 位浮点压到 8 位甚至 4 位，省显存、提速度。今天有 MXFP8 scale 计算、FP8 MLA、KV cache FP8 等多个 commit。
  - 为什么社区在关注：#63689 修的是 MXFP8 在零值和次正规数上的 scale 编码 bug，这是非常底层的数值问题。说明量化已经从"能跑"进入"数值精确"阶段。
  - 对新手意味着什么：量化是性价比最高的优化方向。建议从理解 FP8 的 E4M3/E5M2 格式、scale 因子、per-tensor vs per-block 量化开始。

- **关键词：工具调用（Tool Calling）解析器引擎化**
  - 一句话解释：LLM 输出 JSON 格式的工具调用时，需要一个解析器把它变成结构化数据。今天 #63569 把 llama3_json 解析器迁移到统一的 parser engine。
  - 为什么社区在关注：Agent 应用爆发，工具调用是核心能力。统一 parser engine 意味着不同模型的解析逻辑可以复用、可测试、可维护。
  - 对新手意味着什么：如果你想做 Agent 开发，理解"流式解析"（边生成边解析 JSON）是必备技能，因为用户不想等整个 JSON 生成完才看到结果。

---

## 🧪 重要知识点 & 动手实验

### 知识点 1：弱引用与 GPU 内存释放（来自 Commit #63427）

- **是什么**：Python 的 `weakref` 允许你"引用"一个对象但不阻止它被垃圾回收。vLLM 用弱引用来持有 CUDA Graph 捕获的模型输出，避免这些 tensor 被 Python 变量"钉住"导致显存泄漏。
- **为什么重要**：CUDA Graph 捕获的输出可能嵌套在 tuple/list 里（比如 `(hidden_states, [aux_hidden_states])`）。如果只用浅层弱引用，内层 tensor 仍然被强引用持有，显存永远不释放，长时间跑就 OOM。今天这个 commit 把 `weak_ref_tensors` 改成递归处理嵌套结构。
- **动手试试**：
  ```python
  import torch, weakref, gc

  # 模拟"嵌套输出"
  hidden = torch.empty(1 << 20, device="cuda")  # 4MB
  aux = torch.empty(1 << 20, device="cuda")
  out = (hidden, [aux])

  # 浅层弱引用：只弱引用外层 tuple，内层 tensor 仍被强引用
  weak_out = weakref.ref(out)
  del out
  gc.collect()
  print("浅层弱引用后显存:", torch.cuda.memory_allocated() / 1e6, "MB")
  # 仍然 ~8MB，因为 hidden/aux 还被 out 的内部结构持有？其实不会，但如果
  # 你只 weakref 外层而让内层被别的容器持有，就会泄漏。

  # 正确做法：递归弱引用（vLLM 的做法）
  def weak_ref_tensors(x):
      if isinstance(x, torch.Tensor):
          return weakref.ref(x)
      if isinstance(x, list):
          return [weak_ref_tensors(i) for i in x]
      if isinstance(x, tuple):
          return tuple(weak_ref_tensors(i) for i in x)
      return x

  hidden = torch.empty(1 << 20, device="cuda")
  aux = torch.empty(1 << 20, device="cuda")
  out = weak_ref_tensors((hidden, [aux]))
  del hidden, aux
  gc.collect()
  print("递归弱引用后显存:", torch.cuda.memory_allocated() / 1e6, "MB")
  ```
- **预期结果**：递归弱引用后显存应该降到接近 0（只剩 CUDA context 的固定开销）。如果你看到浅层版本显存不降，说明内层 tensor 被强引用钉住了。

---

### 知识点 2：CUDA 共享内存与 Triton kernel 的 tile 调优（来自 Commit #60942）

- **是什么**：GPU 的 shared memory（共享内存）是每个 SM 上的一块高速缓存，Triton kernel 的 tile 大小（BLOCK）必须让 Q、K、P 三个 tile 塞进 shared memory，否则会报 `OutOfResources`。
- **为什么重要**：不同 GPU 的 shared memory 大小不同（比如 99KiB vs 164KiB）。如果 kernel 写死 BLOCK=64，在 99KiB 的卡上 head_dim=512 就会炸。今天这个 commit 加了 `_fit_block_to_shared_memory`，自动把 BLOCK 减半直到装得下。
- **动手试试**：
  ```python
  import torch
  from vllm.utils.mem_utils import get_max_shared_memory_bytes

  max_smem = get_max_shared_memory_bytes()
  print(f"当前 GPU shared memory: {max_smem / 1024:.1f} KiB")

  def fit_block(block, head_dim, elem_size):
      while block > 16 and elem_size * block * (2 * head_dim + block) > max_smem:
          block //= 2
      return block

  for head_dim in [128, 256, 512]:
      for elem_size in [2, 4]:  # bf16=2, fp32=4
          b = fit_block(64, head_dim, elem_size)
          need = elem_size * b * (2 * head_dim + b) / 1024
          print(f"head_dim={head_dim}, elem={elem_size}B -> BLOCK={b}, 需要 {need:.1f} KiB")
  ```
- **预期结果**：你会看到 head_dim 越大、元素字节越大，BLOCK 被自动减得越小。在 99KiB 的卡上，head_dim=512 + fp32 时 BLOCK 可能降到 16 甚至更小。

---

### 知识点 3：MXFP8 量化中的次正规数处理（来自 Commit #63689）

- **是什么**：MXFP8 是一种块量化格式，每 32 个元素共享一个 8 位的 scale（E8M0，即只有指数、没有尾数）。当一组数的最大值（amax）非常小（次正规数）或为 0 时，scale 的计算要特殊处理，否则会溢出或精度丢失。
- **为什么重要**：如果 amax 是 0（整组全零），`log2(0)` 会变成 `-inf`。如果 amax 是次正规数（比如 1e-40），直接算指数会得到负数。今天这个 commit 用位操作判断 `bits < 0x800000`（即小于最小正规数 2^-126）时，把 scale 设为 1 或 0，避免错误。
- **动手试试**：
  ```python
  import numpy as np

  FP8_MAX = 448.0  # E4M3 的最大值
  FLT_MIN = np.finfo(np.float32).tiny  # 最小正规数 ~1.18e-38

  def mx_code_naive(amax):
      a = max(amax, FLT_MIN)
      q = a / FP8_MAX
      return np.clip(np.ceil(np.log2(q)), 0, 254)

  def mx_code_fixed(amax):
      a = max(amax, FLT_MIN)
      q = a / FP8_MAX
      bits = np.float32(q).view(np.int32)
      sub = 1 if bits > 0x400000 else 0  # > 2^-126
      e = sub if bits < 0x800000 else np.ceil(np.log2(q))
      return int(np.clip(e, 0, 254))

  for amax in [0.0, 1e-40, 1e-38, 1e-30, 1.0, 448.0]:
      print(f"amax={amax:>10.3e}  naive={mx_code_naive(amax):>3}  fixed={mx_code_fixed(amax):>3}")
  ```
- **预期结果**：对 amax=0 和 amax=1e-40，naive 版本会给出奇怪的值（`log2(0)=-inf` 被 clip 到 0，或者 `log2(1e-40/448)` 是负数被 clip 到 0），而 fixed 版本会稳定给出 0 或 1。这就是"零组和次正规组"的正确 scale 编码。

---

### 知识点 4：KV Cache 块拷贝中的"叠加视图"陷阱（来自 Commit #55112）

- **是什么**：混合模型（比如 Mamba + Attention）会把不同类型的层"叠加"在同一块显存上。同一个 `data_ptr` 可能对应两个不同形状的 tensor——一个 Mamba 状态视图（只占块的前 24 字节）和一个 Attention 视图（占满 64 字节）。拷贝 KV block 时如果只按 `data_ptr` 去重，就会漏掉另一种几何形状的视图。
- **为什么重要**：漏拷贝会导致目标 block 的尾部残留旧数据，推理结果错误。今天这个 commit 把去重 key 从 `(device, data_ptr)` 改成 `(device, data_ptr, shape, stride, element_size)`，确保每种几何形状都被拷贝。
- **动手试试**：
  ```python
  import torch

  # 模拟一块 256 字节的 storage，两个 tensor 叠加在上面
  raw = torch.arange(256, dtype=torch.uint8)
  storage = raw.untyped_storage()

  # Mamba 视图：每块 64 字节中只占前 24 字节
  mamba = torch.empty(0, dtype=torch.int8).set_(storage)
  mamba = mamba.as_strided((4, 1, 1, 24), (64, 1, 1, 1), 0)

  # Attention 视图：每块 64 字节全占
  attn = torch.empty(0, dtype=torch.int16).set_(storage)
  attn = attn.as_strided((8, 1, 8, 2), (16, 16, 2, 1), 0)

  print("data_ptr 相同:", mamba.data_ptr() == attn.data_ptr())
  print("shape 不同:", mamba.shape, attn.shape)

  # 如果按 data_ptr 去重，只会拷贝 mamba 的 24 字节，attn 的剩余 40 字节漏掉
  # 正确做法：key 包含 shape/stride/element_size
  key1 = (mamba.data_ptr(), tuple(mamba.shape), tuple(mamba.stride()), mamba.element_size())
  key2 = (attn.data_ptr(), tuple(attn.shape), tuple(attn.stride()), attn.element_size())
  print("两个 key 不同:", key1 != key2)
  ```
- **预期结果**：你会看到 `data_ptr` 相同但 `shape`/`stride`/`element_size` 不同，说明必须用完整几何信息做去重 key。这就是今天 commit 的核心修复。

---

### 知识点 5：投机解码的 draft token 预算分配（来自 Commit #63421）

- **是什么**：投机解码里，每个请求可以"猜"几个 token 是有预算上限的（比如总共 800 个 draft token 分给 300 个请求）。今天这个 commit 把原来用 `torch.compile` 实现的预算分配换成了手写 Triton kernel，还加了 tie-breaking（平局时按请求顺序分配）。
- **为什么重要**：`torch.compile` 编译一个简单的 top-k 逻辑会引入编译开销和不确定性。手写 Triton kernel 更可控、更快，而且能精确处理"平局"这种边界情况（比如所有请求的置信度都是 1.0 时，谁先拿到预算？）。
- **动手试试**：
  ```python
  import torch

  def global_topk_budget(confidence_probs, idx_mapping, capacities, draft_budget):
      # confidence_probs: [max_reqs, num_steps]
      # idx_mapping: [num_reqs] 每个请求在 confidence_probs 里的行号
      survival = confidence_probs[idx_mapping].cumprod(dim=1)
      steps = torch.arange(survival.shape[1], device=survival.device)
      survival = survival.masked_fill(steps >= capacities[:, None], -float("inf"))
      admitted = torch.zeros(survival.numel(), dtype=torch.bool, device=survival.device)
      admitted[survival.flatten().topk(draft_budget).indices] = True
      return admitted.view_as(survival).sum(dim=1, dtype=torch.int32)

  # 模拟：3 个请求，每个最多 3 步，预算 5
  probs = torch.tensor([[0.9, 0.8, 0.7],
                        [0.6, 0.5, 0.4],
                        [0.3, 0.2, 0.1]])
  idx = torch.arange(3)
  caps = torch.tensor([3, 3, 3], dtype=torch.int32)
  result = global_topk_budget(probs, idx, caps, draft_budget=5)
  print("每个请求分到的 draft 数:", result.tolist())
  # 预期：survival = [0.9, 0.72, 0.504], [0.6, 0.3, 0.12], [0.3, 0.06, 0.006]
  # top-5 会选中前两个请求的所有 3 步 + 第三个请求的第 1 步
  # 结果应该是 [3, 3, 1] 或类似
  ```
- **预期结果**：你会看到置信度高的请求优先拿到预算。如果所有 probs 都是 1.0（平局），kernel 会按请求顺序分配，前 266 个请求拿满 3 个，第 267 个拿 2 个，后面拿 0 个——这正是 commit 里 `test_draft_budget_kernel_admits_exact_budget_on_ties` 验证的行为。

---

## 🔥 重要更新

1. **#63427 [Bugfix][CUDA Graph] 弱引用嵌套输出**：修复了 CUDA Graph 捕获输出嵌套在 list/tuple 里时显存不释放的严重 bug，长时间服务会 OOM。所有用 CUDA Graph 的用户都受益。

2. **#55112 [Bugfix][Core] 修复叠加 KV 视图拷贝**：混合模型（Mamba+Attention）的 KV block 拷贝漏数据，会导致推理结果错误。这是正确性 bug，影响所有混合架构模型。

3. **#63689 [Bugfix][ROCm][DSv4] MXFP8 零组/次正规组 scale**：DeepSeek-V4 在 AMD 上的量化精度 bug，零值和极小值组的 scale 编码错误。影响所有用 MXFP8 量化的 DSv4 推理。

4. **#63554 [Spec Decode] Batch-sharded draft logits**：投机解码的 draft logits 现在可以按 batch 分片，大幅降低大 batch 下的显存峰值。这是性能/显存双重优化。

---

## 📋 逐条解读

### 1. [Frontend][Mistral] Forward response_format to mistral-common chat templating (#63812)

- **代码层面**：改的是 `tests/entrypoints/serve/tokenize/test_serving_tokenization.py` 和 `tests/renderers/test_mistral.py`。核心是把 OpenAI 风格的 `response_format`（比如 `json_schema`）透传给 Mistral 的 chat template 渲染器。以前 tokenize 接口不认 `response_format`，现在会把它转成 Mistral 内部的格式。
- **新手概念课堂**：`response_format` 就像你点外卖时备注"不要辣"——你告诉模型"我要 JSON 格式的输出"。Mistral 有自己的模板系统（mistral-common），需要把 OpenAI 的格式翻译成它的格式，就像把中文菜单翻译成英文给外国厨师看。
- **对你有什么影响**：如果你用 vLLM 跑 Mistral 模型并需要结构化输出（JSON），tokenize 接口现在能正确反映 `response_format` 对 token 数的影响。以前 tokenize 和实际生成可能不一致。

---

### 2. [ROCm] DeepSeek-V4-Pro PD Disaggregation through MORI IO KV Connector on AMD GPUs (#48989)

- **代码层面**：改了 `.buildkite/amd-disagg/models.yaml` 和 `pipeline-disagg.yaml`（CI 配置），以及 `vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py`。新增了 DeepSeek-V4-Pro 在 AMD MI350 上的 PD 分离测试配置（1P1D TP8、EP8/DP8 WideEP、2P2D EP16/DP16 三种拓扑）。
- **新手概念课堂**：PD 分离（Prefill-Decode Disaggregation）就像餐厅把"备菜"和"炒菜"分成两个厨房——prefill 阶段处理长 prompt（备菜），decode 阶段逐 token 生成（炒菜）。两个厨房通过 MORI IO（一种高速 KV 传输通道）传递半成品。这样每个厨房可以独立扩容。
- **对你有什么影响**：如果你在 AMD GPU 上跑 DeepSeek-V4，现在有了官方验证的 PD 分离方案。普通用户感知不到，但运维团队可以按这个配置部署。

---

### 3. [Bugfix][CUDA Graph] Weak-ref nested outputs of captured graphs (#63427)

- **代码层面**：改了 `vllm/utils/torch_utils.py` 的 `weak_ref_tensors` 函数。原来只处理单层 tensor/list/tuple，现在递归处理嵌套结构，并且支持 `int` 和 `LayerName` 原样返回。测试文件 `tests/utils_/test_torch_utils.py` 新增了两个测试：一个验证嵌套 tensor 能被正确回收，一个验证未知类型会报错。
- **新手概念课堂**：弱引用就像"便利贴"——你贴在冰箱上提醒自己"牛奶快没了"，但便利贴不会阻止牛奶被喝掉。强引用就像"把牛奶锁在保险柜里"，只要锁着就没人能扔。CUDA Graph 捕获的输出如果被强引用钉住，显存就永远不释放。
- **对你有什么影响**：长时间运行 vLLM 服务时，显存不会再因为 CUDA Graph 输出而缓慢泄漏。这是稳定性提升。

---

### 4. [Bugfix][ROCm] Wait on per-slot events in packed NCCL consumer (#63833)

- **代码层面**：改了 `vllm/distributed/weight_transfer/packed_tensor.py` 的 `packed_nccl_broadcast_consumer` 函数。原来用 `streams[buffer_idx].synchronize()` 同步整个 stream，现在改成每个 slot 记录一个 `torch.Event`，只等这个 slot 自己的 event。
- **新手概念课堂**：在 AMD GPU 上，多个 stream 可能共享同一个硬件队列。如果你 synchronize 一个 stream，会连带等待其他 stream 上后来排队的任务——就像你在 1 号窗口排队，结果 2 号窗口的人也插到你前面了。用 per-slot event 就像给每个窗口发一个"叫号器"，只等自己那个号。
- **对你有什么影响**：AMD GPU 上做权重传输（比如 RLHF 训练时更新权重）不会再因为 stream 串扰而变慢或死锁。

---

### 5. [revert] [Build] Fix QuTLASS compilation with PyTorch 2.13 (#61844)

- **代码层面**：改了 `cmake/external_projects/qutlass.cmake`。回退了之前为 PyTorch 2.13 做的编译修复，改成检查 Qutlass 源码里是否有 `registration.h`（Stable ABI 支持），并设置 `TORCH_TARGET_VERSION=0x020D000000000000ULL`（对应 2.13）。
- **新手概念课堂**：QuTLASS 是一个 CUDA 库，依赖 PyTorch 的 ABI（二进制接口）。PyTorch 2.13 改了 ABI，导致 QuTLASS 编译失败。这个 commit 是"回退 + 加检查"，确保只有支持 Stable ABI 的 Qutlass 版本才能编译。
- **对你有什么影响**：如果你从源码编译 vLLM 并启用了 QuTLASS，需要确保你的 Qutlass 包足够新（有 `registration.h`）。否则编译会报错并提示你更新。

---

### 6. [Bugfix][Core] Copy every overlaid KV view in copy_kv_cache_blocks_inplace (#55112)

- **代码层面**：改了 `vllm/v1/worker/utils.py` 的 `copy_kv_cache_blocks_inplace` 函数。去重 key 从 `(device, data_ptr)` 扩展为 `(device, data_ptr, shape, stride, element_size)`。测试文件 `tests/v1/worker/test_attn_utils.py` 新增了一个测试，模拟 Mamba 视图和 Attention 视图叠加在同一 storage 上的场景。
- **新手概念课堂**：想象一块地皮上盖了两栋楼——一栋是矮平房（Mamba 状态，只占前 24 米），一栋是高楼（Attention，占满 64 米）。如果拆迁时只按"地皮地址"去重，就会以为只拆了平房，高楼没动。正确做法是按"地皮地址 + 楼的形状"去重。
- **对你有什么影响**：混合模型（Mamba + Attention，比如 Jamba）的 KV cache 拷贝现在正确了。以前可能出现"目标 block 尾部残留旧数据"导致的推理错误。

---

### 7. [Docs][Recipe] Align MiniMax-M3 MI355X AgentX monokernel profile (#63868)

- **代码层面**：改了 `recipes/models/MiniMaxAI/MiniMax-M3.yaml`。新增了 `mi355x_agentx` 预设，包含 TP4 和 TP2 两种配置，TP4 启用 monokernel（单 kernel 融合），TP2 用普通 decode。还配了一大堆环境变量（AITER、QuickReduce、KV cache shuffle 等）。
- **新手概念课堂**：Monokernel 就像把"切菜、炒菜、装盘"三个步骤合并成一个机器人动作，减少中间交接开销。AgentX 是 InferenceX 的 agentic 工作负载基准，模拟真实 Agent 场景（多轮工具调用、长上下文）。
- **对你有什么影响**：如果你在 AMD MI355X 上跑 MiniMax-M3 做 Agent 应用，可以直接抄这个 recipe 的配置，省去调参时间。

---

### 8. [Bugfix] Fix Triton prefill attention OutOfResources on GPUs with 99KiB shared memory (#60942)

- **代码层面**：改了 `vllm/v1/attention/ops/triton_prefill_attention.py`。新增 `_fit_block_to_shared_memory` 函数（带 `@cache`），在 CUDA 平台上根据 shared memory 大小自动把 BLOCK 减半。测试文件 `tests/kernels/attention/test_triton_prefill_attention.py` 把 head_dim 参数从 `[128]` 扩展到 `[128, 256, 512]`。
- **新手概念课堂**：Shared memory 就像厨房的操作台——太小了放不下所有食材（Q、K、P tile），就得把食材分批处理（减小 BLOCK）。99KiB 的卡（比如某些消费级 GPU）操作台小，164KiB 的卡（A100/H100）操作台大。
- **对你有什么影响**：在 shared memory 较小的 GPU 上跑 Triton prefill attention，head_dim 较大时不会再报 `OutOfResources`。kernel 会自动选择合适的 BLOCK 大小。

---

### 9. [Bugfix][Spec Decode] Apply the draft max_model_len limit before model config hooks (#63853)

- **代码层面**：改了 `vllm/config/vllm.py` 的 `__post_init__`。把 `self._limit_max_model_len_to_draft()` 的调用从后面移到 `try_verify_and_update_config()` 之前。测试文件新增了一个针对混合 Mamba 模型的测试。
- **新手概念课堂**：模型配置的初始化顺序就像做菜的步骤——先切菜再炒菜。如果先炒菜（model config hooks）再切菜（限制 max_model_len），就会用错食材。混合模型（Jamba）的 `mamba_block_size` 默认等于 `max_model_len`，如果 draft 模型的 `max_model_len` 限制没先应用，就会触发 `--mamba-block-size` 的验证错误。
- **对你有什么影响**：用 EAGLE3/draft model 加混合模型（Jamba）做投机解码时，不会再因为配置顺序问题启动失败。

---

### 10. [Bugfix][ROCm][DSv4] Give zero and subnormal mono MXFP8 groups vLLM's scale code (#63689)

- **代码层面**：改了 4 个文件：`vllm/models/deepseek_v4/amd/mono/attention/device.py`、`vllm/models/deepseek_v4/amd/mono/stages/moe.py`、`vllm/models/deepseek_v41/amd/mono/attention/device.py`、`vllm/models/deepseek_v41/amd/mono/stages/moe.py`。在 `mx_code` 和 `vllm_mx_code` 函数里加了次正规数判断：`bits < 0x800000` 时用 `sub`（0 或 1）代替 `ceil_exp`。
- **新手概念课堂**：MXFP8 的 scale 是 8 位指数（E8M0），只能表示 2 的幂。当一组数的最大值是 0 或极小（次正规数）时，`log2` 会得到 `-inf` 或负数。代码用位操作判断"这个数是不是小于最小正规数 2^-126"，如果是，就把 scale 设为 0 或 1，避免数值错误。
- **对你有什么影响**：DeepSeek-V4 在 AMD GPU 上用 MXFP8 量化时，零值组和次正规组的 scale 编码正确了。以前可能出现精度异常或 NaN。

---

### 11. [Bugfix][MRV2] Release the warmup memory cached on the capture streams (#63154)

- **代码层面**：改了 `vllm/v1/worker/gpu/model_runner.py` 的 `capture_model` 方法。在捕获 CUDA Graph 之后、测量空闲显存之前，加了一行 `torch.accelerator.empty_cache()`。
- **新手概念课堂**：CUDA Graph 捕获时会做 warmup（预热），warmup 产生的临时 tensor 会被缓存在 capture stream 上。捕获完成后，没有代码再从这些 stream 分配内存，所以这些缓存永远不释放。加一行 `empty_cache()` 就像"打扫完厨房把垃圾倒掉"。
- **对你有什么影响**：启动 vLLM 服务时，CUDA Graph 捕获后的空闲显存会更多（warmup 内存被释放）。这意味着你可以用更高的 `gpu_memory_utilization` 或更大的 batch。

---

### 12. [Spec Decode][Sampling] Batch-sharded draft logits (#63554)

- **代码层面**：改了多个测试文件和 `vllm/v1/worker/gpu/spec_decode/` 下的采样逻辑。新增 `draft_logits_num_shards` 和 `draft_logits_shard_idx` 两个属性，让 draft logits 可以按 batch 分片存储和访问。
- **新手概念课堂**：draft logits 是"小模型猜的每个 token 的概率分布"。如果 batch 很大（比如 512 个请求 × 5 个 draft token × 128K 词表），logits 会占大量显存。分片就像把一本大字典拆成几册，每次只查一册。
- **对你有什么影响**：大 batch 投机解码时显存峰值降低，可以跑更大的 batch 或更长的 draft 长度。

---

### 13. [Doc] Kimi-K3: refresh the GB300 AgentX xPyD preset and add the 3P1D DEP16 preset (#63852)

- **代码层面**：改了 `recipes/models/moonshotai/Kimi-K3.yaml`。更新了 GB300 AgentX xPyD 预设的镜像版本、`gpu-memory-utilization`（0.92→0.95）、删除了两个 Mooncake 环境变量、新增了 `TOKENSPEED_MLA_FP16_PARTIALS: "0"`。prefill 角色新增了 CUDA Graph 捕获到 8192 token 的配置。还新增了 3P1D DEP16 预设。
- **新手概念课堂**：xPyD 是 PD 分离的拓扑表示——x 个 prefill 节点、y 个 decode 节点。3P1D 就是 3 个 prefill + 1 个 decode。DEP16 是 Decode 的 EP16（专家并行 16 路）。这些 recipe 就像"官方推荐配置单"，照着抄就能跑出不错的性能。
- **对你有什么影响**：如果你在 GB300 上部署 Kimi-K3，直接抄这个 YAML 的配置即可。普通用户感知不到。

---

### 14. [Perf][Kimi K3] Opt-in KV-balanced FP8 MLA decode for agentic workloads (#63638)

- **代码层面**：新增了 `tests/models/kimi_k3/test_kv_balanced_mla.py`（446 行测试）和 `vllm/models/kimi_k3/nvidia/kv_balanced_mla/` 目录下的 backend 实现。新增 `KIMI_K3_KV_BALANCED_MLA` 这个 attention backend 枚举，但**不会自动被平台选中**，必须显式 opt-in。
- **新手概念课堂**：MLA（Multi-head Latent Attention）是 DeepSeek 提出的注意力变体，把 KV 压缩到一个低秩 latent 空间。KV-balanced 是一种负载均衡策略，让不同 GPU 上的 KV 计算量更均匀。Agentic 工作负载（多轮对话、工具调用）的 KV 分布和普通 chat 不同，需要专门优化。
- **对你有什么影响**：如果你在 SM100 系列 GPU（B200/GB200）上跑 Kimi-K3 做 Agent 应用，可以显式启用这个 backend 获得更好的 decode 性能。默认不启用，所以不影响现有用户。

---

### 15. [NIXL] Bump NIXL version to 1.5.1 (#63817)

- **代码层面**：改了 `requirements/kv_connectors.txt` 和 `requirements/kv_connectors_cu12.txt`，把 `nixl` 从 `1.5.0` 升到 `1.5.1`。
- **新手概念课堂**：NIXL 是 NVIDIA 的 KV 传输库，用于 PD 分离时在节点间传 KV cache。版本升级通常包含 bug 修复和性能改进。
- **对你有什么影响**：用 NIXL 做 KV 传输的用户升级后可能获得稳定性/性能提升。普通用户无感。

---

### 16. [Tool Calling] Migrate llama3_json parser to parser engine (#63569)

- **代码层面**：新增 `tests/parser/engine/test_llama_json.py`（398 行），改了 `tests/parser/engine/test_delegating_replay.py` 和 `test_granite.py`。把 Llama 3 的 JSON 工具调用解析器迁移到统一的 parser engine 架构。
- **新手概念课堂**：Llama 3 的工具调用格式是分号分隔的 JSON 对象（`{"name": "search", "parameters": {...}};{"name": "calc", ...}`）。parser engine 是一个统一框架，让不同模型的解析器共享流式解析、增量输出、错误处理等基础设施。
- **对你有什么影响**：用 Llama 3 做工具调用时，解析更稳定、更可测试。流式输出时能更早看到工具调用参数。

---

### 17. [ROCm][CI] Release checkpoint page cache between language generation tests (#63828)

- **代码层面**：新增 `tests/models/language/generation/conftest.py`。加了一个 autouse fixture，在 ROCm 平台上，当测试的模型变化时调用 `drop_checkpoint_cache(_previous_model)` 释放上一个模型的 page cache。
- **新手概念课堂**：Page cache 是操作系统缓存的文件数据。在容器里，page cache 算进 cgroup 的内存限制。ROCm CI 跑多个模型测试时，checkpoint 文件（几十 GB）的 page cache 会撑爆 pod 内存。这个 fixture 就像"每道菜做完把案板洗干净"。
- **对你有什么影响**：只影响 ROCm CI，普通用户无感。但如果你在容器里跑多个模型测试，可以参考这个做法避免 OOM。

---

### 18. [Perf][MoE] Size the DeepGEMM MegaMoE symmetric buffer for one sequence-parallel shard (#63541)

- **代码层面**：改了 `vllm/model_executor/layers/fused_moe/deep_gemm_mega_moe.py` 和 `vllm/models/kimi_k3/nvidia/model.py`。`DeepGemmMegaMoEExperts.__init__` 新增 `max_num_tokens` 参数，Kimi-K3 在 sequence parallel 时传入 `cdiv(max_num_batched_tokens, tp_size)`。
- **新手概念课堂**：DeepGEMM

## sglang
## 💡 今日关键词

- **关键词：FP8 量化 + 校准 scale**
  - **一句话解释**：把模型权重/激活从 16 位浮点压缩到 8 位浮点，用"缩放因子"（scale）保证精度，scale 需要提前校准好。
  - **为什么社区在关注**：Commit 1 修复了 AMD aiter 后端在"FP8 KV cache + 校准过的 KV scale"场景下输出乱码的问题——说明 FP8 推理已经从"能跑"进入"精度正确性"的深水区，各硬件厂商都在补细节坑。
  - **对新手意味着什么**：量化不再是"调个 flag"，你需要理解 scale 的语义（K 的 scale 不能拿来当 Q 的 scale 用），这是做推理优化岗的必备知识。

- **关键词：异构硬件后端（AMD / Intel XPU / NVIDIA）**
  - **一句话解释**：同一套推理框架要跑在 AMD ROCm、Intel XPU、NVIDIA CUDA 等不同芯片上。
  - **为什么社区在关注**：今天 20 个 commit 里有 3 个是 AMD/Intel 专属优化或修复，说明 SGLang 正在认真做多硬件支持，GPU 不再是 NVIDIA 一家独大。
  - **对新手意味着什么**：如果你只会 CUDA，建议了解一下 ROCm/Triton；跨平台 kernel 开发是稀缺技能。

- **关键词：CUDA Graph / Breakable CUDA Graph（BCG）**
  - **一句话解释**：把一串 GPU 操作"录"下来一次性重放，减少 CPU 调度开销；"可打断的"版本允许在中间插入自定义逻辑。
  - **为什么社区在关注**：Commit 7、12、13 都在把"写入 caller 分配的 buffer"这种模式引入 BCG——这是为了在 graph 内部做内存复用和避免动态分配，是高性能推理的核心技巧。
  - **对新手意味着什么**：想搞推理加速，CUDA Graph 是绕不开的一课；"预分配输出 buffer"是工程落地的基本功。

- **关键词：KV Cache 内存池与分页（Paged KV / Pool Configurator）**
  - **一句话解释**：把 KV cache 切成固定大小的"页"，像操作系统管理内存一样管理显存。
  - **为什么社区在关注**：Commit 10 修复了 DeepSeek-V4 的 KV 容量计算与物理布局不匹配的问题——容量算错会导致 OOM 或浪费显存，这是长上下文推理的关键。
  - **对新手意味着什么**：理解 PagedAttention 和显存预算公式，是读懂 vLLM/SGLang 源码的敲门砖。

- **关键词：多模态（Multimodal）工程化重构**
  - **一句话解释**：让模型同时处理文本、图像、视频、音频。
  - **为什么社区在关注**：Commit 8 是"MM refactor 1/9"系列的第一步，说明多模态代码正在从"能跑就行"走向模块化、可维护。
  - **对新手意味着什么**：多模态是当前最热的应用方向，参与这类重构能快速理解大型项目的架构演进。

---

## 🧪 重要知识点 & 动手实验

- **知识点 1：FP8 量化中的 scale 语义（来自 Commit 1）**
  - **是什么**：量化时用 `x_fp8 = x / scale`，反量化用 `x = x_fp8 * scale`。**K 的 scale 和 Q 的 scale 是两回事**，不能混用。
  - **为什么重要**：Commit 1 的 bug 就是拿 `k_scale` 当 `q_scale` 用，导致 attention 输出乱码。真实场景里，校准数据只对特定张量有效，混用 scale 会静默产生错误结果（不报错，但输出是垃圾）。
  - **动手试试**：
    ```python
    import torch
    # 模拟：Q 的真实范围是 [-10, 10]，K 的真实范围是 [-1, 1]
    q = torch.randn(4, 8) * 10
    k = torch.randn(4, 8) * 1
    fp8_max = 448.0  # e4m3 最大值
    q_scale = q.abs().max() / fp8_max
    k_scale = k.abs().max() / fp8_max
    # 错误做法：用 k_scale 量化 q
    q_wrong = (q / k_scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    q_wrong_deq = q_wrong.float() * k_scale
    # 正确做法
    q_right = (q / q_scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    q_right_deq = q_right.float() * q_scale
    print("错误量化误差:", (q - q_wrong_deq).abs().mean().item())
    print("正确量化误差:", (q - q_right_deq).abs().mean().item())
    ```
  - **预期结果**：错误做法的误差远大于正确做法（甚至可能因超出 fp8 范围被 clamp 而严重失真），直观理解"scale 不能串用"。

- **知识点 2：Triton 写一个 NaN 检测 kernel（来自 Commit 11）**
  - **是什么**：用 Triton 写 GPU kernel，在设备端直接断言"没有 NaN"，避免 CPU-GPU 同步。
  - **为什么重要**：`torch.isnan(x).any().item()` 会强制同步，打断流水线。异步断言让检查"免费"地挂在计算流上，出错时才暴露——这是高性能调试的标准手法。
  - **动手试试**：
    ```python
    import torch, triton
    import triton.language as tl

    @triton.jit(debug=True)
    def assert_no_nan(x_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        vals = tl.load(x_ptr + offs, mask=offs < n, other=0.0)
        tl.device_assert(vals == vals, "NaN detected!")

    x = torch.randn(1024, device="cuda")
    assert_no_nan[(1,)](x, x.numel(), BLOCK=1024)  # 正常
    x[0] = float("nan")
    assert_no_nan[(1,)](x, x.numel(), BLOCK=1024)
    torch.cuda.synchronize()  # 这里才报错
    ```
  - **预期结果**：第一次调用静默通过；第二次在 `synchronize()` 时抛出 device assert 错误，且过程中没有提前同步。

- **知识点 3：Python 列表的"原地修改 vs 重新绑定"（来自 Commit 5）**
  - **是什么**：`list.extend()` 会原地修改；`list = list + [...]` 会创建新列表。如果别人持有旧列表的引用，原地修改会"污染"别人。
  - **为什么重要**：Commit 5 的 bug 是 padding 时 `extend` 了共享列表，导致预先规划好的 metadata 行数被意外改变。这是 Python 里最隐蔽的 bug 类型之一。
  - **动手试试**：
    ```python
    planned = [1, 2]
    shared = planned          # 别名
    # 错误：原地 extend
    shared.extend([0])
    print("原地修改后 planned:", planned)   # [1, 2, 0] 被污染了！

    planned = [1, 2]
    shared = planned
    # 正确：重新绑定
    shared = shared + [0]
    print("重新绑定后 planned:", planned)   # [1, 2] 安全
    ```
  - **预期结果**：第一种写法 `planned` 被意外修改，第二种不受影响，理解"别名 vs 拷贝"。

- **知识点 4：mmap + MADV_HUGEPAGE 大页分配（来自 Commit 6）**
  - **是什么**：用 `mmap` 分配匿名内存，并用 `madvise(MADV_HUGEPAGE)` 让内核尽量用 2MB 大页；多线程并行 prefault 加速首次触页。
  - **为什么重要**：Host KV cache 动辄几十 GB，普通 4KB 页的缺页中断会拖慢启动；大页 + 并行预热能把分配时间从分钟级降到秒级。
  - **动手试试**：
    ```python
    import mmap, ctypes, os
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    size = 256 * 1024 * 1024  # 256MB
    mm = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    # 建议内核使用大页
    MADV_HUGEPAGE = 14
    libc.madvise(ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(mm))),
                 ctypes.c_size_t(size), ctypes.c_int(MADV_HUGEPAGE))
    # 并行 prefault
    from concurrent.futures import ThreadPoolExecutor
    chunk = 2 * 1024 * 1024
    def touch(i):
        mm[i*chunk] = 0
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(touch, range(size // chunk)))
    print("分配并预热完成")
    ```
  - **预期结果**：观察到分配/预热明显快于单线程逐字节 touch，`/proc/self/smaps` 中 `AnonHugePages` 会增长。

- **知识点 5：HuggingFace 本地缓存直接解析（来自 Commit 19）**
  - **是什么**：用 `snapshot_download(local_files_only=True)` 拿到本地已缓存的 snapshot 目录，直接列文件，避免调用 Hub API。
  - **为什么重要**：离线环境或网络抖动时，Hub API 调用会超时；本地解析让"已下载的权重"能秒级定位，且能判断"文件是否真的在缓存里"。
  - **动手试试**：
    ```python
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    from pathlib import Path

    try:
        snap = Path(snapshot_download("bert-base-uncased", local_files_only=True))
        files = [p.relative_to(snap).as_posix() for p in snap.rglob("*") if p.is_file()]
        print("缓存中文件数:", len(files))
        print("前 5 个:", files[:5])
    except LocalEntryNotFoundError:
        print("未缓存，需要联网下载")
    ```
  - **预期结果**：若之前下载过该模型，会列出 `config.json`、`pytorch_model.bin` 等；否则提示未缓存。整个过程零网络请求。

---

## 🔥 重要更新

1. **Commit 1（AMD aiter FP8 KV 修复）**：修复了 FP8 量化下 Q/K scale 混用导致的乱码，是精度正确性级别的关键修复，AMD 用户必看。
2. **Commit 10（DeepSeek-V4 KV 容量匹配物理布局）**：修复显存预算计算与真实分页布局不一致的问题，避免 OOM 或显存浪费，长上下文场景影响大。
3. **Commit 5（DP attention padding 原地修改 bug）**：典型的"共享列表被原地污染"问题，可能导致 DSA metadata 行数错乱，属于隐蔽但影响推理正确性的修复。
4. **Commit 20（无需 draft 模型也能捕获 aux hidden states）**：新增 `--enable-return-aux-hidden-states`，让 target 模型直接输出辅助隐状态，简化了 EAGLE3/DFLASH/DSPARK 的训练数据采集流程。

---

## 📋 逐条解读

### 1. [Fix] [AMD] aiter: Fix garbage output with an fp8 KV cache and calibrated KV scales
- **代码层面**：改动 `aiter_backend.py`。新增 `_quantize_q_fp8()` 方法，用专门的 `q_scale` 量化 Q（而不是复用 `k_scale`），并修正 decode 路径传给 kernel 的 k/v descale。
- **新手概念课堂**：想象考试评分——数学卷满分 100，语文卷满分 150。你不能用数学的"满分"去归一化语文分数。Q 和 K 就是两张"不同量纲"的卷子，各自需要自己的 scale。
- **对你有什么影响**：用 AMD GPU + FP8 KV cache 的用户，之前可能遇到"输出乱码/重复"，升级后恢复正常。

### 2. [Feat] [Intel] docker/xpu: Upgrade oneAPI to 2026.1.3 and sglang-kernel-xpu to 0.5.0
- **代码层面**：只改了两处版本号——`xpu.Dockerfile` 的基础镜像和 `pyproject_xpu.toml` 里的 wheel 地址。
- **新手概念课堂**：就像手机系统升级，底层驱动（oneAPI）和 App（sglang-kernel-xpu）都要配套更新，否则可能不兼容。
- **对你有什么影响**：Intel XPU 用户重新构建镜像即可获得新版本，普通用户无感。

### 3. [Perf] [AMD] Qwen3-Next: fused QK Gemma RMSNorm + partial RoPE on gfx950
- **代码层面**：在 `qwen3_next.py` 新增 `forward_prepare_fused()`，把 QKV 投影后的"Q/gate 拆分 + QK RMSNorm + 部分 RoPE"融合成一个 aiter kernel 调用（仅 gfx950 且开启 `SGLANG_USE_AITER` 时生效），失败则回退原生路径。
- **新手概念课堂**：原来要跑三个小工序（拆、归一化、旋转编码），现在合并成一台机器一次完成，减少数据来回搬运——这就是"算子融合"。
- **对你有什么影响**：AMD gfx950 上跑 Qwen3-Next 会更快；其他硬件无变化。

### 4. [Fix] elementwise: Avoid data-dependent token-count guards in sigmoid-mul launchers
- **代码层面**：`elementwise.py` 里把 `num_tokens < 1024` 这种依赖具体数值的判断，换成 `guard_or_true` / `guard_or_false`，让 torch.compile 在符号化 shape 下也能安全编译。
- **新手概念课堂**：编译器不知道 `num_tokens` 具体是多少（它是个"符号"），直接 `if num_tokens < 1024` 会触发"guard 失败"重新编译。`guard_or_true` 相当于告诉编译器"你就当它成立/不成立"，避免反复编译。
- **对你有什么影响**：使用 torch.compile 的用户，sigmoid-mul 相关路径编译更稳定、更少重编译。

### 5. [Fix] dp-attention: Pad extend host lists out of place
- **代码层面**：`forward_batch_info.py` 里把 `self.xxx_cpu.extend(...)` 改成 `self.xxx_cpu = self.xxx_cpu + [...]`，避免原地修改被预规划 metadata 共享的列表。新增单测验证。
- **新手概念课堂**：你在图书馆借了一本书（共享列表），别人也在看同一本。你直接在书上写笔记（extend），别人看到的就是被你改过的书。正确做法是复印一份再改（重新绑定）。
- **对你有什么影响**：DP attention + DSA 场景下，padding 后 metadata 行数不再错乱，推理结果更可靠。

### 6. [Perf] hicache: Parallel THP-backed host pool allocation (opt-in)
- **代码层面**：`mmap_allocator.py` 新增 `_mmap_private_parallel_prefaulted()`，用 `MAP_PRIVATE|MAP_ANONYMOUS` + `MADV_HUGEPAGE` + `MADV_DONTFORK`，多线程并行 prefault；通过环境变量 `SGLANG_HICACHE_HOST_PARALLEL_ALLOC_THREADS` 开启。
- **新手概念课堂**：分配一大块内存就像给新房子铺地板，传统方式是"一个人一块块铺"（单线程缺页），新方式是"叫 8 个工人分区域同时铺"（多线程 prefault），还用了更大的地砖（大页）。
- **对你有什么影响**：HiCache 用户设置该环境变量后，host KV cache 分配显著加速；不设置则行为不变。

### 7. [Refactor] dsv4: Set the tail source KV layer id in a capture stub
- **代码层面**：`deepseek_v4.py` 给 `eager_on_graph` 加了 `capture_stub` 参数，在 CUDA Graph 捕获阶段只设置 `tail_source_kv_layer_id`，不真正写 KV。
- **新手概念课堂**：录制 CUDA Graph 时，有些操作依赖"运行时才有的数据"，录制阶段没法真跑。capture stub 就是"录制时先占个位，重放时再真正执行"。
- **对你有什么影响**：DeepSeek-V4 的 CUDA Graph 捕获更安全，避免录制阶段读到未计算的数据。

### 8. [Refactor] multimodal: [MM refactor 1/9] Move multimodal enums to a shared module
- **代码层面**：把 `Modality` 和 `MultimodalInputFormat` 两个枚举从 `schedule_batch.py` 挪到新文件 `multimodal/modality.py`，其他文件改为 import。
- **新手概念课堂**：就像把散落在各个房间的"公共工具"集中到工具箱里，谁要用谁去拿，避免循环依赖。
- **对你有什么影响**：纯重构，行为不变；但为后续多模态模块化打下基础。

### 9. [Fix] diffusion/disagg: Recompute automatic topology per role
- **代码层面**：`launch_server.py` 把一堆手动拼 `role_overrides` 的逻辑抽成 `ServerArgs.for_disagg_role()` 方法；自动并行度按每个 role 的 GPU 数重新推导，而不是复用父配置。
- **新手概念课堂**：分离式部署里，Encoder 和 Decoder 用的 GPU 数可能不同。原来"照抄父配置"会导致并行度不匹配，现在"按各自 GPU 数重新算"。
- **对你有什么影响**：Diffusion 分离式部署（disagg）在多 role 不同 GPU 数时能正确启动。

### 10. [Fix] deepseek-v4.1: Match KV capacity to physical layouts
- **代码层面**：`pool_configurator.py` 新增 `_get_low_ratio_kv_padding_bytes()`，计算主 KV 池和 indexer 池的"预留页"开销；`calculate_pool_sizes` 里对齐 page_size 并迭代扣减，直到总占用不超过可用内存。
- **新手概念课堂**：你以为能装 100 件行李（按每件体积算），但行李箱有隔板（页对齐），实际只能装 96 件。这个修复就是把"隔板占的空间"也算进去。
- **对你有什么影响**：DeepSeek-V4.1 长上下文用户不再因容量算错而 OOM 或浪费显存。

### 11. [Perf] async-probe: Fuse asynchronous NaN validation into one kernel
- **代码层面**：`async_probe.py` 新增 Triton kernel `_assert_no_nan_kernel`，一次 launch 检查整个 tensor 的 NaN，替代 `torch.isnan().any()` 多次小 kernel；仅在 CUDA + 浮点 + 连续或 2D 时启用，否则回退。
- **新手概念课堂**：原来查 NaN 要"先标记每个元素、再归约、再判断"三步，现在一个 kernel 搞定，且不触发 CPU 同步。
- **对你有什么影响**：开启 NaN 检查（`SGLANG_ENABLE_NAN_LOGITS_CHECK`）时，性能开销更小。

### 12. [Refactor] diffusion: Write DiT attention break outputs into caller-allocated buffers
- **代码层面**：`attention/layer.py` 给各个 attention 类加 `_new_bcg_output()`，返回预分配的输出 buffer；BCG 路径改为"调用方分配、被调方写入"。
- **新手概念课堂**：就像餐厅不再每次给你新盘子（动态分配），而是你自带餐盘（预分配），服务员直接往里盛菜（写入）。减少分配开销，CUDA Graph 也更友好。
- **对你有什么影响**：Diffusion 模型在 BCG 下显存分配更稳定，减少碎片。

### 13. [Refactor] diffusion: Write MiniMax-H3 and Qwen-Image 2.1 break outputs into caller-allocated buffers
- **代码层面**：`minimax_h3.py` 把 `_minimax_h3_attention_core_bcg` 改成接收 `out=` 参数并 `out.copy_(result)`，并加断言校验 shape/dtype；`_embed` 也改为写入 caller 提供的 buffer。
- **新手概念课堂**：和第 12 条同一思路，但这里多了一层"断言"——如果预分配的形状猜错了，立刻报错而不是静默广播。
- **对你有什么影响**：MiniMax-H3、Qwen-Image 2.1 在 BCG 下更省显存，且错误更早暴露。

### 14. [Fix] [CPU] memory: budget overestimation caused by filtering empty NUMA nodes
- **代码层面**：`common.py` 给 `get_cpu_ids_by_node()` 加了 `@lru_cache(maxsize=1)`，避免重复计算。
- **新手概念课堂**：NUMA 节点查询是"读系统文件 + 解析"，很慢；加缓存相当于"第一次算完记在便利贴上，下次直接看"。
- **对你有什么影响**：CPU 推理启动更快，内存预算计算不再重复开销。

### 15. [Fix] http_server: Make SageMaker /ping reflect server readiness
- **代码层面**：新增 `/ready` 端点；`/ping` 在启动/不健康/关闭时返回 503，暂停生成时仍返回 200；`grpc_bridge.py` 的 `health_check` 统一走 `tokenizer_manager.is_healthy()`。
- **新手概念课堂**：`/ping` 是 SageMaker 的"心跳检测"，以前服务器还没准备好就返回 200，导致流量打进来报错。现在"没准备好就说没准备好"。
- **对你有什么影响**：SageMaker 部署更可靠；本地用户可新增用 `/ready` 判断服务可用性。

### 16. [Fix] diffusion/cli: Preserve explicit argv and registered aliases
- **代码层面**：`server_args.py` 用 argparse 的 `_actions` 自动生成"CLI 别名 → dest"映射，替代硬编码的 `cli_aliases` 字典；`from_cli_args` 新增 `argv` 参数。
- **新手概念课堂**：原来别名是"手写清单"，加新别名容易漏；现在"从 argparse 注册表自动生成"，加别名只需注册一次。
- **对你有什么影响**：Diffusion CLI 的参数别名（如 `--dp`、`--cfg_parallel_size`）更可靠，新增别名不用改两处。

### 17. [Fix] diffusion/realtime: Reserve generation slots after initialization
- **代码层面**：`realtime_video_api.py` 把"占槽位"从 WebSocket 连接时移到"收到初始化消息后"；清理时只有真正启动过生成任务才发 `ReleaseRealtimeSessionReq`。
- **新手概念课堂**：以前"进门就占座"（连接即占槽），现在"点完菜才占座"（初始化后才占）。空闲连接不浪费资源。
- **对你有什么影响**：Realtime 视频服务能容忍更多空闲连接；初始化后 1 秒内拿不到槽位会收到 1008 关闭码。

### 18. [Fix] multimodal: Clip pad spans to a truncated prompt
- **代码层面**：`mm_utils.py` 的 `pad_mm_tokens` 里对每个 offset 做 `end = min(end + 1, len(padded_input_ids))`，只填充仍在 prompt 内的部分。
- **新手概念课堂**：图片占位符 span 是 [2, 7]，但 prompt 被截断到长度 5，原来会越界写入或报错，现在只填 [2, 5)。
- **对你有什么影响**：开启自动截断 + 多模态输入时，不再因 span 越界崩溃，且截断前缀的 radix key 保持一致（利于缓存复用）。

### 19. [Fix] diffusion: Resolve cached named weights without the Hub API
- **代码层面**：`weights/source.py` 新增 `_cached_hub_inventory()`，用 `snapshot_download(local_files_only=True)` 列本地缓存；`resolve_weight` 在指定文件名时优先查缓存，命中则跳过 Hub API。
- **新手概念课堂**：你要找一本书，原来每次都要"打电话问图书馆"（Hub API），现在"先翻自己书架"（本地缓存），有就直接用。
- **对你有什么影响**：离线环境或网络差时，已缓存的权重解析秒级完成，不再超时。

### 20. [Feat] spec: Capture aux hidden states on the target without a speculative draft
- **代码层面**：新增 `--enable-return-aux-hidden-states` 和 `--return-aux-hidden-state-layer-ids`；开启后 `--speculative-algorithm` 只用来指定"捕获约定"（EAGLE3/DFLASH/DSPARK），随后被重置为 `None`，不加载 draft 模型。
- **新手概念课堂**：以前要采集 EAGLE3 训练数据，得先跑一个 draft 模型；现在 target 模型自己就能"顺手"输出中间层隐状态，省掉一整个模型。
- **对你有什么影响**：做投机解码训练数据采集的用户，部署更简单、显存更省；普通推理用户无感（需显式开启）。

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)