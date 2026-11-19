# 🚀 vLLM & SGLang 每日更新 — 2026-11-19

> 自动生成于 2026-11-19 09:27 UTC | AI 解读: ✅ 含代码解读

## vllm
## 💡 今日关键词

- **关键词：异构计算 / 多硬件后端**
  - **一句话解释**：同一套 vLLM 代码要能跑在 NVIDIA GPU、AMD GPU（ROCm）、Intel GPU（XPU）、ARM CPU 上，每个后端都要单独适配和测试。
  - **为什么社区在关注**：今天 20 个 commit 里有 6 个（#62440、#67133、#68541、#48552、#67833、#67364/#67363）都在给 AMD 或 Intel 补能力、补 CI。这说明"只支持 CUDA 就够"的时代结束了，推理框架正在变成跨硬件平台的基础设施。
  - **对新手意味着什么**：如果你会写 Triton/CUDA kernel 并且懂某一种非 NVIDIA 架构（ROCm、XPU、ARM NEON/SVE），现在是非常稀缺的技能组合，很多 PR 都是"给 X 平台补一个 Y kernel"。

- **关键词：量化（Quantization）的工程细节**
  - **一句话解释**：把模型权重从 16 位浮点压成 8 位、4 位甚至 FP4，省显存、提速度，但每种硬件的"数据摆放顺序"都不一样。
  - **为什么社区在关注**：今天有 AWQ→GPTQ 布局转换（#48552）、MXFP4 indexer（#67363）、torchao 重载清理（#68329），说明量化已经不是"加个 `--quantization` 参数"这么简单，而是深入到 kernel 级别的数据布局适配。
  - **对新手意味着什么**：想进这个方向，需要理解"量化格式 = 数据类型 + 打包方式 + 内存布局"三件套，光会调 API 不够。

- **关键词：投机解码 / 推测解码（Speculative Decoding）**
  - **一句话解释**：用小模型"猜"后面几个 token，再用大模型一次性验证，从而加速生成。
  - **为什么社区在关注**：今天有 DSpark 的 packed_modules_mapping 修复（#65647）、接受率估计器的数值稳定性修复（#68164）、Gemma-4 MTP 前缀缓存修复（#65263）——全都是在打磨"投机解码在生产环境中的边角 bug"。
  - **对新手意味着什么**：投机解码是当前 LLM 推理加速最热的方向之一，但它的 bug 往往非常隐蔽（数值溢出、缓存命中丢失），适合喜欢"抠细节"的人。

- **关键词：批处理不变性（Batch Invariance）**
  - **一句话解释**：同一个请求，无论它跟多少其他请求一起跑，输出都应该完全一样。
  - **为什么社区在关注**：#68563 用 `_fixed_order_cumsum` 让 DASH 路由的求和顺序固定，#68164 拒绝非有限牛顿步——都是在对抗"浮点加法不满足结合律"这个根本问题。
  - **对新手意味着什么**：如果你以后做推理服务，客户抱怨"同样的问题两次答案不一样"，很可能就是这个原因。这是一个非常实用的调试方向。

- **关键词：Python 3.14 迁移**
  - **一句话解释**：vLLM 把默认 Python 版本从 3.12 升到 3.14。
  - **为什么社区在关注**：#68457 一次性改了 requirements、CI、pre-commit、macOS workflow 等十几个文件，说明大型项目的版本升级是"牵一发动全身"的工程。
  - **对新手意味着什么**：学会读 CI 配置（`.buildkite/*.yaml`、`.github/workflows/*.yml`）是参与开源项目的必备技能，比读业务代码更能快速理解项目结构。

---

## 🧪 重要知识点 & 动手实验

### 知识点 1：CPU 上的 Top-K / Top-P 采样 kernel（来自 #62440）

- **是什么**：LLM 生成下一个 token 时，要从几万个候选词里按概率挑一个。Top-K 是"只保留概率最高的 K 个"，Top-P 是"保留累计概率达到 P 的最小集合"。这两个操作在 GPU 上很成熟，但 CPU（尤其是 ARM 的 AArch64）上需要专门的 SIMD kernel 才能快。
- **为什么重要**：很多边缘设备、笔记本、ARM 服务器（如 AWS Graviton）跑推理时没有 GPU，采样如果慢，整个生成就慢。而且 ARM NEON/SVE 的向量化写法和 x86 AVX 完全不同，不懂就会写出"能跑但慢 10 倍"的代码。
- **动手试试**：
  ```bash
  # 1. 先看新增的 benchmark 脚本长什么样
  sed -n '1,80p' benchmarks/kernels/cpu/benchmark_cpu_topk_topp.py

  # 2. 在任意有 torch 的机器上跑 PyTorch 参考实现（不需要 ARM）
  python -c "
  import torch
  from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
  logits = torch.randn(4, 32000)
  k = torch.tensor([10, 50, 100, 200])
  p = torch.tensor([0.9, 0.95, 0.8, 0.99])
  out = apply_top_k_top_p_pytorch(logits, k, p, allow_cpu_sync=True)
  print('被保留的位置数量（每行）:', (out > float('-inf')).sum(dim=-1))
  "
  ```
- **预期结果**：你会看到每行保留的 token 数量大致在 K 附近（Top-P 可能进一步裁剪）。理解了这个参考实现，再去看 C++ kernel 就知道它在优化什么了——本质是把"排序 + 累加 + 截断"融合成一次遍历。

### 知识点 2：定点顺序累加与批处理不变性（来自 #68563）

- **是什么**：浮点加法不满足结合律：`(a+b)+c` 可能不等于 `a+(b+c)`。GPU 上不同 batch size 会导致不同的并行归约顺序，于是同一个请求的输出会"抖动"。`_fixed_order_cumsum` 用固定顺序的两两归约（类似树形求和），保证无论 batch 多大，加法顺序都一样。
- **为什么重要**：如果你做的是需要可复现性的场景（评测、A/B 测试、合规审计），输出抖动是致命的。而且这种 bug 极难复现——"单条请求跑是对的，批量跑就错"。
- **动手试试**：
  ```python
  import torch

  def fixed_order_cumsum(x):
      offset = 1
      while offset < x.shape[-1]:
          x = torch.cat((x[..., :offset], x[..., :-offset] + x[..., offset:]), dim=-1)
          offset *= 2
      return x

  torch.manual_seed(0)
  # 构造一个"朴素 cumsum 会因顺序不同而不同"的例子
  x = torch.randn(1, 1024, dtype=torch.float32) * 1e4
  a = x.cumsum(-1)
  b = fixed_order_cumsum(x.clone())
  print("普通 cumsum 最后一项:", a[0, -1].item())
  print("定点 cumsum 最后一项:", b[0, -1].item())
  print("差值:", (a[0, -1] - b[0, -1]).item())

  # 关键验证：把 x 复制成 batch=4，定点版本每行结果应完全一致
  x4 = x.repeat(4, 1)
  r = fixed_order_cumsum(x4)
  print("4 行结果是否完全一致:", torch.equal(r[0], r[1]) and torch.equal(r[1], r[2]))
  ```
- **预期结果**：你会看到两个 cumsum 的最后一项有微小差异（这就是浮点误差），但定点版本在不同 batch 下结果严格一致。这就是"批处理不变性"的核心。

### 知识点 3：量化权重布局转换（来自 #48552）

- **是什么**：AWQ 和 GPTQ 都是 4-bit 量化方法，但它们把 4-bit 数值"打包"进 int32 的顺序不一样（哪个是低半字节、哪个是高半字节）。同一个权重，AWQ 格式直接喂给 GPTQ 的 kernel 会算出错误结果。这个 commit 在 XPU 上加了 `AWQUtils.moe_repack` 做转换。
- **为什么重要**：这是量化落地最典型的坑——"模型加载成功、推理不报错、但输出是乱码"。因为 kernel 只是按约定读 bit，不会检查你的打包顺序对不对。
- **动手试试**：
  ```python
  import torch

  # 模拟：两个 4-bit 值打包进一个 uint8
  # 约定 A（AWQ 风格）：低 4 位是第一个元素
  # 约定 B（GPTQ 风格）：高 4 位是第一个元素
  v0, v1 = 0x3, 0xA   # 两个 4-bit 值

  packed_A = (v1 << 4) | v0   # 低半字节 = v0
  packed_B = (v0 << 4) | v1   # 高半字节 = v0

  print(f"A 格式打包: {packed_A:#04x}, B 格式打包: {packed_B:#04x}")

  # 如果 kernel 期望 A 格式，但拿到 B 格式的数据，解出来就是错的
  def unpack_A(p):
      return p & 0xF, (p >> 4) & 0xF
  def unpack_B(p):
      return (p >> 4) & 0xF, p & 0xF

  print("用 A 解 A:", unpack_A(packed_A))      # (3, 10) 正确
  print("用 A 解 B:", unpack_A(packed_B))      # (10, 3) 顺序反了！
  ```
- **预期结果**：你会直观看到"同样的字节、不同的解读方式、完全不同的数值"。这就是为什么量化模型加载后必须做布局转换，也是为什么每个新硬件后端都要单独验证量化正确性。

### 知识点 4：投机解码的数值稳定性（来自 #68164）

- **是什么**：投机解码会在线估计"草稿 token 被接受的概率"，用一个在线回归（牛顿法）不断更新参数。原代码只检查了 `NaN`，但 `+inf` 能通过 `w + inf >= 0` 这个判断，导致斜率被污染成无穷大，之后所有估计都失效。
- **为什么重要**：这是"防御性编程"的经典案例——`x == x` 只能挡住 NaN，挡不住 inf。在 GPU kernel 里这种 bug 会静默传播，表现为"跑着跑着加速比突然掉到 1.0"。
- **动手试试**：
  ```python
  import math

  def bad_check(step, w):
      # 原代码：只挡 NaN
      return step if (step == step and w + step >= 0.0) else 0.0

  def good_check(step, w):
      # 修复后：挡所有非有限值
      return step if (abs(step) < float("inf") and w + step >= 0.0) else 0.0

  cases = [float("nan"), float("inf"), float("-inf"), 0.5]
  for s in cases:
      print(f"step={s!r:>8}  bad={bad_check(s, 1.0)!r:>8}  good={good_check(s, 1.0)!r}")
  ```
- **预期结果**：`bad_check` 会放行 `inf`，`good_check` 会把它归零。一行代码的差别，决定了估计器会不会被一个异常值永久带偏。

### 知识点 5：前缀缓存与块哈希（来自 #65263）

- **是什么**：vLLM 的前缀缓存（prefix caching）把相同的 prompt 前缀对应的 KV 缓存复用。它靠"块哈希"来识别哪些块是相同的。EAGLE 类投机解码会"丢弃最后一个块"（因为最后一个块的 KV 依赖尚未确定的 token），但 Gemma-4 的 MTP drafter 根本不写 KV，所以不该丢弃——这个 commit 修正了判断逻辑。
- **为什么重要**：前缀缓存命中率直接决定吞吐量。一个错误的"丢块"策略会让命中率从 90% 掉到 0%，而且日志里看不出来，只有 benchmark 才能发现。
- **动手试试**：
  ```python
  # 用 vLLM 自带的测试观察命中率差异（需要一台有 GPU 的机器）
  # 先跑不带投机解码的基线
  python -m vllm.entrypoints.openai.api_server \
      --model google/gemma-2-2b-it \
      --enable-prefix-caching \
      --port 8000 &

  # 发两次相同前缀的请求，看第二次的 cached_tokens
  curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{
    "model": "google/gemma-2-2b-it",
    "prompt": "Explain the theory of relativity in simple terms. " * 20,
    "max_tokens": 1
  }' | python -c "import sys,json; d=json.load(sys.stdin); print('cached:', d['usage'].get('prompt_tokens_details'))"
  ```
- **预期结果**：第二次请求应该显示大量 `cached_tokens`。如果命中率异常低，就要怀疑块哈希策略（比如这个 commit 修的问题）是否把不该丢的块丢了。

---

## 🔥 重要更新

- **#68457 Python 3.14 成为默认版本**：影响所有开发者和 CI，是今天改动面最广的 commit（涉及 requirements、buildkite、GitHub Actions、pre-commit 等十几个文件）。如果你在本地开发 vLLM，需要检查自己的 Python 版本。
- **#64145 可配置的异步调度深度**：新增 `async_scheduling_depth` 参数，允许同时有更多 batch 在飞，理论上能提升吞吐。这是面向生产环境性能调优的重要开关。
- **#67133 / #67364 / #67363 AMD ROCm 三项性能优化**：AITER split-KV kernel、压缩 KV 缓存融合、MXFP4 indexer 单次 launch——说明 AMD 平台正在从"能跑"走向"跑得快"。
- **#68048 CPU offload 的 Engram 原地更新**：修复了 layerwise reload 会破坏 CPU 驻留权重的问题，对使用 DeepSeek-V4.1 + 权重重载的场景是硬 bug 修复。

---

## 📋 逐条解读

### 1. [CPU] Add AArch64 top-k and top-p sampling kernels (#62440)

- **代码层面**：新增了 `benchmarks/kernels/cpu/benchmark_cpu_topk_topp.py`（一个对比 C++ kernel 和 PyTorch sort 实现的基准脚本），并在 ARM CI 脚本 `run-cpu-test-arm.sh` 里加了一行 pytest，把 `test_topk_topp_sampler.py::TestCpuTopkTopp` 纳入 ARM 测试。
- **新手概念课堂**：想象你在一个巨大的水果摊上挑苹果。Top-K 是"我只从最大的 10 个里挑"，Top-P 是"我从最大的开始挑，直到累计重量达到总重量的 90%"。GPU 上可以用几千个线程同时比较，CPU 上只有几个核，所以要靠 SIMD 指令（一次比较 4 个或 8 个数）来加速。AArch64 的 NEON 就是 ARM 的 SIMD。
- **对你有什么影响**：如果你在 ARM 服务器（Graviton、Ampere）或苹果 M 系列上跑 vLLM，采样会更快。普通 CUDA 用户无感知。

### 2. [Bugfix] Make Unlimited-OCR-2 DASH routing batch invariant (#68563)

- **代码层面**：在 `vllm/model_executor/layers/dash_attention.py` 里新增 `_fixed_order_cumsum` 函数，并在 `entmax15` 里根据 `envs.VLLM_BATCH_INVARIANT` 开关决定用定点求和还是普通 `cumsum`。
- **新手概念课堂**：`entmax15` 是一种"稀疏注意力"的归一化方法，需要计算累积和来找阈值。累积和就像"排队报数"，如果队伍分组方式不同（batch 大小不同），每个人报出的累计数会有微小差异。定点求和相当于规定"必须按固定分组方式报数"，这样结果就稳定了。
- **对你有什么影响**：开启 `VLLM_BATCH_INVARIANT` 后，Unlimited-OCR-2 模型在批量推理时输出更稳定，不会因为 batch 大小变化导致识别结果抖动。

### 3. [Bugfix][Spec Decode] Add packed_modules_mapping to DeepSeek-V4 DSpark models (#65647)

- **代码层面**：新增测试文件 `tests/models/test_deepseek_dspark_packed_mapping.py`，并在多个 CI yaml 里注册它。核心是给 DSpark 的投机解码模型补上 `packed_modules_mapping`——一个告诉加载器"哪些权重应该被合并成一个张量加载"的映射表。
- **新手概念课堂**：`packed_modules_mapping` 就像搬家时的"打包清单"：q_proj、k_proj、v_proj 三个抽屉的东西要装进一个箱子（qkv_proj），加载器必须知道这个对应关系，否则会找不到权重或加载错位。
- **对你有什么影响**：使用 DeepSeek-V4 + DSpark 投机解码时，权重加载不再报错或静默错位。

### 4. [Bugfix][Multimodal] Turn off the extensible KV cache for Unlimited-OCR-2 (#68254)

- **代码层面**：在 `vllm/model_executor/models/config.py` 的 `_configure_v2` 里，检测到 Unlimited-OCR-2 时强制把 `enable_extensible_kv_cache` 设为 `False`，并打一条 warning。同时加了两个测试：一个验证启动时日志有提示，一个验证实际生成结果正确。
- **新手概念课堂**：可扩展 KV 缓存（extensible KV cache）是一种让 KV 缓存"动态增长"的优化，但某些模型（比如带 DASH 注意力的 OCR 模型）的缓存布局假设和它冲突。这就像给一辆改装车装了个不兼容的油箱——能装上去，但会漏油。
- **对你有什么影响**：如果你用 Unlimited-OCR-2 并手动开了 `enable_extensible_kv_cache=True`，现在会被自动关掉并提示，避免生成出错。

### 5. [ROCm][Attention] AITER FA: dedicated split-KV kernels for decode and speculative-decode verify (#67133)

- **代码层面**：新增 `benchmarks/kernels/benchmark_rocm_gqa_verify.py`，为 AMD gfx950 上的 GQA（8 组）、head size 64、FP8 KV 缓存的解码和投机验证场景提供专用 split-KV kernel，并用 HIP Graph 捕获来测冷缓存性能。
- **新手概念课堂**：Split-KV 是把一个很长的 KV 序列切成几段，让多个计算单元并行处理，最后合并结果。就像打扫一个长走廊，一个人从头扫到尾很慢，四个人分段扫再汇合就快。投机解码的"验证阶段"要一次性验证多个草稿 token，对 KV 的访问模式和解码阶段不同，所以需要专门的 kernel。
- **对你有什么影响**：在 AMD MI300/MI355 上跑投机解码会更快。NVIDIA 用户无感知。

### 6. [XPU][CI] Enable sampler-flag and logits-processor tests on XPU (#68541)

- **代码层面**：把 `test_gpu_logits_processors.py` 和 `test_gpu_sampler_flags.py` 里的 `torch.cuda.is_available()` 判断改成 `current_platform.is_cuda_alike() or current_platform.is_xpu()`，并把硬编码的 `torch.device("cuda")` 改成 `torch.device(current_platform.device_type)`。
- **新手概念课堂**：这是"平台抽象"的标准做法——不要在业务代码里写死 `cuda`，而是问平台"你是什么设备"。就像写代码不要写死"Windows 路径用反斜杠"，而是用 `os.path.join`。
- **对你有什么影响**：Intel GPU 用户现在能跑这些采样相关测试，说明 XPU 后端的采样路径已经和 CUDA 对齐。

### 7. [XPU][UT] Convert awq-packed MoE qweight to the gptq-equivalent layout on xpu (#48552)

- **代码层面**：在 `vllm/model_executor/layers/fused_moe/oracle/int_wna16.py` 的 `_process_weights_xpu` 里，检测到 `AutoAWQConfig` 时调用 `AWQUtils.moe_repack` 把 AWQ 打包的 MoE 权重转成 GPTQ 等价布局，再喂给 XPU 的 int4 fused MoE kernel。
- **新手概念课堂**：见"知识点 3"。简单说，AWQ 和 GPTQ 都是 4-bit 量化，但"4 个 bit 怎么塞进 32 位整数"的约定不同。XPU 的 kernel 只认一种约定，所以要在加载时转格式。
- **对你有什么影响**：在 Intel GPU 上跑 AWQ 量化的 MoE 模型（如 Mixtral、DeepSeek-MoE）不再输出乱码。

### 8. [Do Not Merge][Build] Point vllm-flash-attn at vllm-project/flash-attention#222 for CI (#68348)

- **代码层面**：只改了一行——`cmake/external_projects/vllm_flash_attn.cmake` 里的 `GIT_TAG` 从一个 commit hash 换到另一个。标题带 `[Do Not Merge]` 说明这是临时给 CI 验证用的。
- **新手概念课堂**：vLLM 的 flash-attention 不是 pip 装的，而是编译时从 GitHub 拉源码。`GIT_TAG` 就是"锁定到哪个版本"。这就像 `package.json` 里的版本号，只不过这里是 git commit。
- **对你有什么影响**：如果你自己从源码编译 vLLM，这个 commit 会改变你拉到的 flash-attention 版本。生产环境建议等正式版。

### 9. [LoRA][Model] Add LoRA support for GLM-5.3-Flash (#56327)

- **代码层面**：新增 `MLAKVBProjWithLoRA` / `MLAKVBProjWithShardedLoRA` 两个 LoRA 层（针对 MLA 的 KV B 投影），在 `tests/lora/test_layers.py` 里加了大量测试，并引入 `FusedMoEWithLoRA` 和 `_Glm5NextMergedColumnParallelLinear` 的支持。
- **新手概念课堂**：LoRA 是"给大模型外挂一个小补丁"，只训练补丁不训练原模型。但不同层的形状不同，补丁要"长"得匹配。MLA（Multi-head Latent Attention）的 KV B 投影是一个特殊形状的矩阵，需要专门的 LoRA 包装类。
- **对你有什么影响**：现在可以在 GLM-5.3-Flash 上挂 LoRA 做微调/多租户服务了。

### 10. [Bugfix][LoRA] Allow non-divisible BLIP-2 encoder budgets (#67271)

- **代码层面**：在 `vllm/model_executor/models/blip2.py` 的 `get_mm_lora_token_counts` 里，把"图像 token 数必须是 query token 数整数倍"的断言改成只在 `mm_kwargs is not None`（真实请求）时检查，`mm_kwargs is None`（仅用于估算显存的 sizing budget）时放行。
- **新手概念课堂**：BLIP-2 的视觉编码器输出固定数量的 query token（比如 32 个），每张图对应 32 个 token。但显存估算时可能给出一个"预算值"（比如 100 个 token），100 不是 32 的整数倍，原代码就崩了。这就像"我预算买 100 块钱的苹果，但苹果 32 块一斤，你不能因为除不尽就报错"。
- **对你有什么影响**：用 LoRA + BLIP-2 多模态模型时，显存估算不再因为除不尽而崩溃。

### 11. [XPU][CI] Expand attention and worker tests in Intel GPU CI (#67833)

- **代码层面**：大幅改写 `.buildkite/intel_jobs/attention_intel.yaml`，把原来零散跑的 7 个 attention 测试扩展成"跑整个 `tests/v1/attention` 目录"（排除 ROCm/CUDA 专属测试），并新增一个可选的 MLA 专项 job。同时清理了 `misc_intel.yaml` 里被注释掉的 executor 依赖。
- **新手概念课堂**：CI 配置就像"考试科目表"。原来 Intel GPU 只考 7 道题，现在考一整章，说明 Intel 后端的成熟度提高了，敢让更多测试跑在上面。
- **对你有什么影响**：Intel GPU 用户能更早发现 attention 相关的回归问题。

### 12. [Core] Configurable async scheduling depth for Model Runner V2 (#64145)

- **代码层面**：新增 `SchedulerConfig.async_scheduling_depth` 参数（默认 2，最小 2），在 `VllmConfig.max_concurrent_batches` 计算里加入这个深度（并考虑 pipeline parallel 的额外 stage）。同时加了大量测试，包括"深度队列下不会调度超过 max_tokens 的步数"。
- **新手概念课堂**：异步调度就像"流水线做饭"——切菜、炒菜、装盘可以同时进行。`async_scheduling_depth` 决定"同时有几道菜在流水线上"。深度越大吞吐越高，但显存和调度复杂度也越高。默认 2 是为了兼容旧行为。
- **对你有什么影响**：生产环境可以通过 `--async-scheduling-depth` 调优吞吐，但要注意显存占用。

### 13. [ROCm][Perf][DSv4.1] Insert the compressed KV cache inside the compression launch (#67364)

- **代码层面**：在 DeepSeek-V4.1 的 compressor 里，把"写压缩后的 KV 到主缓存"这个操作从单独的 kernel launch 合并进压缩 kernel 内部。新增测试 `test_v41_compressor_inserts_main_cache_in_compression_launch` 验证合并后结果正确（包括 NaN/inf 边界值）。
- **新手概念课堂**：每次 kernel launch 都有固定开销（就像每次去银行都要排队取号）。把两个小操作合并成一个大操作，省掉一次排队。这在 GPU 上叫"kernel fusion"。
- **对你有什么影响**：AMD 上跑 DeepSeek-V4.1 长上下文时解码更快。

### 14. [ROCm][Perf][DSv4.1] Build the MXFP4 indexer decode metadata in one launch (#67363)

- **代码层面**：把 MXFP4 indexer 解码阶段需要的多个 metadata（block ends、row shard 等）从多次 kernel launch 合并成一次。测试里新增 `_step` 辅助函数和 `DeepseekV41RocmMxfp4IndexerMetadataBuilder` 的覆盖。
- **新手概念课堂**：和上一条类似，都是 kernel fusion。metadata 就像"索引目录"，原来要分几次去图书馆查目录，现在一次查完。
- **对你有什么影响**：AMD 上 DeepSeek-V4.1 的稀疏注意力解码更快。

### 15. [Frontend] generate: keep sample logprobs as engine rows and render them without objects (#60542)

- **代码层面**：新增 `tests/entrypoints/scale_out/token_in_token_out/test_logprobs_render.py`（381 行），验证"用 FlatLogprobs 引擎行直接渲染 JSON"和"用 pydantic 对象逐条渲染"产生完全相同的 JSON（包括 key 顺序）。核心是把 logprobs 从"每 token 一个 Python 对象"改成"紧凑的 numpy 行"。
- **新手概念课堂**：原来每个 token 的 logprob 都创建一个 Python 对象（`Logprob(token_id, logprob, rank, decoded_token)`），几万个 token 就是几万个对象，内存和 GC 压力大。改成 numpy 数组后，内存连续、序列化快。这就像"把散装的快递一个个寄"改成"打托盘整批发"。
- **对你有什么影响**：请求大量 logprobs 时（比如 `logprobs=5` + 长输出），前端渲染更快、内存更省，但输出 JSON 完全不变。

### 16. [CI/Build] Make Python 3.14 the default for images and CI (#68457)

- **代码层面**：改了 `.buildkite/scripts/check-ray-compatibility.sh`（`--python-version 3.12` → `3.14`）、`.pre-commit-config.yaml` 的 pip-compile 参数、`.github/workflows/macos-smoke-test.yml`、requirements 文件等。同时把原来"Python 3.14 Smoke"的 nightly job 改成"Python 3.12 Smoke"（因为 3.14 变成默认了，需要反向验证 3.12 兼容性）。
- **新手概念课堂**：大型项目的 Python 版本升级不是改一行 `python_requires` 就完事，还要重新编译所有依赖的 lock 文件（`requirements/*.txt`），因为不同 Python 版本解析出的依赖树可能不同。
- **对你有什么影响**：如果你从源码安装 vLLM，现在默认用 Python 3.14。老项目如果锁在 3.12，需要显式指定。

### 17. [Bugfix][Model Loader] Preserve CPU-offloaded Engram storage during layerwise reload (#68048)

- **代码层面**：修改了 sharded_rdt 的权重重载逻辑，让 DeepSeek-V4.1 的 CPU offload Engram 表在 layerwise reload 时"原地更新"而不是重新分配。新增测试 `test_replay_updates_cpu_offloaded_engram_in_place` 验证多次 reload 后 `data_ptr()` 不变。文档 `docs/training/layerwise.md` 也加了说明。
- **新手概念课堂**：Layerwise reload 是"一层一层地换权重"，为了省显存。但如果某个权重在 CPU 上（offload），重新加载时如果重新分配内存，原来指向它的指针就失效了。原地更新就像"换轮胎而不是换车"。
- **对你有什么影响**：用 DeepSeek-V4.1 + CPU offload + 权重重载（比如 RLHF 训练循环）时，不会因为指针失效而崩溃或读到旧数据。

### 18. [Bugfix][Spec Decode] Reject non-finite Newton steps in the acceptance estimator refit (#68164)

- **代码层面**：在 `vllm/v1/worker/gpu/spec_decode/acceptance_estimator.py` 的 Triton kernel 里，把 `step_w == step_w`（只挡 NaN）改成 `tl.abs(step_w) < float("inf")`（挡 NaN 和 ±inf）。对 intercept 的更新也加了类似的有限性检查。新增 3 个测试覆盖 inf/-inf/NaN。
- **新手概念课堂**：见"知识点 4"。牛顿法是用"当前点 + 步长"迭代逼近最优解，如果步长是 inf，下一个点就飞到无穷远，再也回不来。
- **对你有什么影响**：长时间运行投机解码时，接受率估计器不会因为一次数值异常而永久失效。

### 19. [Bugfix][Core] Fix Gemma-4 MTP prefix-cache hit loss when lookahead block hashes are off (#65263)

- **代码层面**：在 `vllm/config/speculative.py` 的 `needs_eagle_block_drop()` 里加上 `or self.use_gemma4_mtp()`，让 Gemma-4 的 MTP drafter 不再触发"丢弃最后一个块"的逻辑。新增测试验证 `use_gemma4_mtp()` 为真时 `needs_eagle_block_drop()` 为假。
- **新手概念课堂**：见"知识点 5"。EAGLE 类 drafter 需要"看下一个 token 的 hidden state"来构建自己的 KV，所以最后一个块的 KV 不可信，要丢掉。但 Gemma-4 的 MTP drafter 只读 target 的 KV、不写自己的 KV，所以不需要丢。
- **对你有什么影响**：用 Gemma-4 + MTP 投机解码 + 前缀缓存时，缓存命中率恢复正常，吞吐提升。

### 20. [Model Loader] Remove the torchao reload decorator (#68329)

- **代码层面**：删除了 `vllm/model_executor/model_loader/reload/torchao_decorator.py` 及 `__init__.py` 里的导出，清理了 `sharded_rdt_engine.py` 里恢复装饰器的逻辑，并把测试 `test_online_quant_config_dict_json` 从"手动调 `get_all_weights` + `load_weights`"改成直接调 `llm.collective_rpc("reload_weights")`。
- **新手概念课堂**：装饰器（decorator）是"给函数套一层壳"。原来为了让 torchao 量化的模型支持重载，给 `load_weights` 套了个壳。现在重载路径已经统一到 `reload_weights` RPC，壳就不需要了。删掉死代码能减少维护负担和 bug 面。
- **对你有什么影响**：用 torchao 在线量化 + 权重重载时，走的是统一路径，行为更可预测。

## sglang
## 💡 今日关键词（放在最前面！）

- **关键词：DFLASH 草稿 KV 环形缓冲（Draft KV Ring）默认关闭**
  - **一句话解释**：投机解码（Speculative Decoding）里，草稿模型（draft）需要缓存自己算过的 KV。DFLASH 用的是一个"环形缓冲"（ring buffer），只保留最近 window + 验证块大小的行，节省显存。今天这个开关从"默认开"改成"默认关"，要用得显式设置环境变量 `SGLANG_ENABLE_DFLASH_DRAFT_KV_RING=1`。
  - **为什么社区在关注**：说明社区在"性能默认值"上越来越保守——一个能省 15% 显存但会增加复杂度的特性，宁可让用户显式打开，也不默认替用户做决定。这反映出推理框架从"追新功能"转向"稳定性优先"的成熟期。
  - **对新手意味着什么**：如果你在学推理优化，要养成"看清默认值"的习惯。很多线上事故不是代码 bug，而是"我以为是默认开的，其实不是"。

- **关键词：MXFP8 / FP8 量化 KV Cache**
  - **一句话解释**：把 KV Cache 从 BF16 压到 8 位浮点（MXFP8 是一种带共享 scale 的 FP8 格式），显存直接减半，能塞更多并发请求。
  - **为什么社区在关注**：连续多个 commit（#51296、#51297、#51302）都在处理 MXFP8 的 scale 存储、per-layer 覆盖、SM90 Marlin 内核，说明"低精度推理"已经从实验走向生产主线。
  - **对新手意味着什么**：量化不再是"调个参数"，而是涉及 kernel 布局、scale 对齐、tensor 内存页的硬核工程。想进这个方向，Triton 和 CUDA 是必修课。

- **关键词：Disaggregation（PD 分离）与状态注册**
  - **一句话解释**：把"预填充（Prefill）"和"解码（Decode）"拆到不同机器上跑，各司其职。分离后 KV Cache、DSA 索引缓存这些"状态"必须正确地在两边注册和传输。
  - **为什么社区在关注**：#41334 修的是混合池里草稿模型的 DSA 索引缓存没注册的问题，这属于"分离架构下的状态一致性"——一个非常容易出隐性 bug 的领域。
  - **对新手意味着什么**：分布式推理不只是"多卡跑"，状态管理才是真正的难点。建议从 KV Cache 生命周期开始理解。

- **关键词：多硬件后端（NPU / SM103 / SM90 Marlin）**
  - **一句话解释**：同一个框架要跑在 NVIDIA Hopper（SM90）、Blackwell（SM103）、华为昇腾 NPU 等不同芯片上，每个芯片都要专门的 kernel。
  - **为什么社区在关注**：今天有 NPU 测试修复、SM103 Beam MoE 内核、SM90 Marlin MoE GEMV 内核，说明 sglang 正在做"全硬件覆盖"。
  - **对新手意味着什么**：异构计算是长期趋势。学会写一次 Triton kernel，理解不同架构的差异（SM 版本、warp 布局），比死磕某一个 CUDA 版本更有前途。

- **关键词：Chat 渲染器（Renderer）与 Tokenizer 解耦**
  - **一句话解释**：模型对话的"模板"（system/user/assistant 怎么拼、工具调用怎么序列化）不再用 Jinja 模板，而是用专门的渲染器代码（inkling renderer、dynamo-renderer）。
  - **为什么社区在关注**：#51301、#50721、#51231 都在推进这件事，说明"对话格式"正在从"模板字符串"变成"可测试的代码模块"，因为 Jinja 模板太脆弱、难调试。
  - **对新手意味着什么**：如果你做 LLM 应用，别只会写 prompt；理解 tokenizer + renderer + parser 三段式管线，才能 debug "为什么我的多轮对话格式错乱"。

---

## 🧪 重要知识点 & 动手实验

### 知识点 1：投机解码中的"草稿 KV 环形缓冲"

- **是什么**：投机解码时，草稿模型（draft model）需要缓存自己的 KV。如果草稿模型只看滑动窗口（sliding window），那它只需要保存最近 `window + 验证块` 行的 KV，用一个环形缓冲循环覆盖即可，不需要完整 KV Cache。
- **为什么重要**：这个缓冲能省约 15% 的目标模型 KV 显存，但会引入"prefill 和 decode 必须同时开启"的耦合（PD 分离时两边都要设同一个环境变量）。今天 #51386 把它从默认开改成默认关，就是因为这个耦合容易让用户踩坑——一边开一边没开，decode 读到空的 ring，结果报错或结果错误。
- **动手试试**：
  ```bash
  # 先看默认值（应该是关闭状态）
  python -c "
  from sglang.srt.arg_groups.fields.spec import Spec
  import inspect
  src = inspect.getsource(Spec)
  print('DFLASH_DRAFT_KV_RING' in src)
  "

  # 启动一个带投机解码的服务，显式开启
  python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --speculative-algorithm EAGLE \
    --speculative-draft-model-path lmsys/sglang-EAGLE-llama3.1-8B-Instruct \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --port 30000
  ```
- **预期结果**：服务能正常启动。如果你在 PD 分离模式下只给 decode 设了 `SGLANG_ENABLE_DFLASH_DRAFT_KV_RING=1` 而 prefill 没设，会看到"bootstrap room id"相关的错误或结果异常——这就是今天这个 commit 想让你避免的坑。

---

### 知识点 2：MXFP8 量化的 scale 存储布局

- **是什么**：MXFP8 把每 32 个元素共享一个 8 位 scale（UE8M0，只有指数没有尾数）。存储时，scale 和量化后的数据要分开放在不同 tensor 里，而且为了 kernel 高效访问，scale 的排列顺序（layout）要和数据对齐。
- **为什么重要**：#51296 给 `store_sf_token_major` 加了一个 Triton kernel，专门把"每个 head 4 个 scale"写进"strided unified token envelope"。如果 scale 布局和 kernel 期待的不一致，量化结果就是错的——而且这种错误不会崩溃，只会让模型输出变得"有点怪"，极难排查。
- **动手试试**：
  ```python
  import torch
  import triton
  import triton.language as tl

  # 最小复现：理解 token-major scale 存储
  @triton.jit
  def store_scale_kernel(src, dst, loc, H: tl.constexpr, BLOCK: tl.constexpr):
      token = tl.program_id(0)
      col = tl.arange(0, BLOCK)
      slot = tl.load(loc + token).to(tl.int64)
      mask = (col < H * 4) & (slot != 0)
      # 每个 head 4 个 scale，从 src 的 [token, head, 4] 搬到 dst 的 [slot, head*4]
      value = tl.load(src + token * H * 4 + col, mask=mask, other=0)
      tl.store(dst + slot * H * 4 + col, value, mask=mask)

  H = 8
  src = torch.randn(4, H, 4, dtype=torch.uint8, device='cuda')
  dst = torch.zeros(16, H * 4, dtype=torch.uint8, device='cuda')
  loc = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device='cuda')
  store_scale_kernel[(4,)](src, dst, loc, H=H, BLOCK=32)
  print(dst[1:5].view(4, H, 4))  # 应该等于 src
  ```
- **预期结果**：`dst[1:5]` 的内容和 `src` 完全一致，说明 scale 被正确"散射"到了统一内存页的槽位里。

---

### 知识点 3：PD 分离下的"健康检查竞态"

- **是什么**：PD 分离时，router 启动时只读一次每个 worker 的 `/server_info` 来判断它是 prefill 还是 decode。但 `/server_info` 在 HTTP server 绑定端口时就能响应，而 `/health` 要等引擎 warmup 完才返回 200。如果 router 在 warmup 期间读，就会把 worker 误判成 `Plain`，然后拒绝所有请求。
- **为什么重要**：#51417 的修复很典型——不是加锁，而是"先等 `/health == 200`，再读 `/server_info`"，把两个竞态窗口都关掉。这种"读之前先确认就绪"的模式在分布式系统里到处都是。
- **动手试试**：
  ```python
  import requests, time

  def wait_until_ready(url, timeout=120):
      start = time.time()
      while time.time() - start < timeout:
          try:
              h = requests.get(f"{url}/health", timeout=5)
              if h.status_code == 200:
                  info = requests.get(f"{url}/server_info", timeout=5).json()
                  print("ready, role =", info.get("role"))
                  return info
          except Exception as e:
              print("not ready:", e)
          time.sleep(2)
      raise RuntimeError("timeout")

  # 启动一个 sglang server 后执行
  # wait_until_ready("http://127.0.0.1:30000")
  ```
- **预期结果**：你会看到先打印几次 "not ready"，然后突然 "ready, role = ..."。如果你把 `/health` 检查删掉只读 `/server_info`，在慢启动的大模型上就会看到 role 是 `Plain` 或 `None`——这就是今天修的 bug。

---

### 知识点 4：Chat 渲染器（Renderer）与 Jinja 模板的取舍

- **是什么**：以前模型对话格式靠 `tokenizer_config.json` 里的 Jinja 模板拼字符串；现在 sglang 开始用"渲染器代码"（inkling renderer、dynamo-renderer）直接生成 token id 序列。
- **为什么重要**：Jinja 模板没法处理"图片占位符 -101"、"工具调用参数重新序列化"这类逻辑，而且模板错误只会表现成"模型答非所问"，极难 debug。渲染器是代码，可以单元测试（今天的 commit 里就有 `test_inkling_tmlv0_golden.json` 这种 golden 文件）。
- **动手试试**：
  ```python
  from sglang.srt.parser.inkling_renderer import render_inkling_messages

  messages = [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"},
  ]
  # 注意：需要装了 tml-renderers 才能跑
  ids = render_inkling_messages(messages, spec="inkling2")
  print(ids)
  ```
- **预期结果**：你会拿到一串 token id（不是字符串！）。和 Jinja 模板输出的字符串对比，会发现渲染器直接跳过了"字符串再 tokenize"这一步，对多模态和工具调用更友好。

---

### 知识点 5：Triton 里用 inline asm 对齐 CUDA fast-math

- **是什么**：#46401 的 `silu_approx_f32` 用 Triton 的 `tl.inline_asm_elementwise` 直接写 PTX 汇编，复刻 CUDA 的 `__expf` 和 `__fdividef` 行为。为什么？因为 Triton 默认的 `tl.exp` 和除法是"精确"的，和 CUDA fast-math 的近似结果有微小差异，会导致 kernel 和参考实现数值对不上。
- **为什么重要**：写高性能 kernel 时，"数值一致性"是硬要求。你不能说"反正差 1e-6 无所谓"——下游可能因为这点差异触发不同的路由决策（MoE 选专家），结果雪崩。用 inline asm 是保证 bit-level 一致的手段。
- **动手试试**：
  ```python
  import torch, triton
  import triton.language as tl

  @triton.jit
  def silu_approx_f32(x):
      exp_neg = tl.inline_asm_elementwise(
          asm="{ mul.ftz.f32 $0, $1, 0fBFB8AA3B; ex2.approx.ftz.f32 $0, $0; }",
          constraints="=f,f", args=[x], dtype=tl.float32, is_pure=True, pack=1,
      )
      return tl.inline_asm_elementwise(
          asm="div.approx.ftz.f32 $0, $1, $2;",
          constraints="=f,f,f", args=[x, 1.0 + exp_neg],
          dtype=tl.float32, is_pure=True, pack=1,
      )

  @triton.jit
  def silu_kernel(x_ptr, out_ptr, N: tl.constexpr):
      pid = tl.program_id(0)
      offs = pid * 128 + tl.arange(0, 128)
      mask = offs < N
      x = tl.load(x_ptr + offs, mask=mask)
      tl.store(out_ptr + offs, silu_approx_f32(x), mask=mask)

  x = torch.randn(1024, device='cuda')
  out = torch.empty_like(x)
  silu_kernel[(8,)](x, out, N=1024)
  ref = torch.nn.functional.silu(x)
  print((out - ref).abs().max())  # 应该是 0 或极小的 ulp 级差异
  ```
- **预期结果**：`max diff` 应该是 0 或 1 个 ulp 级别。如果你换成 `tl.exp` 和 `/`，diff 会稍大——这就是为什么需要 inline asm。

---

## 🔥 重要更新

- **#51386 [Fix] spec/dflash: DFLASH 草稿 KV 环形缓冲默认关闭**：影响所有用 DFLASH 投机解码的用户。默认行为变了，PD 分离部署必须显式在两边设 `SGLANG_ENABLE_DFLASH_DRAFT_KV_RING=1`，否则 decode 读不到 ring 会报错。这是今天最需要"升级前读 changelog"的 commit。

- **#41334 [Fix] disaggregation: 混合池 NextN 草稿 DSA 索引缓存注册**：修了 PD 分离 + EAGLE/NextN + DSA（如 GLM-5.3-Flash）组合下的状态注册 bug。之前的代码只注册了 target 的 DSA 索引缓存，没注册 draft 的，导致分离传输时草稿侧状态丢失。对用新模型跑分离部署的用户很关键。

- **#51296 / #51297 / #51302 [Feat] MXFP8 KV scale 统一内存页 + per-layer FA 覆盖 + SM90 Marlin MoE 内核**：三个 commit 串起来是"低精度推理"的完整链路——scale 存储、attention 覆盖、MoE 内核。对追求极致吞吐的用户是重大利好。

- **#46401 [Feat] NVIDIA beam: SM103 单 token BF16 MoE 内核**：Blackwell（SM103）上新 MoE 内核，用 inline asm 对齐 CUDA fast-math 数值。新硬件用户能直接受益。

---

## 📋 逐条解读

### 1. [Feat] cli: Export Cookbook commands for every config-driven LLM/VLM page (#50106)

- **代码层面**：改的是 `docs/scripts/export_commands.mjs`。以前只导出 `Qwen/qwen3.8-flash-next.jsx` 一个配置的命令，现在遍历 `docs/cookbook/autoregressive` 目录下所有 `.mdx` 页面，凡是 `<Deployment>` 渲染的配置都导出。还加了 `gpuCount` 函数，从 `--tp`/`--pp`/`--dp` 参数推算每副本需要几张 GPU，以及 `UNDETECTED_HARDWARE` 集合（CPU、Mac、TPU、昇腾 NPU、Vera Rubin）跳过自动检测。
- **新手概念课堂**：这就像"文档自动生成器"——以前文档里的启动命令是手写的，容易和代码不同步；现在从配置源文件自动生成，保证文档和代码永远一致。`gpuCount` 就像"根据你点的菜（tp/pp/dp 参数）自动算要几张桌子（GPU）"。
- **对你有什么影响**：文档里的 `sglang serve` 命令会更全、更准。如果你照着文档启动新模型，命令不会再因为"文档没更新"而失败。

---

### 2. [Test] diffusion: exercise action HTTP and WebSocket client contracts (#51360)

- **代码层面**：新增 `test_pi05_action_api.py` 里的测试。用 FastAPI 的 `TestClient` 起一个假服务，`monkeypatch` 掉 `async_scheduler_client.forward`，让它把输入的 observation 原样当 action 返回。然后测两种客户端格式：JSON 和 msgpack（二进制），验证 `action.generation` 响应信封、`__ndarray__` 二进制布局、`shape`/`dtype`/`data` 字段。
- **新手概念课堂**：这叫"契约测试"——不测业务逻辑，只测"客户端和服务端约定好的数据格式"。就像快递单：不管包裹里装什么，单子上的字段（收件人、电话、地址）必须对得上，否则快递员就送错。msgpack 是二进制版 JSON，更省带宽。
- **对你有什么影响**：如果你在写调用 sglang 多模态生成 API 的客户端，这个测试告诉你响应里 `data[0].action.values` 是二维数组、二进制格式用 `__ndarray__` 标记——照着写就不会解析错。

---

### 3. [Fix] disaggregation: Register the NextN draft DSA index cache for hybrid pools (#41334)

- **代码层面**：改 `static_kv_storage.py` 的 `add` 逻辑。原来 `append_dsa_tail` 只处理 target 池的 DSA 索引缓存和 tail；现在在混合池分支里，如果 `draft_token_to_kv_pool` 也是 `DSATokenToKVPool`，就把 draft 的 `get_state_buf_infos()` 和 `get_compress_tail_buf_infos()` 拼在 target 后面一起注册。新增测试文件验证"draft 的索引缓存和 tail 排在 target 后面，在同一个 component 里"。
- **新手概念课堂**：PD 分离时，KV Cache 要像搬家一样从 prefill 机器搬到 decode 机器。搬家前得先列清单（注册），清单上要写清楚每个箱子（buffer）的地址和长度。之前漏列了"草稿模型的索引缓存"这个箱子，导致搬家时它没被搬走，decode 侧用的时候是空的。这个 commit 就是补上清单。
- **对你有什么影响**：用 EAGLE/NextN 投机解码 + DSA 模型（如 GLM-5.3-Flash）+ PD 分离的用户，之前可能遇到"decode 结果异常"或"索引缓存未初始化"的报错，现在修好了。

---

### 4. [Feat] mem_cache: Store MXFP8 KV scales in unified memory pages (#51296)

- **代码层面**：`mxfp8_interleave_sf.py` 新增 `_store_sf_token_major_kernel` 和 `store_sf_token_major`，把"每个 head 4 个 UE8M0 scale"按 token-major 布局写进 strided 的统一内存页。`mxfp8_quant_store_kv.py` 的量化 kernel 加了 `cache_stride`/`sf_stride`/`INTERLEAVED` 参数，并把"slot 0 是 CUDA graph padding sink，跳过写入"的判断从内层提到外层（只对 KV 部分判断，Q 部分不判断）。
- **新手概念课堂**：想象 KV Cache 是一排储物柜，每个柜子放一个 token 的 KV。MXFP8 量化后，每个柜子里除了数据还要放"刻度尺"（scale）。以前刻度尺单独放一个仓库，现在统一放进柜子的"夹层"里——省了一次内存跳转，kernel 读写更快。`slot 0` 是 CUDA graph 的"占位柜"，专门给 padding 用，往里写会破坏 graph，所以要跳过。
- **对你有什么影响**：用 MXFP8 量化 + CUDA graph 的用户，KV Cache 显存利用率更高、kernel 更快。如果你自己写量化 kernel，注意"slot 0 跳过"这个约定。

---

### 5. [Refactor] model_loader: Let models name their safetensors index file (#51299)

- **代码层面**：`loader.py` 的 `Source` dataclass 加了 `safetensors_index_file` 字段（默认 `SAFE_WEIGHTS_INDEX_NAME`），`init_new` 从 model 对象读 `safetensors_index_file` 属性。`_prepare_weights` 加参数 `safetensors_index_file`，允许为 `None`（表示"没有 index 文件，别去找"）。MISTRAL 格式下如果 index_file 是 None 就不覆盖成 `consolidated.safetensors.index.json`。
- **新手概念课堂**：大模型权重通常切成多个 `.safetensors` 文件，配一个 `model.safetensors.index.json` 说明"哪个权重在哪个文件里"。有些特殊模型（比如 Mistral 的 consolidated 格式）索引文件名不一样，或者干脆没有索引文件。以前代码写死了默认名，现在让模型自己声明。
- **对你有什么影响**：加载非标准格式的模型（Mistral consolidated、自定义分片）时不会再因为"找不到 index 文件"而失败。普通用户基本无感，但如果你在微调或转换模型格式，这个改动让 loader 更灵活。

---

### 6. [Feat] attention: Add per-layer FA overrides and block-anchored windows (#51297)

- **代码层面**：`flash_attention.py`、`flash_attention_v4.py`、`cute/interface.py` 都加了 `rel_bias_presheared` 参数（默认 False）。当为 True 时，`rel_extent = rel_bias.shape[-1] - 256`，即认为 rel_bias 的前 256 列是"预剪切"的 padding，不参与计算。`_flash_attn_fwd` 里相应调整 `rel_extent_padded` 和 tile 逻辑。
- **新手概念课堂**：相对位置偏置（relative bias）是 attention 里给"距离近的 token 更高权重"的机制。`presheared` 意思是"已经预剪切过"——就像你买布，店家已经按尺寸裁好了，你就不用再量再剪。这里是为了让 kernel 复用同一块预对齐的 bias 缓存，省一次 reshape/copy。
- **对你有什么影响**：用 FlashAttention + 相对位置偏置的模型（某些长上下文模型）推理更快。普通用户无感，但如果你在调 attention kernel，注意 `rel_bias` 的 shape 约定变了。

---

### 7. [Feat] dspark: Let the draft fold its verify commit and cap step logits (#51298)

- **代码层面**：`mamba_state_scatter_triton.py` 的 `_fused_conv_window_scatter_multi_kernel` 加了 `CONTIGUOUS` 编译期常量——如果源 tensor 的 stride 满足 `stride(4)==1 and stride(3)==dst.shape[3]`，就直接用 `e` 做偏移（连续访问），否则走原来的 `(e//last_axis)*stride + (e%last_axis)*stride` 逻辑。`_fused_commit_track_indices_kernel` 加了 `HAS_ACCEPT_INDEX` 常量，没有 accept_index 时 `last = al - 1`（直接用 accept 长度）。
- **新手概念课堂**：Mamba 模型的"卷积窗口"需要把状态从源 buffer 散射到目标 buffer。如果源 buffer 是连续排列的，就可以像"倒豆子"一样连续倒；如果是跳跃排列的，就得"一颗一颗挑"。加 `CONTIGUOUS` 分支就是检测到连续时走快速路径。`HAS_ACCEPT_INDEX` 类似"有清单就按清单点货，没清单就按数量点货"。
- **对你有什么影响**：用 dspark/Mamba 类模型做投机解码时，verify commit 更快。普通用户感知是"decode 速度略快"。

---

### 8. [Feat] inkling2: Add SM90 Marlin, expert and MTP kernel ops (#51302)

- **代码层面**：新增 `marlin_moe_gemv.cuh`（358 行 CUDA），实现 SM90 上 Marlin 重排后的 NVFP4/MXFP8 专家权重的"小批量 routed MoE GEMM"。核心技巧：Marlin 的 16(k)×64(n) tile 里，一个 lane 的 4 字节正好是 `mma.m16n8k16` 的 A fragment，所以权重可以当 A 操作数、8 行 routed token 当 B 操作数。K 维拆分用 cluster + 分布式共享内存求和。
- **新手概念课堂**：MoE（混合专家）模型每个 token 只激活几个专家，所以是"小批量"矩阵乘。Marlin 是一种把量化权重重排成"kernel 友好"布局的格式。这个 kernel 的妙处是"布局对齐"——权重的存储格式恰好就是 MMA 指令要的格式，省掉了一次转置/重排。就像乐高积木，出厂就是按你要拼的形状切的，直接拼就行。
- **对你有什么影响**：在 Hopper（H100/H200）上跑 NVFP4/MXFP8 量化的 MoE 模型（如 inkling2 系列）时，decode 阶段更快。

---

### 9. [Feat] linker: Place hybrid-SWA sidecar draft pools on the target pages (#51300)

- **代码层面**：`linker_pool_assembler.py` 的 `DevicePoolGroup` 加 `restores_sidecar_drafts` 字段。新增 `_split_sidecar_draft_pools` 函数：把 hybrid-SWA 的 sidecar 草稿池拆成"跟随 target KV 索引的"和"跟随 target SWA 索引的"两类。`_build_mamba_swa_device_pool_group` 里调用它，把拆出来的池子加到 `kv_drafts`/`swa_drafts`。
- **新手概念课堂**：Hybrid-SWA 模型（如某些用滑动窗口注意力的模型）的 KV 池分"全量池"和"滑动窗口池"。草稿模型（sidecar draft）也有自己的池，这些池要么跟着全量索引走，要么跟着窗口索引走。以前没区分，导致草稿池被放错位置，访问越界或读到脏数据。这个 commit 就是"按索引类型分拣行李"。
- **对你有什么影响**：用 hybrid-SWA + 投机解码 + Mamba 的组合时，显存布局更正确，不会因为池子放错而崩溃。

---

### 10. [Feat] inkling2: Render and parse chat through the inkling2 renderer spec (#51301)

- **代码层面**：`chat_encoding.py` 的 `resolve_chat_encoding_spec` 加 `"Inkling2ForConditionalGeneration" in arch` 判断返回 `"inkling2"`；`spec_renders_prompt_ids` 加 `"inkling2"`；`encode_simple_chat` 里 `spec in ("inkling", "inkling2")` 时调 `check_served_tokenizer` 和 `render_inkling_messages(..., spec=spec)`。`chat_encoding_adapters.py` 加 `_build_inkling2`（tokenizer 为 None 时返回 None，否则复用 `_build_inkling`），注册到 `_BUILTIN_ADAPTERS`。`protocol.py` 删了几行（truncated）。
- **新手概念课堂**：这是"给新模型接入对话格式"的标准流程——识别模型架构 → 选渲染器 → 检查 tokenizer 匹配 → 生成 token id。`inkling2` 是 `inkling` 的升级版，复用大部分逻辑但传不同的 `spec` 参数。
- **对你有什么影响**：部署 inkling2 系列模型时，chat 接口能正确工作，不会因为"没有 Jinja 模板"而报错。

---

### 11. [Test] [NPU] multi-node: Fix the PD warmup race and raise the DSV4-Flash 128K prefill memory fraction (#51417)

- **代码层面**：`test_npu_multi_node_utils.py` 的 `wait_for_pd_worker_roles` 现在先 GET `/health`，只有 200 才 GET `/server_info`。`launch_router` 的注释也更新了，说明"只等 `/server_info` 不够，要等 `/health == 200`"。另外把 DSV4-Flash 128K prefill 的 memory fraction 调高了（truncated 部分）。
- **新手概念课堂**：这就是知识点 3 讲的竞态。`/health` 是"引擎 warmup 完了没"，`/server_info` 是"我是 prefill 还是 decode"。router 启动时只读一次 `/server_info`，如果读到 warmup 中的状态就会误判。修复方式是"先确认健康，再问身份"。
- **对你有什么影响**：在昇腾 NPU 多机 PD 分离测试中，不会再因为 warmup 竞态导致 router 把 worker 注册成 `Plain` 然后拒绝请求。如果你自己写部署脚本，记得"先等 health 再查 info"。

---

### 12. [Feat] [NVIDIA] beam: Add the SM103 single-token BF16 MoE kernel (#46401)

- **代码层面**：`numerics.py` 新增 `silu_approx_f32`（用 inline PTX 复刻 CUDA fast-math 的 SiLU）。`moe/__init__.py` 注册 `moe.beam_decode_gemv` kernel（SM103 专用，`min_sm=(10,3), max_sm=(10,3)`）。新增 `beam_decode_gemv.py`（156 行 Triton），实现"单 token Beam MoE + 加权专家输出"，gate/up 和激活保持 BF16 舍入边界，down projection 先乘 FP32 路由系数再舍入到 BF16。
- **新手概念课堂**：Beam search 解码时每个 token 要算多个候选（beam），所以是"单 token 多专家"的小批量 MoE。这个 kernel 的关键是"数值一致性"——用 inline asm 让 SiLU 和 CUDA 参考实现 bit-level 一致，否则 beam 分数排序会乱。
- **对你有什么影响**：在 Blackwell（SM103）上跑 Beam search + MoE 模型时，decode 更快且数值和参考实现一致。

---

### 13. [Perf] qwen4: pin the offloaded PLE table at its exact size (#48249)

- **代码层面**：`cuda_wrapper.py` 加 `cudaHostAlloc`/`cudaFreeHost` 的 ctypes 绑定。`pool_host/common.py` 把 `_alloc_hip_host_tensor` 重构成通用的 `alloc_owned_host_tensor(dims, dtype, alloc, free)`，不再依赖 HIP 专用函数。`_uses_hip_owned_host_memory` 保留。
- **新手概念课堂**：PLE（Positional Language Embedding）表很大，放 GPU 显存放不下，要 offload 到主机内存（pinned memory）。torch 的 `pin_memory` 会把分配大小向上取整到 2 的幂——180GB 的表会占 256GiB！这个 commit 改成"精确大小分配"，用 `cudaHostAlloc` 直接要 180GB。
- **对你有什么影响**：用 qwen4 系列 + PLE offload 的用户，主机内存占用从 256GiB 降到 180GiB，能省出一大块内存给别的进程。

---

### 14. [Chore] sglang-processor: Bump dynamo-renderer to 10.4.3 and dynamo-tokenizers to 4.3.1 (#50721)

- **代码层面**：`rust/Cargo.lock` 里 `dynamo-renderer` 从 7.1.1 升到 10.4.3，`dynamo-parsers` 从 9.2.8 升到 11.0.13，`dynamo-protocols` 从 7.0.0 升到 10.0.2，新增 `dynamo-structural-tag` 0.1.2。依赖里加了 `aho-corasick`、`memchr`、`indexmap`。
- **新手概念课堂**：这是 Rust 生态的"依赖升级"。`dynamo-renderer` 是对话渲染器（知识点 4 讲的），`dynamo-tokenizers` 是 tokenizer。大版本跳跃（7→10）说明 API 有 breaking change，但 sglang 侧已经适配好了。
- **对你有什么影响**：如果你用 Rust 组件或自己编译 sglang 的 Rust 部分，需要重新 `cargo build`。Python 用户无感。

---

### 15. [Test] [NPU] graph: Drop tp1 compile_graph test and run tp2 in base-b PR CI (#50972)

- **代码层面**：删除 `test_npu_compile_graph_tp1_bf16.py`（72 行），修改 `test_npu_compile_graph_tp2_bf16.py`（truncated 部分）。tp1 测试跑 Qwen3-8B + torch.compile + GSM8K 准确率，耗时 1800 秒，现在删掉；tp2 测试挪到 base-b PR CI。
- **新手概念课堂**：CI 测试要平衡"覆盖率"和"耗时"。tp1 的 compile_graph 测试和 tp2 重叠度高，且 tp1 编译耗时 565 秒，不值得在 PR CI 里跑。删掉 tp1、把 tp2 挪到更快的 CI 池，是典型的"测试套件瘦身"。
- **对你有什么影响**：NPU 用户的 PR CI 会更快。如果你在贡献 NPU 相关代码，注意 tp1 compile_graph 不再单独测了。

---

### 16. [Chore] inkling: Bump tml-renderers to 0.4.2 (#51231)

- **代码层面**：`pyproject.toml` 里 `tml-renderers` 从 0.4.1 升到 0.4.2。`inkling_tmlv0_golden.json` 的 `source` 字段同步更新版本号。
- **新手概念课堂**：`tml-renderers` 是 inkling 模型的对话渲染器库。golden 文件是"标准答案"，用来回归测试渲染器输出是否和预期一致。升级版本号时同步更新 golden 的 source

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)