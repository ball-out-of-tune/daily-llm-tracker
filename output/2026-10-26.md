# 🚀 vLLM & SGLang 每日更新 — 2026-10-26

> 自动生成于 2026-10-26 06:55 UTC | AI 解读: ✅ 含代码解读

## vllm
## 💡 今日关键词

- **关键词：投机解码前缀缓存（Spec Decode + Prefix Caching）**
  - **一句话解释**：在 EAGLE 这类"草稿模型"投机解码时，让 KV 缓存能"提前看一眼"草稿 token 的哈希，从而命中前缀缓存。
  - **为什么社区在关注**：投机解码是当前提升 LLM 推理吞吐最有效的手段之一，但它和前缀缓存（Prefix Caching）过去经常"打架"——草稿 token 的 KV 无法复用。这次提交让两者协同，说明社区正在把"性能优化组合拳"打磨到极致。
  - **对新手意味着什么**：如果你在学推理加速，理解"缓存命中率"和"投机接受率"这两个指标如何互相影响，是进阶的关键。可以关注 `lookahead_token_ids` 这种工程化数据结构的用法。

- **关键词：异构计算后端适配（XPU / ROCm / AMD）**
  - **一句话解释**：让 vLLM 在 Intel XPU、AMD ROCm 等非 NVIDIA 平台上也能跑对、跑快。
  - **为什么社区在关注**：今天有 3 个 commit（XPU、ROCm Octave KV、AMD 测试）都在处理异构后端，说明"GPU 不只是 NVIDIA"正在成为现实。AMD MI300/MI355、Intel XPU 都在被正式支持。
  - **对新手意味着什么**：如果你手头没有 N 卡，现在学 vLLM 也有实际落地的硬件路径。同时，理解"平台抽象层"（`current_platform`）是读懂 vLLM 代码的必备技能。

- **关键词：权重缓存与快速启动（Fast Start / Weight Cache）**
  - **一句话解释**：把加载好的模型权重缓存在共享内存/IPC 中，多个引擎进程零拷贝复用，实现"秒级启动"。
  - **为什么社区在关注**：大模型加载动辄几十秒到几分钟，是 RL 训练、多副本服务的痛点。今天两个 commit 都在加固这套机制（缓存 key 加后端维度、检测 daemon 退出），说明它正在从实验走向生产。
  - **对新手意味着什么**：这是一个很好的"系统工程"学习样本——IPC、共享内存、缓存失效策略、进程生命周期管理，都是后端工程师的硬技能。

- **关键词：基准测试工具链完善（vllm bench forward）**
  - **一句话解释**：vLLM 自带的 `bench forward` 命令现在支持投机解码、流水线并行（PP）、上下文并行（PCP）的精确计时。
  - **为什么社区在关注**：性能优化必须有可信的测量工具。今天连续 3 个 commit（#63622、#63616、#63613）都在扩展这个工具，说明社区在系统性地建立"可复现的性能基线"。
  - **对新手意味着什么**：学会用 `vllm bench` 做微基准测试，是理解"为什么这个 kernel 快"的第一步。不要只会跑端到端 benchmark。

- **关键词：元设备初始化（meta device）与内存安全**
  - **一句话解释**：用 PyTorch 的 `meta` 设备"空壳"初始化模型，不真正分配显存/内存，用于形状推导和分布式规划。
  - **为什么社区在关注**：今天有 2 个 bugfix（MOSS-Audio、Engram）都是因为代码在 meta 设备上调用了 `.item()` 或分配了 host 内存而崩溃。说明 meta 初始化路径是容易被忽视的"暗坑"。
  - **对新手意味着什么**：如果你写模型代码，永远要问一句"这段代码在 meta 设备上能跑吗？"——这是一个很好的代码审查 checklist。

---

## 🧪 重要知识点 & 动手实验

### 知识点 1：Meta 设备初始化——为什么不能随便调 `.item()`

- **是什么**：PyTorch 的 `torch.device("meta")` 会创建一个"只有形状、没有数据"的张量。它常用于在真正分配内存前推导模型结构（比如分布式切分规划）。
- **为什么重要**：在 meta 张量上调用 `.item()`、`.numpy()` 或做需要真实数值的运算会直接报错。今天 Commit 5（MOSS-Audio）和 Commit 15（Engram）都是因为这个坑。真实场景中，如果你在写一个支持张量并行的模型，meta 初始化是必经之路，踩坑会导致加载阶段直接崩溃。
- **动手试试**：
  ```python
  import torch

  # 在 meta 设备上创建张量
  with torch.device("meta"):
      x = torch.empty(128, 256)
      print("shape:", x.shape, "device:", x.device)

  # 下面这行会报错：Cannot convert a meta tensor to a Python number
  try:
      print(x.shape[0].item() if hasattr(x.shape[0], 'item') else x.shape[0])
  except Exception as e:
      print("shape 是 int，没问题")

  # 但如果你对 meta 张量本身调 .item()：
  try:
      val = x.sum().item()
  except Exception as e:
      print("报错:", type(e).__name__, e)

  # 正确做法：用纯 Python 整数运算推导形状
  num_mel_bins = 129
  conv_freq = (num_mel_bins + 7) // 8  # 模拟三次 stride=2 卷积
  print("conv_freq =", conv_freq)
  ```
- **预期结果**：你会看到 meta 张量求和后 `.item()` 抛异常，而纯整数算术正常输出 `conv_freq = 17`。这就是 Commit 5 修复的核心——把 `torch.tensor(...).item()` 换成整数除法。

---

### 知识点 2：投机解码 + 前缀缓存如何协同

- **是什么**：投机解码用一个小"草稿模型"先猜几个 token，大模型再验证。前缀缓存则是把已算过的 KV 存起来复用。问题是：草稿 token 还没被验证，它们的 KV 能不能缓存？
- **为什么重要**：如果两者不能协同，投机解码的加速效果会被前缀缓存失效抵消。Commit 1 引入 `lookahead_token_ids`，让 BlockStored 事件携带"前瞻 token"，使得缓存系统能预判草稿 token 的哈希。这是推理加速的组合优化。
- **动手试试**：用 vLLM 的 bench forward 观察投机解码的计时（需要 CUDA GPU）：
  ```bash
  # 安装 vllm 后，用 EAGLE3 草稿模型跑 forward benchmark
  vllm bench forward \
      --model Qwen/Qwen3-1.7B \
      --speculative-config '{"model":"AngelSlim/Qwen3-1.7B_eagle3","method":"eagle3","num_speculative_tokens":3}' \
      --case verify=16*4@4096:3

  # 对比：排除 drafter，只看 target forward
  vllm bench forward \
      --model Qwen/Qwen3-1.7B \
      --speculative-config '{"model":"AngelSlim/Qwen3-1.7B_eagle3","method":"eagle3","num_speculative_tokens":3}' \
      --exclude-drafter \
      --case verify=16*4@4096:3
  ```
- **预期结果**：第一次输出包含 drafter 提案的耗时，第二次只测 target。两者差值就是草稿模型的成本。你会看到 `median_us` 的差异，理解"投机解码不是免费的"。

---

### 知识点 3：权重缓存的 Key 设计——为什么后端也要进 Key

- **是什么**：vLLM 的 Fast Start 把加载好的权重缓存在 IPC 共享内存里。缓存必须有一个"key"来判断能否复用。Commit 6 把 MoE 后端和 linear 后端也加进了 key。
- **为什么重要**：不同 kernel 后端（如 `flashinfer_cutlass` vs `flashinfer_trtllm`）会对权重做不同的**布局变换**（比如重排、量化打包）。如果引擎换了后端却复用了旧缓存，权重布局不匹配，结果就是**静默错误**——不崩溃，但输出全错。这是最可怕的 bug 类型。
- **动手试试**：用 Python 模拟一个缓存 key 的比对逻辑：
  ```python
  from dataclasses import dataclass, field

  @dataclass
  class KernelConfig:
      moe_backend: str = "flashinfer_cutlass"
      linear_backend: str = "cutlass"
      linear_backend_per_quant: dict = field(default_factory=dict)

  @dataclass
  class WeightCacheKey:
      model: str
      dtype: str
      moe_backend: str
      linear_backend: str
      linear_backend_per_quant: tuple

      def mismatched_fields(self, other):
          diffs = []
          for f in ("moe_backend", "linear_backend", "linear_backend_per_quant"):
              if getattr(self, f) != getattr(other, f):
                  diffs.append(f)
          return diffs

  k1 = WeightCacheKey("m", "bf16", "flashinfer_cutlass", "cutlass", ())
  k2 = WeightCacheKey("m", "bf16", "flashinfer_trtllm", "cutlass", ())
  print("mismatch:", k1.mismatched_fields(k2))  # ['moe_backend']
  ```
- **预期结果**：输出 `mismatch: ['moe_backend']`，说明后端变化会被正确识别为缓存失效。这让你理解"缓存 key 必须覆盖所有影响数据布局的维度"。

---

### 知识点 4：平台抽象层——一个模型如何在 XPU / ROCm / CUDA 上分发

- **是什么**：vLLM 用 `current_platform.is_xpu()` / `is_rocm()` / `is_cuda()` 在 import 时决定加载哪个后端实现。Commit 3 给 Kimi-Linear 加了 XPU 分支。
- **为什么重要**：模型代码往往有平台特定的 kernel（NVIDIA 的 fused attention、AMD 的参考实现）。如果分发逻辑写错，会在某个平台上 import 失败或跑错 kernel。今天这个 commit 还修了测试：`KimiLinearForCausalLM` 在 XPU 上应该走 `amd/linear.py` 的参考路径。
- **动手试试**：模拟平台分发逻辑：
  ```python
  class FakePlatform:
      def __init__(self, name): self.name = name
      def is_rocm(self): return self.name == "rocm"
      def is_xpu(self): return self.name == "xpu"
      def is_cuda(self): return self.name == "cuda"

  def pick_impl(platform):
      if platform.is_rocm():
          return "amd.linear.KimiLinearForCausalLM"
      elif platform.is_xpu():
          return "amd.linear.KimiLinearForCausalLM (reference path)"
      else:
          return "nvidia.model.KimiLinearForCausalLM"

  for name in ["cuda", "rocm", "xpu"]:
      print(f"{name:5s} -> {pick_impl(FakePlatform(name))}")
  ```
- **预期结果**：你会看到三行输出，XPU 和 ROCm 都走 AMD 参考路径，CUDA 走 NVIDIA 融合 kernel。这解释了为什么 Commit 3 的测试要改 `not (is_rocm() or is_xpu())`。

---

### 知识点 5：流水线并行的稳态计时——为什么"完成间隔"不等于"延迟"

- **是什么**：PP（Pipeline Parallel）把模型切成多段放在不同 GPU 上，microbatch 像流水线一样穿过各段。Commit 18 让 `bench forward` 测量"稳态下每个 microbatch 从第一段开始到最后一段结束"的时间。
- **为什么重要**：PP 的"填充（fill）"和"排空（drain）"阶段不计入稳态延迟。如果你用端到端时间除以 batch 数，会低估真实延迟。理解"稳态"和"完成间隔"的区别，是做分布式性能分析的基本功。
- **动手试试**：用纯 Python 模拟 PP 流水线的时间线：
  ```python
  # 4 个 microbatch，3 个 stage，每个 stage 耗时 1 单位
  num_micro, num_stage, stage_time = 4, 3, 1.0
  start = {}
  end = {}
  for m in range(num_micro):
      for s in range(num_stage):
          # 第 m 个 microbatch 在第 s 段：必须等前一段完成，且该段空闲
          prev_end = end.get((m, s-1), 0) if s > 0 else 0
          stage_free = max([end.get((mm, s), 0) for mm in range(m)], default=0)
          start[(m, s)] = max(prev_end, stage_free)
          end[(m, s)] = start[(m, s)] + stage_time

  for m in range(num_micro):
      latency = end[(m, num_stage-1)] - start[(m, 0)]
      print(f"microbatch {m}: start={start[(m,0)]:.1f}, end={end[(m,num_stage-1)]:.1f}, latency={latency:.1f}")

  total = max(end.values())
  print(f"总时间={total:.1f}, 完成间隔={total/num_micro:.1f}, 稳态延迟={stage_time*num_stage:.1f}")
  ```
- **预期结果**：你会看到每个 microbatch 的延迟都是 3.0（3 段 × 1 单位），但总时间/完成间隔是 6/4=1.5。这直观展示了"稳态延迟"和"吞吐"的区别，正是 Commit 18 文档强调的点。

---

## 🔥 重要更新

- **Commit 1（EAGLE 前缀缓存 lookahead）**：投机解码与前缀缓存的协同是推理加速的关键组合，这个改动直接影响高并发场景的吞吐，值得所有做推理优化的人关注。
- **Commit 4（ROCm Octave 3-bit KV cache）**：AMD GPU 上原生 3-bit KV 缓存后端，意味着 KV 显存占用可降到约 3/16（相比 FP16），对长上下文服务是重大利好，也标志着 AMD 后端能力在快速补齐。
- **Commit 13（避免 import vllm 时解析平台）**：这是一个"基础设施级"修复——import 时就解析平台会导致插件加载顺序问题、循环依赖。对插件开发者和打包者影响很大。
- **Commit 20（FP8 indexer WK 对跨 load_weights 保留）**：这是一个典型的"静默数据丢失"bug，权重加载器重排序时会把配对的一半丢掉，导致模型输出错误但不报错。对使用 `runai_streamer` 等流式加载器的用户至关重要。

---

## 📋 逐条解读

### 1. [Core][Spec Decode] Lookahead block hashes for EAGLE prefix caching (#61589)

- **代码层面**：在 `kv_events_subscriber.py` 的 `BlockStored` 事件里新增 `lookahead_token_ids: list[list[int]] | None` 字段；测试改为比较"解码后的 msgpack 内容"而非"编码后的字节"，因为新增字段会改变 MessagePack map header 的宽度。
- **新手概念课堂**：想象你在图书馆借书，前缀缓存是"已经整理好的书架"。EAGLE 草稿模型像"提前猜你要借哪几本书"。以前猜的书不能提前上架（因为还没确认），现在系统允许你把"猜的书单"（lookahead_token_ids）也登记进去，等确认后直接上架，省去重复整理。
- **对你有什么影响**：开启 EAGLE 投机解码 + 前缀缓存时，缓存命中率更高，首 token 延迟（TTFT）在多轮对话场景下会改善。普通用户感知为"重复 prompt 响应更快"。

---

### 2. [Perf][DSv4.1] Read GPU Engram tables from peers' HBM instead of DP collectives (#62091)

- **代码层面**：修改 `test_engram_dp_shard.py` 的测试参数化，新增 `tp_size` 维度；测试里用 `_engram_dp_shard_size` 替代 `get_node_count`；新增对 NVLink P2P 可用性的检查（CI 的 L4 没有 P2P 会跳过 GPU peer 读取测试）。
- **新手概念课堂**：Engram 是 DeepSeek-V4.1 里的一种"查表"结构（类似超大 embedding）。以前每个数据并行（DP）rank 要互相广播表数据（collective），现在改成"直接去邻居的显存（HBM）里读"（peer read）。就像以前全班传纸条，现在直接走到同桌桌上看。
- **对你有什么影响**：DSv4.1 在多卡 DP 部署时，Engram 查表的通信开销降低，吞吐提升。需要 NVLink 全互联（P2P）才能生效。

---

### 3. [XPU] Fix Kimi-Linear model dispatch (#63531)

- **代码层面**：`vllm/models/kimi_k3/__init__.py` 新增 `elif current_platform.is_xpu():` 分支，XPU 上 `KimiLinearForCausalLM` 走 `amd/linear.py`（参考 KDA/MLA 实现），而 `KimiK3ForConditionalGeneration` 仍走 nvidia 路径。测试 `test_registry.py` 相应调整 ReplaySSM 的期望值。
- **新手概念课堂**：这就像一个"多语言菜单"——同一个菜（模型），在 NVIDIA 厨房用专用厨具（fused kernel），在 Intel XPU 厨房用通用厨具（参考实现）。分发逻辑就是"看你在哪个厨房，给你对应的菜谱"。
- **对你有什么影响**：Intel XPU 用户现在能正确加载 Kimi-Linear 模型，之前会 import 失败或走错 kernel。

---

### 4. [ROCm] Octave KV (`ROCM_OCTAVE`): native 3-bit KV cache backend for AMD GPUs (#59774)

- **代码层面**：新增 `csrc/attention/octave_attn.cu`、`octave_decode_wmma_rdna3.cu`、`octave_prefill_rdna3.cu` 等 HIP kernel；`CMakeLists.txt` 里只在 `gfx942`（MI300）和 `gfx1100`（RDNA3）架构上编译；新增 `vllm/v1/attention/backends/rocm_octave_attn.py` 后端和 selector 逻辑；Buildkite 加了 MI300 的 GSM8K 精度测试。
- **新手概念课堂**：KV cache 是推理时存的"对话记忆"。FP16 是 16 位，3-bit 就是每个数只占 3 位——相当于把一本厚词典压缩成口袋本。Octave 是 AMD 上的专用压缩格式，用 WMMA（矩阵乘指令）做高效解压。
- **对你有什么影响**：AMD MI300 用户可以把 KV cache 显存压到约 3/16，长上下文（如 128K）能塞进更多并发请求。精度损失需要 GSM8K 测试验证。

---

### 5. [Bugfix] Fix MOSS-Audio meta initialization (#63629)

- **代码层面**：`moss_audio.py` 里把 `self._compute_downsampled_length(torch.tensor(config.num_mel_bins)).item()` 换成整数算术 `(config.num_mel_bins + 7) // 8`，因为 meta 张量不支持 `.item()`。新增测试在 `cpu` 和 `meta` 设备上验证 `stem_proj.weight.shape`。
- **新手概念课堂**：meta 设备是"纸板模型"——只有形状没内容。你问纸板"你有多重？"（`.item()`），它答不出来。正确做法是用尺子量（整数算术）直接算。
- **对你有什么影响**：MOSS-Audio 在 meta 初始化（分布式规划阶段）不再崩溃。普通用户感知不到，但多卡部署或使用 `torch.device("meta")` 的高级用户会受益。

---

### 6. [Fast Start] Key the weight cache on MoE and linear kernel backends (#63506)

- **代码层面**：`WeightCacheKey` 新增 `moe_backend`、`linear_backend`、`linear_backend_per_quant` 字段；`WeightCacheDaemon` 构造时传入 `kernel_config`；新增测试参数化验证不同后端组合会产生正确的 `mismatched_fields`。
- **新手概念课堂**：缓存 key 就像快递柜的取件码。如果两个包裹（不同后端处理的权重）用了同一个取件码，你会拿错包裹。这次把"后端类型"也编进取件码，确保不会拿错。
- **对你有什么影响**：使用 Fast Start（权重缓存）时，切换 MoE/linear 后端不会错误复用旧缓存，避免"模型加载成功但输出乱码"的静默 bug。

---

### 7. [Fast Start] Detect weight cache daemon exit in zero-copy engines (#62734)

- **代码层面**：`ipc_loader.py` 新增 `_watch_daemon()` 方法，用 socket 连接监视 daemon 存活；daemon 侧新增 `_prune_watchers()` 清理已退出的引擎；测试用 `socketpair` 模拟 daemon 退出并验证 `daemon_lost` 事件被设置。
- **新手概念课堂**：daemon 是"共享仓库管理员"，引擎是"借货的顾客"。如果管理员悄悄下班了，顾客还在等货会永远卡住。这次加了"心跳线"——管理员一走，线就断，顾客立刻知道。
- **对你有什么影响**：使用零拷贝权重缓存的引擎，在 daemon 崩溃时能快速失败并报错，而不是挂起。

---

### 8. [Rust Frontend] Select one structured outputs backend for every route (#62456)

- **代码层面**：`rust/src/chat/src/lib.rs` 里根据 `text_processor.structured_outputs_backend()` 决定是否应用 output grammar；非 xgrammar 后端时跳过 grammar 并打一次 warning（用 `Once` 保证只警告一次）；`structural_tag.rs` 移除对 `StructuredOutputBackend` 的直接依赖。
- **新手概念课堂**：结构化输出（JSON schema、tool call）需要一个"语法检查器"。vLLM 支持多种后端（xgrammar、guidance 等），但 output grammar 这种"结构标签"只有 xgrammar 能理解。这就像不同品牌的充电器——你不能拿 A 品牌的线插 B 品牌的头。
- **对你有什么影响**：使用 Rust 前端且选择非 xgrammar 结构化输出后端时，tool call 和组合响应格式**不会被约束**（会打 warning 提醒），而不是静默出错。

---

### 9. [Recipe] DeepSeek-V4.1-Flash: AgentX lanes as tuned presets (#63560)

- **代码层面**：主要是 YAML 配置文件（`recipes/models/MiniMaxAI/MiniMax-M3.yaml` 等）的文案和 label 调整，把 "N clients" 改成纯数字或 "(KV offload)" 标注；description 里 "client count" 改成 "concurrency"。
- **新手概念课堂**：这是"菜谱"文件——告诉用户在不同硬件、不同并发下用什么参数。这次主要是把术语统一（client count → concurrency），避免歧义。
- **对你有什么影响**：查阅 recipe 时术语更清晰。功能上无变化，纯文档/配置整理。

---

### 10. [Frontend] Add --reasoning-effort-map to rewrite request reasoning efforts (#61094)

- **代码层面**：新增 `--reasoning-effort-map` CLI 参数，接受 JSON map（如 `{"medium": "low", "xhigh": "max"}`）；在 `OnlineRenderer` 里应用映射；测试覆盖 CLI 解析、非法 key 拒绝、以及 Chat/Responses/Anthropic 三个 API 路径。
- **新手概念课堂**：不同模型的 chat template 认识的 `reasoning_effort` 值不一样。有的只认 `low/high/max`，你发 `medium` 它会静默 fallback 到 `max`（烧更多 token）。这个参数让你在服务端统一"翻译"客户端的 effort 值。
- **对你有什么影响**：部署模型时可以统一 effort 语义，避免客户端发送不支持的 effort 导致意外的高推理成本。映射到 `none` 还能关闭 thinking。

---

### 11. [Benchmark] Include speculative drafting in bench forward (#63622)

- **代码层面**：`vllm/benchmarks/forward/` 里新增 drafter 支持，默认包含 MTP/EAGLE/DSpark/DFlash 提案计时，`--exclude-drafter` 可排除；测试参数化四种 method；文档更新说明"所有 draft token 都被接受"。
- **新手概念课堂**：以前 bench forward 只测"大模型前向"，现在把"草稿模型猜 token"也算进去。这就像测汽车加速——以前只测发动机，现在把变速箱换挡时间也算上，更接近真实驾驶。
- **对你有什么影响**：做投机解码性能分析时，可以直接用 CLI 得到包含 drafter 的精确耗时，不用自己拼脚本。

---

### 12. [Bugfix] Fix Qwen3-VL reranker crash on missing tie_word_embeddings in text config (#59989)

- **代码层面**：`adapters.py` 里把 `text_config.tie_word_embeddings` 改成 `language_model.config.get_text_config().tie_word_embeddings`，因为某些 Qwen3-VL 的 text config 里没有这个字段，但外层 config 有。
- **新手概念课堂**：`tie_word_embeddings` 表示"输入 embedding 和输出 lm_head 是否共享权重"。有些模型的 config 把它放在外层，有些放在 text_config 里。代码假设了错误的位置，就 AttributeError 崩溃。
- **对你有什么影响**：Qwen3-VL reranker 模型现在能正常加载，之前会崩溃。

---

### 13. [Bugfix][Platform] Keep PIN_MEMORY eager; avoid resolving platform during import vllm (#63601)

- **代码层面**：把 `is_torch_equal` / `is_torch_equal_or_newer` 从 `vllm.utils.torch_utils` 移到新的 `vllm.utils.torch_version`，避免 `torch_utils` 在 import 时触发平台解析；`env_override.py` 相关 import 更新；Buildkite 加了 import 时平台解析的测试依赖。
- **新手概念课堂**：import 一个包时，Python 会执行包里的顶层代码。如果顶层代码去"探测当前是什么 GPU"，就会触发插件加载、CUDA 初始化等副作用。这就像你打开一本书的目录页，结果整本书被打印出来了——太慢且容易出错。
- **对你有什么影响**：`import vllm` 更快、更安全，插件加载顺序问题减少。对插件开发者尤其重要。

---

### 14. [Bugfix][MRV2] Reserve encoder graph memory without decoder graphs (#58243)

- **代码层面**：`profile_cudagraph_memory` 现在支持"只有 encoder graph、没有 decoder graph"的情况（如纯多模态编码器场景）；测试新增 `encoder_only` 参数，验证在 `CUDAGraphMode.NONE` 或只有 encoder 时也能正确测量并 teardown。
- **新手概念课堂**：CUDA graph 是把一串 GPU 操作"录下来"重放，省去启动开销。显存预留需要知道要录多少 graph。以前代码假设"一定有 decoder graph"，遇到纯 encoder 模型（如某些多模态场景）会算错显存。
- **对你有什么影响**：多模态模型（encoder-only 路径）的 CUDA graph 显存预留更准确，避免 OOM 或浪费。

---

### 15. [Bugfix][Multimodal] Respect meta initialization for Engram offload (#63068)

- **代码层面**：`engram.py` 里 `_allocate_weights` 新增判断：如果默认设备是 meta 或 `cpu_offload=False`，直接返回空的 meta 张量，不分配 host 内存；`cpu_offload` 的 dummy weight 设置也加了 `not self.weight.is_meta` 保护。
- **新手概念课堂**：Engram 的 CPU offload 会分配巨大的 host 内存表。但在 meta 初始化阶段（只是规划形状），不应该真的分配——否则会浪费几十 GB 内存甚至 OOM。这就像建筑蓝图阶段不应该真的去搬砖。
- **对你有什么影响**：使用 Engram offload 的模型在分布式初始化阶段不再错误分配 host 内存，启动更快、更省内存。

---

### 16. [Bugfix][Frontend] Preserve DP rank in Responses requests (#62450)

- **代码层面**：`serving_responses.py` 的 `_create_responses` 现在从 `x-data-parallel-rank` header 解析 DP rank 并传给 `engine_client.generate(data_parallel_rank=...)`；测试参数化 header 缺失/非法/合法值，覆盖 stream 和非 stream。
- **新手概念课堂**：数据并行（DP）有多个引擎副本，请求需要路由到特定 rank（比如为了 session 亲和性）。Chat Completions 路径一直支持这个 header，但 Responses API 之前丢了它——请求会被随机路由。
- **对你有什么影响**：使用 Responses API 且有 DP 部署时，可以通过 header 指定 rank，实现会话粘性。之前这个 header 被静默忽略。

---

### 17. [Model] Add Muse Spark Open support (#61879)

- **代码层面**：新增 `vllm/models/muse_spark/` 目录（含 nvidia 的 CuTe-DSL qk_norm kernel）；新增 kernel benchmark `benchmark_muse_spark_qknorm_rope_kv_insert.py`；修改 `test_comm_ops.py` 里 MNNVL 的显存期望值（新增 Compute Capability 100 的用例）。
- **新手概念课堂**：Muse Spark 是一个新模型架构，特点是 QK-Norm + RoPE + KV insert 融合。CuTe-DSL 是 NVIDIA 的 kernel 编写 DSL（类似 Triton 但更底层）。新增模型支持意味着 vLLM 的模型 zoo 又扩大了。
- **对你有什么影响**：可以直接用 vLLM 部署 Muse Spark 模型，享受融合 kernel 的性能。

---

### 18. [Benchmark] Add steady-state PP support to bench forward (#63616)

- **代码层面**：`vllm/benchmarks/forward/` 新增 PP 支持，streaming 相同 microbatch 穿过已捕获的 stage；`median_us` 测量每个 microbatch 从首 stage 开始到尾 stage 结束（含 PP 传输），排除 fill/drain；JSON 输出含 stage 时间线和时钟同步不确定性；Buildkite 加了 PP 测试。
- **新手概念课堂**：PP 就像工厂流水线，4 个工人（stage）依次加工。稳态下每个产品从第一道工序到最后一道的时间，才是真实"单个产品延迟"。填线和排空阶段（第一批和最后一批）不算。
- **对你有什么影响**：可以用 `vllm bench forward --pipeline-parallel-size N` 精确测量 PP 稳态延迟，指导并行策略选择。

---

### 19. [Benchmark] Add PCP support to vllm bench forward (#63613)

- **代码层面**：`vllm/benchmarks/forward/` 支持 `--prefill-context-parallel-size`；case 描述的是 DP rank 的全局 batch，PCP 切分后每个 rank 的 JSON 含 `num_local_tokens`、`num_tokens_after_padding`、`batch_order`（一个请求可能出现两次）；延迟取所有 rank 的最大值；新增 `test_pcp_query_tokens` 测试 query token 切片正确性。
- **新手概念课堂**：PCP（Prefill Context Parallel）是把长 prefill 的序列切到多卡上并行算。就像把一本长篇小说分给几个人同时读，每人读一段。但"读"的时候要注意边界（query token 的切片）。
- **对你有什么影响**：MLA 模型做长上下文 prefill 时，可以用 bench forward 精确测量 PCP 的收益。

---

### 20. [Bugfix] Preserve FP8 indexer WK pairs across load_weights calls (#59223)

- **代码层面**：`deepseek_v32/nvidia/model.py` 和 `mtp.py` 里，FP8 indexer 的 `wk.weight` 和 `weight_scale_inv` 需要配对融合成 `wk_weights_proj`。但 `AutoWeightsLoader` 会在 `model` ↔ 非 `model` 前缀切换时切分 checkpoint 流，导致两次 `load_weights()` 调用。修复让 pending pair buffer 跨调用保留。新增专门测试文件模拟 `runai_streamer` 的乱序加载。
- **新手概念课堂**：FP8 权重需要配套的 scale（缩放因子）才能正确反量化。`wk` 和 `scale_inv` 是一对，必须一起处理。如果加载器把这对拆到两次调用里，第一次只拿到一半，另一半被丢弃——模型不报错，但 WK 那部分权重是随机初始化的，输出质量下降。
- **对你有什么影响**：使用 `runai_streamer` 等流式/乱序权重加载器部署 DeepSeek-V3.2 时，FP8 indexer 权重不再静默丢失。这是"不崩溃但结果错"的高危 bug 修复。

## sglang
## 💡 今日关键词

- **关键词：Breakable CUDA Graph / 可打断的计算图（BCG）**
  - 一句话通俗解释：把一大段 GPU 计算图切成能中途"暂停"的小段，在暂停点插入需要 CPU 参与的动态逻辑（比如分配 buffer、读回数据），再继续录制/回放。
  - 为什么社区在关注：Commit 1、6、19 都在围绕"capture 时 buffer 谁分配、谁持有"打转，说明 SGLang 正在把 DeepSeek V4 这类超大模型的 prefill 图优化到极致——连一次隐式内存分配都要消灭。
  - 对新手意味着什么：CUDA Graph 是推理性能优化的必修课。如果你只会写 eager 模式，看到 `assert table is not None or _current_capture_var.get() is None` 这种代码会一脸懵。建议从理解 "capture 期间不能有动态分配" 这条铁律开始学。

- **关键词：异构后端适配（NPU / XPU / ROCm）**
  - 一句话通俗解释：同一套 SGLang 代码要跑在 NVIDIA、AMD、Intel、华为昇腾等不同芯片上，需要针对每个后端做条件分支或专用 kernel。
  - 为什么社区在关注：Commit 3、9、16、17、18、19、20 全部涉及非 NVIDIA 后端。昇腾 NPU 的 commit 尤其密集（llguidance bitmask、DeepEP MXFP8、D2H-free gather），说明国产芯片在 LLM 推理生态里的投入在加速。
  - 对新手意味着什么：如果你在学 AI 基础设施，只懂 CUDA 已经不够了。了解 `is_xpu()`、`is_npu_arch35()` 这类硬件探测模式，以及"用空 tuple `()` 做 isinstance 短路"这种小技巧，是跨平台开发的日常。

- **关键词：量化（MXFP4 / MXFP8 / Block-FP8）**
  - 一句话通俗解释：把模型权重和激活值从 16 位浮点压到 4/8 位，省显存、提吞吐，但可能损失精度。
  - 为什么社区在关注：Commit 15、16、17 分别涉及"加载时在线量化 shared expert"、"FP4 专家走 MXFP8 dispatch"、"block-FP8 是否 requant 成 MXFP8 的开关"。量化已经从"离线转换"进化到"加载时动态决策 + 精度/性能权衡开关"。
  - 对新手意味着什么：量化不再是"跑个 GPTQ 脚本"这么简单，而是要理解 block scale 布局、UE8M0 与 fp32 scale 的区别、以及为什么 requant 是"有损"的。这是模型部署工程师的核心技能。

- **关键词：PD 分离（Prefill-Decode Disaggregation）与 WAR barrier**
  - 一句话通俗解释：把"读 prompt 算 KV"（prefill）和"逐 token 生成"（decode）拆到不同机器上，中间靠 KV 传输。WAR（Write-After-Read）barrier 是防止新写入覆盖还在被读的数据的同步点。
  - 为什么社区在关注：Commit 7 把粗粒度 barrier 推迟到"上一个结果处理完之后"，这是典型的"用调度换吞吐"优化，说明 PD 分离场景的流水线重叠已经做到很细的颗粒度。
  - 对新手意味着什么：PD 分离是当前大规模推理服务的主流架构。理解 `event_loop_overlap_disagg_prefill` 这种重叠事件循环，比背 API 有用得多。

- **关键词：CPU 镜像消除（Mirrorless / D2H-free）**
  - 一句话通俗解释：以前 GPU 上的 seq_lens 等元数据要复制一份到 CPU 方便调度器读，现在尽量不复制，减少 Device-to-Host 同步开销。
  - 为什么社区在关注：Commit 19 的标题直接写了 "D2H-free block-table gather"，把 `needs_cpu_seq_lens` 的静态开关改成了"看 mirror 在不在"的数据驱动分支。这是推理引擎消除同步点的典型手法。
  - 对新手意味着什么：D2H 拷贝和 `.cpu()` 调用是隐藏的性能杀手。学会用 `if forward_batch.seq_lens_cpu is not None` 这种"有就用、没有就走另一条路"的写法，是写高性能推理代码的基本功。

---

## 🧪 重要知识点 & 动手实验

- **知识点 1：CUDA Graph capture 期间为什么不能动态分配内存**
  - **是什么**：CUDA Graph 把一串 kernel launch 录制成一张图，回放时一次性提交，省掉 CPU launch 开销。但录制期间如果代码走了"第一次运行才分配 buffer"的分支，回放时这个分配不会重演，就会读到野指针或空 buffer。
  - **为什么重要**：Commit 1 和 6 的核心就是"warmup 阶段先分配好 caller 持有的 buffer，capture 阶段只写不分配"。不懂这条，你写的图优化代码会在第二次回放时静默出错（或 assert 失败）。
  - **动手试试**：
    ```python
    import torch

    # 模拟：capture 时第一次调用会分配 buffer
    class Bad:
        def __init__(self): self.buf = None
        def run(self, x):
            if self.buf is None:
                self.buf = torch.empty_like(x)   # capture 期间分配 → 危险
            self.buf.copy_(x * 2)
            return self.buf

    m = Bad()
    x = torch.randn(4, device="cuda")
    g = torch.cuda.CUDAGraph()
    # 先 warmup（在 capture 外分配）
    m.run(x)
    with torch.cuda.graph(g):
        y = m.run(x)
    x.copy_(torch.arange(4, device="cuda", dtype=torch.float32))
    g.replay()
    print(y)  # 观察是否随输入更新
    ```
  - **预期结果**：因为 warmup 时 `self.buf` 已分配，capture 走的是 `copy_` 分支，回放正常。如果你把 warmup 那行删掉，capture 期间会分配，回放结果可能不对或报错。理解这个差异，你就懂了 Commit 1 里那句 `assert ... _current_capture_var.get() is None` 的用意。

- **知识点 2：用空 tuple 做 isinstance 短路，跳过不可用后端的 import**
  - **是什么**：Commit 9 里 `CustomAllReduceV2 = ()`，因为 `isinstance(x, ())` 永远返回 False，所以后续 `isinstance(obj, CustomAllReduceV2)` 检查自然不成立，从而避免 import 一个 XPU 上不存在的模块。
  - **为什么重要**：跨平台代码里，顶层 import 一个只有特定后端才有的模块，会让整个包在别的平台上 import 失败。用空 tuple 是零成本、无需 try/except 的惯用法。
  - **动手试试**：
    ```python
    # 模拟 XPU 环境
    IS_XPU = True
    if IS_XPU:
        CustomAllReduceV2 = ()
    else:
        class CustomAllReduceV2: pass

    class Foo: pass
    print(isinstance(Foo(), CustomAllReduceV2))   # False，不报错
    print(isinstance((), CustomAllReduceV2))      # False
    ```
  - **预期结果**：两次都打印 `False`，不会抛 `TypeError`。对比一下如果写成 `CustomAllReduceV2 = None` 再 `isinstance(x, None)` 会怎样——会抛 `TypeError: isinstance() arg 2 must be a type`。这就是为什么用 `()` 而不是 `None`。

- **知识点 3：量化中的 block scale 与 UE8M0 vs fp32 scale**
  - **是什么**：Block-FP8 把权重按 128×128 分块，每块存一个 fp32 scale；MXFP8 按 1×32 分块，scale 是 UE8M0（只有 8 位指数、无尾数，即只能是 2 的幂）。从 fp32 scale 转 UE8M0 是有损的（要舍入到最近的 2 的幂）。
  - **为什么重要**：Commit 17 加了一个环境变量 `SGLANG_NPU_ARCH35_REQUANT_BLOCK_FP8`，默认关闭，因为"lossy"。如果你盲目开启，精度可能掉；但不开，就走 Triton 模拟路径，性能差。这是典型的"精度 vs 性能"开关。
  - **动手试试**：
    ```python
    import torch
    # 模拟 fp32 block scale → UE8M0 的舍入
    scales = torch.tensor([1.0, 1.3, 2.0, 3.7, 0.6])
    # UE8M0 只能表示 2^k，用 log2 取整模拟
    ue8m0 = torch.pow(2.0, torch.round(torch.log2(scales)))
    print("原始:", scales)
    print("UE8M0:", ue8m0)
    print("相对误差:", ((ue8m0 - scales) / scales).abs())
    ```
  - **预期结果**：你会看到 1.3 被舍入到 1.0 或 2.0（取决于 round 行为），误差可达 20%+。这直观解释了为什么 requant 要"validate accuracy before using it"。

- **知识点 4：PD 分离中的 WAR barrier 与流水线重叠**
  - **是什么**：在 prefill 节点，GPU 可能在读某块 KV buffer（Read），同时下一轮想写它（Write）。WAR barrier 保证"写"等"读"完成。但如果 barrier 是粗粒度的（等整个 forward 结束），就会阻塞流水线。
  - **为什么重要**：Commit 7 把粗粒度 barrier 推迟到"上一个 batch result 处理完之后"，因为"result 处理和 inflight 队列不会修改这个 forward 的请求"。这是用对数据依赖的精确分析换取重叠度。
  - **动手试试**（用 Python 线程模拟流水线重叠）：
    ```python
    import threading, time

    read_done = threading.Event()
    log = []

    def gpu_forward(tag):
        log.append(f"{tag}: read start")
        time.sleep(0.1)
        log.append(f"{tag}: read done")
        read_done.set()

    def next_forward(tag):
        read_done.wait()          # WAR barrier：等读完成
        log.append(f"{tag}: write start")
        time.sleep(0.1)
        log.append(f"{tag}: write done")

    # 粗粒度：barrier 在 forward 返回后立即执行
    t1 = threading.Thread(target=gpu_forward, args=("F1",))
    t2 = threading.Thread(target=next_forward, args=("F2",))
    t1.start(); t2.start(); t1.join(); t2.join()
    print("\n".join(log))
    ```
  - **预期结果**：你会看到 "F1: read done" 之后 F2 才开始写。把 `read_done.set()` 挪到 `time.sleep` 之后（模拟"推迟 barrier"），观察 F2 是否能更早启动。这就理解了 Commit 7 里 `war_barrier_pending` 标志的作用。

- **知识点 5：HiCache 故障注入测试（随机延迟 + 随机失败）**
  - **是什么**：Commit 11 加了一个装饰器 `HiCacheStorageFailureInjector`，包在存储后端外面，每次调用随机 sleep 或随机返回失败，用来暴露多 rank 下的竞态和分歧。
  - **为什么重要**：分布式缓存系统最怕"某个 rank 的 IO 慢了一点"导致的状态不一致。用故障注入在 CI 里复现，比线上出问题再查便宜一万倍。
  - **动手试试**：
    ```python
    import random, time

    class Injector:
        def __init__(self, backend, delay_limit=0.0, fail_prob=0.0):
            self.backend = backend
            self.delay_limit = delay_limit
            self.fail_prob = fail_prob
        def get(self, key):
            if self.delay_limit:
                time.sleep(random.uniform(0, self.delay_limit))
            if self.fail_prob and random.random() < self.fail_prob:
                return None            # 模拟 cache miss
            return self.backend.get(key)

    class RealBackend:
        def get(self, key): return f"value-{key}"

    inj = Injector(RealBackend(), delay_limit=0.05, fail_prob=0.3)
    results = [inj.get(i) for i in range(10)]
    print(results)
    print("miss 数:", sum(r is None for r in results))
    ```
  - **预期结果**：每次运行 miss 数不同（约 3 个），且有随机延迟。这正是 Commit 11 想制造的不确定性——如果上层代码在这种输入下仍能保证各 rank 一致，就说明逻辑健壮。

---

## 🔥 重要更新

1. **Commit 6 + Commit 1（dsv4 prefill graph buffer 重构）**：把 candidate table 和 replay tail 的输出从"被调用方隐式分配"改成"caller 持有、warmup 预分配"，这是 CUDA Graph 正确性的关键修复，影响 DeepSeek V4 的 prefill 性能与稳定性。
2. **Commit 15（AMD Quark MXFP4 shared expert 在线量化）**：让"未量化的 shared expert"在加载时被量化并融合进 routed experts，直接影响 Qwen3.5/3.8 MXFP4 在 MI355X 上的吞吐，但**有精度风险**，文档明确要求验证。
3. **Commit 19（NPU D2H-free block-table gather）**：把 `needs_cpu_seq_lens` 的静态开关改为数据驱动，为"无 CPU 镜像"的 DSA 图回放铺路，是昇腾后端消除同步点的重要一步。
4. **Commit 7（PD prefill WAR barrier 推迟）**：粗粒度 barrier 延后到结果处理之后，提升 PD 分离 prefill 节点的流水线重叠度，属于"零精度损失"的纯性能优化。

---

## 📋 逐条解读

### 1. [Refactor] dsv4: Write the replay tail's outputs into caller-held prefill graph bridges (#46516)
- **代码层面**：改的是 `deepseek_v4_backend.py` 和 `deepseek_v4_replay_tail.py`。`run_in_prefill_graph` 不再让被调用方返回新分配的 tensor，而是先通过 `self._bridge(bridge_key, block_rows)` 拿到 caller 持有的 bridge（一组预分配 buffer），传给 `bcg_tail_layers_in_prefill_graph` 写入，再从中取 `residual, pre, *aux`。删除 `_bridge_outputs` 方法。加了 assert：capture 期间 bridge 必须已存在。
- **新手概念课堂**：想象你在拍电影（capture），演员（kernel）必须用剧组事先准备好的道具（buffer）。如果拍到一半演员说"我需要一个新道具"，这台摄影机（graph）就录不下去了，因为回放时没有"买道具"这个动作。所以 warmup 阶段先把所有道具买齐，capture 只做表演。
- **对你有什么影响**：普通用户感知不到，但 DeepSeek V4 长上下文 prefill 的稳定性和显存占用会更好，减少了 capture 期间的隐式分配失败风险。

### 2. [Perf] [NVIDIA] kimi-k3: Route draft/dense bf16 GEMMs to TGV and narrow the MLA output gate vector (#46468)
- **代码层面**：`mla_output_gate.cuh` 里 `kVecN` 从 8 改成 4，`n_vecs = N / 4`，即每个线程处理更短的向量（更细的并行粒度）；`cutedsl_bf16_gemm.py` 的 `_pick_tactic` 里给 DSpark draft down/o_proj 和 dense MLP gate_up 加了新的 (M, N) 形状到 TGV（Tensor Core GEMM 变体）的映射表。
- **新手概念课堂**：`kVecN=8` 是"每个线程一次搬 8 个 bf16"（16 字节，正好一个 128-bit load）。改成 4 是"一次搬 8 字节"，可能是为了匹配某些 N 不是 8 的倍数的场景，避免尾部浪费。GEMM tactic 表就像"菜谱"：给定菜名（M,N,K），查表决定用哪种炒法（kernel 配置）。
- **对你有什么影响**：Kimi-K3 在 NVIDIA GPU 上跑 draft 模型和 dense 层会更快，尤其是 MLA 输出门这个小 kernel 的 launch 效率提升。

### 3. [Feat] [NPU] llguidance: add bitmask path for Ascend (#36074)
- **代码层面**：`llguidance_backend.py` 的 `apply_vocab_mask` 里加分支：如果 `logits.device.type == "npu"`，import `sgl_kernel_npu` 并调用 `torch.ops.npu.apply_token_bitmask`，否则走原来的 `apply_token_bitmask_inplace`。新增手动测试文件，用 mock 的 `torch.ops.npu.apply_token_bitmask` 在无昇腾硬件时也能测。
- **新手概念课堂**：约束解码（constrained decoding）像"给模型一张只能选合法词的菜单"，bitmask 就是菜单上打勾/打叉的位图。不同硬件打勾的方式不同，昇腾有自己的 kernel。
- **对你有什么影响**：在昇腾 NPU 上用 llguidance 做 JSON/正则约束生成，不再走通用慢路径，速度更快。

### 4. [Test] [NPU] return-output-ids: Add e2e test for --return-output-ids parameter (#39774)
- **代码层面**：新增 `test_npu_return_output_ids.py`，启动一个 Llama-3.2-1B 的昇腾 server（带 `--return-output-ids`），发 chat completion 请求，断言响应里有 `sglext.output_ids`，且能 decode 回生成文本。
- **新手概念课堂**：`sglext` 是 SGLang 在 OpenAI 兼容响应里塞的"扩展字段"，用来返回标准 API 没有的信息（token id、logprob 等）。e2e 测试就是"真的起一个 server 发 HTTP 请求"，比单元测试更接近用户视角。
- **对你有什么影响**：用 `--return-output-ids` 时，你可以拿到原始 token id 做后处理（比如精确计费、对齐训练数据），现在这个功能在昇腾上有回归保障。

### 5. [Test] [NPU] return-input-ids: Add e2e test for --return-input-ids parameter (#39772)
- **代码层面**：与上一条对称，新增 `test_npu_return_input_ids.py`，验证 `sglext.input_ids` 与 tokenizer 对 prompt 的编码完全一致。
- **新手概念课堂**：`--return-input-ids` 让你知道"服务器实际看到的 prompt token 是什么"，这在排查"为什么模型答非所问"时非常有用——可能是 chat template 加错了。
- **对你有什么影响**：调试 prompt 模板问题时，可以直接对比客户端 tokenize 结果和服务端返回的 input_ids。

### 6. [Refactor] dsv41: Write the candidate table into caller-held prefill graph buffers (#46399)
- **代码层面**：`_prefill_graph_full_topk` 增加 `table` 参数，`_prefill_graph_full_topk_break` 从"返回 table"改成"写入传入的 table"；`forward_low_ratio_sources` 里先 `candidates.prefill_graph_table(...)` 取 buffer，传进去，若为 None（首次运行）则 break 分配后再取一次。删除了 `is_capturing()` 的调用。
- **新手概念课堂**：这是 Commit 1 的"姊妹篇"——candidate table 也是 capture 期间不能新分配的东西。把"谁分配"从被调用方上移到调用方，责任更清晰，也更容易在 warmup 阶段一次性备齐。
- **对你有什么影响**：DeepSeek V4.1 的 prefill 图更稳定，长序列 top-k 索引器在 capture/回放时不会因 buffer 缺失而失败。

### 7. [Perf] pd: Apply the prefill WAR barrier after the previous result is processed (#46220)
- **代码层面**：`prefill.py` 新增 `_war_barrier_is_coarse()` 判断；`event_loop_overlap_disagg_prefill` 里如果 barrier 是粗粒度的，就设 `war_barrier_pending = True` 推迟执行，在"等 batch result"和"处理完 inflight queue"之后各补一次 `_apply_war_barrier()`。
- **新手概念课堂**：WAR barrier 像"图书馆还书后才能借同一本书"。粗粒度 barrier 是"等所有人还完才开放借阅"，推迟执行相当于"先把还书的人排队，等处理完上一批借阅请求再统一开放"，减少等待。
- **对你有什么影响**：PD 分离部署下 prefill 节点的吞吐会提升，尤其是 chunked prefill 场景。

### 8. [Doc] cookbook: Add MiniMax-M3.1-Flash AgentX launches and refresh with nine InferenceX PRs (#46506)
- **代码层面**：新增 `agentx/MiniMaxAI/minimax-m3.1.jsx`（2599 行生成代码），`_agentx.jsx` 的 `LINKER_FLAGS` 加了 `--hicache-storage-backend-extra-config`，MiniMax-M3.1.mdx 加了 "Agentic Long-Context Deployment" 章节。
- **新手概念课堂**：AgentX 是 SemiAnalysis InferenceX 的 agentic 长上下文 benchmark 配置集，cookbook 把这些"实测过的启动命令"搬进文档，用户复制粘贴就能复现。
- **对你有什么影响**：部署 MiniMax-M3.1-Flash 做 coding agent 时，可以直接抄一份经过 benchmark 验证的启动命令。

### 9. [Fix] [Intel] dp-attention: Skip the CustomAllReduceV2 import on XPU (#46417)
- **代码层面**：`dp_attention.py` 把 `CustomAllReduceV2` 的 import 从顶层移到 `if is_xpu(): CustomAllReduceV2 = () else: from ... import CustomAllReduceV2`。
- **新手概念课堂**：见知识点 2。`custom_all_reduce_v2` 依赖 `tvm_ffi`，XPU 上没有这个包，顶层 import 会让整个模块加载失败。
- **对你有什么影响**：Intel XPU 上启用 DP attention 不再因 import 错误启动失败。

### 10. [Perf] [NVIDIA] deepseek-v4.1: Align BF16 prefill indexer score width (#45794)
- **代码层面**：`scoring.py` 的 `prefill_requests` 里，当 `req is None`（CP 场景）时用 `rows_per_request` 构造 req；对 BF16 且 `lc >= 1024` 的 `index_k` 做 64 对齐 padding；`_score_chunks` 的 `rows_per_chunk` 改用 `index_k.shape[0]`，打分后 `[:, :lc]` 切回。
- **新手概念课堂**：Hopper tensor core 的 GEMM 对 K 维度对齐敏感，K 不是 64 的倍数时尾部会浪费。padding 到 64 的倍数让 GEMM 走满速，代价是多算几个 padding 列（但 selection 时切掉）。
- **对你有什么影响**：DeepSeek V4.1 在 Hopper 上做长上下文 prefill 时，indexer 打分更快。

### 11. [Test] HiCache: Inject random delays and failures for testing (#43539)
- **代码层面**：新增两个环境变量 `SGLANG_TEST_HICACHE_BACKEND_RANDOM_DELAY` / `..._FAILURE`，新增 `HiCacheStorageFailureInjector` 装饰器类（用 `threading.local` 保证每线程独立随机序列），在 `_maybe_wrap_hicache_storage_failure_injector` 里按 env 决定是否包裹。
- **新手概念课堂**：见知识点 5。故障注入是分布式系统的"疫苗"——在受控环境里打小剂量病毒，让系统产生免疫力。
- **对你有什么影响**：普通用户无感知；开发者可以用这两个 env 在 CI 里跑"IO 抖动下的正确性"测试。

### 12. [Perf] diffusion: Write Wan's normed q/k straight into the Ulysses exchange (#46247)
- **代码层面**：`rmsnorm_interleaved_rope_cutedsl.py` 的 kernel 增加 `blocked` 模式和 `mDst` 参数，normed+rope 后的结果直接按列块写入 Ulysses 通信的 buffer，省掉一次中间 tensor 和拷贝；`__init__.py` 导出 `fused_rmsnorm_interleaved_rope_into_blocks`。
- **新手概念课堂**：Ulysses 是序列并行的一种，要把 q/k 按序列维切分后 all-to-all。原来流程是"算完写 mY → 再拷到通信 buffer"，现在"直接算完写通信 buffer"，省一次显存往返。
- **对你有什么影响**：Wan 视频生成模型在多卡序列并行下，注意力前的 norm+rope 更快、显存更省。

### 13. [Feat] mem_cache: tag Mooncake offloads with the writing session (#44611)
- **代码层面**：`mooncake_direct_linker.py` 的 `offload` 增加 `session_id` 参数，队列元素多带一个 `session_id`，`batch_set_v2` 调用时构造 `HiCacheStorageExtraInfo(extra_info={"session_id": ...})`；`mooncake_store.py` 增加 `_replicate_config_supports` 做字段兼容性探测。
- **新手概念课堂**：给缓存写入打 session 标签，就像给快递包裹贴"发件人 ID"，后续可以按 session 清理/审计，避免不同会话的 KV 混在一起。
- **对你有什么影响**：用 Mooncake 做 KV offload 时，可以按会话追踪和管理缓存，多租户场景更安全。

### 14. [Feat] diffusion/openai: Report per-step progress for async video jobs (#45919)
- **代码层面**：新增 `video_progress.py` 的 `run_progress_listener`，在 `http_server.py` 的 lifespan 里按 `server_args.progress_endpoint` 启动监听任务；文档补充 `GET /v1/videos/{video_id}` 的 progress 语义（多阶段均分、best effort、进程内存储）。
- **新手概念课堂**：异步视频生成像"点外卖"，`progress` 是"骑手到哪了"。文档明确说了它不是时间估计，而是去噪步数进度，且多阶段（含 refine）平分区间。
- **对你有什么影响**：做视频生成前端时，可以轮询 `progress` 显示进度条，但要注意"重启后 job 元数据丢失"和"要轮询到接受 job 的那台 server"。

### 15. [Perf] [AMD] quark/Qwen3.8: Quantize the BF16 shared expert of Quark MXFP4 checkpoints to MXFP4 while loading for shared-expert fusion (#41899)
- **代码层面**：`fused_moe_triton/layer.py` 新增 `_maybe_load_bf16_shared_expert_as_fp4`，在 expert_id 映射到 local slot 之后、TP 分片之前，把未量化的 shared expert 整体量化再分片（保证与离线量化结果一致）；`exec_.py` 和文档更新 `--enforce-shared-experts-fusion` 的语义。
- **新手概念课堂**：见知识点 3。这里的关键细节是"先整体量化再分片"，因为 MX 的 block 沿 K 方向，TP 分片如果先切再量化，block 边界会对不上。
- **对你有什么影响**：MI355X 上跑 Qwen3.5/3.8 MXFP4 时，shared expert 可以融合进 routed experts 的 W4A4 grouped GEMM，吞吐提升；但**输入激活也变 MXFP4**（原来 BF16），务必验证精度。

### 16. [Perf] [NPU] deepep: enable MXFP8 low-latency dispatch for Fp8MoEMethod FP4 experts (#40519)
- **代码层面**：把 `NPUW4A8MXFP4MoEMethod` 里的 dispatch 量化配置提成类属性 `DISPATCHER_QUANT_CONFIG`，`fp8.py` 的 `process_weights_after_loading` 优先读 `layer.w13_kernel.DISPATCHER_QUANT_CONFIG`，没有才回退到 `weight_dtype/use_mxfp8`。
- **新手概念课堂**：DeepEP 的 dispatch 是把 token 发给对应 expert 的通信步骤。低延迟模式把激活量化成 MXFP8 传输（省带宽），普通模式保持 BF16（因为 A5 的 MXFP8 普通 dispatch 只支持单机）。
- **对你有什么影响**：昇腾上 FP4 专家的 MoE 在低延迟 decode 场景通信量减半，decode 更快；多机 prefill 仍走 BF16，不受影响。

### 17. [Fix] [NPU] requantization: gate arch35 block-FP8 -> MXFP8 requantization (#40435)
- **代码层面**：新增 `SGLANG_NPU_ARCH35_REQUANT_BLOCK_FP8`（默认 False）；`fp8.py` 里 arch35 分支增加条件：只有 `scale_fmt == "ue8m0"`（无损重排）或显式开启该 env（有损 requant）才走 MXFP8 路径，否则走通用路径保留原布局。
- **新手概念课堂**：见知识点 3。之前代码无条件下 requant，导致普通 block-FP8 checkpoint 精度悄悄下降；这个 fix 把"有损操作"变成显式 opt-in。
- **对你有什么影响**：昇腾 arch35 上加载普通 block-FP8 模型，默认不再损失精度（走 Triton 模拟，慢一点）；追求性能可开 env，但要先验证 accuracy。

### 18. [Fix] [AMD] dsv4.1: Fix the V4.1 k-norm-rope store build on ROCm 10 (#46236)
- **代码层面**：`main_norm_rope.cuh` 新增 `shfl_xor_in_warp`，ROCm 下用 `__shfl_xor`（不带 `_sync` 和 mask），CUDA 下保持 `__shfl_xor_sync(0xffffffff, ...)`；测试用例从 4096 tokens 改成 4097（奇数，让最后一个 wave64 有一半 lane 空闲）。
- **新手概念课堂**：HIP 的 `__shfl_xor_sync` 如果有 masked lane 已经 return 会 trap（硬件异常）。wave64 是 AMD 的 warp 宽度（64 线程），奇数 token 数会让最后一个 wave 有一半线程提前退出，正好触发这个 bug。测试改成 4097 就是为了覆盖这个边界。
- **对你有什么影响**：ROCm 10 上跑 DeepSeek V4.1 的 k-norm-rope 不再编译失败/运行时 trap。

### 19. [Perf] [NPU] AscendAttnBackend: D2H-free block-table gather for mirrorless DSA graph replay (#45949)
- **代码层面**：`ascend_backend.py` 里 `needs_cpu_seq_lens=False` 时不再 assert 非 hybrid SWA，而是如果 `use_fia` 就警告并保留 mirror；`init_forward_metadata` 里所有 `if self.needs_cpu_seq_lens` 改成 `if forward_batch.seq_lens_cpu is not None`（数据驱动）；mirrorless decode 时把 draft offset 加到 device 上的 `seq_lens` 而不是 CPU mirror。
- **新手概念课堂**：见关键词"CPU 镜像消除"。静态开关（`needs_cpu_seq_lens`）假设"整个进程要么有 mirror 要么没有"，但实际不同 forward_batch 可能不一样。改成"看这个 batch 有没有 mirror"更灵活，也让 mirrorless 路径真正可用。
- **对你有什么影响**：昇腾上 DSA（DeepSeek Sparse Attention）图回放不再需要 D2H 拷贝 seq_lens，decode 延迟降低。

### 20. [Fix] [AMD] mamba: Keep a chunked prefill cut on the mamba chunk grid so DCP can donate checkpoints (#41701)
- **代码层面**：`schedule_policy.py` 新增 `mamba_prefill_chunk_grid()`（只有启用了 mamba extra buffer 且 DCP 开启时才返回 chunk size，否则返回 1）；`PrefillAdder` 新增 `floor_chunked_tokens`，把被截断的 chunk 向下取整到 grid，但"如果取整后为 0 就不取整"（保证有进展）。
- **新手概念课堂**：Mamba 的状态缓存（checkpoint）按固定 chunk 大小组织，DCP（Decode Context Parallel）只有在"缓存前缀正好是整数个 chunk"时才能捐赠 checkpoint 给别的 rank。prefill 截断如果落在 chunk 中间，后续就再也捐不了。所以宁可少 prefill 几个 token，也要让截断点对齐 chunk 边界。
- **对你有什么影响**：AMD 上跑 Mamba 类模型 + DCP + chunked prefill 时，checkpoint 捐赠不再因为截断点不对齐而失效，长上下文 decode 效率提升。

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)