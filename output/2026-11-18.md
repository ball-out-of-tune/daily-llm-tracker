# 🚀 vLLM & SGLang 每日更新 — 2026-11-18

> 自动生成于 2026-11-18 09:03 UTC | AI 解读: ✅ 含代码解读

## vllm
## 💡 今日关键词（放在最前面！）

- **关键词：ROCm / AMD GPU 适配**
  - **一句话通俗解释**：ROCm 是 AMD 的 GPU 计算平台，相当于 AMD 版的 CUDA。今天 20 个 commit 里有 8 个带 `[ROCm]` 标签。
  - **为什么社区在关注**：vLLM 正在系统性地把原本只在 NVIDIA 上跑得好的功能（量化、MoE、KV 卸载、多模态）逐个移植到 AMD MI 系列显卡上，说明 AI 推理的硬件格局正在从"英伟达一家独大"走向异构。
  - **对新手意味着什么**：如果你只会 CUDA，现在补一点 HIP/ROCm 知识（API 几乎一一对应）会让你在求职市场上更稀缺；如果你在学 Triton，好消息是 Triton 代码在两边都能跑。

- **关键词：量化（Quantization）**
  - **一句话通俗解释**：把模型权重从 16 位浮点压成 8 位、4 位甚至更少，让显存占用和带宽需求大幅下降。
  - **为什么社区在关注**：今天有 MXFP8 融合、INT8_PB_WO 上 ROCm、WNA16 Triton kernel 支持 8-bit 等多个量化相关 commit——量化已经是推理加速的第一生产力。
  - **对新手意味着什么**：理解"打包（packing）"和"分组（group-wise）"这两个概念，是读懂现代量化 kernel 的钥匙。

- **关键词：KV Cache 卸载（KV Offload）**
  - **一句话通俗解释**：把注意力机制的 KV 缓存从 GPU 显存挪到 CPU 内存或磁盘上，用"空间换显存"来支持更长上下文。
  - **为什么社区在关注**：今天有 3 个 KV Offload 相关 commit（多 rank CPU 区域、短写错误诊断、私有区域生命周期），说明这块正在快速成熟。
  - **对新手意味着什么**：这是理解"内存层级（GPU→CPU→磁盘）"设计的绝佳案例，也是长上下文推理的核心工程问题。

- **关键词：Triton 与手写 kernel 优化**
  - **一句话通俗解释**：Triton 是 OpenAI 出的 Python 风格 GPU 编程语言，比 CUDA 好写，比 PyTorch 原生算子灵活。
  - **为什么社区在关注**：今天有去掉 `tl.make_block_ptr`、避免 host 同步、融合量化 producer 等多个 kernel 级优化，说明性能优化已经下沉到"每一次 kernel launch、每一次 H2D 拷贝"。
  - **对新手意味着什么**：学会用 `torch.profiler` 找同步点、理解"异步拷贝"和"pinned memory"，比背 API 更有价值。

- **关键词：MoE（混合专家）与并行策略**
  - **一句话通俗解释**：MoE 模型每次只激活一小部分"专家"网络，用大参数量换低成本推理；但专家要分布在多张卡上，通信成了瓶颈。
  - **为什么社区在关注**：今天有 Muse Spark 的 reduce-scatter 融合、gpt-oss 的 MXFP4 专家按块切分、EPLB 的 NPU 设备索引修复——MoE 的并行细节是当前最卷的工程战场。
  - **对新手意味着什么**：理解 all-reduce / reduce-scatter / all-gather 三种集合通信的代价差异，是读懂 MoE 推理代码的前提。

---

## 🧪 重要知识点 & 动手实验

### 知识点 1：进程级显存 vs 设备级显存（Commit 1）

- **是什么**：`torch.accelerator.memory_allocated()` 只统计**当前进程**分配的显存；而 `total - free` 统计的是**整张卡**的已用显存（包括别的进程）。
- **为什么重要**：写显存泄漏测试时，如果用设备级指标，别的进程随便动一下就会让测试误报。这是"测量方法比测量工具更重要"的经典案例。
- **动手试试**：
```python
import torch

# 分配 100MB
x = torch.empty(100 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")

print("进程已分配 (allocated):", torch.cuda.memory_allocated() / 2**20, "MiB")
print("进程已保留 (reserved) :", torch.cuda.memory_reserved() / 2**20, "MiB")
free, total = torch.cuda.mem_get_info()
print("设备级已用          :", (total - free) / 2**20, "MiB")
print("设备总显存          :", total / 2**20, "MiB")
```
- **预期结果**：`allocated` 约等于 100MiB（加上少量开销），`reserved` 可能更大（缓存分配器预留），而设备级已用会包含驱动和别的进程占用，通常明显大于 100MiB。你会直观看到"三个数字为什么不一样"。

---

### 知识点 2：短写（short write）与 `write(2)` 的返回值语义（Commit 4）

- **是什么**：`write()` 系统调用不保证一次写完你要求的字节数，它可能只写一部分就返回（磁盘满、信号中断等）。返回值小于请求长度就叫"短写"。
- **为什么重要**：很多新手代码写 `write(fd, buf, n)` 后不检查返回值，结果磁盘满时数据静默丢失。正确做法是循环写直到写完，或者像这个 commit 一样明确报错。
- **动手试试**（用 `RLIMIT_FSIZE` 人为制造短写）：
```python
import os, resource, tempfile

# 把单文件大小软限制设为 100 字节
resource.setrlimit(resource.RLIMIT_FSIZE, (100, resource.RLIM_INFINITY))

path = tempfile.mktemp()
with open(path, "wb", buffering=0) as f:
    try:
        n = f.write(b"A" * 1000)   # 想写 1000 字节
        print("实际写入:", n)
    except OSError as e:
        print("写入失败:", e)
os.unlink(path)
```
- **预期结果**：你会看到写入字节数小于 1000，或者直接抛出 `OSError`（`File too large`）。这解释了为什么 vLLM 要把这种情况单独报告成"可能是磁盘满了"，而不是笼统的 EIO。

---

### 知识点 3：WNA16 量化里的"打包（packing）"（Commit 19）

- **是什么**：4-bit 权重在内存里不是每个值占 4 bit，而是把 8 个 4-bit 值塞进一个 32-bit 整数里（`int32`），这叫"打包"。8-bit 则是 4 个塞进一个 `int32`。
- **为什么重要**：量化 kernel 的第一步永远是"解包"。如果你不理解打包布局（沿 K 还是沿 N、有没有交错），读量化代码会完全懵。
- **动手试试**：
```python
import torch

def pack_int4_along_n(w, bits=4):
    """[K, N] int32 值 -> [K, N//pack] int32 打包"""
    pack, mask = 32 // bits, (1 << bits) - 1
    K, N = w.shape
    shifts = torch.arange(pack) * bits
    return torch.sum((w.view(K, N // pack, pack) & mask) << shifts, dim=2)

def unpack_int4_along_n(packed, bits=4):
    pack, mask = 32 // bits, (1 << bits) - 1
    K, Np = packed.shape
    shifts = torch.arange(pack) * bits
    return ((packed.unsqueeze(-1) >> shifts) & mask).reshape(K, Np * pack)

w = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
p = pack_int4_along_n(w)
print("打包后:", p.tolist(), "->", hex(p.item()))
print("解包后:", unpack_int4_along_n(p).tolist())
```
- **预期结果**：打包后是一个 32 位整数（十六进制形如 `0x87654321`），解包后恢复成 `[1,2,3,4,5,6,7,8]`。把 `bits` 改成 8，验证 4 个值打包成一个 `int32`。

---

### 知识点 4：避免 host 同步——`async_tensor_h2d` 与 pinned memory（Commit 9）

- **是什么**：`tensor.to(device, non_blocking=True)` 只有在**源张量是 pinned memory（页锁定内存）**时才真正异步；否则 PyTorch 会悄悄退化成同步拷贝，卡住 CPU。
- **为什么重要**：在推理循环里，一次隐式同步就能让 GPU 空转几十微秒。vLLM 的 `async_tensor_h2d` 就是"先搬到 pinned 内存，再异步拷到 GPU"的封装。
- **动手试试**：
```python
import torch, time

src = torch.arange(1024 * 1024)
pinned = src.pin_memory()   # 页锁定

def bench(t, name):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        d = t.to("cuda", non_blocking=True)
    torch.cuda.synchronize()
    print(f"{name}: {(time.perf_counter()-t0)*10:.2f} ms / 100 次")

bench(src, "普通内存")
bench(pinned, "pinned 内存")
```
- **预期结果**：pinned 版本通常明显更快，因为拷贝真正异步了。如果你在循环里马上对 `d` 做 `.item()` 或 `print`，两者都会变慢——那正是"隐式同步"的代价。

---

### 知识点 5：`reduce-scatter` 为什么比 `all-reduce` 省带宽（Commit 8）

- **是什么**：`all-reduce` 让所有卡都拿到完整结果；`reduce-scatter` 只让每张卡拿到结果的一个分片，通信量约为 `all-reduce` 的一半。MoE 里"每卡只负责一部分 token 的输出"正好匹配 `reduce-scatter`。
- **为什么重要**：MoE 推理的瓶颈常在通信。选对集合通信原语，比优化计算 kernel 收益更大。
- **动手试试**（需要 ≥2 张 GPU 或多进程）：
```python
# 用 torchrun 起 2 个进程：torchrun --nproc_per_node=2 test.py
import os, torch, torch.distributed as dist

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
x = torch.ones(4, device="cuda") * (rank + 1)

y = x.clone(); dist.all_reduce(y)
print(f"rank {rank} all_reduce 结果: {y.tolist()}")

z = x.clone(); dist.reduce_scatter_tensor(z, [torch.ones(4 // world, device='cuda') for _ in range(world)])
print(f"rank {rank} reduce_scatter 分片: {z.tolist()}")
dist.destroy_process_group()
```
- **预期结果**：`all_reduce` 后每张卡的 `y` 都是 `[3,3,3,3]`（sum）；`reduce_scatter` 后每张卡只拿到结果的一部分。对比两者耗时，你会理解为什么 MoE 的"最终归约"偏爱 `reduce-scatter`。

---

## 🔥 重要更新

1. **Commit 3（KV Offload 多 rank 私有 CPU 区域）**：这是架构级改动——让每个 rank 在复制的 CPU 布局之外拥有自己的私有区域，直接影响多卡 KV 卸载的正确性和内存效率，是长上下文推理的基础设施。
2. **Commit 19（INT8_PB_WO 上 ROCm）**：让 ModelOpt 的 INT8 权重量化模型能在 AMD 卡上跑起来，补齐了 ROCm 量化生态的关键一环，对 AMD 用户是实打实的新能力。
3. **Commit 10（gpt-oss MXFP4 专家按块切分）**：修的是"按填充后尺寸切分导致权重错位"的隐蔽 bug，同时保留调用方的 `w2_bias`——这类 bug 不修会让模型输出悄悄变差，属于"沉默的正确性"问题。
4. **Commit 11（Direct reload 校验每个参数都被加载）**：在 RL 训练场景里，权重热更新如果漏加载某个参数，模型会静默退化。这个 commit 把"漏加载"变成显式报错，是可靠性上的重要补强。

---

## 📋 逐条解读

### 1. [ROCm][CI] 在 Qwen3-VL 泄漏测试中测量进程级显存

- **代码层面**：改的是 `tests/models/multimodal/generation/test_memory_leak.py`。把 `_gpu_used_bytes()`（用 `total - free` 算设备级已用）换成 `_gpu_memory_bytes()`（返回 `memory_allocated()` 和 `memory_reserved()`）。同时把注意力后端从硬编码的 `TRITON_ATTN` 改成参数化 fixture，ROCm 上额外测试 `ROCM_ATTN` 和 `ROCM_AITER_FA`（仅 gfx9 架构）。
- **新手概念课堂**：想象你在测"自己家用了多少水"，结果看的是"整个小区的水表"。别的邻居一开水龙头，你的测试就失败。进程级显存就像自家水表，只统计本进程的分配。
- **对你有什么影响**：普通用户无感，但如果你在 AMD 卡上跑 Qwen3-VL，CI 现在会覆盖更多注意力后端，回归问题能更早被发现。

---

### 2. [Perf][Frontend] 在线 beam search 中限制已完成假设的历史

- **代码层面**：改的是 `vllm/entrypoints/generate/beam_search/online.py`。核心问题是：beam search 过程中已完成的假设（finished hypothesis）如果一直保留完整历史，内存会随步数线性增长。改动对已完成序列的历史做了有界化处理，并新增测试用 `weakref.WeakValueDictionary` 追踪对象生命周期，断言同时存活的已完成序列不超过 beam width。
- **新手概念课堂**：beam search 像"同时走多条迷宫路线"，每走一步都可能有一条路线到达终点。如果你把每条到达终点的路线都完整抄一份留着，纸会越用越多。正确做法是只留最好的几条。
- **对你有什么影响**：用在线 beam search（比如 `n>1` 的采样）时，长输出场景下内存不再无限涨。

---

### 3. [KV Offload] 每个 rank 在复制布局之外拥有自己的 CPU 区域

- **代码层面**：改动横跨 `tests/v1/kv_offload/` 和 `csrc` 相关逻辑。`OffloadingParallelConfig` 新增 `local_world_size` 字段；`SharedOffloadRegion` 新增 `private=True` 模式——私有区域没有 joiner，映射后立即 unlink 文件名（映射仍有效），避免退出路径泄漏。测试新增 `test_private_region_unlinked_once_mapped`。
- **新手概念课堂**：`/dev/shm` 里的共享内存像"公告栏"，大家都能看，但撤掉公告栏要等所有人都不看了（barrier）。私有区域像"自己的笔记本"，写完就能把封面撕掉（unlink），内容还在手里。
- **对你有什么影响**：多卡部署 + KV 卸载时，每个 rank 的内存布局更合理，减少不必要的复制和同步开销。

---

### 4. [Bugfix][KV Offload] FS 层短写报告为"可能磁盘满"而非 EIO

- **代码层面**：改 `csrc/fs_io.cpp`。新增 `kShortWrite = -2` 常量（不是 errno，因为短写本身不设 errno）。`_store_block` 在 `written >= 0 && written != size` 时返回 `kShortWrite`；`batch_store_block` 捕获后格式化成明确的 `OSError`，提示"可能是硬件错误或文件系统已满（ENOSPC）"。测试用 `RLIMIT_FSIZE` 人为制造短写。
- **新手概念课堂**：EIO 是"设备出错"，听起来像硬盘坏了；ENOSPC 是"没空间了"，用户知道该去清磁盘。把前者误报成后者，会让运维排查方向完全跑偏。
- **对你有什么影响**：KV 卸载到文件系统时，磁盘满会给出可操作的错误信息，而不是吓人的设备错误。

---

### 5. [Multimodal] 新增 TorchCodec 图像解码后端

- **代码层面**：新增 `image_backend` 参数，可选 `pil`（默认）或 `torchcodec`。TorchCodec 后端支持 JPEG/PNG/WebP/AVIF，直接返回 CHW `uint8` 张量，跳过 HF 图像处理器的 PIL→张量转换。带透明通道、CMYK、16-bit、动图等情况自动回退 Pillow。Dockerfile.s390x 升级 TorchCodec 到 v0.17.0 并关闭图像解码器编译。
- **新手概念课堂**：Pillow 解码图片像"先拆快递再重新打包"；TorchCodec 直接给你"已经摆好位置的零件"，少一道工序。
- **对你有什么影响**：多模态推理的图像预处理更快了，且像素值与 Pillow 后端完全一致（不会影响模型输出）。

---

### 6. [ROCm][Bugfix][DSv4.1] 在构建目录的副本上编译 gfx942 top-512 扩展

- **代码层面**：改 `vllm/models/deepseek_v41/amd/ops/topk.py`。原来 `load(sources=[_SOURCE])` 会让 hipify 在源文件旁边生成 `.hip` 文件——如果 vLLM 装在只读目录（非 root 用户），构建就失败。现在先用 `_get_build_directory` 拿到构建目录，`shutil.copyfile` 复制源文件进去（用 `os.replace` 原子替换避免并发问题），再在副本上编译。
- **新手概念课堂**：hipify 是"把 CUDA 代码翻译成 HIP 代码"的工具，它习惯把翻译结果写在输入文件旁边。就像翻译软件默认把译文存在原文同目录——如果那个目录是只读的，就写不进去。
- **对你有什么影响**：非 root 用户在 AMD 卡上跑 DeepSeek V4.1 不再因权限问题构建失败。

---

### 7. [ROCm][Perf][DSv4.1] 把 fused_wqa_wkv 和共享专家的 MXFP8 输入量化融合进 producer

- **代码层面**：把原本独立的 `mxfp8_e4m3_quantize` kernel launch 融合进上游 producer（Mega-MHC seam 的 `fused_wqa_wkv`、共享专家的 `gate_up_proj`，以及 fused SiLU-and-mul 的 `down_proj`）。新增测试 `test_dsv41_mxfp8_input_fusion.py` 断言融合后与未融合路径**逐位一致**。测试辅助函数 `_stub_sublayers` 用于替换 decoder 的 attention/FFN。
- **新手概念课堂**：融合 kernel 像"洗菜和切菜在一个工位完成"，省掉了把菜从水槽搬到案板的时间（一次 kernel launch + 一次显存读写）。
- **对你有什么影响**：AMD MI350/MI355 上 DeepSeek V4.1 推理更快，且数值完全不变。

---

### 8. [ROCm][Perf] Muse Spark：用一次 reduce-scatter 完成 token 分片的 MoE 归约

- **代码层面**：改 `vllm/models/muse_spark/amd/muse_moe_runner.py`。新增 `token_sharded` 判断（pre_norm_all_reduce + 有 fused_norm + TP>1）。当 `routed.dtype == reduce_dtype`（无需类型转换拷贝）时，把 chunk_size 设为全部 token，一次 reduce-scatter + all-gather 替代"每个 chunk 一对"。对超出默认 chunk 的尾部 token 做 TP 对齐处理，剩余部分走 all-reduce。
- **新手概念课堂**：原来是把一车货分成小批、每批都跑一趟快递；现在发现不用重新打包，直接整车发一次更划算。但如果最后有几个零头装不满一车，就单独走另一条路。
- **对你有什么影响**：Muse Spark 在 AMD 多卡上的 MoE 推理吞吐提升。

---

### 9. [Bugfix][Model] 避免 Dots3Note 视觉塔中的 host 同步

- **代码层面**：改 `vllm/models/dots3_note/nvidia/vision.py` 和 `vision_attention.py`。两处 `.to(device, non_blocking=True)` 换成 `async_tensor_h2d`（经 pinned 内存中转）。`_segment_ids` 用 `torch.searchsorted` 在 GPU 上计算每个 token 属于哪个 varlen 段，替代原来用 Python 循环 `for i in range(1, len(cu_seqlens))` 逐段切片——后者每次 `cu_seqlens[i]` 都会把标量拷回 host。attention mask 也从"先填 min 再逐段置 0"改成"一次性 `masked_fill_`"。
- **新手概念课堂**：Python 循环里访问 GPU 张量的单个元素（`cu_seqlens[i]`）会触发一次"GPU 算完 → 拷回 CPU"的往返，像每问一个字都要寄一封信。`searchsorted` 是把所有问题一次性打包寄过去。
- **对你有什么影响**：Dots3Note 视觉编码更快，尤其是多图/长序列场景。

---

### 10. [Bugfix][gpt-oss] 流式 MXFP4 专家按整块切分并保留调用方的 w2_bias

- **代码层面**：改 gpt-oss 的 per-expert 权重加载逻辑。原来按"填充后的参数尺寸"切分 TP 分片，但 MXFP4 的 checkpoint 是按"整块（block）"组织的，两者不一致会导致权重错位。现在改为按整块切分（w13：4 块分 2 卡各 2 块；w2：3 块分 2+1），并保留 padding 区域不动。同时修复 `w2_bias` 被覆盖的问题。测试用"每行一个不同值"的权重来捕捉行偏移错误。
- **新手概念课堂**：MXFP4 把 32 个 4-bit 值打包成一个块，块是不可分割的最小单位。按"字节数"切分就像把一盒鸡蛋按重量分给两个人——蛋会碎。必须按"盒"分。
- **对你有什么影响**：用 gpt-oss + TP 推理时，输出正确性得到保证（之前是静默错误）。

---

### 11. [RL][Model Loader] Direct reload：检查一次更新加载了每个参数

- **代码层面**：新增 `track_loaded_params(model)`，在冷启动和 direct reload 时记录哪些参数被权重加载器实际写过。`finish_direct_reload` 时对比"模型所有参数"与"已加载参数"，缺失则报错。配套修复 NCCL 权重传输：一次失败的 update（finish 抛异常）后，worker 保留的是那次失败的 metadata，下次 update 必须重发自己的 metadata 而不能复用旧的。新增测试 `test_nccl_trainer_resends_metadata_after_a_failed_update`。
- **新手概念课堂**：RL 训练里模型权重会频繁热更新。如果某次更新漏了一个参数，模型不会报错，只会"悄悄变笨"。这就像考试交卷前不检查有没有漏答题。
- **对你有什么影响**：用 vLLM 做 RL 训练的团队，权重更新出错会立刻暴露，而不是训练几小时后才发现指标异常。

---

### 12. [Perf][KDA] 为 Kimi-Linear 导出 Triton prefill checkpoint

- **代码层面**：改 `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py` 等。把 Triton 实现的 KDA（Kimi Delta Attention）prefill checkpoint 暴露出来，让 ROCm 上的 Kimi-Linear 也能用 `chunk_kda_with_fused_gate`（之前只有 NVIDIA 有）。测试新增 `kimi_amd` 实现项，并调用 `set_triton_allocator` 为非 ROCm 环境提供 scratch 分配器。另外把测试里 Llama 的 `vocab_size` 从 1000 改成 32000 以覆盖 TinyLlama tokenizer。
- **新手概念课堂**：线性注意力（如 KDA）的 prefill 阶段可以"分块并行计算"，checkpoint 就是块与块之间的中间状态。有了 checkpoint，prefill 可以拆成多段流水线执行。
- **对你有什么影响**：Kimi-Linear 在 AMD 卡上的长 prompt 预填速度提升。

---

### 13. [Bugfix][Multimodal] Pixtral (HF)：尊重处理器的 size 覆盖

- **代码层面**：改 `vllm/model_executor/models/llava.py` 里的 `PixtralHFProcessingInfo`，新增 `get_vision_encoder_info` 方法，让 `mm_processor_kwargs` 里的 `size`（或 `images_kwargs.size`）能正确传到视觉编码器，而不是被默认值覆盖。测试覆盖 `{}`（默认 3072 token）、`{"size": {"longest_edge": 512}}`（768 token）两种写法，以及"初始化时传"和"请求时传"两种时机。
- **新手概念课堂**：多模态模型把图片切成 patch，patch 数量决定 token 数。`longest_edge=512` 就是"最长边缩到 512 像素"，图变小了 token 自然变少。如果这个参数被静默忽略，你会以为调了但没生效。
- **对你有什么影响**：用 Pixtral (HF 格式) 时，`--mm-processor-kwargs` 的 size 参数终于生效了。

---

### 14. [Bugfix][EPLB][Hardware][Ascend] 在异步 EPLB worker 中为 NPU 设置 device_index

- **代码层面**：改 `vllm/distributed/eplb/eplb_state.py` 一行：`if self.device.type in ("cuda", "xpu")` 加上 `"npu"`。EPLB（专家并行负载均衡）的异步 worker 线程需要知道用哪张卡，之前只处理 CUDA/XPU，NPU 上 `device_index` 没设置。
- **新手概念课堂**：多卡机器上，"用哪张卡"必须显式指定。异步线程不会自动继承主线程的当前设备，所以要把 index 传过去，否则线程可能跑到卡 0 上去。
- **对你有什么影响**：昇腾 NPU 用户开 EPLB 时不再有设备索引问题。

---

### 15. [ROCm][Bugfix] 在 op 构建时探测 AITER mHC 支持，而非 import 时

- **代码层面**：改 `vllm/model_executor/layers/mhc.py`。原来在模块 import 时就计算 `HAS_AITER_MHC_PRE_NORM` 等布尔常量；现在改成运行时方法 `_aiter_mhc_op_accepts_norm("mhc_pre")`，op 实例上存 `_aiter_supports_norm` / `_use_aiter` 标志。测试相应从 monkeypatch 模块级常量改为设置实例属性。
- **新手概念课堂**：import 时做探测就像"进门前先测体温"，但门可能还没装好。AITER 的 op 是延迟构建的，import 时它还不存在，探测结果必然错。改成"用的时候再问"就对了。
- **对你有什么影响**：ROCm 上 AITER mHC 的可用性判断更准确，不会因为 import 顺序导致误判。

---

### 16. [Recipe] 新增 HY4 Flash 配方

- **代码层面**：新增 `recipes/models/tencent/Hy4-flash.yaml`。记录腾讯混元 HY4 Flash（220B 总参数 / 12B 激活，1M 上下文，多模态 MoE）的部署配方：专用 Docker 镜像、`--limit-mm-per-prompt '{"image":1,"video":0}'`、DFly 投机解码（5 个 draft token）、hy_v4 tool-call 和 reasoning parser 成对启用。AMD 全系标为 unsupported（无 ROCm 镜像且不支持 pip 安装）。
- **新手概念课堂**：Recipe 是"菜谱"——告诉你用什么镜像、什么参数、什么硬件能跑某个模型。省得你从零试错。
- **对你有什么影响**：想部署 HY4 Flash 的话，直接照抄 YAML 里的参数即可。

---

### 17. [Kernel][MiniMax-M3] 从 Hopper 稀疏 prefill 中去掉 `tl.make_block_ptr`

- **代码层面**：改 `vllm/models/minimax_m3/nvidia/ops/sparse_prefill.py` 的 `_gqa_sparse_fwd_tiled_kernel`。把 `tl.make_block_ptr` + `boundary_check` 的写法换成手动计算指针 + 显式 mask（`qo_mask` 同时处理 q_len、gqa_group_size、head_dim 三个维度的越界）。store 也做同样替换。
- **新手概念课堂**：`make_block_ptr` 是 Triton 的"自动挡"——帮你算地址和处理边界，但在某些形状下编译器优化不好。手动算指针是"手动挡"，代码更长但生成的 PTX 更干净。
- **对你有什么影响**：MiniMax-M3 在 Hopper 上的稀疏注意力 prefill 更快。

---

### 18. [Bugfix][Frontend] gpt-oss legacy tool-call header 中把特殊 token 文本当普通文本编码

- **代码层面**：改 `vllm/entrypoints/openai/parser/harmony_utils.py` 的 `_use_legacy_tool_call_headers`。`encoding.encode(f"assistant to={recipient}")` 加上 `disallowed_special=()`，让形如 `<|end|>` 的文本按普通字符编码，而不是被识别成特殊 token 或直接抛异常。测试构造了一个 recipient 名为 `functions.get_weather<|end|>` 的 tool call。
- **新手概念课堂**：tiktoken 默认会把 `<|end|>` 这种字符串当成"特殊 token"处理（可能报错）。加 `disallowed_special=()` 等于说"这里全是普通文本，别特殊对待"。就像在 Word 里输入 `*bold*` 不想被自动加粗，得先关掉自动格式。
- **对你有什么影响**：客户端回传的 tool call 名字里如果碰巧含特殊 token 文本，不再导致 500 错误。

---

### 19. [ROCm][Quantization] 用 8-bit Triton WNA16 kernel 在 ROCm 上服务 ModelOpt INT8_PB_WO

- **代码层面**：扩展 `TritonW4A16LinearKernel` 支持 8-bit 权重（`bits=8`，4 个值打包进一个 `int32`）。测试里的 `_pack_int4_along_n` / `_unpack_int4_along_n` / `_w4a16_reference` 都加了 `bits` 参数。文档更新：INT8_PB_WO 在 NVIDIA 上用 Marlin/Humming，在 ROCm 上用 Triton WNA16。
- **新手概念课堂**：WNA16 是"权重 N-bit、激活 16-bit"的量化范式。4-bit 和 8-bit 的 kernel 结构几乎一样，只是每 32 位整数里塞的个数不同（8 个 vs 4 个）。
- **对你有什么影响**：AMD 卡上现在能跑 ModelOpt 量化的 INT8 模型了，显存占用比 BF16 减半。

---

### 20. [ROCm][DSv4.1] 用 HIP VMM 在 DP rank 间共享 Engram host 表

- **代码层面**：新增环境变量 `VLLM_ROCM_ENGRAM_HIP_VMM=1`，开启后用 HIP VMM（虚拟内存管理）的 host memory 存放共享 Engram 表，比 `/dev/shm` 填充更快。默认关闭，因为某些 amdgpu 驱动（如 6.16.6）在副本退出后不释放 host 内存直到重启（引用 ROCm issue #13573）。任何 HIP VMM 失败都会在所有副本上回退到 `/dev/shm`。测试重构出 `_engram_host_table()` 辅助函数，统一 NVIDIA（cuMem）和 ROCm（HIP VMM）两条路径的测试。
- **新手概念课堂**：HIP VMM 像"虚拟地址映射"——多张卡可以把同一块物理内存映射到各自的地址空间，读写零拷贝。`/dev/shm` 也能共享，但要经过文件系统层，慢一些。代价是驱动 bug 可能导致内存泄漏。
- **对你有什么影响**：AMD 多卡跑 DeepSeek V4.1 且开启 Engram 时，可以手动开这个开关换性能；遇到内存不释放就关掉回退。

## sglang
## 💡 今日关键词（放在最前面！）

- **关键词：AMD gfx1250 / gfx950 内核适配**
  - 一句话通俗解释：SGLang 正在为 AMD 的新一代 GPU（gfx1250、gfx950）手写大量专用 CUDA/HIP 内核，包括 MoE、注意力、通信、投机解码。
  - 为什么社区在关注：Commit 6/7/13/14 连续出现 AMD 专属 kernel（KDA verify、sparse MLA、A8W4 MoE、custom all-reduce 调优），说明 SGLang 正在把 AMD 从"能跑"推进到"跑得快"。异构计算（NVIDIA + AMD + NPU）已是硬需求。
  - 对新手意味着什么：如果你只会写 PyTorch 层，AMD 内核方向人才缺口很大；学 HIP/ROCm、Triton、CUTLASS 风格 kernel 编程会有明显红利。

- **关键词：多模态重构（token-space 策略）**
  - 一句话通俗解释：把 Qwen VL 系列的多模态处理从"像素空间"迁移到"token 空间"，并统一到 `multimodal/` 目录下。
  - 为什么社区在关注：Commit 1/2 是一个 9 步重构的第 7 步，涉及 `mrope_rope_index` 文件迁移、processor 拆分，说明多模态代码在走向模块化。
  - 对新手意味着什么：想贡献多模态模型支持，先读懂 `python/sglang/srt/multimodal/` 的目录结构，比盲改 model 文件高效得多。

- **关键词：投机解码（Speculative Decoding）稳定性**
  - 一句话通俗解释：DFlash / DSpark / EAGLE 等投机解码路径的 buffer 越界、graph 捕获、采样 staging 问题被集中修复（Commit 5/10）。
  - 为什么社区在关注：投机解码是当前推理加速的主力，但 graph capture + 动态 batch 极易踩坑，社区在补"生产级"鲁棒性。
  - 对新手意味着什么：想学推理加速，投机解码是必修课；理解 CUDA Graph 与静态 buffer 的关系是核心门槛。

- **关键词：PD 分离 / HiCache / KV 卸载**
  - 一句话通俗解释：Prefill-Decode 分离架构下的资源清理（Commit 8）、HiCache 对象命名（Commit 16）、KV cache 卸载到主机内存（Commit 11）。
  - 为什么社区在关注：长上下文 + 高并发场景下，KV cache 管理是瓶颈；PD 分离成为大规模部署标配。
  - 对新手意味着什么：分布式推理不只是"多卡"，还要懂生命周期管理、命名空间、缓存层级。

- **关键词：Triton autotune 配置调优**
  - 一句话通俗解释：为特定 GPU（H20）+ 特定模型（GLM-5.3-Flash）+ 特定量化（FP8）手调 Triton kernel 参数（Commit 15）。
  - 为什么社区在关注：通用配置跑不满硬件，厂商/模型组合需要专属调优，这是"最后一公里"性能工程。
  - 对新手意味着什么：性能优化不只是改算法，读 JSON 配置、理解 `BLOCK_SIZE_M/N/K`、`num_stages`、`USE_TMA` 是实用技能。

## 🧪 重要知识点 & 动手实验

- **知识点 1：CUDA Graph 与静态 buffer 的"越界"陷阱（投机解码）**
  - **是什么**：CUDA Graph 会把一次前向的所有 kernel 和内存地址"录"下来，回放时地址固定；如果实际 batch 超过录制时的 buffer 大小，就会读到脏数据或越界。
  - **为什么重要**：Commit 5 修复的正是这个问题——`_MegakernelSelectorDraftSampler` 的 `max_bs` 为 `None`（表示用 megakernel 自己的 buffer），但旧代码无条件调用 `stage_sampling_params`，超出 buffer 的 batch 会踩内存。不懂这个，写投机解码时会在"小 batch 测试通过、大 batch 线上崩"。
  - **动手试试**：
    ```python
    # 模拟：graph 录制 bs=4，回放 bs=8
    import torch
    buf = torch.zeros(4, 8, device="cuda")  # 只够 4 个请求
    bs = 8
    try:
        # 错误做法：不检查直接写
        buf[:bs] = 1.0
    except RuntimeError as e:
        print("捕获到越界:", e)
    # 正确做法：先判断
    if bs <= buf.shape[0]:
        buf[:bs] = 1.0
    else:
        print("走 eager 路径，不用 graph buffer")
    ```
  - **预期结果**：看到 `捕获到越界: The expanded size of the tensor (8) must match...`，理解为什么 Commit 5 要加 `if sampler.max_bs is None or bs <= sampler.max_bs` 这个守卫。

- **知识点 2：Triton autotune 配置文件的字段含义（性能优化）**
  - **是什么**：Triton kernel 的 tile 大小、流水线级数、是否用 TMA（Tensor Memory Accelerator）等参数，会随 GPU 型号和问题规模变化，SGLang 用 JSON 文件存这些"最优解"。
  - **为什么重要**：Commit 15 给 H20 上的 GLM-5.3-Flash 加了 `USE_TMA: true` 和 `num_stages: 5`，这些不是随便填的——TMA 让大块数据搬运不占寄存器，`num_stages` 控制软件流水深度。填错会 OOM 或变慢。
  - **动手试试**：
    ```bash
    # 找到配置文件
    find python/sglang/srt/layers/moe/moe_runner/triton_utils/configs -name "*H20*GLM*" 2>/dev/null
    # 或直接看今天的 diff 文件
    cat "python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_8_0/E=288,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json" | head -30
    ```
  - **预期结果**：看到 `BLOCK_SIZE_M/N/K`、`GROUP_SIZE_M`、`num_warps`、`num_stages`、`USE_TMA` 等字段，理解每个数字对应"一次算多大的块、并行多少 warp、流水几级"。

- **知识点 3：多模态 processor 的"token 空间"vs"像素空间"**
  - **是什么**：图像输入有两种处理路径——传统是先把图片编码成 embedding（像素空间），新路径是先把图片压成离散 token 再和文本 token 一起处理（token 空间）。
  - **为什么重要**：Commit 2 把 `disable_fast_image_processor` 改成 `image_processor_backend="pil"`，Commit 1 是配套测试修复。token 空间能让多模态和纯文本走同一套调度/缓存逻辑，是架构统一的关键。
  - **动手试试**：
    ```python
    # 看 processor 基类如何声明 backend
    import inspect
    from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens
    print(inspect.getsource(MultimodalSpecialTokens)[:800])
    # 搜索 image_processor_backend 的所有用法
    # grep -rn "image_processor_backend" python/sglang/srt/multimodal/ | head
    ```
  - **预期结果**：看到 `image_token_id`、`mm_tokens` 等字段，理解 processor 如何把图片占位符映射到 token id。

- **知识点 4：分布式资源的生命周期管理（PD 分离）**
  - **是什么**：PD 分离架构里，KV manager 持有跨进程的共享内存/共识存储（`PPConsensusStore`），进程退出时必须显式 `close()`，否则会泄漏或让其他 rank 卡住。
  - **为什么重要**：Commit 8 在 Mooncake 和 Mori 两个 KV manager 的 `teardown()` 里加了 `if isinstance(self.request_status, PPConsensusStore): self.request_status.close()`。这类 bug 在单机测试里完全看不出来，只在多机长跑时暴露。
  - **动手试试**：
    ```python
    # 模拟一个需要显式关闭的资源
    class Store:
        def __init__(self): self.closed = False
        def close(self): self.closed = True
    class Manager:
        def __init__(self): self.request_status = Store()
        def teardown(self):
            # 漏掉这行就会泄漏
            if hasattr(self.request_status, "close"):
                self.request_status.close()
    m = Manager(); m.teardown()
    print("已关闭:", m.request_status.closed)  # True
    ```
  - **预期结果**：输出 `已关闭: True`；把 `close` 那行注释掉再跑，输出 `False`，体会"忘记清理"的后果。

- **知识点 5：LoRA 包装层必须"透传"基座属性**
  - **是什么**：LoRA 是在原模型层外面套一层，但上层代码（如 attention 的 `assert not self.o_proj.reduce_results`）会读 `layer.bias`、`layer.reduce_results` 等属性，包装层必须把这些"透传"下去。
  - **为什么重要**：Commit 18 修的是 `if hasattr(self.base_layer, "bias") and self.base_layer.bias is not None` —— 当 bias 是 `None` 时属性没被设置，上层读 `layer.bias is None` 会 `AttributeError`。同时给 MoE 包装层加了 `supports_deferred_finalize = False`，防止上层误以为支持延迟 finalize。
  - **动手试试**：
    ```python
    class Base:
        def __init__(self): self.bias = None
    class Wrapper:
        def __init__(self, base):
            self.base = base
            # 错误写法
            # if hasattr(base, "bias") and base.bias is not None:
            #     self.bias = base.bias
            # 正确写法
            if hasattr(base, "bias"):
                self.bias = base.bias
    w = Wrapper(Base())
    print("bias is None:", w.bias is None)  # True，没 AttributeError
    ```
  - **预期结果**：输出 `bias is None: True`；用错误写法会抛 `AttributeError: 'Wrapper' object has no attribute 'bias'`。

## 🔥 重要更新

- **Commit 2（多模态重构 7/9）**：把 Qwen VL 的 `mrope_rope_index` 从 `layers/rotary_embedding/` 迁到 `multimodal/`，并新增 token-space 策略。这是架构级重构，影响所有 Qwen VL 多模态模型，值得关注后续第 8/9 步。
- **Commit 7 + 13 + 14（AMD 三连发）**：MLA draft verify、KDA verify、sparse MLA、A8W4 MoE、custom all-reduce 调优，一天内三个 AMD 大 PR，说明 AMD 支持已进入"逐 kernel 打磨"阶段，对 AMD 用户是重大利好。
- **Commit 5（DFlash 采样 staging 越界修复）**：投机解码线上崩溃类 bug，影响所有用 DFlash/DSpark 的大 batch 场景，属于"不修就随机崩"的高危修复。
- **Commit 8（PD 分离资源泄漏修复）**：Mooncake/Mori KV manager 的 `PPConsensusStore` 未关闭，多机长跑会累积泄漏，属于生产环境必须跟进的修复。

## 📋 逐条解读

### 1. [Fix] tests: Use image_processor_backend in the Qwen token-space test (#51207)
- **代码层面**：改 `test/registered/unit/multimodal/test_qwen_tokenized_media.py`，把 `disable_fast_image_processor=True` 换成 `image_processor_backend="pil"`。只是一个测试参数名更新。
- **新手概念课堂**：就像你家的插座从"两孔"换成"三孔"，旧插头（`disable_fast_image_processor`）不能用了，得换新插头（`image_processor_backend`）。这是 Commit 2 重构的配套动作。
- **对你有什么影响**：普通用户无感；如果你有自定义脚本传了 `disable_fast_image_processor`，需要改成 `image_processor_backend="pil"`。

### 2. [Refactor] multimodal: [MM refactor 7/9] Add a token-space strategy for Qwen VL models (#46413)
- **代码层面**：把 `mrope_rope_index.py` 从 `layers/rotary_embedding/` 整体搬到 `multimodal/`，更新所有 import（`mrope.py`、`qwen3_omni_moe.py`、`ernie45_vl.py` 等），并新增 token-space 策略类。
- **新手概念课堂**：像把厨房里的调料从"按瓶罐分类"改成"按菜系分类"——文件还在，但归属变了。`mrope`（多模态 RoPE）是给图像/视频 token 算位置编码的，它本质上属于多模态，不该放在通用 rotary 层里。
- **对你有什么影响**：如果你 fork 过 SGLang 并 import 了 `sglang.srt.layers.rotary_embedding.mrope_rope_index`，需要改路径；Qwen VL 系列多模态推理行为不变，但代码更清晰。

### 3. [Fix] openai: Render DSV4.1 message tools with recipe (#49514)
- **代码层面**：改 `serving_deepseek_recipe.py` 把 `tool_choice` 从"只在有 tools 时传"改成"总是传"；改 `deepseek_v41.py` 的 `process_mm_data_async` 支持 `input_text` 是 list 时走 `fast_load_mm_data`；改 `deepseek_recipe_renderer.py` 新增 `_take_message_tools` 用 marker system message 包裹声明 tools 的消息。
- **新手概念课堂**：就像快递单上"是否保价"这一栏，以前只有买了保险才填，现在不管买不买都填——因为渲染器需要知道"用户到底想不想用工具"来决定哪些消息里声明的工具该显示。
- **对你有什么影响**：用 DeepSeek V4.1 的 OpenAI 兼容接口时，`tool_choice` 语义更准确；多模态输入（图片 URL 列表）处理更健壮。

### 4. [Fix] trtllm_mha: Pack a graph-built page table for eager reads (#47703)
- **代码层面**：`trtllm_mha_backend.py` 里把 `metadata.page_table = kv_view.ids` 改成 `.contiguous()`，SWA page table 同样处理；删掉了原来的 `assert is_contiguous()`。测试文件同步改成验证"eager 路径会 pack"。
- **新手概念课堂**：page table 是 KV cache 的"页表"，XQA/fmha_v2 这些 kernel 读它时**只看宽度不看 stride**。如果 table 是某个更宽 buffer 的 view（strided），kernel 会把第 2 行开始的数据全读错。`.contiguous()` 就是"把 view 拷贝成独立紧凑数组"。
- **对你有什么影响**：用 trtllm_mha backend + CUDA Graph + SWA（滑动窗口注意力）时，之前可能读到脏页导致输出乱码，现在修好了。

### 5. [Fix] dflash: Stage the draft sampler only for batches its buffers hold (#47644)
- **代码层面**：`dflash_worker_v2.py` 给 `_MegakernelSelectorDraftSampler` 加 `self.max_bs = None`，新增 `_stage_draft_sampler` 方法做 `bs <= sampler.max_bs` 守卫，把原来无条件的 `stage_sampling_params` 调用替换掉。
- **新手概念课堂**：就像演唱会检票——你的票只能坐前 4 排（buffer 大小），来了 8 个人（bs=8）就不能硬塞进去，得走"站票区"（eager 路径）。CUDA Graph 的 buffer 是静态的，超了必须绕开。
- **对你有什么影响**：用 DFlash 投机解码 + 大 batch 时不再随机崩溃或输出错乱。

### 6. [Feat] [AMD] muse-spark: Add gfx1250 support (#50707)
- **代码层面**：新增 `docker/muse_gfx1250/launch.sh` 启动脚本、`muse_gfx1250_tp8_ep8_tuned_grouped_fmoe.csv` 调优配置；改 `custom_all_reduce.cuh` 给 gfx1250 单机场景按 SM 数量分配 block；改 `sp_collective.cuh`。
- **新手概念课堂**：就像给新款 AMD 显卡写"专属驱动 + 专属游戏配置"——launch.sh 是启动参数模板，CSV 是 MoE kernel 的"最佳参数表"，custom all-reduce 是 GPU 间通信的加速库。
- **对你有什么影响**：在 AMD gfx1250 上跑 Muse-Spark 816B 模型现在有官方配方；all-reduce 在单机 8 卡场景下延迟更低。

### 7. [Feat] [AMD] speculative: MLA draft verify, DFLASH MLA context and target-only sampling kernels (#50999)
- **代码层面**：`extend_attention.py` 新增 `BF16_DOT` 编译期常量，在 gfx1250 或 `bf16_dot=True` 时把 fp8 K/V 上采样到 bf16 再 dot（绕开 gfx1250 fp8 tl.dot 的 bug）；新增 MLA draft verify、DFLASH MLA context、target-only sampling 内核。
- **新手概念课堂**：就像计算器算小数会丢精度，GPU 的 fp8 矩阵乘在某些架构上有 bug，那就"先转成 bf16 再算"——多花点带宽换正确性。
- **对你有什么影响**：AMD 上跑 MLA 架构（DeepSeek 系）+ 投机解码现在有专用 kernel，速度和正确性都提升。

### 8. [Fix] disaggregation: Close the PP consensus store when a KV manager is torn down (#51134)
- **代码层面**：`mooncake/conn.py` 和 `mori/conn.py` 的 `teardown()` 里加 `if isinstance(self.request_status, PPConsensusStore): self.request_status.close()`；测试里加 `m.request_status.close.assert_called_once_with()`。
- **新手概念课堂**：就像退租时要把水电煤气都过户/结清——KV manager 销毁时，它持有的跨进程共识存储（PP 流水线各 stage 之间同步用）必须显式关闭，否则资源泄漏。
- **对你有什么影响**：PD 分离 + PP 流水线长跑时不再泄漏共享内存/句柄。

### 9. [Test] [NPU] deepseek-v4: Restore the combined CP + DSPARK cases in the nightly multi-node pipeline (#51171)
- **代码层面**：改 `.github/workflows/nightly-test-npu.yml`，把之前因"dspark/dp/cp 冲突"被注释掉的 CP 组合测试恢复为默认，把 off_cp/off_dspark 版本注释掉。
- **新手概念课堂**：就像 CI 里"之前因为 bug 临时跳过某组测试，bug 修好了就把测试加回来"。CP（Context Parallel）、DP（Data Parallel）、DSPARK（投机解码）三者组合的测试恢复，说明冲突已解决。
- **对你有什么影响**：普通用户无感；关注 NPU 多机部署的开发者能看到官方验证覆盖更全。

### 10. [Perf] [NPU] dsv4: mirrorless seq_lens path for DFlash-family spec decode (#50763)
- **代码层面**：`ascend_dsv4_backend.py` 支持 `seq_lens_cpu is None`（mirrorless），新增 `_stable_compact_1d` 做静态形状 compact 避免 D2H 同步，graph replay ctx 里所有 `*_cpu` 字段允许是 device tensor；文档更新环境变量说明。
- **新手概念课堂**：就像以前每做一步都要"把答案抄到纸上"（D2H 拷贝），现在直接在"脑子里算"（device 侧完成），省掉来回搬运的时间。NPU 上 D2H 同步是性能杀手。
- **对你有什么影响**：NPU 上跑 DeepSeek-V4 + DFlash 投机解码更快，尤其 decode 阶段每步都省一次同步。

### 11. [Doc] [AMD] cookbook: Qwen3.5 MI355X FP8 agentic sets SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION=1 on the v0.5.23 image (#51152)
- **代码层面**：纯文档 + JSX 配置生成器，MI355X 镜像从 v0.5.22 升到 v0.5.23，新增 `SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION=1` 环境变量说明（AITER 后端会把显式 `--mem-fraction-static 0.8` 乘 0.85 变成 0.68，设这个变量后不缩放，KV pool +45%）。
- **新手概念课堂**：就像你点了"大杯"奶茶，店员默认给你做成"中杯"（乘 0.85），设这个变量就是告诉店员"我要的就是大杯，别缩"。
- **对你有什么影响**：AMD MI355X 上部署 Qwen3.5-397B-FP8 长上下文 agentic 场景，KV cache 容量提升 45%，能扛更高并发。

### 12. [Test] [NPU] kimi_k3: retune the 128k decode prefix case to C6 (#48894)
- **代码层面**：`test_npu_performance_utils.py` 新增 `REQUEST_THROUGHPUT_TOLERANCE = 0.98` 和 `request_throughput` 断言/打点；PD mix 测试启动时改成轮询 `sglang_thread.is_alive()` 避免死等 3600 秒。
- **新手概念课堂**：就像性能测试加了"每秒请求数"这个指标，并且启动服务时如果进程死了立刻报错，而不是傻等一小时。
- **对你有什么影响**：普通用户无感；NPU 性能回归测试更敏感、失败更快。

### 13. [Feat] [AMD] kimi-k3: gfx950 KDA verify, A8W4 MoE, latent-tail and LL collective kernels (#50998)
- **代码层面**：新增 `kda_verify_gfx950.cuh`（KDA 链式投机验证 kernel，含 causal conv1d + gated delta-rule 递推）、A8W4 MoE、latent-tail、LL collective 等 kernel。
- **新手概念课堂**：KDA（Kimi Delta Attention）是 Kimi-K3 的线性注意力变体，verify 阶段要把 T 个候选 token 一次性验证，kernel 里用 LDS（共享内存）存中间量、用 DPP（Data Parallel Primitives）做 warp 内归约——这是手写高性能 kernel 的典型套路。
- **对你有什么影响**：AMD gfx950 上跑 Kimi-K3 + 投机解码有专用 kernel，速度显著提升。

### 14. [Feat] [AMD] dsa: gfx950 sparse MLA, DCP top-k and FP4 index-K kernels for Kimi-K3.1 (#50997)
- **代码层面**：新增 `sparse_mla_decode_gfx950.cuh`（从 aiter 移植的稀疏 MLA decode，处理 Kimi-K3.1 的 fp8_ds_mla 656 字节/ token 布局），加 DCP top-k、FP4 index-K kernel。
- **新手概念课堂**：MLA（Multi-head Latent Attention）是 DeepSeek 系的压缩注意力，sparse MLA 只算 top-k 相关的 KV；"656 字节/ token"是精心设计的布局（512 fp8 NoPE + 4 fp32 scale + 64 bf16 RoPE），kernel 直接按字节偏移读，省掉反序列化。
- **对你有什么影响**：AMD 上跑 Kimi-K3.1 长上下文推理更快，显存占用更低。

### 15. [Perf] [NVIDIA] fused_moe: Add H20 down-projection (TMA) configs for GLM-5.3-Flash TP8/TP4 (#48313)
- **代码层面**：改 `triton_3_8_0/E=288,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json`，给每个 batch size 桶调 `BLOCK_SIZE_M/N/K`、`GROUP_SIZE_M`、`num_stages`，并加 `USE_TMA: true`。
- **新手概念课堂**：TMA（Tensor Memory Accelerator）是 Hopper 引入的硬件单元，专门做大块数据搬运，不占寄存器；`num_stages` 是软件流水深度，越大越能掩盖延迟但越吃 shared memory。
- **对你有什么影响**：H20 上跑 GLM-5.3-Flash 的 MoE down-projection 更快（具体提升看 benchmark）。

### 16. [Fix] hicache: Expect the KV dtype suffix in page-unified L3 object names (#51081)
- **代码层面**：测试里 `model_id=self.model_path` 改成 `model_id=f"{self.model_path}-dtype_bfloat16"`，匹配 `storage_model_name()` 的命名规则。
- **新手概念课堂**：就像文件命名从 `model.bin` 改成 `model-bf16.bin`——L3 缓存的对象名带上了 dtype 后缀，避免 bf16 和 fp8 缓存互相污染。
- **对你有什么影响**：用 HiCache + page-unified 布局时，不同 dtype 的 KV cache 不会串味。

### 17. [Doc] cookbook: Add Hy4-flash cookbook (#51160)
- **代码层面**：新增 `docs/cookbook/autoregressive/Tencent/Hy4-flash.mdx` 和配套 JSX 配置，去掉 Hy4-Preview 的 `tag: NEW`。
- **新手概念课堂**：就是给新模型（腾讯 Hy4-flash，~220B MoE，稀疏 + 滑动窗口注意力 + Engram 记忆）写"开箱即用"的部署文档。
- **对你有什么影响**：想部署 Hy4-flash 有官方配方了，支持 BF16/FP8 + H200/B300/GB300。

### 18. [Fix] lora: Expose the bias and deferred-finalize flags on LoRA-wrapped layers (#51083)
- **代码层面**：`lora/layers.py` 里把 `if hasattr(base, "bias") and base.bias is not None` 改成 `if hasattr(base, "bias")`（bias 为 None 也透传）；MoE 包装层加 `self.supports_deferred_finalize = False` 和 `self.supports_routed_deferred_finalize = False`。
- **新手概念课堂**：包装层要"如实转告"基座层的属性——基座说"我没 bias"（`bias=None`），包装层也得说"我没 bias"，不能装作这个属性不存在。
- **对你有什么影响**：用 LoRA + 某些模型（如 DeepSeek MLA）时不再因 `layer.bias` 不存在而 `AttributeError`；MoE + LoRA 不会被误判支持延迟 finalize。

### 19. [Fix] model-loader: Support RunAI for PP Nemotron MTP drafts (#49348)
- **代码层面**：`weight_utils.py` 的 `runai_safetensors_weights_iterator` 新增 `load_group` 参数，当 load_group 是 stage-local（world_size 不匹配）时强制 `is_distributed=False`；`nemotron_h_mtp.py` 加 `loaded_input_embedding` 标志；`load_config.py` 更新注释。
- **新手概念课堂**：PP（流水线并行）下每个 stage 只加载自己那部分权重，但 RunAI streamer 默认"全世界一起加载"，会让 stage 0 等 stage 3 的 draft 权重——所以要改成"各 stage 独立流式加载"。
- **对你有什么影响**：Nemotron MTP draft 模型 + PP + RunAI 加载不再卡死。

### 20. [Fix] cpu: Pin outlines to the version the outlines grammar backend imports (#51086)
- **代码层面**：commit 说明是"把 outlines 库 pin 到 outlines grammar backend 实际 import 的版本"（diff 未给出，推测是 requirements 或 pyproject 里的版本约束）。
- **新手概念课堂**：就像你代码里 `import requests`，但 requirements 里写 `requests>=2.0`——某天 3.0 发布把 API 改了，你就崩了。pin 版本就是"锁死在我验证过的那个版本"。
- **对你有什么影响**：CPU 上跑结构化输出（grammar-constrained decoding）不再因 outlines 版本漂移而 import 失败。

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)