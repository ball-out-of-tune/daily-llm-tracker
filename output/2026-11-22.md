# 🚀 vLLM & SGLang 每日更新 — 2026-11-22

> 自动生成于 2026-11-22 08:47 UTC | AI 解读: ✅ 含代码解读

## vllm
## 💡 今日关键词

- **关键词：KV Offload 的读写方向控制（access_mode）**
  - 一句话通俗解释：给 KV 缓存卸载的存储层加一个"只读 / 只写 / 读写"开关，让一个实例可以专门写缓存、另一个专门读缓存。
  - 为什么社区在关注：这说明 KV 缓存卸载已经从"单机省显存"进化到"多实例共享缓存池"的架构，社区在认真做缓存的分工协作（一个实例预热、多个实例消费）。
  - 对新手意味着什么：如果你在学推理优化，理解"缓存分层 + 读写解耦"是迈向大规模推理系统的关键一步，这类设计在真实生产里非常常见。

- **关键词：Inkling 2 模型适配 + SM120（5090 / DGX Spark）性能优化**
  - 一句话通俗解释：vLLM 一天内连续多个 commit 在为同一个新模型做适配，还专门针对消费级 Blackwell 显卡（sm120）优化小 batch 的矩阵乘法。
  - 为什么社区在关注：说明新模型上线速度在加快，同时"消费级 GPU 也能跑好推理"正在成为现实需求，不再是数据中心专属。
  - 对新手意味着什么：如果你只有一张 5090 或类似显卡，这是好消息——社区在认真优化你能用得起的硬件。

- **关键词：输入边界与安全加固（Security Hardening）**
  - 一句话通俗解释：给 stop strings、stop_token_ids、bad words 这些用户可控参数加上数量上限，防止恶意请求耗尽服务器资源。
  - 为什么社区在关注：vLLM 越来越多被部署为公网服务，安全从"可选"变成"必须"，这是成熟开源项目的标志。
  - 对新手意味着什么：学 AI 推理不只要懂性能，还要懂"攻击面"——任何用户能传的字段都是潜在风险点。

- **关键词：多模态输入的健壮性（torch.Tensor / NumPy / PIL 兼容）**
  - 一句话通俗解释：多个 commit 在修"用户传进来的图片/视频格式和处理器预期不一致"的问题。
  - 为什么社区在关注：多模态是当下最热的方向，但输入格式五花八门（PIL、NumPy、Tensor），兼容性 bug 是高频痛点。
  - 对新手意味着什么：写多模态代码时，"入口处统一格式转换"是最值得养成的习惯。

- **关键词：错误分类与可观测性（Error Classification）**
  - 一句话通俗解释：把笼统的 `ValueError` 换成带参数名、带上下文的 `VLLMValidationError` / `RuntimeError`。
  - 为什么社区在关注：错误信息质量直接影响用户体验和排障效率，这是工程成熟度的体现。
  - 对新手意味着什么：写代码时"抛什么异常、带什么信息"是基本功，好的错误信息能省下几小时的 debug。

---

## 🧪 重要知识点 & 动手实验

- **知识点 1：KV 缓存卸载的读写方向控制（access_mode）**
  - **是什么**：KV Offload 把 GPU 上放不下的 KV 缓存挪到 CPU 内存或磁盘/对象存储。`access_mode` 让你决定这个存储层是"只提供读取"、"只接收写入"还是"两者都行"。
  - **为什么重要**：在多实例部署里，你可以让 A 实例只写（预热共享缓存），B、C 实例只读（直接命中缓存，跳过 prefill）。不懂这个，你可能白白浪费算力重复做 prefill；配错了（比如只读层被当成可写）会导致缓存永远不命中或写入失败。
  - **动手试试**：
    ```bash
    # 先看看文档里 access_mode 的定义
    python -c "print(open('docs/features/kv_offloading_usage.md').read()[:2000])"

    # 跑一下相关的单元测试，观察 access_mode 如何影响调度
    pytest -v tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py -k "access_mode or store_beyond"
    ```
  - **预期结果**：测试通过，你能在输出里看到 `OffloadAccessMode.READWRITE` 等枚举被传入配置；读文档能理解 `read` / `write` / `readwrite` 三档的语义差异。

- **知识点 2：多模态输入的格式归一化（PIL / NumPy / torch.Tensor）**
  - **是什么**：不同的图像/视频处理器对输入类型有假设——有的只吃 `np.ndarray`，有的只吃 `PIL.Image`。修复方式是在入口处做 `torch.Tensor → numpy`、`np.ndarray → PIL` 的转换。
  - **为什么重要**：用户从不同来源拿到的数据格式各异（视频解码器给 Tensor、OpenCV 给 NumPy、PIL 读图给 Image），不做归一化就会在深层代码里炸出难以定位的 `AttributeError`。
  - **动手试试**：
    ```python
    import numpy as np, torch
    from PIL import Image
    from transformers.image_transforms import to_pil_image

    # 模拟三种输入格式
    arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    t = torch.from_numpy(arr).permute(2, 0, 1)  # CHW tensor
    img = Image.fromarray(arr)

    # 统一转成 PIL
    for x in [arr, t, img]:
        pil = to_pil_image(x if not isinstance(x, np.ndarray) else x)
        print(type(x).__name__, "->", type(pil).__name__, pil.size)
    ```
  - **预期结果**：三种输入都能成功转成 `PIL.Image`，你会看到统一入口的价值——后续代码只需要处理一种类型。

- **知识点 3：采样阶段的 token 屏蔽（masked_token_ids）**
  - **是什么**：在 softmax 之前把指定 token 的 logits 设为 `-inf`，让模型永远不会采样到它们。典型场景：lm_head 里有 padding 行、或占位符 token 不该被生成。
  - **为什么重要**：不做屏蔽时，模型偶尔会吐出无意义的占位符 token，导致下游解析失败。这是"生成可控性"的基础能力，也是结构化输出、工具调用的前置条件。
  - **动手试试**：
    ```python
    from vllm import LLM, SamplingParams

    MODEL = "vllm-project/tiny-random-LlamaForCausalLM"
    VOCAB = 32000
    KEEP = 1234
    masked = [i for i in range(VOCAB) if i != KEEP]

    llm = LLM(MODEL, enforce_eager=True, masked_token_ids=masked)
    out = llm.generate("Hello my name is", SamplingParams(temperature=0, max_tokens=8))
    print(out[0].outputs[0].token_ids)  # 应该全是 1234
    ```
  - **预期结果**：输出 token 全部是 `1234`，因为其他 token 的 logits 都是 `-inf`，这是验证屏蔽生效最直观的方法。

- **知识点 4：异常分类——从 `ValueError` 到带上下文的 `VLLMValidationError`**
  - **是什么**：把"参数非法"这类错误从泛泛的 `ValueError` 换成携带 `parameter` 和 `value` 字段的专用异常；把"用法错误"（比如对非生成模型调 `generate()`）从 `ValueError` 改为 `RuntimeError`。
  - **为什么重要**：调用方可以按异常类型做不同处理（参数错误→返回 400，运行时错误→返回 500），错误信息里带参数名让用户一眼知道改哪里。
  - **动手试试**：
    ```python
    from vllm import LLM
    from vllm.exceptions import VLLMValidationError

    try:
        LLM("vllm-project/tiny-random-LlamaForCausalLM",
            data_parallel_size=2)  # 单进程下不支持
    except VLLMValidationError as e:
        print("parameter =", e.parameter)
        print("message   =", str(e)[:120])
    ```
  - **预期结果**：捕获到 `VLLMValidationError`，`e.parameter == "data_parallel_size"`，你能从异常对象直接拿到出错字段，而不是去正则匹配字符串。

- **知识点 5：TMA + 共享内存约束下的 kernel 调参（SM120 案例）**
  - **是什么**：Triton kernel 用 TMA（Tensor Memory Accelerator）搬运数据时，每个 SM 的共享内存有限（SM120 约 99 KiB）。commit 8 把 `num_stages` 从 4 降到 2、`block_n` 从 128 降到 64，就是为了让两个 stage 能塞进共享内存。
  - **为什么重要**：GPU kernel 调参不是"越大越好"——pipeline stage 太多会超出共享内存，导致编译失败或性能骤降。理解"资源约束决定参数选择"是性能优化的核心思维。
  - **动手试试**：
    ```python
    # 用 Triton 官方教程感受 num_stages 对共享内存的影响
    # 先确认环境
    import torch, triton
    print("triton", triton.__version__)
    print("cuda capability", torch.cuda.get_device_capability())

    # 观察一个简单 matmul 在不同 num_stages 下的编译结果
    # （在支持 TMA 的 GPU 上运行；否则跳过）
    import triton.language as tl
    @triton.jit
    def k(X, Y, N: tl.constexpr):
        offs = tl.arange(0, N)
        x = tl.load(X + offs)
        tl.store(Y + offs, x * 2)
    ```
  - **预期结果**：打印出你的 GPU compute capability（如 `(12, 0)` 表示 sm120）；在 sm120 上你会理解为什么社区要把 stage 数压到 2——共享内存就是硬预算。

---

## 🔥 重要更新

- **Commit 12 `[Model] Add Inkling 2 Mini`**：新增一个完整模型支持，配套 CI 任务（含 DGX Spark sm12 专项），说明社区正在快速跟进前沿模型，且覆盖消费级硬件。
- **Commit 16 `[Bugfix][Security] Complete input-bound hardening`**：补齐之前安全 PR 遗漏的边界检查（stop strings / stop_token_ids / bad words 数量上限），是面向公网部署的关键加固。
- **Commit 14 `[Model] Add masked_token_ids`**：新增"永不采样某些 token"的能力，是生成可控性和结构化输出的基础设施。
- **Commit 1 `[KV Offload] Add access_mode`**：让 KV 卸载支持读写解耦，为多实例共享缓存池铺路，是架构级能力扩展。

---

## 📋 逐条解读

### 1. [KV Offload] Add access_mode (read|write|readwrite)
- **代码层面**：在 `vllm/v1/kv_offload/config.py` 新增 `OffloadAccessMode` 枚举（`READ` / `WRITE` / `READWRITE`），并把它接入 offloading spec 的配置解析；`fs`、`obj`、`p2p` 等二级存储层都新增可选 `access_mode` 字段，默认 `readwrite`。调度器在 `prepare_store` / `prepare_load` 时根据该模式决定是否真正执行写或读。测试里大量构造配置的地方补上了 `access_mode=OffloadAccessMode.READWRITE`。
- **新手概念课堂**：想象一个共享图书馆。`readwrite` 是普通读者（能借能还）；`read` 是"只借不还"的读者（消费别人放进去的书）；`write` 是"只放不借"的捐赠者（预热书库给别人用）。这个开关让多个 vLLM 实例可以分工：一个专门把算好的 KV 写进共享存储，其他实例只读命中，省掉重复 prefill。
- **对你有什么影响**：如果你部署多实例共享缓存，现在可以显式声明每个实例的角色，避免"只读实例误写"或"只写实例永远不命中"的配置错误；普通单机用户默认值不变，无感知。

### 2. [Bugfix][Multimodal] Pass image kwargs through the OpenCUA processor
- **代码层面**：修复 `vllm/model_executor/models/opencua.py` 的处理器，让 `mm_processor_kwargs`（比如 `max_pixels`）真正透传到内部的 Qwen2-VL 图像处理器；新增 `tests/models/multimodal/generation/test_opencua.py` 和 `processing/test_opencua.py` 验证 1920×1080 图片在 `max_pixels=512²` 下产生 312 个 image token。
- **新手概念课堂**：`mm_processor_kwargs` 就像给相机设置的"分辨率上限"。之前 OpenCUA 的处理器把用户设的参数"吞掉了"，导致图片按默认尺寸处理，token 数爆炸。修复后参数一路传到真正干活的处理器手里。
- **对你有什么影响**：用 OpenCUA 模型时，你设的 `max_pixels` 终于生效了，大图不再意外产生巨量 token 拖慢推理。

### 3. [Frontend] Classify errors in offline LLM APIs
- **代码层面**：`vllm/entrypoints/llm.py` 中，`kv_transfer_config` 解析失败和 `data_parallel_size` 单进程不支持这两处从 `ValueError` 改为 `VLLMValidationError`（带 `parameter=` 和 `value=`）；`generate` / `enqueue` / `chat` / `enqueue_chat` 在模型不是生成式时从 `ValueError` 改为 `RuntimeError`。
- **新手概念课堂**：异常类型就像快递单上的"问题分类"。"参数填错了"（Validation）和"你拿错了工具"（Runtime）是两类问题，应该分开放。之前都塞进 `ValueError` 大筐，调用方没法区分。
- **对你有什么影响**：写脚本调用 vLLM 时，你可以 `except VLLMValidationError` 精确捕获配置错误并提示用户改哪个参数，而不是笼统地抓 `ValueError`。

### 4. [Bugfix][KV Offload] Store the chunks that fit when a request exceeds the CPU tier
- **代码层面**：调度器在 `prepare_store` 时，如果请求的 chunk 数超过 CPU 层容量，不再整体失败，而是"能存多少存多少"，剩下的留在 `keys_to_retry` 里下个 step 继续尝试；只有请求结束时仍有未存的 chunk 才计一次 `ALLOCATION_FAILURE`。测试新增 `test_store_beyond_manager_capacity_continues_on_later_steps` 等。
- **新手概念课堂**：就像搬家时卡车装不下所有箱子——以前是"装不下就一件都不搬"，现在是"先装满一车，剩下的下一趟再搬"。只要最后都搬完了就不算失败。
- **对你有什么影响**：长请求的 KV 卸载不再因为"一次装不下"就完全放弃，缓存命中率提升，显存压力更平滑。

### 5. [Bugfix][Multimodal] Accept torch.Tensor videos in vLLM's own HF processors
- **代码层面**：在 `internvl.py`、`nano_nemotron_vl.py`、`ovis2_5.py` 三个处理器的视频入口处加 `if isinstance(video, torch.Tensor): video = video.cpu().numpy()`。
- **新手概念课堂**：不同工具对"视频"的表示不同——PyTorch 用 Tensor，OpenCV/NumPy 用 ndarray。就像有人用 Word 文档、有人用 PDF，处理器只认一种。加一行转换就是"入口处统一翻译"。
- **对你有什么影响**：你传 `torch.Tensor` 格式的视频不再报 `AttributeError: 'Tensor' object has no attribute 'shape'` 之类的错。

### 6. [Bugfix][Multimodal] Accept NumPy images and PIL frame videos in Muse Spark Open
- **代码层面**：`vllm/models/muse_spark/common/processor.py` 的 `preprocess_image` 签名扩展为 `Image.Image | np.ndarray | torch.Tensor`，入口用 `to_pil_image()` 统一转换；视频分支检测"PIL 帧列表"并 `np.stack` 成 ndarray。
- **新手概念课堂**：和第 5 条同一思路，但方向相反——这次是把 NumPy/Tensor 转成 PIL，把 PIL 帧列表转成 NumPy 数组。核心思想都是"在边界处归一化"。
- **对你有什么影响**：Muse Spark Open 模型对输入格式更宽容，你用 PIL 逐帧读视频也能直接喂进去。

### 7. [Perf][Inkling2-Mini] Improve small M GEMM on 5090 (sm120)
- **代码层面**：新增 `vllm/models/inkling2/nvidia/ops/ll_gemm_sm120.py`，提供 `sm12x_bf16_ll_gemm`（小 M 场景的 bf16 GEMM）和 `qkv_bmm`（Q/K/V 三个投影合并成一次 batched matmul）；`attention.py` 的 `_project` 在 `use_qkv_bmm` 为真时走融合路径；`loop.py` 的 `blend` 用 `sm12x_bf16_ll_gemm` 替换 `torch.bmm`。测试从 `test_sm12x_gemv_*` 改名为 `test_sm12x_ll_gemm_*`。
- **新手概念课堂**：小 batch（M 很小）时，普通 GEMM 的"准备时间"比"计算时间"还长，像开大卡车送一个包裹。专用小 M kernel 就像换小面包车，启动快、浪费少。`qkv_bmm` 则是"三个快递合并成一车送"。
- **对你有什么影响**：在 5090 / DGX Spark 这类 sm120 显卡上跑 Inkling2-Mini，短序列（decode 阶段）吞吐会明显提升。

### 8. [Perf][Model] Inkling 2: keep the TMA k_group_partials path on SM12x
- **代码层面**：`linear.py` 的 `k_group_partials` 去掉 `and not sm120` 的限制，让 SM120 也能走 TMA 路径；但针对 SM120 把 `block_n` 限制为 64、`num_stages` 设为 2，因为 SM120 每 block 共享内存约 99 KiB，两个 48 KiB stage 刚好塞下。
- **新手概念课堂**：TMA 是 GPU 上的"专用搬运工"，比普通 load 高效。但每个 SM 的"仓库"（共享内存）有大小限制，SM120 的仓库比数据中心卡小，所以一次只能搬更少、用更少的流水线级数。
- **对你有什么影响**：SM120 用户不再被排除在 TMA 加速路径之外，但代价是 block 更小——这是硬件约束下的合理权衡。

### 9. [CI/Build][NVIDIA] Use PyTorch's cu135 nightly for Rubin
- **代码层面**：`.buildkite/release-pipeline.yaml` 里 Rubin 构建从 `cuda13.4` builder 切到 `cuda13.5`，删掉 `TRITON_INSTALL_FROM_SOURCE_*` 相关 build-arg；`docker/Dockerfile` 删除 `tools/build_triton_from_source.sh` 的调用和对应 ARG。原因是 PyTorch 已发布 cu135 nightly，不再需要从源码编译 Triton 来绕过 ptxas 对 SM107 的支持问题。
- **新手概念课堂**：之前因为 ptxas 13.3 不认识 SM107 指令，只能自己编译 Triton；现在上游修好了，就像"零件厂终于生产出标准件，不用自己 3D 打印了"。
- **对你有什么影响**：普通用户无感知；如果你自己构建 Rubin 镜像，构建流程简化、不再需要从源码编 Triton。

### 10. [Model] Hy4-flash Cleanup
- **代码层面**：`vllm/models/hy_v4/amd/model.py` 大幅重构 `load_weights`——把权重映射抽到 `vllm/models/hy_v4/weight_utils.py`（`HYV4_MLP_MAPPER`、`HYV4_WEIGHTS_MAPPER`），删掉本地缓存的 `split_expert_params_mapping` 和一堆内联逻辑；`test_tensor_schema.py` 里非 CUDA 平台从"断言抛 `NotImplementedError`"改为 `pytest.skip`。
- **新手概念课堂**：这是典型的"重构清理"——把一大坨 if-else 权重加载逻辑拆成声明式的映射表。就像把"手写 200 行 if 判断"换成"查表"，可读性和可维护性都大幅提升。
- **对你有什么影响**：Hy4-flash 在非 CUDA 平台上的测试行为从"报错"变成"跳过"，CI 更干净；权重加载逻辑更清晰，后续加新变体更容易。

### 11. [Bugfix] Whisper prompt conditioning: use the real `<|startofprev|>` token
- **代码层面**：Whisper 的 prompt 条件化之前用错了特殊 token（`<|prev|>`），改为正确的 `<|startofprev|>`；`get_generation_prompt` 保证 decoder prompt 布局是 `get_prompt_ids(prompt)` + SOT 序列，不再额外前置 decoder start token。新增 `test_decoder_prompt_layout` 验证。
- **新手概念课堂**：Whisper 用特殊 token 告诉模型"接下来是上一段的上下文"。用错 token 就像给信封贴错邮票——模型看不懂，prompt 就白传了。
- **对你有什么影响**：用 Whisper 做语音转写时，你传的 `prompt`（词汇/风格提示）终于真正影响解码结果，转写准确率（尤其专有名词）会提升。

### 12. [Model] Add Inkling 2 Mini
- **代码层面**：新增 `vllm/models/inkling2/` 完整模型实现（attention、MoE、loop、sconv 等），配套 CUDA kernel（`csrc/inkling2/allreduce_conv_mix.cu` 等），CI 新增 `models_basic.yaml` 里的 Inkling2 测试任务和 DGX Spark sm12 专项任务。`allreduce_conv_mix.cu` 把 `kTokenThreads` 从常量改为模板参数，支持 hidden size 4608 和 2304 两种。
- **新手概念课堂**：这是一个"新模型全家桶"——模型结构、kernel、测试、CI 一次性到位。`kTokenThreads` 模板化就像"一件衣服做成可调节尺码"，同一套 kernel 服务两个不同 hidden size 的模型变体。
- **对你有什么影响**：你现在可以直接用 vLLM 跑 Inkling 2 Mini，且在消费级 sm120 显卡上有专门优化和 CI 保障。

### 13. [Bugfix] Normalize native GTE positional embedding configuration
- **代码层面**：`vllm/model_executor/models/config.py` 的 `verify_and_update_model_config` 里，如果 GTE 模型的 `position_embedding_type` 是 `None`，显式设为 `"rope"`。
- **新手概念课堂**：有些模型配置文件里"位置编码类型"字段是空的，代码后续要读它就会 `None` 崩溃。这行代码就是"填空默认值"——没写就按 RoPE 处理。
- **对你有什么影响**：加载某些 GTE 模型不再因为配置缺字段而报错，开箱即用。

### 14. [Model] Add masked_token_ids to never sample given tokens
- **代码层面**：`ModelConfig` 新增 `masked_token_ids: list[int] | None`，在采样前把这些 token 的 logits 设为 `-inf`；`__post_init__` 校验 token id 范围（`[-1, VOCAB_SIZE]` 之外报错）；`SpeculativeConfig` 拒绝与"有损"投机解码方法（`fly`、`synthetic`）组合，因为那些方法不保证严格按 logits 采样。新增 `tests/v1/sample/test_masked_token_ids.py`。
- **新手概念课堂**：`masked_token_ids` 就像考试时的"排除选项"——正确答案只可能在剩下的选项里。实现上是在 softmax 前把不要的 token 概率压到 0。
- **对你有什么影响**：你可以强制模型永不输出某些 token（比如 lm_head 的 padding 行、占位符），提升生成可控性；但注意它和某些投机解码方法不兼容，配置时会报错提醒。

### 15. [Bugfix][Multimodal] Return 400 for unreadable embedding payloads
- **代码层面**：`vllm/renderers/embed_utils.py` 的 `safe_load_prompt_embeds` 等函数在解码失败时抛 `VLLMValidationError`（带 `param="image_embeds"` 等），HTTP 层转成 400；内存耗尽等服务器侧问题仍是 500。文档 `multimodal_inputs.md` 补充说明。新增测试验证"传垃圾 base64 返回 400 且服务仍健康"。
- **新手概念课堂**：客户端传了坏数据（400）和服务器自己崩了（500）是两回事。之前坏 base64 可能触发 500，让用户以为是服务器问题；现在明确告诉用户"你传的数据读不了"。
- **对你有什么影响**：调 API 传 embedding 时，如果格式错了会收到清晰的 400 和参数名，方便你定位；服务端不会因为一个坏请求挂掉。

### 16. [Bugfix][Security] Complete input-bound hardening left incomplete by PR #51447
- **代码层面**：`SamplingParams` 新增 `VLLM_MAX_STOP_STRINGS`（4）、`VLLM_MAX_STOP_TOKEN_IDS`（128）、`VLLM_MAX_NUM_BAD_WORDS`（128）、`VLLM_MAX_BAD_WORDS_TOTAL_TOKENS`（1024）上限；`stop_token_ids` 改为 `frozenset` 加速查找；`stop` 字符串去重；`CompletionRequest` / `ChatCompletionRequest` 的 pydantic 模型同步加 `max_length` 校验；DeepSeek V3.2 的 tool-message 处理从线性扫描改为更高效的结构。文档 `security.md` 补充说明。
- **新手概念课堂**：这些参数都是用户可控的，如果允许传 100 万个 stop string，服务器光解析就能被拖垮（DoS）。给每个字段加上限就像"自助餐每人限拿 4 个盘子"——防止一个人把整个餐厅吃空。
- **对你有什么影响**：正常使用无感知；如果你之前传了超过 128 个 stop_token_ids，现在会收到明确的 400 错误。部署公网服务时建议按文档调低这些上限。

### 17. [XPU] skip not applied tests in lora/test_layer.py and test_config.py on XPU
- **代码层面**：`.buildkite/intel_jobs/lora_intel.yaml` 加 `--deselect` 跳过 XPU 不支持的 GLM5next KDA 测试；`test_tensor_schema.py` 把 XPU 不支持模型列表扩展为 `{GLM-5.3-Flash, Hy4-preview, Hy4}`；`test_config.py` 给 CUDA-only 测试加 `skipif(current_platform.is_xpu())`；`vllm/models/glm5next/__init__.py` 删除"XPU 直接 raise NotImplementedError"的 import 期检查，改为测试层跳过。
- **新手概念课堂**：之前 GLM5next 在 XPU 上一 import 就炸，导致整个测试文件都跑不了。改成"测试层跳过"就像"门口贴个告示：本房间 XPU 用户请勿入内"，而不是"整栋楼锁死"。
- **对你有什么影响**：XPU 用户跑测试时不再被无关的 CUDA-only 用例卡住；GLM5next 在 XPU 上仍不支持，但错误更友好。

### 18. [Perf][Model] Export ZAYA1 CCA prefill checkpoints
- **代码层面**：`vllm/model_executor/models/zaya.py` 的 `ZayaCCA._mix_with_state` 支持 `MambaPrefillCheckpointMetadata`——在 prefill 进行到指定 offset 时，把 CCA 状态导出到 checkpoint slot，供后续复用。新增测试 `test_zaya_cca_exports_state_at_checkpoint` 验证"checkpoint 状态 == 单独跑该 offset 的结果"。
- **新手概念课堂**：CCA（连续卷积注意力）的状态像"滚动窗口的历史摘要"。prefill 很长时，中间状态以后可能被复用（比如分块 prefill、投机解码），所以在特定位置"存个档"能省掉重算。
- **对你有什么影响**：ZAYA1 模型在长 prefill + 复用场景下性能提升；普通用户无感知，但如果你做分块推理或前缀缓存，这个能力很关键。

### 19. [Bugfix][EC Connector] Keep marker token embeddings for pruned Qwen3-VL video on encoder-only instances
- **代码层面**：Qwen3-VL 视频裁剪（`video_pruning_rate`）后，时间戳和视觉标记 token 的 embedding 必须保留；之前 encoder-only 实例（`mm_encoder_only=True`）会把这些标记丢掉，导致和完整模型输出不一致。修复让 encoder-only 路径也产出这些标记 embedding。新增 `test_encoder_only_matches_full_model_for_pruned_video` 对比两者输出。
- **新手概念课堂**：视频裁剪就像"删减电影片段"，但片头的时间戳字幕不能删——下游模型靠它对齐时间。encoder-only 实例之前把字幕也删了，导致结果不对。
- **对你有什么影响**：用 encoder-only 部署 Qwen3-VL 做视频推理时，输出和完整模型一致，不再出现"裁剪后结果漂移"的问题。

### 20. [Doc] Add vLLM security review skill
- **代码层面**：新增 `.agents/skills/review-vllm-security/SKILL.md`，定义安全审查工作流（追踪输入→快速筛查→查兄弟路径→确认资源释放），按变更类型（端点/多模态/模型加载/结构化输出/引擎/缓存/部署/日志）列出检查清单；`pr-checklist/SKILL.md` 里加链接，要求涉及不可信输入、共享资源、权限的 PR 走安全审查。
- **新手概念课堂**：这是给 AI agent（和人类 reviewer）的"安全检查清单"。就像飞机起飞前的 checklist——不靠记忆，靠清单保证不漏项。
- **对你有什么影响**：如果你给 vLLM 提 PR 且涉及用户输入处理，会被要求对照这个清单自查；对普通用户意味着 vLLM 的安全审查流程更系统化，公网部署更放心。

## sglang
## 💡 今日关键词

- **关键词：GPU 数值一致性（CPU/GPU 浮点差异）**
  - 一句话解释：同样的数学公式在 CPU 和 GPU 上算出来的结果可能不完全一样，尤其是 `pow`、`exp` 这类函数。
  - 为什么社区在关注：Commit 1 发现 Qwen-Image-Edit 在"编码器卸载到 CPU"和"常驻 GPU"两种模式下生成的图不一样，根因就是 RoPE 频率表在 CPU 上初始化、搬到 GPU 后没有重建。随着大模型显存越来越紧张，"部分组件卸载"成为常态，这类隐蔽的数值差异会越来越多。
  - 对新手意味着什么：如果你做推理部署，一定要知道"同一份权重、不同执行路径"可能输出不同结果。学会写"逐位对比测试"是必备技能。

- **关键词：int32 溢出（大模型 LoRA 服务）**
  - 一句话解释：32 位整数最大约 21 亿，当张量元素超过这个数量，行偏移计算就会"绕回"变成负数或错误值。
  - 为什么社区在关注：Commit 2 里 LoRA 的 chunked SGMV kernel 用 int32 计算行偏移，当 `(rows-1)*width > 2^31` 时最后一行读错位置。大模型 + 大词表 + 大 batch 的场景越来越常见。
  - 对新手意味着什么：写 kernel 或索引计算时，先问自己"这个索引最大能到多少"。养成用 `int64` 处理"元素总数"类索引的习惯。

- **关键词：投机解码（Speculative Decoding）重构**
  - 一句话解释：用小模型"猜"多个 token，大模型一次性验证，加速生成。
  - 为什么社区在关注：今天有 4 个 commit（9、10、15、16）都在重构投机解码的 verify 流程，把 EAGLE、NGRAM、DSpark 三种 draft 算法的"接受逻辑"统一成 `VerifyAccept` 结构。这是典型的"多实现收敛到统一抽象"的工程演进。
  - 对新手意味着什么：投机解码是当前 LLM 推理加速最热门的方向之一。理解"draft-verify-accept"三段式是入门必修课。

- **关键词：内存卸载与映射保留（Offload / mmap）**
  - 一句话解释：显存不够时把权重放到 CPU 内存，需要时再搬回 GPU；"映射保留"指不复制数据、直接复用 checkpoint 文件在内存里的映射。
  - 为什么社区在关注：Commit 11 让 diffusion 在主机内存紧张时保留 mmap 映射而不是复制权重，Commit 1 处理卸载后的数值一致性。说明"低显存部署"已是 diffusion 推理的核心痛点。
  - 对新手意味着什么：学会用 `mmap`、`torch.load(mmap=True)`、理解 pinned memory 与 page cache，是部署大模型的基本功。

- **关键词：AMD / AITER 后端适配**
  - 一句话解释：AITER 是 AMD 的 AI 算子库，类似 NVIDIA 的 cuBLAS/cuDNN 生态位。
  - 为什么社区在关注：Commit 6、7 都是 AMD 相关修复（KV layout 读取、MoE topk 断言移除），说明 SGLang 正在认真做 ROCm 支持，异构计算不再是"能跑就行"。
  - 对新手意味着什么：如果你只有 AMD 显卡，现在入门 LLM 推理的门槛在降低；同时"跨后端一致性"是很好的练手方向。

---

## 🧪 重要知识点 & 动手实验

### 知识点 1：CPU 与 GPU 的浮点数值差异（Commit 1）

- **是什么**：`torch.pow`、`torch.exp` 在 CPU 和 GPU 上的实现不同，最后一位有效数字可能不同。对 RoPE 这种"频率表"敏感的位置编码，微小差异会逐层放大成肉眼可见的图像差异。
- **为什么重要**：模型卸载（offload）时权重在 CPU 初始化再搬到 GPU，如果不重建频率表，结果就和常驻 GPU 不一致。这类 bug 极难排查，因为 loss/精度指标看起来都正常。
- **动手试试**：
```python
import torch

dim, theta = 128, 10000.0
idx = torch.arange(0, dim, 2, dtype=torch.float32)

# CPU 上算
inv_cpu = 1.0 / (theta ** (idx / dim))

# 搬到 GPU 后再算（模拟 offload 后重建）
inv_gpu = (1.0 / (theta ** (idx.cuda() / dim))).cpu()

print("bitwise equal:", torch.equal(inv_cpu, inv_gpu))
print("max abs diff:", (inv_cpu - inv_gpu).abs().max().item())
```
- **预期结果**：`bitwise equal` 很可能是 `False`，`max abs diff` 是一个极小的数（1e-8 量级）。这说明"公式一样≠结果一样"，需要用 `torch.equal` 而不是 `allclose` 来验证一致性。

### 知识点 2：int32 溢出与安全索引（Commit 2）

- **是什么**：`torch.int32` 最大值是 2,147,483,647。当 `rows * width` 超过它，`row * width` 的乘法结果会溢出。
- **为什么重要**：LoRA 服务里 `permutation` 表用 int32 存物理行号，乘上 `width`（hidden size）后可能超界，导致读到错误内存——不报错，只是结果全错。
- **动手试试**：
```python
import torch

rows, width = 65538, 32768
print("rows*width =", rows * width, "> 2^31?", rows * width > 2**31)

idx_i32 = torch.tensor([rows - 1], dtype=torch.int32)
offset_i32 = idx_i32 * width          # 溢出！
idx_i64 = idx_i32.to(torch.int64)
offset_i64 = idx_i64 * width          # 正确

print("int32 offset:", offset_i32.item())
print("int64 offset:", offset_i64.item())
print("expected    :", (rows - 1) * width)
```
- **预期结果**：`int32 offset` 会变成负数或错误值，`int64 offset` 与 `expected` 一致。这就是 Commit 2 里 `.to(tl.int64)` 修复的本质。

### 知识点 3：投机解码的"draft-verify-accept"三段式（Commit 9/10/15）

- **是什么**：draft 模型先猜 `k` 个 token，target 模型一次前向验证这 `k` 个位置，接受最长的正确前缀，再额外吐一个 bonus token。
- **为什么重要**：这是当前 LLM 推理加速最主流的手段。理解 `accept_lens`、`accept_index`、`accept_mask` 三个量的含义，就能读懂所有投机解码实现。
- **动手试试**（纯 CPU，模拟接受逻辑）：
```python
import torch

# 假设 draft 猜了 4 个 token，target 验证后每行接受长度不同
accept_lens = torch.tensor([2, 4, 1])          # bs=3，含 bonus token
width = 4
tokens = torch.arange(3 * width).reshape(3, width)  # 每行的候选 token

# 构造 accept_mask（参考 VerifyAccept.accept_mask 的实现）
positions = torch.arange(width)
mask = positions[None, :] < accept_lens[:, None]
print("mask:\n", mask)

# 取出每行被接受的 token
accepted = [tokens[i][mask[i]].tolist() for i in range(3)]
print("accepted:", accepted)
```
- **预期结果**：第 0 行取前 2 个、第 1 行取全部 4 个、第 2 行取第 1 个。这就是 `VerifyAccept.accept_tokens()` 的核心逻辑。

### 知识点 4：Triton kernel 里的 stride 与缓冲区容量（Commit 4）

- **是什么**：张量的 `stride(0)` 是"沿第 0 维走一步跨多少个元素"。缓冲区可能分配了 `capacity=8` 步，但本次 verify 只用 4 步，若 kernel 里用 `cache_steps=4` 当 stride 就会读错行。
- **为什么重要**：Mamba 的状态缓存、KV cache 都是"预分配大缓冲区、按需使用"的模式。混淆"容量"和"使用量"是经典 bug。
- **动手试试**：
```python
import torch

capacity, steps = 8, 4
buf = torch.zeros(2, capacity, 3, 4)   # [batch, capacity, heads, dim]
print("stride(0) =", buf.stride(0))    # 每跨一个 batch 走 capacity*3*4 步

# 错误写法：用 steps 当 stride
wrong_base = buf.data_ptr() + 1 * steps * 3 * 4 * buf.element_size()
# 正确写法：用 stride(0)
correct_base = buf.data_ptr() + 1 * buf.stride(0) * buf.element_size()
print("wrong == correct?", wrong_base == correct_base)
```
- **预期结果**：`False`。当 `capacity != steps` 时两者不等，这正是 Commit 4 修复的 bug。

### 知识点 5：mmap 与内存映射保留（Commit 11）

- **是什么**：`safetensors` 加载时用 `mmap` 把文件映射到虚拟内存，不真正复制数据。当系统内存紧张时，可以"释放映射"而不是"复制一份 CPU 权重"。
- **为什么重要**：diffusion 模型动辄几十 GB，复制一份 CPU 权重可能直接 OOM。保留映射意味着"需要时从磁盘页缓存重新读"，牺牲一点延迟换内存安全。
- **动手试试**：
```python
import torch, tempfile, os
from safetensors.torch import save_file, load_file

with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "w.safetensors")
    save_file({"w": torch.randn(1024, 1024)}, p)

    # 普通加载：数据进内存
    w1 = load_file(p)["w"]
    # mmap 加载：惰性映射（safetensors 内部默认用 mmap）
    w2 = load_file(p, device="cpu")["w"]
    print("shape:", w2.shape, "dtype:", w2.dtype)
    print("file size MB:", os.path.getsize(p) / 1e6)
```
- **预期结果**：文件大小约 4MB，加载后内存占用不会翻倍（因为 mmap 共享页缓存）。如果你用 `psutil` 观察进程 RSS，会发现 mmap 加载的增量远小于文件大小。

---

## 🔥 重要更新

1. **Commit 13 `[Feat] diffusion: Request watchdog and /health 503`**：新增 `--request-time-limit`，请求超时后 `/health` 返回 503 并 dump 所有线程栈。这是生产环境"卡死检测"的关键能力，对运维影响巨大。

2. **Commit 9/10/15/16 投机解码统一重构**：把 EAGLE/NGRAM/DSpark 的 verify-accept 逻辑收敛为 `VerifyAccept` + `TargetVerify` + `commit_verify_accept`。这是架构级重构，未来新增 draft 算法会容易得多。

3. **Commit 11 `[Fix] diffusion: Preserve mapped component weights during low-memory offload`**：低内存卸载时保留 mmap 映射，避免复制整份权重导致 OOM。对消费级显卡跑 diffusion 是实打实的体验提升。

4. **Commit 12 `[Perf] diffusion/lingbot-video-moe: compute timestep modulation once per sample`**：把 `(B*S, D)` 的 modulation 计算降为 `(B, D)`，序列越长收益越大，是典型的"消除冗余计算"优化。

---

## 📋 逐条解读

### 1. [Fix] diffusion: Rebuild the Qwen2.5-VL vision RoPE table on the GPU so Qwen-Image-Edit gives the same image with the encoder offloaded
- **代码层面**：在 `qwen2_5vl_vision.py` 里自定义了一个 `Qwen2_5VLVisionRotaryEmbedding`，加了一个 `recompute_on_device_change` 开关；`forward` 时如果发现 `inv_freq` 的设备或 dtype 变了，就在当前设备上重新计算频率表。`qwen_image.py` 里在创建 pipeline 时给非 Edit-2509 的模型打开这个开关。
- **新手概念课堂**：RoPE 就像给每个位置发一个"频率身份证"，让模型知道 token 在序列里的相对位置。这个身份证是在 CPU 上批量印好的，搬到 GPU 后如果直接复用，可能和"在 GPU 上现场印"的版本有细微差别——就像同一份菜谱，两个厨师放盐的克数差 0.01g，最后味道就不一样。
- **对你有什么影响**：如果你用 Qwen-Image-Edit 并开启了编码器卸载（省显存），以前生成的图和常驻 GPU 模式不一样，现在一致了。Edit-2509 特意保留旧行为，因为它的一致性基准是用旧表生成的。

### 2. [Fix] kernels/lora: Use int64 token row offsets in the csgmv shrink and expand kernels
- **代码层面**：在 `chunked_sgmv_expand.py` 和 `chunked_sgmv_shrink.py` 里，把 `tl.load(permutation + ...)` 的结果显式 `.to(tl.int64)`。新增了一个 10GB 显存占用的回归测试，构造 `(65538-1)*32768 > 2^31` 的场景。
- **新手概念课堂**：`permutation` 表存的是"逻辑 token → 物理行号"的映射，像电影院座位号。int32 只能表示到 21 亿号座位，当影院有 21 亿个座位时，最后几排的座位号算出来就"绕回"到前面的排了。
- **对你有什么影响**：在多 LoRA 大模型服务里，当 batch × 序列长度 × hidden size 特别大时，以前可能读到错误的 LoRA 权重行，输出乱码；现在修好了。普通小规模使用感知不到。

### 3. [Fix] diffusion/nomas: Run each requested video of a /v1/videos job as its own sample
- **代码层面**：`nomas.py` 的 pipeline config 新增 `supports_sequential_multi_output_inference()` 返回 `True`；新增测试验证 `num_outputs_per_prompt=2` 时跑两次独立的 sample（seed 42 和 43），而不是一次生成两个。
- **新手概念课堂**：Nomas 的"latent session"一次只能装一个样本，就像一台洗衣机一次只能洗一件衣服。你要求洗两件，正确做法是分两次洗，而不是硬塞进去。
- **对你有什么影响**：调用 `/v1/videos` 时设置 `n=2`，以前可能报错或行为异常，现在会顺序生成两个视频，且每个 seed 独立递增，结果可复现。

### 4. [Fix] mamba: Use the intermediate state buffer stride during Triton verify
- **代码层面**：`mamba_ssm.py` 的 `_selective_scan_update_kernel` 把参数 `cache_steps` 改名为 `stride_intermediate_batch`，调用处传 `intermediate_states_buffer.stride(0)` 而不是 `cache_steps`。新增测试参数化 `capacity=[4,8]`、`tree=[False,True]` 验证行 stride。
- **新手概念课堂**：缓冲区像一排预分配的储物柜，`capacity=8` 个柜子，这次只用了 4 个。如果 kernel 以为"每行占 4 个柜子"去算偏移，就会读到隔壁 batch 的数据。正确做法是问缓冲区"你每行实际占多少柜子"（`stride(0)`）。
- **对你有什么影响**：使用 Mamba 模型 + 投机解码（EAGLE tree verify）时，中间状态缓存可能被读错，导致生成质量下降或结果不稳定；现在修复。

### 5. [Fix] moe/tuning: Tune NemotronH's non-gated relu2 MoE with its real shape
- **代码层面**：`common_utils.py` 新增 `get_moe_activation()` 返回 `(activation, is_gated)`；`calculate_shard_intermediate_size` 增加 `is_gated` 参数，非 gated 时不再乘 2。benchmark 脚本对非 gated SiLU 模型直接报错退出。配置文件命名加入 `is_gated` 区分。
- **新手概念课堂**：gated MoE 的 w1 里"门"和"上投影"叠在一起，宽度是 `2*intermediate`；non-gated 只有上投影，宽度是 `intermediate`。用错宽度调出来的 kernel 配置就像给 40 码的脚买 42 码的鞋——能穿但不合脚。
- **对你有什么影响**：NemotronH 这类 non-gated MoE 模型以前可能用了错误的 tuning 配置，性能不达标；现在用真实 shape 调优。普通用户感知是"推理更快了"。

### 6. [Fix] [AMD] aiter: Read the KV layout of the pool wrapped by HybridLinearKVPool
- **代码层面**：`aiter_backend.py` 的 `_kv_cache_is_vectorized_5d` 改为先判断 `has_full`，只要 full 或 sliding_window 子池存在就取对应 view，不再要求"必须有 sliding_window 才取 view"。新增 Qwen3.5-0.8B（hybrid linear attention）的 GSM8K 测试。
- **新手概念课堂**：`HybridLinearKVPool` 是个"套娃"——外面一层是混合池，里面包着 full attention 和 sliding window 两个子池。AITER 后端以前只会在"有 sliding window"时往里看，遇到"只有 full"的混合池就看不到真正的 KV 布局。
- **对你有什么影响**：在 AMD MI300X/MI355X 上跑 Qwen3.5 这类 hybrid linear attention 模型，以前可能因为读错布局而 decode 出错；现在正常。

### 7. [Fix] [AMD] moe/topk: stop AITER rejecting apply_routed_scaling_factor_on_output
- **代码层面**：`topk.py` 里删掉一行 `assert not apply_routed_scaling_factor_on_output, "Not implemented"`。
- **新手概念课堂**：`apply_routed_scaling_factor_on_output` 是 MoE 的一个开关——把 routed scaling factor 乘在输出上而不是 gate 上。AITER 后端以前直接断言"不支持"，现在放开。
- **对你有什么影响**：AMD 用户启用某些 MoE 模型（需要这个开关）时不再启动失败。改动极小但解锁了一个配置组合。

### 8. [Perf] diffusion: Avoid boolean indexing in placeholder validation
- **代码层面**：`gemma_3.py` 和 `qwen_vl.py` 里把 `inputs_embeds[mask].numel()` 这种"布尔索引取元素个数"改成 `mask.sum().item()` 或 `mask[..., 0].sum().item()`。
- **新手概念课堂**：布尔索引 `x[mask]` 会真的把满足条件的元素"挑出来"生成新张量，很慢；而 `mask.sum()` 只是数一数有多少个 True，快得多。就像清点人数，不需要把每个人拉出来排队，只需数举手的人。
- **对你有什么影响**：多模态编码器每次前向都做这个校验，改成 O(1) 后省掉一次大张量拷贝。对长序列、大 batch 的多模态推理有可感知的加速。

### 9. [Refactor] speculative: One target-verify ring for EAGLE, NGRAM and DSpark
- **代码层面**：`dspark_worker_v2.py` 里删掉散落的 grammar mask、watermark、logprob 处理代码，统一走 `TargetVerify.prepare(...)` 返回的 `verify` 对象，再调 `verify.grammar_mask.apply()`、`verify.watermark.apply()`。大量 import 被清理。
- **新手概念课堂**：以前三种 draft 算法各自写一套"验证后处理"逻辑，像三家分店各有一套收银流程；现在统一成一个"总收银台"（`TargetVerify`），新店开张直接复用。
- **对你有什么影响**：使用投机解码时行为更一致，新算法接入更快。普通用户感知不到，但社区贡献者受益。

### 10. [Refactor] speculative: Describe verify accepts with VerifyAccept and commit them in one call
- **代码层面**：新增 `sampling/verify_accept.py`，定义 `VerifyAccept`（`tokens`、`accept_lens`、可选 `accept_index`）和 `commit_verify_accept()`。`accept_tokens()` / `accept_mask()` 提供统一视图。watermark adapter 里的 `append_accept_tokens` 被替换为 `commit(accept)`。
- **新手概念课堂**：`VerifyAccept` 是一张"验货单"——记录"哪些 token 被接受了、接受了几个"。`commit` 是把这张单子一次性入账到惩罚历史和水印上下文，而不是各处零散地改状态。
- **对你有什么影响**：带 penalty（重复惩罚等）或 watermark 的投机解码，以前可能因为提交顺序问题出现状态不一致；现在原子化提交。

### 11. [Fix] diffusion: Preserve mapped component weights during low-memory offload
- **代码层面**：`component_residency_strategies.py` 的 `_load_component` 增加判断：CUDA + 独立显存 + 无 snapshot + 主机内存不足 + 全文件映射 + dtype 匹配时，保留 mmap 视图不复制。`component_manager.py` 的 `HostPinBudget` 传入 `node_local_ranks` 分摊预算。文档补充了详细说明。
- **新手概念课堂**：普通卸载像"把书从书架搬到桌上"，需要额外空间；保留映射像"在书上贴个书签，需要时按书签翻回原页"——不占额外空间，但翻页慢一点。当桌子（主机内存）不够时，书签是唯一选择。
- **对你有什么影响**：低内存机器跑 diffusion 时，卸载编码器/VAE 不再因为"复制一份 CPU 权重"而 OOM。代价是后续 H2D 可能稍慢。文档同时说明：breakable CUDA graph 要求 DiT 常驻，不能同时开 DiT 卸载。

### 12. [Perf] diffusion/lingbot-video-moe: compute timestep modulation once per sample
- **代码层面**：`lingbot_video_moe.py` 里 `temb6` 从 `(B*S, 6D)` 降为 `(B, 6D)`，block 里用 `(temb6 + scale_shift_table).unsqueeze(1)` 广播；只有启用 `LINGBOT_VIDEO_GATED_RESIDUAL` 时才 expand 成 `(B, S, D)`。`norm_out_modulation` 同样只算一次。
- **新手概念课堂**：timestep modulation 只依赖时间步 `t`，和序列位置无关。以前每个 token 都算一遍同样的东西，像给全班每人发一份一模一样的通知；现在只印一份贴在墙上。
- **对你有什么影响**：LingBot 视频生成模型在长序列（视频 token 多）时明显加速。测试验证了输出与旧实现数值一致。

### 13. [Feat] diffusion: Request watchdog and /health 503 when a request runs past its limit
- **代码层面**：新增 `--request-time-limit {SECONDS|TASK=SECONDS}` 参数；`sampling_params.py` 加 `request_task()` 钩子；超时后 dump 所有线程栈、`/health` 返回 503、拒绝新请求。文档新增"Request time limit"章节。
- **新手概念课堂**：watchdog 像厨房里的定时器——菜做太久就报警，告诉经理"这桌出问题了"，并把当前所有厨师在干什么都记录下来（线程栈）。注意它不"杀"请求，只是标记不健康。
- **对你有什么影响**：生产部署时，GPU kernel 卡死不再表现为"服务看起来健康但请求永远不返回"。配合负载均衡摘除 + 重启，能快速恢复。注意：队列等待时间不计入。

### 14. [Perf] diffusion/registry: Skip native discovery for metadata-only lookups
- **代码层面**：`registry.py` 里 `has_registered_diffusion_model_path`、`_get_config_info`、`get_model_info` 从 `_ensure_registry_initialized()`（会导入所有 pipeline）改为 `load_external_model_package()`（只加载外部包）。`get_model_info` 里 `_PIPELINE_REGISTRY.get` 改为 `get_pipeline_class`（惰性）。
- **新手概念课堂**：以前查"这个模型叫什么名字"要把整个 pipeline 库都 import 一遍，像查电话簿先把整本书复印一份；现在只翻到那一页。
- **对你有什么影响**：`sglang` 启动、模型列表查询、配置解析变快，尤其是 import 大量 pipeline 模块的场景。

### 15. [Refactor] speculative: One definition of logit adjustments and truncated sampling for verify gates
- **代码层面**：`sampling_batch_info.py` 新增 `adjusts_logits(info, has_grammar)` 和 `need_truncated_sampling(info)` 两个纯函数；`logits_processor.py` 的 `verify_uses_vocab_shards` 改为调用 `adjusts_logits`；`dflash_worker_v2.py` 等改用 `need_truncated_sampling`。
- **新手概念课堂**：`adjusts_logits` 回答"这次采样会不会改 logits"（有 grammar / penalty / bias / watermark 就会）；`need_truncated_sampling` 回答"会不会截断分布"（top-k/p/min-p）。以前这两个判断散落各处，容易不一致。
- **对你有什么影响**：投机解码 + 分片词表（vocab shards）的兼容性判断更准确，减少"该走分片却走了全量"或反之的性能损失/正确性问题。

### 16. [Refactor] sampling: Build watermark rows per step
- **代码层面**：`scheduler.py` 的 `run_batch` 里新增 `batch.sampling_info.watermark = build_watermark_rows(batch.reqs, batch.device)`；`SamplingBatchInfo.from_schedule_batch` 不再构建 watermark；`filter_batch` 里删掉 watermark 的过滤逻辑。
- **新手概念课堂**：水印策略以前在"建 batch"时算一次，但 batch 会动态增删请求（`filter_batch`），导致水印行和请求对不上。现在改成"每次 launch 前按当前 reqs 重建"，像每次开会前重新点名。
- **对你有什么影响**：使用 watermark（如 Gumbel 水印）时，动态 batch 下不再出现行错位导致的水印失效或误标。

### 17. [Fix] glm5_next: Gather DFlash aux hidden states for every scattered MoE all-to-all backend
- **代码层面**：`glm5_next.py` 用 `is_moe_input_scattered_across_dp_ranks()` 替代硬编码的四个 backend 判断；新增 `_captures_attn_tp_slices()` 静态方法，根据 `layer.attn_boundary.plan` 判断当前 capture 是否看到的是 ATTN_TP 分片，只有分片时才 `all_gather`。
- **新手概念课堂**：MoE 的 all-to-all 会把 token 打散到不同 rank，此时 capture 到的 hidden state 是"分片"的，需要 `all_gather` 拼回来。但如果 `--disable-attn-tp-gather` 让 batch 不分片，就不该再 gather，否则重复。
- **对你有什么影响**：GLM5-Next 使用 DFlash 投机解码 + 各种 MoE all-to-all 后端（DeepEP、Mooncake、Ascend ops 等）时，aux hidden state 收集正确，draft 质量稳定。

### 18. [Test] transformers: Fix three unit tests that fail on transformers main
- **代码层面**：`test_hf_transformers.py` 里把 `PretrainedConfig()` 实例改成 `SimpleNamespace`（因为新版 transformers 的 `PretrainedConfig` 对 `rope_scaling` 有校验）；`test_tied_embedding_plan_normalizes` 改用 `AutoModel.from_config(config).tp_plan` 而不是 `config.base_model_tp_plan`。
- **新手概念课堂**：这是"追上游"的典型——transformers 库升级后 API 行为变了，测试要跟着改。用 `SimpleNamespace` 是为了绕过新校验，只测被测函数本身。
- **对你有什么影响**：普通用户无感。但对 CI 维护者重要：这些测试在 transformers main 上不再红。

### 19. [Perf] diffusion/loader: Read remote DiT shard headers concurrently
- **代码层面**：`loader/utils.py` 的 `_hub_dit_parameter_counts` 在 shard 数 > 1 时用 `ThreadPoolExecutor(max_workers=4)` 并发调 `parse_safetensors_file_metadata`（每个是一次 Hub range request）。测试用 `Barrier(2)` 验证两个 shard 确实并发读取。
- **新手概念课堂**：safetensors 文件头包含参数元数据，读它是一次网络请求。多个 shard 串行读像"一个一个打电话问"，并发读像"同时发微信问"。
- **对你有什么影响**：从 HuggingFace Hub 加载多 shard 的 DiT（如大视频模型）时，参数统计阶段更快。注意：只并发"读 header"，不并发"下载权重"。

### 20. [Fix] cuda-graph: Initialize Breakable backend flag in prefill test fixture
- **代码层面**：`test_prefill_cuda_graph_runner.py` 的 fixture 里补一行 `runner._is_breakable_backend = isinstance(backend, BreakableCudaGraphBackend)`。
- **新手概念课堂**：测试 fixture 手动构造 runner 对象时，漏初始化了一个布尔标志，导致后续代码读到未定义属性或走错分支。就像组装家具时漏装一个螺丝，看起来能用但一受力就散。
- **对你有什么影响**：纯测试修复，普通用户无感。但它提醒我们：手动 mock 对象时，要对照 `__init__` 补齐所有状态字段。

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)