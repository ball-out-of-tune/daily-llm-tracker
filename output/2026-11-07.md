# 🚀 vLLM & SGLang 每日更新 — 2026-11-07

> 自动生成于 2026-11-07 07:44 UTC | AI 解读: ✅ 含代码解读

## vllm — 最近 24h commits

- **[[Perf][MiniMax-M3] Warm up DeepGEMM NVFP4 MegaMoE kernels before serving (#66227)](https://github.com/vllm-project/vllm/commit/2f58f985571fb6b33a8b1f2c8a27d5956b41adf6)** — Yongye Zhu
  `2f58f985` 2026-11-07T07:24:10Z
- **[[KV Offload] SimpleCPUOffload: one host buffer per rank, opt-in HugeTLB pages (#60130)](https://github.com/vllm-project/vllm/commit/7a07e2342e7e857133d3964bd4b7022103a5b030)** — Summer Yang
  `7a07e234` 2026-11-07T07:12:21Z
- **[[ROCm][CI] Fix skip condition for test_block_fp8.py  (#66205)](https://github.com/vllm-project/vllm/commit/f98575ec133dce6a759d1d0f27667eb246fc3123)** — rasmith
  `f98575ec` 2026-11-07T06:11:28Z
- **[[ROCm][CI] Set speculative_config in test_rocm_triton_attn_dsv4.py to fix (MI355 DPX) Attention Kernels Shard %N (#66217)](https://github.com/vllm-project/vllm/commit/204e2ece5f232baa9814ea7b5e451deda2087653)** — rasmith
  `204e2ece` 2026-11-07T06:03:53Z
- **[[Test][UMBP][8/10] Add standalone runtime coverage (#61796)](https://github.com/vllm-project/vllm/commit/1648fefab91204f3b15bf8062217316f07f2c4bb)** — CharlesWu
  `1648fefa` 2026-11-07T05:54:04Z
- **[[XPU] skip test_online_hadamard_target_wraps_resolved_method UT (#66125)](https://github.com/vllm-project/vllm/commit/5629b26146a39705c814ddae6a0f98321e14d2a6)** — Yan Ma
  `5629b261` 2026-11-07T04:05:52Z
- **[[Doc][Recipe] MiMo-V2.5-Pro: split fused --max-model-len base arg (#64588)](https://github.com/vllm-project/vllm/commit/d90f5573a3bc6e7f31051841552ee4b78a95d023)** — namgyu-yun
  `d90f5573` 2026-11-07T04:04:57Z
- **[[Recipe][ROCm] MiniMax-M3.1-Flash MI355X AgentX: nightly image and the opt-in optimizations (#66188)](https://github.com/vllm-project/vllm/commit/80e0cd68f812bfef5e6e7cffef3350daa01ff222)** — Fangzhou Ai
  `80e0cd68` 2026-11-07T02:52:56Z
- **[[CPU][Perf] Speed up GDN speculative-decode recurrent update (#64829)](https://github.com/vllm-project/vllm/commit/b4dffa5095ffa673ed1530524a503b168f5b611c)** — ekintel
  `b4dffa50` 2026-11-07T02:40:46Z
- **[[ROCm][Bugfix] Stop declaring the MXFP4 MoE intermediate padding to AITER (#61816)](https://github.com/vllm-project/vllm/commit/8176238c97f982ddc266e8bacb2aca9e11926af7)** — PerryZhang01
  `8176238c` 2026-11-07T01:01:48Z
- **[[BugFix][DP] Hash the parallel config before engine init for DP validation (#65792)](https://github.com/vllm-project/vllm/commit/5a763c36496034fc6e4afd75ffdaee005e7c9410)** — Maroon Ayoub
  `5a763c36` 2026-11-07T00:05:54Z
- **[[Perf] Vectorize silu_and_mul_per_block_quant for large inputs (#65791)](https://github.com/vllm-project/vllm/commit/c2a60420142f37a2a1613fbb94d41e0f4bf295e4)** — Misha Goin
  `c2a60420` 2026-11-06T23:48:11Z
- **[[Bugfix][MoE] Respect padded row strides in top-k routing (#57823)](https://github.com/vllm-project/vllm/commit/8b60d59945b2b651fd09a3d511a9dc0c392b06d3)** — Idder Ghanbaja
  `8b60d599` 2026-11-06T23:45:47Z
- **[[Bugfix][Frontend] Return 400 for images rejected by Pillow's bomb check (#60308)](https://github.com/vllm-project/vllm/commit/0bebae99a37d9af43e0ca2828108129c28272f7b)** — Sriraj
  `0bebae99` 2026-11-06T23:44:42Z
- **[[Bugfix][DP] Log cross-engine step statistics ordering at debug level (#64159)](https://github.com/vllm-project/vllm/commit/51300e32e4821b6e262a04edf4d473c238307ac7)** — Yifan Qiao
  `51300e32` 2026-11-06T23:43:58Z
- **[[Tests] Add TP2 PCP2 canonical mapping byte coverage (#62529)](https://github.com/vllm-project/vllm/commit/bc01b9b83d63783c1f169ed8998cbbda8e65c491)** — Nuriel32
  `bc01b9b8` 2026-11-06T23:43:24Z
- **[[Startup] Skip the CUDA graph memory profile when its estimate is disabled (#64622)](https://github.com/vllm-project/vllm/commit/74bb7664f798ec6fd53cad029375c948135fdbd8)** — Paul Emmerich
  `74bb7664` 2026-11-06T23:38:34Z
- **[[Rust Frontend] Kimi K3.1 chat renderer (Kimi K3.1 split 2/5) (#66028)](https://github.com/vllm-project/vllm/commit/ff05baf8ba821b1f15cc77dd93d4baf8833dcbf0)** — Jiangyun Zhu
  `ff05baf8` 2026-11-06T22:31:30Z
- **[[Model] Kimi K3.1: model files and NVIDIA specialized decode (Kimi K3.1 split 1/5) (#66030)](https://github.com/vllm-project/vllm/commit/9c345f09510012486291d2c2e07e3f3282220263)** — Jiangyun Zhu
  `9c345f09` 2026-11-06T22:31:14Z
- **[[Sleep Mode] Level 2 discards only parameters (#66010)](https://github.com/vllm-project/vllm/commit/f8b116409c4cdf02cee4bb7eca8e6bb9780c2e35)** — aoshen02
  `f8b11640` 2026-11-06T20:33:59Z

<details><summary>📄 diff: [Perf][MiniMax-M3] Warm up DeepGEMM NVFP4 MegaMoE kernels be</summary>

```diff
diff --git a/tests/model_executor/test_deep_gemm_warmup.py b/tests/model_executor/test_deep_gemm_warmup.py
index 26c410d1bf99..9337ef08a225 100644
--- a/tests/model_executor/test_deep_gemm_warmup.py
+++ b/tests/model_executor/test_deep_gemm_warmup.py
@@ -10,6 +10,10 @@
 from vllm.model_executor.kernels.linear.scaled_mm.deep_gemm import (
     DeepGemmFp8BlockScaledMMKernel,
 )
+from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
+    MEGA_MOE_MXFP4,
+    MEGA_MOE_NVFP4,
+)
 from vllm.model_executor.warmup import deep_gemm_warmup
 
 
@@ -93,14 +97,16 @@ def test_kernel_registers_itself_as_warmup_provider(is_bmm) -> None:
 
 def test_mega_moe_predicate_matches_only_native_mega_moe_layers(monkeypatch) -> None:
     """Without the MegaMoE model modules loaded, nothing matches and nothing is
-    imported; with them loaded, only DeepSeek-V4 and Kimi K3 MegaMoE layers match."""
+    imported; with them loaded, only DeepSeek-V4, Kimi K3 and MiniMax M3 MegaMoE
+    layers match."""
     name = "vllm.models.deepseek_v4.nvidia.model"
     kimi_name = "vllm.models.kimi_k3.nvidia.model"
-    monkeypatch.delitem(sys.modules, name, raising=False)
-    monkeypatch.delitem(sys.modules, kimi_name, raising=False)
+    minimax_name = "vllm.models.minimax_m3.nvidia.model"
+    for model_module in (name, kimi_name, minimax_name):
+        monkeypatch.delitem(sys.modules, model_module, raising=False)
     other = SimpleNamespace(use_native_mega_moe=True, use_mega_moe=True)
     assert not deep_gemm_warmup._mega_moe_may_use_deep_gemm(other)
-    assert name not in sys.modules and kimi_name not in sys.modules
+    assert not {name, kimi_name, minimax_name} & set(sys.modules)
 
     moe_cls = type("DeepseekV4MoE", (), {})
     monkeypatch.setitem(sys.modules, name, SimpleNamespace(DeepseekV4MoE=moe_cls))
@@ -117,6 +123,38 @@ def test_mega_moe_predicate_matches_only_native_mega_moe_layers(monkeypatch) ->
     assert deep_gemm_warmup._mega_moe_may_use_deep_gemm(kimi_mega)
     assert not deep_gemm_warmup._mega_moe_may_use_deep_gemm(kimi_fused)
 
+    minimax_cls = type("MiniMaxM3MoE", (), {})
+    monkeypatch.setitem(
+        sys.modules, minimax_name, SimpleNamespace(MiniMaxM3MoE=minimax_cls)
+    )
+    minimax_mega, minimax_fused = minimax_cls(), minimax_cls()
+    minimax_mega.use_mega_moe, minimax_fused.use_mega_moe = True, False
+    assert deep_gemm_warmup._mega_moe_may_use_deep_gemm(minimax_mega)
+    assert not deep_gemm_warmup._mega_moe_may_use_deep_gemm(minimax_fused)
+
+
+@pytest.mark.parametrize(
+    ("weight_format", "expected"),
+    [(MEGA_MOE_MXFP4, [2, 5]), (MEGA_MOE_NVFP4, [1, 2, 3, 5])],
+)
+def test_mega_moe_m_values_cover_nvfp4_launch_variants(
+    monkeypatch, weight_format, expected
+) -> None:
+    """NVFP4 switches from the decode-shaped to the bulk kernel inside one BLOCK_M,
+    so both ends are warmed; MX kernels depend on BLOCK_M alone."""
+    monkeypatch.setattr(
+        deep_gemm_warmup, "_import_deep_gemm", lambda: SimpleNamespace()
+    )
+    module = SimpleNamespace(
+        gate=SimpleNamespace(),
+        experts=SimpleNamespace(
+            get_block_m=lambda num_tokens: 64 if num_tokens <= 2 else 128,
+            weight_format=weight_format,
+        ),
+        use_sequence_parallel=False,
+    )
+    assert deep_gemm_warmup._get_fp8_fp4_mega_moe_m_values(module, 5) == expected
+
 
 def test_mega_moe_m_values_reach_mega_gate(monkeypatch) -> None:
     """Each bf16_mega_gate config is warmed with an M above the GateLinear
@@ -128,7 +166,11 @@ def test_mega_moe_m_values_reach_mega_gate(monkeypatch) -> None:
     monkeypatch.setattr(deep_gemm_warmup, "_mega_moe_uses_mega_gate", lambda _: True)
     module = SimpleNamespace(
         gate=SimpleNamespace(input_size=4096, output_size=128),
-        experts=SimpleNamespace(top_k=6, get_block_m=lambda num_tokens: 16),
... (truncated, 154 more lines)
```
</details>

<details><summary>📄 diff: [KV Offload] SimpleCPUOffload: one host buffer per rank, opt</summary>

```diff
diff --git a/docs/features/kv_offloading_usage.md b/docs/features/kv_offloading_usage.md
index a7c263007d13..a316f67116d6 100644
--- a/docs/features/kv_offloading_usage.md
+++ b/docs/features/kv_offloading_usage.md
@@ -469,6 +469,32 @@ Example (OpenAI-compatible completions request):
 }
 ```
 
+## Huge pages for `SimpleCPUOffloadConnector`
+
+With `kv_offload_backend="cpu"`, set `cpu_hugepage_block_size` to `"2MB"` or
+`"1GB"` in `kv_connector_extra_config` to back the CPU KV caches with Linux
+HugeTLB pages (Linux 5.14 or newer):
+
+```json
+{
+  "kv_connector": "SimpleCPUOffloadConnector",
+  "kv_role": "kv_both",
+  "kv_connector_extra_config": {
+    "cpu_bytes_to_use_per_rank": 137438953472,
+    "cpu_hugepage_block_size": "1GB"
+  }
+}
+```
+
+Reserve enough pages on the host for every rank, rounding each rank's cache
+size up to a whole page. Containers also need a matching hugetlb allowance, for
+example Kubernetes `hugepages-1Gi`. If the pool or the limit is too small,
+startup fails; there is no fallback to ordinary pages.
+
+Without this key, the caches use ordinary pages, and `THP_MEM_ALLOC_ENABLE=1`
+requests transparent huge pages. In both cases, the memory follows the worker's
+NUMA policy, for example from `--numa-bind`.
+
 ## Explicit pool sizing for `SimpleCPUOffloadConnector`
 
 `SimpleCPUOffloadConnector` accepts an optional `num_cpu_blocks` in
diff --git a/tests/v1/simple_kv_offload/conftest.py b/tests/v1/simple_kv_offload/conftest.py
new file mode 100644
index 000000000000..2619868c9fa4
--- /dev/null
+++ b/tests/v1/simple_kv_offload/conftest.py
@@ -0,0 +1,31 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import pytest
+
+
+def _free_hugepages(page: int) -> int:
+    path = f"/sys/kernel/mm/hugepages/hugepages-{page >> 10}kB/free_hugepages"
+    try:
+        with open(path) as f:
+            return int(f.read())
+    except OSError:
+        return -1
+
+
+@pytest.fixture
+def free_hugepages():
+    """Free pages in the host's HugeTLB pool of a size; -1 if it has none."""
+    return _free_hugepages
+
+
+@pytest.fixture
+def hugepage_with_free():
+    """Return a HugeTLB page size with at least ``n`` free pages, else skip."""
+
+    def pick(n: int) -> int:
+        pages = [p for p in (2 << 20, 1 << 30) if _free_hugepages(p) >= n]
+        if not pages:
+            pytest.skip(f"needs {n} free hugepages of some size")
+        return pages[0]
+
+    return pick
diff --git a/tests/v1/simple_kv_offload/test_host_buffer.py b/tests/v1/simple_kv_offload/test_host_buffer.py
new file mode 100644
index 000000000000..963d53d4f7ae
--- /dev/null
+++ b/tests/v1/simple_kv_offload/test_host_buffer.py
@@ -0,0 +1,48 @@
... (truncated, 763 more lines)
```
</details>

<details><summary>📄 diff: [ROCm][CI] Fix skip condition for test_block_fp8.py  (#66205</summary>

```diff
diff --git a/tests/kernels/moe/test_block_fp8.py b/tests/kernels/moe/test_block_fp8.py
index 3c9018aa8e79..0f8c4be99d31 100644
--- a/tests/kernels/moe/test_block_fp8.py
+++ b/tests/kernels/moe/test_block_fp8.py
@@ -315,8 +315,11 @@ def test_w8a8_block_fp8_fused_moe(
 @torch.inference_mode()
 def test_block_fp8_up_and_down_tma_matches_pointer(M, modular, workspace_init):
     if not (
-        current_platform.is_device_capability_family(90)
-        or current_platform.is_device_capability_family(100)
+        current_platform.is_cuda()
+        and (
+            current_platform.is_device_capability_family(90)
+            or current_platform.is_device_capability_family(100)
+        )
     ):
         pytest.skip("This block FP8 TMA test requires SM90 or SM100")
 

```
</details>

<details><summary>📄 diff: [ROCm][CI] Set speculative_config in test_rocm_triton_attn_d</summary>

```diff
diff --git a/tests/kernels/attention/test_rocm_triton_attn_dsv4.py b/tests/kernels/attention/test_rocm_triton_attn_dsv4.py
index 0c37a127c5a1..5aec1dbc5ef2 100644
--- a/tests/kernels/attention/test_rocm_triton_attn_dsv4.py
+++ b/tests/kernels/attention/test_rocm_triton_attn_dsv4.py
@@ -1880,6 +1880,8 @@ def test_dsv4_adaptive_mla_swa_metadata_graph_replay(monkeypatch) -> None:
         parallel_drafting=False,
         enable_adaptive_verification=True,
         use_dspark=lambda: True,
+        method="dspark",
+        speculator_config=SimpleNamespace(speculator_class=None),
     )
     mla_spec = MLAAttentionSpec(
         block_size=block_size,
@@ -2129,6 +2131,8 @@ def test_dsv41_adaptive_mla_swa_metadata_graph_replay(monkeypatch) -> None:
         parallel_drafting=False,
         enable_adaptive_verification=True,
         use_dspark=lambda: True,
+        method="dspark",
+        speculator_config=SimpleNamespace(speculator_class=None),
     )
     mla_spec = MLAAttentionSpec(
         block_size=block_size,

```
</details>

<details><summary>📄 diff: [Test][UMBP][8/10] Add standalone runtime coverage (#61796)</summary>

```diff
diff --git a/.buildkite/test-amd.yaml b/.buildkite/test-amd.yaml
index 4e2e19a1884f..a0c4954d4a97 100644
--- a/.buildkite/test-amd.yaml
+++ b/.buildkite/test-amd.yaml
@@ -5884,7 +5884,10 @@ steps:
   - docker/Dockerfile.rocm_base
   commands:
   - export VLLM_WORKER_MULTIPROC_METHOD=spawn
-  - pytest -v -s -rs "v1/kv_connector/umbp_integration/test_mori_token_e2e.py::test_restore_matches_recompute[tp2]"
+  - >-
+    pytest -v -s -rs
+    "v1/kv_connector/umbp_integration/test_mori_token_e2e.py::test_restore_matches_recompute[tp2]"
+    "v1/kv_connector/umbp_integration/test_mori_standalone_gpu.py::test_objects_outlive_the_writer_and_restore_on_another_gpu"
 
 - label: ":amd: (MI355 DPX) V1 Attention Shard %N" # TBD
   timeout_in_minutes: 125
diff --git a/.buildkite/test_areas/misc.yaml b/.buildkite/test_areas/misc.yaml
index 4be063988c90..2bc1e82acfef 100644
--- a/.buildkite/test_areas/misc.yaml
+++ b/.buildkite/test_areas/misc.yaml
@@ -221,7 +221,14 @@ steps:
   - docker/Dockerfile.rocm_base
   commands:
   - export VLLM_WORKER_MULTIPROC_METHOD=spawn
-  - pytest -v -s -rs v1/kv_connector/unit/test_umbp_async_lookup.py v1/kv_connector/unit/test_umbp_core.py v1/kv_connector/unit/test_umbp_embedded.py v1/kv_connector/unit/test_umbp_events.py v1/kv_connector/unit/test_umbp_hybrid.py
+  - >-
+    pytest -v -s -rs
+    v1/kv_connector/unit/test_umbp_async_lookup.py
+    v1/kv_connector/unit/test_umbp_core.py
+    v1/kv_connector/unit/test_umbp_embedded.py
+    v1/kv_connector/unit/test_umbp_events.py
+    v1/kv_connector/unit/test_umbp_hybrid.py
+    v1/kv_connector/unit/test_umbp_standalone.py
 
 - label: ":amd: (MI355 DPX) UMBP Connector Integration"
   key: amd-umbp-connector-integration-mi355
@@ -265,7 +272,10 @@ steps:
   - docker/Dockerfile.rocm_base
   commands:
   - export VLLM_WORKER_MULTIPROC_METHOD=spawn
-  - pytest -v -s -rs "v1/kv_connector/umbp_integration/test_mori_token_e2e.py::test_restore_matches_recompute[tp2]"
+  - >-
+    pytest -v -s -rs
+    "v1/kv_connector/umbp_integration/test_mori_token_e2e.py::test_restore_matches_recompute[tp2]"
+    "v1/kv_connector/umbp_integration/test_mori_standalone_gpu.py::test_objects_outlive_the_writer_and_restore_on_another_gpu"
 
 # Single-node MoRIIO P/D on dummy weights, WRITE and READ mode: prefill and
 # decode on separate GPUs over xGMI, compared exactly against a direct serve.
diff --git a/tests/v1/kv_connector/umbp_integration/test_mori_standalone_gpu.py b/tests/v1/kv_connector/umbp_integration/test_mori_standalone_gpu.py
new file mode 100644
index 000000000000..a2e68d99ce0f
--- /dev/null
+++ b/tests/v1/kv_connector/umbp_integration/test_mori_standalone_gpu.py
@@ -0,0 +1,350 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""StandaloneRuntime against a real umbp_standalone_server on ROCm GPUs."""
+
+import multiprocessing as mp
+import os
+import signal
+import subprocess
+import time
+from concurrent.futures import ThreadPoolExecutor
+from pathlib import Path
+
+import pytest
+import torch
+
+import vllm.envs as envs
+from tests.v1.kv_connector.rocm_pd_accuracy_utils import _REVISIONS, QWEN_MODEL
+from tests.v1.kv_connector.umbp_integration.test_mori_token_e2e import (
+    BLOCK_SIZE,
+    PARAMS,
+    _assert_same_outputs,
+    _external_tokens,
+    _prompt,
+    _reset_gpu_prefix_cache,
+)
+from tests.v1.kv_connector.umbp_test_utils import run_in_new_process
... (truncated, 1691 more lines)
```
</details>

<details><summary>📄 diff: [XPU] skip test_online_hadamard_target_wraps_resolved_method</summary>

```diff
diff --git a/tests/quantization/test_online.py b/tests/quantization/test_online.py
index 1ad7f0a4e2cb..8e7fcf7d0deb 100644
--- a/tests/quantization/test_online.py
+++ b/tests/quantization/test_online.py
@@ -308,6 +308,10 @@ def test_online_mxfp4_activation_key_is_honoured_or_rejected(
         assert computed_key == activation_quant_key
 
 
+@pytest.mark.skipif(
+    current_platform.is_xpu(),
+    reason="MXFP4 linear kernels on XPU only support dynamic activation.",
+)
 def test_online_hadamard_target_wraps_resolved_method(default_vllm_config) -> None:
     """`hadamard_size` on a `targets` spec wraps the method built with the
     resolved activation key in `InputHadamardLinearMethod`."""

```
</details>

<details><summary>📄 diff: [Doc][Recipe] MiMo-V2.5-Pro: split fused --max-model-len bas</summary>

```diff
diff --git a/recipes/models/XiaomiMiMo/MiMo-V2.5-Pro.yaml b/recipes/models/XiaomiMiMo/MiMo-V2.5-Pro.yaml
index f252177c4d3b..4e0833df385d 100644
--- a/recipes/models/XiaomiMiMo/MiMo-V2.5-Pro.yaml
+++ b/recipes/models/XiaomiMiMo/MiMo-V2.5-Pro.yaml
@@ -21,7 +21,8 @@ model:
   context_length: 1048576
   base_args:
     - "--trust-remote-code"
-    - "--max-model-len auto"
+    - "--max-model-len"
+    - "auto"
     - "--generation-config"
     - "vllm"
   base_env: {}
@@ -63,6 +64,8 @@ variants:
 compatible_strategies:
   - single_node_tp
   - single_node_tep
+  - multi_node_tp
+  - multi_node_tep
   - pd_cluster
 
 hardware_overrides: {}

```
</details>

<details><summary>📄 diff: [Recipe][ROCm] MiniMax-M3.1-Flash MI355X AgentX: nightly ima</summary>

```diff
diff --git a/recipes/models/MiniMaxAI/MiniMax-M3.1-Flash.yaml b/recipes/models/MiniMaxAI/MiniMax-M3.1-Flash.yaml
index 2a7541260832..cedf59c21fc3 100644
--- a/recipes/models/MiniMaxAI/MiniMax-M3.1-Flash.yaml
+++ b/recipes/models/MiniMaxAI/MiniMax-M3.1-Flash.yaml
@@ -1404,9 +1404,9 @@ presets:
     strategy: single_node_tp
     variants: [mxfp4]
     workload: "Agentic (InferenceX AgentX)"
-    description: "One TP4 engine on four of the node's eight MI355X GPUs, serving the Quark MXFP4 checkpoint: seven-token DSpark with mono decode and an FP8 KV cache up to concurrency 8, seven-token DSpark with DBO and an NVFP4 KV cache at 16 and 32, and three-token DSpark with DBO and a host-DRAM KV tier at 64."
-    source: { label: "InferenceX #4671", href: "https://github.com/SemiAnalysisAI/InferenceX/pull/4671", key: "minimaxm31flash-fp4-mi355x-vllm-agentic-dspark", revision: a328b443582c45e45eef7f0c4f133f3e173f0211 }
-    image: "vllm/vllm-openai-rocm:minimax-m31-flash@sha256:ae8681212aa5291d8c27f06e90016435e3fca144a68fd5c439231f6475f74569"
+    description: "One TP4 engine on four of the node's eight MI355X GPUs, serving the Quark MXFP4 checkpoint: seven-token DSpark with mono decode and an FP8 KV cache up to concurrency 8, seven-token DSpark with an NVFP4 KV cache and breakable CUDA graphs at 16 and 32, and three-token DSpark with a host-DRAM KV tier at 64, on a nightly image with the MiniMax-M3.1 ROCm optimizations that need an opt-in switch enabled."
+    source: { label: "InferenceX #4866", href: "https://github.com/SemiAnalysisAI/InferenceX/pull/4866", key: "minimaxm31flash-fp4-mi355x-vllm-agentic-dspark", revision: ce4079c48b85472cc4be60f40c0fa1f7f067e8e9 }
+    image: "vllm/vllm-openai-rocm:nightly-4fab5a07475822c6eaa971dceb2e2220d0446808@sha256:6f126f27a3b58959da3c16076248fc99eb31e26d0a3ae9f8563bda7317d6a7e6"
     tp: 4
     select:
       features: { text_only: true, spec_decoding: true }
@@ -1417,11 +1417,15 @@ presets:
       # the DSpark drafter kernels (seven-token drafts; c64's three-token draft keeps
       # the stock draft layers)
       VLLM_ROCM_MINIMAX_M3_MONO_DRAFTER: "1"
+      # the FP8 indexer of the mono points bypasses it
+      VLLM_ROCM_PARALLEL_INDEX_TOPK: "1"
+      VLLM_USE_RUST_FRONTEND: "1"
+      # the draft LM head's skinny GEMM is 5x slower at exactly 5 requests
+      VLLM_ROCM_USE_SKINNY_GEMM: "0"
     json_args:
       "--speculative-config": { num_speculative_tokens: 7, rejection_sample_method: block }
-    # The DBO points pass the DBO flags directly: the DBO / SP prefill feature's
-    # modes set their own thresholds (8192, or 16384 with SP on the default
-    # checkpoint only), and this lane runs the 16384-token threshold without SP.
+    # json_args cannot add --compilation-config (mono_decode owns it). The capture list is
+    # the V2 runner's deprecated alias for the prefill and decode graph sizes.
     points:
       c1:
         label: "1 (mono)"
@@ -1444,20 +1448,20 @@ presets:
         select: { features: { mono_decode: mono } }
         args: ["--max-num-seqs", "16", "--max-num-batched-tokens", "8192"]
       c16:
-        label: "16 (DBO)"
+        label: "16 (graphs)"
         concurrency: 16
-        args: ["--max-num-seqs", "32", "--max-num-batched-tokens", "32768", "--gpu-memory-utilization", "0.85", "--enable-dbo", "--dbo-prefill-token-threshold", "16384", "--dbo-decode-token-threshold", "32768"]
-        env: { VLLM_USE_V2_MODEL_RUNNER: "1", DEBUG_HIP_DYNAMIC_QUEUES: "0" }
+        args: ["--max-num-seqs", "32", "--max-num-batched-tokens", "32768", "--gpu-memory-utilization", "0.85", "--compilation-config", '{"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]}']
+        env: { VLLM_USE_V2_MODEL_RUNNER: "1", DEBUG_HIP_DYNAMIC_QUEUES: "0", VLLM_USE_BREAKABLE_CUDAGRAPH: "1" }
       c32:
-        label: "32 (DBO)"
+        label: "32 (graphs)"
         concurrency: 32
-        args: ["--max-num-seqs", "64", "--max-num-batched-tokens", "32768", "--gpu-memory-utilization", "0.85", "--enable-dbo", "--dbo-prefill-token-threshold", "16384", "--dbo-decode-token-threshold", "32768"]
-        env: { VLLM_USE_V2_MODEL_RUNNER: "1", DEBUG_HIP_DYNAMIC_QUEUES: "0" }
+        args: ["--max-num-seqs", "64", "--max-num-batched-tokens", "32768", "--gpu-memory-utilization", "0.85", "--compilation-config", '{"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]}']
+        env: { VLLM_USE_V2_MODEL_RUNNER: "1", DEBUG_HIP_DYNAMIC_QUEUES: "0", VLLM_USE_BREAKABLE_CUDAGRAPH: "1" }
       c64:
-        label: "64 (DBO, KV offload)"
+        label: "64 (KV offload)"
         concurrency: 64
         select: { kv_offload: simple }
-        args: ["--max-num-seqs", "128", "--max-num-batched-tokens", "32768", "--gpu-memory-utilization", "0.9", "--enable-dbo", "--dbo-prefill-token-threshold", "16384", "--dbo-decode-token-threshold", "32768"]
+        args: ["--max-num-seqs", "128", "--max-num-batched-tokens", "32768", "--gpu-memory-utilization", "0.9"]
         env: { VLLM_USE_V2_MODEL_RUNNER: "1", DEBUG_HIP_DYNAMIC_QUEUES: "0", PYTHONHASHSEED: "42" }
         json_args:
           "--speculative-config": { num_speculative_tokens: 3 }
@@ -1465,43 +1469,69 @@ presets:
     runbook: |
       The MI355X AgentX lane in [SemiAnalysis InferenceX](https://inferencex.semianalysis.com/)
       (`minimaxm31flash-fp4-mi355x-vllm-agentic-dspark`,
-      [SemiAnalysisAI/InferenceX#4671](https://github.com/SemiAnalysisAI/InferenceX/pull/4671)),
+      [SemiAnalysisAI/InferenceX#4866](https://github.com/SemiAnalysisAI/InferenceX/pull/4866)),
       on the Quark MXFP4 checkpoint with `--max-num-seqs` at twice the concurrency
       (c64 admits 128). Throughput runs replaced block verification with synthetic
       rejection at the committed golden DSpark acceptance length (4.04 for seven
       tokens, 2.88 for three); the commands above keep block verification.
 
-      - **Concurrency 1-8** run mono decode with an FP8 KV cache and no DBO: a decode
-        step of up to 16 tokens runs each layer as one fused kernel, and prefill steps
-        stay at 8192 tokens.
-      - **Concurrency 16 and 32** run seven-token DSpark with DBO and an NVFP4 KV cache
... (truncated, 84 more lines)
```
</details>

<details><summary>📄 diff: [CPU][Perf] Speed up GDN speculative-decode recurrent update</summary>

```diff
diff --git a/benchmarks/kernels/cpu/benchmark_cpu_gdn_spec.py b/benchmarks/kernels/cpu/benchmark_cpu_gdn_spec.py
new file mode 100644
index 000000000000..c58a94f171b8
--- /dev/null
+++ b/benchmarks/kernels/cpu/benchmark_cpu_gdn_spec.py
@@ -0,0 +1,142 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Benchmark the CPU GDN recurrent-update kernels used for decode and
+speculative-decode verify (defaults match Qwen3.6-35B-A3B)."""
+
+import math
+import time
+
+import numpy as np
+import torch
+
+import vllm._custom_ops as ops
+from vllm.utils.argparse_utils import FlexibleArgumentParser
+from vllm.utils.torch_utils import set_random_seed
+
+STATE_DTYPES = {
+    "float32": torch.float32,
+    "bfloat16": torch.bfloat16,
+    "float16": torch.float16,
+}
+
+
+@torch.inference_mode()
+def main(args) -> None:
+    set_random_seed(args.seed)
+    batch_size = args.batch_size
+    q_len = args.num_spec_tokens + 1 if args.mode == "spec" else 1
+    num_tokens = batch_size * q_len
+    hk, hv, dk, dv = args.num_k_heads, args.num_v_heads, args.k_dim, args.v_dim
+    state_dtype = STATE_DTYPES[args.state_dtype]
+    num_cols = q_len
+
+    q = torch.randn(num_tokens, hk, dk, dtype=torch.bfloat16)
+    k = torch.randn(num_tokens, hk, dk, dtype=torch.bfloat16)
+    v = torch.randn(num_tokens, hv, dv, dtype=torch.bfloat16)
+    a = torch.randn(num_tokens, hv, dtype=torch.bfloat16)
+    b = torch.randn(num_tokens, hv, dtype=torch.bfloat16)
+    A_log = torch.rand(hv, dtype=torch.float32)
+    dt_bias = torch.rand(hv, dtype=torch.bfloat16)
+
+    # In serving, ~10 GB of weights stream between two visits to a layer's
+    # state, so the state is always cold. Rotate through a footprint far larger
+    # than the LLC so reads and write-backs of the state hit DRAM here as well.
+    num_slots = batch_size * num_cols
+    pool_bytes = num_slots * hv * dk * dv * state_dtype.itemsize
+    num_pools = max(
+        args.num_layers, math.ceil(args.footprint_gb * 1024**3 / pool_bytes)
+    )
+    pools = [
+        torch.randn(num_slots, hv, dk, dv).to(state_dtype) for _ in range(num_pools)
+    ]
+    cu_seqlens = torch.arange(batch_size + 1, dtype=torch.int32) * q_len
+    spec_indices = torch.arange(num_slots, dtype=torch.int32).view(batch_size, num_cols)
+    num_accepted = torch.full((batch_size,), max(1, q_len // 2), dtype=torch.int32)
+    state_indices = torch.arange(batch_size, dtype=torch.int32)
+    decode_cu_seqlens = torch.arange(batch_size + 1, dtype=torch.int32)
+
+    def run_layer(pool: torch.Tensor) -> None:
+        if args.mode == "spec":
+            ops.fused_sigmoid_gating_delta_rule_update_spec_cpu(
+                A_log=A_log,
+                dt_bias=dt_bias,
+                q=q,
+                k=k,
+                v=v,
+                a=a,
+                b=b,
+                initial_state_source=pool,
+                spec_state_indices=spec_indices,
+                num_accepted_tokens=num_accepted,
+                cu_seqlens=cu_seqlens,
+                use_qk_l2norm_in_kernel=True,
+            )
+        else:
... (truncated, 471 more lines)
```
</details>

<details><summary>📄 diff: [ROCm][Bugfix] Stop declaring the MXFP4 MoE intermediate pad</summary>

```diff
diff --git a/tests/kernels/moe/test_ocp_mx_moe.py b/tests/kernels/moe/test_ocp_mx_moe.py
index a799fc6c39d2..851da4e26ca3 100644
--- a/tests/kernels/moe/test_ocp_mx_moe.py
+++ b/tests/kernels/moe/test_ocp_mx_moe.py
@@ -2083,21 +2083,28 @@ class MockLayer:
 
 
 @pytest.mark.parametrize("intermediate_size_unpadded", [640, 704])
+@pytest.mark.parametrize(
+    ("backend_name", "activation_name"),
+    [("AITER_MXFP4_MXFP4", "SILU"), ("AITER_MXFP4_BF16", "SWIGLUOAI")],
+)
 @pytest.mark.skipif(not ROCM_GFX950, reason="Requires GFX950")
 @pytest.mark.skipif(not ROCM_AITER_AVAILABLE, reason="Requires AITER")
 @torch.inference_mode()
-def test_aiter_mxfp4_w4a4_moe_pad_masks_only_padding(
+def test_aiter_mxfp4_moe_pad_masks_only_padding(
     intermediate_size_unpadded: int,
+    backend_name: str,
+    activation_name: str,
     monkeypatch: pytest.MonkeyPatch,
 ):
-    """At MoE tp_size == 1 the W4A4 MoE masks the padding and nothing else.
+    """At MoE tp_size == 1 the MXFP4 MoE masks the padding and nothing else.
 
     The experts are stored 768 wide with zeros past the real width, so the
     layer computes the same function whether the kernel is told that the tail
     is padding or not. A pad that is doubled twice (once by vLLM, once inside
     AITER for the fused gate/up weight) masks live intermediate rows as well.
-    640 stored at 768 is Qwen3.8 Flash Next's shape at TP1 and under expert
-    parallelism before the W4A4 intermediate was aligned to 128.
+    640 stored at 768 is Qwen3.8 Flash Next's W4A4 shape at TP1 and under
+    expert parallelism before the W4A4 intermediate was aligned to 128. The
+    W4A16 case uses gpt-oss's interleaved, biased layout.
     """
     import vllm.distributed.parallel_state as ps
     from vllm.config import VllmConfig, set_current_vllm_config
@@ -2120,7 +2127,9 @@ def test_aiter_mxfp4_w4a4_moe_pad_masks_only_padding(
     monkeypatch.setattr(ps, "_TP", types.SimpleNamespace(world_size=1))
     monkeypatch.setattr(rocm_aiter_ops, "_AITER_ENABLED", True)
 
-    backend = Mxfp4MoeBackend.AITER_MXFP4_MXFP4
+    backend = Mxfp4MoeBackend[backend_name]
+    activation = MoEActivation[activation_name]
+    gpt_oss = backend == Mxfp4MoeBackend.AITER_MXFP4_BF16
     experts_cls = backend_to_kernel_cls(backend)[0]
     num_experts, topk, num_tokens = 8, 2, 64
     hidden_size, intermediate_size = 512, 768
@@ -2130,12 +2139,24 @@ def test_aiter_mxfp4_w4a4_moe_pad_masks_only_padding(
     w13 = torch.randn(
         num_experts, 2 * intermediate_size, hidden_size, dtype=dtype, device=device
     )
-    w13[:, intermediate_size_unpadded:intermediate_size] = 0
-    w13[:, intermediate_size + intermediate_size_unpadded :] = 0
     w2 = torch.randn(
         num_experts, hidden_size, intermediate_size, dtype=dtype, device=device
     )
     w2[..., intermediate_size_unpadded:] = 0
+    w13_bias = w2_bias = None
+    if gpt_oss:
+        # Checkpoint layout: gate/up rows interleaved, padding at the end.
+        w13[:, 2 * intermediate_size_unpadded :] = 0
+        w13_bias = torch.randn(
+            num_experts, 2 * intermediate_size, dtype=torch.float32, device=device
+        )
+        w13_bias[:, 2 * intermediate_size_unpadded :] = 0
+        w2_bias = torch.randn(
+            num_experts, hidden_size, dtype=torch.float32, device=device
+        )
+    else:
+        w13[:, intermediate_size_unpadded:intermediate_size] = 0
+        w13[:, intermediate_size + intermediate_size_unpadded :] = 0
     w13_q, w13_scale = dynamic_mxfp4_quant(w13.reshape(-1, hidden_size))
     w2_q, w2_scale = dynamic_mxfp4_quant(w2.reshape(-1, intermediate_size))
     w13_q = w13_q.reshape(num_experts, 2 * intermediate_size, -1)
@@ -2144,7 +2165,7 @@ def test_aiter_mxfp4_w4a4_moe_pad_masks_only_padding(
     w2_scale = w2_scale.reshape(num_experts, hidden_size, -1)
 
     layer = types.SimpleNamespace(w13_input_scale=None, w2_input_scale=None)
-    w13_q, w2_q, w13_scale, w2_scale, _, _ = (
... (truncated, 157 more lines)
```
</details>

<details><summary>📄 diff: [BugFix][DP] Hash the parallel config before engine init for</summary>

```diff
diff --git a/tests/v1/engine/test_startup_watch_processes.py b/tests/v1/engine/test_startup_watch_processes.py
index d4d57c305f17..9cabcaad2aa2 100644
--- a/tests/v1/engine/test_startup_watch_processes.py
+++ b/tests/v1/engine/test_startup_watch_processes.py
@@ -7,12 +7,14 @@
 from threading import Event
 from types import SimpleNamespace
 
+import msgspec
 import pytest
 import torch
 import zmq
 
 import vllm.platforms as platforms
-from vllm.config import WatchdogConfig
+from vllm.config import ParallelConfig, WatchdogConfig
+from vllm.config.utils import hash_config
 from vllm.utils import torch_utils
 from vllm.v1.engine import core as core_module
 from vllm.v1.engine import utils as engine_utils
@@ -21,6 +23,7 @@
     CoreEngine,
     CoreEngineLaunch,
     CoreEngineProcManager,
+    CoreEngineState,
     EngineZmqAddresses,
     wait_for_engine_startup,
 )
@@ -341,3 +344,82 @@ def test_wait_for_engine_startup_reports_watched_process_exit():
         exc_info.value
     )
     assert "Failed frontend proc(s): {'RustFrontend': 1}" in str(exc_info.value)
+
+
+def _engine_config_hash(**overrides) -> str:
+    """Hash of an engine's ParallelConfig after engine initialization."""
+    config = ParallelConfig(data_parallel_size=2, data_parallel_size_local=2)
+    for name, value in overrides.items():
+        setattr(config, name, value)
+    return hash_config(config)
+
+
+def _run_dp_startup_handshake(engine_hashes: list[str]) -> list[CoreEngine]:
+    """Drive wait_for_engine_startup with HELLO and READY from each engine."""
+    parallel_config = ParallelConfig(
+        data_parallel_size=len(engine_hashes),
+        data_parallel_size_local=len(engine_hashes),
+    )
+    core_engines = [CoreEngine(index=i) for i in range(len(engine_hashes))]
+    launch = CoreEngineLaunch(
+        engine_manager=None,
+        coordinator=None,
+        addresses=EngineZmqAddresses(inputs=[], outputs=[]),
+        tensor_queue=None,
+    )
+    ctx = zmq.Context()
+    handshake_socket = ctx.socket(zmq.ROUTER)
+    handshake_socket.bind("inproc://dp-startup-handshake")
+    engine_sockets = []
+    try:
+        for engine, config_hash in zip(core_engines, engine_hashes):
+            engine_socket = ctx.socket(zmq.DEALER)
+            engine_socket.setsockopt(zmq.IDENTITY, engine.identity)
+            engine_socket.connect("inproc://dp-startup-handshake")
+            engine_sockets.append(engine_socket)
+            for msg in (
+                {"status": "HELLO", "local": True, "headless": False},
+                {
+                    "status": "READY",
+                    "local": True,
+                    "headless": False,
+                    "parallel_config_hash": config_hash,
+                },
+            ):
+                engine_socket.send(msgspec.msgpack.encode(msg))
+        wait_for_engine_startup(
+            handshake_socket,
+            core_engines,
+            parallel_config,
+            coordinated_dp=True,
... (truncated, 81 more lines)
```
</details>

<details><summary>📄 diff: [Perf] Vectorize silu_and_mul_per_block_quant for large inpu</summary>

```diff
diff --git a/csrc/quantization/fused_kernels/fused_silu_mul_block_quant.cu b/csrc/quantization/fused_kernels/fused_silu_mul_block_quant.cu
index bd50f43bf49f..89d1d3e6b4e1 100644
--- a/csrc/quantization/fused_kernels/fused_silu_mul_block_quant.cu
+++ b/csrc/quantization/fused_kernels/fused_silu_mul_block_quant.cu
@@ -4,6 +4,7 @@
 #include "../../torch_utils.h"
 
 #include "../../dispatch_utils.h"
+#include "../../core/cuda_compat.h"
 #include "quant_conversions.cuh"
 
 namespace vllm {
@@ -102,6 +103,84 @@ __global__ void silu_and_mul_per_block_quant_kernel(
       vllm::ScaledQuant<scalar_out_t, false>::quant_fn(result, group_scale);
 }
 
+// Vectorized variant: each group is handled by group_size / 8 lanes of a
+// warp, each loading 8 contiguous 16-bit elements of gate and up (16B loads)
+// and storing 8 quantized bytes. The group max is reduced with warp shuffles,
+// so there is no shared memory or __syncthreads. Requires 16B-aligned input
+// and 8B-aligned output rows. Same arithmetic as the kernel above.
+template <typename scalar_t, typename scalar_out_t, bool is_scale_transposed,
+          int32_t group_size>
+__global__ void silu_and_mul_per_block_quant_vec_kernel(
+    scalar_out_t* __restrict__ out, float* __restrict__ scales,
+    scalar_t const* __restrict__ input, float const* scale_ub,
+    int32_t const hidden_size, int32_t const num_groups,
+    int64_t const num_tokens) {
+  constexpr int kVec = 16 / sizeof(scalar_t);
+  constexpr int kLanes = group_size / kVec;
+  static_assert(kVec == 8, "expects 16-bit input");
+  static_assert(kLanes > 0 && kLanes <= 32 && (32 % kLanes) == 0);
+
+  int64_t const gid =
+      (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / kLanes;
+  int const lane = threadIdx.x % kLanes;
+  bool const valid = gid < num_tokens * num_groups;
+  int64_t const token_idx = valid ? gid / num_groups : 0;
+  int const group_idx = valid ? static_cast<int>(gid % num_groups) : 0;
+
+  int64_t const in_off = token_idx * (2 * static_cast<int64_t>(hidden_size)) +
+                         group_idx * group_size + lane * kVec;
+  alignas(16) scalar_t g[kVec];
+  alignas(16) scalar_t u[kVec];
+  if (valid) {
+    *reinterpret_cast<uint4*>(g) =
+        *reinterpret_cast<uint4 const*>(input + in_off);
+    *reinterpret_cast<uint4*>(u) =
+        *reinterpret_cast<uint4 const*>(input + in_off + hidden_size);
+  } else {
+    *reinterpret_cast<uint4*>(g) = make_uint4(0, 0, 0, 0);
+    *reinterpret_cast<uint4*>(u) = make_uint4(0, 0, 0, 0);
+  }
+
+  float r[kVec];
+  float local_max = 0.f;
+#pragma unroll
+  for (int i = 0; i < kVec; ++i) {
+    float const gate = static_cast<float>(g[i]);
+    float const sigmoid_gate = 1.0f / (1.0f + expf(-gate));
+    r[i] = gate * sigmoid_gate * static_cast<float>(u[i]);
+    local_max = fmaxf(local_max, fabsf(r[i]));
+  }
+#pragma unroll
+  for (int off = kLanes / 2; off > 0; off >>= 1) {
+    local_max =
+        fmaxf(local_max, VLLM_SHFL_XOR_SYNC_WIDTH(local_max, off, kLanes));
+  }
+
+  float group_scale = local_max / quant_type_max_v<scalar_out_t>;
+  if (scale_ub != nullptr) {
+    group_scale = fminf(group_scale, *scale_ub);
+  }
+  group_scale = fmaxf(group_scale, min_scaling_factor<scalar_out_t>::val());
+  if (!valid) {
+    return;
+  }
+
+  alignas(8) scalar_out_t q[kVec];
+#pragma unroll
... (truncated, 99 more lines)
```
</details>

<details><summary>📄 diff: [Bugfix][MoE] Respect padded row strides in top-k routing (#</summary>

```diff
diff --git a/csrc/moe/topk_softmax_kernels.cu b/csrc/moe/topk_softmax_kernels.cu
index b687e95f75e8..b9e8b319b462 100644
--- a/csrc/moe/topk_softmax_kernels.cu
+++ b/csrc/moe/topk_softmax_kernels.cu
@@ -78,7 +78,8 @@ enum ScoringFunc {
 // in the softmax kernel when we extend this module to support expert-choice routing.
 template <int TPB, typename InputType>
 __launch_bounds__(TPB) __global__
-    void moeSoftmax(const InputType* input, const bool* finished, float* output, const int num_cols)
+    void moeSoftmax(const InputType* input, const bool* finished, float* output, const int num_cols,
+        const int64_t input_row_stride)
 {
     using BlockReduce = cub::BlockReduce<float, TPB>;
     __shared__ typename BlockReduce::TempStorage tmpStorage;
@@ -86,7 +87,8 @@ __launch_bounds__(TPB) __global__
     __shared__ float normalizing_factor;
     __shared__ float float_max;
 
-    const int thread_row_offset = blockIdx.x * num_cols;
+    const int64_t input_row_offset = blockIdx.x * input_row_stride;
+    const int output_row_offset = blockIdx.x * num_cols;
 
     float threadData(-FLT_MAX);
 
@@ -98,8 +100,7 @@ __launch_bounds__(TPB) __global__
 
     for (int ii = threadIdx.x; ii < num_cols; ii += TPB)
     {
-        const int idx = thread_row_offset + ii;
-        const float val = toFloat(input[idx]);
+        const float val = toFloat(input[input_row_offset + ii]);
         threadData = max(val, threadData);
     }
 
@@ -114,8 +115,7 @@ __launch_bounds__(TPB) __global__
 
     for (int ii = threadIdx.x; ii < num_cols; ii += TPB)
     {
-        const int idx = thread_row_offset + ii;
-        const float val = toFloat(input[idx]);
+        const float val = toFloat(input[input_row_offset + ii]);
         threadData += expf(val - float_max);
     }
 
@@ -129,20 +129,21 @@ __launch_bounds__(TPB) __global__
 
     for (int ii = threadIdx.x; ii < num_cols; ii += TPB)
     {
-        const int idx = thread_row_offset + ii;
-        const float val = toFloat(input[idx]);
+        const float val = toFloat(input[input_row_offset + ii]);
         float softmax_val = expf(val - float_max) * normalizing_factor;
         // Clamp NaN/Inf to 0 to prevent duplicate expert IDs downstream.
         if (isnan(softmax_val) || isinf(softmax_val)) softmax_val = 0.f;
-        output[idx] = softmax_val;
+        output[output_row_offset + ii] = softmax_val;
     }
 }
 
 template <int TPB, typename InputType>
 __launch_bounds__(TPB) __global__
-    void moeSigmoid(const InputType* input, const bool* finished, float* output, const int num_cols)
+    void moeSigmoid(const InputType* input, const bool* finished, float* output, const int num_cols,
+        const int64_t input_row_stride)
 {
-    const int thread_row_offset = blockIdx.x * num_cols;
+    const int64_t input_row_offset = blockIdx.x * input_row_stride;
+    const int output_row_offset = blockIdx.x * num_cols;
 
     // Don't touch finished rows.
     if ((finished != nullptr) && finished[blockIdx.x])
@@ -152,12 +153,11 @@ __launch_bounds__(TPB) __global__
 
     for (int ii = threadIdx.x; ii < num_cols; ii += TPB)
     {
-        const int idx = thread_row_offset + ii;
-        const float val = toFloat(input[idx]);
+        const float val = toFloat(input[input_row_offset + ii]);
         float sigmoid_val = 1.0f / (1.0f + __expf(-val));
         // Clamp NaN/Inf to 0 to prevent duplicate expert IDs downstream.
... (truncated, 187 more lines)
```
</details>

<details><summary>📄 diff: [Bugfix][Frontend] Return 400 for images rejected by Pillow'</summary>

```diff
diff --git a/tests/multimodal/media/test_image.py b/tests/multimodal/media/test_image.py
index a754746fea62..74ed5dacea37 100644
--- a/tests/multimodal/media/test_image.py
+++ b/tests/multimodal/media/test_image.py
@@ -2,6 +2,7 @@
 # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 import inspect
 import struct
+from io import BytesIO
 from pathlib import Path
 
 import numpy as np
@@ -260,7 +261,6 @@ def test_image_pixel_limit_respected():
     import vllm.envs as envs
 
     image = Image.new("RGB", (100, 100), (255, 0, 0))
-    from io import BytesIO
 
     buf = BytesIO()
     image.save(buf, format="PNG")
@@ -280,7 +280,6 @@ def test_image_pixel_limit_rejected(monkeypatch):
     monkeypatch.setattr(envs, "VLLM_MAX_IMAGE_PIXELS", 100)
 
     image = Image.new("RGB", (20, 20), (0, 255, 0))
-    from io import BytesIO
 
     buf = BytesIO()
     image.save(buf, format="PNG")
@@ -291,6 +290,23 @@ def test_image_pixel_limit_rejected(monkeypatch):
         image_io.load_bytes(data)
 
 
+def test_image_decompression_bomb_rejected(monkeypatch):
+    """Pillow rejects images above twice `Image.MAX_IMAGE_PIXELS` with its own
+    non-OSError exception, which must surface as a client-side ValueError."""
+    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 50)
+
+    image = Image.new("RGB", (20, 20), (0, 255, 0))
+
+    buf = BytesIO()
+    image.save(buf, format="PNG")
+    data = buf.getvalue()
+
+    image_io = ImageMediaIO()
+    with pytest.raises(ValueError, match="Failed to load image") as exc_info:
+        image_io.load_bytes(data)
+    assert isinstance(exc_info.value.__cause__, Image.DecompressionBombError)
+
+
 def test_image_broken_png_rejected():
     """A PNG with a corrupted chunk structure surfaces as a client-side
     ValueError (Pillow raises SyntaxError, which is not an OSError)."""
@@ -357,7 +373,6 @@ def test_image_pixel_limit_disabled(monkeypatch):
     monkeypatch.setattr(envs, "VLLM_MAX_IMAGE_PIXELS", 0)
 
     image = Image.new("RGB", (1000, 1000), (0, 0, 255))
-    from io import BytesIO
 
     buf = BytesIO()
     image.save(buf, format="PNG")
@@ -369,8 +384,6 @@ def test_image_pixel_limit_disabled(monkeypatch):
 
 
 def _png_bytes(width: int, height: int) -> bytes:
-    from io import BytesIO
-
     buf = BytesIO()
     Image.new("RGB", (width, height), (1, 2, 3)).save(buf, format="PNG")
     return buf.getvalue()
diff --git a/vllm/multimodal/media/image.py b/vllm/multimodal/media/image.py
index 108d684afff0..3651db9b98e4 100644
--- a/vllm/multimodal/media/image.py
+++ b/vllm/multimodal/media/image.py
@@ -103,6 +103,7 @@ def load_bytes(self, data: bytes) -> MediaWithBytes[Image.Image]:
             Image.UnidentifiedImageError,
             SyntaxError,
             RuntimeError,
+            Image.DecompressionBombError,
         ) as e:
             raise ValueError(f"Failed to load image: {e}") from e
... (truncated, 2 more lines)
```
</details>

<details><summary>📄 diff: [Bugfix][DP] Log cross-engine step statistics ordering at de</summary>

```diff
diff --git a/docs/usage/troubleshooting.md b/docs/usage/troubleshooting.md
index bbd4109a4196..0e70923f2c46 100644
--- a/docs/usage/troubleshooting.md
+++ b/docs/usage/troubleshooting.md
@@ -40,6 +40,8 @@ If other strategies don't solve the problem, it's likely that the vLLM instance
 - `export NCCL_DEBUG=TRACE` to turn on more logging for NCCL.
 - `export VLLM_TRACE_FUNCTION=1` to record all function calls for inspection in the log files to tell which function crashes or hangs. (WARNING: This flag will slow down the token generation by **over 100x**. Do not use unless absolutely needed.)
 
+The DP coordinator's `Received stats for out-of-order step` diagnostic is logged at DEBUG level. Statistics from different engines can arrive in a different step order; this message alone does not indicate a failed request. Set `VLLM_LOGGING_LEVEL=DEBUG` to inspect these reports.
+
 ## Breakpoints
 
 Setting normal `pdb` breakpoints may not work in vLLM's codebase if they are executed in a subprocess. You will experience something like:
diff --git a/vllm/v1/engine/coordinator.py b/vllm/v1/engine/coordinator.py
index 524794dac0c2..7d2c7106d049 100644
--- a/vllm/v1/engine/coordinator.py
+++ b/vllm/v1/engine/coordinator.py
@@ -391,7 +391,9 @@ def process_input_socket(
                             elif stats_wave != last_stats_wave or (
                                 stats_step != last_stats_step
                             ):
-                                logger.warning(
+                                # Stats from different engines can arrive out of
+                                # step order even when each engine reports in order.
+                                logger.debug(
                                     "Received stats for out-of-order "
                                     "step (%d, %d) from engine %d (expected "
                                     "> (%d, %d))",

```
</details>

<details><summary>📄 diff: [Tests] Add TP2 PCP2 canonical mapping byte coverage (#62529</summary>

```diff
diff --git a/tests/v1/kv_connector/unit/offloading_connector/test_canonical_mapping.py b/tests/v1/kv_connector/unit/offloading_connector/test_canonical_mapping.py
index 9c3ba6d555dd..a38562532465 100644
--- a/tests/v1/kv_connector/unit/offloading_connector/test_canonical_mapping.py
+++ b/tests/v1/kv_connector/unit/offloading_connector/test_canonical_mapping.py
@@ -337,6 +337,63 @@ def mappings_at(tp: int):
         assert _store_all(mappings, pages, canonical_size) == reference
 
 
+def _reference_page(form, tokens, heads, head_size) -> bytes:
+    """Serialize logical K/V, token, head, and dimension coordinates."""
+    if form.endswith("nhd"):
+        token_heads = [(token, head) for token in tokens for head in heads]
+    else:
+        token_heads = [(token, head) for head in heads for token in tokens]
+    if form.startswith("split"):
+        cells = ((kv, token, head) for kv in range(2) for token, head in token_heads)
+    else:
+        cells = ((kv, token, head) for token, head in token_heads for kv in range(2))
+    return bytes(
+        (32 * token + 8 * head + 4 * kv + dim) % 256
+        for kv, token, head in cells
+        for dim in range(head_size)
+    )
+
+
+@pytest.mark.parametrize("form", sorted(CACHE_BUILDERS))
+@pytest.mark.parametrize("interleave", [1, 2])
+def test_tp2_pcp2_store_load_matches_reference(form, interleave):
+    """TP head shards and PCP token shards preserve independent reference bytes."""
+    spec = _full_spec()
+    cache = CACHE_BUILDERS[form](spec)
+    mappings = [
+        _mapping(
+            spec,
+            cache,
+            _ctx(rank, tp=2, dcp=1, pcp=2, interleave=interleave, total=4),
+        )
+        for rank in range(4)
+    ]
+    pcp_tokens = {
+        1: ((0, 2, 4, 6), (1, 3, 5, 7)),
+        2: ((0, 1, 4, 5), (2, 3, 6, 7)),
+    }[interleave]
+    # Rank order: PCP0/TP0, PCP0/TP1, PCP1/TP0, PCP1/TP1.
+    rank_coordinates = (
+        (pcp_tokens[0], (0, 1)),
+        (pcp_tokens[0], (2, 3)),
+        (pcp_tokens[1], (0, 1)),
+        (pcp_tokens[1], (2, 3)),
+    )
+    pages = [
+        _reference_page(form, tokens, heads, spec.head_size)
+        for tokens, heads in rank_coordinates
+    ]
+    reference = _reference_page(form, range(8), range(4), spec.head_size)
+
+    _verify_tiling("tp2-pcp2", mappings)
+    assert _store_all(mappings, pages, len(reference)) == reference
+    for rank, (mapping, page) in enumerate(zip(mappings, pages)):
+        assert mapping.canonical_page_size_bytes == len(reference)
+        assert mapping.local_page_size_bytes == len(page)
+        assert not mapping.parallelism_agnostic
+        assert _load_one(mapping, reference) == page, f"rank {rank}"
+
+
 def test_cp_round_trip():
     # tp=4 / dcp=2 / 2 KV heads: 4 workers jointly hold one canonical page
     spec = _full_spec(num_kv_heads=1)

```
</details>

<details><summary>📄 diff: [Startup] Skip the CUDA graph memory profile when its estima</summary>

```diff
diff --git a/tests/v1/worker/test_gpu_worker.py b/tests/v1/worker/test_gpu_worker.py
index ad8ea27bbec2..49ebc3c87e01 100644
--- a/tests/v1/worker/test_gpu_worker.py
+++ b/tests/v1/worker/test_gpu_worker.py
@@ -278,6 +278,98 @@ def profile_run(*, randomize_inputs=False):
     assert get_current_vllm_config_or_none() is None
 
 
+@pytest.mark.parametrize(
+    "use_v2,extensible,cudagraphs,estimate_enabled,estimate,warns",
+    [
+        # MRV1: the flag decides whether the estimate runs and is applied.
+        (False, False, True, True, 3, False),
+        (False, False, True, False, 3, True),
+        # MRV2 with the extensible cache: the estimate is 0 by design, and
+        # neither flag value warns about unaccounted graph memory.
+        (True, True, True, True, 0, False),
+        (True, True, True, False, 0, False),
+        # MRV2 fallback (extensible cache off).
+        (True, False, True, True, 3, False),
+        (True, False, True, False, 3, True),
+        # MRV2 with no graphs to capture and the flag at 1.
+        (True, False, False, True, 0, False),
+    ],
+)
+def test_cudagraph_memory_profile_runs_only_when_applied(
+    monkeypatch, use_v2, extensible, cudagraphs, estimate_enabled, estimate, warns
+):
+    """The graph memory estimate is a throwaway capture pass; with the
+    estimate disabled it is neither run nor subtracted from the KV cache
+    budget, and the disabled-estimate warning only fires when graph memory
+    goes unaccounted for."""
+    from vllm.config import CUDAGraphMode
+
+    estimate *= GiB_bytes
+    calls = []
+    warnings = []
+
+    def profile_cudagraph_memory():
+        calls.append(1)
+        return estimate
+
+    worker = SimpleNamespace(
+        vllm_config=SimpleNamespace(
+            compilation_config=SimpleNamespace(
+                cudagraph_mode=CUDAGraphMode.FULL if cudagraphs else CUDAGraphMode.NONE
+            )
+        ),
+        randomize_dummy_inputs=False,
+        use_v2_model_runner=use_v2,
+        cache_config=SimpleNamespace(
+            kv_cache_memory_bytes=None,
+            gpu_memory_utilization=0.9,
+            resolved_gpu_memory_utilization=0.9,
+            enable_extensible_kv_cache=extensible,
+        ),
+        model_config=SimpleNamespace(multimodal_config=None),
+        parallel_config=SimpleNamespace(),
+        model_runner=SimpleNamespace(
+            profile_run=lambda **kwargs: None,
+            model_memory_usage=0,
+            profile_cudagraph_memory=profile_cudagraph_memory,
+        ),
+        init_snapshot=SimpleNamespace(
+            free_memory=80 * GiB_bytes, total_memory=80 * GiB_bytes
+        ),
+        requested_memory=72 * GiB_bytes,
+        _scoped_allocator_max_split=lambda **kwargs: nullcontext(),
+    )
+    profile_result = SimpleNamespace(
+        after_profile=SimpleNamespace(free_memory=60 * GiB_bytes),
+        total_consumed=20 * GiB_bytes,
+        non_kv_cache_memory=20 * GiB_bytes,
+        transient_peak_headroom=0,
+    )
+    monkeypatch.setattr(gpu_worker, "maybe_apply_startup_plan", lambda _: None)
+    monkeypatch.setattr(
+        gpu_worker,
+        "memory_profiling",
+        lambda *args, **kwargs: nullcontext(profile_result),
... (truncated, 145 more lines)
```
</details>

<details><summary>📄 diff: [Rust Frontend] Kimi K3.1 chat renderer (Kimi K3.1 split 2/5</summary>

```diff
diff --git a/rust/src/chat/src/backend/hf.rs b/rust/src/chat/src/backend/hf.rs
index 4cc929ac4e0c..601643552426 100644
--- a/rust/src/chat/src/backend/hf.rs
+++ b/rust/src/chat/src/backend/hf.rs
@@ -27,6 +27,7 @@ use crate::renderer::hf::{HfChatRenderer, MultimodalRenderInfo};
 use crate::renderer::{
     DeepSeekV4ChatRenderer, DeepSeekV32ChatRenderer, DeepSeekV41ChatRenderer, DynChatRenderer,
     Glm5NextChatRenderer, HarmonyChatRenderer, InklingChatRenderer, KimiK3ChatRenderer,
+    KimiK3Variant,
 };
 use crate::request::ChatRequest;
 use crate::{DynChatOutputProcessor, RendererSelection};
@@ -103,6 +104,7 @@ impl HfChatBackend {
             RendererSelection::KimiK3 => Arc::new(KimiK3ChatRenderer::new(
                 tokenizer.clone(),
                 options.default_chat_template_kwargs,
+                KimiK3Variant::from_model_type(model_type),
             )),
             RendererSelection::Glm5Next => Arc::new(
                 Glm5NextChatRenderer::new(options.default_chat_template_kwargs)
@@ -253,7 +255,7 @@ mod tests {
     use super::HfChatBackend;
     use crate::backend::{ChatBackend, LoadModelBackendsOptions, NewChatOutputProcessorOptions};
     use crate::request::{ChatContent, ChatMessage, ChatRequest, ResolvedToolContext};
-    use crate::{ChatTool, ParserSelection, RendererSelection};
+    use crate::{ChatTool, EffortValue, ParserSelection, RendererSelection};
 
     fn request_with_user_text(text: &str) -> ChatRequest {
         ChatRequest {
@@ -356,6 +358,25 @@ mod tests {
         );
     }
 
+    #[test]
+    fn auto_uses_kimi_k3_renderer_with_k31_efforts_for_kimi_k31_model_type() {
+        let backend = backend_for_selection(
+            RendererSelection::Auto,
+            r#"{"model_type":"kimi_k31"}"#,
+            r#"{}"#,
+        );
+        let mut request = request_with_user_text("hello");
+        request.chat_options.reasoning_effort = Some(EffortValue::from("medium"));
+
+        let rendered = backend.chat_renderer().render(&request).unwrap();
+
+        assert!(matches!(rendered.prompt, Prompt::TokenIds(_)));
+        assert_eq!(
+            rendered.effective_template_kwargs["thinking_effort"],
+            "medium"
+        );
+    }
+
     #[test]
     fn auto_uses_hf_renderer_for_other_model_types() {
         let prompt = render_prompt(
diff --git a/rust/src/chat/src/lib.rs b/rust/src/chat/src/lib.rs
index f4db0dff6996..8ab29b7c951a 100644
--- a/rust/src/chat/src/lib.rs
+++ b/rust/src/chat/src/lib.rs
@@ -39,8 +39,8 @@ pub use reasoning::EffortValue;
 pub use renderer::hf::ChatTemplateContentFormatOption;
 pub use renderer::{
     ChatRenderer, DeepSeekV4ChatRenderer, DeepSeekV32ChatRenderer, DeepSeekV41ChatRenderer,
-    DynChatRenderer, HarmonyChatRenderer, InklingChatRenderer, KimiK3ChatRenderer, MediaPartSource,
-    RenderedPrompt, RendererSelection,
+    DynChatRenderer, HarmonyChatRenderer, InklingChatRenderer, KimiK3ChatRenderer, KimiK3Variant,
+    MediaPartSource, RenderedPrompt, RendererSelection,
 };
 pub use request::{
     ChatContent, ChatContentPart, ChatMessage, ChatOptions, ChatRequest, ChatRole, ChatTool,
diff --git a/rust/src/chat/src/multimodal.rs b/rust/src/chat/src/multimodal.rs
index f3121693826e..a503269189b1 100644
--- a/rust/src/chat/src/multimodal.rs
+++ b/rust/src/chat/src/multimodal.rs
@@ -1702,6 +1702,25 @@ mod tests {
         test_info("deepseek_v41", config, tokenizer)
     }
 
+    #[test]
+    fn kimi_k31_resolves_kimi_k3_image_support_by_model_type() {
... (truncated, 350 more lines)
```
</details>

<details><summary>📄 diff: [Model] Kimi K3.1: model files and NVIDIA specialized decode</summary>

```diff
diff --git a/docs/models/supported_models.md b/docs/models/supported_models.md
index 67304fc24173..14fe17b0cafc 100644
--- a/docs/models/supported_models.md
+++ b/docs/models/supported_models.md
@@ -593,6 +593,7 @@ These models primarily accept the [`LLM.generate`](./generative_models.md#llmgen
 | `KimiAudioForConditionalGeneration` | Kimi-Audio | T + A<sup>+</sup> | `moonshotai/Kimi-Audio-7B-Instruct` | | ✅︎ |
 | `KimiK25ForConditionalGeneration` | Kimi-K2.5 | T + I<sup>+</sup> | `moonshotai/Kimi-K2.5` | | ✅︎ |
 | `KimiK3ForConditionalGeneration` | Kimi-K3 | T + I<sup>+</sup> | `moonshotai/Kimi-K3` | | ✅︎ |
+| `KimiK31ForConditionalGeneration` | Kimi-K3.1 | T + I<sup>+</sup> | `moonshotai/Kimi-K3.1` | | ✅︎ |
 | `KimiVLForConditionalGeneration` | Kimi-VL-A3B-Instruct, Kimi-VL-A3B-Thinking | T + I<sup>+</sup> | `moonshotai/Kimi-VL-A3B-Instruct`, `moonshotai/Kimi-VL-A3B-Thinking` | | ✅︎ |
 | `LightOnOCRForConditionalGeneration` | LightOnOCR-1B | T + I<sup>E+</sup> | `lightonai/LightOnOCR-1B`, etc | ✅︎ | ✅︎ |
 | `Lfm2VlForConditionalGeneration` | LFM2-VL | T + I<sup>+</sup> | `LiquidAI/LFM2-VL-450M`, `LiquidAI/LFM2-VL-3B`, `LiquidAI/LFM2-VL-8B-A1B`, etc. | ✅︎ | ✅︎ |
diff --git a/tests/kernels/test_bf16_skinny_gemm.py b/tests/kernels/test_bf16_skinny_gemm.py
index 9824490147c3..00c5994be345 100644
--- a/tests/kernels/test_bf16_skinny_gemm.py
+++ b/tests/kernels/test_bf16_skinny_gemm.py
@@ -34,7 +34,7 @@
     (2112, 7168): (set(), set(range(1, 17))),
     (2304, 1536): (set(), set(range(1, 17))),
     (4608, 1536): (set(), set(range(1, 17))),
-    (3584, 7168): ({1}, set(range(2, 9))),
+    (3584, 7168): ({1}, set()),
     (6288, 7168): (set(range(1, 5)), set()),
     (12448, 7168): (set(range(1, 4)), set()),
     (7168, 768): (set(), set(range(1, 17))),
@@ -164,6 +164,14 @@
     (10240, 7168, 2): (32, 2, 4, 8),
     (10240, 7168, 3): (64, 4, 1, 8),
     (10240, 7168, 4): (64, 4, 1, 8),
+    # Kimi K3.1 DSA packed projections: q_a/kv_a/wk/weights, q_b/wq (TP1, TP8).
+    (2272, 7168, 1): (224, 4, 2, 8),
+    (2272, 7168, 2): (224, 4, 2, 8),
+    (2272, 7168, 3): (128, 2, 2, 8),
+    (2272, 7168, 4): (224, 4, 1, 8),
+    (2272, 7168, 5): (128, 2, 1, 8),
+    (6400, 1536, 1): (64, 4, 2, 8),
+    (22528, 1536, 1): (32, 8, 2, 8),
 }
 
 EXPECTED_RESIDUAL_CUTE_CONFIGS = {
@@ -662,8 +670,8 @@ def test_build_plan_matches_selector() -> None:
 
 # Keyed by local (N, K): (cute token counts, dsv3 token counts). Mirrors
 # KIMI_K3_PROJECTIONS_SM100, measured on B200; intentionally different from
-# EXPECTED_SELECTIONS (e.g. 3584x7168 routes dsv3 at M2..8 on SM103 but only
-# wins with CuTe at M1..3 on SM100).
+# EXPECTED_SELECTIONS (e.g. 3584x7168 uses CuTe at M1 only on SM103 but wins
+# with CuTe at M1..3 on SM100).
 EXPECTED_SELECTIONS_SM100 = {
     (1536, 128): (set(), {1, 16}),
     (3072, 128): (set(), {8}),
diff --git a/tests/models/kimi_k3/nvidia/__init__.py b/tests/models/kimi_k3/nvidia/__init__.py
new file mode 100644
index 000000000000..e69de29bb2d1
diff --git a/tests/models/kimi_k3/nvidia/conftest.py b/tests/models/kimi_k3/nvidia/conftest.py
new file mode 100644
index 000000000000..99a74b0444a1
--- /dev/null
+++ b/tests/models/kimi_k3/nvidia/conftest.py
@@ -0,0 +1,24 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""The specialized decode tests need SM100 multi-GPU nodes and real-rank setups
+that CI does not provide. They are collected only when ``specialized_decode`` (or
+a path inside it) is passed to pytest explicitly, e.g.
+``pytest tests/models/kimi_k3/nvidia/specialized_decode``; a run over
+``tests/models/kimi_k3`` skips them."""
+
+from pathlib import Path
+
+import pytest
+
+_SPECIALIZED_DECODE = Path(__file__).parent / "specialized_decode"
+
+
+def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
+    if collection_path != _SPECIALIZED_DECODE:
+        return None
+    root = config.invocation_params.dir
+    for arg in config.args:
... (truncated, 41066 more lines)
```
</details>

<details><summary>📄 diff: [Sleep Mode] Level 2 discards only parameters (#66010)</summary>

```diff
diff --git a/docs/features/sleep_mode.md b/docs/features/sleep_mode.md
index 8c73dd16a624..cb1b38b76f72 100644
--- a/docs/features/sleep_mode.md
+++ b/docs/features/sleep_mode.md
@@ -11,14 +11,33 @@ Key benefits:
 - **Fine-grained control**: Optionally wake up only model weights or KV cache to avoid OOM during weight updates.
 
 !!! note
-    This feature is now supported on CUDA and ROCm platform.
+    This feature is now supported on CUDA, ROCm and XPU platforms.
 
 !!! note
     For more information, see this [Blog Post](https://blog.vllm.ai/2025/10/26/sleep-mode.html).
 
 ## Sleep levels
 
-Level 1 sleep will offload the model weights and discard the KV cache. The content of KV cache is forgotten. Level 1 sleep is good for sleeping and waking up the engine to run the same model again. The model weights are backed up in CPU memory. Please make sure there's enough CPU memory to store the model weights. Level 2 sleep will discard both the model weights and the KV cache (while the model's buffers are kept in CPU, like rope scaling tensors). The content of both the model weights and KV cache is forgotten. Level 2 sleep is good for sleeping and waking up the engine to run a different model or update the model, where previous model weights are not needed, e.g. RLHF weight update. Level 2 sleep is also useful when there is not enough CPU memory to back up the model weights, e.g. when a colocated trainer already uses CPU memory for offloading its own state; since nothing is backed up, restore the weights after waking up with `collective_rpc("reload_weights")`.
+Level 1 sleep will offload the model weights and discard the KV cache. The content of KV cache is forgotten. Level 1 sleep is good for sleeping and waking up the engine to run the same model again. The model weights are backed up in CPU memory. Please make sure there's enough CPU memory to store the model weights. Level 2 sleep will discard the model parameters and the KV cache, and back up the rest of what the model allocated on the GPU while loading, like rope scaling tensors, in CPU memory. The content of the parameters and KV cache is forgotten. Level 2 sleep is good for sleeping and waking up the engine to update the model's weights, where the previous weights are not needed, e.g. RLHF weight update. Level 2 sleep is also useful when there is not enough CPU memory to back up the model weights, e.g. when a colocated trainer already uses CPU memory for offloading its own state; since the parameters are not backed up, restore them after waking up with `collective_rpc("reload_weights")`.
+
+### What level 2 keeps
+
+Level 2 discards only what is rebuilt on wake, which is what the weight update writes.
+
+| State | After a level-2 wake, before the update |
+| --- | --- |
+| Parameters | Undefined, unless selected by `sleep_preserve_parameter_names` |
+| Anything else allocated while the model loads: buffers, plain attributes, tensors held in closures, pointer tables | Kept, at the same address |
+| Tensors derived from weights | Kept with their old values until post-load processing refreshes them |
+| KV cache and workspace scratch | Contents undefined |
+
+This is the same on CUDA, ROCm and XPU.
+
+**If you send weights to a sleeping engine,** send every parameter that isn't selected by `sleep_preserve_parameter_names`, including the ones that didn't change. `collective_rpc("reload_weights")` does this from the checkpoint. With DeepSeek V4.1, send `engram.wkv`: only the Engram embedding table weights and scales can be left out.
+
+**If you write a kernel,** state built during model loading needs nothing special, whether or not it is registered as a buffer. A tensor computed from weights must be recomputed in post-load processing (see `refill_derived`), or it goes stale after an update. Take per-step scratch from the workspace manager, so that it isn't backed up.
+
+**Host memory** is what non-parameter state takes, plus any preserved parameters. For example, DeepSeek-V4.1-Flash at TP 4 frees 80 GiB per GPU and backs up 2.35 GiB per rank. An allocation that mixes parameters with other tensors is backed up whole.
 
 ## Usage
 
@@ -44,7 +63,7 @@ llm.wake_up()
 
 ```python
 # Sleep level 2
-# Put the engine to sleep (level=2: discard both weights and KV cache)
+# Put the engine to sleep (level=2: discard parameters and KV cache)
 llm.sleep(level=2)
 
 # Reallocate weights memory only
@@ -88,9 +107,9 @@ training. It covers two use cases:
 Target patterns match
 `model.named_parameters()`; checkpoint names and `requires_grad` are not used.
 
-Level-2 sleep backs up selected GPU parameters to pageable CPU memory; weights
-wake-up restores them in place and releases the backups. CPU parameters need no
-copy. Allow enough host memory for backups. Level-1 behavior is unchanged.
+Level-2 sleep backs up selected GPU parameters to CPU memory; weights wake-up
+restores them in place. CPU parameters need no copy. Allow enough host memory
+for backups. Level-1 behavior is unchanged.
 
 The trainer must omit these parameters from updates, and reload/post-processing
 must preserve their values and storage (including avoiding replacement with meta
diff --git a/tests/basic_correctness/memory/cumem/test_cumem.py b/tests/basic_correctness/memory/cumem/test_cumem.py
index 4fb94f0c4e8f..7e53770457c8 100644
--- a/tests/basic_correctness/memory/cumem/test_cumem.py
+++ b/tests/basic_correctness/memory/cumem/test_cumem.py
@@ -87,12 +87,14 @@ def sleep_and_check_backups(llm, level: int) -> dict[int, tuple]:
     state, host_after = llm.collective_rpc(backup_state)[0]
     assert host_after == host_before
     for tag, size, backup_ptr, backup_bytes, pinned in state.values():
-        if tag in OFFLOAD_TAGS[level]:
+        # Level 2 keeps the weights allocations holding more than parameters.
+        if tag in OFFLOAD_TAGS[level] or (tag == "weights" and backup_ptr is not None):
             assert backup_ptr is not None
             assert (backup_bytes, pinned) == (size, PIN_MEMORY)
         else:
             assert backup_ptr is None
-    assert any(tag == "weights" for tag, *_ in state.values())
+    weights_backed_up = [s[2] is not None for s in state.values() if s[0] == "weights"]
+    assert weights_backed_up and (level == 1 or not all(weights_backed_up))
... (truncated, 812 more lines)
```
</details>

## sglang — 最近 24h commits

- **[[Fix] hicache: Return int page slots from `_storage_page_index` without DCP (#49034)](https://github.com/sgl-project/sglang/commit/1c8740ecfc4f0840c9009fb24282a979d947ff9d)** — Khoa Pham
  `1c8740ec` 2026-11-07T07:34:38Z
- **[[Perf] [NVIDIA] unquant: Autotune low-M BF16 GEMMs via FlashInfer mm_bf16 (#43703)](https://github.com/sgl-project/sglang/commit/19cee50c290e399ba4742adaeeade6dbfc817092)** — Chunan Zeng
  `19cee50c` 2026-11-07T07:33:09Z
- **[[Fix] radix-cache: Use Rust TreeCore with PyTorch 2.15 in the cu134 image instead of slow python backend (#48988)](https://github.com/sgl-project/sglang/commit/bcd5ec1195a06f3861973b0ca7adbaa24446f754)** — Trevor Morris
  `bcd5ec11` 2026-11-07T06:53:17Z
- **[[Feat] kimi-k3.1: HiCache under decode context parallelism (#48877)](https://github.com/sgl-project/sglang/commit/b165323cb648e89d0922ce15613f984f4f84b4fd)** — Khoa Pham
  `b165323c` 2026-11-07T06:26:25Z
- **[[Fix] kimi-k3: Cap cached_tokens at prompt_tokens in the Rust processor (#48393)](https://github.com/sgl-project/sglang/commit/fee391881ad2076316032bd8f7455d1c7688b498)** — Aryan Sinha
  `fee39188` 2026-11-07T06:14:12Z
- **[[Test] rust-server: Cover the embedded Rust frontend in a two-node TP engine (#48511)](https://github.com/sgl-project/sglang/commit/286cb448b1c03fc15f4ab10b39c449dfdb47794a)** — metamergebot
  `286cb448` 2026-11-07T05:41:57Z
- **[[Test] load-snapshot: Pin the Python server mode in TestZmqReaderOwner (#48453)](https://github.com/sgl-project/sglang/commit/a6ae2a54e504f837a281a348478f8a3b114c65af)** — metamergebot
  `a6ae2a54` 2026-11-07T05:41:18Z
- **[[Refactor] scheduler: Run a prefill batch's KV calls from one overridable _flush_prefill (#48449)](https://github.com/sgl-project/sglang/commit/d333aded47f97c682ff68db7e0a4179066a7b000)** — metamergebot
  `d333aded` 2026-11-07T05:40:15Z
- **[[Fix] io_struct: Give each request its own sampling_params dict during normalization (#48446)](https://github.com/sgl-project/sglang/commit/c7d154f910c9d5d459c15b2eaefb1d548c31591b)** — metamergebot
  `c7d154f9` 2026-11-07T05:39:48Z
- **[[Doc] [AMD] cookbook: Match the Kimi-K3.1-Pro MI35x single-node cells to the current recipe (#49022)](https://github.com/sgl-project/sglang/commit/9a57115efc9b2b06e5b3893f4367139cfea8a732)** — Kevin Mi
  `9a57115e` 2026-11-07T05:31:16Z
- **[[Fix] mem_cache: Chunk MambaPool temporal state copy to avoid OOM spike (#47984)](https://github.com/sgl-project/sglang/commit/784cffe3d90bb625041f9c0b4f06e66b0a122e42)** — Tanish Malekar
  `784cffe3` 2026-11-07T04:49:53Z
- **[[Fix] rust-server: Let DP rank 0's drain wait for every attention-DP listener (#48492)](https://github.com/sgl-project/sglang/commit/99833a441eb53fb2afedf36d97f9b0b400b6a275)** — metamergebot
  `99833a44` 2026-11-07T04:44:21Z
- **[[Perf] kimi-k3.1: Faster DCP decode reduce (push-combine, V pre-projection, direct exchange) (#48740)](https://github.com/sgl-project/sglang/commit/00c3b143c081b9edf5b8c30b23641f91ce9dfe8b)** — Khoa Pham
  `00c3b143` 2026-11-07T04:38:05Z
- **[[Feat] kimi-k3.1: DSPARK and DFLASH under decode context parallelism (#48739)](https://github.com/sgl-project/sglang/commit/709b8b3c79467c67bff11397b62ebe251a62559e)** — Khoa Pham
  `709b8b3c` 2026-11-07T04:37:39Z
- **[[Perf] [NVIDIA] fp8: Let the Hopper block GEMV take K=6144 for MiniMax M3.1 (#48531)](https://github.com/sgl-project/sglang/commit/ad951803d69a050efe2113f5188814e1a65312f2)** — Chen Cheng
  `ad951803` 2026-11-07T04:28:03Z
- **[[Perf] [NVIDIA] muse-spark: Bound B200 projection dispatch and allocator compilation (#47129)](https://github.com/sgl-project/sglang/commit/0ced08c3dbaa4d0b21cb7c11d670b77dfa74765d)** — Zhuofeng-Li
  `0ced08c3` 2026-11-07T04:27:29Z
- **[[Perf] dspark: Build the compact verify window and stage its padded layout in one launch each (#48833)](https://github.com/sgl-project/sglang/commit/a418db307bfa2d43932cfd0c3711373665618868)** — Qiaolin Yu
  `a418db30` 2026-11-07T04:11:26Z
- **[[Fix] moe: Match the Triton renormalize sum order in the warp MoE router (#49017)](https://github.com/sgl-project/sglang/commit/780baaaf6173786b8baa4f20ad2aaaf77024d53f)** — Baizhou Zhang
  `780baaaf` 2026-11-07T04:03:20Z
- **[[Feat] scheduler: Run the state checks in the PD and PP loops and after a flush (#48997)](https://github.com/sgl-project/sglang/commit/baef1bbb2cdba2e4d007e2bb328bcccad2b7ea67)** — Liangsheng Yin
  `baef1bbb` 2026-11-07T03:58:48Z
- **[[Perf] diffusion: Tune LingBot MoE TMA for B300 SXM6 AC (#48470)](https://github.com/sgl-project/sglang/commit/b8529362f52cf5c0ed9f732f26fc973977f79c10)** — Xiaoyu Zhang
  `b8529362` 2026-11-07T03:54:18Z

<details><summary>📄 diff: [Fix] hicache: Return int page slots from `_storage_page_ind</summary>

```diff
diff --git a/python/sglang/srt/mem_cache/pool_host/base.py b/python/sglang/srt/mem_cache/pool_host/base.py
index f39387cc0fd5..ed0302434b46 100644
--- a/python/sglang/srt/mem_cache/pool_host/base.py
+++ b/python/sglang/srt/mem_cache/pool_host/base.py
@@ -625,9 +625,9 @@ def logical_page_size(self) -> int:
     def _storage_page_index(self, index: int) -> int:
         """Physical slot of the page an L3 key covers; storage backends address
         pages by logical (widened) slot, one physical page per rank under DCP."""
+        index = int(index)
         if self.dcp_size == 1:
             return index
-        index = int(index)
         if index < 0 or index % self.logical_page_size != 0:
             raise ValueError(
                 "DCP L3 page start must be a nonnegative logical page boundary."
diff --git a/test/registered/unit/mem_cache/test_hicache_dcp_host_pool.py b/test/registered/unit/mem_cache/test_hicache_dcp_host_pool.py
index 2831a1e13f2c..1311bbb4a843 100644
--- a/test/registered/unit/mem_cache/test_hicache_dcp_host_pool.py
+++ b/test/registered/unit/mem_cache/test_hicache_dcp_host_pool.py
@@ -18,10 +18,16 @@
 import torch
 
 from sglang.srt.layers.dcp.layout import maybe_dcp_kernel_indices
-from sglang.srt.managers.cache_controller import HiCacheController
+from sglang.srt.managers.cache_controller import (
+    HiCacheController,
+    check_dcp_l3_extra_pool,
+)
+from sglang.srt.mem_cache.hicache_storage import PoolName, l3_page_span
 from sglang.srt.mem_cache.kv.layout.mha import MHAView
 from sglang.srt.mem_cache.kv.layout.mla import MLAView
 from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
+from sglang.srt.mem_cache.pool_host.dsa import DSAIndexerPoolHost
+from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
 from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
 from sglang.test.ci.ci_register import register_cpu_ci
 from sglang.test.test_utils import CustomTestCase
@@ -530,6 +536,132 @@ def test_zero_copy_rejects_invalid_pages_and_split_buffers(self):
             pool.get_page_buffer_meta(torch.arange(128))
 
 
+def _make_indexer_host(anchor: MLATokenToKVPoolHost, stride: int = 6):
+    """CPU DSAIndexerPoolHost carrying only the storage-page state (the real
+    constructor needs a CUDA DSA device pool)."""
+    pool = DSAIndexerPoolHost.__new__(DSAIndexerPoolHost)
+    pool.page_size = anchor.page_size
+    pool.dcp_size = anchor.dcp_size
+    pool.dcp_rank = anchor.dcp_rank
+    pool.layout = anchor.layout
+    pool.size = anchor.size
+    pool.page_num = anchor.page_num
+    pool.layer_num = 2
+    pool.indexer_dtype = torch.uint8
+    pool.indexer_page_stride_size = stride
+    pool.indexer_page_num = (pool.size + pool.page_size + 1) // pool.page_size
+    shape = (
+        (pool.layer_num, pool.indexer_page_num, stride)
+        if pool.layout == "layer_first"
+        else (pool.indexer_page_num, pool.layer_num, 1, stride)
+    )
+    pool.index_k_with_scale_buffer = (
+        (torch.arange(torch.Size(shape).numel()) % 251).to(torch.uint8).reshape(shape)
+    )
+    return pool
+
+
+class TestDcpIndexerStoragePages(CustomTestCase):
+    """L3 keys address widened pages; the row-sharded indexer stores one
+    physical page per logical page, the same row the L2 transfer uses."""
+
+    @staticmethod
+    def _page(pool, page):
+        if pool.layout == "layer_first":
+            return pool.index_k_with_scale_buffer[:, page]
+        return pool.index_k_with_scale_buffer[page]
+
+    def test_logical_pages_map_to_physical_pages(self):
+        for dcp in (1, 2, 8):
+            for layout in ("layer_first", "page_first", "page_first_direct"):
+                with self.subTest(dcp=dcp, layout=layout):
... (truncated, 91 more lines)
```
</details>

<details><summary>📄 diff: [Perf] [NVIDIA] unquant: Autotune low-M BF16 GEMMs via Flash</summary>

```diff
diff --git a/python/sglang/srt/arg_groups/fields/exec_.py b/python/sglang/srt/arg_groups/fields/exec_.py
index 6755049752c4..7ac16d79c580 100644
--- a/python/sglang/srt/arg_groups/fields/exec_.py
+++ b/python/sglang/srt/arg_groups/fields/exec_.py
@@ -309,7 +309,7 @@ class ExecKernel(msgspec.Struct):
     bf16_gemm_backend: A[
         str,
         Arg(
-            help="Choose the backend for unquantized BF16 GEMM operations. Options: 'auto' (default; selects 'cutedsl' on SM10x GPUs, except deterministic inference selects 'torch'; otherwise uses cuBLAS via torch.nn.functional.linear), 'cutedsl' (SGLang JIT CuTe DSL TGV BF16 GEMM on SM10x; dispatches between the allowlisted low-M Split-K kernel, the CuTe DSL kernel, and cuBLAS), 'gemv', 'torch' (always uses cuBLAS via torch.nn.functional.linear).",
+            help="Choose the backend for unquantized BF16 GEMM operations. Options: 'auto' (default; selects 'cutedsl' on SM10x GPUs, except deterministic inference selects 'torch'; otherwise uses cuBLAS via torch.nn.functional.linear), 'cutedsl' (SGLang JIT CuTe DSL TGV BF16 GEMM on SM10x; M <= 32 uses FlashInfer's autotuned mm_bf16 (off with --disable-flashinfer-autotune; SM107 uses allowlisted warp Split-K instead), other shapes dispatch between the CuTe DSL kernel and cuBLAS), 'gemv', 'torch' (always uses cuBLAS via torch.nn.functional.linear).",
             cli_name="--bf16-gemm-backend",
             choices=["auto", "cutedsl", "gemv", "torch"],
         ),
diff --git a/python/sglang/srt/layers/quantization/unquant.py b/python/sglang/srt/layers/quantization/unquant.py
index 0df541059e5c..d0fdd153a979 100644
--- a/python/sglang/srt/layers/quantization/unquant.py
+++ b/python/sglang/srt/layers/quantization/unquant.py
@@ -3,7 +3,7 @@
 import collections
 import logging
 from enum import Enum
-from typing import TYPE_CHECKING, Callable, Iterable, List, Optional
+from typing import TYPE_CHECKING, Callable, List, Optional
 
 logger = logging.getLogger(__name__)
 
@@ -137,175 +137,13 @@ def is_gemv(self) -> bool:
 _use_hopper_bf16_gemv = None
 _hopper_bf16_tiny_gemv = None
 _use_hopper_bf16_tiny_gemv = None
-_splitk_tactic = None
-_run_splitk_dense = None
-_direct_default_tactic = None
-_prefer_direct = None
-_run_direct_dense = None
 _warp_splitk_tactic = None
 _run_warp_splitk_dense = None
-_enable_bf16_splitk_gemm = False
-
-# GB300 TP16 tactics measured under CUDA graph replay with PDL and cold weights.
-# Unlisted shapes, including M=64, retain the existing TGV/cuBLAS path.
-_BF16_SPLITK_TUNED_TACTICS = {
-    (1, 256, 8192): (64, 8, 4, 11),
-    (2, 256, 8192): (64, 8, 4, 11),
-    (4, 256, 8192): (64, 8, 4, 11),
-    (8, 256, 8192): (64, 8, 4, 11),
-    (16, 256, 8192): (64, 8, 4, 10),
-    (24, 256, 8192): (64, 8, 4, 11),
-    (32, 256, 8192): (64, 8, 4, 12),
-    (1, 512, 8192): (64, 8, 4, 11),
-    (2, 512, 8192): (64, 8, 4, 12),
-    (4, 512, 8192): (64, 8, 4, 10),
-    (8, 512, 8192): (64, 8, 4, 12),
-    (12, 512, 8192): (64, 8, 4, 12),
-    (16, 512, 8192): (64, 8, 4, 12),
-    (24, 512, 8192): (64, 8, 4, 12),
-    (32, 512, 8192): (64, 16, 4, 9),
-    (24, 2304, 8192): (64, 32, 2, 9),
-    (32, 2304, 8192): (64, 32, 2, 9),
-    (1, 2560, 8192): (64, 8, 2, 10),
-    (2, 2560, 8192): (64, 8, 2, 10),
-    (4, 2560, 8192): (64, 8, 2, 10),
-    (8, 2560, 8192): (64, 8, 2, 10),
-    (16, 2560, 8192): (64, 16, 2, 11),
-    (24, 2560, 8192): (64, 32, 2, 9),
-    (32, 2560, 8192): (64, 32, 2, 9),
-    # MuseSpark TP8 unpooler slice; only swept M are listed.
-    (4, 1024, 4096): (64, 8, 4, 12),
-    (8, 1024, 4096): (64, 8, 4, 12),
-    (12, 1024, 4096): (64, 16, 4, 8),
-    (16, 1024, 4096): (64, 16, 4, 8),
-    # MuseSpark TP4 qkvg and pooler/unpooler slices on GB300, same method;
-    # only swept M are listed; other M were not tuned and stay on TGV/cuBLAS.
-    (24, 4608, 8192): (64, 32, 2, 9),
-    (32, 4608, 8192): (64, 32, 2, 9),
-    (1, 1024, 8192): (64, 16, 4, 10),
-    (2, 1024, 8192): (64, 16, 4, 10),
-    (4, 1024, 8192): (64, 16, 4, 10),
-    (8, 1024, 8192): (64, 16, 4, 10),
-    (12, 1024, 8192): (64, 16, 4, 10),
... (truncated, 437 more lines)
```
</details>

<details><summary>📄 diff: [Fix] radix-cache: Use Rust TreeCore with PyTorch 2.15 in th</summary>

```diff
diff --git a/docker/Dockerfile.cu134 b/docker/Dockerfile.cu134
index 939e9cf731da..402f4563ecf8 100644
--- a/docker/Dockerfile.cu134
+++ b/docker/Dockerfile.cu134
@@ -894,9 +894,6 @@ ARG CUDA_VERSION
 ARG SGL_VERSION
 ARG USE_LATEST_SGLANG
 
-# TODO(Jialin): Set SGLANG_RUST_BUILD_MODE=never after Rust TreeCore supports
-# the PyTorch 2.15 used by this preview image.
-
 WORKDIR /sgl-workspace
 
 COPY --from=local_src /src /tmp/local_src
@@ -912,13 +909,19 @@ RUN if [ "$BRANCH_TYPE" = "local" ]; then \
     && rm -rf /tmp/local_src
 
 # Editable install (fast - dependencies already installed via constraints)
+# --no-build-isolation builds the Rust extensions (including the Rust TreeCore,
+# which links libtorch) against the image's torch==${TORCH_VERSION}; an isolated
+# build would compile them against python/pyproject.toml's [build-system] torch pin.
 # Clean up __pycache__/tests/pyc in same RUN to avoid writing ~28k files to layer
 RUN --mount=type=cache,target=/root/.cache/pip \
     cd /sgl-workspace/sglang \
-    && python3 -m pip install --no-deps -e "python[${BUILD_TYPE}]" \
+    && python3 -m pip install "setuptools>=61.0" "setuptools-rust>=1.11" "setuptools-scm>=8.0" wheel \
+    && python3 -m pip install --no-deps --no-build-isolation -e "python[${BUILD_TYPE}]" \
     && mkdir -p /root/.cache/huggingface /root/.cache/sglang \
     && ( find /opt/sglang/lib/python3.12/site-packages -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true )
 
+# Use the Rust extensions built above; never rebuild them at runtime.
+ENV SGLANG_RUST_BUILD_MODE=never
 
 # Install pre-built gateway artifacts from parallel builder
 COPY --from=gateway_builder /build/sgl-model-gateway-bin /opt/sglang/bin/sgl-model-gateway
diff --git a/python/sglang/srt/rust_extensions/torch_build.py b/python/sglang/srt/rust_extensions/torch_build.py
index e54c5935763f..f36e7fdbb90a 100644
--- a/python/sglang/srt/rust_extensions/torch_build.py
+++ b/python/sglang/srt/rust_extensions/torch_build.py
@@ -13,7 +13,7 @@
 from typing import Mapping
 
 _MIN_SUPPORTED_TORCH = (2, 11)
-_MAX_SUPPORTED_TORCH = (2, 14)
+_MAX_SUPPORTED_TORCH = (2, 15)
 
 
 @dataclass(frozen=True)
diff --git a/test/registered/rust/test_rust_extension.py b/test/registered/rust/test_rust_extension.py
index 15c8a55b7e76..8e8f734853f1 100644
--- a/test/registered/rust/test_rust_extension.py
+++ b/test/registered/rust/test_rust_extension.py
@@ -14,7 +14,11 @@
 
 from sglang.srt.rust_extensions import load_rust_extension
 from sglang.srt.rust_extensions import loader as rust_extension
-from sglang.srt.rust_extensions.torch_build import torch_build_configuration
+from sglang.srt.rust_extensions.torch_build import (
+    _MAX_SUPPORTED_TORCH,
+    _MIN_SUPPORTED_TORCH,
+    torch_build_configuration,
+)
 from sglang.test.ci.ci_register import register_cpu_ci
 from sglang.test.test_utils import CustomTestCase
 
@@ -534,8 +538,14 @@ def test_torch_build_configuration_is_versioned_and_relocatable(self):
             self.assertEqual(spaced_build.environment["CC_SHELL_ESCAPED_FLAGS"], "1")
             self.assertIn(f"'{spaced_header}'", spaced_build.environment["CXXFLAGS"])
 
-            fake_torch.__version__ = "2.15.0"
-            with self.assertRaisesRegex(RuntimeError, "PyTorch 2.11 through 2.14"):
+            fake_torch.__version__ = (
+                f"{_MAX_SUPPORTED_TORCH[0]}.{_MAX_SUPPORTED_TORCH[1] + 1}.0"
+            )
+            with self.assertRaisesRegex(
+                RuntimeError,
+                f"PyTorch {_MIN_SUPPORTED_TORCH[0]}.{_MIN_SUPPORTED_TORCH[1]} "
+                f"through {_MAX_SUPPORTED_TORCH[0]}.{_MAX_SUPPORTED_TORCH[1]}",
+            ):
                 torch_build_configuration(
... (truncated, 25 more lines)
```
</details>

<details><summary>📄 diff: [Feat] kimi-k3.1: HiCache under decode context parallelism (</summary>

```diff
diff --git a/python/sglang/srt/arg_groups/model_overrides/kimi_k3.py b/python/sglang/srt/arg_groups/model_overrides/kimi_k3.py
index a9e72b2a6215..4616d3be793f 100644
--- a/python/sglang/srt/arg_groups/model_overrides/kimi_k3.py
+++ b/python/sglang/srt/arg_groups/model_overrides/kimi_k3.py
@@ -191,11 +191,10 @@ def _kimi_k31_overrides(server_args: Any, hf_config: Any) -> dict:
         if (
             cfg.speculative_algorithm not in (None, "DSPARK", "DFLASH")
             or cfg.disaggregation_mode != "null"
-            or cfg.enable_hierarchical_cache
         ):
             raise ValueError(
                 "Kimi-K3.1 DCP supports only DSPARK and DFLASH speculative "
-                "decoding, and not PD disaggregation or HiCache yet."
+                "decoding, and not PD disaggregation yet."
             )
         if cfg.speculative_algorithm == "DSPARK":
             from sglang.srt.speculative.ragged_verify import (
diff --git a/python/sglang/srt/managers/cache_controller.py b/python/sglang/srt/managers/cache_controller.py
index 357e17404e95..8c1b2cf71008 100644
--- a/python/sglang/srt/managers/cache_controller.py
+++ b/python/sglang/srt/managers/cache_controller.py
@@ -298,6 +298,36 @@ def is_terminated(self) -> bool:
             return self._terminated_flag
 
 
+def check_dcp_l3_extra_pool(entry: Any, anchor: Any) -> None:
+    """Reject a host pool that HiCache L3 cannot store under DCP.
+
+    The DCP L3 path stores one physical page per logical page on each rank, so
+    every pool besides the MLA anchor must be either per-slot state or sharded
+    by the same DCP rows as the anchor.
+    """
+    from sglang.srt.layers.dcp.layout import dsa_shards_index_k
+    from sglang.srt.mem_cache.pool_host.dsa import DSAIndexerPoolHost
+    from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
+
+    pool = entry.host_pool
+    if pool is anchor:
+        return
+    if entry.name == PoolName.MAMBA and isinstance(pool, MambaPoolHost):
+        return  # KDA state: one slot per key
+    if (
+        entry.name == PoolName.INDEXER
+        and isinstance(pool, DSAIndexerPoolHost)
+        and dsa_shards_index_k()
+        and pool.dcp_size == anchor.dcp_size
+        and pool.size == anchor.size
+    ):
+        return  # index-K row-sharded like the MLA KV (Kimi-K3.1)
+    raise NotImplementedError(
+        f"HiCache L3 with DCP does not support the {entry.name} host pool "
+        f"({type(pool).__name__})."
+    )
+
+
 class HiCacheController:
     def __init__(
         self,
@@ -570,7 +600,6 @@ def attach_storage_backend(
                 raise NotImplementedError(
                     "HiCache L3 with DCP does not support buffer_only host memory."
                 )
-            from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
             from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
 
             if storage_backend not in ("file", "mooncake"):
@@ -584,17 +613,9 @@ def attach_storage_backend(
                 raise NotImplementedError(
                     "HiCache L3 with DCP requires a materialized MLA host pool."
                 )
-            if self.mem_pool_host is not self.storage_host_pool and any(
-                entry.host_pool is not self.storage_host_pool
-                and (
-                    entry.name != PoolName.MAMBA
-                    or not isinstance(entry.host_pool, MambaPoolHost)
-                )
-                for entry in self.mem_pool_host.entries
-            ):
-                raise NotImplementedError(
-                    "HiCache L3 with DCP supports only MLA KV and optional KDA state."
... (truncated, 262 more lines)
```
</details>

<details><summary>📄 diff: [Fix] kimi-k3: Cap cached_tokens at prompt_tokens in the Rus</summary>

```diff
diff --git a/rust/sglang-processor/src/openai/chat.rs b/rust/sglang-processor/src/openai/chat.rs
index f3af775697d9..60cbde2c4300 100644
--- a/rust/sglang-processor/src/openai/chat.rs
+++ b/rust/sglang-processor/src/openai/chat.rs
@@ -1420,10 +1420,14 @@ impl ChatResponder {
         events
     }
 
-    /// `_reported_prompt_tokens`.
+    /// `_reported_prompt_tokens` and `_reported_cached_tokens`.
     fn report_prompt_tokens(&self, meta: &mut GenerateMetaInfo) {
         meta.prompt_tokens =
             u64::from(meta.prompt_tokens).saturating_sub(self.unreported_prompt_tokens) as u32;
+        if self.unreported_prompt_tokens > 0 {
+            // A full-prompt cache hit also covers the prompt tokens left out above.
+            meta.cached_tokens = meta.cached_tokens.min(meta.prompt_tokens);
+        }
     }
 
     /// A `ToolCall` as Pydantic dumps it, with `_process_tool_call_id`'s id.
@@ -1668,4 +1672,94 @@ mod tests {
         assert!(created.len() > 1, "{events:?}");
         assert!(created.iter().all(|c| *c == json!(1)), "{created:?}");
     }
+
+    /// A K3 cache hit reaching the generation stub left out of `prompt_tokens`
+    /// is capped at `prompt_tokens`; one ending before the stub, and other
+    /// models, keep the engine's count. Unary, final-usage and continuous-usage
+    /// replies agree.
+    #[test]
+    fn kimi_k3_cached_tokens_exclude_the_generation_stub() {
+        let k3 = ChatModel {
+            chat_encoding_spec: Some("kimi_k3".into()),
+            ..ChatModel::default()
+        };
+        let other = ChatModel::default();
+        // (model, engine prompt, engine cached, reported prompt, reported cached)
+        let cases = [
+            (&k3, 2075, 2073, 2072, 2072),
+            (&k3, 2075, 2000, 2072, 2000),
+            (&other, 2075, 2073, 2075, 2073),
+        ];
+        let stream_options = [
+            None,
+            Some(json!({"include_usage": true})),
+            Some(json!({"continuous_usage_stats": true})),
+        ];
+        for (model, prompt, cached, want_prompt, want_cached) in cases {
+            for options in &stream_options {
+                let usage = one_frame_usage(model, options.as_ref(), prompt, cached);
+                assert_eq!(
+                    usage["prompt_tokens"],
+                    json!(want_prompt),
+                    "{options:?} {usage}"
+                );
+                assert_eq!(
+                    usage["prompt_tokens_details"]["cached_tokens"],
+                    json!(want_cached),
+                    "{options:?} {usage}"
+                );
+            }
+        }
+    }
+
+    /// The usage of a one-frame chat reply with cache reporting on: unary
+    /// without `stream_options`, streamed with them.
+    fn one_frame_usage(
+        model: &ChatModel,
+        stream_options: Option<&Value>,
+        prompt_tokens: u32,
+        cached_tokens: u32,
+    ) -> Value {
+        let mut body = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]});
+        if let Some(options) = stream_options {
+            body["stream"] = json!(true);
+            body["stream_options"] = options.clone();
+        }
+        let settings = OpenAiSettings {
+            enable_cache_report: true,
+            ..OpenAiSettings::default()
... (truncated, 36 more lines)
```
</details>

<details><summary>📄 diff: [Test] rust-server: Cover the embedded Rust frontend in a tw</summary>

```diff
diff --git a/test/registered/core/test_rust_server_multi_node.py b/test/registered/core/test_rust_server_multi_node.py
new file mode 100644
index 000000000000..661b982dfa80
--- /dev/null
+++ b/test/registered/core/test_rust_server_multi_node.py
@@ -0,0 +1,274 @@
+"""The embedded Rust server in a two-node engine, both nodes on one host.
+
+One TP2 engine spans two "nodes" (``--nnodes 2``), each a separate launcher
+process with one GPU: node 0 on GPU 0 and node 1 on GPU 1, sharing a
+``--dist-init-addr`` on loopback. With ``SGLANG_RUST_SERVER=1`` the Rust
+listener runs inside the global rank-0 scheduler on node 0. Node 1 hosts no
+listener: it runs only its scheduler and the launcher's placeholder health
+server on its own ``--port``, as in Python mode.
+
+Every request crosses nodes: the rank-0 scheduler broadcasts it to the node-1
+rank, and every forward runs collectives across both nodes.
+
+Usage:
+python3 -m unittest test_rust_server_multi_node
+"""
+
+import json
+import os
+import subprocess
+import unittest
+from urllib.parse import urlparse
+
+import requests
+
+from sglang.srt.utils import kill_process_tree
+from sglang.srt.utils.network import get_free_port_below_ephemeral
+from sglang.test.ci.ci_register import register_cuda_ci
+from sglang.test.test_utils import (
+    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
+    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
+    DEFAULT_URL_FOR_TEST,
+    CustomTestCase,
+    collect_process_tree_pids,
+    find_available_port,
+    is_rust_server_built,
+    popen_launch_server,
+    wait_for_gpu_release,
+)
+
+register_cuda_ci(est_time=300, stage="base-b", runner_config="2-gpu-large")
+
+# Maximum time this test allows a node to exit after its peer is killed.
+PEER_EXIT_TIMEOUT_SECS = 120
+
+
+class TwoNodeRustServerGang:
+    """One TP2 engine as two single-GPU launcher processes in Rust mode."""
+
+    def __init__(self, model: str):
+        parsed = urlparse(DEFAULT_URL_FOR_TEST)
+        self.host = parsed.hostname
+        self.node0_url = DEFAULT_URL_FOR_TEST
+        node1_port = find_available_port(parsed.port + 100)
+        self.node1_url = f"http://{self.host}:{node1_port}"
+        self.model = model
+        # Node 0's TCPStore binds this port; pick it below the ephemeral range
+        # so no outgoing connection can take it first.
+        dist_init_addr = f"{self.host}:{get_free_port_below_ephemeral()}"
+        self.common_args = [
+            "--tp-size",
+            "2",
+            "--nnodes",
+            "2",
+            "--dist-init-addr",
+            dist_init_addr,
+            # What is under test is the frontend and the cross-node plumbing,
+            # not kernels: portable backends and no CUDA graphs keep the two
+            # launches short on any runner GPU.
+            "--attention-backend",
+            "triton",
+            "--sampling-backend",
+            "pytorch",
+            "--cuda-graph-backend-decode",
+            "disabled",
... (truncated, 201 more lines)
```
</details>

<details><summary>📄 diff: [Test] load-snapshot: Pin the Python server mode in TestZmqR</summary>

```diff
diff --git a/test/registered/unit/managers/test_load_snapshot_backends.py b/test/registered/unit/managers/test_load_snapshot_backends.py
index b6f729a98310..ff7e0177c26a 100644
--- a/test/registered/unit/managers/test_load_snapshot_backends.py
+++ b/test/registered/unit/managers/test_load_snapshot_backends.py
@@ -330,6 +330,13 @@ class TestZmqReaderOwner(CustomTestCase):
 
     CALLERS = ("TokenizerManager", "MultiTokenizerRouter", "DataParallelController")
 
+    def setUp(self):
+        super().setUp()
+        # These are the Python server's callers. The Rust server runs no zmq
+        # collector (`test_should_use_zmq` covers it), so pin the mode against
+        # an inherited SGLANG_RUST_SERVER=1.
+        self.enterContext(envs.SGLANG_RUST_SERVER.override(False))
+
     def _owners(self, **overrides):
         """Publish a config and return the callers that claim the socket."""
         fields = dict(

```
</details>

<details><summary>📄 diff: [Refactor] scheduler: Run a prefill batch's KV calls from on</summary>

```diff
diff --git a/python/sglang/srt/managers/scheduler_components/batch_result_processor.py b/python/sglang/srt/managers/scheduler_components/batch_result_processor.py
index fba0075a96e6..05ef7ac8e2c7 100644
--- a/python/sglang/srt/managers/scheduler_components/batch_result_processor.py
+++ b/python/sglang/srt/managers/scheduler_components/batch_result_processor.py
@@ -2,6 +2,7 @@
 
 import logging
 from dataclasses import dataclass
+from functools import partial
 from http import HTTPStatus
 from typing import (
     TYPE_CHECKING,
@@ -310,6 +311,28 @@ def consume_auxiliary_output(
     ) -> None:
         output.consume(batch, cls._build_auxiliary_commits(batch, output_starts))
 
+    def _flush_prefill(
+        self,
+        batch: ScheduleBatch,
+        kv_ops: List[Callable[[], None]],
+        skip_stream_req: Optional[Req],
+    ) -> None:
+        """Run a prefill batch's queued KV calls in order, end the free group, then
+        send the outputs.
+
+        The loop queues each row's KV work (HiSparse finish, release, checkpoint,
+        staging admit) in row order, so an override that sends first moves only the
+        send. The send and the KV calls are coupled both ways: stream_output sets
+        finished_output and fills finished_len where it is None, and release_kv_cache
+        reads finished_len.
+        """
+        for op in kv_ops:
+            op()
+        self.token_to_kv_pool_allocator.free_group_end()
+        self.output_streamer.stream_output(
+            batch.reqs, batch.return_logprob, skip_stream_req
+        )
+
     def process_batch_result_prefill(
         self,
         batch: ScheduleBatch,
@@ -317,6 +340,8 @@ def process_batch_result_prefill(
     ):
         skip_stream_req = None
         self.token_to_kv_pool_allocator.free_group_begin()
+        # Each generation row's KV work, queued in row order for _flush_prefill.
+        kv_ops: List[Callable[[], None]] = []
 
         if self.is_generation:
             if result.copy_done is not None:
@@ -445,16 +470,27 @@ def process_batch_result_prefill(
                             self._maybe_collect_routed_experts(req)
                             self._maybe_collect_indexer_topk(req)
                         if batch.decoding_reqs and req in batch.decoding_reqs:
-                            self._prepare_kv_cache_release(req)
-                        self.request_retirer.release(
-                            req, checkpoint=discard_finish_reason is None
+                            kv_ops.append(partial(self._prepare_kv_cache_release, req))
+                        kv_ops.append(
+                            partial(
+                                self.request_retirer.release,
+                                req,
+                                checkpoint=discard_finish_reason is None,
+                            )
                         )
                         req.time_stats.set_completion_time()
                     elif not batch.decoding_reqs or req not in batch.decoding_reqs:
-                        checkpoint_kv_cache(req, self.tree_cache)
+                        kv_ops.append(
+                            partial(checkpoint_kv_cache, req, self.tree_cache)
+                        )
                         if get_memory().enable_hisparse:
                             self.leave_prefill_launched([req])
-                            self.hisparse_coordinator.admit_request_into_staging(req)
+                            kv_ops.append(
+                                partial(
+                                    self.hisparse_coordinator.admit_request_into_staging,
+                                    req,
+                                )
+                            )
... (truncated, 195 more lines)
```
</details>

<details><summary>📄 diff: [Fix] io_struct: Give each request its own sampling_params d</summary>

```diff
diff --git a/python/sglang/srt/managers/io_struct.py b/python/sglang/srt/managers/io_struct.py
index 83414ee3774e..adf062d5bd0b 100644
--- a/python/sglang/srt/managers/io_struct.py
+++ b/python/sglang/srt/managers/io_struct.py
@@ -577,6 +577,8 @@ def _normalize_single_inputs(self):
         """Normalize inputs for a single example."""
         if self.sampling_params is None:
             self.sampling_params = {}
+        else:
+            self.sampling_params = self.sampling_params.copy()
         if self.rid is None:
             self.rid = uuid.uuid4().hex
         if self.return_logprob is None:
@@ -757,11 +759,15 @@ def _normalize_audio_data(self, num):
     def _normalize_sampling_params(self, num):
         """Normalize sampling parameters for batch processing."""
         if self.sampling_params is None:
-            self.sampling_params = [{}] * num
+            self.sampling_params = [{} for _ in range(num)]
         elif isinstance(self.sampling_params, dict):
-            self.sampling_params = [self.sampling_params] * num
+            self.sampling_params = [self.sampling_params.copy() for _ in range(num)]
         else:  # Already a list
-            self.sampling_params = self.sampling_params * self.parallel_sample_num
+            self.sampling_params = [
+                params.copy()
+                for params in self.sampling_params * self.parallel_sample_num
+            ]
+        self.__dict__.pop("_sub_obj_cache", None)
 
     def _normalize_watermark(self, num):
         if isinstance(self.watermark, list):
@@ -1391,6 +1397,8 @@ def normalize_batch_and_arguments(self):
                 self.rid = uuid.uuid4().hex
             if self.sampling_params is None:
                 self.sampling_params = {}
+            else:
+                self.sampling_params = self.sampling_params.copy()
             self.sampling_params["max_new_tokens"] = 0
         else:
             if self.rid is None:
@@ -1399,14 +1407,21 @@ def normalize_batch_and_arguments(self):
                 assert isinstance(self.rid, list), "The rid should be a list."
 
             if self.sampling_params is None:
-                self.sampling_params = [{}] * self.batch_size
+                self.sampling_params = [{} for _ in range(self.batch_size)]
             elif isinstance(self.sampling_params, dict):
-                self.sampling_params = [self.sampling_params] * self.batch_size
+                self.sampling_params = [
+                    self.sampling_params.copy() for _ in range(self.batch_size)
+                ]
+            else:
+                self.sampling_params = [
+                    params.copy() for params in self.sampling_params
+                ]
             for i in range(self.batch_size):
                 self.sampling_params[i]["max_new_tokens"] = 0
 
             self._normalize_lora_paths(self.batch_size)
 
+        self.__dict__.pop("_sub_obj_cache", None)
         self._validate_rid_uniqueness()
 
     def _validate_mm_batch_lengths(self):
diff --git a/test/registered/unit/managers/test_io_struct.py b/test/registered/unit/managers/test_io_struct.py
index 2ed0ff599dc5..e6a5daad926a 100644
--- a/test/registered/unit/managers/test_io_struct.py
+++ b/test/registered/unit/managers/test_io_struct.py
@@ -433,6 +433,70 @@ def setUp(self):
             rid=["id1", "id2"],
         )
 
+    def test_sampling_params_are_owned_by_each_request(self):
+        """Adjusting one request's token budget must preserve peers and caller defaults."""
+        for n in (1, 2):
+            shared = {"max_new_tokens": 20, "n": n}
+            cases = (None,) if n == 1 else ()
+            cases += (
+                shared,
... (truncated, 164 more lines)
```
</details>

<details><summary>📄 diff: [Doc] [AMD] cookbook: Match the Kimi-K3.1-Pro MI35x single-n</summary>

```diff
diff --git a/docs/cookbook/autoregressive/Moonshotai/Kimi-K3.1-Pro.mdx b/docs/cookbook/autoregressive/Moonshotai/Kimi-K3.1-Pro.mdx
index c4b30fcdb623..6a87b44d02f2 100644
--- a/docs/cookbook/autoregressive/Moonshotai/Kimi-K3.1-Pro.mdx
+++ b/docs/cookbook/autoregressive/Moonshotai/Kimi-K3.1-Pro.mdx
@@ -247,16 +247,16 @@ Kimi K3.1 Pro always thinks; set the depth with the top-level `reasoning_effort`
 - **Environment.** Every MI350X / MI355X cell sets `SGLANG_USE_AITER=1 SGLANG_AITER_K3_OPT=1 AITER_FLYDSL_FORCE=1 AITER_SITUV2_A8W4=1`, as on [Kimi-K3](/cookbook/autoregressive/Moonshotai/Kimi-K3#amd-env): the AITER ROCm path, its K3 fused kernels, the FlyDSL MoE kernels and the SiTU v2 MoE with A8W4 activation quantization.
 - **Pick the recipe by agent count.**
   - 1 to 4: `Unified` `Low-Latency (TP8)`, on the monokernel.
-  - 8 to 56: `Unified` `High-Throughput (DCP8)`; set `--max-running-requests` and `--cuda-graph-max-bs-decode` to the agent count (64 for 56). At 8 agents, add the monokernel and turn HiCache off.
+  - 8 to 56: `Unified` `High-Throughput (DCP8)` with HiCache; set `--max-running-requests` and `--cuda-graph-max-bs-decode` to the agent count. The cell is the 48-agent shape: 32k prefill chunks every 8 decode steps, block 5, and the row-sharded prefill (`SGLANG_ENABLE_ROCM_K3_ROW_SHARDED_ROUTED_FRONT=1`, `SGLANG_ENABLE_ROCM_K3_ROW_SHARDED_STREAM=1`) with `SGLANG_OPT_DFLASH_RING_LOCS_FUSED=1`. From 8 to 32 agents, use `--chunked-prefill-size 16384`, `--prefill-decode-interval 16` and block 6, without those three variables.
   - Past 56: the 1P1D, `High-Throughput` prefill and decode on two nodes (see [On AMD MI355X](#pd-amd)).
-- **DFLASH** proposes 5 draft tokens (block 6) on the Triton draft attention backend, with ReplaySSM on every role but the PD prefill.
-- **Monokernel (1 to 8 agents).** `--enable-amd-monokernel` with `SGLANG_AMD_MONO_K31_WIDE=1` runs decode and verify steps through the persistent kernels of `sgl-amd-mono` (in the image); other steps use the standard forward. It takes TP8 or DCP8 with DFLASH but not HiCache, and SGLang stops at startup with the reasons if a setup is outside what it supports. The log shows `[k31 wide] first wide step` once it engages.
-- **HiCache.** `L1+L2 (host)` is the default: write-back, `--hicache-ratio 2.5` and `SGLANG_HICACHE_MAMBA_HOST_RATIO=12` (8 on the PD prefill). Low-Latency and the PD decode run without it.
+- **DFLASH** proposes 5 draft tokens (block 6) on the Triton draft attention backend, 4 (block 5) on the single-node `High-Throughput` cell from 48 agents, with ReplaySSM on every role but the PD prefill. The single-node cells add `--speculative-use-block-verification`, which the measured acceptance used for sampled drafts.
+- **Monokernel (1 to 4 agents).** `--enable-amd-monokernel` with `SGLANG_AMD_MONO_K31_WIDE=1` runs decode and verify steps through the persistent kernels of `sgl-amd-mono` (in the image); other steps use the standard forward. It takes TP8 or DCP8 with DFLASH but not HiCache, and SGLang stops at startup with the reasons if a setup is outside what it supports. The log shows `[k31 wide] first wide step` once it engages. The `Low-Latency` cell keeps 150 KDA state slots (`--max-mamba-cache-size 150`) so agent sessions' prefixes stay cached.
+- **HiCache.** `L1+L2 (host)` is the default: write-back, `--hicache-ratio 2.5` and `SGLANG_HICACHE_MAMBA_HOST_RATIO=10` (8 on the PD prefill). Low-Latency and the PD decode run without it.
   - On agentic replay the host KDA-state slots bound the prefix hit rate before the host KV does, so `SGLANG_HICACHE_MAMBA_HOST_RATIO` sizes them apart from `--hicache-ratio`. The recipes keep every KDA state per path; `--mamba-max-states-per-path 1` lowered the hit rate.
-  - The host pools pin about 190 GB per GPU at DCP8, about 1.5 TB per node. `HSA_USERPTR_FOR_PAGED_MEM=0` backs them with `hipHostMalloc` (GTT memory, which never migrates); page migration in registered host memory stalls every GPU queue.
+  - The host pools pin 170-185 GB per GPU at DCP8, and the node's 8 ranks share its GTT limit (half of RAM, 1511 GiB on MI355X). At ratio 12 they overflowed it at 32 agents, and `hipHostMalloc` failed at startup. `HSA_USERPTR_FOR_PAGED_MEM=0` backs them with `hipHostMalloc` (GTT memory, which never migrates); page migration in registered host memory stalls every GPU queue.
   - L3 has not run on MI350X / MI355X.
 - **Memory.** `SGLANG_ROCM_RUNTIME_RESERVE_GB` (10 at DCP8, 6 at TP8) keeps GPU memory free for the HIP queues' scratch, which is allocated outside torch. `--mem-fraction-static` 0.87 or 0.88 overflowed GTT at startup or ran slower on the single-node recipe.
-- **`--prefill-decode-interval 16`** runs 16 decode steps after each prefill batch before the next prefill. At 48 agents, 16 matched the throughput of 8 with 16% more interactivity; p90 TTFT rose from 5.2 s to 10.4 s.
+- **`--prefill-decode-interval`** sets how many decode steps run after each prefill batch before the next prefill. The single-node `High-Throughput` cell runs 8 with 32k prefill chunks from 48 agents; below 48 agents, 16 with 16k chunks.
 - **MI350X** runs the MI355X commands, single-node and 1P1D.
 
 ## 3. Advanced Usage
diff --git a/docs/src/snippets/configs/moonshotai/kimi-k3.1-pro.jsx b/docs/src/snippets/configs/moonshotai/kimi-k3.1-pro.jsx
index 671531953b37..49702b5017ad 100644
--- a/docs/src/snippets/configs/moonshotai/kimi-k3.1-pro.jsx
+++ b/docs/src/snippets/configs/moonshotai/kimi-k3.1-pro.jsx
@@ -60,8 +60,11 @@ export const config = {
   },
   // Throughput shapes verify K = 5 (block 6): at large batch the verify rows (requests ×
   // block) dominate a step. The High-Throughput decode and every MI35x cell (whose PD prefill must
-  // match its decode) use block 6; everything else runs K = 7 (block 8).
+  // match its decode) use block 6; everything else runs K = 7 (block 8). The MI35x unified
+  // High-Throughput cell, at its 48-agent shape, verifies K = 4 (block 5): from 48 requests the
+  // shorter verify outweighs its lower acceptance.
   dflashBlock(s) {
+    if (config.isAmd(s) && s.pdMode === "unified" && s.strategy === "high-throughput") return 5;
     return (s.pdMode === "decode" && s.strategy === "high-throughput") || config.isAmd(s) ? 6 : 8;
   },
 
@@ -133,6 +136,9 @@ export const config = {
             // KV pool collapses (5.13M -> 0.77M tokens measured on decode). The PD prefill role
             // never runs verify and rejects the flag at startup.
             ...(s.pdMode !== "prefill" ? ["--enable-linear-replayssm-spec"] : []),
+            // The MI35x single-node acceptance was measured with block verification of sampled
+            // drafts; greedy requests skip it.
+            ...(config.isAmd(s) && s.pdMode === "unified" ? ["--speculative-use-block-verification"] : []),
           ],
           hints: (s) =>
             s.pdMode === "decode" && s.strategy === "high-throughput"
@@ -208,8 +214,8 @@ export const config = {
             "--hicache-ratio 2.5",
           ],
           // Keeps every rank on the Rust TreeCore; write-back asserts on a mix with the Python one.
-          env: (s) => [`SGLANG_HICACHE_MAMBA_HOST_RATIO=${s.pdMode === "prefill" ? 8 : 12}`, "RUSTUP_TOOLCHAIN=stable"],
-          hints: ["The host pools pin about 190 GB of host memory per GPU; check the node's GTT limit before raising either ratio."],
+          env: (s) => [`SGLANG_HICACHE_MAMBA_HOST_RATIO=${s.pdMode === "prefill" ? 8 : 10}`, "RUSTUP_TOOLCHAIN=stable"],
+          hints: ["The host pools pin 170-185 GB of host memory per GPU, and the 8 ranks share the node's GTT (1511 GiB on MI355X); ratio 12 overflowed it at 32 agents. Check the node's GTT limit before raising either ratio."],
         },
         { id: "amd-off", label: "Off", showWhen: (s) => config.isAmd(s) },
         {
@@ -1066,7 +1072,8 @@ sgl-router \\
         "--cuda-graph-max-bs-decode 4",
         "--schedule-policy lpm",
         "--prefill-decode-interval 16",
-        "--max-mamba-cache-size 60",
+        // 150 state slots keep agent sessions' prefixes cached.
+        "--max-mamba-cache-size 150",
         "--mamba-radix-cache-strategy extra_buffer_lazy",
         "--cuda-graph-backend-prefill breakable",
         "--cuda-graph-bs-prefill 256 512 768 1024 1536 2048 3072 4096",
@@ -1094,6 +1101,11 @@ sgl-router \\
         "SGLANG_ROCM_RUNTIME_RESERVE_GB=10",
         "SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1",
         "HSA_USERPTR_FOR_PAGED_MEM=0",
+        // From 48 agents: the MoE front and residual stream on row shards in prefill (no extra
... (truncated, 60 more lines)
```
</details>

<details><summary>📄 diff: [Fix] mem_cache: Chunk MambaPool temporal state copy to avoi</summary>

```diff
diff --git a/python/sglang/srt/mem_cache/memory_pool.py b/python/sglang/srt/mem_cache/memory_pool.py
index e475618ac0f7..986397be133f 100644
--- a/python/sglang/srt/mem_cache/memory_pool.py
+++ b/python/sglang/srt/mem_cache/memory_pool.py
@@ -117,6 +117,8 @@
 _MAMBA_DEBUG_ASSERTS = os.environ.get("SGLANG_MAMBA_DEBUG_ASSERTS", "0") == "1"
 
 GB = 1024 * 1024 * 1024
+# Arbitrary; no evidence this is the best chunk size for temporal state copies.
+_MAMBA_TEMPORAL_COPY_CHUNK_BYTES = 64 * 1024 * 1024
 _is_cuda = is_cuda()
 _is_npu = is_npu()
 _is_cpu = is_cpu()
@@ -1183,6 +1185,30 @@ def clear_slots(self, indices: torch.Tensor):
         if temporal.numel() > 0:
             temporal.index_fill_(1, idx64, 0)
 
+    def _copy_temporal_slots(
+        self, *, src_indices: torch.Tensor, dst_indices: torch.Tensor
+    ) -> None:
+        temporal = self.mamba_cache.temporal
+        if temporal.numel() == 0:
+            return
+        num_slots = dst_indices.shape[0]
+        slot_bytes = temporal.element_size() * (temporal.numel() // temporal.shape[1])
+        step = max(1, _MAMBA_TEMPORAL_COPY_CHUNK_BYTES // slot_bytes)
+        if num_slots <= step:
+            temporal[:, dst_indices] = temporal[:, src_indices]
+            return
+        # A batched gather allocates num_slots * slot_bytes at once and can OOM
+        # under admission bursts; chunking requires disjoint src/dst slots.
+        if envs.SGLANG_DEBUG_MEMORY_POOL.get():
+            overlap = set(src_indices.tolist()) & set(dst_indices.tolist())
+            assert not overlap, (
+                "chunked temporal copy requires disjoint src/dst slots; "
+                f"overlap={sorted(overlap)}"
+            )
+        for start in range(0, num_slots, step):
+            chunk = slice(start, start + step)
+            temporal[:, dst_indices[chunk]] = temporal[:, src_indices[chunk]]
+
     def copy_from(self, src_indices: torch.Tensor, dst_indices: torch.Tensor):
         """Clone mamba state (conv + temporal) from src slots into dst slots.
 
@@ -1212,17 +1238,13 @@ def copy_from(self, src_indices: torch.Tensor, dst_indices: torch.Tensor):
                     f"overlap={sorted(overlap)}"
                 )
             fused_copy_conv_slots(self._conv_slot_desc, src_indices, dst_indices)
-            temporal = self.mamba_cache.temporal
-            if temporal.numel() > 0:
-                temporal[:, dst_indices] = temporal[:, src_indices]
+            self._copy_temporal_slots(src_indices=src_indices, dst_indices=dst_indices)
         else:
             for i in range(len(self.mamba_cache.conv)):
                 self.mamba_cache.conv[i][:, dst_indices] = self.mamba_cache.conv[i][
                     :, src_indices
                 ]
-            self.mamba_cache.temporal[:, dst_indices] = self.mamba_cache.temporal[
-                :, src_indices
-            ]
+            self._copy_temporal_slots(src_indices=src_indices, dst_indices=dst_indices)
         if self.replayssm_write_pos is not None:
             self.replayssm_write_pos[dst_indices] = 0
         for sibling in self._slot_siblings:
diff --git a/test/registered/unit/mem_cache/test_mamba_unittest.py b/test/registered/unit/mem_cache/test_mamba_unittest.py
index 9da87f7fc335..59422378870d 100755
--- a/test/registered/unit/mem_cache/test_mamba_unittest.py
+++ b/test/registered/unit/mem_cache/test_mamba_unittest.py
@@ -9,6 +9,7 @@
 from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
 from sglang.srt.environ import envs
 from sglang.srt.managers.schedule_batch import Req
+from sglang.srt.mem_cache import memory_pool as memory_pool_module
 from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
 from sglang.srt.mem_cache.memory_pool import (
     HybridLinearKVPool,
@@ -462,6 +463,63 @@ def aligned_for(chunk_size, track_seqlens, prefix_lens):
         server_args._mamba_cache_chunk_size = FLA_CHUNK_SIZE
         set_global_server_args_for_scheduler(server_args)
 
... (truncated, 61 more lines)
```
</details>

<details><summary>📄 diff: [Fix] rust-server: Let DP rank 0's drain wait for every atte</summary>

```diff
diff --git a/python/sglang/srt/entrypoints/engine.py b/python/sglang/srt/entrypoints/engine.py
index 88143bbd6910..38142692dc28 100644
--- a/python/sglang/srt/entrypoints/engine.py
+++ b/python/sglang/srt/entrypoints/engine.py
@@ -2142,8 +2142,11 @@ def _block_until_follower_schedulers_exit(
     backstop. Rust listeners on this node (one per attention DP group whose
     first rank is here) still drain as node 0's do: SIGTERM is relayed to them
     (``RustServerSigtermDrain``), so their health answers 503 and their
-    in-flight requests finish before they close; their schedulers then exit on
-    rank 0's ShutdownReq. Without a local listener, the caller has already
+    in-flight requests finish before they close; rank 0's drain waits for them
+    (the MLP-sync gather carries every DP listener's drain state, and a drain
+    on any listener begins on all of them), and their schedulers then exit on
+    rank 0's ShutdownReq. Without a SIGTERM here, the first scheduler exiting
+    0 bounds the others. Without a local listener, the caller has already
     routed SIGTERM to ``Engine._handle_follower_sigterm``.
     """
     on_main_thread = threading.current_thread() is threading.main_thread()
diff --git a/python/sglang/srt/managers/scheduler.py b/python/sglang/srt/managers/scheduler.py
index 120d5612d7fd..c1f8a88c2fa9 100644
--- a/python/sglang/srt/managers/scheduler.py
+++ b/python/sglang/srt/managers/scheduler.py
@@ -258,7 +258,12 @@
 from sglang.srt.managers.scheduler_components.deferred_responses import (
     DeferredResponses,
 )
-from sglang.srt.managers.scheduler_components.dp_attn import SchedulerDPAttnAdapter
+from sglang.srt.managers.scheduler_components.dp_attn import (
+    DRAIN_STATE_NO_LISTENER,
+    MLP_SYNC_CARRIES_DRAIN_STATE,
+    SchedulerDPAttnAdapter,
+    should_skip_scheduler_all_gather,
+)
 from sglang.srt.managers.scheduler_components.dynamic_chunk_sizer import (
     DynamicChunkSizer,
 )
@@ -2553,7 +2558,8 @@ def maybe_init_rust_server(self, port_args: PortArgs) -> None:
             self, frontend_cpu_share=frontend_cpu_share
         )
         rust_server.install_sigterm_drain(
-            notify=drain_notify_name(port_args.instance_id)
+            notify=drain_notify_name(port_args.instance_id),
+            hold_drain_end=self._holds_rust_drain_end(),
         )
         self.rust_server = rust_server
         # The rust server *is* the ingress source: SchedulerRequestReceiver
@@ -2565,6 +2571,31 @@ def maybe_init_rust_server(self, port_args: PortArgs) -> None:
             rust_server, can_empty_cache=lambda: not self._engine_paused
         )
 
+    def _holds_rust_drain_end(self) -> bool:
+        """Whether this listener's drain ends only once every attention-DP
+        listener closed (``RustServer.on_drain_rows``): its ``ShutdownReq``
+        stops every rank. The DP groups share their drain state through the
+        MLP-sync gather of every step, which PP > 1 and
+        ``SGLANG_SCHEDULER_SKIP_ALL_GATHER`` do not run."""
+        parallel = get_parallel()
+        if parallel.attn_dp_size == 1 or parallel.tp_rank != 0:
+            return False
+        if (
+            parallel.dp_size == 1
+            and len(self.rust_server_endpoints) == parallel.attn_dp_size
+            and self.require_mlp_sync
+            and not should_skip_scheduler_all_gather(parallel.num_dp_ranks)
+            and parallel.pp_size == 1
+            and MLP_SYNC_CARRIES_DRAIN_STATE
+        ):
+            return True
+        logger.warning(
+            "rust server: no per-step MLP-sync gather between the attention-DP "
+            "listeners; DP rank 0's drain ends once its own listener closed, so "
+            "the other DP listeners' last requests may be cut"
+        )
+        return False
+
     def get_prefill_adder_class(self) -> type[PrefillAdder]:
         return UocrPrefillAdder if self.uocr_runtime is not None else PrefillAdder
 
@@ -2649,8 +2680,21 @@ def init_dp_attn_adapter(self) -> None:
             get_require_mlp_sync=lambda: self.require_mlp_sync,
... (truncated, 1293 more lines)
```
</details>

<details><summary>📄 diff: [Perf] kimi-k3.1: Faster DCP decode reduce (push-combine, V </summary>

```diff
diff --git a/python/sglang/kernels/jit/csrc/dsa/dcp_push_combine.cuh b/python/sglang/kernels/jit/csrc/dsa/dcp_push_combine.cuh
new file mode 100644
index 000000000000..89fa12ec7d95
--- /dev/null
+++ b/python/sglang/kernels/jit/csrc/dsa/dcp_push_combine.cuh
@@ -0,0 +1,271 @@
+#include <sgl_kernel/tensor.h>
+#include <sgl_kernel/utils.h>
+
+#include <sgl_kernel/runtime.cuh>
+#include <sgl_kernel/utils.cuh>
+
+#include <sgl_kernel/distributed/communicator.cuh>
+#include <sgl_kernel/distributed/ptx.cuh>
+
+#include <tvm/ffi/extra/stl.h>
+
+#include <algorithm>
+#include <cmath>
+#include <cstdint>
+
+// DCP decode reduce in one kernel: every rank holds the attention partials of
+// ALL heads over its own KV shard ([B, H, D], with the per-head LSE) and owns
+// H / N heads of the result. Each rank pushes the partials of the heads rank n
+// owns straight into rank n's slot of the CustomAllReduceV2 push plane over
+// NVLink, then combines the N partials of its own heads by LSE weight -- no
+// staging copy, no separate all-to-all, no separate combine.
+//
+// Arrival is lamport-style: a slot reads +0.0 until its producer has written it
+// (a producer rewrites +0.0 words to -0.0), and the consumer restores +0.0
+// after reading, so the plane needs no barrier. Slot (dst, src) holds rows
+// R = b * (H / N) + j for dst's local head j: R * 256 B of payload (D = 128 bf16)
+// then, at `lse_offset`, one 16 B word [lse, -0, -0, -0] per row.
+//
+// A warp owns a row at a time, on the same (block, warp) on every rank, so the
+// producer and the consumer of a row share a block and its phase counter; the
+// plane is this op's own, so a call may use fewer blocks than the plane has.
+
+namespace sglang {
+
+using device::distributed::Counter;
+using host::distributed::CommunicatorRef;
+
+namespace dcp_push_combine {
+
+constexpr uint32_t kHeadDim = 128;
+constexpr uint32_t kRowBytes = kHeadDim * 2;   // bf16 payload
+constexpr uint32_t kRowVecs = kRowBytes / 16;  // 16
+constexpr uint32_t kBlockSize = 256;
+constexpr uint32_t kWarps = kBlockSize / 32;
+constexpr uint32_t kNegZeroWord = 0x80000000u;
+
+struct alignas(16) Vec16 {
+  uint32_t w[4];
+};
+
+template <uint32_t kWorldSize>
+struct Params {
+  const uint8_t* v;    // [B, H, kHeadDim] bf16, last dim contiguous
+  int64_t v_stride_b;  // bytes
+  int64_t v_stride_h;  // bytes
+  const float* lse;    // [B, H]
+  int64_t lse_stride_b;
+  int64_t lse_stride_h;
+  uint8_t* out;  // [B, H / N * kHeadDim] bf16, contiguous
+  uint8_t* workspaces[kWorldSize];
+  Counter* counter;
+  uint32_t slot_bytes;
+  uint32_t lse_offset;
+  uint32_t rank;
+  uint32_t num_rows;  // B * H / N
+  uint32_t heads_per_rank;
+  bool lse_base_e;
+};
+
+SGL_DEVICE float bf16_lo(uint32_t w) {
+  return __uint_as_float(w << 16);
+}
+
+SGL_DEVICE float bf16_hi(uint32_t w) {
... (truncated, 1483 more lines)
```
</details>

<details><summary>📄 diff: [Feat] kimi-k3.1: DSPARK and DFLASH under decode context par</summary>

```diff
diff --git a/python/sglang/srt/arg_groups/model_overrides/kimi_k3.py b/python/sglang/srt/arg_groups/model_overrides/kimi_k3.py
index 20c49310c322..a9e72b2a6215 100644
--- a/python/sglang/srt/arg_groups/model_overrides/kimi_k3.py
+++ b/python/sglang/srt/arg_groups/model_overrides/kimi_k3.py
@@ -189,14 +189,35 @@ def _kimi_k31_overrides(server_args: Any, hf_config: Any) -> dict:
     overrides = {}
     if cfg.dcp_size > 1:
         if (
-            cfg.speculative_algorithm is not None
+            cfg.speculative_algorithm not in (None, "DSPARK", "DFLASH")
             or cfg.disaggregation_mode != "null"
             or cfg.enable_hierarchical_cache
         ):
             raise ValueError(
-                "Kimi-K3.1 DCP does not support speculative decoding, PD "
-                "disaggregation or HiCache yet."
+                "Kimi-K3.1 DCP supports only DSPARK and DFLASH speculative "
+                "decoding, and not PD disaggregation or HiCache yet."
             )
+        if cfg.speculative_algorithm == "DSPARK":
+            from sglang.srt.speculative.ragged_verify import (
+                RaggedVerifyMode,
+                read_ragged_verify_mode,
+            )
+
+            if read_ragged_verify_mode() is not RaggedVerifyMode.STATIC:
+                raise ValueError(
+                    "Kimi-K3.1 DCP + DSPARK requires SGLANG_RAGGED_VERIFY_MODE=static."
+                )
+        if cfg.speculative_algorithm == "DFLASH":
+            if cfg.speculative_draft_attention_backend not in (None, "cutedsl_mla"):
+                raise ValueError(
+                    "Kimi-K3.1 DCP + DFLASH needs "
+                    "--speculative-draft-attention-backend cutedsl_mla."
+                )
+            overrides["speculative_draft_attention_backend"] = "cutedsl_mla"
+        if cfg.speculative_algorithm is not None:
+            # NOTE(yhyang201): DCP target verify runs the DSA decode path; each
+            # rank attends over its KV shard and the (o, lse) partials are merged.
+            overrides["speculative_attention_mode"] = "decode"
         if (
             cfg.dsa_topk_backend != "sgl-kernel"
             or not envs.SGLANG_OPT_USE_TOPK_V2.get()
diff --git a/python/sglang/srt/arg_groups/speculative_hook.py b/python/sglang/srt/arg_groups/speculative_hook.py
index 4958abebf5a4..3671fe1d8bf6 100644
--- a/python/sglang/srt/arg_groups/speculative_hook.py
+++ b/python/sglang/srt/arg_groups/speculative_hook.py
@@ -937,6 +937,17 @@ def is_supported_backend(backend: str) -> bool:
             return current_platform.supports_speculative_draft_attention_backend(
                 "DFLASH", backend
             )
+        if backend == "cutedsl_mla":
+            from sglang.srt.hf_transformers import get_config
+            from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config
+
+            draft_hf_config = get_config(
+                cfg.speculative_draft_model_path,
+                trust_remote_code=cfg.trust_remote_code,
+                revision=cfg.speculative_draft_model_revision,
+                model_override_args=json.loads(cfg.json_model_override_args),
+            )
+            return parse_dflash_draft_config(draft_hf_config=draft_hf_config).is_mla
         return backend in supported_draft_backends
 
     def get_fallback_backend() -> str:
diff --git a/python/sglang/srt/layers/attention/dsa/dsa_indexer.py b/python/sglang/srt/layers/attention/dsa/dsa_indexer.py
index de16544c52da..4c0d621d6995 100644
--- a/python/sglang/srt/layers/attention/dsa/dsa_indexer.py
+++ b/python/sglang/srt/layers/attention/dsa/dsa_indexer.py
@@ -1409,9 +1409,11 @@ def _get_topk_paged(
         B = metadata.get_seqlens_int32().shape[0]
         next_n = q_offset // B if B > 0 else 0
         dcp_size, _ = _index_k_dcp()
-        assert dcp_size == 1 or forward_batch.forward_mode.is_decode_or_idle(), (
-            "DSA DCP runs the paged top-k in decode only"
-        )
+        assert (
+            dcp_size == 1
+            or forward_batch.forward_mode.is_decode_or_idle()
+            or forward_batch.forward_mode.is_target_verify()
... (truncated, 297 more lines)
```
</details>

<details><summary>📄 diff: [Perf] [NVIDIA] fp8: Let the Hopper block GEMV take K=6144 f</summary>

```diff
diff --git a/python/sglang/kernels/ops/gemm/hopper_fp8_block_gemv.py b/python/sglang/kernels/ops/gemm/hopper_fp8_block_gemv.py
index 8ccd25a55725..ccb949b1cb01 100644
--- a/python/sglang/kernels/ops/gemm/hopper_fp8_block_gemv.py
+++ b/python/sglang/kernels/ops/gemm/hopper_fp8_block_gemv.py
@@ -12,6 +12,10 @@
 
 _MAX_M = 16
 _MAX_K = 4096  # 16 x K bytes of activation slab in shared memory
+# MiniMax M3.1 projections, M <= 8 only (8 x K bytes of slab). Measured against
+# DeepGEMM on H200: this K pays up to this N, N=9856 is slower.
+_WIDE_K = 6144
+_WIDE_K_MAX_N = 8192
 _MAX_SIDE_N = 8  # side projection columns, one extra block each
 
 # Occupancy inputs measured with ncu on H20 for the shipped variants.
@@ -89,11 +93,15 @@ def use_hopper_fp8_block_gemv(
     if not (
         1 <= m <= _MAX_M
         and tuple(block_size) == (128, 128)
-        and k % 128 == 0
-        and 128 <= k <= _MAX_K
+        and (128 <= k <= _MAX_K or k == _WIDE_K)
+        # An odd K block count above one can neither split across warps nor unroll the
+        # loads; DeepGEMM wins there on H200 (K=384: GEMV takes 1.1x-1.4x the time).
+        and (k == 128 or k % 256 == 0)
         and n >= 16
     ):
         return False
+    if k == _WIDE_K:
+        return m <= 8 and n <= _WIDE_K_MAX_N
     # Past M=8 a second column group doubles MMA work; it wins on H20 only for N <= 1536.
     return m <= 8 or n <= 1536
 

```
</details>

<details><summary>📄 diff: [Perf] [NVIDIA] muse-spark: Bound B200 projection dispatch a</summary>

```diff
diff --git a/python/sglang/kernels/ops/communication/muse_spark_all_gather.py b/python/sglang/kernels/ops/communication/muse_spark_all_gather.py
index fb14b13d6cb7..9c3bac773a66 100644
--- a/python/sglang/kernels/ops/communication/muse_spark_all_gather.py
+++ b/python/sglang/kernels/ops/communication/muse_spark_all_gather.py
@@ -32,6 +32,7 @@
     )
 
 SMALL_PROJECTION_MAX_TOKENS = 8
+B200_PROJECTION_MAX_TOKENS = 128
 MAX_TOKENS = 512
 _VEC = 8
 
@@ -75,6 +76,12 @@ def gemm_ag_supported(
                 and n in (4096, 8192)
                 and current_arch().is_sm(10, 3)
             )
+            or (
+                comm.world_size == 8
+                and n in (4096, 8192)
+                and m <= B200_PROJECTION_MAX_TOKENS
+                and torch.cuda.get_device_capability(x.device) == (10, 0)
+            )
         )
         and weight.shape[1] == k
         and n % (comm.world_size * _VEC) == 0
diff --git a/python/sglang/kernels/ops/memory/allocator.py b/python/sglang/kernels/ops/memory/allocator.py
index f0f4a77b6a75..c61a90cb7c5e 100644
--- a/python/sglang/kernels/ops/memory/allocator.py
+++ b/python/sglang/kernels/ops/memory/allocator.py
@@ -2,6 +2,14 @@
 import triton
 import triton.language as tl
 
+# The request-to-token assignment scan still specializes on this width.
+# Paged extend/decode allocation below uses fixed blocks for every batch size.
+_MIN_BATCH_SCAN_WIDTH = 64
+
+
+def batch_scan_width(bs: int) -> int:
+    return max(1 << max(bs - 1, 0).bit_length(), _MIN_BATCH_SCAN_WIDTH)
+
 
 @triton.jit(do_not_specialize=["num_pages", "full_page_representatives"])
 def get_and_clear_swa_pages_kernel(
@@ -119,15 +127,27 @@ def alloc_extend_kernel(
     last_loc_ptr,
     free_page_ptr,
     out_indices,
-    bs_upper: tl.constexpr,
     page_size: tl.constexpr,
 ):
     pid = tl.program_id(0)
 
-    load_offset = tl.arange(0, bs_upper)
-    seq_lens = tl.load(seq_lens_ptr + load_offset, mask=load_offset <= pid)
-    pre_lens = tl.load(pre_lens_ptr + load_offset, mask=load_offset <= pid)
+    # A fixed-width scan avoids recompilation as the running batch grows.
+    BLOCK_BS: tl.constexpr = 256
+    load_offset = tl.arange(0, BLOCK_BS)
+    seq_lens = tl.load(seq_lens_ptr + load_offset, mask=load_offset <= pid, other=0)
+    pre_lens = tl.load(pre_lens_ptr + load_offset, mask=load_offset <= pid, other=0)
     extend_lens = seq_lens - pre_lens
+    num_new_pages = (seq_lens + page_size - 1) // page_size - (
+        pre_lens + page_size - 1
+    ) // page_size
+    for block_start in range(BLOCK_BS, pid + 1, BLOCK_BS):
+        load_offset = block_start + tl.arange(0, BLOCK_BS)
+        seq_lens = tl.load(seq_lens_ptr + load_offset, mask=load_offset <= pid, other=0)
+        pre_lens = tl.load(pre_lens_ptr + load_offset, mask=load_offset <= pid, other=0)
+        extend_lens += seq_lens - pre_lens
+        num_new_pages += (seq_lens + page_size - 1) // page_size - (
+            pre_lens + page_size - 1
+        ) // page_size
 
     seq_len = tl.load(seq_lens_ptr + pid)
     pre_len = tl.load(pre_lens_ptr + pid)
@@ -136,10 +156,6 @@ def alloc_extend_kernel(
     sum_extend_lens = tl.sum(extend_lens)
     output_start_loc = sum_extend_lens - extend_len
 
... (truncated, 429 more lines)
```
</details>

<details><summary>📄 diff: [Perf] dspark: Build the compact verify window and stage its</summary>

```diff
diff --git a/python/sglang/kernels/ops/speculative/dspark/dspark_verify_window.py b/python/sglang/kernels/ops/speculative/dspark/dspark_verify_window.py
index 194744bbb333..9be778e63abf 100644
--- a/python/sglang/kernels/ops/speculative/dspark/dspark_verify_window.py
+++ b/python/sglang/kernels/ops/speculative/dspark/dspark_verify_window.py
@@ -139,33 +139,85 @@ def build_ragged_verify_window(
     )
 
 
-@triton.jit
-def _ragged_finalize_kernel(
-    req_ptr,
-    within_ptr,
-    prefix_ptr,
-    cache_ptr,
-    pos_out_ptr,
-    cache_out_ptr,
+@triton.jit(
+    do_not_specialize=[
+        "req_to_token_stride",
+        "draft_block_ids_stride",
+        "draft_tokens_stride",
+        "num_rows",
+        "bs",
+        "n",
+        "width",
+    ]
+)
+def _ragged_verify_window_kernel(
+    verify_lens_ptr,
+    prefix_lens_ptr,
+    req_pool_indices_ptr,
+    req_to_token_ptr,
+    req_to_token_stride,
+    draft_block_ids_ptr,
+    draft_block_ids_stride,
+    draft_tokens_ptr,
+    draft_tokens_stride,
+    positions_ptr,
+    cache_loc_ptr,
+    verify_ids_ptr,
+    window_index_ptr,
+    num_rows,
     bs,
     n,
-    real_len,
-    BLOCK: tl.constexpr,
+    width,
+    BLOCK_T: tl.constexpr,
+    BLOCK_BS: tl.constexpr,
 ):
-    pid = tl.program_id(0)
-    offs = pid * BLOCK + tl.arange(0, BLOCK)
+    offs = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
     mask = offs < n
-    req = tl.load(req_ptr + offs, mask=mask, other=0)
-    within = tl.load(within_ptr + offs, mask=mask, other=0)
-    valid = req < bs
+    req = tl.zeros([BLOCK_T], dtype=tl.int32)
+    start = tl.zeros([BLOCK_T], dtype=tl.int32)
+    total = 0
+    for row0 in range(0, num_rows, BLOCK_BS):
+        rows = row0 + tl.arange(0, BLOCK_BS)
+        lens = tl.load(verify_lens_ptr + rows, mask=rows < num_rows, other=0)
+        lens = lens.to(tl.int32)
+        incl = total + tl.cumsum(lens, axis=0)
+        before = incl[None, :] <= offs[:, None]
+        req += tl.sum(before.to(tl.int32), axis=1)
+        start += tl.sum(tl.where(before, lens[None, :], 0), axis=1)
+        total += tl.sum(lens, axis=0)
+    valid = mask & (offs < total)
     safe_req = tl.minimum(req, bs - 1)
-    prefix = tl.load(prefix_ptr + safe_req, mask=mask, other=0)
-    pos = tl.where(valid, prefix + within, 0)
-    lmask = mask & (offs < real_len)
-    cl = tl.load(cache_ptr + offs, mask=lmask, other=0)
-    cl = tl.where(valid, cl, 0)
-    tl.store(pos_out_ptr + offs, pos, mask=mask)
-    tl.store(cache_out_ptr + offs, cl, mask=mask)
+    within = tl.where(valid, offs - start, 0)
+    prefix = tl.load(prefix_lens_ptr + safe_req, mask=mask, other=0).to(tl.int64)
... (truncated, 836 more lines)
```
</details>

<details><summary>📄 diff: [Fix] moe: Match the Triton renormalize sum order in the war</summary>

```diff
diff --git a/python/sglang/kernels/jit/csrc/moe/route_warp.cuh b/python/sglang/kernels/jit/csrc/moe/route_warp.cuh
index 9a5a8e7251ec..46b13fbb9b25 100644
--- a/python/sglang/kernels/jit/csrc/moe/route_warp.cuh
+++ b/python/sglang/kernels/jit/csrc/moe/route_warp.cuh
@@ -374,7 +374,7 @@ __global__ __launch_bounds__(kNumWarps * 32) void route_warp_kernel(const __grid
     }
   }
   if (!kMuseSpark && p.renormalize) {
-    const auto sum = warp::reduce_sum<std::bit_ceil(K)>(w);
+    const auto sum = warp::reduce_sum<1, std::bit_ceil(K)>(w);
     const auto norm = p.renorm_eps > 0.0f ? sum + p.renorm_eps : (sum > 0.0f ? sum : 1.0f);
     w = ptx::div_full(w, norm);
   }

```
</details>

<details><summary>📄 diff: [Feat] scheduler: Run the state checks in the PD and PP loop</summary>

```diff
diff --git a/python/sglang/srt/disaggregation/decode.py b/python/sglang/srt/disaggregation/decode.py
index 57425e2018f5..f4fb5ba5165c 100644
--- a/python/sglang/srt/disaggregation/decode.py
+++ b/python/sglang/srt/disaggregation/decode.py
@@ -100,6 +100,7 @@
     ListKind,
     ReqList,
 )
+from sglang.srt.managers.scheduler_components.state_check import CheckPhase
 from sglang.srt.managers.utils import (
     GenerationBatchResult,
     is_war_barrier_enabled,
@@ -3685,6 +3686,7 @@ def event_loop_normal_disagg_decode(self: Scheduler):
 
             # Update last_batch
             self.last_batch = batch
+            self.invariant_checker.run(CheckPhase.BUSY, batch)
 
     @torch.no_grad()
     def event_loop_overlap_disagg_decode(self: Scheduler):
@@ -3762,6 +3764,7 @@ def pop_and_process():
 
             # Update last_batch
             self.last_batch = batch
+            self.invariant_checker.run(CheckPhase.BUSY, batch)
 
     def _flush_staging_scatters(self: Scheduler) -> None:
         handler = self.disagg_decode_transfer_queue.staging_handler
diff --git a/python/sglang/srt/disaggregation/prefill.py b/python/sglang/srt/disaggregation/prefill.py
index 5e15e37b1652..e18d60f8eef0 100644
--- a/python/sglang/srt/disaggregation/prefill.py
+++ b/python/sglang/srt/disaggregation/prefill.py
@@ -78,6 +78,7 @@
     ReqList,
     move,
 )
+from sglang.srt.managers.scheduler_components.state_check import CheckPhase
 from sglang.srt.mem_cache.base_prefix_cache import CacheRequestOutcome
 from sglang.srt.mem_cache.common import (
     checkpoint_kv_cache,
@@ -772,6 +773,7 @@ def event_loop_normal_disagg_prefill(self: Scheduler) -> None:
 
             # Update last_batch
             self.last_batch = batch
+            self.invariant_checker.run(CheckPhase.BUSY, batch)
 
     def _supports_result_wait_polling(self: Scheduler) -> bool:
         """Whether the overlap loop can wait for a result without blocking.
@@ -996,6 +998,7 @@ def event_loop_overlap_disagg_prefill(self: Scheduler) -> None:
             self._sched_idled = batch is None
             self.last_batch = batch
             waiting_for_batch_result = False
+            self.invariant_checker.run(CheckPhase.BUSY, batch)
 
     def is_disagg_prefill_batch_result_ready(
         self: Scheduler,
diff --git a/python/sglang/srt/hardware_backend/mlx/scheduler_mixin.py b/python/sglang/srt/hardware_backend/mlx/scheduler_mixin.py
index 186e10967df7..b7fa758edcfd 100644
--- a/python/sglang/srt/hardware_backend/mlx/scheduler_mixin.py
+++ b/python/sglang/srt/hardware_backend/mlx/scheduler_mixin.py
@@ -22,7 +22,7 @@
 import mlx.core as mx
 
 from sglang.srt.managers.overlap_utils import resolve_forward_inputs
-from sglang.srt.managers.scheduler_components.invariant_checker import CheckPhase
+from sglang.srt.managers.scheduler_components.state_check import CheckPhase
 from sglang.srt.runtime_context import get_device
 from sglang.srt.utils import DynamicGradMode
 
@@ -255,7 +255,7 @@ def _launch_chained(prev: MlxPendingJob) -> MlxPendingJob:
             ):
                 self.cur_batch_for_debug = pending_next.schedule_batch
                 self.last_batch = pending_next.schedule_batch
-                self.invariant_checker.run(CheckPhase.BUSY)
+                self.invariant_checker.run(CheckPhase.BUSY, pending_next.schedule_batch)
                 continue
 
             # 4. Chain is broken. Finalise pending_next (if any), then
@@ -276,4 +276,4 @@ def _launch_chained(prev: MlxPendingJob) -> MlxPendingJob:
                 self.on_idle()
... (truncated, 861 more lines)
```
</details>

<details><summary>📄 diff: [Perf] diffusion: Tune LingBot MoE TMA for B300 SXM6 AC (#48</summary>

```diff
diff --git a/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_8_0/E=128,N=768,device_name=NVIDIA_B300_SXM6_AC.json b/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_8_0/E=128,N=768,device_name=NVIDIA_B300_SXM6_AC.json
new file mode 100644
index 000000000000..c900b640e211
--- /dev/null
+++ b/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_8_0/E=128,N=768,device_name=NVIDIA_B300_SXM6_AC.json
@@ -0,0 +1,33 @@
+{
+    "128": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 32,
+        "BLOCK_SIZE_K": 64,
+        "GROUP_SIZE_M": 1
+    },
+    "129": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 32,
+        "GROUP_SIZE_M": 8
+    },
+    "16384": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 32,
+        "GROUP_SIZE_M": 8
+    },
+    "32768": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 32,
+        "GROUP_SIZE_M": 8,
+        "USE_TMA": true
+    },
+    "65536": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 32,
+        "GROUP_SIZE_M": 8
+    }
+}
diff --git a/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_8_0/E=128,N=768,device_name=NVIDIA_B300_SXM6_AC_down.json b/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_8_0/E=128,N=768,device_name=NVIDIA_B300_SXM6_AC_down.json
new file mode 100644
index 000000000000..c900b640e211
--- /dev/null
+++ b/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_8_0/E=128,N=768,device_name=NVIDIA_B300_SXM6_AC_down.json
@@ -0,0 +1,33 @@
+{
+    "128": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 32,
+        "BLOCK_SIZE_K": 64,
+        "GROUP_SIZE_M": 1
+    },
+    "129": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 32,
+        "GROUP_SIZE_M": 8
+    },
+    "16384": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 32,
+        "GROUP_SIZE_M": 8
+    },
+    "32768": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 32,
+        "GROUP_SIZE_M": 8,
+        "USE_TMA": true
+    },
+    "65536": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 32,
+        "GROUP_SIZE_M": 8
+    }
+}
diff --git a/python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py b/python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py
index 25bb96c58a06..1d3a4582f7b7 100644
... (truncated, 113 more lines)
```
</details>

---
> 🤖 Generated by [daily-llm-tracker](https://github.com/ball-out-of-tune/daily-llm-tracker)