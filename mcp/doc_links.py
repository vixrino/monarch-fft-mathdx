# Map a failure to the NVIDIA documentation section an agent should read next. Every URL was checked to
# resolve (HTTP 200) on 2026-11-17; the cuBLASDx markdown pages are the agent-targeted guide NVIDIA ships
# in doc/cublasdx/html/markdown/ and also serves online. cuFFTDx has no agent guide, so its entries point at
# the regular HTML reference plus this repo's unofficial guide.
import re

CB = "https://docs.nvidia.com/cuda/cublasdx/"
CF = "https://docs.nvidia.com/cuda/cufftdx/"
UNOFFICIAL = "dxbench://agent_guides/cufftdx"

# (pattern on the compiler message or the failure kind, title, url)
COMPILE = [
    (r"cublasdx.*(Size|Precision|Arrangement|Type|Function|BlockDim|SM|Alignment)|is_complete_blas",
     "cuBLASDx operators", CB + "api/operators.html"),
    (r"suggest_layout|get_layout|make_tensor|slice_shared_memory|get_shared_storage_size",
     "cuBLASDx shared-memory tensors and layouts", CB + "api/other_tensors.html"),
    (r"suggest_accumulator|get_accumulator|partition_and_store|map_fragment_index|get_results",
     "cuBLASDx register fragments (supermma card)", CB + "markdown/performance_guide_api_supermma.md#perf-guide-api-card-supermma-label"),
    (r"cublasdx.*execute|execute.*cublasdx|blas_block_gemm_execution",
     "cuBLASDx GEMM execution methods", CB + "api/gemm_methods.html"),
    (r"cufftdx.*(Size|Type|Direction|Precision|ElementsPerThread|FFTsPerBlock|SM|Block)",
     "cuFFTDx operators", CF + "api/operators.html"),
    (r"elements_per_thread|storage_size|shared_memory_size|block_dim|ffts_per_block|implicit_type_batching|"
     r"requires_workspace|input_length|output_length|stride",
     "cuFFTDx traits", CF + "api/traits.html"),
    (r"cufftdx.*execute|working_group|block_fft_execution",
     "cuFFTDx execution methods", CF + "api/methods.html"),
    (r"CMake Error|mathdx|find_package",
     "cuBLASDx build card (release flags, targets)", CB + "markdown/blas_performance_guide.md"),
]

KIND = {
    "accuracy_monarch": [
        ("Correctness harness for a ported GEMM kernel", CB + "markdown/performance_guide_porting.md#perf-guide-port-harness-label"),
        ("Monarch index maps and the three easy mistakes", "docs/derivation.md"),
    ],
    "accuracy_cufftdx": [
        ("cuFFTDx: data layout per thread and execute()", CF + "api/methods.html"),
        ("Using cuFFTDx (load / execute / store pattern)", CF + "introduction1.html"),
        ("Unofficial cuFFTDx agent guide: I/O and fp16 batching", UNOFFICIAL),
    ],
    "no_tensor_cores": [
        ("supermma card: which precisions reach tensor cores, suggested layouts", CB + "markdown/performance_guide_api_supermma.md#perf-guide-api-card-supermma-label"),
        ("Porting guide step 5: check geometry and generated instructions", CB + "markdown/performance_guide_porting.md#perf-guide-port-validation-label"),
        ("Shapes that do not divide the tile", CB + "markdown/performance_guide_problem_shape.md#perf-guide-partial-tiles-label"),
    ],
    "timeout": [
        ("Measurement protocol", CB + "markdown/blas_performance_guide.md#perf-guide-measurement-label"),
    ],
}


def for_errors(errors):
    out, seen = [], set()
    for e in errors:
        text = e["message"] + " " + " ".join(e.get("context", []))
        for pat, title, url in COMPILE:
            if url not in seen and re.search(pat, text):
                seen.add(url)
                out.append({"title": title, "url": url, "matched": e["message"][:120]})
    return out


def for_kind(kind):
    return [{"title": t, "url": u} for t, u in KIND[kind]]
