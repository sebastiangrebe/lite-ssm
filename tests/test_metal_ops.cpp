// tests/test_metal_ops.cpp
// Phase 4 smoke test: confirm every kernel listed in our op surface is
// present in the compiled metallib and can be turned into a
// MTLComputePipelineState without errors. Does NOT validate numerical
// output (that comes in Phase 5 with parity tests against PyTorch).
//
// Exits 77 (CTest SKIP) if the metallib was not built — typically because
// the Metal Toolchain isn't installed yet (Xcode 26 makes it an opt-in
// download).

#include "lite_ssm/ops.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <string_view>

int main() {
    constexpr int SKIP = 77;

    lite_ssm::MetalOps ops;
    if (!ops.metallib_loaded()) {
        std::fprintf(stderr,
            "[test_metal_ops] SKIP: lite_ssm.metallib not found.\n"
            "  Install Metal Toolchain:  xcodebuild -downloadComponent MetalToolchain\n"
            "  Then re-run:              cmake --build build\n");
        return SKIP;
    }
    std::printf("[test_metal_ops] metallib: %s\n", ops.metallib_path().c_str());

    constexpr std::array<std::string_view, 16> kernels = {
        "lite_ssm_common_noop",
        "rmsnorm_f16",
        "silu_f16",
        "silu_gated_f16",
        "linear_f16_gemv",
        "linear_silu_f16_gemv",
        "rmsnorm_linear_f16_gemv",
        "linear_f16_gemm",
        "linear_silu_f16_gemm",
        "causal_conv1d_f16",
        "causal_conv1d_update_f16",
        "ssd_chunked_f16",
        "ssd_step_f16",
        "linear_int4_gemv",      // Phase 14
        "linear_int4_gemm",
        "seed_conv_state_f16",   // Phase 16
    };

    int failures = 0;
    for (auto name : kernels) {
        bool ok = false;
        try {
            ok = ops.has_pipeline(name);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[test_metal_ops]  %.*s -> THROW: %s\n",
                         (int)name.size(), name.data(), e.what());
            ++failures;
            continue;
        }
        if (ok) {
            std::printf("[test_metal_ops]  %.*s  ok\n", (int)name.size(), name.data());
        } else {
            std::fprintf(stderr, "[test_metal_ops]  %.*s  MISSING in metallib\n",
                         (int)name.size(), name.data());
            ++failures;
        }
    }

    if (failures) {
        std::fprintf(stderr, "[test_metal_ops] FAIL: %d kernel(s) missing or unbuildable\n",
                     failures);
        return 1;
    }
    std::printf("[test_metal_ops] OK: all %zu kernels present\n", kernels.size());
    return 0;
}
