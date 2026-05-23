#pragma once
// lite_ssm/detail/metal_backend.hpp — INTERNAL cross-TU helpers for the
// Metal backend. Not part of the public API.
//
// The void* values returned here are id<...> objects bridge-cast back to
// raw pointers so this header can be included from pure C++ TUs.

#include <cstddef>
#include <string>

namespace lite_ssm::detail {

// Returns a bridge-cast id<MTLDevice> as a void*. Lazily creates the system
// default device on first call. Throws std::runtime_error if Metal is not
// available on this system.
void* metal_default_device();

// Returns a bridge-cast id<MTLCommandQueue> as a void*. Lazily created.
void* metal_default_queue();

// Resolves the location of `lite_ssm.metallib` on disk, in priority order:
//   1. $LITE_SSM_METALLIB env var
//   2. compile-time LITE_SSM_METALLIB_PATH (set by CMake to the build dir)
//   3. <executable_dir>/lite_ssm.metallib
// Returns an empty string if none of those exist. The caller decides how
// to react (error out, skip a smoke test, etc.).
std::string metallib_resolve_path();

// KernelRegistry shims (impl in kernel_registry.mm).
//
//   registry_pipeline(name)   bridge-cast id<MTLComputePipelineState>* or nullptr
//                             if the named function is not in the metallib.
//                             Builds + caches on first call. May throw if
//                             pipeline construction itself fails.
//   registry_library_loaded() true once the metallib has been opened.
//   registry_metallib_path()  the resolved path (or empty).
void*              registry_pipeline(const char* name);
bool               registry_library_loaded();
const std::string& registry_metallib_path();

}  // namespace lite_ssm::detail
