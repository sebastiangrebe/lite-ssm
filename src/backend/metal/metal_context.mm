// src/backend/metal/metal_context.mm
// MTLDevice + MTLCommandQueue singletons (lazy) and metallib path resolver.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "lite_ssm/detail/metal_backend.hpp"

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <mach-o/dyld.h>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef LITE_SSM_METALLIB_PATH
#define LITE_SSM_METALLIB_PATH ""
#endif

namespace lite_ssm::detail {

namespace {

std::atomic<void*> g_device{nullptr};
std::atomic<void*> g_queue{nullptr};

}  // namespace

void* metal_default_device() {
    void* cached = g_device.load(std::memory_order_acquire);
    if (cached) return cached;

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) {
        throw std::runtime_error(
            "Metal: MTLCreateSystemDefaultDevice() returned nil — no Metal-capable GPU?");
    }
    void* raw = (__bridge_retained void*)dev;

    void* expected = nullptr;
    if (!g_device.compare_exchange_strong(expected, raw,
                                          std::memory_order_acq_rel,
                                          std::memory_order_acquire)) {
        CFBridgingRelease(raw);
        return expected;
    }
    return raw;
}

void* metal_default_queue() {
    void* cached = g_queue.load(std::memory_order_acquire);
    if (cached) return cached;

    id<MTLDevice>       dev = (__bridge id<MTLDevice>)metal_default_device();
    id<MTLCommandQueue> q   = [dev newCommandQueue];
    if (!q) {
        throw std::runtime_error("Metal: newCommandQueue returned nil");
    }
    void* raw = (__bridge_retained void*)q;

    void* expected = nullptr;
    if (!g_queue.compare_exchange_strong(expected, raw,
                                         std::memory_order_acq_rel,
                                         std::memory_order_acquire)) {
        CFBridgingRelease(raw);
        return expected;
    }
    return raw;
}

namespace {

std::string current_executable_dir() {
    char buf[1024];
    uint32_t sz = sizeof(buf);
    if (_NSGetExecutablePath(buf, &sz) != 0) return {};
    std::error_code ec;
    auto canon = std::filesystem::weakly_canonical(std::filesystem::path(buf), ec);
    if (ec) return std::filesystem::path(buf).parent_path().string();
    return canon.parent_path().string();
}

bool file_exists(const std::string& p) {
    if (p.empty()) return false;
    std::error_code ec;
    return std::filesystem::exists(p, ec);
}

}  // namespace

std::string metallib_resolve_path() {
    if (const char* env = std::getenv("LITE_SSM_METALLIB"); env && file_exists(env)) {
        return env;
    }
    const std::string compiled = LITE_SSM_METALLIB_PATH;
    if (file_exists(compiled)) {
        return compiled;
    }
    std::string sib = current_executable_dir() + "/lite_ssm.metallib";
    if (file_exists(sib)) {
        return sib;
    }
    return {};
}

}  // namespace lite_ssm::detail
