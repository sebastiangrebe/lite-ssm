// src/backend/metal/kernel_registry.mm
// Loads lite_ssm.metallib and lazily compiles MTLComputePipelineState
// objects for each named kernel. Used by ops.mm.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "lite_ssm/detail/metal_backend.hpp"

#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace lite_ssm::detail {

// Exposed as detail::KernelRegistry for ops.mm to consume. We keep it a
// raw class (not in a header) since only one .mm file actually uses it.
class KernelRegistry {
public:
    static KernelRegistry& instance() {
        static KernelRegistry r;
        return r;
    }

    bool library_loaded() const { return library_ != nullptr; }
    const std::string& metallib_path() const { return path_; }

    // Returns a bridge-cast id<MTLComputePipelineState> as void*. nullptr if
    // the named function does not exist in the library (or the library is
    // not loaded). Throws std::runtime_error on a real pipeline build error.
    void* pipeline(const std::string& name) {
        std::lock_guard<std::mutex> lock(mu_);
        if (!ensure_library_locked_()) return nullptr;

        auto it = cache_.find(name);
        if (it != cache_.end()) return it->second;

        id<MTLLibrary>            lib  = (__bridge id<MTLLibrary>)library_;
        id<MTLFunction>           func = [lib newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
        if (!func) {
            cache_[name] = nullptr;
            return nullptr;
        }
        id<MTLDevice>             dev  = (__bridge id<MTLDevice>)metal_default_device();
        NSError*                  err  = nil;
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:func error:&err];
        if (!pso) {
            std::string msg = "Metal: failed to build pipeline for '" + name + "': ";
            msg += err ? [[err localizedDescription] UTF8String] : "unknown error";
            throw std::runtime_error(msg);
        }
        void* raw = (__bridge_retained void*)pso;
        cache_[name] = raw;
        return raw;
    }

    ~KernelRegistry() {
        for (auto& [_, v] : cache_) if (v) CFBridgingRelease(v);
        if (library_) CFBridgingRelease(library_);
    }

private:
    KernelRegistry() = default;
    KernelRegistry(const KernelRegistry&) = delete;
    KernelRegistry& operator=(const KernelRegistry&) = delete;

    bool ensure_library_locked_() {
        if (library_ || tried_) return library_ != nullptr;
        tried_ = true;

        path_ = metallib_resolve_path();
        if (path_.empty()) return false;

        id<MTLDevice> dev = (__bridge id<MTLDevice>)metal_default_device();
        NSURL* url        = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path_.c_str()]];
        NSError* err      = nil;
        id<MTLLibrary> lib = [dev newLibraryWithURL:url error:&err];
        if (!lib) {
            std::string msg = "Metal: failed to load metallib '" + path_ + "': ";
            msg += err ? [[err localizedDescription] UTF8String] : "unknown error";
            throw std::runtime_error(msg);
        }
        library_ = (__bridge_retained void*)lib;
        return true;
    }

    std::mutex                                mu_;
    void*                                     library_ = nullptr;
    bool                                      tried_   = false;
    std::string                               path_;
    std::unordered_map<std::string, void*>    cache_;
};

}  // namespace lite_ssm::detail

// Tiny C-ABI surface so ops.mm doesn't need to know KernelRegistry's type.
namespace lite_ssm::detail {

void* registry_pipeline(const char* name) {
    return KernelRegistry::instance().pipeline(name);
}

bool registry_library_loaded() {
    return KernelRegistry::instance().library_loaded();
}

const std::string& registry_metallib_path() {
    return KernelRegistry::instance().metallib_path();
}

}  // namespace lite_ssm::detail
