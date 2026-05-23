// src/backend/metal/metal_buffer.mm — bridge mmap region to MTLBuffer with
// MTLResourceStorageModeShared via newBufferWithBytesNoCopy.
//
// Apple constraints we must respect:
//   * The pointer must be page-aligned. mmap guarantees this; we assert.
//   * The length must be a multiple of the page size. MmapFile rounds up
//     the mapping size for exactly this reason.
//   * The bytes must remain valid for the MTLBuffer's lifetime. We hand a
//     nil deallocator and rely on the owning SSMFile to destroy the
//     UnifiedBuffer BEFORE its MmapFile (members declared in that order).

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "lite_ssm/allocator.hpp"
#include "lite_ssm/detail/metal_backend.hpp"

#include <cassert>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>

namespace lite_ssm {

namespace {

std::size_t page_size_bytes() {
    long ps = ::sysconf(_SC_PAGESIZE);
    return (ps > 0) ? static_cast<std::size_t>(ps) : 16384;
}

}  // namespace

UnifiedBuffer::UnifiedBuffer(const MmapFile& mapping) {
    if (!mapping.valid()) {
        throw std::runtime_error("UnifiedBuffer: source MmapFile is empty");
    }
    const std::size_t page = page_size_bytes();
    const void*       ptr  = mapping.data();
    const std::size_t len  = mapping.mapped_size();

    // mmap returns page-aligned pointers; double-check we didn't break that
    // invariant somewhere upstream.
    assert((reinterpret_cast<std::uintptr_t>(ptr) % page) == 0 &&
           "UnifiedBuffer: mmap pointer must be page-aligned");
    assert((len % page) == 0 &&
           "UnifiedBuffer: MmapFile::mapped_size() must be page-aligned");
    (void)page;

    id<MTLDevice> dev = (__bridge id<MTLDevice>)detail::metal_default_device();
    id<MTLBuffer> buf = [dev newBufferWithBytesNoCopy:const_cast<void*>(ptr)
                                              length:len
                                             options:MTLResourceStorageModeShared
                                         deallocator:nil];
    if (!buf) {
        throw std::runtime_error(
            "UnifiedBuffer: newBufferWithBytesNoCopy returned nil "
            "(length=" + std::to_string(len) + ")");
    }

    buffer_   = (__bridge_retained void*)buf;
    contents_ = ptr;
    length_   = len;
}

UnifiedBuffer::~UnifiedBuffer() {
    reset_();
}

UnifiedBuffer::UnifiedBuffer(UnifiedBuffer&& other) noexcept
    : buffer_(other.buffer_),
      contents_(other.contents_),
      length_(other.length_) {
    other.buffer_   = nullptr;
    other.contents_ = nullptr;
    other.length_   = 0;
}

UnifiedBuffer& UnifiedBuffer::operator=(UnifiedBuffer&& other) noexcept {
    if (this != &other) {
        reset_();
        buffer_         = other.buffer_;
        contents_       = other.contents_;
        length_         = other.length_;
        other.buffer_   = nullptr;
        other.contents_ = nullptr;
        other.length_   = 0;
    }
    return *this;
}

void UnifiedBuffer::reset_() noexcept {
    if (buffer_) {
        CFBridgingRelease(buffer_);
        buffer_   = nullptr;
        contents_ = nullptr;
        length_   = 0;
    }
}

// ---------------------------------------------------------------------------
//  DeviceBuffer — owns a fresh MTLBuffer (StorageModeShared)
// ---------------------------------------------------------------------------
DeviceBuffer::DeviceBuffer(std::size_t nbytes) {
    if (nbytes == 0) return;
    id<MTLDevice> dev = (__bridge id<MTLDevice>)detail::metal_default_device();
    id<MTLBuffer> buf = [dev newBufferWithLength:nbytes options:MTLResourceStorageModeShared];
    if (!buf) {
        throw std::runtime_error("DeviceBuffer: newBufferWithLength returned nil");
    }
    buffer_   = (__bridge_retained void*)buf;
    contents_ = buf.contents;
    length_   = nbytes;
}

DeviceBuffer::~DeviceBuffer() { reset_(); }

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : buffer_(other.buffer_), contents_(other.contents_), length_(other.length_) {
    other.buffer_   = nullptr;
    other.contents_ = nullptr;
    other.length_   = 0;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
        reset_();
        buffer_         = other.buffer_;
        contents_       = other.contents_;
        length_         = other.length_;
        other.buffer_   = nullptr;
        other.contents_ = nullptr;
        other.length_   = 0;
    }
    return *this;
}

void DeviceBuffer::zero() {
    if (contents_ && length_) std::memset(contents_, 0, length_);
}

void DeviceBuffer::reset_() noexcept {
    if (buffer_) {
        CFBridgingRelease(buffer_);
        buffer_   = nullptr;
        contents_ = nullptr;
        length_   = 0;
    }
}

}  // namespace lite_ssm
