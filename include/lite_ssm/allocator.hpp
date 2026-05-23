#pragma once
// lite_ssm/allocator.hpp — zero-copy memory plumbing.
//
//   MmapFile      RAII wrapper around mmap(MAP_PRIVATE | PROT_READ).
//                 Returns the page-aligned base pointer of the file.
//
//   UnifiedBuffer RAII wrapper around an MTLBuffer constructed via
//                 newBufferWithBytesNoCopy:length:options:deallocator:.
//                 Views the MmapFile region; no bytes copied.
//                 MUST outlive any Metal command encoder that binds it,
//                 and MUST be destroyed BEFORE the MmapFile (so the
//                 MTLBuffer release happens before munmap). When both
//                 are members of the same owner, declare MmapFile first
//                 and UnifiedBuffer second — C++ destroys in reverse order.

#include <cstddef>
#include <string>

namespace lite_ssm {

class MmapFile {
public:
    MmapFile() = default;
    explicit MmapFile(const std::string& path);
    ~MmapFile();

    MmapFile(const MmapFile&)            = delete;
    MmapFile& operator=(const MmapFile&) = delete;

    MmapFile(MmapFile&& other) noexcept;
    MmapFile& operator=(MmapFile&& other) noexcept;

    const void*  data()        const { return ptr_; }
    std::size_t  file_size()   const { return file_size_; }   // exact file size
    std::size_t  mapped_size() const { return mapped_size_; } // page-rounded
    bool         valid()       const { return ptr_ != nullptr; }

private:
    void        reset_() noexcept;

    void*       ptr_         = nullptr;
    std::size_t file_size_   = 0;
    std::size_t mapped_size_ = 0;
};

// DeviceBuffer owns a newly-allocated MTLBuffer with StorageModeShared. Used
// for activations and recurrent state — anything that isn't a view into the
// mmap'd weights file. CPU and GPU see the same bytes (unified memory).
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t nbytes);
    ~DeviceBuffer();

    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&&) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&&) noexcept;

    void*       metal_buffer() const { return buffer_; }     // id<MTLBuffer>
    void*       contents()     const { return contents_; }   // CPU pointer
    std::size_t length()       const { return length_; }
    bool        valid()        const { return buffer_ != nullptr; }

    void zero();   // memset(contents, 0, length) — for state reset

private:
    void reset_() noexcept;

    void*       buffer_   = nullptr;
    void*       contents_ = nullptr;
    std::size_t length_   = 0;
};

class UnifiedBuffer {
public:
    UnifiedBuffer() = default;
    // Constructs an MTLBuffer viewing the entire MmapFile region with
    // MTLResourceStorageModeShared. The `mapping` must remain valid for
    // the lifetime of this UnifiedBuffer.
    explicit UnifiedBuffer(const MmapFile& mapping);
    ~UnifiedBuffer();

    UnifiedBuffer(const UnifiedBuffer&)            = delete;
    UnifiedBuffer& operator=(const UnifiedBuffer&) = delete;

    UnifiedBuffer(UnifiedBuffer&& other) noexcept;
    UnifiedBuffer& operator=(UnifiedBuffer&& other) noexcept;

    // Opaque id<MTLBuffer>. Cast back in .mm files via
    //   (__bridge id<MTLBuffer>)ub.metal_buffer()
    void*        metal_buffer() const { return buffer_; }
    const void*  contents()     const { return contents_; } // == MmapFile::data()
    std::size_t  length()       const { return length_; }   // page-aligned
    bool         valid()        const { return buffer_ != nullptr; }

private:
    void        reset_() noexcept;

    void*       buffer_   = nullptr;  // bridge-retained id<MTLBuffer>
    const void* contents_ = nullptr;
    std::size_t length_   = 0;
};

}  // namespace lite_ssm
