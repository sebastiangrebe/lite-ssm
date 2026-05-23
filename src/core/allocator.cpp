// src/core/allocator.cpp — POSIX mmap RAII implementation.

#include "lite_ssm/allocator.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace lite_ssm {

namespace {

std::size_t page_size_bytes() {
    long ps = ::sysconf(_SC_PAGESIZE);
    return (ps > 0) ? static_cast<std::size_t>(ps) : 16384;  // 16K on Apple Silicon
}

std::size_t align_up_sz(std::size_t v, std::size_t a) {
    return (v + a - 1) & ~(a - 1);
}

}  // namespace

MmapFile::MmapFile(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error("MmapFile: open('" + path + "') failed: " + std::strerror(errno));
    }

    struct stat st{};
    if (::fstat(fd, &st) != 0) {
        int err = errno;
        ::close(fd);
        throw std::runtime_error("MmapFile: fstat failed: " + std::string(std::strerror(err)));
    }
    if (st.st_size <= 0) {
        ::close(fd);
        throw std::runtime_error("MmapFile: empty or invalid file '" + path + "'");
    }

    const std::size_t file_size   = static_cast<std::size_t>(st.st_size);
    const std::size_t page_size   = page_size_bytes();
    const std::size_t mapped_size = align_up_sz(file_size, page_size);

    void* ptr = ::mmap(nullptr, mapped_size, PROT_READ, MAP_PRIVATE, fd, 0);
    // Closing the fd is safe immediately after mmap on POSIX — mapping
    // holds its own reference.
    ::close(fd);

    if (ptr == MAP_FAILED) {
        throw std::runtime_error("MmapFile: mmap failed: " + std::string(std::strerror(errno)));
    }

    file_size_   = file_size;
    mapped_size_ = mapped_size;
    ptr_         = ptr;
}

MmapFile::~MmapFile() {
    reset_();
}

MmapFile::MmapFile(MmapFile&& other) noexcept
    : ptr_(other.ptr_),
      file_size_(other.file_size_),
      mapped_size_(other.mapped_size_) {
    other.ptr_         = nullptr;
    other.file_size_   = 0;
    other.mapped_size_ = 0;
}

MmapFile& MmapFile::operator=(MmapFile&& other) noexcept {
    if (this != &other) {
        reset_();
        ptr_              = other.ptr_;
        file_size_        = other.file_size_;
        mapped_size_      = other.mapped_size_;
        other.ptr_        = nullptr;
        other.file_size_  = 0;
        other.mapped_size_= 0;
    }
    return *this;
}

void MmapFile::reset_() noexcept {
    if (ptr_) {
        ::munmap(ptr_, mapped_size_);
        ptr_         = nullptr;
        file_size_   = 0;
        mapped_size_ = 0;
    }
}

}  // namespace lite_ssm
