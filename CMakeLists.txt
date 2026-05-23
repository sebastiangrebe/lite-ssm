cmake_minimum_required(VERSION 3.24)
project(lite_ssm
    VERSION 0.1.0
    DESCRIPTION "Ultra-lightweight standalone C++ inference engine for State Space Models"
    LANGUAGES C CXX OBJC OBJCXX
)

# -----------------------------------------------------------------------------
# Build settings
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT APPLE)
    message(FATAL_ERROR "lite-ssm targets Apple Silicon. Build on macOS only.")
endif()

# Enable ARC for ObjC / ObjC++ sources
add_compile_options(
    $<$<COMPILE_LANGUAGE:OBJC>:-fobjc-arc>
    $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# -----------------------------------------------------------------------------
# Apple frameworks (Metal, Foundation, QuartzCore)
# -----------------------------------------------------------------------------
find_package(Metal REQUIRED)

# -----------------------------------------------------------------------------
# Google SentencePiece — fetched + built from source as a static lib.
#
# We disable every optional SP subproject so the build stays self-contained:
#   * no shared-library variant (we only link the static archive)
#   * no test binaries
#   * no command-line tools
#   * use the bundled protobuf so we don't pull a system dep
#   * no tcmalloc dependency
# -----------------------------------------------------------------------------
include(FetchContent)
set(SPM_ENABLE_SHARED      OFF CACHE BOOL "" FORCE)
set(SPM_BUILD_TEST         OFF CACHE BOOL "" FORCE)
set(SPM_ENABLE_TCMALLOC    OFF CACHE BOOL "" FORCE)
set(SPM_USE_BUILTIN_PROTOBUF ON CACHE BOOL "" FORCE)
set(SPM_BUILD_LIBRARY_ONLY  ON CACHE BOOL "" FORCE)
# SP v0.2.0 declares `cmake_minimum_required(VERSION 3.1)` — CMake 4.x dropped
# support for floors that old. The policy override re-enables compatibility
# without forking SP.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)
FetchContent_Declare(
    sentencepiece
    GIT_REPOSITORY https://github.com/google/sentencepiece.git
    GIT_TAG        v0.2.0
)
FetchContent_MakeAvailable(sentencepiece)
# SP's CMake stamps the static lib as `sentencepiece-static`. Headers live
# under <build>/sentencepiece-src/src/. We expose them as a private include
# dir on lite_ssm — public callers don't need SP.

# -----------------------------------------------------------------------------
# Metal shader compilation: .metal -> .air -> lite_ssm.metallib
#
# Probe the Metal Toolchain at configure time. Xcode 26+ ships `metal` as a
# stub that errors with "missing Metal Toolchain" until the user runs:
#     xcodebuild -downloadComponent MetalToolchain
# If the toolchain is unavailable we skip the metallib build with a loud
# warning rather than failing — the C++ scaffold should still compile so
# developers can work on non-kernel code.
# -----------------------------------------------------------------------------
set(LITE_SSM_SHADER_DIR "${CMAKE_SOURCE_DIR}/shaders")
file(GLOB LITE_SSM_METAL_SOURCES "${LITE_SSM_SHADER_DIR}/*.metal")

set(LITE_SSM_METALLIB "${CMAKE_BINARY_DIR}/lite_ssm.metallib")
set(LITE_SSM_AIR_DIR "${CMAKE_BINARY_DIR}/air")
file(MAKE_DIRECTORY "${LITE_SSM_AIR_DIR}")

# Toolchain probe: try compiling a trivial shader through `metal`.
set(LITE_SSM_METAL_PROBE_SRC "${CMAKE_BINARY_DIR}/metal_probe.metal")
file(WRITE "${LITE_SSM_METAL_PROBE_SRC}"
    "#include <metal_stdlib>\nusing namespace metal;\nkernel void probe(uint tid [[thread_position_in_grid]]) { (void)tid; }\n"
)
execute_process(
    COMMAND xcrun -sdk macosx metal -c "${LITE_SSM_METAL_PROBE_SRC}" -o "${CMAKE_BINARY_DIR}/metal_probe.air"
    RESULT_VARIABLE LITE_SSM_METAL_PROBE_RC
    OUTPUT_VARIABLE LITE_SSM_METAL_PROBE_OUT
    ERROR_VARIABLE  LITE_SSM_METAL_PROBE_ERR
)

if(LITE_SSM_METAL_PROBE_RC EQUAL 0)
    set(LITE_SSM_HAVE_METAL_TOOLCHAIN TRUE)
else()
    set(LITE_SSM_HAVE_METAL_TOOLCHAIN FALSE)
    message(WARNING
        "Metal Toolchain not available — skipping metallib build.\n"
        "Install with:  xcodebuild -downloadComponent MetalToolchain\n"
        "Probe error: ${LITE_SSM_METAL_PROBE_ERR}"
    )
endif()

if(LITE_SSM_HAVE_METAL_TOOLCHAIN AND LITE_SSM_METAL_SOURCES)
    set(LITE_SSM_AIR_FILES "")
    foreach(metal_src ${LITE_SSM_METAL_SOURCES})
        get_filename_component(metal_name "${metal_src}" NAME_WE)
        set(air_file "${LITE_SSM_AIR_DIR}/${metal_name}.air")
        add_custom_command(
            OUTPUT "${air_file}"
            COMMAND xcrun -sdk macosx metal
                    -gline-tables-only
                    -frecord-sources
                    -ffast-math
                    -I "${LITE_SSM_SHADER_DIR}"
                    -c "${metal_src}"
                    -o "${air_file}"
            DEPENDS "${metal_src}"
            COMMENT "Compiling Metal shader ${metal_name}.metal"
            VERBATIM
        )
        list(APPEND LITE_SSM_AIR_FILES "${air_file}")
    endforeach()

    # Modern Xcode (15+) ships only the `metal` driver — it both compiles
    # .metal -> .air and links .air -> .metallib based on the -o extension.
    # The standalone `metallib` / `metal-link` tools are no longer present.
    add_custom_command(
        OUTPUT "${LITE_SSM_METALLIB}"
        COMMAND xcrun -sdk macosx metal ${LITE_SSM_AIR_FILES} -o "${LITE_SSM_METALLIB}"
        DEPENDS ${LITE_SSM_AIR_FILES}
        COMMENT "Linking lite_ssm.metallib"
        VERBATIM
    )
    add_custom_target(lite_ssm_metallib ALL DEPENDS "${LITE_SSM_METALLIB}")
else()
    add_custom_target(lite_ssm_metallib
        COMMENT "Skipping metallib build (Metal Toolchain unavailable or no .metal sources)"
    )
endif()

# -----------------------------------------------------------------------------
# Core static library
# -----------------------------------------------------------------------------
set(LITE_SSM_CORE_SOURCES
    src/core/tensor.cpp
    src/core/allocator.cpp
    src/core/ssm_loader.cpp
    src/core/tokenizer.cpp
    src/model/mamba2_block.cpp
    src/model/mamba2_model.cpp
    src/model/state.cpp
    src/runtime/inference.cpp
    src/runtime/sampler.cpp
)

set(LITE_SSM_METAL_BACKEND_SOURCES
    src/backend/metal/metal_context.mm
    src/backend/metal/metal_buffer.mm
    src/backend/metal/kernel_registry.mm
    src/backend/metal/ops.mm
)

add_library(lite_ssm STATIC
    ${LITE_SSM_CORE_SOURCES}
    ${LITE_SSM_METAL_BACKEND_SOURCES}
)

add_dependencies(lite_ssm lite_ssm_metallib)

target_include_directories(lite_ssm
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
)

target_link_libraries(lite_ssm
    PUBLIC
        Metal::Metal
        Metal::Foundation
        Metal::QuartzCore
    PRIVATE
        sentencepiece-static
)
target_include_directories(lite_ssm PRIVATE
    ${sentencepiece_SOURCE_DIR}/src
    ${sentencepiece_BINARY_DIR}
)

target_compile_options(lite_ssm PRIVATE
    -Wall -Wextra -Wpedantic
    $<$<CONFIG:Release>:-O3>
)

# Expose metallib path to C++ via compile definition
target_compile_definitions(lite_ssm PRIVATE
    LITE_SSM_METALLIB_PATH="${LITE_SSM_METALLIB}"
)

# -----------------------------------------------------------------------------
# CLI binary
# -----------------------------------------------------------------------------
add_executable(lite_ssm_cli apps/lite_ssm_cli.cpp)
set_target_properties(lite_ssm_cli PROPERTIES OUTPUT_NAME "lite-ssm")
target_link_libraries(lite_ssm_cli PRIVATE lite_ssm)

# -----------------------------------------------------------------------------
# Benchmarks
# -----------------------------------------------------------------------------
add_executable(benchmark_litessm examples/benchmark_litessm.cpp)
target_link_libraries(benchmark_litessm PRIVATE lite_ssm)

# -----------------------------------------------------------------------------
# Stress / endurance probes — built as standalone executables only. NOT
# registered with CTest because they are long-running by design (memory
# bleed runs until budget exhaustion; concurrency saturates the GPU).
# -----------------------------------------------------------------------------
set(LITE_SSM_STRESS
    test_memory_bleed
    test_concurrency
)
foreach(stress ${LITE_SSM_STRESS})
    add_executable(${stress} tests/stress/${stress}.cpp)
    target_link_libraries(${stress} PRIVATE lite_ssm)
endforeach()

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
enable_testing()

set(LITE_SSM_TESTS
    test_tensor
    test_mmap_loader
    test_metal_ops
    test_ssd_parity
    test_tokenizer
)

# Point the parity test at the dump generated by tools/dump_parity.py.

foreach(test_name ${LITE_SSM_TESTS})
    add_executable(${test_name} tests/${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE lite_ssm)
    add_test(NAME ${test_name} COMMAND ${test_name})
    # 77 == autoconf/CTest convention for "skipped"; tests use it when a
    # required artifact (e.g. model.ssm) isn't available.
    set_tests_properties(${test_name} PROPERTIES SKIP_RETURN_CODE 77)
endforeach()

# Point tests at the repo-root model.ssm (generated by tools/export_mamba2.py).
set_tests_properties(test_mmap_loader PROPERTIES
    ENVIRONMENT "LITE_SSM_TEST_MODEL=${CMAKE_SOURCE_DIR}/model.ssm"
)
set_tests_properties(test_ssd_parity PROPERTIES
    ENVIRONMENT "LITE_SSM_TEST_MODEL=${CMAKE_SOURCE_DIR}/model.ssm;LITE_SSM_PARITY_DUMP=${CMAKE_SOURCE_DIR}/parity.bin"
)
# Run the tokenizer test from the repo root so it finds tokenizer.bin /
# tokenizer_refs.bin produced by tools/export_tokenizer.py.
set_tests_properties(test_tokenizer PROPERTIES
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
message(STATUS "")
message(STATUS "lite-ssm configuration:")
message(STATUS "  Version:       ${PROJECT_VERSION}")
message(STATUS "  Build type:    ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard:  ${CMAKE_CXX_STANDARD}")
message(STATUS "  Metal toolchain: ${LITE_SSM_HAVE_METAL_TOOLCHAIN}")
message(STATUS "  Metal sources: ${LITE_SSM_METAL_SOURCES}")
message(STATUS "  Metallib out:  ${LITE_SSM_METALLIB}")
message(STATUS "")
