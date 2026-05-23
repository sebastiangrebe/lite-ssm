# FindMetal.cmake
# Locates Apple's Metal, Foundation, and QuartzCore frameworks and exposes
# them as imported targets: Metal::Metal, Metal::Foundation, Metal::QuartzCore.

if(NOT APPLE)
    set(Metal_FOUND FALSE)
    return()
endif()

find_library(METAL_FRAMEWORK Metal)
find_library(FOUNDATION_FRAMEWORK Foundation)
find_library(QUARTZCORE_FRAMEWORK QuartzCore)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Metal
    REQUIRED_VARS
        METAL_FRAMEWORK
        FOUNDATION_FRAMEWORK
        QUARTZCORE_FRAMEWORK
)

if(Metal_FOUND)
    if(NOT TARGET Metal::Metal)
        add_library(Metal::Metal INTERFACE IMPORTED)
        set_target_properties(Metal::Metal PROPERTIES
            INTERFACE_LINK_LIBRARIES "${METAL_FRAMEWORK}"
        )
    endif()
    if(NOT TARGET Metal::Foundation)
        add_library(Metal::Foundation INTERFACE IMPORTED)
        set_target_properties(Metal::Foundation PROPERTIES
            INTERFACE_LINK_LIBRARIES "${FOUNDATION_FRAMEWORK}"
        )
    endif()
    if(NOT TARGET Metal::QuartzCore)
        add_library(Metal::QuartzCore INTERFACE IMPORTED)
        set_target_properties(Metal::QuartzCore PROPERTIES
            INTERFACE_LINK_LIBRARIES "${QUARTZCORE_FRAMEWORK}"
        )
    endif()
endif()

mark_as_advanced(METAL_FRAMEWORK FOUNDATION_FRAMEWORK QUARTZCORE_FRAMEWORK)
