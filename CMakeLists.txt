cmake_minimum_required(VERSION 3.23)

project(rnnoise
    VERSION 0.2.0
    DESCRIPTION "RNN-based noise suppression"
    LANGUAGES C
)

include(CheckCCompilerFlag)
include(CheckLibraryExists)
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# ------------------------------------------------------------------------------
# Options
# ------------------------------------------------------------------------------

option(RNNOISE_BUILD_EXAMPLES
    "Build example applications"
    ON
)

option(RNNOISE_BUILD_DOCS
    "Build API documentation when Doxygen is available"
    ON
)

option(RNNOISE_ENABLE_DNN_DEBUG_FLOAT
    "Use floating-point DNN computation everywhere"
    OFF
)

option(RNNOISE_DOWNLOAD_MODEL
    "Download the default RNNoise model"
    OFF
)

if(WIN32)
    add_compile_definitions(
        _WIN32_WINNT=0x0601
        WIN32_LEAN_AND_MEAN
        UNICODE
        _UNICODE
    )
endif()

# ------------------------------------------------------------------------------
# Model
#
# The model is always built from src/rnnoise_data.c. When downloading is
# enabled, the archive is downloaded and extracted into the source tree.
# ------------------------------------------------------------------------------

if(RNNOISE_DOWNLOAD_MODEL)
    file(READ
        "${CMAKE_CURRENT_SOURCE_DIR}/model_version"
        RNNOISE_MODEL_HASH
    )

    string(STRIP
        "${RNNOISE_MODEL_HASH}"
        RNNOISE_MODEL_HASH
    )

    set(RNNOISE_MODEL_NAME
        "rnnoise_data-${RNNOISE_MODEL_HASH}.tar.gz"
    )

    set(RNNOISE_MODEL_ARCHIVE
        "${CMAKE_CURRENT_SOURCE_DIR}/${RNNOISE_MODEL_NAME}"
    )

    file(DOWNLOAD
        "https://media.xiph.org/rnnoise/models/${RNNOISE_MODEL_NAME}"
        "${RNNOISE_MODEL_ARCHIVE}"
        EXPECTED_HASH "SHA256=${RNNOISE_MODEL_HASH}"
        SHOW_PROGRESS
        STATUS RNNOISE_MODEL_DOWNLOAD_STATUS
    )

    list(GET
        RNNOISE_MODEL_DOWNLOAD_STATUS
        0
        RNNOISE_MODEL_DOWNLOAD_RESULT
    )

    if(NOT RNNOISE_MODEL_DOWNLOAD_RESULT EQUAL 0)
        list(GET
            RNNOISE_MODEL_DOWNLOAD_STATUS
            1
            RNNOISE_MODEL_DOWNLOAD_MESSAGE
        )

        message(FATAL_ERROR
            "Failed to download RNNoise model: "
            "${RNNOISE_MODEL_DOWNLOAD_MESSAGE}"
        )
    endif()

    file(ARCHIVE_EXTRACT
        INPUT "${RNNOISE_MODEL_ARCHIVE}"
        DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}"
    )
endif()

# ------------------------------------------------------------------------------
# Library
# ------------------------------------------------------------------------------

set(RNNOISE_SOURCES
    src/denoise.c
    src/rnn.c
    src/pitch.c
    src/kiss_fft.c
    src/celt_lpc.c
    src/nnet.c
    src/nnet_default.c
    src/parse_lpcnet_weights.c
    src/rnnoise_data.c
    src/rnnoise_tables.c
)

add_library(rnnoise
    ${RNNOISE_SOURCES}
)

add_library(rnnoise::rnnoise ALIAS rnnoise)

target_compile_features(rnnoise
    PRIVATE
        c_std_90
)

target_include_directories(rnnoise
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_definitions(rnnoise
    PRIVATE
        RNNOISE_BUILD
        $<$<CONFIG:Debug>:OP_ENABLE_ASSERTIONS=1>
)

if(NOT RNNOISE_ENABLE_DNN_DEBUG_FLOAT)
    target_compile_definitions(rnnoise
        PRIVATE
            DISABLE_DEBUG_FLOAT=1
    )
endif()

if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(rnnoise
        PUBLIC
            RNNOISE_STATIC
    )
endif()

set_target_properties(rnnoise PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    C_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# ------------------------------------------------------------------------------
# Compiler warnings
# ------------------------------------------------------------------------------

if(MSVC)
    target_compile_options(rnnoise
        PRIVATE
            /W4
    )
else()
    target_compile_options(rnnoise
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wno-sign-compare
            -Wno-parentheses
            -Wno-long-long
    )
endif()

# ------------------------------------------------------------------------------
# x86-64 compiler ISA support
#
# v1: baseline x86-64
# v2: SSE4.2
# v3: AVX2 + FMA
# v4: AVX-512
# ------------------------------------------------------------------------------

if(MSVC)
    set(RNNOISE_X86_64_V2_FLAG "/arch:SSE4.2")
    set(RNNOISE_X86_64_V3_FLAG "/arch:AVX2")
    set(RNNOISE_X86_64_V4_FLAG "/arch:AVX512")
else()
    set(RNNOISE_X86_64_V2_FLAG "-march=x86-64-v2")
    set(RNNOISE_X86_64_V3_FLAG "-march=x86-64-v3")
    set(RNNOISE_X86_64_V4_FLAG "-march=x86-64-v4")
endif()

set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

check_c_compiler_flag(
    "${RNNOISE_X86_64_V2_FLAG}"
    RNNOISE_HAVE_X86_64_V2
)

check_c_compiler_flag(
    "${RNNOISE_X86_64_V3_FLAG}"
    RNNOISE_HAVE_X86_64_V3
)

check_c_compiler_flag(
    "${RNNOISE_X86_64_V4_FLAG}"
    RNNOISE_HAVE_X86_64_V4
)

# ------------------------------------------------------------------------------
# x86-64 RTCD
# ------------------------------------------------------------------------------

if(RNNOISE_HAVE_X86_64_V2 OR
   RNNOISE_HAVE_X86_64_V3 OR
   RNNOISE_HAVE_X86_64_V4)

    target_sources(rnnoise
        PRIVATE
            src/x86/x86_dnn_map.c
            src/x86/x86cpu.c
    )

    target_compile_definitions(rnnoise
        PRIVATE
            RNN_ENABLE_X86_RTCD=1
            CPU_INFO_BY_ASM=1
    )
endif()

# ------------------------------------------------------------------------------
# x86-64 v2
# ------------------------------------------------------------------------------

if(RNNOISE_HAVE_X86_64_V2)
    target_sources(rnnoise
        PRIVATE
            src/x86/nnet_sse4_2.c
    )

    set_source_files_properties(
        src/x86/nnet_sse4_2.c
        PROPERTIES
            COMPILE_OPTIONS "${RNNOISE_X86_64_V2_FLAG}"
    )

    target_compile_definitions(rnnoise
        PRIVATE
            RNNOISE_X86_64_V2=1
    )
endif()

# ------------------------------------------------------------------------------
# x86-64 v3
# ------------------------------------------------------------------------------

if(RNNOISE_HAVE_X86_64_V3)
    target_sources(rnnoise
        PRIVATE
            src/x86/nnet_avx2.c
    )

    set_source_files_properties(
        src/x86/nnet_avx2.c
        PROPERTIES
            COMPILE_OPTIONS "${RNNOISE_X86_64_V3_FLAG}"
    )

    target_compile_definitions(rnnoise
        PRIVATE
            RNNOISE_X86_64_V3=1
    )
endif()

# ------------------------------------------------------------------------------
# x86-64 v4
# ------------------------------------------------------------------------------

if(RNNOISE_HAVE_X86_64_V4)
    target_sources(rnnoise
        PRIVATE
            src/x86/nnet_avx512.c
    )

    set_source_files_properties(
        src/x86/nnet_avx512.c
        PROPERTIES
            COMPILE_OPTIONS "${RNNOISE_X86_64_V4_FLAG}"
    )

    target_compile_definitions(rnnoise
        PRIVATE
            RNNOISE_X86_64_V4=1
    )
endif()

# ------------------------------------------------------------------------------
# Libm
# ------------------------------------------------------------------------------

set(lrintf_lib)

if(NOT WIN32)
    check_library_exists(
        m
        lrintf
        ""
        RNNOISE_HAVE_LIBM
    )

    if(RNNOISE_HAVE_LIBM)
        target_link_libraries(rnnoise
            PRIVATE
                m
        )

        set(lrintf_lib "-lm")
    endif()
endif()

# ------------------------------------------------------------------------------
# Example
# ------------------------------------------------------------------------------

if(RNNOISE_BUILD_EXAMPLES)
    add_executable(rnnoise_demo
        examples/rnnoise_demo.c
    )

    target_link_libraries(rnnoise_demo
        PRIVATE
            rnnoise
    )
endif()

# ------------------------------------------------------------------------------
# Tools
# ------------------------------------------------------------------------------

add_executable(dump_features
    src/dump_features.c
)

target_compile_definitions(dump_features
    PRIVATE
        TRAINING
)

target_include_directories(dump_features
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(dump_features
    PRIVATE
        rnnoise
)

add_executable(dump_weights_blob
    src/write_weights.c
)

target_compile_definitions(dump_weights_blob
    PRIVATE
        DUMP_BINARY_WEIGHTS
)

target_include_directories(dump_weights_blob
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(NOT WIN32 AND RNNOISE_HAVE_LIBM)
    target_link_libraries(dump_features
        PRIVATE
            m
    )

    target_link_libraries(dump_weights_blob
        PRIVATE
            m
    )
endif()

# ------------------------------------------------------------------------------
# Documentation
# ------------------------------------------------------------------------------

if(RNNOISE_BUILD_DOCS)
    find_package(Doxygen QUIET)

    if(Doxygen_FOUND)
        doxygen_add_docs(
            rnnoise_docs
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            COMMENT "Generating RNNoise API documentation"
        )
    endif()
endif()

# ------------------------------------------------------------------------------
# pkg-config
# ------------------------------------------------------------------------------

set(PACKAGE_VERSION "${PROJECT_VERSION}")

set(prefix "${CMAKE_INSTALL_PREFIX}")
set(exec_prefix "\${prefix}")
set(libdir "\${prefix}/${CMAKE_INSTALL_LIBDIR}")
set(includedir "\${prefix}/${CMAKE_INSTALL_INCLUDEDIR}")

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/rnnoise.pc.in"
    "${CMAKE_CURRENT_BINARY_DIR}/rnnoise.pc"
    @ONLY
)

# ------------------------------------------------------------------------------
# CMake package
# ------------------------------------------------------------------------------

set(RNNOISE_CMAKE_INSTALL_DIR
    "${CMAKE_INSTALL_LIBDIR}/cmake/rnnoise"
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/rnnoise-config-version.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
)

file(WRITE
    "${CMAKE_CURRENT_BINARY_DIR}/rnnoise-config.cmake"
    "include(\"\${CMAKE_CURRENT_LIST_DIR}/rnnoise-targets.cmake\")\n"
)

# ------------------------------------------------------------------------------
# Installation
# ------------------------------------------------------------------------------

install(
    TARGETS rnnoise
    EXPORT rnnoise-targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
    DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
    FILES
        COPYING
        AUTHORS
        README
    DESTINATION ${CMAKE_INSTALL_DOCDIR}
)

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/rnnoise.pc"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)

install(
    EXPORT rnnoise-targets
    FILE rnnoise-targets.cmake
    NAMESPACE rnnoise::
    DESTINATION ${RNNOISE_CMAKE_INSTALL_DIR}
)

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/rnnoise-config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/rnnoise-config-version.cmake"
    DESTINATION ${RNNOISE_CMAKE_INSTALL_DIR}
)

if(Doxygen_FOUND)
    install(
        DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/html/"
        DESTINATION ${CMAKE_INSTALL_DOCDIR}/html
    )
endif()
