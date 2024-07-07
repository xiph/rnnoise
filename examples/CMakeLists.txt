set (CMAKE_CXX_STANDARD 17)

add_executable(rnnoise_demo rnnoise_demo.c)

target_link_libraries(rnnoise_demo PRIVATE rnnoise)

find_package(SndFile REQUIRED)
find_package(cxxopts REQUIRED)
find_package(fmt REQUIRED)
find_package(mimalloc REQUIRED)

add_executable(rnnoise_libsoundfile
    rnnoise_libsndfile.cpp
    profiling/xcorr_impl.cpp
    profiling/xcorr_offload_kernel.cpp
)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    message(WARNING "Compiling RNNOISE for Windows...")
    target_compile_definitions(rnnoise_libsoundfile PRIVATE WINDOWS_SPECIFIC_MACRO)
endif()

target_link_libraries(
    rnnoise_libsoundfile
    PRIVATE
    rnnoise_amalgamated
    SndFile::sndfile
    cxxopts::cxxopts
    fmt::fmt
    mimalloc-static
)
target_include_directories(rnnoise_libsoundfile PRIVATE ${CMAKE_CURRENT_LIST_DIR})