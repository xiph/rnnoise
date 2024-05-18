set (CMAKE_CXX_STANDARD 17)

add_executable(rnnoise_demo rnnoise_demo.c)

target_link_libraries(rnnoise_demo PRIVATE rnnoise)

find_package(SndFile REQUIRED)
find_package(cxxopts REQUIRED)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)


add_executable(rnnoise_libsoundfile rnnoise_libsndfile.cpp)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    message(WARNING "Compiling RNNOISE for Windows...")
    target_compile_definitions(rnnoise_libsoundfile PRIVATE WINDOWS_SPECIFIC_MACRO)
endif()

target_link_libraries(
    rnnoise_libsoundfile
    PRIVATE
    rnnoise
    SndFile::sndfile
    cxxopts::cxxopts
    fmt::fmt
    spdlog::spdlog
)
target_include_directories(rnnoise_libsoundfile PRIVATE ${CMAKE_CURRENT_LIST_DIR})