cmake_minimum_required(VERSION 3.21)
project(rnnoise)

option(RNNOISE_COMPILE_OPUS OFF)
option(RNNOISE_COMPILE_DEMO ON)

if(RNNOISE_COMPILE_OPUS)
	add_definitions(-DCOMPILE_OPUS)
endif()

# Ignore CRT warnings on MSVC
if(MSVC)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

#Pre-Generate library resources
set(RNNOISE_MODEL_VERSION "0b50c45")
set(RNNOISE_MODEL_TAR "rnnoise_data-${RNNOISE_MODEL_VERSION}.tar.gz")
set(FILE_URL "https://media.xiph.org/rnnoise/models/${RNNOISE_MODEL_TAR}")
set(RNNOISE_DATA_DOWNLOADS_DIR ${CMAKE_CURRENT_LIST_DIR}/downloads/)
set(RNNOISE_DOWNLOADED_PATH ${RNNOISE_DATA_DOWNLOADS_DIR}/${RNNOISE_MODEL_TAR})



if(NOT EXISTS "${RNNOISE_DOWNLOADED_PATH}")
    
        file(DOWNLOAD ${FILE_URL} ${RNNOISE_DOWNLOADED_PATH}
                STATUS RNNOISE_DOWNLOAD_STATUS
                TIMEOUT 60
                SHOW_PROGRESS)

        if(RNNOISE_DOWNLOAD_STATUS EQUAL 0)
                message("File downloaded successfully.")
        else()
                message(FATAL_ERROR "Failed to download file: ${RNNOISE_DOWNLOAD_STATUS}")

        endif()

        execute_process(
                COMMAND ${CMAKE_COMMAND} -E tar xzf ${RNNOISE_DOWNLOADED_PATH}
                WORKING_DIRECTORY ${RNNOISE_DATA_DOWNLOADS_DIR}
)

endif()

if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/src/rnnoise_data.c")
        set(RNNOISE_MODEL_UNPACKED_SOURCE_DIR ${RNNOISE_DATA_DOWNLOADS_DIR}/src)
        set(RNNOIDE_MODEL_FILES_DESTINATION ${CMAKE_CURRENT_LIST_DIR}/src)

        file(COPY ${RNNOISE_MODEL_UNPACKED_SOURCE_DIR}/rnnoise_data_little.h DESTINATION ${RNNOIDE_MODEL_FILES_DESTINATION})
        file(COPY ${RNNOISE_MODEL_UNPACKED_SOURCE_DIR}/rnnoise_data_little.c DESTINATION ${RNNOIDE_MODEL_FILES_DESTINATION})
        file(COPY ${RNNOISE_MODEL_UNPACKED_SOURCE_DIR}/rnnoise_data.c DESTINATION ${RNNOIDE_MODEL_FILES_DESTINATION})
        file(COPY ${RNNOISE_MODEL_UNPACKED_SOURCE_DIR}/rnnoise_data.h DESTINATION ${RNNOIDE_MODEL_FILES_DESTINATION})
endif()

# Get source files

set(RN_NOISE_SRC
        src/_kiss_fft_guts.h
        src/arch.h
        src/celt_lpc.h
        src/common.h
        src/cpu_support.h
        src/denoise.h
        src/kiss_fft.h
        src/nnet_arch.h
        src/nnet.h
        src/opus_types.h
        src/pitch.h
        src/rnn.h
        src/rnnoise_data.h
        include/rnnoise.h
        src/vec.h

            src/celt_lpc.c
            src/nnet_default.c
            src/rnn.c
            src/rnnoise_tables.c
            src/denoise.c
            src/kiss_fft.c
            src/parse_lpcnet_weights.c
            src/rnnoise_data.c
            #write_weights.c
            #dump_features.c
            src/nnet.c
            src/pitch.c
        )

add_library(rnnoise STATIC ${RN_NOISE_SRC})

add_definitions(-DRNNOISE_BUILD)

# Include dirs
target_include_directories(rnnoise PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        PRIVATE src)

if (UNIX)
        target_link_libraries(rnnoise PUBLIC m)
endif(UNIX)

add_subdirectory(examples)