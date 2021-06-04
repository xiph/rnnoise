set (library_sources 
	denoise.c
	rnn.c
	rnn_data.c
	rnn_reader.c
	pitch.c
	kiss_fft.c
	celt_lpc.c
)

add_library(rnnoise SHARED ${library_sources})
include_directories(rnnoise ../include)
add_library(rnnoise_static ${library_sources})
include_directories(rnnoise_static ../include)
set_target_properties(rnnoise_static PROPERTIES OUTPUT_NAME rnnoise)


