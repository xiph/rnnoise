#!/bin/sh
echo ##########
echo "Compiling Demo..."
echo ##########

gcc -o rnnoise_demo \
	examples/rnnoise_demo.c \
	src/celt_lpc.c src/denoise.c src/kiss_fft.c \
	src/nnet.c src/nnet_default.c src/parse_lpcnet_weights.c \
	src/pitch.c src/rnn.c src/rnnoise_data.c src/rnnoise_tables.c \
	-I./include -I./src \
	-lm

echo ##########
echo "Compiled rnnoise_demo !!"
ls -alh | grep rnnoise_demo
echo ##########
