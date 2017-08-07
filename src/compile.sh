#!/bin/sh

gcc -Wall -W -O3 -g denoise.c kiss_fft.c pitch.c celt_lpc.c rnn.c rnn_data.c -o denoise -lm
