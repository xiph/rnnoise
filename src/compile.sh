#!/bin/sh

gcc -DTRAINING=1 -Wall -W -O3 -g -I../include rnn_denoise.c rnn_kiss_fft.c rnn_pitch.c rnn_celt_lpc.c rnn.c rnn_data.c -o denoise_training -lm
