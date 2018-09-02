/* Copyright (c) 2018 Gregor Richards
 * Copyright (c) 2017 Mozilla */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include "rnnoise-nu.h"

#define FRAME_SIZE 480

int main(int argc, char **argv) {
  int i, ci;
  int first = 1;
  int channels;
  float x[FRAME_SIZE];
  short *tmp;
  int sample_rate;
  RNNModel *model = NULL;
  DenoiseState **sts;
  float max_attenuation;
  if (argc < 4) {
    fprintf(stderr, "usage: %s <sample rate> <channels> <max attenuation dB> [model]\n", argv[0]);
    return 1;
  }

  sample_rate = atoi(argv[1]);
  if (sample_rate <= 0) sample_rate = 48000;
  channels = atoi(argv[2]);
  if (channels < 1) channels = 1;
  max_attenuation = pow(10, -atof(argv[3])/10);

  if (argc >= 5) {
      model = rnnoise_get_model(argv[4]);
      if (!model) {
          fprintf(stderr, "Model not found!\n");
          return 1;
      }
  }

  sts = malloc(channels * sizeof(DenoiseState *));
  if (!sts) {
    perror("malloc");
    return 1;
  }
  tmp = malloc(channels * FRAME_SIZE * sizeof(short));
  if (!tmp) {
      perror("malloc");
      return 1;
  }
  for (i = 0; i < channels; i++) {
    sts[i] = rnnoise_create(model);
    rnnoise_set_param(sts[i], RNNOISE_PARAM_MAX_ATTENUATION, max_attenuation);
    rnnoise_set_param(sts[i], RNNOISE_PARAM_SAMPLE_RATE, sample_rate);
  }

  while (1) {
    fread(tmp, sizeof(short), channels * FRAME_SIZE, stdin);
    if (feof(stdin)) break;

    for (ci = 0; ci < channels; ci++) {
        for (i=0;i<FRAME_SIZE;i++) x[i] = tmp[i*channels+ci];
        rnnoise_process_frame(sts[ci], x, x);
        for (i=0;i<FRAME_SIZE;i++) tmp[i*channels+ci] = x[i];
    }

    if (!first) fwrite(tmp, sizeof(short), channels * FRAME_SIZE, stdout);
    first = 0;
  }

  for (i = 0; i < channels; i++)
    rnnoise_destroy(sts[i]);
  free(tmp);
  free(sts);
  return 0;
}
