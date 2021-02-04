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

#include <stdio.h>
#include "rnnoise.h"

#define FRAME_SIZE 480

int main(int argc, char **argv) {
  int i;
  int first = 1;
  float x[FRAME_SIZE];
  RNNModel *model = NULL;
  FILE *f1, *fout;
  FILE *model_fptr;
  DenoiseState *st;

  if (argc < 3) {
    fprintf(stderr, "usage: %s <noisy speech> <output denoised> [<model file>]\n", argv[0]);
    return 1;
  }
  if (argc >= 4) {
    model_fptr = fopen(argv[3], "r");
    if (!model_fptr) {
      fprintf(stderr, "Error opening model file \n");
      return 1;
    }
    model = rnnoise_model_from_file(model_fptr);
    if (!model) {
      fprintf(stderr, "Model not found \n");
      return 1;
    }
  }

  st = rnnoise_create(model);
  f1 = fopen(argv[1], "rb");
  fout = fopen(argv[2], "wb");
  if (!f1) {
    fprintf(stderr, "Error opening input audio file\n");
    return 1;
  }

  while (1) {
    short tmp[FRAME_SIZE];
    fread(tmp, sizeof(short), FRAME_SIZE, f1);
    if (feof(f1)) break;
    for (i=0;i<FRAME_SIZE;i++) x[i] = tmp[i];
    rnnoise_process_frame(st, x, x);
    for (i=0;i<FRAME_SIZE;i++) tmp[i] = x[i];
    if (!first) fwrite(tmp, sizeof(short), FRAME_SIZE, fout);
    first = 0;
  }
  rnnoise_destroy(st);
  if (model) {
    rnnoise_model_free(model);
  }
  fclose(f1);
  fclose(fout);
  if (model_fptr) {
    fclose(model_fptr);
  }
  return 0;
}
