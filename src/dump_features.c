/* Copyright (c) 2017 Mozilla */
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif


#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include "rnnoise.h"
#include "common.h"
#include "denoise.h"
#include "arch.h"
#include "kiss_fft.h"

int lowpass = FREQ_SIZE;
int band_lp = NB_BANDS;

#define SEQUENCE_LENGTH 2000

static unsigned rand_lcg(unsigned *seed) {
  *seed = 1664525**seed + 1013904223;
  return *seed;
}

static float uni_rand() {
  return rand()/(double)RAND_MAX-.5;
}

static void rand_resp(float *a, float *b) {
  a[0] = .75*uni_rand();
  a[1] = .75*uni_rand();
  b[0] = .75*uni_rand();
  b[1] = .75*uni_rand();
}

short speech16[SEQUENCE_LENGTH*FRAME_SIZE];
short noise16[SEQUENCE_LENGTH*FRAME_SIZE];
float x[SEQUENCE_LENGTH*FRAME_SIZE];
float n[SEQUENCE_LENGTH*FRAME_SIZE];
float xn[SEQUENCE_LENGTH*FRAME_SIZE];

    
int main(int argc, char **argv) {
  int i;
  int count=0;
  static const float a_hp[2] = {-1.99599, 0.99600};
  static const float b_hp[2] = {-2, 1};
  float a_noise[2] = {0};
  float b_noise[2] = {0};
  float a_sig[2] = {0};
  float b_sig[2] = {0};
  float speech_gain = 1, noise_gain = 1;
  FILE *f1, *f2, *fout;
  long speech_length, noise_length;
  int maxCount;
  unsigned seed;
  DenoiseState *st;
  DenoiseState *noisy;
  seed = getpid();
  srand(seed);
  st = rnnoise_create(NULL);
  noisy = rnnoise_create(NULL);
  if (argc!=5) {
    fprintf(stderr, "usage: %s <speech> <noise> <output> <count>\n", argv[0]);
    return 1;
  }
  f1 = fopen(argv[1], "rb");
  f2 = fopen(argv[2], "rb");
  fout = fopen(argv[3], "wb");

  fseek(f1, 0, SEEK_END);
  speech_length = ftell(f1);
  fseek(f1, 0, SEEK_SET);
  fseek(f2, 0, SEEK_END);
  noise_length = ftell(f2);
  fseek(f2, 0, SEEK_SET);

  maxCount = atoi(argv[4]);
  for (count=0;count<maxCount;count++) {
    long speech_pos, noise_pos;
    int start_pos;
    float E[SEQUENCE_LENGTH] = {0};
    float mem[2]={0};
    int frame;
    int silence;
    kiss_fft_cpx X[FREQ_SIZE], Y[FREQ_SIZE], P[WINDOW_SIZE];
    float Ex[NB_BANDS], Ey[NB_BANDS], Ep[NB_BANDS];
    float Exp[NB_BANDS];
    float features[NB_FEATURES];
    float g[NB_BANDS];
    if ((count%1000)==0) fprintf(stderr, "%d\r", count);
    speech_pos = (rand_lcg(&seed)*2.3283e-10)*speech_length;
    noise_pos = (rand_lcg(&seed)*2.3283e-10)*noise_length;
    if (speech_pos > speech_length-(long)sizeof(speech16)) speech_pos = speech_length-sizeof(speech16);
    if (noise_pos > noise_length-(long)sizeof(noise16)) noise_pos = noise_length-sizeof(noise16);
    speech_pos -= speech_pos&1;
    noise_pos -= noise_pos&1;
    fseek(f1, speech_pos, SEEK_SET);
    fseek(f2, noise_pos, SEEK_SET);
    fread(speech16, sizeof(speech16), 1, f1);
    fread(noise16, sizeof(noise16), 1, f2);
    if (rand()%4) start_pos = 0;
    else start_pos = -(int)(1000*log(rand()/(float)RAND_MAX));
    start_pos = IMIN(start_pos, SEQUENCE_LENGTH*FRAME_SIZE);
    RNN_CLEAR(speech16, start_pos);

    speech_gain = pow(10., (-40+(rand()%60))/20.);
    noise_gain = pow(10., (-30+(rand()%50))/20.);
    if (rand()%10==0) noise_gain = 0;
    noise_gain *= speech_gain;
    rand_resp(a_noise, b_noise);
    rand_resp(a_sig, b_sig);
    lowpass = FREQ_SIZE * 3000./24000. * pow(50., rand()/(double)RAND_MAX);
    for (i=0;i<NB_BANDS;i++) {
      if (eband20ms[i] > lowpass) {
        band_lp = i;
        break;
      }
    }

    for (frame=0;frame<SEQUENCE_LENGTH;frame++) {
      int j;
      E[frame] = 0;
      for(j=0;j<FRAME_SIZE;j++) {
        float s = speech16[frame*FRAME_SIZE+j];
        E[frame] += s*s;
        x[frame*FRAME_SIZE+j] = speech_gain*speech16[frame*FRAME_SIZE+j];
        n[frame*FRAME_SIZE+j] = noise_gain*noise16[frame*FRAME_SIZE+j];
      }
    }

    RNN_CLEAR(mem, 2);
    rnn_biquad(x, mem, x, b_hp, a_hp, SEQUENCE_LENGTH*FRAME_SIZE);
    RNN_CLEAR(mem, 2);
    rnn_biquad(x, mem, x, b_sig, a_sig, SEQUENCE_LENGTH*FRAME_SIZE);
    RNN_CLEAR(mem, 2);
    rnn_biquad(n, mem, n, b_hp, a_hp, SEQUENCE_LENGTH*FRAME_SIZE);
    RNN_CLEAR(mem, 2);
    rnn_biquad(n, mem, n, b_noise, a_noise, SEQUENCE_LENGTH*FRAME_SIZE);

    for (frame=0;frame<SEQUENCE_LENGTH;frame++) {
      int j;
      int vad;
      for(j=0;j<FRAME_SIZE;j++) {
        xn[frame*FRAME_SIZE+j] = x[frame*FRAME_SIZE+j] + n[frame*FRAME_SIZE+j];
      }
      rnn_frame_analysis(st, Y, Ey, &x[frame*FRAME_SIZE]);
      silence = rnn_compute_frame_features(noisy, X, P, Ex, Ep, Exp, features, &xn[frame*FRAME_SIZE]);
      /*rnn_pitch_filter(X, P, Ex, Ep, Exp, g);*/
      vad = (E[frame] > 1e9f);
      for (i=0;i<NB_BANDS;i++) {
        g[i] = sqrt((Ey[i]+1e-3)/(Ex[i]+1e-3));
        if (g[i] > 1) g[i] = 1;
        if (silence || i > band_lp) g[i] = -1;
        if (Ey[i] < 5e-2 && Ex[i] < 5e-2) g[i] = -1;
        if (vad==0 && noise_gain==0) g[i] = -1;
      }
#if 0
      {
        short tmp[FRAME_SIZE];
        for (j=0;j<FRAME_SIZE;j++) tmp[j] = MIN16(32767, MAX16(-32767, xn[frame*FRAME_SIZE+j]));
        fwrite(tmp, FRAME_SIZE, 2, fout);
      }
#endif
#if 1
      fwrite(features, sizeof(float), NB_FEATURES, fout);
      fwrite(g, sizeof(float), NB_BANDS, fout);
      fwrite(&vad, sizeof(float), 1, fout);
#endif
    }
  }

  fclose(f1);
  fclose(f2);
  fclose(fout);
  return 0;
}
