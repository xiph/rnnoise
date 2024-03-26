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

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "rnnoise.h"
#include "common.h"
#include "denoise.h"
#include "arch.h"
#include "kiss_fft.h"

int lowpass = FREQ_SIZE;
int band_lp = NB_BANDS;

static float uni_rand() {
  return rand()/(double)RAND_MAX-.5;
}

static void rand_resp(float *a, float *b) {
  a[0] = .75*uni_rand();
  a[1] = .75*uni_rand();
  b[0] = .75*uni_rand();
  b[1] = .75*uni_rand();
}

int main(int argc, char **argv) {
  int i;
  int count=0;
  static const float a_hp[2] = {-1.99599, 0.99600};
  static const float b_hp[2] = {-2, 1};
  float a_noise[2] = {0};
  float b_noise[2] = {0};
  float a_sig[2] = {0};
  float b_sig[2] = {0};
  float mem_hp_x[2]={0};
  float mem_hp_n[2]={0};
  float mem_resp_x[2]={0};
  float mem_resp_n[2]={0};
  float x[FRAME_SIZE];
  float n[FRAME_SIZE];
  float xn[FRAME_SIZE];
  int vad_cnt=0;
  int gain_change_count=0;
  float speech_gain = 1, noise_gain = 1;
  FILE *f1, *f2;
  int maxCount;
  DenoiseState *st;
  DenoiseState *noise_state;
  DenoiseState *noisy;
  st = rnnoise_create(NULL);
  noise_state = rnnoise_create(NULL);
  noisy = rnnoise_create(NULL);
  if (argc!=4) {
    fprintf(stderr, "usage: %s <speech> <noise> <count>\n", argv[0]);
    return 1;
  }
  f1 = fopen(argv[1], "r");
  f2 = fopen(argv[2], "r");
  maxCount = atoi(argv[3]);
  for(i=0;i<150;i++) {
    short tmp[FRAME_SIZE];
    fread(tmp, sizeof(short), FRAME_SIZE, f2);
  }
  while (1) {
    kiss_fft_cpx X[FREQ_SIZE], Y[FREQ_SIZE], N[FREQ_SIZE], P[WINDOW_SIZE];
    float Ex[NB_BANDS], Ey[NB_BANDS], En[NB_BANDS], Ep[NB_BANDS];
    float Exp[NB_BANDS];
    float Ln[NB_BANDS];
    float features[NB_FEATURES];
    float g[NB_BANDS];
    short tmp[FRAME_SIZE];
    float vad=0;
    float E=0;
    if (count==maxCount) break;
    if ((count%1000)==0) fprintf(stderr, "%d\r", count);
    if (++gain_change_count > 2821) {
      speech_gain = pow(10., (-40+(rand()%60))/20.);
      noise_gain = pow(10., (-30+(rand()%50))/20.);
      if (rand()%10==0) noise_gain = 0;
      noise_gain *= speech_gain;
      if (rand()%10==0) speech_gain = 0;
      gain_change_count = 0;
      rand_resp(a_noise, b_noise);
      rand_resp(a_sig, b_sig);
      lowpass = FREQ_SIZE * 3000./24000. * pow(50., rand()/(double)RAND_MAX);
      for (i=0;i<NB_BANDS;i++) {
        if (eband20ms[i] > lowpass) {
          band_lp = i;
          break;
        }
      }
    }
    if (speech_gain != 0) {
      fread(tmp, sizeof(short), FRAME_SIZE, f1);
      if (feof(f1)) {
        rewind(f1);
        fread(tmp, sizeof(short), FRAME_SIZE, f1);
      }
      for (i=0;i<FRAME_SIZE;i++) x[i] = speech_gain*tmp[i];
      for (i=0;i<FRAME_SIZE;i++) E += tmp[i]*(float)tmp[i];
    } else {
      for (i=0;i<FRAME_SIZE;i++) x[i] = 0;
      E = 0;
    }
    if (noise_gain!=0) {
      fread(tmp, sizeof(short), FRAME_SIZE, f2);
      if (feof(f2)) {
        rewind(f2);
        fread(tmp, sizeof(short), FRAME_SIZE, f2);
      }
      for (i=0;i<FRAME_SIZE;i++) n[i] = noise_gain*tmp[i];
    } else {
      for (i=0;i<FRAME_SIZE;i++) n[i] = 0;
    }
    rnn_biquad(x, mem_hp_x, x, b_hp, a_hp, FRAME_SIZE);
    rnn_biquad(x, mem_resp_x, x, b_sig, a_sig, FRAME_SIZE);
    rnn_biquad(n, mem_hp_n, n, b_hp, a_hp, FRAME_SIZE);
    rnn_biquad(n, mem_resp_n, n, b_noise, a_noise, FRAME_SIZE);
    for (i=0;i<FRAME_SIZE;i++) xn[i] = x[i] + n[i];
    if (E > 1e9f) {
      vad_cnt=0;
    } else if (E > 1e8f) {
      vad_cnt -= 5;
    } else if (E > 1e7f) {
      vad_cnt++;
    } else {
      vad_cnt+=2;
    }
    if (vad_cnt < 0) vad_cnt = 0;
    if (vad_cnt > 15) vad_cnt = 15;

    if (vad_cnt >= 10) vad = 0;
    else if (vad_cnt > 0) vad = 0.5f;
    else vad = 1.f;

    rnn_frame_analysis(st, Y, Ey, x);
    rnn_frame_analysis(noise_state, N, En, n);
    for (i=0;i<NB_BANDS;i++) Ln[i] = log10(1e-2+En[i]);
    int silence = rnn_compute_frame_features(noisy, X, P, Ex, Ep, Exp, features, xn);
    rnn_pitch_filter(X, P, Ex, Ep, Exp, g);
    //printf("%f %d\n", noisy->last_gain, noisy->last_period);
    for (i=0;i<NB_BANDS;i++) {
      g[i] = sqrt((Ey[i]+1e-3)/(Ex[i]+1e-3));
      if (g[i] > 1) g[i] = 1;
      if (silence || i > band_lp) g[i] = -1;
      if (Ey[i] < 5e-2 && Ex[i] < 5e-2) g[i] = -1;
      if (vad==0 && noise_gain==0) g[i] = -1;
    }
    count++;
#if 1
    fwrite(features, sizeof(float), NB_FEATURES, stdout);
    fwrite(g, sizeof(float), NB_BANDS, stdout);
    fwrite(&vad, sizeof(float), 1, stdout);
#endif
  }
  fprintf(stderr, "matrix size: %d x %d\n", count, NB_FEATURES + 2*NB_BANDS + 1);
  fclose(f1);
  fclose(f2);
  return 0;
}
