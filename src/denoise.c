#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "kiss_fft.h"

#define FRAME_SIZE_SHIFT 2
#define FRAME_SIZE (120<<FRAME_SIZE_SHIFT)
#define WINDOW_SIZE (2*FRAME_SIZE)
#define FREQ_SIZE (FRAME_SIZE + 1)

#define NB_BANDS 21

#define SQUARE(x) ((x)*(x))

static const opus_int16 eband5ms[] = {
/*0  200 400 600 800  1k 1.2 1.4 1.6  2k 2.4 2.8 3.2  4k 4.8 5.6 6.8  8k 9.6 12k 15.6 20k*/
  0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100
};


typedef struct {
  int init;
  kiss_fft_state *kfft;
} CommonState;

typedef struct {
  float analysis_mem[FRAME_SIZE];
  float synthesis_mem[FRAME_SIZE];
  
} DenoiseState;

void compute_band_energy(float *bandE, const kiss_fft_cpx *X) {
  int i;
  for (i=0;i<NB_BANDS;i++)
  {
    int j;
    opus_val32 sum = 1e-27;
    for (j=0;j<(eband5ms[i+1]-eband5ms[i])<<FRAME_SIZE_SHIFT;j++) {
      sum += SQUARE(X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j].r);
      sum += SQUARE(X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j].i);
    }
    bandE[i] = sqrt(sum);
  }
}

void interp_band_gain(float *g, const float *bandE) {
  int i;
  for (i=0;i<NB_BANDS;i++)
  {
    int j;
    for (j=0;j<(eband5ms[i+1]-eband5ms[i])<<FRAME_SIZE_SHIFT;j++)
      g[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j] = bandE[i];
  }
}

CommonState common;

static void check_init() {
  if (common.init) return;
  common.kfft = opus_fft_alloc_twiddles(2*FRAME_SIZE, NULL, NULL, NULL, 0);
  common.init = 1;
}


void forward_transform(kiss_fft_cpx *out, const float *in) {
  int i;
  kiss_fft_cpx x[WINDOW_SIZE];
  kiss_fft_cpx y[WINDOW_SIZE];
  check_init();
  for (i=0;i<WINDOW_SIZE;i++) {
    x[i].r = in[i];
    x[i].i = 0;
  }
  opus_fft(common.kfft, x, y, 0);
  for (i=0;i<FREQ_SIZE;i++) {
    out[i] = y[i];
  }
}

void inverse_transform(float *out, const kiss_fft_cpx *in) {
  check_init();
}

int main() {
  int i;
  kiss_fft_state *kfft;
  float x[2*FRAME_SIZE];
  kiss_fft_cpx y[2*FRAME_SIZE];
  float bandE[NB_BANDS];
  memset(x, 0, sizeof(x));
  x[0] = 1;
  x[1] = 1;
  kfft = opus_fft_alloc_twiddles(2*FRAME_SIZE, NULL, NULL, NULL, 0);
  //opus_fft(kfft, x, y, 0);
  forward_transform(y, x);
  compute_band_energy(bandE, y);
  /*for (i=0;i<2*FRAME_SIZE;i++)
    printf("%f %f\n", y[i].r, y[i].i);*/
  for (i=0;i<NB_BANDS;i++)
    printf("%f\n", bandE[i]);
  return 0;
}
