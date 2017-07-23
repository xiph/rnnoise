#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "kiss_fft.h"
#include "common.h"
#include <math.h>

#define FRAME_SIZE_SHIFT 2
#define FRAME_SIZE (120<<FRAME_SIZE_SHIFT)
#define WINDOW_SIZE (2*FRAME_SIZE)
#define FREQ_SIZE (FRAME_SIZE + 1)


#define SQUARE(x) ((x)*(x))

#define SMOOTH_BANDS 1

#if SMOOTH_BANDS
#define NB_BANDS 22
#else
#define NB_BANDS 21
#endif

#define CEPS_MEM 8
#define NB_DELTA_CEPS 6

#define NB_FEATURES (NB_BANDS+2*NB_DELTA_CEPS+1)

static const opus_int16 eband5ms[] = {
/*0  200 400 600 800  1k 1.2 1.4 1.6  2k 2.4 2.8 3.2  4k 4.8 5.6 6.8  8k 9.6 12k 15.6 20k*/
  0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100
};


typedef struct {
  int init;
  kiss_fft_state *kfft;
  float half_window[FRAME_SIZE];
  float dct_table[NB_BANDS*NB_BANDS];
} CommonState;

typedef struct {
  float analysis_mem[FRAME_SIZE];
  float cepstral_mem[CEPS_MEM][NB_BANDS];
  int memid;
  float synthesis_mem[FRAME_SIZE];
} DenoiseState;

#if SMOOTH_BANDS
void compute_band_energy(float *bandE, const kiss_fft_cpx *X) {
  int i;
  float sum[NB_BANDS] = {0};
  for (i=0;i<NB_BANDS-1;i++)
  {
    int j;
    int band_size;
    band_size = (eband5ms[i+1]-eband5ms[i])<<FRAME_SIZE_SHIFT;
    for (j=0;j<band_size;j++) {
      float tmp;
      float frac = (float)j/band_size;
      tmp = SQUARE(X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j].r);
      tmp += SQUARE(X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j].i);
      sum[i] += (1-frac)*tmp;
      sum[i+1] += frac*tmp;
    }
  }
  sum[0] *= 2;
  sum[NB_BANDS-1] *= 2;
  for (i=0;i<NB_BANDS;i++)
  {
    bandE[i] = sum[i];
  }
}

void interp_band_gain(float *g, const float *bandE) {
  int i;
  memset(g, 0, FREQ_SIZE);
  for (i=0;i<NB_BANDS-1;i++)
  {
    int j;
    int band_size;
    band_size = (eband5ms[i+1]-eband5ms[i])<<FRAME_SIZE_SHIFT;
    for (j=0;j<band_size;j++) {
      float frac = (float)j/band_size;
      g[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j] = (1-frac)*bandE[i] + frac*bandE[i+1];
    }
  }
}
#else
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
    bandE[i] = sum;
  }
}

void interp_band_gain(float *g, const float *bandE) {
  int i;
  memset(g, 0, FREQ_SIZE);
  for (i=0;i<NB_BANDS;i++)
  {
    int j;
    for (j=0;j<(eband5ms[i+1]-eband5ms[i])<<FRAME_SIZE_SHIFT;j++)
      g[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j] = bandE[i];
  }
}
#endif


CommonState common;

static void check_init() {
  int i;
  if (common.init) return;
  common.kfft = opus_fft_alloc_twiddles(2*FRAME_SIZE, NULL, NULL, NULL, 0);
  for (i=0;i<FRAME_SIZE;i++)
    common.half_window[i] = sin(.5*M_PI*sin(.5*M_PI*(i+.5)/FRAME_SIZE) * sin(.5*M_PI*(i+.5)/FRAME_SIZE));
  for (i=0;i<NB_BANDS;i++) {
    int j;
    for (j=0;j<NB_BANDS;j++) {
      common.dct_table[i*NB_BANDS + j] = cos((i+.5)*j*M_PI/NB_BANDS);
      if (j==0) common.dct_table[i*NB_BANDS + j] *= sqrt(.5);
    }
  }
  common.init = 1;
}

static void dct(float *out, const float *in) {
  int i;
  check_init();
  for (i=0;i<NB_BANDS;i++) {
    int j;
    float sum = 0;
    for (j=0;j<NB_BANDS;j++) {
      sum += in[j] * common.dct_table[j*NB_BANDS + i];
    }
    out[i] = sum*sqrt(2./22);
  }
}

#if 0
static void idct(float *out, const float *in) {
  int i;
  check_init();
  for (i=0;i<NB_BANDS;i++) {
    int j;
    float sum = 0;
    for (j=0;j<NB_BANDS;j++) {
      sum += in[j] * common.dct_table[i*NB_BANDS + j];
    }
    out[i] = sum*sqrt(2./22);
  }
}
#endif

static void forward_transform(kiss_fft_cpx *out, const float *in) {
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

static void inverse_transform(float *out, const kiss_fft_cpx *in) {
  int i;
  kiss_fft_cpx x[WINDOW_SIZE];
  kiss_fft_cpx y[WINDOW_SIZE];
  check_init();
  for (i=0;i<FREQ_SIZE;i++) {
    x[i] = in[i];
  }
  for (;i<WINDOW_SIZE;i++) {
    x[i].r = x[WINDOW_SIZE - i].r;
    x[i].i = -x[WINDOW_SIZE - i].i;
  }
  opus_fft(common.kfft, x, y, 0);
  /* output in reverse order for IFFT. */
  out[0] = WINDOW_SIZE*y[0].r;
  for (i=1;i<WINDOW_SIZE;i++) {
    out[i] = WINDOW_SIZE*y[WINDOW_SIZE - i].r;
  }
}

static void apply_window(float *x) {
  int i;
  check_init();
  for (i=0;i<FRAME_SIZE;i++) {
    x[i] *= common.half_window[i];
    x[WINDOW_SIZE - 1 - i] *= common.half_window[i];
  }
}

int rnnoise_init(DenoiseState *st) {
  memset(st, 0, sizeof(*st));
  return 0;
}

DenoiseState *rnnoise_create() {
  DenoiseState *st;
  st = malloc(sizeof(DenoiseState));
  rnnoise_init(st);
  return st;
}


static void frame_analysis(DenoiseState *st, kiss_fft_cpx *y, float *Ey, float *features, const float *in) {
  float x[WINDOW_SIZE];
  int i;
  RNN_COPY(x, st->analysis_mem, FRAME_SIZE);
  for (i=0;i<FRAME_SIZE;i++) x[FRAME_SIZE + i] = in[i];
  RNN_COPY(st->analysis_mem, in, FRAME_SIZE);
  apply_window(x);
  forward_transform(y, x);
  if (Ey != NULL) {
    compute_band_energy(Ey, y);
    if (features != NULL) {
      float *ceps_0, *ceps_1, *ceps_2;
      float spec_variability = 0;
      float Ly[NB_BANDS];
      for (i=0;i<NB_BANDS;i++) Ly[i] = log10(1e-10+Ey[i]);
      dct(features, Ly);
      features[0] -= 12;
      features[1] -= 4;
      ceps_0 = st->cepstral_mem[st->memid];
      ceps_1 = (st->memid < 1) ? st->cepstral_mem[CEPS_MEM+st->memid-1] : st->cepstral_mem[st->memid-1];
      ceps_2 = (st->memid < 2) ? st->cepstral_mem[CEPS_MEM+st->memid-2] : st->cepstral_mem[st->memid-2];
      for (i=0;i<NB_BANDS;i++) ceps_0[i] = features[i];
      st->memid++;
      for (i=0;i<NB_DELTA_CEPS;i++) {
        features[i] = ceps_0[i] + ceps_1[i] + ceps_2[i];
        features[NB_BANDS+i] = ceps_0[i] - ceps_2[i];
        features[NB_BANDS+NB_DELTA_CEPS+i] =  ceps_0[i] - 2*ceps_1[i] + ceps_2[i];
      }
      /* Spectral variability features. */
      if (st->memid == CEPS_MEM) st->memid = 0;
      for (i=0;i<CEPS_MEM;i++)
      {
        int j;
        float mindist = 1e15f;
        for (j=0;j<CEPS_MEM;j++)
        {
          int k;
          float dist=0;
          for (k=0;k<NB_BANDS;k++)
          {
            float tmp;
            tmp = st->cepstral_mem[i][k] - st->cepstral_mem[j][k];
            dist += tmp*tmp;
          }
          if (j!=i)
            mindist = MIN32(mindist, dist);
        }
        spec_variability += mindist;
      }
      features[NB_BANDS+2*NB_DELTA_CEPS] = spec_variability/CEPS_MEM-2.1;
    }
  }
}

static void frame_synthesis(DenoiseState *st, float *out, const kiss_fft_cpx *y) {
  float x[WINDOW_SIZE];
  int i;
  inverse_transform(x, y);
  apply_window(x);
  for (i=0;i<FRAME_SIZE;i++) out[i] = x[i] + st->synthesis_mem[i];
  RNN_COPY(st->synthesis_mem, &x[FRAME_SIZE], FRAME_SIZE);
}

void rnnoise_process_frame(DenoiseState *st, float *out, const float *in) {
  kiss_fft_cpx y[FREQ_SIZE];
  frame_analysis(st, y, NULL, NULL, in);
  /* Do processing here. */
  frame_synthesis(st, out, y);
}

int main(int argc, char **argv) {
  int i;
  float x[FRAME_SIZE];
  float n[FRAME_SIZE];
  float xn[FRAME_SIZE];
  int vad_cnt=0;
  FILE *f1, *f2, *fout;
  DenoiseState *st;
  DenoiseState *noise_state;
  DenoiseState *noisy;
  st = rnnoise_create();
  noise_state = rnnoise_create();
  noisy = rnnoise_create();
  if (argc!=4) {
    fprintf(stderr, "usage: %s <speech> <noise> <output denoised>\n", argv[0]);
    return 1;
  }
  f1 = fopen(argv[1], "r");
  f2 = fopen(argv[2], "r");
  fout = fopen(argv[3], "w");
  for(i=0;i<150;i++) {
    short tmp[FRAME_SIZE];
    fread(tmp, sizeof(short), FRAME_SIZE, f2);
  }
  while (1) {
    kiss_fft_cpx X[FREQ_SIZE], Y[FREQ_SIZE], N[FREQ_SIZE];
    float Ex[NB_BANDS], Ey[NB_BANDS], En[NB_BANDS];
    float Ln[NB_BANDS];
    float features[NB_FEATURES];
    float g[NB_BANDS];
    float gf[FREQ_SIZE];
    short tmp[FRAME_SIZE];
    float vad=0;
    float E=0;
    fread(tmp, sizeof(short), FRAME_SIZE, f1);
    if (feof(f1)) break;
    for (i=0;i<FRAME_SIZE;i++) x[i] = tmp[i];
    fread(tmp, sizeof(short), FRAME_SIZE, f2);
    if (feof(f2)) break;
    for (i=0;i<FRAME_SIZE;i++) n[i] = tmp[i];
    for (i=0;i<FRAME_SIZE;i++) xn[i] = x[i] + n[i];
    for (i=0;i<FRAME_SIZE;i++) E += x[i]*(float)x[i];
    if (E > 1e9f) {
      vad_cnt=0;
    } else if (E > 1e8f) {
      vad_cnt -= 5;
      if (vad_cnt < 0) vad_cnt = 0;
    } else {
      vad_cnt++;
      if (vad_cnt > 15) vad_cnt = 15;
    }
    if (vad_cnt >= 10) vad = 0;
    else if (vad_cnt > 0) vad = 0.5f;
    else vad = 1.f;

    frame_analysis(st, X, Ex, NULL, x);
    frame_analysis(noise_state, N, En, NULL, n);
    for (i=0;i<NB_BANDS;i++) Ln[i] = log10(1e-10+En[i]);
    frame_analysis(noisy, Y, Ey, features, xn);
    for (i=0;i<NB_FEATURES;i++) printf("%f ", features[i]);
    for (i=0;i<NB_BANDS;i++) {
      g[i] = sqrt((Ex[i]+1e-15)/(Ey[i]+1e-15));
      if (g[i] > 1) g[i] = 1;
    }
    interp_band_gain(gf, g);
#if 1
    for (i=0;i<NB_BANDS;i++) printf("%f ", g[i]);
    for (i=0;i<NB_BANDS;i++) printf("%f ", Ln[i]);
    printf("%f\n", vad);
#endif
#if 1
    for (i=0;i<FREQ_SIZE;i++) {
      Y[i].r *= gf[i];
      Y[i].i *= gf[i];
    }
#endif
    frame_synthesis(noisy, xn, Y);

    for (i=0;i<FRAME_SIZE;i++) tmp[i] = xn[i];
    fwrite(tmp, sizeof(short), FRAME_SIZE, fout);
  }
  fclose(f1);
  fclose(f2);
  fclose(fout);
  return 0;
}
