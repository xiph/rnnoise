/* Copyright (c) 2008-2011 Octasic Inc.
                 2012-2017 Jean-Marc Valin */
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

#include <math.h>
#include "opus_types.h"
#include "common.h"
#include "arch.h"
#include "tansig_table.h"
#include "rnn.h"
#include "rnn_data.h"
#include <stdio.h>

// SIMD
#include <immintrin.h>
#include <cpuid.h>
#include <xsaveintrin.h>


/**************************************
 * GCC
 *************************************/

int is_avx2_supported() {
#if defined(__AVX2__)
   int cpuInfo[4];
   int max_function_id;
   int os_enables_XSAVE_XRSTORE = 0;
   int os_enables_avx = 0;
   int os_enables_avx2 = 0;
#ifdef __FMA__
   int os_enables_fma = 0;
#endif

   // Check for GCC or WIN32, other compilers not supported
#if !defined(__GNUC__) && !defined(_WIN32)
   return 0;
#endif

   // WIN32 must support CPUID
#if defined(_WIN32) && !defined(HAS_CPUID)
   return 0;
#endif


   // Check CPU support
   // See: https://github.com/gcc-mirror/gcc/blob/master/gcc/config/i386/cpuid.h

#if defined(__GNUC__)
   __cpuid_count(0, 0, cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3]);
#else // _WIN32
   __cpuid(cpuInfo, 0);
#endif
   max_function_id = cpuInfo[0];
   if (max_function_id < 1) {
      return 0;
   }

#if defined(__GNUC__)
   __cpuid_count(1, 0, cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3]);
#else // _WIN32
   __cpuid(cpuInfo, 1);
#endif
   os_enables_XSAVE_XRSTORE = cpuInfo[2] & 0x08000000;
   if(!os_enables_XSAVE_XRSTORE) {
      return 0;
   }

#ifdef __FMA__
   os_enables_fma = cpuInfo[2] & 0x00001000;
#endif
   os_enables_avx = cpuInfo[2] & 0x10000000;

   if (max_function_id >= 7) {
#if defined(__GNUC__)
      __cpuid_count(7, 0, cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3]);
#else // _WIN32
      __cpuid(cpuInfo, 7);
#endif
      os_enables_avx2 = cpuInfo[1] & 0x00000020;
   }


   // Check OS support
   // See: https://stackoverflow.com/a/22521619/2750093
   // AVX2 and FMA: no check available, checking AVX only is your best bet
   if(os_enables_avx) {
      unsigned long long xcrFeatureMask = _xgetbv(0); // _XCR_XFEATURE_ENABLED_MASK
      os_enables_avx = (xcrFeatureMask & 0x6) == 0x6;
   }

#ifdef __FMA__
   return os_enables_avx && os_enables_avx2 && os_enables_fma;
#else
   return os_enables_avx && os_enables_avx2;
#endif

#else
   return 0;
#endif
}


static OPUS_INLINE float tansig_approx(float x)
{
    int i;
    float y, dy;
    float sign=1;
    /* Tests are reversed to catch NaNs */
    if (!(x<8))
        return 1;
    if (!(x>-8))
        return -1;
#ifndef FIXED_POINT
    /* Another check in case of -ffast-math */
    if (celt_isnan(x))
       return 0;
#endif
    if (x<0)
    {
       x=-x;
       sign=-1;
    }
    i = (int)floor(.5f+25*x);
    x -= .04f*i;
    y = tansig_table[i];
    dy = 1-y*y;
    y = y + x*dy*(1 - y*x);
    return sign*y;
}

static OPUS_INLINE float sigmoid_approx(float x)
{
   return .5 + .5*tansig_approx(.5*x);
}

static OPUS_INLINE float relu(float x)
{
   return x < 0 ? 0 : x;
}

void compute_dense(const DenseLayer *layer, float *output, const float *input)
{
   int i, j;
   int N, M;
   int stride;
   M = layer->nb_inputs;
   N = layer->nb_neurons;
   stride = N;
   for (i = 0;i < N;i++)
   {
      /* Compute update gate. */
      float sum = layer->bias[i];
      for (j = 0; j<M;j++)
         sum += layer->input_weights[j*stride + i]*input[j];
      output[i] = WEIGHTS_SCALE*sum;
   }
   if (layer->activation == ACTIVATION_SIGMOID) {
      for (i = 0;i < N;i++)
         output[i] = sigmoid_approx(output[i]);
   } else if (layer->activation == ACTIVATION_TANH) {
      for (i = 0;i < N;i++)
         output[i] = tansig_approx(output[i]);
   } else if (layer->activation == ACTIVATION_RELU) {
      for (i = 0;i < N;i++)
         output[i] = relu(output[i]);
   } else {
     *(int*)0=0;
   }
}

#if defined(__AVX2__)
#include <immintrin.h>

// Use native FMA if available, otherwise fall back to multiply + add
#ifdef __FMA__
#define _MM256_FMADD_PS(a, b, c) _mm256_fmadd_ps(a, b, c)
#else
static OPUS_INLINE __m256 _mm256_fmadd_ps_fallback(__m256 a, __m256 b, __m256 c) {
   __m256 multiplied = _mm256_mul_ps(a, b);
   return _mm256_add_ps(c, multiplied);
}

#define _MM256_FMADD_PS(a, b, c) _mm256_fmadd_ps_fallback(a, b, c)
#endif

void compute_gru_avx2(const GRULayer *gru, float *state, const float *input)
{
   int i, j;
   int N, M;
   int stride;
   float z[MAX_NEURONS];
   float r[MAX_NEURONS];
   float h[MAX_NEURONS];
   M = gru->nb_inputs;
   N = gru->nb_neurons;
   stride = 3 * N;

   int chunk_size = 8;
   int n_remainder = N % chunk_size;
   int n_chunk_count = (N - n_remainder) / chunk_size;

   for (int i_chunk = 0; i_chunk < n_chunk_count; i_chunk++) {
      // Load i8s
      __m128i i8_z_sum = _mm_loadu_si128((__m128i*) &gru->bias[i_chunk * chunk_size]);
      __m128i i8_r_sum = _mm_loadu_si128((__m128i*) &gru->bias[N + (i_chunk * chunk_size)]);
      // Sign-extend to i32s
      __m256i i32_z_sum = _mm256_cvtepi8_epi32(i8_z_sum);
      __m256i i32_r_sum = _mm256_cvtepi8_epi32(i8_r_sum);
      // Convert to f32s
      __m256 z_sum = _mm256_cvtepi32_ps(i32_z_sum);
      __m256 r_sum = _mm256_cvtepi32_ps(i32_r_sum);

      for (j = 0; j<M; j++) {
         // Load i8s
         __m128i z_input_weights_i8 = _mm_loadu_si128((__m128i*) &gru->input_weights[j*stride + (i_chunk * chunk_size)]);
         __m128i r_input_weights_i8 = _mm_loadu_si128((__m128i*) &gru->input_weights[N + j*stride + (i_chunk * chunk_size)]);
         // Sign-extend to i32s
         __m256i z_input_weights_i32 = _mm256_cvtepi8_epi32(z_input_weights_i8);
         __m256i r_input_weights_i32 = _mm256_cvtepi8_epi32(r_input_weights_i8);
         // Convert to f32s
         __m256 z_input_weights = _mm256_cvtepi32_ps(z_input_weights_i32);
         __m256 r_input_weights = _mm256_cvtepi32_ps(r_input_weights_i32);

         __m256 input_v = _mm256_broadcast_ss(&input[j]);

         z_sum = _MM256_FMADD_PS(z_input_weights, input_v, z_sum);
         r_sum = _MM256_FMADD_PS(r_input_weights, input_v, r_sum);
      }
      for (j = 0; j<N; j++) {
         // Load i8s
         __m128i z_recurrent_weights_i8 = _mm_loadu_si128((__m128i*) &gru->recurrent_weights[j*stride + (i_chunk * chunk_size)]);
         __m128i r_recurrent_weights_i8 = _mm_loadu_si128((__m128i*) &gru->recurrent_weights[N + j*stride + (i_chunk * chunk_size)]);
         // Sign-extend to i32s
         __m256i z_recurrent_weights_i32 = _mm256_cvtepi8_epi32(z_recurrent_weights_i8);
         __m256i r_recurrent_weights_i32 = _mm256_cvtepi8_epi32(r_recurrent_weights_i8);
         // Convert to f32s
         __m256 z_recurrent_weights = _mm256_cvtepi32_ps(z_recurrent_weights_i32);
         __m256 r_recurrent_weights = _mm256_cvtepi32_ps(r_recurrent_weights_i32);

         __m256 state_v = _mm256_broadcast_ss(&state[j]);

         z_sum = _MM256_FMADD_PS(z_recurrent_weights, state_v, z_sum);
         r_sum = _MM256_FMADD_PS(r_recurrent_weights, state_v, r_sum);
      }

      // Store sums
      _mm256_storeu_ps(&z[i_chunk * chunk_size], z_sum);
      _mm256_storeu_ps(&r[i_chunk * chunk_size], r_sum);
   }
   // Remainders
   for (int i = n_chunk_count * chunk_size; i < N; i++) {
      float z_sum = gru->bias[i];
      float r_sum = gru->bias[N + i];

      for (j = 0; j<M;j++) {
         /* Compute update gate. */
         z_sum += gru->input_weights[j*stride + i]*input[j];
         /* Compute reset gate. */
         r_sum += gru->input_weights[N + j*stride + i]*input[j];
      }
      for (j = 0; j<N;j++) {
         /* Compute update gate. */
         z_sum += gru->recurrent_weights[j*stride + i]*state[j];
         /* Compute reset gate. */
         r_sum += gru->recurrent_weights[N + j*stride + i]*state[j];
      }

      z[i] = z_sum;
      r[i] = r_sum;
   }
   // Apply sigmoid to sums
   for (i = 0; i < N; i++) {
      z[i] = sigmoid_approx(WEIGHTS_SCALE * z[i]);
      r[i] = sigmoid_approx(WEIGHTS_SCALE * r[i]);
   }

   /* Compute output. */
   for (int i_chunk = 0; i_chunk < n_chunk_count; i_chunk++) {
      // Load i8s
      __m128i i8_sum = _mm_loadu_si128((__m128i*) &gru->bias[2*N + (i_chunk * chunk_size)]);
      // Sign-extend to i32s
      __m256i i32_sum = _mm256_cvtepi8_epi32(i8_sum);
      // Convert to f32s
      __m256 sum = _mm256_cvtepi32_ps(i32_sum);

      for (j = 0; j < M; j++) {
         // Load i8s
         __m128i input_weights_i8 = _mm_loadu_si128((__m128i*) &gru->input_weights[2*N + j*stride + (i_chunk * chunk_size)]);
         // Sign-extend to i32s
         __m256i input_weights_i32 = _mm256_cvtepi8_epi32(input_weights_i8);
         // Convert to f32s
         __m256 input_weights = _mm256_cvtepi32_ps(input_weights_i32);

         __m256 input_v = _mm256_broadcast_ss(&input[j]);

         sum = _MM256_FMADD_PS(input_weights, input_v, sum) ;
      }

      for (j = 0; j < N; j++) {
         // Load i8s
         __m128i recurrent_weights_i8 = _mm_loadu_si128((__m128i*) &gru->recurrent_weights[2*N + j*stride + (i_chunk * chunk_size)]);
         // Sign-extend to i32s
         __m256i recurrent_weights_i32 = _mm256_cvtepi8_epi32(recurrent_weights_i8);
         // Convert to f32s
         __m256 recurrent_weights = _mm256_cvtepi32_ps(recurrent_weights_i32);

         float state_times_r = state[j] * r[j];
         __m256 state_times_r_v = _mm256_set1_ps(state_times_r);

         sum = _MM256_FMADD_PS(recurrent_weights, state_times_r_v, sum);
      }

      // Store sums
      _mm256_storeu_ps(&h[i_chunk * chunk_size], sum);
   }
   // Remainders
   for (int i = n_chunk_count * chunk_size; i < N; i++) {
      float sum = gru->bias[2*N + i];
      for (j = 0; j < M; j++)
         sum += gru->input_weights[2*N + j*stride + i] * input[j];
      for (j = 0; j < N; j++)
         sum += gru->recurrent_weights[2*N + j*stride + i] * state[j] * r[j];

      h[i] = sum;
   }

   for (i = 0; i < N; i++) {
      float sum = h[i];

      if (gru->activation == ACTIVATION_SIGMOID) sum = sigmoid_approx(WEIGHTS_SCALE*sum);
      else if (gru->activation == ACTIVATION_TANH) sum = tansig_approx(WEIGHTS_SCALE*sum);
      else if (gru->activation == ACTIVATION_RELU) sum = relu(WEIGHTS_SCALE*sum);
      else *(int*)0=0;
      state[i] = z[i]*state[i] + (1-z[i])*sum;
   }
}
#endif

void compute_gru(const GRULayer *gru, float *state, const float *input)
{
   int i, j;
   int N, M;
   int stride;
   float z[MAX_NEURONS];
   float r[MAX_NEURONS];
   float h[MAX_NEURONS];
   M = gru->nb_inputs;
   N = gru->nb_neurons;
   stride = 3*N;
   for (i = 0; i < N; i++)
   {
      float z_sum = gru->bias[i];
      float r_sum = gru->bias[N + i];

      for (j = 0; j<M;j++) {
         /* Compute update gate. */
         z_sum += gru->input_weights[j*stride + i]*input[j];
         /* Compute reset gate. */
         r_sum += gru->input_weights[N + j*stride + i]*input[j];
      }
      for (j = 0; j < N; j++) {
         /* Compute update gate. */
         z_sum += gru->recurrent_weights[j*stride + i]*state[j];
         /* Compute reset gate. */
         r_sum += gru->recurrent_weights[N + j*stride + i]*state[j];
      }

      z[i] = sigmoid_approx(WEIGHTS_SCALE*z_sum);
      r[i] = sigmoid_approx(WEIGHTS_SCALE*r_sum);
   }

   /* Compute output. */
   for (i = 0; i < N; i++) {
      float sum = gru->bias[2*N + i];
      for (j = 0; j<M; j++)
         sum += gru->input_weights[2*N + j*stride + i]*input[j];
      for (j = 0; j<N; j++)
         sum += gru->recurrent_weights[2*N + j*stride + i]*state[j]*r[j];
      if (gru->activation == ACTIVATION_SIGMOID) sum = sigmoid_approx(WEIGHTS_SCALE*sum);
      else if (gru->activation == ACTIVATION_TANH) sum = tansig_approx(WEIGHTS_SCALE*sum);
      else if (gru->activation == ACTIVATION_RELU) sum = relu(WEIGHTS_SCALE*sum);
      else *(int*)0=0;
      h[i] = z[i] * state[i] + (1 - z[i]) * sum;
   }
   memcpy((void*) &state, (void*) &h, N * sizeof(float));
}

#define INPUT_SIZE 42

void compute_rnn(RNNState *rnn, float *gains, float *vad, const float *input) {
  int i;
  float dense_out[MAX_NEURONS];
  float noise_input[MAX_NEURONS*3];
  float denoise_input[MAX_NEURONS*3];
  compute_dense(rnn->model->input_dense, dense_out, input);
  rnn->compute_gru_fct(rnn->model->vad_gru, rnn->vad_gru_state, dense_out);
  compute_dense(rnn->model->vad_output, vad, rnn->vad_gru_state);
  for (i = 0;i<rnn->model->input_dense_size;i++) noise_input[i] = dense_out[i];
  for (i = 0;i<rnn->model->vad_gru_size;i++) noise_input[i+rnn->model->input_dense_size] = rnn->vad_gru_state[i];
  for (i = 0;i<INPUT_SIZE;i++) noise_input[i+rnn->model->input_dense_size+rnn->model->vad_gru_size] = input[i];
  rnn->compute_gru_fct(rnn->model->noise_gru, rnn->noise_gru_state, noise_input);

  for (i = 0;i<rnn->model->vad_gru_size;i++) denoise_input[i] = rnn->vad_gru_state[i];
  for (i = 0;i<rnn->model->noise_gru_size;i++) denoise_input[i+rnn->model->vad_gru_size] = rnn->noise_gru_state[i];
  for (i = 0;i<INPUT_SIZE;i++) denoise_input[i+rnn->model->vad_gru_size+rnn->model->noise_gru_size] = input[i];
  rnn->compute_gru_fct(rnn->model->denoise_gru, rnn->denoise_gru_state, denoise_input);
  compute_dense(rnn->model->denoise_output, gains, rnn->denoise_gru_state);
}
