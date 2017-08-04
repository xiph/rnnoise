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
#include <stdio.h>

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
   for (i=0;i<N;i++)
   {
      /* Compute update gate. */
      float sum = layer->bias[i];
      for (j=0;j<M;j++)
         sum += layer->input_weights[j*stride + i]*input[j];
      output[i] = WEIGHTS_SCALE*sum;
   }
   if (layer->activation == activation_sigmoid) {
      for (i=0;i<N;i++)
         output[i] = sigmoid_approx(output[i]);
   } else if (layer->activation == activation_tanh) {
      for (i=0;i<N;i++)
         output[i] = tansig_approx(output[i]);
   } else if (layer->activation == activation_relu) {
      for (i=0;i<N;i++)
         output[i] = relu(output[i]);
   } else {
     *(int*)0=0;
   }
}

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
   for (i=0;i<N;i++)
   {
      /* Compute update gate. */
      float sum = gru->bias[i];
      for (j=0;j<M;j++)
         sum += gru->input_weights[j*stride + i]*input[j];
      for (j=0;j<N;j++)
         sum += gru->recurrent_weights[j*stride + i]*state[j];
      z[i] = sigmoid_approx(WEIGHTS_SCALE*sum);
   }
   for (i=0;i<N;i++)
   {
      /* Compute reset gate. */
      float sum = gru->bias[N + i];
      for (j=0;j<M;j++)
         sum += gru->input_weights[N + j*stride + i]*input[j];
      for (j=0;j<N;j++)
         sum += gru->recurrent_weights[N + j*stride + i]*state[j];
      r[i] = sigmoid_approx(WEIGHTS_SCALE*sum);
   }
   for (i=0;i<N;i++)
   {
      /* Compute output. */
      float sum = gru->bias[2*N + i];
      for (j=0;j<M;j++)
         sum += gru->input_weights[2*N + j*stride + i]*input[j];
      for (j=0;j<N;j++)
         sum += gru->recurrent_weights[2*N + j*stride + i]*state[j]*r[j];
      if (gru->activation == activation_sigmoid) sum = sigmoid_approx(WEIGHTS_SCALE*sum);
      else if (gru->activation == activation_tanh) sum = tansig_approx(WEIGHTS_SCALE*sum);
      else if (gru->activation == activation_relu) sum = relu(WEIGHTS_SCALE*sum);
      else *(int*)0=0;
      h[i] = z[i]*state[i] + (1-z[i])*sum;
   }
   for (i=0;i<N;i++)
      state[i] = h[i];
}

#if 1
#define INPUT_SIZE 42
#define DENSE_SIZE 12
#define VAD_SIZE 12
#define NOISE_SIZE 48
#define DENOISE_SIZE 128

extern const DenseLayer input_dense;
extern const GRULayer vad_gru;
extern const GRULayer noise_gru;
extern const GRULayer denoise_gru;
extern const DenseLayer denoise_output;
extern const DenseLayer vad_output;

int main() {
  float vad_state[MAX_NEURONS] = {0};
  float vad_out[MAX_NEURONS] = {0};
  float input[INPUT_SIZE];
  float dense_out[MAX_NEURONS];
  float noise_input[MAX_NEURONS*3];
  float denoise_input[MAX_NEURONS*3];
  float noise_state[MAX_NEURONS] = {0};
  float denoise_state[MAX_NEURONS] = {0};
  float gains[22];
  while (1)
  {
    int i;
    for (i=0;i<INPUT_SIZE;i++) scanf("%f", &input[i]);
    for (i=0;i<45;i++) scanf("%f", &vad_out[0]);
    if (feof(stdin)) break;
    compute_dense(&input_dense, dense_out, input);
    compute_gru(&vad_gru, vad_state, dense_out);
    compute_dense(&vad_output, vad_out, vad_state);
#if 1
    for (i=0;i<DENSE_SIZE;i++) noise_input[i] = dense_out[i];
    for (i=0;i<VAD_SIZE;i++) noise_input[i+DENSE_SIZE] = vad_state[i];
    for (i=0;i<INPUT_SIZE;i++) noise_input[i+DENSE_SIZE+VAD_SIZE] = input[i];
    compute_gru(&noise_gru, noise_state, noise_input);

    for (i=0;i<VAD_SIZE;i++) denoise_input[i] = vad_state[i];
    for (i=0;i<NOISE_SIZE;i++) denoise_input[i+VAD_SIZE] = noise_state[i];
    for (i=0;i<INPUT_SIZE;i++) denoise_input[i+VAD_SIZE+NOISE_SIZE] = input[i];
    compute_gru(&denoise_gru, denoise_state, denoise_input);

    compute_dense(&denoise_output, gains, denoise_state);

    for (i=0;i<22;i++) printf("%f ", gains[i]);
#endif
    printf("%f\n", vad_out[0]);
  }
}
#endif
