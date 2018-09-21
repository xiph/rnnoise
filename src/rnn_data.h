#ifndef RNN_DATA_H
#define RNN_DATA_H

#include "rnn.h"

struct RNNModel {
  int input_dense_size;
  DenseLayer *input_dense;

  int vad_gru_size;
  GRULayer *vad_gru;

  int noise_gru_size;
  GRULayer *noise_gru;

  int denoise_gru_size;
  GRULayer *denoise_gru;

  int denoise_output_size;
  DenseLayer *denoise_output;

  int vad_output_size;
  DenseLayer *vad_output;
};

struct RNNState {
  const RNNModel *model;
  float *vad_gru_state;
  float *noise_gru_state;
  float *denoise_gru_state;
};


#endif
