#!/usr/bin/python 
# requires pip install tensorflow==1.5
# requires pip install keras==2.3.0
# for latest TF reset_after=False for GRU layers

from __future__ import print_function

import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import SimpleRNN
from keras.layers import Dropout
from keras.layers import concatenate
from keras import losses
from keras import regularizers
from keras.constraints import min_max_norm
import h5py

from keras.constraints import Constraint
from keras import backend as K
import numpy as np

#import tensorflow as tf
#from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.42
#set_session(tf.Session(config=config))


def my_crossentropy(y_true, y_pred):
    return K.mean(2*K.abs(y_true-0.5) * K.binary_crossentropy(y_pred, y_true), axis=-1)

def mymask(y_true):
    return K.minimum(y_true+1., 1.)

def msse(y_true, y_pred):
    return K.mean(mymask(y_true) * K.square(K.sqrt(y_pred) - K.sqrt(y_true)), axis=-1)

def mycost(y_true, y_pred):
    return K.mean(mymask(y_true) * (10*K.square(K.square(K.sqrt(y_pred) - K.sqrt(y_true))) + K.square(K.sqrt(y_pred) - K.sqrt(y_true)) + 0.01*K.binary_crossentropy(y_pred, y_true)), axis=-1)

def my_accuracy(y_true, y_pred):
    return K.mean(2*K.abs(y_true-0.5) * K.equal(y_true, K.round(y_pred)), axis=-1)

class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range
    '''
    def __init__(self, c=2):
        self.c = c

    def __call__(self, p):
        return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__,
            'c': self.c}

reg = 0.000001
constraint = WeightClip(0.499)

print('Build model...')
# main_input = Input(shape=(100, 42), name='main_input')

# tmp = Dense(24, activation='tanh', name='input_dense')(main_input)
# vad_gru = GRU(24, unroll=True, activation='relu', recurrent_activation='sigmoid', return_sequences=True, name='vad_gru', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg))(tmp)
# vad_output = Dense(1, activation='sigmoid', name='vad_output')(vad_gru)

# noise_input = keras.layers.concatenate([tmp, vad_gru, main_input])
# noise_gru = GRU(48, unroll=True, activation='relu', recurrent_activation='sigmoid', return_sequences=True, name='noise_gru', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg))(noise_input)

# denoise_input = keras.layers.concatenate([vad_gru, noise_gru, main_input])
# denoise_gru = GRU(96, unroll=True, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, name='denoise_gru', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg))(denoise_input)
# denoise_output = Dense(22, activation='sigmoid', name='denoise_output')(denoise_gru)

main_input = Input(shape=(None, 42), name='main_input')

tmp = Dense(24, activation='tanh', name='input_dense', kernel_constraint=constraint, bias_constraint=constraint)(main_input)
vad_gru = GRU(24, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, name='vad_gru', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(tmp)
vad_output = Dense(1, activation='sigmoid', name='vad_output', kernel_constraint=constraint, bias_constraint=constraint)(vad_gru)

noise_input = keras.layers.concatenate([tmp, vad_gru, main_input])
noise_gru = GRU(48, activation='relu', recurrent_activation='sigmoid', return_sequences=True, name='noise_gru', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(noise_input)

denoise_input = keras.layers.concatenate([vad_gru, noise_gru, main_input])
denoise_gru = GRU(96, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, name='denoise_gru', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(denoise_input)
denoise_output = Dense(22, activation='sigmoid', name='denoise_output', kernel_constraint=constraint, bias_constraint=constraint)(denoise_gru)

model = Model(inputs=main_input, outputs=[denoise_output, vad_output])

#model.compile(loss=[mycost, my_crossentropy],
#              metrics=[msse],
#              optimizer='adam', loss_weights=[10, 0.5])


# Load weights from C array
import rnn_data

# Layer shape
# | Num |    Layer    |   Kernel  | Recurrent Kernel |  Bias  | 
# |  1  |    dense    |  (42, 24) | ---------------- | (24,)  |
# |  2  |    vad gru  |  (24, 72) |     (24, 72)     | (72,)  |
# |  4  |  noise gru  | (90, 144) |     (48, 144)    | (144,) |
# |  6  | denoise gru |(114, 288) |     (96, 288)    | (288,) |
# |  7  | denoise out |  (96, 22) | ---------------- | (22,)  |
# |  8  |  vad output |  (24, 1)  | ---------------- |  (1,)  |

def float_weights(weights, shape):
    weights = np.array(weights)
    float_weights = np.true_divide(weights, 256)
    reshaped_weights = float_weights.reshape(shape)
    return reshaped_weights

def float_bias(bias):
    bias = np.array(bias)
    float_bias = np.true_divide(bias, 256)
    return float_bias

dense_layer_map = {
    1 : ["input_dense",  (42, 24)],
    7 : ["denoise_output", (96, 22)],
    8 : ["vad_output", (24, 1)]
}

gru_layer_map = {
    2 : ["vad_gru", (24, 72), (24, 72)],
    4 : ["noise_gru", (90, 144), (48, 144)],
    6 : ["denoise_gru", (114, 288), (96, 288)]
}

for index in range(9):
    if index in dense_layer_map:
        # Load weights, bias for dense layers
        layer_name = dense_layer_map[index][0]

        weights = eval("rnn_data.{}_weights".format(layer_name))
        weights_shape = dense_layer_map[index][1]

        bias = eval("rnn_data.{}_bias".format(layer_name))

        model.layers[index].set_weights([float_weights(weights, weights_shape), float_bias(bias)])
    elif index in gru_layer_map:
        # Load weights, recurrent weights, bias for gru layers
        layer_name = gru_layer_map[index][0]

        weights = eval("rnn_data.{}_weights".format(layer_name))
        weights_shape = gru_layer_map[index][1]

        recurrent_weights = eval("rnn_data.{}_recurrent_weights".format(layer_name))
        recurrent_weights_shape = gru_layer_map[index][2]

        bias = eval("rnn_data.{}_bias".format(layer_name))

        model.layers[index].set_weights([float_weights(weights, weights_shape), \
                                         float_weights(recurrent_weights, recurrent_weights_shape), \
                                         float_bias(bias)])
    else:
        print("{} LAYER DOES NOT NEED TO LOAD WEIGHTS...".format(index))

print("WEIGHTS LOADED!")

model.save("rnnoise.h5")

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
sess = K.get_session()
constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), ['denoise_output/Sigmoid','vad_output/Sigmoid'])
graph_io.write_graph(constant_graph, './', 'rnnoise.pb', as_text=False)
