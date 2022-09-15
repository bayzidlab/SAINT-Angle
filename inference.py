# Parsing Arguments from the Command Line
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--verbose", dest="verbose", action="store_true", help="whether to print out detailed messages (Default: False)")
parser.add_argument("--output", dest="output_predictions", action="store_true", help="whether to output predicted angles (Default: False)")
parser.add_argument("-m", "--model", dest="model_name", default="model_1", help="model(s) to use for inference/prediction")
parser.add_argument("-d", "--dataset", dest="dataset_name", default="TEST2016", help="dataset to use for inference/prediction")

parser.set_defaults(verbose=False)
parser.set_defaults(output_predictions=False)

args = parser.parse_args()

verbose = args.verbose
output_predictions = args.output_predictions
model_name = args.model_name
dataset_name = args.dataset_name

# Import Statements
import tensorflow as tf
import keras

if verbose:
    print(f"Tensorflow version: {tf.__version__}")
    print(f"Keras version: {keras.__version__}\n")

from keras import backend, optimizers, callbacks
from keras.models import Sequential, Model, load_model
from keras.layers import Layer, add, Add, Concatenate, Multiply, Flatten, Dot, Softmax, BatchNormalization, LayerNormalization, RepeatVector, \
    Input, SpatialDropout1D, Embedding, LSTM, Dense, merge, Convolution2D, Lambda, GRU, TimeDistributed, Reshape, Permute, Convolution1D, \
    Masking, Bidirectional
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from keras.initializers import Ones, Zeros

from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.metrics import Metric
from tensorflow.python.client import device_lib

import numpy as np
import gzip
import os
import math
from itertools import combinations

if verbose:
    print(f"Local devices available:\n{'\n'.join([device.name + '(' + device.device_type +')' for device in device_lib.list_local_devices()])}\n")

# Performance Evaluation Functions Definitions

# Performance Metrics Description:-
#     mse -> Mean Squared Error (MSE)
#     mae -> Mean Absolute Error (MAE)
#     sae -> Sum of Absolute Error (SAE) = MAE * 'total residue count'

def get_total_mse_sin_cos(y_true, y_predict):
    mask = 1 - backend.cast((backend.equal(y_true[:, :, 0], -100)), 'float32')
    count = backend.sum(mask)
    
    y_truephi_sin = y_true[:, :, 0] * mask
    y_truephi_cos = y_true[:, :, 1] * mask
    y_truepsi_sin = y_true[:, :, 2] * mask
    y_truepsi_cos = y_true[:, :, 3] * mask

    y_predphi_sin = y_predict[:, :, 0] * mask
    y_predphi_cos = y_predict[:, :, 1] * mask
    y_predpsi_sin = y_predict[:, :, 2] * mask
    y_predpsi_cos = y_predict[:, :, 3] * mask

    phi_diff_sin = backend.abs(y_truephi_sin - y_predphi_sin)
    phi_diff_cos = backend.abs(y_truephi_cos - y_predphi_cos)
    psi_diff_sin = backend.abs(y_truepsi_sin - y_predpsi_sin)
    psi_diff_cos = backend.abs(y_truepsi_cos - y_predpsi_cos)

    phi_mse_sin = backend.sum(backend.square(phi_diff_sin)) / count
    phi_mse_cos = backend.sum(backend.square(phi_diff_cos)) / count
    psi_mse_sin = backend.sum(backend.square(psi_diff_sin)) / count
    psi_mse_cos = backend.sum(backend.square(psi_diff_cos)) / count

    final_mse = .25 * (phi_mse_sin + phi_mse_cos + psi_mse_sin + psi_mse_cos)
    return final_mse

def get_total_mae(y_true, y_predict):
    y_truephi_angle = (tf.atan2(y_true[:, :, 0], y_true[:, :, 1]) * 180) / np.pi
    y_truepsi_angle = (tf.atan2(y_true[:, :, 2], y_true[:, :, 3]) * 180) / np.pi

    y_predphi_angle = (tf.atan2(y_predict[:, :, 0], y_predict[:, :, 1]) * 180) / np.pi
    y_predpsi_angle = (tf.atan2(y_predict[:, :, 2], y_predict[:, :, 3]) * 180) / np.pi

    mask = 1 - backend.cast((backend.equal(y_true[:, :, 0], -100)), 'float32')
    count = backend.sum(mask)

    phi_diff = backend.abs(y_truephi_angle - y_predphi_angle)
    psi_diff = backend.abs(y_truepsi_angle - y_predpsi_angle)

    phi_exp_diff = Lambda(lambda x: 360 - x)(phi_diff)
    psi_exp_diff = Lambda(lambda x: 360 - x)(psi_diff)

    phi_mask1 = backend.cast((backend.greater(phi_diff[:, :], 180)), 'float32')
    phi_mask2 = 1 - phi_mask1
    psi_mask1 = backend.cast((backend.greater(psi_diff[:, :], 180)), 'float32')
    psi_mask2 = 1 - psi_mask1

    phi_error = phi_diff * phi_mask2 + phi_exp_diff * phi_mask1
    psi_error = psi_diff * psi_mask2 + psi_exp_diff * psi_mask1

    phi_mae = backend.sum(phi_error * mask) / count
    psi_mae = backend.sum(psi_error * mask) / count

    final_mae = .5 * (phi_mae + psi_mae)
    return final_mae

def get_phi_mae(y_true, y_predict):
    y_truephi_angle = (tf.atan2(y_true[:, :, 0], y_true[:, :, 1]) * 180) / np.pi
    y_predphi_angle = (tf.atan2(y_predict[:, :, 0], y_predict[:, :, 1]) * 180) / np.pi

    mask = 1 - backend.cast((backend.equal(y_true[:, :, 0], -100)), 'float32')
    count = backend.sum(mask)

    phi_diff = backend.abs(y_truephi_angle - y_predphi_angle)
    phi_exp_diff = Lambda(lambda x: 360 - x)(phi_diff)

    phi_mask1 = backend.cast((backend.greater(phi_diff[:, :], 180)), 'float32')
    phi_mask2 = 1 - phi_mask1

    phi_error = phi_diff * phi_mask2 + phi_exp_diff * phi_mask1
    phi_mae = backend.sum(phi_error * mask) / count
    return phi_mae

def get_psi_mae(y_true, y_predict):
    y_truepsi_angle = (tf.atan2(y_true[:, :, 2], y_true[:, :, 3]) * 180) / np.pi
    y_predpsi_angle = (tf.atan2(y_predict[:, :, 2], y_predict[:, :, 3]) * 180) / np.pi

    mask = 1 - backend.cast((backend.equal(y_true[:, :, 0], -100)), 'float32')
    count = backend.sum(mask)

    psi_diff = backend.abs(y_truepsi_angle - y_predpsi_angle)
    psi_exp_diff = Lambda(lambda x: 360 - x)(psi_diff)

    psi_mask1 = backend.cast((backend.greater(psi_diff[:, :], 180)), 'float32')
    psi_mask2 = 1 - psi_mask1

    psi_error = psi_diff * psi_mask2 + psi_exp_diff * psi_mask1
    psi_mae = backend.sum(psi_error * mask) / count
    return psi_mae

def get_total_sae(y_true, y_predict):
    y_truephi_angle = (tf.atan2(y_true[:, :, 0], y_true[:, :, 1]) * 180) / np.pi
    y_truepsi_angle = (tf.atan2(y_true[:, :, 2], y_true[:, :, 3]) * 180) / np.pi

    y_predphi_angle = (tf.atan2(y_predict[:, :, 0], y_predict[:, :, 1]) * 180) / np.pi
    y_predpsi_angle = (tf.atan2(y_predict[:, :, 2], y_predict[:, :, 3]) * 180) / np.pi

    mask = 1 - backend.cast((backend.equal(y_true[:, :, 0], -100)), 'float32')
    count = backend.sum(mask)

    phi_diff = backend.abs(y_truephi_angle - y_predphi_angle)
    psi_diff = backend.abs(y_truepsi_angle - y_predpsi_angle)

    phi_exp_diff = Lambda(lambda x: 360 - x)(phi_diff)
    psi_exp_diff = Lambda(lambda x: 360 - x)(psi_diff)

    phi_mask1 = backend.cast((backend.greater(phi_diff[:, :], 180)), 'float32')
    phi_mask2 = 1 - phi_mask1
    psi_mask1 = backend.cast((backend.greater(psi_diff[:, :], 180)), 'float32')
    psi_mask2 = 1 - psi_mask1

    phi_error = phi_diff * phi_mask2 + phi_exp_diff * phi_mask1
    psi_error = psi_diff * psi_mask2 + psi_exp_diff * psi_mask1

    phi_sae = backend.sum(phi_error * mask)
    psi_sae = backend.sum(psi_error * mask)

    final_sae = .5 * (phi_sae + psi_sae)
    return final_sae

def get_phi_sae(y_true, y_predict):
    y_truephi_angle = (tf.atan2(y_true[:, :, 0], y_true[:, :, 1]) * 180) / np.pi
    y_predphi_angle = (tf.atan2(y_predict[:, :, 0], y_predict[:, :, 1]) * 180) / np.pi

    mask = 1 - backend.cast((backend.equal(y_true[:, :, 0], -100)), 'float32')
    count = backend.sum(mask)

    phi_diff = backend.abs(y_truephi_angle - y_predphi_angle)
    phi_exp_diff = Lambda(lambda x: 360 - x)(phi_diff)

    phi_mask1 = backend.cast((backend.greater(phi_diff[:, :], 180)), 'float32')
    phi_mask2 = 1 - phi_mask1

    phi_error = phi_diff * phi_mask2 + phi_exp_diff * phi_mask1
    phi_sae = backend.sum(phi_error * mask)
    return phi_sae

def get_psi_sae(y_true, y_predict):
    y_truepsi_angle = (tf.atan2(y_true[:, :, 2], y_true[:, :, 3]) * 180) / np.pi
    y_predpsi_angle = (tf.atan2(y_predict[:, :, 2], y_predict[:, :, 3]) * 180) / np.pi

    mask = 1 - backend.cast((backend.equal(y_true[:, :, 0], -100)), 'float32')
    count = backend.sum(mask)

    psi_diff = backend.abs(y_truepsi_angle - y_predpsi_angle)
    psi_exp_diff = Lambda(lambda x: 360 - x)(psi_diff)

    psi_mask1 = backend.cast((backend.greater(psi_diff[:, :], 180)), 'float32')
    psi_mask2 = 1 - psi_mask1

    psi_error = psi_diff * psi_mask2 + psi_exp_diff * psi_mask1
    psi_sae = backend.sum(psi_error * mask)
    return psi_sae

# Performance Evaluation Functions Definitions (in NumPy for cross-checking purpose)
def get_total_mae_np(y_true, y_predict, ground_truth_phi, ground_truth_psi, padding_val=-500):
    y_predphi_sin = y_predict[:, :, 0]
    y_predphi_cos = y_predict[:, :, 1]
    y_predpsi_sin = y_predict[:, :, 2]
    y_predpsi_cos = y_predict[:, :, 3]

    y_truephi_sin = y_true[:, :, 0]
    y_truephi_cos = y_true[:, :, 1]
    y_truepsi_sin = y_true[:, :, 2]
    y_truepsi_cos = y_true[:, :, 3]

    mask_phi = 1 - np.equal(ground_truth_phi[:, :], padding_val).astype(np.float32)
    mask_psi = 1 - np.equal(ground_truth_psi[:, :], padding_val).astype(np.float32)

    count_phi = mask_phi.sum() + 1
    count_psi = mask_psi.sum() + 1
    
    assert count_phi == count_psi, "Phi count and Psi count mismatched"

    y_predphi_angle = (np.arctan2(y_predphi_sin, y_predphi_cos) * 180) / np.pi
    y_predpsi_angle = (np.arctan2(y_predpsi_sin, y_predpsi_cos) * 180) / np.pi

    y_truephi_angle = (np.arctan2(y_truephi_sin, y_truephi_cos) * 180) / np.pi
    y_truepsi_angle = (np.arctan2(y_truepsi_sin, y_truepsi_cos) * 180) / np.pi

    phi_diff = np.abs(y_truephi_angle - y_predphi_angle)
    psi_diff = np.abs(y_truepsi_angle - y_predpsi_angle)

    rotate_difference = lambda x: 360 - x
    phi_exp_diff = rotate_difference(phi_diff)
    psi_exp_diff = rotate_difference(psi_diff)

    phi_mask1 = np.greater(phi_diff[:, :], 180).astype(np.float32)
    phi_mask2 = 1 - phi_mask1
    psi_mask1 = np.greater(psi_diff[:, :], 180).astype(np.float32)
    psi_mask2 = 1 - psi_mask1

    phi_error = phi_diff * phi_mask2 + phi_exp_diff * phi_mask1
    psi_error = psi_diff * psi_mask2 + psi_exp_diff * psi_mask1

    phi_mae = np.sum(np.multiply(phi_error, mask_phi)) / count_phi
    psi_mae = np.sum(np.multiply(psi_error, mask_psi)) / count_psi

    final_mae = .5 * (phi_mae + psi_mae)
    return final_mae

def get_phi_mae_np(y_true, y_predict, ground_truth, padding_val=-500):
    y_predphi_sin = y_predict[:, :, 0]
    y_predphi_cos = y_predict[:, :, 1]

    y_truephi_sin = y_true[:, :, 0]
    y_truephi_cos = y_true[:, :, 1]

    mask = 1 - np.equal(ground_truth[:, :], padding_val).astype(np.float32)
    count = mask.sum() + 1

    y_predphi_angle = (np.arctan2(y_predphi_sin, y_predphi_cos) * 180) / np.pi
    y_truephi_angle = (np.arctan2(y_truephi_sin, y_truephi_cos) * 180) / np.pi

    phi_diff = np.abs(y_truephi_angle - y_predphi_angle)

    rotate_difference = lambda x: 360 - x
    phi_exp_diff = rotate_difference(phi_diff)

    phi_mask1 = np.greater(phi_diff[:, :], 180).astype(np.float32)
    phi_mask2 = 1 - phi_mask1

    phi_error = phi_diff * phi_mask2 + phi_exp_diff * phi_mask1
    phi_mae = np.sum(np.multiply(phi_error, mask)) / count
    return phi_mae

def get_psi_mae_np(y_true, y_predict, ground_truth, padding_val=-500):
    y_predpsi_sin = y_predict[:, :, 2]
    y_predpsi_cos = y_predict[:, :, 3]

    y_truepsi_sin = y_true[:, :, 2]
    y_truepsi_cos = y_true[:, :, 3]

    mask = 1 - np.equal(ground_truth[:, :], padding_val).astype(np.float32)
    count = mask.sum() + 1

    y_predpsi_angle = (np.arctan2(y_predpsi_sin, y_predpsi_cos) * 180) / np.pi
    y_truepsi_angle = (np.arctan2(y_truepsi_sin, y_truepsi_cos) * 180) / np.pi

    psi_diff = np.abs(y_truepsi_angle - y_predpsi_angle)

    rotate_difference = lambda x: 360 - x
    psi_exp_diff = rotate_difference(psi_diff)

    psi_mask1 = np.greater(psi_diff[:, :], 180).astype(np.float32)
    psi_mask2 = 1 - psi_mask1

    psi_error = psi_diff * psi_mask2 + psi_exp_diff * psi_mask1
    psi_mae = np.sum(np.multiply(psi_error, mask)) / count
    return psi_mae

# Helping Functions & Classes Definitions
def load_gz(path):
    return np.load(gzip.open(path, 'rb'))

def get_shape_list(x):
    ## reference: https://github.com/Separius/BERT-keras/blob/master/transformer/funcs.py
    if backend.backend() != 'theano':
        tmp = backend.int_shape(x)
    else:
        tmp = x.shape

    tmp = list(tmp)
    tmp[0] = -1
    return tmp

def load_labels(file_path, dataset_name, force_padding=True, padding_val=-500, ignore_first_and_last=True, verbose=False):
    if verbose:
        print("\nload_labels function in action...\n\nLoading protein lengths...")

    with open(f'{file_path}/{dataset_name}_len.txt', 'r') as length_file:
        data_len = [int(length) for length in length_file.read().split('\n') if length != '']

    if verbose:
        print(f"Lengths loaded for {len(data_len)} proteins.\n")
        print("Loading protein labels...")

    data_phi = np.loadtxt(f'{file_path}/{dataset_name}_phi.txt')
    data_psi = np.loadtxt(f'{file_path}/{dataset_name}_psi.txt')

    if verbose:
        print(f"Phi file shape: {data_phi.shape}\nPsi file shape: {data_psi.shape}")

    assert len(data_len) == data_phi.shape[0] == data_psi.shape[0], "Protein counts mismatched"

    data_label = np.zeros((len(data_len), 700, 4))
    data_label[:, :, 0] = np.sin(data_phi * (np.pi / 180))
    data_label[:, :, 1] = np.cos(data_phi * (np.pi / 180))
    data_label[:, :, 2] = np.sin(data_psi * (np.pi / 180))
    data_label[:, :, 3] = np.cos(data_psi * (np.pi / 180))

    if ignore_first_and_last:
        # force-padding first index of phi
        data_phi[:, 0] = padding_val
        data_label[:, 0, 0] = padding_val
        data_label[:, 0, 1] = padding_val

        # force-padding last index of psi
        for index, protein_length in enumerate(data_len):
            data_psi[index, protein_length - 1] = padding_val
            data_label[index, protein_length - 1, 2] = padding_val
            data_label[index, protein_length - 1, 3] = padding_val

    if force_padding:
        for index in range(len(data_len)):
            data_phi[index, data_len[index]:] = padding_val
            data_psi[index, data_len[index]:] = padding_val
            data_label[index, data_len[index]:, :] = padding_val

    if verbose:
        print(f"Labels shape: {data_label.shape}")

    return data_phi, data_psi, data_label

def get_scores(y_predict):
    total_score = get_total_mae_np(test_label, y_predict, test_phi, test_psi, -500)
    phi_score = get_phi_mae_np(test_label, y_predict, test_phi, -500)
    psi_score = get_psi_mae_np(test_label, y_predict, test_psi, -500)
    avg_score = 0.5 * (phi_score + psi_score)
    return phi_score, psi_score

class MyLayer(Layer):
    def __init__(self, **kwargs):
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self._x = backend.variable(0.2)
        self._x._trainable = True
        self._trainable_weights = [self._x]
        super(MyLayer, self).build(input_shape)

    def call(self, x):
        A, B = x
        result = add([self._x * A, (1 - self._x) * B])
        return result

    def compute_output_shape(self, input_shape):
        return input_shape[0]

class DataLoader(tf.keras.utils.Sequence):
    def __init__(
            self,
            base_path,
            protbert_path,
            dataset_name,
            batch_size,
            length_file_path,
            phi_file_path,
            psi_file_path,
            num_features,
            use_protbert,
            force_padding=True,
            padding_val=-500,
            shuffle=False,
            ignore_first_and_last=True,
            verbose=False
    ):
        self.base_path = base_path
        self.protbert_path = protbert_path
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_features = num_features
        self.use_protbert = use_protbert

        if verbose:
            print("\nDataLoader in action...\n\nLoading protein lengths...")

        with open(length_file_path, 'r') as length_file:
            self.data_len = [int(length) for length in length_file.read().split('\n') if length != '']

        if verbose:
            print(f"Lengths loaded for {len(self.data_len)} proteins.\n")
            print("Loading protein labels...")

        self.data_phi = np.loadtxt(phi_file_path)
        self.data_psi = np.loadtxt(psi_file_path)

        if verbose:
            print(f"Phi file shape: {self.data_phi.shape}\nPsi file shape: {self.data_psi.shape}")

        assert len(self.data_len) == self.data_phi.shape[0] == self.data_psi.shape[0], "Protein counts mismatched"

        self.data_label = np.zeros((len(self.data_len), 700, 4))
        self.data_label[:, :, 0] = np.sin(self.data_phi * (np.pi / 180))
        self.data_label[:, :, 1] = np.cos(self.data_phi * (np.pi / 180))
        self.data_label[:, :, 2] = np.sin(self.data_psi * (np.pi / 180))
        self.data_label[:, :, 3] = np.cos(self.data_psi * (np.pi / 180))

        if ignore_first_and_last:
            # force-padding first index of phi
            self.data_phi[:, 0] = padding_val
            self.data_label[:, 0, 0] = padding_val
            self.data_label[:, 0, 1] = padding_val

            # force-padding last index of psi
            for index, protein_length in enumerate(self.data_len):
                self.data_psi[index, protein_length - 1] = padding_val
                self.data_label[index, protein_length - 1, 2] = padding_val
                self.data_label[index, protein_length - 1, 3] = padding_val

        if force_padding:
            for index in range(len(self.data_len)):
                self.data_phi[index, self.data_len[index]:] = padding_val
                self.data_psi[index, self.data_len[index]:] = padding_val
                self.data_label[index, self.data_len[index]:, :] = padding_val

        if verbose:
            print(f"Labels shape: {self.data_label.shape}\n")
            print("Preparing masks...")

        self.data_att_mask = np.zeros((len(self.data_len), 700))

        for index in range(len(self.data_len)):
            self.data_att_mask[index, self.data_len[index]:] = -1000000

        if verbose:
            print(f"Attention mask shape: {self.data_att_mask.shape}")

        self.data_position_ids = np.repeat([np.array(range(700))], len(self.data_len), axis=0)
        self.weight_mask = np.zeros((len(self.data_len), 700))

        for index in range(len(self.data_len)):
            self.weight_mask[index, :self.data_len[index]] = 1.0

        if verbose:
            print(f"Weight mask shape: {self.weight_mask.shape}")

        self.on_epoch_end(shuffle)

    def on_epoch_end(self, shuffle):
        # updating indices after each epoch
        self.indices = np.arange(len(self.data_len))

        if shuffle:
            np.random.shuffle(self.indices)

    def __load_length(self, index):
        return self.data_len[index]

    def __load_labels(self, index):
        return self.data_phi[index, :], self.data_psi[index, :], self.data_label[index, :]

    def __load_training_masks(self, index):
        return self.data_att_mask[index, :], self.data_position_ids[index, :], self.weight_mask[index, :]

    def __len__(self):
        # returning total number of batches
        return int(math.ceil(len(self.data_len) / self.batch_size))

    def __getitem__(self, batch_index):
        # generating indices of a particualr batch
        batch_indices = self.indices[batch_index * self.batch_size: (batch_index + 1) * self.batch_size]

        features = np.zeros((len(batch_indices), 700, self.num_features))
        att_mask = np.zeros((len(batch_indices), 700))
        position_ids = np.zeros((len(batch_indices), 700))
        weight_mask = np.zeros((len(batch_indices), 700))
        label = np.zeros((len(batch_indices), 700, 4))

        for batch_idx, sample_idx in enumerate(batch_indices):
            base_features = np.load(f'{self.base_path}/{self.dataset_name}_sample_{sample_idx}')

            if self.num_features == 57 or self.num_features == 1081:
                base_features = base_features[:, :57]

            if self.use_protbert:
                protbert_features = np.load(f'{self.protbert_path}/{self.dataset_name}_sample_{sample_idx}')

            sample_phi, sample_psi, label[batch_idx] = self.__load_labels(sample_idx)
            att_mask[batch_idx], position_ids[batch_idx], weight_mask[batch_idx] = self.__load_training_masks(
                sample_idx)

            if self.use_protbert:
                features[batch_idx] = np.concatenate((protbert_features, base_features), axis=-1)
            else:
                features[batch_idx] = base_features

        return [features, att_mask, position_ids], label, weight_mask

# Loading Model(s), Dataset, and Labels
# Prediction/Inference with SAINT-Angle

# Base Models Description:-
#     model_1 -> Basic architecture trained with 57 Base features
#     model_2 -> Basic architecture trained with 57 Base and 16 Window10 features
#     model_3 -> ProtTrans architecture trained with 57 Base and 1024 ProtTrans features
#     model_4 -> ProtTrans architecture trained with 57 Base, 16 Window10, and 1024 ProtTrans features
#     model_5 -> ProtTrans architecture trained with 57 Base, 36 Window20, and 1024 ProtTrans features
#     model_6 -> ProtTrans architecture trained with 57 Base, 96 Window50, and 1024 ProtTrans features
#     model_7 -> Residual architecture trained with 57 Base and 1024 ProtTrans features
#     model_8 -> Residual architecture trained with 57 Base, 16 Window10, and 1024 ProtTrans features
#     model_9 -> Basic architecture trained with 232 ESIDEN features
#     model_10 -> Basic architecture trained with 232 ESIDEN and 30 HMM features
#     model_11 -> ProtTrans architecture trained with 232 ESIDEN, 30 HMM, and 1024 ProtTrans features
#     model_12 -> Residual architecture trained with 232 ESIDEN, 30 HMM, and 1024 ProtTrans features

base_models_dict = {
    "model_1": {
        "name": "SAINT-Angle (model_1)",
        "details": "Basic architecture trained with 57 Base features",
        "num_features": 57,
        "use_protbert": False,
        "base_path": "split-base(57)-win10(16)"
    },
    "model_2": {
        "name": "SAINT-Angle (model_2)",
        "details": "Basic architecture trained with 57 Base and 16 Window10 features",
        "num_features": 73,
        "use_protbert": False,
        "base_path": "split-base(57)-win10(16)"
    },
    "model_3": {
        "name": "SAINT-Angle (model_3)",
        "details": "ProtTrans architecture trained with 57 Base and 1024 ProtTrans features",
        "num_features": 1081,
        "use_protbert": True,
        "base_path": "split-base(57)-win10(16)"
    },
    "model_4": {
        "name": "SAINT-Angle (model_4)",
        "details": "ProtTrans architecture trained with 57 Base, 16 Window10, and 1024 ProtTrans features",
        "num_features": 1097,
        "use_protbert": True,
        "base_path": "split-base(57)-win10(16)"
    },
    "model_5": {
        "name": "SAINT-Angle (model_5)",
        "details": "ProtTrans architecture trained with 57 Base, 36 Window20, and 1024 ProtTrans features",
        "num_features": 1117,
        "use_protbert": True,
        "base_path": "split-base(57)-win20(36)"
    },
    "model_6": {
        "name": "SAINT-Angle (model_6)",
        "details": "ProtTrans architecture trained with 57 Base, 96 Window50, and 1024 ProtTrans features",
        "num_features": 1177,
        "use_protbert": True,
        "base_path": "split-base(57)-win50(96)"
    },
    "model_7": {
        "name": "SAINT-Angle (model_7)",
        "details": "Residual architecture trained with 57 Base and 1024 ProtTrans features",
        "num_features": 1081,
        "use_protbert": True,
        "base_path": "split-base(57)-win10(16)"
    },
    "model_8": {
        "name": "SAINT-Angle (model_8)",
        "details": "Residual architecture trained with 57 Base, 16 Window10, and 1024 ProtTrans features",
        "num_features": 1097,
        "use_protbert": True,
        "base_path": "split-base(57)-win10(16)"
    },
    "model_9": {
        "name": "SAINT-Angle (model_9)",
        "details": "Basic architecture trained with 232 ESIDEN features",
        "num_features": 232,
        "use_protbert": False,
        "base_path": "split-ESIDEN(232)"
    },
    "model_10": {
        "name": "SAINT-Angle (model_10)",
        "details": "Basic architecture trained with 232 ESIDEN and 30 HMM features",
        "num_features": 262,
        "use_protbert": False,
        "base_path": "split-sa_es(262)-win0(0)"
    },
    "model_11": {
        "name": "SAINT-Angle (model_11)",
        "details": "ProtTrans architecture trained with 232 ESIDEN, 30 HMM, and 1024 ProtTrans features",
        "num_features": 1286,
        "use_protbert": True,
        "base_path": "split-sa_es(262)-win0(0)"
    },
    "model_12": {
        "name": "SAINT-Angle (model_12)",
        "details": "Residual architecture trained with 232 ESIDEN, 30 HMM, and 1024 ProtTrans features",
        "num_features": 1286,
        "use_protbert": True,
        "base_path": "split-sa_es(262)-win0(0)"
    }
}

ensemble_3_model_names = ["model_10", "model_11", "model_12"]
ensemble_8_model_names = ["model_1", "model_2", "model_3", "model_4", "model_5", "model_6", "model_7", "model_8"]
model_names = ensemble_8_model_names + ["model_9"] + ensemble_3_model_names + ["ensemble_3", "ensemble_8"]

additional_layers = {
    "MyLayer": MyLayer,
    "backend": backend,
    "shape_list": get_shape_list,
    "total_mse_sin_cos": get_total_mse_sin_cos,
    "total_mae_apu": get_total_mae,
    "phi_mae": get_phi_mae,
    "psi_mae": get_psi_mae,
    "total_sae_apu": get_total_sae,
    "phi_sae": get_phi_sae,
    "psi_sae": get_psi_sae
}

assert model_name in model_names, "Invalid model name"

if model_name == "ensemble_3":
    print("Loading models...\n")
    model_10 = load_model('./models/model_10.h5', custom_objects=additional_layers)
    model_11 = load_model('./models/model_11.h5', custom_objects=additional_layers)
    model_12 = load_model('./models/model_12.h5', custom_objects=additional_layers)
    
    models_loaded = '\n'.join(
        [f"{base_models_dict[model_name]['name']}: {base_models_dict[model_name]['details']}" for model_name in ensemble_3_model_names]
    )
    print(f"Following models loaded:-\n{models_loaded}\n")

    print("Loading dataset and labels...")
    configs = [
        {"base_path": "split-sa_es(262)-win0(0)", "num_features": 262, "use_protbert": False},
        {"base_path": "split-sa_es(262)-win0(0)", "num_features": 1286, "use_protbert": True}
    ]
    test_data_loaders = []
    
    for config in configs:
        test_data_loaders.append(
            DataLoader(
                base_path=f'./datasets/{dataset_name}/{config["base_path"]}',
                protbert_path=f'./datasets/{dataset_name}/split-ProtBERT(1024)',
                dataset_name=dataset_name,
                batch_size=1,
                length_file_path=f'./datasets/{dataset_name}/{dataset_name}_len.txt',
                phi_file_path=f'./datasets/{dataset_name}/{dataset_name}_phi.txt',
                psi_file_path=f'./datasets/{dataset_name}/{dataset_name}_psi.txt',
                num_features=config['num_features'],
                use_protbert=config['use_protbert'],
                ignore_first_and_last=True,
                verbose=verbose
            )
        )
    testphi, testpsi, testlabel = load_labels(f'./datasets/{dataset_name}', dataset_name, ignore_first_and_last=True, verbose=verbose)

    print("\nPredicting with SAINT-Angle...")
    y_predict = model_10.predict(test_data_loaders[0])
    y_predict = y_predict + model_11.predict(test_data_loaders[1])
    y_predict = y_predict + model_12.predict(test_data_loaders[1])

    y_predict = y_predict / 3
    phi_score, psi_score = get_scores(y_predict)
elif model_name == "ensemble_8":
    print("Loading models...\n")
    model_1 = load_model('./models/model_1.h5', custom_objects=additional_layers)
    model_2 = load_model('./models/model_2.h5', custom_objects=additional_layers)
    model_3 = load_model('./models/model_3.h5', custom_objects=additional_layers)
    model_4 = load_model('./models/model_4.h5', custom_objects=additional_layers)
    model_5 = load_model('./models/model_5.h5', custom_objects=additional_layers)
    model_6 = load_model('./models/model_6.h5', custom_objects=additional_layers)
    model_7 = load_model('./models/model_7.h5', custom_objects=additional_layers)
    model_8 = load_model('./models/model_8.h5', custom_objects=additional_layers)

    models_loaded = '\n'.join(
        [f"{base_models_dict[model_name]['name']}: {base_models_dict[model_name]['details']}" for model_name in ensemble_8_model_names]
    )
    print(f"Following models loaded:-\n{models_loaded}\n")

    print("Loading dataset and labels...")
    configs = [
        {"base_path": "split-base(57)-win10(16)", "num_features": 57, "use_protbert": False},
        {"base_path": "split-base(57)-win10(16)", "num_features": 73, "use_protbert": False},
        {"base_path": "split-base(57)-win10(16)", "num_features": 1081, "use_protbert": True},
        {"base_path": "split-base(57)-win10(16)", "num_features": 1097, "use_protbert": True},
        {"base_path": "split-base(57)-win20(36)", "num_features": 1117, "use_protbert": True},
        {"base_path": "split-base(57)-win50(96)", "num_features": 1177, "use_protbert": True}
    ]
    test_data_loaders = []

    for config in configs:
        test_data_loaders.append(
            DataLoader(
                base_path=f'./datasets/{dataset_name}/{config["base_path"]}',
                protbert_path=f'./datasets/{dataset_name}/split-ProtBERT(1024)',
                dataset_name=dataset_name,
                batch_size=1,
                length_file_path=f'./datasets/{dataset_name}/{dataset_name}_len.txt',
                phi_file_path=f'./datasets/{dataset_name}/{dataset_name}_phi.txt',
                psi_file_path=f'./datasets/{dataset_name}/{dataset_name}_psi.txt',
                num_features=config['num_features'],
                use_protbert=config['use_protbert'],
                ignore_first_and_last=True,
                verbose=verbose
            )
        )
    testphi, testpsi, testlabel = load_labels(f'./datasets/{dataset_name}', dataset_name, ignore_first_and_last=True, verbose=verbose)

    print("\nPredicting with SAINT-Angle...")
    y_predict = model_1.predict(test_data_loaders[0])
    y_predict = y_predict + model_2.predict(test_data_loaders[1])
    y_predict = y_predict + model_3.predict(test_data_loaders[2])
    y_predict = y_predict + model_4.predict(test_data_loaders[3])
    y_predict = y_predict + model_5.predict(test_data_loaders[4])
    y_predict = y_predict + model_6.predict(test_data_loaders[5])
    y_predict = y_predict + model_7.predict(test_data_loaders[2])
    y_predict = y_predict + model_8.predict(test_data_loaders[3])

    y_predict = y_predict / 8
    phi_score, psi_score = get_scores(y_predict)
else:
    print("Loading model...\n")
    model = load_model(f'./models/{model_name}.h5', custom_objects=additional_layers)
    print(f"Following model loaded:-\n{base_models_dict[model_name]['name']}: {base_models_dict[model_name]['details']}\n")

    print("Loading dataset and labels...")
    test_data_loader = DataLoader(
        base_path=f'./datasets/{dataset_name}/{base_models_dict[model_name]["base_path"]}',
        protbert_path=f'./datasets/{dataset_name}/split-ProtBERT(1024)',
        dataset_name=dataset_name,
        batch_size=1,
        length_file_path=f'./datasets/{dataset_name}/{dataset_name}_len.txt',
        phi_file_path=f'./datasets/{dataset_name}/{dataset_name}_phi.txt',
        psi_file_path=f'./datasets/{dataset_name}/{dataset_name}_psi.txt',
        num_features=base_models_dict[model_name]['num_features'],
        use_protbert=base_models_dict[model_name]['use_protbert'],
        ignore_first_and_last=False,
        verbose=verbose
    )
    test_phi, test_psi, test_label = load_labels(f'./datasets/{dataset_name}', dataset_name, ignore_first_and_last=False, verbose=verbose)

    print("\nPredicting with SAINT-Angle...")
    y_predict = model.predict(test_data_loader)

    total_score = get_total_mae_np(test_label, y_predict, test_phi, test_psi, -500)
    phi_score = get_phi_mae_np(test_label, y_predict, test_phi, -500)
    psi_score = get_psi_mae_np(test_label, y_predict, test_psi, -500)
    avg_score = 0.5 * (phi_score + psi_score)

print(f"\n=== Prediction Results from SAINT-Angle on {dataset_name} ===")
print(f"MAE of phi:\t{phi_score}")
print(f"MAE of psi:\t{psi_score}")

if output_predictions:
    print("\nWriting predictions to output files...")
    phi_predicted = np.arctan2(y_predict[:, :, 0], y_predict[:, :, 1]) * 180 / np.pi
    psi_predicted = np.arctan2(y_predict[:, :, 2], y_predict[:, :, 3]) * 180 / np.pi

    if not os.path.exists(f'./predictions/{dataset_name}'):
        os.makedirs(f'./predictions/{dataset_name}')

    np.savetxt(f'./predictions/{dataset_name}/{dataset_name}_pred_phi.txt', phi_predicted)
    np.savetxt(f'./predictions/{dataset_name}/{dataset_name}_pred_psi.txt', psi_predicted)