import tensorflow as tf
from keras import backend
from keras.layers import add, Layer

import math
import os

from .features import *

def get_shape_list(x):
    if backend.backend() != "theano":
        temp = backend.int_shape(x)
    else:
        temp = x.shape

    temp = list(temp)
    temp[0] = -1
    return temp

class MyLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._x = None
    
    def build(self, input_shape):
        self._x = backend.variable(0.2)
        self._x._trainable = True
        self._trainable_weights = [self._x]
        super().build(input_shape)

    def call(self, x, **kwargs):
        a, b = x
        result = add([self._x * a, (1 - self._x) * b])
        return result

    def compute_output_shape(self, input_shape):
        return input_shape[0]

class CustomDataLoader(tf.keras.utils.Sequence):
    def __init__(self, proteins_dict, inputs_dir_path, batch_size, window_size=0, use_prottrans=False, use_gpu=False):
        self.proteins_dict = proteins_dict
        self.inputs_dir_path = inputs_dir_path
        self.batch_size = batch_size
        self.window_size = window_size
        self.use_prottrans = use_prottrans
        self.use_gpu = use_gpu

    def __len__(self):
        return math.ceil(len(self.proteins_dict) / self.batch_size)

    def __getitem__(self, index):
        batch_protein_names = list(self.proteins_dict.keys())[index * self.batch_size:(index + 1) * self.batch_size]
        num_features = 57 + (self.window_size * 2 - 4 if self.window_size > 0 else 0) + (1024 if self.use_prottrans else 0)

        batch_features = np.zeros(shape=(len(batch_protein_names), 700, num_features))
        batch_attention_masks = np.zeros(shape=(len(batch_protein_names), 700))
        batch_positions_ids = np.zeros(shape=(len(batch_protein_names), 700))
        batch_weight_masks = np.zeros(shape=(len(batch_protein_names), 700))
        batch_labels = None

        for batch_index, protein_name in enumerate(batch_protein_names):
            protein_file_path = self.inputs_dir_path + os.sep + protein_name

            with open(protein_file_path + ".fasta", 'r') as fasta_file:
                pseq = fasta_file.read().split('\n')[1]
			
            hhm = np.nan_to_num(generate_hhm(hhm_file_path=protein_file_path + ".hhm", pseq=pseq), nan=0.0)
            pssm = np.nan_to_num(generate_pssm(pssm_file_path=protein_file_path + ".pssm", pseq=pseq), nan=0.0)
            pcp = np.nan_to_num(generate_pcp(pseq=pseq), nan=0.0)
            contact = np.nan_to_num(generate_contact(contact_file_path=protein_file_path + ".spotcon", pseq=pseq, window_size=self.window_size), nan=0.0)
            prottrans = np.nan_to_num(generate_prottrans(pseq=pseq, use_gpu=self.use_gpu), nan=0.0) if self.use_prottrans else None

            features = np.concatenate([hhm, pssm, pcp], axis=1)

            if contact is not None:
                features = np.concatenate([features, contact], axis=1)

            if prottrans is not None:
                features = np.concatenate([prottrans, features], axis=1)

            batch_features[batch_index, :self.proteins_dict[protein_name]] = features

            attention_masks, position_ids, weight_masks = np.zeros(shape=(700,)), np.arange(700), np.ones(shape=(700,))
            attention_masks[self.proteins_dict[protein_name]:] = -np.inf
            weight_masks[self.proteins_dict[protein_name]:] = 0

            batch_attention_masks[batch_index] = attention_masks
            batch_positions_ids[batch_index] = position_ids
            batch_weight_masks[batch_index] = weight_masks

        return [batch_features, batch_attention_masks, batch_positions_ids], batch_labels, batch_weight_masks