import tensorflow as tf
from keras import backend
from keras.layers import add, Layer

import math
import os
import numpy as np
import pickle

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
    def __init__(self, proteins_dict, protein_names, features_dir_path, batch_size, window_size=0, use_prottrans=False, data_dir_path=None):
        self.proteins_dict, self.protein_names = proteins_dict, protein_names
        self.features_dir_path, self.batch_size = features_dir_path, batch_size
        self.window_size, self.use_prottrans, self.data_dir_path = window_size, use_prottrans, data_dir_path
    
    def __len__(self):
        return math.ceil(len(self.protein_names) / self.batch_size)
    
    def __getitem__(self, index):
        batch_protein_names = self.protein_names[index * self.batch_size:(index + 1) * self.batch_size]
        num_features = 57 + (self.window_size * 2 - 4 if self.window_size > 0 else 0) + (1024 if self.use_prottrans else 0)
        
        batch_features = np.zeros(shape=(len(batch_protein_names), 700, num_features))
        batch_attention_masks = np.zeros(shape=(len(batch_protein_names), 700))
        batch_positions_ids = np.zeros(shape=(len(batch_protein_names), 700))
        batch_weight_masks = np.zeros(shape=(len(batch_protein_names), 700))
        batch_labels = np.zeros(shape=(len(batch_protein_names), 700, 4))
        
        for batch_index, protein_name in enumerate(batch_protein_names):
            features_file_path = self.features_dir_path + os.sep + protein_name
            
            with open(features_file_path + "_hhm.npy", 'rb') as hhm_file:
                hhm = np.load(file=hhm_file)
            
            with open(features_file_path + "_pssm.npy", 'rb') as pssm_file:
                pssm = np.load(file=pssm_file)
            
            with open(features_file_path + "_pcp.npy", 'rb') as pcp_file:
                pcp = np.load(file=pcp_file)
            
            features = np.concatenate([hhm, pssm, pcp], axis=1)
            
            if self.window_size > 0:
                with open(features_file_path + f"_win{self.window_size}.npy", 'rb') as win_file:
                    contact = np.load(file=win_file)
            else:
                contact = None
            
            if self.use_prottrans:
                with open(features_file_path + "_prottrans.npy", 'rb') as prottrans_file:
                    prottrans = np.load(file=prottrans_file)
            else:
                prottrans = None
            
            if contact is not None:
                features = np.concatenate([features, contact], axis=1)
                mu_std_file_mappings = {10: "5norm_data.p", 20: "0norm_data.p", 50: "1norm_data.p"}

                with open(self.data_dir_path + os.sep + mu_std_file_mappings[self.window_size], 'rb') as pickle_file:
                    norm_dict = pickle.load(file=pickle_file, encoding="latin1")

                norm_mu, norm_std = norm_dict["mu1d"], norm_dict["std1d"]
                features = (features - norm_mu) / norm_std
            else:
                with open(self.data_dir_path + os.sep + "5norm_data.p", 'rb') as pickle_file:
                    norm_dict = pickle.load(file=pickle_file, encoding="latin1")

                norm_mu, norm_std = norm_dict["mu1d"][:57], norm_dict["std1d"][:57]
                features = (features - norm_mu) / norm_std

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