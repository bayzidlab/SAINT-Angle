import tensorflow as tf
from keras import backend
from keras.layers import add, Layer
import math

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
        self.shuffle = shuffle

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

        self.on_epoch_end()

    def on_epoch_end(self):
        # updating indices after each epoch
        self.indices = np.arange(len(self.data_len))

        if self.shuffle:
            np.random.shuffle(self.indices)

    def __load_lengths(self, index):
        return self.data_len[index]

    def __load_labels(self, index):
        return self.data_phi[index, :], self.data_psi[index, :], self.data_label[index, :]

    def __load_training_masks(self, index):
        return self.data_att_mask[index, :], self.data_position_ids[index, :], self.weight_mask[index, :]

    def __len__(self):
        # returning total number of batches
        return int(math.ceil(len(self.data_len) / self.batch_size))

    def __getitem__(self, index):
        # generating indices of a particular batch
        batch_indices = self.indices[index * self.batch_size: (index + 1) * self.batch_size]

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