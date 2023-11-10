import torch
from transformers import T5EncoderModel, T5Tokenizer

import pandas as pd
import numpy as np

import re
import gc

import warnings
warnings.simplefilter("ignore")

def generate_pssm(pssm_file_path, pseq):
    num_pssm_columns = 44
    pssm_column_names = [str(index) for index in range(num_pssm_columns)]

    with open(pssm_file_path, 'r') as pssm_file:
        pssm = pd.read_csv(pssm_file, names=pssm_column_names, delim_whitespace=True).dropna().values[:, 2:22].astype(np.float32)

    assert pssm.shape[0] == len(pseq), "PSSM file is in wrong format!"
    return pssm

def generate_hhm(hhm_file_path, pseq):
    num_hhm_columns = 22
    hhm_column_names = [str(index) for index in range(num_hhm_columns)]

    with open(hhm_file_path, 'r') as hhm_file:
        hhm = pd.read_csv(hhm_file, names=hhm_column_names, delim_whitespace=True)

    pos1 = (hhm["0"] == "HMM").idxmax() + 3
    hhm = hhm[pos1:-1].values[:, :num_hhm_columns].reshape((-1, 44))
    hhm[hhm == '*'] = "9999"
    hhm = hhm[:, 2:-12].astype(np.float32)

    assert hhm.shape[0] == len(pseq), "HHM file is in wrong format!"
    return hhm

def get_pcp_dictionary():
    pcp_dictionary = {
        'A': [-0.350, -0.680, -0.677, -0.171, -0.170, 0.900, -0.476],
        'R': [0.105, 0.373, 0.466, -0.900, 0.900, 0.528, -0.371],
        'N': [-0.213, -0.329, -0.243, -0.674, -0.075, -0.403, -0.529],
        'D': [-0.213, -0.417, -0.281, -0.767, -0.900, -0.155, -0.635],
        'C': [-0.140, -0.329, -0.359, 0.508, -0.114, -0.652, 0.476],
        'Q': [-0.230, -0.110, -0.020, -0.464, -0.276, 0.528, -0.371],
        'E': [-0.230, -0.241, -0.058, -0.696, -0.868, 0.900, -0.582],
        'G': [-0.900, -0.900, -0.900, -0.342, -0.179, -0.900, -0.900],
        'H': [0.384, 0.110, 0.138, -0.271, 0.195, -0.031, -0.106],
        'I': [0.900, -0.066, -0.009, 0.652, -0.186, 0.155, 0.688],
        'L': [0.213, -0.066, -0.009, 0.596, -0.186, 0.714, -0.053],
        'K': [-0.088, 0.066, 0.163, -0.889, 0.727, 0.279, -0.265],
        'M': [0.110, 0.066, 0.087, 0.337, -0.262, 0.652, -0.001],
        'F': [0.363, 0.373, 0.412, 0.646, -0.272, 0.155, 0.318],
        'P': [0.247, -0.900, -0.294, 0.055, -0.010, -0.900, 0.106],
        'S': [-0.337, -0.637, -0.544, -0.364, -0.265, -0.466, -0.212],
        'T': [0.402, -0.417, -0.321, -0.199, -0.288, -0.403, 0.212],
        'W': [0.479, 0.900, 0.900, 0.900, -0.209, 0.279, 0.529],
        'Y': [0.363, 0.417, 0.541, 0.188, -0.274, -0.155, 0.476],
        'V': [0.677, -0.285, -0.232, 0.331, -0.191, -0.031, 0.900],
        'X': [0.077, -0.154, -0.062, -0.076, -0.145, 0.0497, -0.0398]
    }
    return pcp_dictionary

def generate_pcp(pseq):
    pcp_dictionary = get_pcp_dictionary()
    pcp = np.array([pcp_dictionary.get(amino_acid_residue, [0] * 7) for amino_acid_residue in pseq], dtype=np.float32)
    return pcp

def generate_contact(contact_file_path, pseq, window_size, min_sep=3):
    if not window_size > 0:
        return None

    pseq_length = len(pseq)
    contact_feats = np.zeros(shape=(pseq_length, pseq_length, 1))

    with open(contact_file_path, 'r') as contact_file:
        contact_map = pd.read_csv(contact_file, names=["pos1", "pos2", "idk1", "idk2", "score"], delim_whitespace=True)

    contact_map = contact_map[contact_map["pos1"].astype(str).str.isdigit()].dropna().values

    if contact_map.shape[0] == 0:
        with open(contact_file_path, 'r') as contact_file:
            contact_map = pd.read_csv(contact_file, names=["pos1", "pos2", "score"], delim_whitespace=True)

        contact_map = contact_map[contact_map["pos1"].astype(str).str.isdigit()].dropna().values
        pos1 = contact_map[:, 0].astype(int)
        pos2 = contact_map[:, 1].astype(int)
    else:
        pos1 = contact_map[:, 0].astype(int) - 1
        pos2 = contact_map[:, 1].astype(int) - 1

    score = contact_map[:, -1:]
    contact_feats[pos1, pos2] = score
    contact_feats = contact_feats + np.transpose(contact_feats, axes=(1, 0, 2)) + np.tril(m=np.triu(m=np.ones(shape=(pseq_length, pseq_length)), k=(-min_sep + 1)), k=(min_sep - 1))[:, :, None]

    contact_image = [contact_feats]
    contact_image = np.concatenate(contact_image, axis=2)

    assert pseq_length == contact_image.shape[0]

    features_depth = contact_image.shape[2]
    window_size = int(window_size)

    resize = np.concatenate([np.zeros(shape=(window_size, pseq_length, features_depth)), np.concatenate([contact_image, np.zeros(shape=(window_size, pseq_length, features_depth))], axis=0)], axis=0)
    contact_array = np.concatenate([resize[index:(index + 2 * window_size + 1), index, :features_depth] for index in range(pseq_length)], axis=1).T
    removal_indices = np.array([window_size + index for index in range(-2, 3)])
    contact = np.delete(contact_array, obj=removal_indices, axis=1).astype(np.float32)
    return contact

def generate_prottrans(pseq, use_gpu=False):
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    gc.collect()

    device = torch.device("cuda:0") if use_gpu and torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model = model.eval()

    p_s_e_q = [re.sub(r"[UZOB]", 'X', ' '.join(pseq))]

    ids = tokenizer.batch_encode_plus(p_s_e_q, add_special_tokens=True, padding=True)
    input_ids = torch.tensor(ids["input_ids"]).to(device)
    attention_mask = torch.tensor(ids["attention_mask"]).to(device)

    with torch.no_grad():
        embedding = model(input_ids=input_ids, attention_mask=attention_mask)

    embedding = embedding.last_hidden_state.cpu().numpy()
    prottrans = []

    for sequence_num in range(len(embedding)):
        sequence_len = (attention_mask[sequence_num] == 1).sum()
        sequence_emd = embedding[sequence_num][:sequence_len - 1]
        prottrans.append(sequence_emd)

    return prottrans[0]