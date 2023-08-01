from keras import backend
from keras.models import load_model

import argparse
import numpy as np
import os
import gc

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

from utils.axils import *
from utils.evaluations import *
from utils.features import *

parser = argparse.ArgumentParser()

parser.add_argument("-l", "--list", dest="proteins_list_path", default="./proteins_list", help="path to text file containing list of proteins")
parser.add_argument("-i", "--inputs", dest="inputs_dir_path", default="./inputs", help="path to directory containing input files")
parser.add_argument("-o", "--outputs", dest="outputs_dir_path", default="./outputs", help="path to directory containing output files")
parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", help="whether to print out detailed messages (Default: False)")

parser.set_defaults(verbose=False)

args = parser.parse_args()

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
    print("\nLoading models...\n")
    model_10 = load_model('./models/model_10.h5', custom_objects=additional_layers)
    model_11 = load_model('./models/model_11.h5', custom_objects=additional_layers)
    model_12 = load_model('./models/model_12.h5', custom_objects=additional_layers)

    models_loaded = '\n'.join(
        [f"{base_models_dict[model_name]['name']}: {base_models_dict[model_name]['details']}" for model_name in
         ensemble_3_model_names]
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
    test_phi, test_psi, test_label = load_labels(f'./datasets/{dataset_name}', dataset_name, ignore_first_and_last=True,
                                                 verbose=verbose)

    print("\nPredicting with SAINT-Angle...")
    y_predict = model_10.predict(test_data_loaders[0])
    y_predict = y_predict + model_11.predict(test_data_loaders[1])
    y_predict = y_predict + model_12.predict(test_data_loaders[1])

    y_predict = y_predict / 3
    phi_score, psi_score = get_scores(test_label, y_predict, test_phi, test_psi)
elif model_name == "ensemble_8":
    print("\nLoading models...\n")
    model_1 = load_model('./models/model_1.h5', custom_objects=additional_layers)
    model_2 = load_model('./models/model_2.h5', custom_objects=additional_layers)
    model_3 = load_model('./models/model_3.h5', custom_objects=additional_layers)
    model_4 = load_model('./models/model_4.h5', custom_objects=additional_layers)
    model_5 = load_model('./models/model_5.h5', custom_objects=additional_layers)
    model_6 = load_model('./models/model_6.h5', custom_objects=additional_layers)
    model_7 = load_model('./models/model_7.h5', custom_objects=additional_layers)
    model_8 = load_model('./models/model_8.h5', custom_objects=additional_layers)

    models_loaded = '\n'.join(
        [f"{base_models_dict[model_name]['name']}: {base_models_dict[model_name]['details']}" for model_name in
         ensemble_8_model_names]
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
    test_phi, test_psi, test_label = load_labels(f'./datasets/{dataset_name}', dataset_name, ignore_first_and_last=True,
                                                 verbose=verbose)

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
    phi_score, psi_score = get_scores(test_label, y_predict, test_phi, test_psi)

if output_predictions:
    print("Writing predictions to output files...\n")
    phi_predicted = np.arctan2(y_predict[:, :, 0], y_predict[:, :, 1]) * 180 / np.pi
    psi_predicted = np.arctan2(y_predict[:, :, 2], y_predict[:, :, 3]) * 180 / np.pi

    if not os.path.exists(f'./predictions/{dataset_name}'):
        os.makedirs(f'./predictions/{dataset_name}')

    np.savetxt(f'./predictions/{dataset_name}/{model_name}_{dataset_name}_pred_phi.txt', phi_predicted)
    np.savetxt(f'./predictions/{dataset_name}/{model_name}_{dataset_name}_pred_psi.txt', psi_predicted)