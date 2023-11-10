from keras.models import load_model

import argparse
import os
from tqdm import tqdm
import shutil

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

from utils.axils import *
from utils.evaluations import *
from utils.features import *

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def main(args):
    with open(args.proteins_list_path, 'r') as proteins_list_file:
        proteins_list = [protein_name for protein_name in proteins_list_file.read().split('\n') if protein_name != '']

    accepted_proteins, rejected_proteins, protein_pseqs = {}, [], {}

    for protein_name in proteins_list:
        with open(args.inputs_dir_path + os.sep + protein_name + ".fasta", 'r') as fasta_file:
            pseq = fasta_file.read().split('\n')[1]

        if len(pseq) > 700:
            rejected_proteins.append(protein_name)
        else:
            accepted_proteins[protein_name] = len(pseq)
            protein_pseqs[protein_name] = pseq

    print(f"\nTotal {len(proteins_list)} proteins provided.")
    print(f"Among them, {', '.join(rejected_proteins) if len(rejected_proteins) > 0 else 'no'} proteins have length longer than 700.")
    print(f"And, {', '.join(accepted_proteins.keys()) if len(accepted_proteins) > 0 else 'no'} proteins have length shorter than or equal to 700.")
    print(f"Working on {len(accepted_proteins)} proteins for phi and psi angles prediction.\n")

    del proteins_list, rejected_proteins
    gc.collect()

    if len(accepted_proteins) == 0:
        del accepted_proteins, protein_pseqs
        gc.collect()
        return True

    spotcon_availables, spotcon_unavailables = {}, {}

    for protein_name in accepted_proteins:
        if os.path.isfile(args.inputs_dir_path + os.sep + protein_name + ".spotcon"):
            spotcon_availables[protein_name] = accepted_proteins[protein_name]
        else:
            spotcon_unavailables[protein_name] = accepted_proteins[protein_name]

    del accepted_proteins
    gc.collect()

    additional_layers = {"MyLayer": MyLayer, "backend": backend, "shape_list": get_shape_list, "total_mse_sin_cos": mean_mse, "total_mae_apu": mean_mae, "phi_mae": phi_mae, "psi_mae": psi_mae, "total_sae_apu": mean_sae, "phi_sae": phi_sae, "psi_sae": psi_sae}
    base_models_dict = {"model_1": {"window_size": 0, "use_prottrans": False}, "model_2": {"window_size": 10, "use_prottrans": False}, "model_3": {"window_size": 0, "use_prottrans": True}, "model_4": {"window_size": 10, "use_prottrans": True}, "model_5": {"window_size": 20, "use_prottrans": True}, "model_6": {"window_size": 50, "use_prottrans": True}, "model_7": {"window_size": 0, "use_prottrans": True}, "model_8": {"window_size": 10, "use_prottrans": True}}
    non_window_models, window_models = ["model_1", "model_3", "model_7"], ["model_2", "model_4", "model_5", "model_6", "model_8"]
    base_models = {}

    for model_name in tqdm(iterable=non_window_models, desc="Loading Non-window Models...", ncols=100, unit="model"):
        base_models[model_name] = load_model("./models" + os.sep + model_name + ".h5", custom_objects=additional_layers)

    print(f"\n{', '.join(non_window_models)} loaded.\n")

    del non_window_models
    gc.collect()

    predictions = {}

    features_dir_path = "./features"

    if not os.path.exists(features_dir_path):
        os.makedirs(features_dir_path)

    if len(spotcon_unavailables) > 0:
        for protein_name in tqdm(iterable=spotcon_unavailables, desc="Generating Features...", ncols=100, unit="protein"):
            protein_file_path = args.inputs_dir_path + os.sep + protein_name
            features_file_path = features_dir_path + os.sep + protein_name

            hhm = generate_hhm(hhm_file_path=protein_file_path + ".hhm", pseq=protein_pseqs[protein_name])
            pssm = generate_pssm(pssm_file_path=protein_file_path + ".pssm", pseq=protein_pseqs[protein_name])
            pcp = generate_pcp(pseq=protein_pseqs[protein_name])
            prottrans = generate_prottrans(pseq=protein_pseqs[protein_name], use_gpu=args.use_gpu)

            with open(features_file_path + "_hhm.npy", 'wb') as hhm_file:
                np.save(file=hhm_file, arr=hhm)

            with open(features_file_path + "_pssm.npy", 'wb') as pssm_file:
                np.save(file=pssm_file, arr=pssm)

            with open(features_file_path + "_pcp.npy", 'wb') as pcp_file:
                np.save(file=pcp_file, arr=pcp)

            with open(features_file_path + "_prottrans.npy", 'wb') as prottrans_file:
                np.save(file=prottrans_file, arr=prottrans)

        print(f"\nSPOTCON file is not available for {', '.join(spotcon_unavailables.keys())} proteins.")
        print("Predicting phi and psi angles of these proteins without window features.\n")

        protein_names = list(spotcon_unavailables.keys())

        for model_name in tqdm(iterable=base_models, desc="Predicting Angles...", ncols=100, unit="model"):
            predictions[model_name] = base_models[model_name].predict(CustomDataLoader(proteins_dict=spotcon_unavailables, protein_names=protein_names, features_dir_path=features_dir_path, batch_size=1, window_size=base_models_dict[model_name]["window_size"], use_prottrans=base_models_dict[model_name]["use_prottrans"], data_dir_path="./data"))

        avg_prediction = sum(predictions.values()) / len(predictions)

        if not os.path.exists(args.outputs_dir_path):
            os.makedirs(args.outputs_dir_path)

        for index, protein_name in enumerate(protein_names):
            for model_name in predictions:
                prediction = predictions[model_name][index, :spotcon_unavailables[protein_name]]
                phi_predictions = np.arctan2(prediction[:, 0], prediction[:, 1]) * 180 / np.pi
                psi_predictions = np.arctan2(prediction[:, 2], prediction[:, 3]) * 180 / np.pi
                phi_predictions[0] = psi_predictions[-1] = 360

                outputs = pd.DataFrame({"Amino Acid": list(protein_pseqs[protein_name]), "Phi": phi_predictions, "Psi": psi_predictions})
                outputs.to_csv(args.outputs_dir_path + os.sep + protein_name + '.' + model_name + ".csv", index=False)

            prediction = avg_prediction[index, :spotcon_unavailables[protein_name]]
            phi_predictions = np.arctan2(prediction[:, 0], prediction[:, 1]) * 180 / np.pi
            psi_predictions = np.arctan2(prediction[:, 2], prediction[:, 3]) * 180 / np.pi
            phi_predictions[0] = psi_predictions[-1] = 360

            outputs = pd.DataFrame({"Amino Acid": list(protein_pseqs[protein_name]), "Phi": phi_predictions, "Psi": psi_predictions})
            outputs.to_csv(args.outputs_dir_path + os.sep + protein_name + '.' + "ensemble3.csv", index=False)

        print("\nDone!\n")

    del spotcon_unavailables, predictions
    gc.collect()

    if len(spotcon_availables) > 0:
        for model_name in tqdm(iterable=window_models, desc="Loading Window Models...", ncols=100, unit="model"):
            base_models[model_name] = load_model("./models" + os.sep + model_name + ".h5", custom_objects=additional_layers)

        print(f"\n{', '.join(window_models)} loaded.\n")

        del window_models
        gc.collect()

        predictions = {}

        for protein_name in tqdm(iterable=spotcon_availables, desc="Generating Features...", ncols=100, unit="protein"):
            protein_file_path = args.inputs_dir_path + os.sep + protein_name
            features_file_path = features_dir_path + os.sep + protein_name

            hhm = generate_hhm(hhm_file_path=protein_file_path + ".hhm", pseq=protein_pseqs[protein_name])
            pssm = generate_pssm(pssm_file_path=protein_file_path + ".pssm", pseq=protein_pseqs[protein_name])
            pcp = generate_pcp(pseq=protein_pseqs[protein_name])
            prottrans = generate_prottrans(pseq=protein_pseqs[protein_name], use_gpu=args.use_gpu)

            with open(features_file_path + "_hhm.npy", 'wb') as hhm_file:
                np.save(file=hhm_file, arr=hhm)

            with open(features_file_path + "_pssm.npy", 'wb') as pssm_file:
                np.save(file=pssm_file, arr=pssm)

            with open(features_file_path + "_pcp.npy", 'wb') as pcp_file:
                np.save(file=pcp_file, arr=pcp)

            with open(features_file_path + "_prottrans.npy", 'wb') as prottrans_file:
                np.save(file=prottrans_file, arr=prottrans)

            contact_10 = generate_contact(contact_file_path=protein_file_path + ".spotcon", pseq=protein_pseqs[protein_name], window_size=10)
            contact_20 = generate_contact(contact_file_path=protein_file_path + ".spotcon", pseq=protein_pseqs[protein_name], window_size=20)
            contact_50 = generate_contact(contact_file_path=protein_file_path + ".spotcon", pseq=protein_pseqs[protein_name], window_size=50)

            with open(features_file_path + "_win10.npy", 'wb') as win10_file:
                np.save(file=win10_file, arr=contact_10)

            with open(features_file_path + "_win20.npy", 'wb') as win20_file:
                np.save(file=win20_file, arr=contact_20)

            with open(features_file_path + "_win50.npy", 'wb') as win50_file:
                np.save(file=win50_file, arr=contact_50)

        print(f"\nSPOTCON file is available for {', '.join(spotcon_availables.keys())} proteins.")
        print("Predicting phi and psi angles of these proteins with window features.\n")

        protein_names = list(spotcon_availables.keys())

        for model_name in tqdm(iterable=base_models, desc="Predicting Angles...", ncols=100, unit="model"):
            predictions[model_name] = base_models[model_name].predict(CustomDataLoader(proteins_dict=spotcon_availables, protein_names=protein_names, features_dir_path=features_dir_path, batch_size=1, window_size=base_models_dict[model_name]["window_size"], use_prottrans=base_models_dict[model_name]["use_prottrans"], data_dir_path="./data"))

        avg_prediction = sum(predictions.values()) / len(predictions)

        if not os.path.exists(args.outputs_dir_path):
            os.makedirs(args.outputs_dir_path)

        for index, protein_name in enumerate(protein_names):
            for model_name in predictions:
                prediction = predictions[model_name][index, :spotcon_availables[protein_name]]
                phi_predictions = np.arctan2(prediction[:, 0], prediction[:, 1]) * 180 / np.pi
                psi_predictions = np.arctan2(prediction[:, 2], prediction[:, 3]) * 180 / np.pi
                phi_predictions[0] = psi_predictions[-1] = 360

                outputs = pd.DataFrame({"Amino Acid": list(protein_pseqs[protein_name]), "Phi": phi_predictions, "Psi": psi_predictions})
                outputs.to_csv(args.outputs_dir_path + os.sep + protein_name + '.' + model_name + ".csv", index=False)

            prediction = avg_prediction[index, :spotcon_availables[protein_name]]
            phi_predictions = np.arctan2(prediction[:, 0], prediction[:, 1]) * 180 / np.pi
            psi_predictions = np.arctan2(prediction[:, 2], prediction[:, 3]) * 180 / np.pi
            phi_predictions[0] = psi_predictions[-1] = 360

            outputs = pd.DataFrame({"Amino Acid": list(protein_pseqs[protein_name]), "Phi": phi_predictions, "Psi": psi_predictions})
            outputs.to_csv(args.outputs_dir_path + os.sep + protein_name + '.' + "ensemble8.csv", index=False)

        print("\nDone!\n")

        del predictions
        gc.collect()
    else:
        del window_models
        gc.collect()

    del protein_pseqs, spotcon_availables, additional_layers, base_models_dict, base_models
    gc.collect()
    shutil.rmtree(path=features_dir_path)
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Python script for predicting protein backbone torsion angles with SAINT-Angle.")
    metavar = 'X'

    parser.add_argument("--list", default="./proteins_list", type=str, metavar=metavar, help="Path to text file containing list of proteins (Default: ./proteins_list)", dest="proteins_list_path")
    parser.add_argument("--inputs", default="./inputs", type=str, metavar=metavar, help="Path to directory containing input files (Default: ./inputs)", dest="inputs_dir_path")
    parser.add_argument("--outputs", default="./outputs", type=str, metavar=metavar, help="Path to directory containing output files (Default: ./outputs)", dest="outputs_dir_path")
    parser.add_argument("--gpu", action="store_true", help="Enables gpu usage for ProtTrans features generation (Default: False)", dest="use_gpu")

    parser.set_defaults(use_gpu=False)
    args = parser.parse_args()

    main(args)