from keras.models import load_model

import argparse
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

from utils.axils import *
from utils.evaluations import *

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

    print()

    if args.verbose:
        print(f"<SAINT-Angle> Total {len(proteins_list)} proteins provided.")
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

    for model_name in non_window_models:
        base_models[model_name] = load_model("./models" + os.sep + model_name + ".h5", custom_objects=additional_layers)

    if args.verbose:
        print(f"<SAINT-Angle> {', '.join(non_window_models)} loaded.\n")

    del non_window_models
    gc.collect()

    predictions = {}

    if len(spotcon_unavailables) > 0:
        if args.verbose:
            print(f"<SAINT-Angle> SPOTCON file is not available for {', '.join(spotcon_unavailables.keys())} proteins.")
            print("Predicting phi and psi angles of these proteins without window features.")

        for model_name in base_models:
            predictions[model_name] = base_models[model_name].predict(CustomDataLoader(proteins_dict=spotcon_unavailables, inputs_dir_path=args.inputs_dir_path, batch_size=1, window_size=base_models_dict[model_name]["window_size"], use_prottrans=base_models_dict[model_name]["use_prottrans"], use_gpu=args.use_gpu))

        avg_prediction = sum(predictions.values()) / len(predictions)

        if not os.path.exists(args.outputs_dir_path):
            os.makedirs(args.outputs_dir_path)

        for index, protein_name in enumerate(spotcon_unavailables):
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
            outputs.to_csv(args.outputs_dir_path + os.sep + protein_name + '.' + "ensemble.csv", index=False)

        if args.verbose:
            print("Done!\n")

    del spotcon_unavailables, predictions
    gc.collect()

    if len(spotcon_availables) > 0:
        for model_name in window_models:
            base_models[model_name] = load_model("./models" + os.sep + model_name + ".h5", custom_objects=additional_layers)

        if args.verbose:
            print(f"<SAINT-Angle> {', '.join(window_models)} loaded.\n")

        del window_models
        gc.collect()

        predictions = {}

        if args.verbose:
            print(f"<SAINT-Angle> SPOTCON file is available for {', '.join(spotcon_availables.keys())} proteins.")
            print("Predicting phi and psi angles of these proteins with window features.")

        for model_name in base_models:
            predictions[model_name] = base_models[model_name].predict(CustomDataLoader(proteins_dict=spotcon_availables, inputs_dir_path=args.inputs_dir_path, batch_size=1, window_size=base_models_dict[model_name]["window_size"], use_prottrans=base_models_dict[model_name]["use_prottrans"], use_gpu=args.use_gpu))

        avg_prediction = sum(predictions.values()) / len(predictions)

        if not os.path.exists(args.outputs_dir_path):
            os.makedirs(args.outputs_dir_path)

        for index, protein_name in enumerate(spotcon_availables):
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
            outputs.to_csv(args.outputs_dir_path + os.sep + protein_name + '.' + "ensemble.csv", index=False)

        if args.verbose:
            print("Done!\n")

        del predictions
        gc.collect()
    else:
        del window_models
        gc.collect()

    del protein_pseqs, spotcon_availables, additional_layers, base_models_dict, base_models
    gc.collect()
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--list", dest="proteins_list_path", default="./proteins_list", help="path to text file containing list of proteins")
    parser.add_argument("--inputs", dest="inputs_dir_path", default="./inputs", help="path to directory containing input files")
    parser.add_argument("--outputs", dest="outputs_dir_path", default="./outputs", help="path to directory containing output files")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="enables detailed messages printing (Default: False)")
    parser.add_argument("--gpu", dest="use_gpu", action="store_true", help="enables gpu usage for features generation (Default: False)")

    parser.set_defaults(verbose=False)
    parser.set_defaults(use_gpu=False)
    args = parser.parse_args()

    main(args)
    print("<SAINT-Angle> Done!\n")