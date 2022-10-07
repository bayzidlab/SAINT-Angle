# SAINT-Angle
This repository contains the official implementation of our paper titled **SAINT-Angle: self-attention augmented inception-inside-inception network and transfer learning improve protein backbone torsion angle prediction**.



## Introduction

Protein structure provides valuable insights into how proteins function within living organisms as well as how proteins interact with one another. Prediction of protein backbone torsion angles (φ and ψ), therefore, is a key subproblem in protein structure prediction. However, existing experimental methods for determining backbone torsion angles are costly and time-consuming. Hence, the concerned community is focusing on developing computational methods for predicting torsion angles.

In this paper, we present **SAINT-Angle**, a highly accurate method for protein backbone torsion angles prediction. **SAINT-Angle** uses a self-attention based deep learning architecture called **[SAINT](https://academic.oup.com/bioinformatics/article/36/17/4599/5841663)** which was previously developed in our lab for the protein secondary structure prediction. We extended and improved the existing **SAINT** architecture as well as used transfer learning for predicting the backbone torsion angles. We conducted a thorough analysis and compared the performance of SAINT-Angle with contemporary state-of-the-art prediction methods on a collection of publicly available benchmark datasets, namely, TEST2016, TEST2018, CAMEO, and CASP. The experimental results suggest the notable improvements that our proposed method has achieved over the best alternate methods.




## Guidelines

### Getting Started
In order to run **SAINT-Angle**, create a *workspace* directory. Download the `inference.py` script and place it inside the *workspace* directory.

You may create a separate ***Conda*** environment for running **SAINT-Angle** and install ***TensorFlow (2.6.0 version)*** and ***Keras (2.6.0 version)***. We have tested our code with the aforementioned versions of ***TensorFlow*** and ***Keras***. For faster inference, we recommend install the GPU version of ***TensorFlow***.



### Downloading Pretrained Model Weights
You will find **[here](https://drive.google.com/drive/folders/1RR9TzAhHoTUkzeDTQ40qjh0qB2-eIfse?usp=sharing)** all the pretrained model weights essential for running **SAINT-Angle**. Download these model weights and place them in a folder named `models` inside the *workspace* directory. There are, in total, 12 model weights files (with `.h5` extension). The details on each model are given as follows.

| Model Name | Architecture | Features               |      | Model Name | Architecture | Features               |
| ---------- | ------------ | ---------------------- | ---- | ---------- | ------------ | ---------------------- |
| model_1    | Basic        | Base                   |      | model_7    | Residual     | Base, ProtTrans        |
| model_2    | Basic        | Base, Win10            |      | model_8    | Residual     | Base, Win10, ProtTrans |
| model_3    | ProtTrans    | Base, ProtTrans        |      | model_9    | Basic        | ESIDEN                 |
| model_4    | ProtTrans    | Base, Win10, ProtTrans |      | model_10   | Basic        | ESIDEN, HMM            |
| model_5    | ProtTrans    | Base, Win20, ProtTrans |      | model_11   | ProtTrans    | ESIDEN, HMM, ProtTrans |
| model_6    | ProtTrans    | Base, Win50, ProtTrans |      | model_12   | Residual     | ESIDEN, HMM, ProtTrans |

Here, ***Base*** means the feature set consisting of *PSSM*, *HMM*, and *PCP* features from **[SPOT-1D](https://academic.oup.com/bioinformatics/article/35/14/2403/5232996)**. ***Win10***, ***Win20***, and ***Win50*** mean the *window* features from **[SPOT-Contact](https://academic.oup.com/bioinformatics/article/34/23/4039/5040307)**. ***ProtTrans*** means the *extracted* features from **[ProtTrans](https://ieeexplore.ieee.org/document/9477085)** **[[Github]](https://github.com/agemagician/ProtTrans/blob/master/Embedding/PyTorch/Advanced/ProtT5-XL-UniRef50.ipynb)**. ***ESIDEN*** means the feature set consisting of *PSSM*, *AA*, *PCP*, *DC*, *RE*, *PSSP*, and *RBP* features from **[ESIDEN](https://www.nature.com/articles/s41598-021-00477-2)**.

In order to predict backbone torsion angles using **SAINT-Angle**, you may use any of the given models. Also, there are 2 ensembles of different combinations of given models available in **SAINT-Angle** for prediction purpose. The details on each ensemble are given as follows.

| Ensemble Name | Base Models Used                                             |
| ------------- | ------------------------------------------------------------ |
| ensemble_3    | model_10, model_11, model_12                                 |
| ensemble_8    | model_1, model_2, model_3, model_4, model_5, model_6, model_7, model_8 |

By providing the appropriate command line argument (discussed in the subsequent section), you may use the aforementioned (ensemble of) models for torsion angles prediction.



### Preparing Input Features



### Running Inference for Torsion Angles Prediction

1. Open up a terminal window and navigate to your *workspace* directory using `cd` command (if necessary).

2. To run **SAINT-Angle** for predicting protein backbone torsion angles, type in the terminal the following command.

    > python inference.py

    This command runs the `inference.py` with default arguments using `models/model_1.h5` as model and `datasets/TEST2016_A` as dataset.

3. You may change the model that you will use for the inference by providing the model parameter.

    `-m MODEL_NAME` or `--model MODEL_NAME` sets the inference script to use either `models/MODEL_NAME.h5` model (in case of inference with single model) or `MODEL_NAME` ensemble model (in case of inference with ensemble of models) for the prediction. You must use one of the following `MODEL_NAME` to specify the model that you want use for the inference.

    |         |         |          |          |          |            |            |
    | ------- | ------- | -------- | -------- | -------- | ---------- | ---------- |
    | model_1 | model_2 | model_3  | model_4  | model_5  | model_6    | model_7    |
    | model_8 | model_9 | model_10 | model_11 | model_12 | ensemble_3 | ensemble_8 |

4. You may change the dataset that you will use for the inference by providing the dataset parameter.

    `-d DATASET_NAME` or `--dataset DATASET_NAME` sets the inference script to use features from the `datasets/DATASET_NAME` dataset to predict the backbone torsion angles of protein(s) belonging to `DATASET_NAME` dataset.

5. `--verbose` sets the inference script to print out detailed messages and `--output` sets the inference script to output predicted angles in `predictions` folder inside *workspace* directory (discussed in the subsequent section).

6. `-h` or `--help` sets the inference script to display help message.



### Locating Outputs




## Citation
