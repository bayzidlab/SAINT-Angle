# SAINT-Angle
This repository contains the official implementation of our paper titled **SAINT-Angle: self-attention augmented inception-inside-inception network and transfer learning improve protein backbone torsion angle prediction**.



## Introduction



## Guidelines

### Getting Started
In order to run **SAINT-Angle**, create a workspace directory. Download the `inference.py` script and place it inside the workspace directory.



### Downloading Pretrained Model Weights
You will find **here** all the pretrained model weights essential for running **SAINT-Angle**. Download these model weights and place them in a folder named `models` inside the workspace directory. There are, in total, 12 model weights files (with `.h5` extension). The details on each model are given as follows.

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



### Running Inference



### Locating Outputs




## Citation
