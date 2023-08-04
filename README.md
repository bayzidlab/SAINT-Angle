# SAINT-Angle

This repository contains the standalone version (command line interface/CLI tool) of the method proposed in our paper titled **SAINT-Angle: self-attention augmented inception-inside-inception network and transfer learning improve protein backbone torsion angle prediction**, published in the journal **[Bioinformatics Advances (Oxford Academic)](https://academic.oup.com/bioinformaticsadvances)**.

If any content in this repository is used for research or other purpose, we shall be obliged if you kindly cite *[**our paper**](https://doi.org/10.1093/bioadv/vbad042)*.



## Guidelines

### Getting Started

In order to run **SAINT-Angle** on your local machine, navigate to a *workspace* directory and run the following commands in a terminal.

```sh
git clone https://github.com/bayzidlab/SAINT-Angle.git
cd SAINT-Angle/saint_angle
```

Then, create a separate ***Conda*** environment for running **SAINT-Angle** with ***Python*** version `3.7`. Install ***TensforFlow*** version `2.6.0`, ***Keras*** version `2.6.0`, and ***pandas*** version `1.3.5`. We tested our method with aforementioned versions of ***TensorFlow*** and ***Keras***. We recommend you install *GPU* version of ***TensorFlow*** for faster inference. Besides, run the following command in a terminal to install ***SentencePiece*** and ***transformers***.

```sh
pip install -q SentencePiece transformers
```



### Downloading Pretrained Model Weights

Pretrained model weights essential for running **SAINT-Angle** can be found ***[here](https://drive.google.com/drive/folders/1OLZ2mFbFhIER-JFxf24snxdUPyt5EiZ6?usp=sharing)***. Download these model weights files and place them in a folder named `models` inside `SAINT-Angle/saint_angle` directory. There are 8 (eight) model weights files with `.h5` extension, each of which corresponding to one of the base models used in ensemble of **SAINT-Angle**. A short description of each base model is provided in the below table.

| Model Name | Architecture | Features                  |
| ---------- | ------------ | ------------------------- |
| model_1    | Basic        | Base                      |
| model_2    | Basic        | Base, Window10            |
| model_3    | ProtTrans    | Base, ProtTrans           |
| model_4    | ProtTrans    | Base, Window10, ProtTrans |
| model_5    | ProtTrans    | Base, Window20, ProtTrans |
| model_6    | ProtTrans    | Base, Window50, ProtTrans |
| model_7    | Residual     | Base, ProtTrans           |
| model_8    | Residual     | Base, Window10, ProtTrans |

For details on different architectures and features, we request you to go through ***[our paper](https://doi.org/10.1093/bioadv/vbad042)***.



### Preparing Input Features

Follow any of the following 2 (two) options to prepare input proteins list and corresponding feature files before running **SAINT-Angle**.

#### Option-1

List the name of input proteins in a file named `proteins_list` and place this file inside `SAINT-Angle/saint_angle` directory. Besides, place the `.fasta`, `.pssm`, `.hhm`, and `.spotcon` files of each input protein in a folder named `inputs` located inside `SAINT-Angle/saint_angle` directory. A `.spotcon` file should contain the contact map of an input protein in ***CASP*** format.

#### Option-2

If you already have a text file containing a list of names of input proteins (one name in each line) and a folder containing the `.fasta`, `.pssm`, `.hhm`, and `.spotcon` files of each input protein, you can simply provide the corresponding paths when running **SAINT-Angle** (discussed in next section).

***Keep in mind that, some of the base models of SAINT-Angle require contact map (window) features of input proteins to predict their torsion φ and ψ angles. If you do not provide `spotcon` file for a particular input protein, then SAINT-Angle will use, in its ensemble, only the base models that do not require window features, in order to predict its torsion angles.***



### Running SAINT-Angle

In order to run **SAINT-Angle** for protein backbone φ and ψ angles prediction, run one of the following commands in a terminal from `SAINT-Angle/saint_angle` directory.

#### To see available command line parameters

```sh
python saint_angle.py -h OR python saint_angle.py --help
```

#### When input features prepared following Option-1

```sh
python saint_angle.py
```

#### When input features prepared following Option-2

```sh
python saint_angle.py --list PATH_TO_PROTEINS_LIST --inputs PATH_TO_INPUT_FILES_FOLDER
```



### Understanding Output Files

**SAINT-Angle** will generate a folder named `outputs` inside `SAINT-Angle/saint_angle` directory *(if not specified otherwise via command line argument)* and place all the output files in this folder. For each input protein, you can expect either 4 (four) or 9 (nine) output `.csv` files inside the output folder. A short description of each output file format is provided in the below table.

| **For Protein without `.spotcon` File** |                               |
| --------------------------------------- | ----------------------------- |
| **Output File Format**                  | **Model Used for Prediction** |
| `PROTEIN_NAME.model_1.csv`              | model_1                       |
| `PROTEIN_NAME.model_3.csv`              | model_3                       |
| `PROTEIN_NAME.model_7.csv`              | model_7                       |
| `PROTEIN_NAME.ensemble3.csv`            | *ensemble* of above 3 models  |
| **For Protein with `.spotcon` File**    |                               |
| **Output File Format**                  | **Model Used for Prediction** |
| `PROTEIN_NAME.model_1.csv`              | model_1                       |
| `PROTEIN_NAME.model_2.csv`              | model_2                       |
| `PROTEIN_NAME.model_3.csv`              | model_3                       |
| `PROTEIN_NAME.model_4.csv`              | model_4                       |
| `PROTEIN_NAME.model_5.csv`              | model_5                       |
| `PROTEIN_NAME.model_6.csv`              | model_6                       |
| `PROTEIN_NAME.model_7.csv`              | model_7                       |
| `PROTEIN_NAME.model_8.csv`              | model_8                       |
| `PROTEIN_NAME.ensemble8.csv`            | *ensemble* of above 8 models  |

The φ angle of first amino acid residue and the ψ angle of last amino acid residue for each input protein are masked with 360&deg; angle.



## Data Availability

Backbone torsion φ and ψ angles of proteins from a recent benchmark dataset, used to evaluate the performance of **SAINT-Angle**, can be found ***[here](https://drive.google.com/drive/folders/19s8KXsilnqdWesy-98oweCFEP-rjPW2S?usp=sharing)***.



## Citation

A K M Mehedi Hasan, Ajmain Yasar Ahmed, Sazan Mahbub, M Saifur Rahman, and Md Shamsuzzoha Bayzid, SAINT-Angle: self-attention augmented inception-inside-inception network and transfer learning improve protein backbone torsion angle prediction, Bioinformatics Advances, Vol. 3(1): vbad042, https://doi.org/10.1093/bioadv/vbad042, 2023.



## BibTeX

```latex
@article{hasan2023saint,
  title={SAINT-Angle: self-attention augmented inception-inside-inception network and transfer learning improve protein backbone torsion angle prediction},
  author={Hasan, AKM Mehedi and Ahmed, Ajmain Yasar and Mahbub, Sazan and Rahman, M Saifur and Bayzid, Md Shamsuzzoha},
  journal={Bioinformatics Advances},
  volume={3},
  number={1},
  pages={vbad042},
  year={2023},
  publisher={Oxford University Press}
}
```