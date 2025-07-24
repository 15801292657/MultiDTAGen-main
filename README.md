# MultiDTAGen

## 🔍 Datasets:
### Dataset Source:
All datasets are in the compressed package.

## 🧠 Model Architecture
The MultiDTAGen architecture consists of the following components:

The model processes drugs via parallel GCN 
(graph) and ChemBERTa (SMILES) encoders, and proteins via parallel CNN and ESM (sequence) 
encoders. These features are fused and utilized for two interconnected tasks: 
1) DTA Prediction, where a fully connected network predicts binding affinity
2) Conditional Generation, where a CVAE Transformer decoder generates novel molecules conditioned on the target protein and a desired affinity.

##🛠️ Preprocessing
+ Drugs: The SMILES string representation are converted to the chemical structure using the RDKit library. We then use NetworkX to further convert it to graph representation.
+ Proteins: The protein sequence convert it into a numerical representation using label encoding. Further some more steps preprocessing steps were applied (more detail are provided in the main text).


## System requirements 
+ GPU: GeForce RTX 4090
+ CUDA: 12.1


## ⚙️ Installation and Requirements
You'll need to run the following commands in order to run the codes
```sh
conda env create -f environment.yml  
```
it will download all the required libraries

Or install Manually...
```sh
conda create -n MultiDTAGen python=3.8
conda activate MultiDTAGen
+ python 3.8.11
+ conda install -y -c conda-forge rdkit
+ conda install pytorch torchvision cudatoolkit -c pytorch
```
```sh
pip install torch-cluster==1.6.2+pt21cu118
```
```sh
pip install torch-scatter==2.1.2+pt21cu118
```
```sh
 pip install torch-sparse==0.6.18+pt21cu118
```
```sh
pip install torch-spline-conv==1.2.2+pt21cu118
```
```sh
pip install torch-geometric==2.6.1
```
```sh
pip pip install fairseq==0.10.2
```
```sh
pip pip install einops==0.8.1
```
+ The whole installation maximum takes about 30 minutes.

## 🤖🎛️ Training
The MultiDTAGen is trained using PyTorch and PyTorch Geometric libraries, with the support of NVIDIA GeForce RTX 4090 GPU for the back-end hardware.

i.Create Data
```sh
conda activate MultiDTAGen
python create_data.py
```
The create_data.py script generates four PyTorch-formatted data files from: kiba_train.csv, kiba_test.csv, davis_train.csv, davis_test.csv, metz_train.csv, metz_test.csv, pdbbindv2020_train.csv, pdbbindv2020_test.csv, bindingdb_train.csv, and bindingdb_test.csv and store it data/processed/, consisting of  kiba_train.pt, kiba_test.pt, davis_train.pt, davis_test.pt, metz_train.pt, metz_test.pt, pdbbindv2020_train.pt, pdbbindv2020_test.pt, bindingdb_train.pt, and bindingdb_test.pt.

ii. Train the model 
```sh
conda activate MultiDTAGen
python training.py
```
