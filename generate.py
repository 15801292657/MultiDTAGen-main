import argparse
import pickle
from pathlib import Path
import pandas as pd
import rdkit

import torch
from rdkit import RDLogger
from rdkit.Chem import MolFromSmiles
from tqdm.auto import tqdm
from model import MultiDTAGen
from torch.utils.data import DataLoader
from rdkit import Chem

from utils import *

import sys
print(sys.executable)

RDLogger.DisableLog('rdApp.*')

def load_model(model_path, tokenizer_path, device):
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    model = MultiDTAGen(tokenizer)
    saved_state_dict = torch.load(model_path, map_location=device)
    model_dict = model.state_dict()
    # 过滤掉不匹配的键
    filtered_state_dict = {k: v for k, v in saved_state_dict.items() if k in model_dict}
    model_dict.update(filtered_state_dict)
    model.load_state_dict(model_dict)
    return model, tokenizer

def format_smiles(smiles):
    mol = MolFromSmiles(smiles)
    if mol is None:
        return None

    smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    return smiles

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, choices=['kiba', 'bindingdb', 'davis', 'metz', 'pdbbindv2020'], help='the dataset name (kiba or davis)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='device to use (cpu or cuda)')
    args = parser.parse_args()
    dataset = args.dataset
    device = args.device

    config = {
        'input_path': 'your_input_path',
        'output_dir': 'your_output_directory',
        'model_path': 'your_model_weights_path.pth',
        'tokenizer_path': 'your_tokenizer_path.pkl',
        'n_mol': 40000,
        'filter': True,
        'batch_size': 64,
        'seed': -1,
        'device': device
    }

    model_path = f'saved_models/multidtagen_model_{dataset}.pth'
    tokenizer_path = f'data/{dataset}_tokenizer.pkl'

    test_data = TestbedDataset(root="data", dataset=f"{dataset}_test")
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    output_dir = Path(f'generated_results/{dataset}')
    output_dir.mkdir(parents=False, exist_ok=True)

    model, tokenizer = load_model(model_path, tokenizer_path, device)

    model.eval()
    model.to(device)

    output_path = output_dir / 'generated_smiles.txt'
    with open(output_path, 'a') as output_file:
        n_epoch = (config['n_mol'] + config['batch_size'] - 1) // config['batch_size']
        generated_smiles = []

        for i, data in enumerate(tqdm(test_loader)):
            data = data.to(device)
            res = tokenizer.get_text(model.generate(data))
            generated_smiles.extend(res)

        generated_smiles = generated_smiles[:config['n_mol']]

        if config['filter']:
            generated_smiles = [format_smiles(smiles) for smiles in generated_smiles]
            generated_smiles = [smiles for smiles in generated_smiles if smiles]
            generated_smiles = list(set(generated_smiles))

        output_file.write('\n'.join(generated_smiles) + '\n')

    print('Generation complete')