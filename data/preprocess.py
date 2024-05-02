### Original code from MoFlow (under MIT License) https://github.com/calvin-zcx/moflow
import os
import sys
sys.path.insert(0, os.getcwd())
import pandas as pd
import argparse
import time
from utils.data_frame_parser import DataFrameParser
from utils.numpytupledataset import NumpyTupleDataset
from utils.smile_to_graph import GGNNPreprocessor
import pdb

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', type=str, default='ZINC250k', choices=['ZINC250k', 'QM9', 'GDSCv2', 'GDSCv2_SMALL'])
args = parser.parse_args()

start_time = time.time()
data_name = args.dataset

if data_name == 'ZINC250k':
    max_atoms = 38
    path = 'data/zinc250k.csv'
    smiles_col = 'smiles'
    label_idx = 1
elif data_name == 'QM9':
    max_atoms = 9
    path = 'data/qm9.csv'
    smiles_col = 'SMILES1'
    label_idx = 2
elif data_name == 'GDSCv2':
    max_atoms = 100
    path = 'data/gdscv2.csv'
    smiles_col = 'smiles'
    label_idx = 1
elif data_name == 'GDSCv2_SMALL':
    max_atoms = 96
    path = 'data/gdscv2_small.csv'
    smiles_col = 'smiles'
    label_idx = 1
else:
    raise ValueError(f"[ERROR] Unexpected value data_name={data_name}")

preprocessor = GGNNPreprocessor(out_size=max_atoms, kekulize=True)

print(f'Preprocessing {data_name} data')
df = pd.read_csv(path, index_col=0)
# Caution: Not reasonable but used in chain_chemistry\datasets\zinc.py:
# 'smiles' column contains '\n', need to remove it.
# Here we do not remove \n, because it represents atom N with single bond

import pdb;pdb.set_trace()

labels = df.keys().tolist()[label_idx:]
parser = DataFrameParser(preprocessor, labels=labels, smiles_col=smiles_col)
result = parser.parse(df, return_smiles=True)

dataset = result['dataset']
smiles = result['smiles']
# import pdb;pdb.set_trace()

# NumpyTupleDataset.save(f'data/{data_name.lower()}_kekulized.npz', dataset)
NumpyTupleDataset.save(f'data/{data_name.lower()}_Small_DRP_kekulized.npz', dataset)

# NumpyTupleDataset.save(f'/home/nas/lk/mol_generate/gdscv2_GDSS/{data_name.lower()}_kekulized.npz', dataset)
print('Total time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
