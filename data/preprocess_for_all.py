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

start_time = time.time()

# data_names = ['GDSCv2', 'ZINC250k', 'QM9']
data_names = ['kiba']

total_data = [[],[]]

for data_name in data_names:
    
    if data_name == 'GDSCv2':
        max_atoms = 100
        path = 'data/gdscv2.csv'
        smiles_col = 'smiles'
        label_idx = 1 
    elif data_name == 'ZINC250k':
        max_atoms = 100
        path = 'data/zinc250k.csv'
        smiles_col = 'smiles'
        label_idx = 1
    elif data_name == 'QM9':
        max_atoms = 100
        path = 'data/qm9.csv'
        smiles_col = 'SMILES1'
        label_idx = 2
    elif data_name == 'zinc_frags_total_split':
        max_atoms = 100
        path = 'data/zinc_frags_total_split.csv'
        smiles_col = 'SMILES1'
        label_idx = 1
    elif data_name == 'kiba':
        max_atoms = 100
        path = 'data/kiba.csv'
        smiles_col = 'compound_iso_smiles'
        label_idx = 1

    preprocessor = GGNNPreprocessor(out_size=max_atoms, kekulize=True)

    print(f'Preprocessing {data_name} data')
    
    if data_name == 'kiba':
        df = pd.read_csv(path)
        min_, max_ = df['affinity'].min(), df['affinity'].max()
        df['affinity'] = (df['affinity'] - min_) / (max_ - min_)
    else:
        df = pd.read_csv(path, index_col=0)
    
    # Caution: Not reasonable but used in chain_chemistry\datasets\zinc.py:
    # 'smiles' column contains '\n', need to remove it.
    # Here we do not remove \n, because it represents atom N with single bond

    # if data_name == 'GDSCv2':
    #     df = df.drop_duplicates(subset='smiles')
    
    labels = df.keys().tolist()[label_idx:]
    parser = DataFrameParser(preprocessor, labels=labels, smiles_col=smiles_col)
    result = parser.parse(df, return_smiles=True)
    
    
    # if len(result['dataset']._datasets)==3:
    #     node_feature, adj_feature, _ = result['dataset']._datasets
    # elif len(result['dataset']._datasets)==2:
    #     node_feature, adj_feature = result['dataset']._datasets

    # total_data[0].extend(node_feature)
    # total_data[1].extend(adj_feature)
    
    NumpyTupleDataset.save(f'/home/lk/project/mol_generate/GDSS/data/for_all_{data_name}_kekulized.npz', result['dataset'])
    
# import pdb;pdb.set_trace()
    
# dataset = NumpyTupleDataset(total_data)

# # import pdb;pdb.set_trace()
# NumpyTupleDataset.save(f'/home/nas/lk/mol_generate/GDSS_data/qm9_zinc250k_gdscv2_kekulized.npz', dataset)

# NumpyTupleDataset.save(f'/home/nas/lk/mol_generate/gdscv2_GDSS/{data_name.lower()}_kekulized.npz', dataset)
# print('Total time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))


