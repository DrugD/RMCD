import os
import csv
import pdb
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import numpy

from rdkit import Chem
from rdkit.Chem import rdmolops
import json
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles


# ,smiles,logP,qed,SAS

dataset = 'kiba'
Data_DTA_file = f"./data/{dataset}.csv"
Data_smiles_file = f"./data/{dataset}/ligands_can.txt"
Data_target_file = f"./data/{dataset}/proteins.txt"

def save_json(data):
    
    import random, json
    # 生成10个[1,100)的随机整数
    valid_idx = random.sample(range(0, len(data)), int(len(data)*0.1))
    dict_save = dict(valid_idxs=valid_idx)
    dict_save_ = json.dumps(dict_save)
    f2 = open(f'./data/valid_idx_{dataset}.json', 'w')
    f2.write(dict_save_)
    f2.close()

    pdb.set_trace()

    # return None


def save_target_mut_matrix():
    f = open(Data_target_file)
    reader = csv.reader(f)
    next(reader)
    features = {}
    target_dict = {}
    mut_dict = {}
    matrix_list = []

    for item in reader:
        target_id = item[1]
        mut = item[5]
        is_mutated = int(item[6])

        if mut in mut_dict:
            col = mut_dict[mut]
        else:
            col = len(mut_dict)
            mut_dict[mut] = col

        if target_id in target_dict:
            row = target_dict[target_id]
        else:
            row = len(target_dict)
            target_dict[target_id] = row
        if is_mutated == 1:
            matrix_list.append((row, col))
    
    target_feature = np.zeros((len(target_dict), len(mut_dict)))

    for item in matrix_list:
        target_feature[item[0], item[1]] = 1

    return target_dict, target_feature

def load_drug_smile():
    # reader = csv.reader(open(Data_smiles_file))
    # next(reader, None)
    ligands = json.load(open(f"./data/{dataset}/ligands_can.txt"), object_pairs_hook=OrderedDict)

    drug_dict = {}
    
    max_num_atom = -1
    for key, value in ligands.items():
        name = key 
        smile = Chem.MolToSmiles(Chem.MolFromSmiles(value),isomericSmiles=True)
        
        mol = Chem.MolFromSmiles(smile)
        canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=False,
                                            canonical=True)
        mol = Chem.MolFromSmiles(canonical_smiles)
        Chem.Kekulize(mol)
        num_atoms = mol.GetNumAtoms()
        
        if num_atoms > max_num_atom:
            max_num_atom = num_atoms
        if name in drug_dict:
            continue
        else:
            drug_dict[name] = smile
            
    print("Max atom number is ",max_num_atom)
    return drug_dict

def load_drug_target_response():
    f = open(Data_DTA_file)
    reader = csv.reader(f)
    next(reader)

    # target_dict, target_features = save_target_mut_matrix()
    target_dict = json.load(open(f"./data/{dataset}/proteins.txt"), object_pairs_hook=OrderedDict)
    drug_smiles = load_drug_smile()
    
    temp_data = []
    
    for item in tqdm(reader):
        # pdb.set_trace()
        if target_dict.get(list(target_dict.keys())[int(item[1])]):
            drug_smile = item[0]
            # target_name = item[3]
            # target_feature = list(target_features[target_dict[item[3]]])
            target_id = item[1]
            ic50 = item[2]
            # ic50 = 1 / (1 + pow(math.exp(float(ic50)), -0.1))
            temp_data.append((drug_smile, target_id, ic50))
        
        
    return temp_data

datas = load_drug_target_response()
save_json(datas)