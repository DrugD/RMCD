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


# ,smiles,logP,qed,SAS

Data_response_file = "PANCANCER_IC.csv"
Data_smiles_file = "drug_smiles.csv"
Data_cell_file = "PANCANCER_Genetic_feature.csv"

def save_csv(data):
   
    data = pd.DataFrame(data, columns=["smiles","cell_name","ic50"])
    # data.to_csv("gdscv2.csv")
    # pdb.set_trace()

    # return None


def save_cell_mut_matrix():
    f = open(Data_cell_file)
    reader = csv.reader(f)
    next(reader)
    features = {}
    cell_dict = {}
    mut_dict = {}
    matrix_list = []

    for item in reader:
        cell_id = item[1]
        mut = item[5]
        is_mutated = int(item[6])

        if mut in mut_dict:
            col = mut_dict[mut]
        else:
            col = len(mut_dict)
            mut_dict[mut] = col

        if cell_id in cell_dict:
            row = cell_dict[cell_id]
        else:
            row = len(cell_dict)
            cell_dict[cell_id] = row
        if is_mutated == 1:
            matrix_list.append((row, col))
    
    cell_feature = np.zeros((len(cell_dict), len(mut_dict)))

    for item in matrix_list:
        cell_feature[item[0], item[1]] = 1

    return cell_dict, cell_feature

def load_drug_smile():
    reader = csv.reader(open(Data_smiles_file))
    next(reader, None)

    drug_dict = {}
    
    
    for item in reader:
        name = item[0]
        smile = item[3]
        
       
        if name in drug_dict:
            continue
        else:
            drug_dict[name] = smile
            
    
    return drug_dict

def load_drug_cell_response():
    f = open(Data_response_file)
    reader = csv.reader(f)
    next(reader)

    cell_dict, cell_features = save_cell_mut_matrix()
    drug_smiles = load_drug_smile()
    
    nei_atom = []
    temp_data = []
    max_num_atom = -1
    for item in tqdm(reader):
        if drug_smiles.get(item[0]) and cell_dict.get(item[3]):
            drug_smile = drug_smiles[item[0]]
            
            mol = Chem.MolFromSmiles(drug_smile)
            canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=False,
                                                canonical=True)
            mol = Chem.MolFromSmiles(canonical_smiles)
            Chem.Kekulize(mol)
            num_atoms = mol.GetNumAtoms()
            
            nei_atom_ = [x for x in set([i.GetAtomicNum() for i in mol.GetAtoms()])]
                
                #    
                
            nei_atom.extend(nei_atom_)
            nei_atom = [i for i in set(nei_atom)]
      
            if num_atoms > max_num_atom:
                max_num_atom = num_atoms
            
            
            cell_name = item[3]
            # cell_feature = list(cell_features[cell_dict[item[3]]])
            ic50 = item[8]
            ic50 = 1 / (1 + pow(math.exp(float(ic50)), -0.1))
            temp_data.append((drug_smile, cell_name, ic50))
            
    print("Max atom number is ",max_num_atom)
    print("Atoms class is ",len(nei_atom),"\n", nei_atom)
    # pdb.set_trace()
    return temp_data

datas = load_drug_cell_response()
save_csv(datas)