import pandas as pd

df = pd.read_csv('./zinc250k.csv')
smiles_list = df['smiles']

from rdkit import Chem
from rdkit.Chem import Recap
from itertools import combinations
from tqdm import tqdm
i = 0
with open('zinc_frags_total_split.csv', 'w') as f:
    f.write(',SMILES1,SMILES2'+'\n')
    for smiles in tqdm(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        fragments = Recap.RecapDecompose(mol).GetLeaves().keys()
      
        if len(fragments)>=2:
            f.write(f"{i},{smiles},{'?'.join(fragments)}\n")
            i += 1

import pandas as pd
import json
import random

data = pd.read_csv("zinc_frags_total_split.csv")

total_rows = len(data)

num_samples = int(total_rows * 0.2)

random_idxs = random.sample(range(total_rows), num_samples)

with open("valid_idx_zinc_frags_total_split.json", "w") as f:
    json.dump({"valid_idxs": [str(idx).zfill(6) for idx in random_idxs]}, f)


import pandas as pd
import json
import random

data = pd.read_csv("zinc_frags_total_split.csv")
print(max([len(x) for x in data['SMILES2']]))