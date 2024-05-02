import pickle
import sys
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
sys.path.insert(0,'/home/lk/project/mol_generate/GDSS')
sys.path.insert(0,'/home/lk/project/mol_generate/GDSS/moses/')

from utils.mol_utils import gen_mol, mols_to_smiles, load_smiles, canonicalize_smiles, mols_to_nx, filter_smiles_with_labels, smiles_to_mols
from moses.metrics.metrics import get_all_metrics
from utils.loader import load_ckpt, load_data, load_seed, load_device
from evaluation.stats import eval_graph_list
import torch
import argparse

from parsers.parser import Parser
from parsers.config import get_config

import os
from tqdm import tqdm
import random

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
import logging
import datetime


import csv



def logger_creat(flag):
        
    '''
    logging
    '''


    # 创建一个logger实例
    logger = logging.getLogger(__name__)

    # 设置日志级别
    logger.setLevel(logging.INFO)



    # 创建一个文件处理器，并为文件名添加时间格式
    current_time = datetime.datetime.now().strftime("%Y-%m-%d")
    log_file = f"./evaluation/eval_regression/{flag}_calculation_results_{current_time}.log"
    file_handler = logging.FileHandler(log_file)


    # 设置文件处理器的日志级别
    file_handler.setLevel(logging.INFO)

    # 创建一个格式化器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 将格式化器添加到文件处理器
    file_handler.setFormatter(formatter)

    # 将文件处理器添加到logger
    logger.addHandler(file_handler)

    return logger


def read_smiles_from_txt(file_path):
    smiles_list = []
    with open(file_path, 'r') as file:
        for line in file:
            smiles_list.append(line.strip())  # 去除每行末尾的换行符并添加到列表中
    return smiles_list


def Ours(label):
    import glob
    file_pattern = '/home/lk/project/mol_generate/GDSS/evaluation/eval_regression/datas/*.txt'
    matching_files = glob.glob(file_pattern)
    # file_path = '/home/lk/project/mol_generate/GDSS/evaluation/eval_others/Ours/Apr03-06:33:01_163-sample-1331032-0.35.txt'
    # aim_txt = [read_smiles_from_txt(x) for x in matching_files if str(label['cell']) in x and str(label['ic50']) in x]

    return matching_files

def get_others_gen_mols(config):
    # Ours_ = Ours(config)
    # Ours_, sample_label_list = Ours_[0], Ours_[1]
    return Ours(config)
    # return  data
    


def truncate_dict_values(gen_smiles_dict, max_length=1000):
    truncated_dict = {}
    for key, value in gen_smiles_dict.items():
        if len(value) > max_length:
            truncated_dict[key] = random.sample(value, max_length)
        else:
            truncated_dict[key] = value
    return truncated_dict



def main(work_type_args):

    # 
    
    args = Parser().parse()
    config = get_config(args.config, args.seed)

    
    # device = load_device()
    device = ['cpu']
    ckpt_dict = load_ckpt(config, device)
    configt = ckpt_dict['config']
    
    sample_label_list = get_others_gen_mols(config.controller.label)
 
    sample_label_list_ = {x.split('/')[-1].split('.')[:-1][0].split('-')[-2]+'_'+x.split('/')[-1].split('.')[:-1][1]:read_smiles_from_txt(x) for x in sample_label_list}
    gen_smiles_dict = truncate_dict_values(sample_label_list_, max_length=1000)

    # train_smiles, _ = load_smiles( configt.data.data)
    # train_smiles = canonicalize_smiles(train_smiles)
    

    for new_task in sample_label_list_:
        
        CellLine = int(new_task.split('_')[0])
        IC50 = float('0.'+new_task.split('_')[-1])

        config.controller.label.cell = CellLine
        config.controller.label.ic50 = IC50

        logger = logger_creat(str(CellLine)+"_"+str(IC50))
        logger.info(config)
        
        logger.info([f'{k}:{len(v)}' for k,v in gen_smiles_dict.items()])

        

        # test_topK_df_1 = filter_smiles_with_labels(config, topk=3)
        # test_topK_df_2 = filter_smiles_with_labels(config, topk=5)
        # test_topK_df_3 = filter_smiles_with_labels(config, topk=7)
        # test_topK_df_4 = filter_smiles_with_labels(config, topk=10)
        # test_topK_df_5 = filter_smiles_with_labels(config, topk=15)

        # train_smiles = canonicalize_smiles(train_smiles)

        # test_smiles_1 = canonicalize_smiles(test_topK_df_1['smiles'].tolist())
        # test_smiles_2 = canonicalize_smiles(test_topK_df_2['smiles'].tolist())
        # test_smiles_3 = canonicalize_smiles(test_topK_df_3['smiles'].tolist())
        # test_smiles_4 = canonicalize_smiles(test_topK_df_4['smiles'].tolist())
        # test_smiles_5 = canonicalize_smiles(test_topK_df_5['smiles'].tolist())
        
        # test_topK_df_nx_graphs_1 = mols_to_nx(smiles_to_mols(test_smiles_1))
        # test_topK_df_nx_graphs_2 = mols_to_nx(smiles_to_mols(test_smiles_2))
        # test_topK_df_nx_graphs_3 = mols_to_nx(smiles_to_mols(test_smiles_3))
        # test_topK_df_nx_graphs_4 = mols_to_nx(smiles_to_mols(test_smiles_4))
        # test_topK_df_nx_graphs_5 = mols_to_nx(smiles_to_mols(test_smiles_5))
        

        for key in sample_label_list_:
            if new_task==key:
                continue
            
            import pdb;pdb.set_trace()
            logger.info(f'==========={key}=============')
            gen_smiles = sample_label_list_[key]
            gen_mols = [Chem.MolFromSmiles(smi) for smi in gen_smiles]

            # -------- Evaluation --------
            scores_1 = get_all_metrics(gen=gen_smiles, k=len(gen_smiles), device=device[0], n_jobs=8, test=test_smiles_1, train=train_smiles)
            scores_2 = get_all_metrics(gen=gen_smiles, k=len(gen_smiles), device=device[0], n_jobs=8, test=test_smiles_2, train=train_smiles)
            scores_3 = get_all_metrics(gen=gen_smiles, k=len(gen_smiles), device=device[0], n_jobs=8, test=test_smiles_3, train=train_smiles)
            scores_4 = get_all_metrics(gen=gen_smiles, k=len(gen_smiles), device=device[0], n_jobs=8, test=test_smiles_4, train=train_smiles)
            scores_5 = get_all_metrics(gen=gen_smiles, k=len(gen_smiles), device=device[0], n_jobs=8, test=test_smiles_5, train=train_smiles)
        
            # 
            
            gen_mols_nx = mols_to_nx(gen_mols)
            logger.info("Length of mols_to_nx(gen_mols): %d", len(gen_mols_nx))
            
            scores_nspdk_1 = eval_graph_list(test_topK_df_nx_graphs_1, gen_mols_nx, methods=['nspdk'])['nspdk']
            scores_nspdk_2 = eval_graph_list(test_topK_df_nx_graphs_2, gen_mols_nx, methods=['nspdk'])['nspdk']
            scores_nspdk_3 = eval_graph_list(test_topK_df_nx_graphs_3, gen_mols_nx, methods=['nspdk'])['nspdk']
            scores_nspdk_4 = eval_graph_list(test_topK_df_nx_graphs_4, gen_mols_nx, methods=['nspdk'])['nspdk']
            scores_nspdk_5 = eval_graph_list(test_topK_df_nx_graphs_5, gen_mols_nx, methods=['nspdk'])['nspdk']


            for metric in ['valid', f'unique@{len(gen_smiles)}', 'FCD/Test', 'Novelty']:
                logger.info(f'{metric}: {scores_1[metric]}')
                logger.info(f'{metric}: {scores_2[metric]}')
                logger.info(f'{metric}: {scores_3[metric]}')
                logger.info(f'{metric}: {scores_4[metric]}')
                logger.info(f'{metric}: {scores_5[metric]}')
                
            logger.info(f'NSPDK MMD: {scores_nspdk_1}')
            logger.info(f'NSPDK MMD: {scores_nspdk_2}')
            logger.info(f'NSPDK MMD: {scores_nspdk_3}')
            logger.info(f'NSPDK MMD: {scores_nspdk_4}')
            logger.info(f'NSPDK MMD: {scores_nspdk_5}')
            

                    
            # test_topK_df_1
            # test_topK_df_2
            # test_topK_df_3
            # test_topK_df_4
            # test_topK_df_5
            
            aim_smiles = test_topK_df_1['smiles'].tolist()
            aim_mols = [Chem.MolFromSmiles(smi) for smi in aim_smiles]
            gen_mols = [Chem.MolFromSmiles(smi) for smi in gen_smiles]

            gen_mols = [Chem.MolFromSmiles(smi) for smi in gen_smiles if Chem.MolFromSmiles(smi) is not None]

            gen_fps = [AllChem.GetMorganFingerprintAsBitVect((mol), 2, 1024) for mol in gen_mols]
            aim_fps = [AllChem.GetMorganFingerprintAsBitVect((mol), 2, 1024) for mol in aim_mols]
            max_sims = []
            for i in range(len(gen_fps)):
                sims = DataStructs.BulkTanimotoSimilarity(gen_fps[i], aim_fps)
                max_sims.append(max(sims))
                    
            thresholds = [0.4, 0.3, 0.2, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05]

            results = {x: sum(1 for value in max_sims if value < x) for x in thresholds}
            logger.info(results)
        
    
    


if __name__ == '__main__':

    work_type_parser = argparse.ArgumentParser()
    work_type_parser.add_argument('--type', type=str, required=True)
    work_type_parser.add_argument('--condition', type=str, required=True)
    main(work_type_parser.parse_known_args()[0])
