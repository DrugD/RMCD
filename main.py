import torch
import argparse
import time
from parsers.parser import Parser
from parsers.config import get_config
from trainer import Trainer

from cldr_trainer import CLDRTrainer
from clfg_trainer import CLFGTrainer
from cldt_trainer import CLDTTrainer
from clddi_trainer import CLDDITrainer

from control_trainer_cldr import ControlTrainer as ControlTrainer_cldr
from control_trainer_cldt import ControlTrainer as ControlTrainer_cldt
from control_trainer_cldr_multi_data import MultiDataControlTrainer as MultiDataControlTrainer_cldr
from control_trainer_cldt_multi_data import MultiDataControlTrainer as MultiDataControlTrainer_cldt
from control_trainer_clfg import ControlTrainer as ControlTrainer_clfg

from sampler import Sampler, Sampler_mol, Sampler_mol_condition_cldr, Sampler_mol_condition_clfg, Sampler_mol_condition_cldt

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def main(work_type_args):

    ts = time.strftime('%b%d-%H:%M:%S', time.gmtime())
    args = Parser().parse()
    config = get_config(args.config, args.seed)

    # -------- Train --------
    if work_type_args.type == 'train':
        trainer = Trainer(config) 
        ckpt = trainer.train(ts)
        if 'sample' in config.keys():
            config.ckpt = ckpt
            sampler = Sampler(config) 
            sampler.sample()

    elif work_type_args.type == 'control_drp':
        trainer = ControlTrainer_cldr(config) 
        ckpt = trainer.train(ts)
        
    elif work_type_args.type == 'control_frag':
        trainer = ControlTrainer_clfg(config) 
        ckpt = trainer.train(ts)
    
    elif work_type_args.type == 'control_dta':
        trainer = ControlTrainer_cldt(config) 
        ckpt = trainer.train(ts)
        
    elif work_type_args.type == 'multidata_control_drp':
        trainer = MultiDataControlTrainer_cldr(config) 
        ckpt = trainer.train(ts)
        
    elif work_type_args.type == 'multidata_control_dta':
        trainer = MultiDataControlTrainer_cldt(config) 
        ckpt = trainer.train(ts)
        
    elif work_type_args.type == 'cldr_train':
        trainer = CLDRTrainer(config) 
        ckpt = trainer.train(ts)
        
    elif work_type_args.type == 'clfg_train':
        trainer = CLFGTrainer(config) 
        ckpt = trainer.train(ts)

    elif work_type_args.type == 'cldt_train':
        trainer = CLDTTrainer(config)
        ckpt = trainer.train(ts)
        
    elif work_type_args.type == 'clddi_train':
        trainer = CLDDITrainer(config)
        ckpt = trainer.train(ts)

    # -------- Generation --------
    elif work_type_args.type == 'sample':
        if config.data.data in ['QM9', 'ZINC250k', 'GDSCv2']:
            sampler = Sampler_mol(config)
        else:
            sampler = Sampler(config) 
        sampler.sample()
        
    # -------- Condition Generation --------
    elif work_type_args.type == 'drp_condition_sample':
        if config.data.data in ['GDSCv2', ['GDSCv2', 'QM9']]:
            sampler = Sampler_mol_condition_cldr(config, w=float(work_type_args.condition))
        sampler.sample()
        
    elif work_type_args.type == 'frag_condition_sample':
        if config.data.data in ['zinc_frags_total_split']:
            sampler = Sampler_mol_condition_clfg(config, w=float(work_type_args.condition))
        sampler.sample()
        
    elif work_type_args.type == 'dta_condition_sample':
        if config.data.data in ['davis', ['davis', 'QM9'], ['kiba', 'QM9']]:
            sampler = Sampler_mol_condition_cldt(config, w=float(work_type_args.condition))
        sampler.sample()
    
    
    else:
        raise ValueError(f'Wrong type : {work_type_args.type}')

if __name__ == '__main__':

    work_type_parser = argparse.ArgumentParser()
    work_type_parser.add_argument('--type', type=str, required=True)
    work_type_parser.add_argument('--condition', type=str, required=True)
    main(work_type_parser.parse_known_args()[0])
