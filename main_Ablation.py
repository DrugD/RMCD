import torch
import argparse
import time
from parsers.parser import Parser
from parsers.config import get_config

from control_trainer_cldr_multi_data_ablation_0 import MultiDataControlTrainer as MultiDataControlTrainer_cldr_0


from sampler import Sampler_mol_condition_cldr

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def main(work_type_args):

    ts = time.strftime('%b%d-%H:%M:%S', time.gmtime())
    args = Parser().parse()
    config = get_config(args.config, args.seed)

    # -------- Train --------
    if work_type_args.type  == 'multidata_control_drp_0':
        trainer = MultiDataControlTrainer_cldr_0(config) 
        ckpt = trainer.train(ts)

    else:
        raise ValueError(f'Wrong type : {work_type_args.type}')

if __name__ == '__main__':

    work_type_parser = argparse.ArgumentParser()
    work_type_parser.add_argument('--type', type=str, required=True)
    work_type_parser.add_argument('--condition', type=str, required=True)
    main(work_type_parser.parse_known_args()[0])
