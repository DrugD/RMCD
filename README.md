# Regressor-free Molecule Generation to support Drug Response Prediction (Regressor-free guidance)

Official Code Repository for the paper 'Regressor-free Molecule Generation to support Drug Response Prediction'.



## Contribution

+ Regressor-free guidance molecule generation is proposed to ensure sampling within a more effective space, where the regression controller model can encode the response value between the molecules and the cell lines as conditional guidance for molecule generation.
+ To enhance noise prediction performance, we introduce a dual-branch controlled noise prediction model for score estimation, named DBControl. DBControl model consists of two GNN-based branches, each undergoing unconditional training and conditional mixed training, respectively.
+ Experimental results demonstrate that our method outperforms the state-of-the-art baselines in conditional molecular graph generation for the DRP task. Furthermore, we provide an effectiveness proof of our method.

## Dependencies

Regressor-free guidance is built in **Python 3.8.18** and **Pytorch 2.0.1**. Please use the following command to install the requirements:

```sh
pip install -r requirements.txt
```

For molecule generation, additionally run the following command:

```sh
conda install -c conda-forge rdkit=2020.09.1.0
```


## Running Experiments


### 1. Preparations

We provide two **molecular graph datasets** (QM9 and ZINC250k) and one **drug response dataset** (GDSCv2). 

We additionally provide the commands for generating generic graph datasets as follows:

```sh
python data/data_generators.py --dataset ${dataset_name}
```

To preprocess the molecular graph datasets for training models, run the following command:

```sh
python data/preprocess.py --dataset ${dataset_name}
python data/preprocess_for_nspdk.py --dataset ${dataset_name}
```

For the evaluation of generic graph generation tasks, run the following command to compile the ORCA program (see http://www.biolab.si/supp/orca/orca.html):

```sh
cd evaluation/orca 
g++ -O2 -std=c++11 -o orca orca.cpp
```


### 2. Configurations

The configurations are provided on the `config/` directory in `YAML` format. 
Hyperparameters used in the experiments are specified in the Appendix C of our paper.


### 3. Training Uncondition Model

We provide the commands for the following task: Molecule Generation.

To train the score models, first modify `config/${dataset}.yaml` accordingly, then run the following command.

```sh
CUDA_VISIBLE_DEVICES=${gpu_ids} python main.py --type train --config ${train_config} --seed ${seed}
```

for example, 

```sh
CUDA_VISIBLE_DEVICES=0 python main.py --type train --config community_small --seed 42
```
and
```sh
CUDA_VISIBLE_DEVICES=0,1 python main.py --type train --config zinc250k --seed 42
```

### 4. Training Condition Contrastive Learning Model

We provide the commands for the following task: Molecule Generation.

To train the score models, first modify `config/$cl{task}_main.yaml` accordingly, such as `dr` for `drug response`, then run the following command.

```sh
CUDA_VISIBLE_DEVICES=${gpu_ids} python main.py --type cl{task}_train --config ${train_config} --seed ${seed}
```

for example, 

```sh
CUDA_VISIBLE_DEVICES=0 python main.py --type cldr_train --config cldr_train --seed 42
```
and
```sh
CUDA_VISIBLE_DEVICES=0,1 python main.py --type cldt_train --config cldt_train --seed 42
```


### 5. Training Condition  Model

We provide the commands for the following task: Molecule Generation.

To train the score models, first modify `config_control/{task}_train{_multi_data}.yaml` accordingly, such as `drp` for `drug response`, `_multi_data` is default， then run the following command.

```sh
CUDA_VISIBLE_DEVICES=${gpu_ids} python main.py --type train --config config_control/${train_config} --seed ${seed}
```

for example, 

```sh
CUDA_VISIBLE_DEVICES=0 python main.py --type control_drp --config config_control/drp_train --seed 42
```

```sh
CUDA_VISIBLE_DEVICES=0 python main.py --type multidata_control_drp --config config_control/drp_train --seed 42
```
and
```sh
CUDA_VISIBLE_DEVICES=0,1 python main.py --type multidata_control_dta --config config_control/dta_train --seed 42
```

### 6. Generation and Evaluation

To generate graphs using the trained score models, run the following command.

```sh
CUDA_VISIBLE_DEVICES=${gpu_ids} python main.py --type {task}_condition_sample --config sample_{dataset}
```
or

```sh
CUDA_VISIBLE_DEVICES=${gpu_ids} python main.py --type sample --config sample_zinc250k
```


## Pretrained checkpoints

We provide checkpoints of the pretrained models on the `checkpoints/` directory, which are used in the main experiments.

+ `['GDSCv2', 'QM9']/date-time.pth` 
+ `QM9/qm9.pth`
+ `ZINC250k/zinc250k.pth` 

We have uploaded a zip pack of code that can be used for conditional sampling directly, which can be downloaded (https://drive.google.com/file/d/15He7YHE39wT9tdNcqSOPR-bvzvO_Isr9/view?usp=drive_link).

```sh
cd ./RFMG_Sampling
CUDA_VISIBLE_DEVICES=${gpu_ids} python main.py
```