import os
import time
from tqdm import tqdm, trange
import numpy as np
import torch

from utils.loader import load_seed, load_device, load_data, load_model_params, load_model_optimizer, \
                         load_ema, load_loss_fn, load_batch
from utils.logger import Logger, set_log, start_log, train_log
from controller import TransEDRP
import pdb
from models.transE import TransE
from tqdm import tqdm

class CLDRTrainer(object):
    def __init__(self, config):
        super(CLDRTrainer, self).__init__()

        self.config = config
        self.log_folder_name, self.log_dir, self.ckpt_dir = set_log(self.config)

        self.seed = load_seed(self.config.seed)
        self.device = load_device()

        self.params_x, self.params_adj = load_model_params(self.config)
        self.transe = TransE(1000, 1, f'cuda:{self.device[0]}', dim=128)
        self.model = TransEDRP(self.config.controller, self.params_x, self.params_adj, f'cuda:{self.device[0]}', self.config.train)


        # load uncondition_model for CLDR
        
        self.un_C_model_pth = './checkpoints/QM9/Mar10-01:49:59.pth'
        
        self.load_uncondition_model_params()
        
        self.model.to(f'cuda:{self.device[0]}')
        self.model.get_cell_matrix(self.config.controller.cell_csv_path)
        self.transe = self.transe.to(f'cuda:{self.device[0]}')

        
        self.train_loader, self.test_loader = load_data(self.config)
    
    def load_uncondition_model_params(self):
        un_C_model = torch.load(self.un_C_model_pth)
        
        un_C_model_x = un_C_model['x_state_dict']
        un_C_model_adj = un_C_model['adj_state_dict']
        
        model_dict =  self.model.state_dict()
        
        self.model_x_dict = [x  for x in model_dict.keys() if 'drug_x_module' in x]
        self.model_adj_dict = [x  for x in model_dict.keys() if 'drug_adj_module' in x]
        
        target_dict = {}
        for k in self.model_x_dict:
            target_dict[k] = un_C_model_x[k.split('drug_x_module')[1:][0][1:]]
            target_dict[k].requires_grad = False
            
        for k in self.model_adj_dict:
            target_dict[k] = un_C_model_adj[k.split('drug_adj_module')[1:][0][1:]]
            target_dict[k].requires_grad = False
            
        model_dict.update(target_dict)
        self.model.load_state_dict(model_dict)
        
        
        
    def train(self, ts):
   
        self.config.exp_name = ts
        self.ckpt = f'{ts}'
        print('\033[91m' + f'{self.ckpt}' + '\033[0m')
        
        optimizer = torch.optim.Adam([
                        {'params': self.model.parameters()},
                        # {'params': self.transe.parameters()}
                    ], lr=self.config.train.lr, weight_decay=0.05)

        cross_entropy_loss = torch.nn.CrossEntropyLoss()

    
        logger = Logger(str(os.path.join(self.log_dir, f'{self.ckpt}.log')), mode='a')
        logger.log(f'{self.ckpt}', verbose=False)
        start_log(logger, self.config)
        train_log(logger, self.config)

        self.loss_fn = load_loss_fn(self.config)

        # -------- Training --------
        for epoch in trange(0, (self.config.train.num_epochs), desc = '[Epoch]', position = 1, leave=False):

            self.train_TransE = []
            self.train_CLIP = []
            self.train_CLIP_Num = []
            self.test_TransE = []
            self.test_CLIP = []
            self.test_CLIP_Num = []
            t_start = time.time()

            self.model.train()

            for _, train_b in tqdm(enumerate(self.train_loader)):
                optimizer.zero_grad()
                x, adj, label = load_batch(train_b, self.device) 
                cell_name, ic50 = label[:,:,0], label[:,:,1]
                
          
                logits_per_dc, logits_per_text, num_logits_per_dc, num_logits_per_text = self.model.forward4gen(x, adj, cell_name, ic50)
                
                number_features = self.model.generate_samples(0, 1000)
                
                loss_TransEs = self.transe(number_features,number_features)
                
                loss_TransE = torch.sum(loss_TransEs[0])+torch.sum(loss_TransEs[1])+torch.sum(loss_TransEs[2])
          
                labels = torch.arange(ic50.shape[0]).long().to(f'cuda:{self.device[0]}')

                loss_dc = cross_entropy_loss(logits_per_dc, labels)

                loss_t = cross_entropy_loss(logits_per_text, labels)
                
                loss_dc_num = cross_entropy_loss(num_logits_per_dc, labels)

                loss_t_num = cross_entropy_loss(num_logits_per_text, labels)
                
                loss_CLIP = (loss_dc + loss_t)/2 
                
                loss_CLIP_Num = (loss_dc_num + loss_t_num)/2 
             
                loss = loss_TransE*0.01 + loss_CLIP * 0.19 + loss_CLIP_Num * 0.8
      
                loss.backward()
                optimizer.step()
                
                # Print the weights of self.transe
                # print("Weights of self.transe:")
                # for name, param in self.transe.state_dict().items():
                #     print(name, param.sum())
                    
                self.train_TransE.append(loss_TransE.item())
                self.train_CLIP.append(loss_CLIP.item())
                self.train_CLIP_Num.append(loss_CLIP_Num.item())
                    
            self.model.eval()
            
            with torch.no_grad():
                for _, test_b in enumerate(self.test_loader):   

                    x, adj, label = load_batch(test_b, self.device) 
                    cell_name, ic50 = label[:,:,0], label[:,:,1]
                    
                    logits_per_dc, logits_per_text, num_logits_per_dc, num_logits_per_text= self.model.forward4gen(x, adj, cell_name, ic50)
                    
                    number_features = self.model.generate_samples(0, 1000)
                    
                    loss_TransEs = self.transe(number_features,number_features)
                    loss_TransE = torch.sum(loss_TransEs[0])+torch.sum(loss_TransEs[1])+torch.sum(loss_TransEs[2])
            
                    labels = torch.arange(ic50.shape[0]).long().to(f'cuda:{self.device[0]}')

                    loss_dc = cross_entropy_loss(logits_per_dc, labels)

                    loss_t = cross_entropy_loss(logits_per_text, labels)
                    
                    loss_dc_num = cross_entropy_loss(num_logits_per_dc, labels)

                    loss_t_num = cross_entropy_loss(num_logits_per_text, labels)
                    
                    loss_CLIP = (loss_dc + loss_t)/2 
                    
                    loss_CLIP_Num = (loss_dc_num + loss_t_num)/2 
             
                    
                    self.test_TransE.append(loss_TransE.item())
                    self.test_CLIP.append(loss_CLIP.item())
                    self.test_CLIP_Num.append(loss_CLIP_Num.item())
                    
            mean_train_TransE = np.mean(self.train_TransE)
            mean_train_CLIP = np.mean(self.train_CLIP)
            mean_test_TransE = np.mean(self.test_TransE)
            mean_test_CLIP = np.mean(self.test_CLIP)
            mean_train_CLIP_Num = np.mean(self.train_CLIP_Num)
            mean_test_CLIP_Num = np.mean(self.test_CLIP_Num)
            
            # # -------- Log losses --------
            # logger.log(f'{epoch+1:03d} | {time.time()-t_start:.2f}s | '
            #             f'test transe: {mean_test_TransE:.3e} | test clip: {mean_test_CLIP:.3e} | '
            #             f'train transe: {mean_train_TransE:.3e} | train clip: {mean_train_CLIP:.3e} | ', verbose=False)

            # # -------- Save checkpoints --------
            # if epoch % self.config.train.save_interval == self.config.train.save_interval-1:
            #     save_name = f'_{epoch+1}' if epoch < self.config.train.num_epochs - 1 else ''

            #     torch.save({ 
            #         'model_config': self.config,
            #         'model_state_dict': self.model.state_dict(), 
            #         'transe_state_dict': self.transe.state_dict(),
            #         }, f'./checkpoints/{self.config.data.data}/{self.ckpt + save_name}.pth')
            
            # if epoch % self.config.train.print_interval == self.config.train.print_interval-1:
            #     tqdm.write(f'[EPOCH {epoch+1:04d}] test clip: {mean_test_CLIP:.3e} | train clip: {mean_train_CLIP:.3e} | '
            #                 f'test transe: {mean_test_TransE:.3e} | train transe: {mean_train_TransE:.3e}')
                        # -------- Log losses --------
                        
            logger.log(f'{epoch+1:03d} | {time.time()-t_start:.2f}s | '
                        f'test transe: {mean_test_TransE:.3e} | test clip: {mean_test_CLIP:.3e} | '
                        f'train transe: {mean_train_TransE:.3e} | train clip: {mean_train_CLIP:.3e} | '
                        f'test num clip: {mean_test_CLIP_Num:.3e} | train num clip: {mean_train_CLIP_Num:.3e} | ', verbose=False)

            # -------- Save checkpoints --------
            if epoch % self.config.train.save_interval == self.config.train.save_interval-1:
                save_name = f'_{epoch+1}' if epoch < self.config.train.num_epochs - 1 else ''

                torch.save({ 
                    'model_config': self.config,
                    'model_state_dict': self.model.state_dict(), 
                    'transe_state_dict': self.transe.state_dict(),
                    }, f'./checkpoints/{self.config.data.data}/CLDR_{self.ckpt + save_name}.pth')
            
            if epoch % self.config.train.print_interval == self.config.train.print_interval-1:
                tqdm.write(f'[EPOCH {epoch+1:04d}] test clip: {mean_test_CLIP:.3e} | train clip: {mean_train_CLIP:.3e} | '
                            f'test transe: {mean_test_TransE:.3e} | train transe: {mean_train_TransE:.3e}'
                            f'test num clip: {mean_test_CLIP_Num:.3e} | train num clip: {mean_train_CLIP_Num:.3e}')
                
        print(' ')
        return self.ckpt
