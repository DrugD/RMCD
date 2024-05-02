import os
import time
from tqdm import tqdm, trange
import numpy as np
import torch

from utils.loader import load_seed, load_device, load_data, load_model_params, load_model_optimizer, \
                         load_ema, load_loss_fn, load_batch
from utils.logger import Logger, set_log, start_log, train_log

import pdb
from clip.drp_label_emb import DRPLabelEmbedding

class MSETrainer(object):
    def __init__(self, config):
        super(MSETrainer, self).__init__()

        self.config = config
        self.log_folder_name, self.log_dir, self.ckpt_dir = set_log(self.config)

        self.seed = load_seed(self.config.seed)
        self.device = load_device()
        self.train_loader, self.test_loader = load_data(self.config)

        self.params_x, self.params_adj = load_model_params(self.config)
        self.params_x, self.params_adj = load_model_params(self.config)
        
        self.drp_label_embedding = DRPLabelEmbedding(self.device)
        self.drp_label_embedding.to('cuda:'+str(self.device[0]))
        
    def train(self, ts):
        self.config.exp_name = ts
        self.ckpt = f'{ts}'
        print('\033[91m' + f'{self.ckpt}' + '\033[0m')
        
        
        
        # -------- Load models, optimizers, ema --------
        self.model_x, self.optimizer_x, self.scheduler_x = load_model_optimizer(self.params_x, self.config.train, 
                                                                                self.device)
        self.model_adj, self.optimizer_adj, self.scheduler_adj = load_model_optimizer(self.params_adj, self.config.train, 
                                                                                        self.device)
        
        self.model_x_condition, self.optimizer_x_condition, self.scheduler_x_condition = load_model_optimizer(self.params_x, self.config.train, 
                                                                                self.device)
        self.model_adj_condition, self.optimizer_adj_condition, self.scheduler_adj_condition = load_model_optimizer(self.params_adj, self.config.train, 
                                                                                        self.device)

        
        # DLE: DRP_LABEL_EMBEDDING
        self.optimizer_DLE = torch.optim.Adam(self.drp_label_embedding.parameters(), lr=self.config.train.lr, 
                                        weight_decay=self.config.train.weight_decay)
        self.scheduler_DLE  = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_DLE, gamma=self.config.train.lr_decay)
        
 
        
        
        self.ema_x = load_ema(self.model_x_condition, decay=self.config.train.ema)
        self.ema_adj = load_ema(self.model_adj_condition, decay=self.config.train.ema)
        
        self.ema_x_condition = load_ema(self.model_x_condition, decay=self.config.train.ema)
        self.ema_adj_condition = load_ema(self.model_adj_condition, decay=self.config.train.ema)
   
        logger = Logger(str(os.path.join(self.log_dir, f'{self.ckpt}.log')), mode='a')
        logger.log(f'{self.ckpt}', verbose=False)
        start_log(logger, self.config)
        train_log(logger, self.config)

        self.loss_fn = load_loss_fn(self.config)

        # -------- Training --------
        for epoch in trange(0, (self.config.train.num_epochs), desc = '[Epoch]', position = 1, leave=False):

            self.train_x = []
            self.train_adj = []
            self.test_x = []
            self.test_adj = []
            
            self.train_x_condition = []
            self.train_adj_condition = []
            self.test_x_condition = []
            self.test_adj_condition = []
            
            
            t_start = time.time()

            self.model_x.train()
            self.model_adj.train()
            
            self.model_x_condition.train()
            self.model_adj_condition.train()
            
            for _, train_b in enumerate(self.train_loader):

                self.optimizer_x.zero_grad()
                self.optimizer_adj.zero_grad()
                
                self.optimizer_x_condition.zero_grad()
                self.optimizer_adj_condition.zero_grad()
                
                self.optimizer_DLE.zero_grad()
                
          
                x, adj, label = load_batch(train_b, self.device) 
                
                loss_subject = (x, adj, self.drp_label_embedding.forward_null_text(label))
                loss_x, loss_adj = self.loss_fn(self.model_x, self.model_adj, *loss_subject)
                
                loss_subject = (x, adj, self.drp_label_embedding(label))
                loss_x_condition, loss_adj_condition = self.loss_fn(self.model_x_condition, self.model_adj_condition, *loss_subject)
                
                

                loss_x.backward(retain_graph=True)
                loss_adj.backward(retain_graph=True)

                loss_x_condition.backward(retain_graph=True)
                loss_adj_condition.backward()
                
                
                
                torch.nn.utils.clip_grad_norm_(self.model_x.parameters(), self.config.train.grad_norm)
                torch.nn.utils.clip_grad_norm_(self.model_adj.parameters(), self.config.train.grad_norm)
                
                torch.nn.utils.clip_grad_norm_(self.model_x_condition.parameters(), self.config.train.grad_norm)
                torch.nn.utils.clip_grad_norm_(self.model_adj_condition.parameters(), self.config.train.grad_norm)
                
                self.optimizer_x.step()
                self.optimizer_adj.step()
                
                self.optimizer_x_condition.step()
                self.optimizer_adj_condition.step()
                
                self.optimizer_DLE.step()
              

                # -------- EMA update --------
                self.ema_x.update(self.model_x.parameters())
                self.ema_adj.update(self.model_adj.parameters())

                self.train_x.append(loss_x.item())
                self.train_adj.append(loss_adj.item())
                
                self.ema_x_condition.update(self.model_x_condition.parameters())
                self.ema_adj_condition.update(self.model_adj_condition.parameters())

                self.train_x_condition.append(loss_x_condition.item())
                self.train_adj_condition.append(loss_adj_condition.item())
                
                
            if self.config.train.lr_schedule:
                self.scheduler_x.step()
                self.scheduler_adj.step()
                self.scheduler_x_condition.step()
                self.scheduler_adj_condition.step()
                self.scheduler_DLE.step()
                
     
            
            self.model_x.eval()
            self.model_adj.eval()
            
            self.model_x_condition.eval()
            self.model_adj_condition.eval()
            
            self.drp_label_embedding.eval()
            
            with torch.no_grad():
                for _, test_b in enumerate(self.test_loader):   
                    
                    x, adj, label = load_batch(train_b, self.device) 
                    
                    loss_subject = (x, adj, self.drp_label_embedding.forward_null_text(label))
                    loss_x, loss_adj = self.loss_fn(self.model_x, self.model_adj, *loss_subject)
                    
                    loss_subject = (x, adj, self.drp_label_embedding(label))
                    loss_x_condition, loss_adj_condition = self.loss_fn(self.model_x_condition, self.model_adj_condition, *loss_subject)
                    
                    self.ema_x.store(self.model_x.parameters())
                    self.ema_x.copy_to(self.model_x.parameters())
                    self.ema_adj.store(self.model_adj.parameters())
                    self.ema_adj.copy_to(self.model_adj.parameters())

                    self.ema_x_condition.store(self.model_x_condition.parameters())
                    self.ema_x_condition.copy_to(self.model_x_condition.parameters())
                    self.ema_adj_condition.store(self.model_adj_condition.parameters())
                    self.ema_adj_condition.copy_to(self.model_adj_condition.parameters())

                    self.test_x.append(loss_x.item())
                    self.test_adj.append(loss_adj.item())
                    
                    self.test_x_condition.append(loss_x_condition.item())
                    self.test_adj_condition.append(loss_adj_condition.item())
                    
                    self.ema_x.restore(self.model_x.parameters())
                    self.ema_adj.restore(self.model_adj.parameters())
                    
                    self.ema_x_condition.restore(self.model_x_condition.parameters())
                    self.ema_adj_condition.restore(self.model_adj_condition.parameters())

            mean_train_x = np.mean(self.train_x)
            mean_train_adj = np.mean(self.train_adj)
            mean_test_x = np.mean(self.test_x)
            mean_test_adj = np.mean(self.test_adj)

            mean_train_x_condition = np.mean(self.train_x_condition)
            mean_train_adj_condition = np.mean(self.train_adj_condition)
            mean_test_x_condition = np.mean(self.test_x_condition)
            mean_test_adj_condition = np.mean(self.test_adj_condition)
            
            # -------- Log losses --------
            logger.log(f'{epoch+1:03d} | {time.time()-t_start:.2f}s | '
                        f'test x: {mean_test_x:.3e} | test adj: {mean_test_adj:.3e} | '
                        f'train x: {mean_train_x:.3e} | train adj: {mean_train_adj:.3e} | ', verbose=False)

            logger.log(f'{epoch+1:03d} | {time.time()-t_start:.2f}s | '
                        f'test x condition: {mean_test_x_condition:.3e} | test adj condition: {mean_test_adj_condition:.3e} | '
                        f'train x condition: {mean_train_x_condition:.3e} | train adj condition: {mean_train_adj_condition:.3e} | ', verbose=False)
            
            # -------- Save checkpoints --------
            if epoch % self.config.train.save_interval == self.config.train.save_interval-1:
                save_name = f'_{epoch+1}' if epoch < self.config.train.num_epochs - 1 else ''
             
                torch.save({ 
                    'model_config': self.config,
                    'params_x' : self.params_x,
                    'params_adj' : self.params_adj,
                    'x_state_dict': self.model_x.state_dict(), 
                    'adj_state_dict': self.model_adj.state_dict(),
                    'ema_x': self.ema_x.state_dict(),
                    'ema_adj': self.ema_adj.state_dict()
                    }, f'/home/nas/lk/GDSS/checkpoints/{self.config.data.data}/{self.ckpt + save_name}.pth')
                
                torch.save({ 
                    'model_config': self.config,
                    'params_x' : self.params_x,
                    'params_adj' : self.params_adj,
                    'x_state_dict_condition': self.model_x_condition.state_dict(), 
                    'adj_state_dict_condition': self.model_adj_condition.state_dict(),
                    'ema_x_condition': self.ema_x_condition.state_dict(),
                    'ema_adj_condition': self.ema_adj_condition.state_dict()
                    }, f'/home/nas/lk/GDSS/checkpoints/{self.config.data.data}/{self.ckpt + save_name}_condition.pth')
                
                torch.save({ 
                    'model_config': self.config,
                    'drp_label_embedding': self.drp_label_embedding.state_dict()
                    }, f'/home/nas/lk/GDSS/checkpoints/{self.config.data.data}/DLE_{self.ckpt + save_name}.pth')
                
            if epoch % self.config.train.print_interval == self.config.train.print_interval-1:
                tqdm.write(f'[EPOCH {epoch+1:04d}] test adj: {mean_test_adj:.3e} | train adj: {mean_train_adj:.3e} | '
                            f'test x: {mean_test_x:.3e} | train x: {mean_train_x:.3e} | test adj condition: {mean_test_adj_condition:.3e} | '
                            f'train adj condition: {mean_train_adj_condition:.3e} | '
                            f'test x condition: {mean_test_x_condition:.3e} | train x condition: {mean_train_x_condition:.3e}')
        print(' ')
        return self.ckpt

