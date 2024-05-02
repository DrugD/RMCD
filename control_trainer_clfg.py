import os
import time
from tqdm import tqdm, trange
import numpy as np
import torch

from utils.loader import load_seed, load_device, load_data, load_model_params, load_model_optimizer, \
                         load_ema, load_loss_fn, load_batch, load_control_net
from utils.logger import Logger, set_log, start_log, train_log

import pdb
# from clip.drp_label_emb import DRPLabelEmbedding
from controller import TransFGDRP


class ControlTrainer(object):
    def __init__(self, config):
        super(ControlTrainer, self).__init__()
        
        self.config = config
        self.log_folder_name, self.log_dir, self.ckpt_dir = set_log(self.config)

        self.seed = load_seed(self.config.seed)
        self.device = load_device()

        self.train_loader, self.test_loader = load_data(self.config)

        self.params_x, self.params_adj = load_model_params(self.config)
        
        # self.drp_label_embedding = DRPLabelEmbedding(self.device)
        
        self.frag_label_embedding = TransFGDRP(self.config.controller, self.params_x, self.params_adj, f'cuda:{self.device[0]}', self.config.train)
        self.frag_label_embedding.load_state_dict(torch.load(self.config.controller.cldr_ckpt)['model_state_dict'], strict = True)
        self.frag_label_embedding.to('cuda:'+str(self.device[0]))
        self.frag_label_embedding.eval()
        # cldr_ckpt
        
    def train(self, ts):
        
        self.config.exp_name = ts
        self.ckpt = f'{ts}'
        print('\033[91m' + f'{self.ckpt}' + '\033[0m')
        
           
        logger = Logger(str(os.path.join(self.log_dir, f'{self.ckpt}.log')), mode='a')
        logger.log(f'{self.ckpt}', verbose=False)
        start_log(logger, self.config)
        train_log(logger, self.config)
        
        
        # -------- Load models, optimizers, ema --------
        self.model_x, self.optimizer_x, self.scheduler_x = load_model_optimizer(self.params_x, self.config.train, 
                                                                                self.device)
        self.model_adj, self.optimizer_adj, self.scheduler_adj = load_model_optimizer(self.params_adj, self.config.train, 
                                                                                        self.device)

        if self.config.train.resume is False:
            logger.log(f'Load origin model pth from {self.config.train.pretrain}', verbose=False)
            self.model_x,self.model_adj = load_control_net(self.model_x,self.model_adj,self.config)
        else:
            logger.log(f'Resume from {self.config.train.resume}', verbose=False)
            self.model_x.load_state_dict(torch.load(self.config.train.resume, map_location=torch.device('cpu'))['x_state_dict'], strict=True)
            self.model_adj.load_state_dict(torch.load(self.config.train.resume, map_location=torch.device('cpu'))['adj_state_dict'], strict=True)

        # if 1:
        #     grad_true_list = ['control', 'zero_convs', 'condition']
        #     # 将包含需要训练的参数的名称的参数设置为可训练，其他参数设置为不可训练
        #     for name, param in self.model_x.named_parameters():
        #         param.requires_grad = any(grad_keyword in name for grad_keyword in grad_true_list)

        #     for name, param in self.model_adj.named_parameters():
        #         param.requires_grad = any(grad_keyword in name for grad_keyword in grad_true_list)
                    
            # frozen_params_x = [param for name, param in self.model_x.named_parameters() if 'control' in name]
            # train_params_x = [param for name, param in model.named_parameters() if 'control' not in name]


            # for param in self.model_x.named_parameters():
            #     import pdb;pdb.set_trace()
            #     if any(substring in param[0] for substring in grad_true_list):
            #         param[1].requires_grad = True
            #         print(param[0],'is not locked.')
            #     else:
            #         print(param[0],'is locked.')
            #         param[1].requires_grad = False
            
            # for param in self.model_adj.named_parameters():
            #     if any(substring in param[0] for substring in grad_true_list):
            #         param[1].requires_grad = True
            #         print(param[0],'is not locked.')
            #     else:
            #         print(param[0],'is locked.')
            #         param[1].requires_grad = False
                    
        self.optimizer_x = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model_x.parameters()), lr=self.config.train.lr, 
                                weight_decay=self.config.train.weight_decay )
        self.optimizer_adj = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model_adj.parameters()), lr=self.config.train.lr, 
                                weight_decay=self.config.train.weight_decay )
        
        print("Molde x has ",len(list(self.model_x.named_parameters()))," params,","and now has ",len(self.optimizer_x.param_groups[0]['params'])," params for training.")
        print("Molde adj has ",len(list(self.model_adj.named_parameters()))," params,","and now has ",len(self.optimizer_adj.param_groups[0]['params'])," params for training.")

        
            
        # import pdb;pdb.set_trace()
        # DLE: DRP_LABEL_EMBEDDING
        # self.optimizer_DLE = torch.optim.Adam(self.drp_label_embedding.parameters(), lr=self.config.train.lr, 
        #                                 weight_decay=self.config.train.weight_decay)
        # self.scheduler_DLE  = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_DLE, gamma=self.config.train.lr_decay)
        
 
        
        
        self.ema_x = load_ema(self.model_x, decay=self.config.train.ema)
        self.ema_adj = load_ema(self.model_adj, decay=self.config.train.ema)
        
     


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
            
            # self.model_x_condition.train()
            # self.model_adj_condition.train()
            
            for _, train_b in enumerate(self.train_loader):

                self.optimizer_x.zero_grad()
                self.optimizer_adj.zero_grad()
                
                # self.optimizer_x_condition.zero_grad()
                # self.optimizer_adj_condition.zero_grad()
                
                # self.optimizer_DLE.zero_grad()
                
          
                x, adj, label = load_batch(train_b, self.device) 

                loss_subject = (x, adj, self.frag_label_embedding.text_forward(label))
                loss_x, loss_adj = self.loss_fn(self.model_x, self.model_adj, *loss_subject)
                
                # loss_subject = (x, adj, self.drp_label_embedding(label))
                # loss_x_condition, loss_adj_condition = self.loss_fn(self.model_x_condition, self.model_adj_condition, *loss_subject)
                
                

                loss_x.backward(retain_graph=True)
                loss_adj.backward()

                # loss_x_condition.backward(retain_graph=True)
                # loss_adj_condition.backward()
                
                
                
                torch.nn.utils.clip_grad_norm_(self.model_x.parameters(), self.config.train.grad_norm)
                torch.nn.utils.clip_grad_norm_(self.model_adj.parameters(), self.config.train.grad_norm)
                
                # torch.nn.utils.clip_grad_norm_(self.model_x_condition.parameters(), self.config.train.grad_norm)
                # torch.nn.utils.clip_grad_norm_(self.model_adj_condition.parameters(), self.config.train.grad_norm)
                
                self.optimizer_x.step()
                self.optimizer_adj.step()
                
                # self.optimizer_x_condition.step()
                # self.optimizer_adj_condition.step()
                
                # self.optimizer_DLE.step()
              

                # -------- EMA update --------
                self.ema_x.update(self.model_x.parameters())
                self.ema_adj.update(self.model_adj.parameters())

                self.train_x.append(loss_x.item())
                self.train_adj.append(loss_adj.item())
                
     
                # for name, param in self.model_x.named_parameters():
                #     print(name, param.sum().item())
                
                # print("------"*10)
                
                # for name, param in self.model_adj.named_parameters():
                #     print(name, param.sum().item())
                
                # import pdb;pdb.set_trace()
                
                # self.ema_x_condition.update(self.model_x_condition.parameters())
                # self.ema_adj_condition.update(self.model_adj_condition.parameters())

                # self.train_x_condition.append(loss_x_condition.item())
                # self.train_adj_condition.append(loss_adj_condition.item())
                
                
            if self.config.train.lr_schedule:
                self.scheduler_x.step()
                self.scheduler_adj.step()
                # self.scheduler_DLE.step()
                
     
            
            self.model_x.eval()
            self.model_adj.eval()
            
            
            
            with torch.no_grad():
                for _, test_b in enumerate(self.test_loader):   
                    
                    x, adj, label = load_batch(train_b, self.device) 
                    
                    loss_subject = (x, adj, self.frag_label_embedding.text_forward(label))
                    loss_x, loss_adj = self.loss_fn(self.model_x, self.model_adj, *loss_subject)

                    self.ema_x.store(self.model_x.parameters())
                    self.ema_x.copy_to(self.model_x.parameters())
                    self.ema_adj.store(self.model_adj.parameters())
                    self.ema_adj.copy_to(self.model_adj.parameters())

         
                    self.test_x.append(loss_x.item())
                    self.test_adj.append(loss_adj.item())
                    
               
                    self.ema_x.restore(self.model_x.parameters())
                    self.ema_adj.restore(self.model_adj.parameters())
                    
                 

            mean_train_x = np.mean(self.train_x)
            mean_train_adj = np.mean(self.train_adj)
            mean_test_x = np.mean(self.test_x)
            mean_test_adj = np.mean(self.test_adj)

            # -------- Log losses --------
            logger.log(f'{epoch+1:03d} | {time.time()-t_start:.2f}s | '
                        f'test x: {mean_test_x:.3e} | test adj: {mean_test_adj:.3e} | '
                        f'train x: {mean_train_x:.3e} | train adj: {mean_train_adj:.3e} | ', verbose=False)

      
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
                    }, f'./checkpoints/{self.config.data.data}/{self.ckpt + save_name}.pth')
         
                # torch.save({ 
                #     'model_config': self.config,
                #     'drp_label_embedding': self.drp_label_embedding.state_dict()
                #     }, f'/home/nas/lk/GDSS/checkpoints/{self.config.data.data}/DLE_{self.ckpt + save_name}.pth')
                
            if epoch % self.config.train.print_interval == self.config.train.print_interval-1:
                tqdm.write(f'[EPOCH {epoch+1:04d}] test adj: {mean_test_adj:.3e} | train adj: {mean_train_adj:.3e} | '
                            f'test x: {mean_test_x:.3e} | train x: {mean_train_x:.3e}')
        print(' ')
        return self.ckpt
