#! /usr/bin/env python

import torch
import torch.nn.functional as F
from torch import nn
import logging

# custom
import data
import math
import matplotlib.pyplot as plt
import os
import numpy as np
import kld_func

class SimpleEncClassifier(nn.Module):
    def __init__(self, enc_dims, mlp_dims, dropout=0.2, verbose=1):
        super().__init__()
        self.enc_dims = enc_dims
        self.mlp_dims = mlp_dims
        self.encoder_model = None
        self.mlp_model = None
        self.encoded = None
        self.mlp_out = None
        self.encoder_modules = []
        self.mlp_modules = []
        self.verbose = verbose
        # encoder
        n_stacks = len(self.enc_dims) - 1 # 4
        # internal layers in encoder
        for i in range(n_stacks - 1): # 3
            self.encoder_modules.append(nn.Linear(self.enc_dims[i], self.enc_dims[i + 1]))
            self.encoder_modules.append(nn.ReLU())
        # encoded features layer. no activation.
        self.encoder_modules.append(nn.Linear(self.enc_dims[-2], self.enc_dims[-1]))
        # encoder model
        self.encoder_model = nn.Sequential(*(self.encoder_modules))

        # MLP
        m_stacks = len(self.mlp_dims) - 1
        for i in range(m_stacks - 1):
            self.mlp_modules.append(nn.Linear(self.mlp_dims[i], self.mlp_dims[i + 1]))
            self.mlp_modules.append(nn.ReLU())
            if dropout > 0:
                self.mlp_modules.append(nn.Dropout(p=dropout))
        # mlp output
        self.mlp_modules.append(nn.Linear(self.mlp_dims[-2], self.mlp_dims[-1]))
        self.mlp_modules.append(nn.Softmax(dim=1))
        self.mlp_model = nn.Sequential(*(self.mlp_modules))

        if self.verbose:
            print(self.encoder_model)
            print(self.mlp_model)
        return

    def update_mlp_head(self, dropout=0.2):
        self.mlp_out = None
        self.mlp_modules = []

        # MLP
        m_stacks = len(self.mlp_dims) - 1
        for i in range(m_stacks - 1):
            self.mlp_modules.append(nn.Linear(self.mlp_dims[i], self.mlp_dims[i + 1]))
            self.mlp_modules.append(nn.ReLU())
            if dropout > 0:
                self.mlp_modules.append(nn.Dropout(p=dropout))
        # mlp output
        self.mlp_modules.append(nn.Linear(self.mlp_dims[-2], self.mlp_dims[-1]))
        self.mlp_modules.append(nn.Softmax(dim=1))
        self.mlp_model = nn.Sequential(*(self.mlp_modules))

        if self.verbose:
            print(self.encoder_model)
            print(self.mlp_model)
        return

    def forward(self, x):
        self.encoded = self.encoder_model(x)
        self.out = self.mlp_model(self.encoded)
        return self.encoded, self.encoded, self.out
    
    def predict_proba(self, x):
        _, _, mlp_out = self.forward(x)
        return mlp_out
    
    def predict(self, x):
        self.encoded = self.encoder_model(x)
        self.out = self.mlp_model(self.encoded)
        preds = self.out.max(1)[1]
        return preds
    
    
    def encode(self, x, is_all_return = False):
        self.encoded = self.encoder_model(x)
        if is_all_return == False:
            return self.encoded
        return None, x, None, None, self.encoded
    
    def discrete_cmap(self, N, base_cmap=None):

        # Note that if base_cmap is a string or None, you can simply do
        #    return plt.cm.get_cmap(base_cmap, N)
        # The following works for string, None, or a colormap instance:

        base = plt.cm.get_cmap(base_cmap)
        color_list = base(np.linspace(0, 1, N))
        cmap_name = base.name + str(N)
        return base.from_list(cmap_name, color_list, N)
    
    def eval_plot(self, x_tensor_cuda, y_bin, epoch, args):
        with torch.no_grad(): 
            x_tensor = x_tensor_cuda
            #1
            x_encoded_tensor = self.encode(x_tensor)
            x_encoded = x_encoded_tensor.detach().cpu().float().numpy()

            N = len(np.unique(y_bin))
            
            for i in range(0, x_encoded.shape[1], 2):
                plt.scatter(x_encoded[:,i], x_encoded[:,i+1], c = y_bin, alpha=0.2, s=10, marker='o', edgecolor='none', cmap=self.discrete_cmap(N, 'jet'))
                #plt.colorbar(ticks=range(N))
                plt.xlabel('x'+str(i+1))
                plt.ylabel('x'+str(i+2))
                plt.colorbar()
                #plt.show()
                
                directory_path = "./maps/hc/"+ str(x_encoded.shape[1]) + "/epoch_" + str(epoch)
                # 디렉토리 생성
                if not os.path.exists(directory_path):
                    os.makedirs(directory_path)
                
                file_name = directory_path + "/dim_" + str(i+1) + "-" + str(i+2) + "_for_x"
                plt.savefig(file_name)
                plt.clf()
                if i==14:
                    break
    
    

# EncKldCustomMlpEnsemble6: EncKldCustomMlpEnsemble2 + 전역 kld loss
class EncKldCustomMlpEnsemble6(nn.Module):
    def __init__(self, enc_dims, mlp_dims, kld_scale = None, kld_dev_scale = None, dropout=0.2, verbose=1):
        super().__init__()
        self.enc_dims = enc_dims
        self.mlp_dims = mlp_dims
        self.pre_encoder_model = None
        self.encoder_model = None
        
        self.mlp_model = None
        self.encoded = None
        self.mlp_out = None
        self.pre_encoder_modules = []
        self.encoder_modules = []
        self.mlp_modules = []
        self.verbose = verbose
        
        self.kld_dev_scale = kld_dev_scale
        
        # encoder
        n_stacks = len(self.enc_dims) - 1 # 4
        # internal layers in encoder
        for i in range(n_stacks - 1): # 3
            self.encoder_modules.append(nn.Linear(self.enc_dims[i], self.enc_dims[i + 1]))
            self.encoder_modules.append(nn.ReLU())
        # pre_encoder model
        self.encoder_modules.append(nn.Linear(self.enc_dims[-2], self.enc_dims[-1]))
        
        # add?
        #self.encoder_modules.append(nn.Tanh())
        
        self.encoder_model = nn.Sequential(*(self.encoder_modules))
        
        # encoder
        n_stacks = len(self.enc_dims) - 1 # 4
        # internal layers in encoder
        for i in range(n_stacks - 1): # 3
            self.pre_encoder_modules.append(nn.Linear(self.enc_dims[i], self.enc_dims[i + 1]))
            self.pre_encoder_modules.append(nn.ReLU())
        # pre_encoder model
        self.pre_encoder_model = nn.Sequential(*(self.pre_encoder_modules))
        
        self.z_mean_fc = nn.Linear(self.enc_dims[-2], self.enc_dims[-1])
        self.z_log_var_fc = nn.Linear(self.enc_dims[-2], self.enc_dims[-1])
        
        # MLP
        self.mlp_dims[0] = self.mlp_dims[0] * 2
        m_stacks = len(self.mlp_dims) - 1
        for i in range(m_stacks - 1):
            self.mlp_modules.append(nn.Linear(self.mlp_dims[i], self.mlp_dims[i + 1]))
            self.mlp_modules.append(nn.ReLU())
            if dropout > 0:
                self.mlp_modules.append(nn.Dropout(p=dropout))
        # mlp output
        self.mlp_modules.append(nn.Linear(self.mlp_dims[-2], self.mlp_dims[-1]))
        self.mlp_modules.append(nn.Softmax(dim=1))
        self.mlp_model = nn.Sequential(*(self.mlp_modules))

        if self.verbose:
            print(self.pre_encoder_model)
            print(self.mlp_model)
        return
    
    def discrete_cmap(self, N, base_cmap=None):

        # Note that if base_cmap is a string or None, you can simply do
        #    return plt.cm.get_cmap(base_cmap, N)
        # The following works for string, None, or a colormap instance:

        base = plt.cm.get_cmap(base_cmap)
        color_list = base(np.linspace(0, 1, N))
        cmap_name = base.name + str(N)
        return base.from_list(cmap_name, color_list, N)
    
    
    def eval_plot(self, x_tensor_cuda, y_bin, epoch, args):
        with torch.no_grad():
            x_tensor = x_tensor_cuda
            #1
            kld_encoded_tensor, _, _, _, c_encoded_tensor = self.encode(x_tensor, is_all_return = True)
            c_encoded = c_encoded_tensor.detach().cpu().float().numpy()
            kld_encoded = kld_encoded_tensor.detach().cpu().float().numpy()

            N = len(np.unique(y_bin))
            
            for i in range(0, c_encoded.shape[1], 2):
                plt.scatter(c_encoded[:,i], c_encoded[:,i+1], c = y_bin, alpha=0.2, s=10, marker='o', edgecolor='none', cmap=self.discrete_cmap(N, 'jet'))
                #plt.colorbar(ticks=range(N))
                plt.xlabel('x'+str(i+1))
                plt.ylabel('x'+str(i+2))
                plt.colorbar()
                #plt.show()
                
                directory_path = "./maps/hc_kld/"+ str(c_encoded.shape[1]) + "/epoch_" + str(epoch)
                # 디렉토리 생성
                if not os.path.exists(directory_path):
                    os.makedirs(directory_path)
                
                file_name = directory_path + "/dim_" + str(i+1) + "-" + str(i+2) + "_for_x"
                plt.savefig(file_name)
                plt.clf()
                if i==14:
                    break
            
            
            #markers = ['o' if y == 0 else 'X' for y in y_bin]
            N = len(np.unique(y_bin))
            for i in range(0, kld_encoded.shape[1], 2):
                plt.scatter(kld_encoded[:,i], kld_encoded[:,i+1], c = y_bin, alpha=0.2, s=10, marker='o', edgecolor='none', cmap=self.discrete_cmap(N, 'jet'))
                #plt.colorbar(ticks=range(N))
                plt.xlabel('z'+str(i))
                plt.ylabel('z'+str(i+1))
                plt.colorbar()
                #plt.show()
                
                file_name = directory_path + "/dim_" + str(i+1) + "-" + str(i+2) + "_for_z"
                plt.savefig(file_name)
                plt.clf()
                if i==14:
                    break
        
        
    def forward(self, x):
        kld_encoded, _, z_mean, z_log_var, c_encoded = self.encode(x, is_all_return = True)
        
        z_concat = torch.cat((c_encoded, kld_encoded), dim=1)
        
        # z = self.reparameterize(z_mean, z_log_var)
        mlp_out = self.mlp_model(z_concat)
        #self.out = self.mlp_model(c_encoded)
        return kld_encoded, x, z_mean, z_log_var, c_encoded, mlp_out



    def predict_proba(self, x):
        kld_encoded, _, z_mean, z_log_var, c_encoded = self.encode(x, is_all_return = True)
        z_concat = torch.cat((c_encoded, kld_encoded), dim=1)
        mlp_out = self.mlp_model(z_concat)
        #mlp_out = torch.clamp(mlp_out, min=1e-5, max=1 - 1e-5)
        return mlp_out
        
        
    def predict(self, x):

        with torch.no_grad():
            mlp_out = self.predict_proba(x)
            preds = mlp_out.max(1)[1]
            return preds
        
    def reparameterize(self, z_mean, z_log_var, kld_dev_scale):
        std = torch.exp(0.5*z_log_var)
        eps = torch.randn_like(std)
        return z_mean + (std * kld_dev_scale) * eps
    
    def encode_c(self, x):
        c_encoded = self.encoder_model(x)
        return c_encoded
    
    def encode_kld(self, x):
        self.pre_encoded = self.pre_encoder_model(x)
        z_mean = self.z_mean_fc(self.pre_encoded)
        z_log_var = self.z_log_var_fc(self.pre_encoded)
        self.encoded = self.reparameterize(z_mean, z_log_var, self.kld_dev_scale)

        return self.encoded, z_mean, z_log_var
    
    
    def encode(self, x, is_all_return = False):
        
        c_encoded = self.encode_c(x)
        
        
        kld_encoded, z_mean, z_log_var = self.encode_kld(x)
        
        if is_all_return == False:
            return c_encoded
        
        return kld_encoded, x, z_mean, z_log_var, c_encoded
    
    
    


class CAEKldEnsembleMlp(nn.Module):
    def __init__(self, enc_dims, mlp_dims, kld_scale = 2.0, dropout=0.2, verbose=1):
        super().__init__()
        self.enc_dims = enc_dims
        self.mlp_dims = mlp_dims
        
        self.encoder_model = None
        self.pre_encoder_model = None
        self.decoder_model = None
        self.mlp_model = None
        
        self.encoded = None
        self.decoded = None
        
        self.encoder_modules = []
        self.pre_encoder_modules = []
        self.decoder_modules = []
        self.mlp_modules = []
        
        self.verbose = verbose
        
        self.kld_scale = kld_scale
        
        self.dim_last = enc_dims[-1]
        self.two_x = kld_func.find_exponent_of_two(self.dim_last)
        
        # encoder
        n_stacks = len(self.enc_dims) - 1
        # internal layers in encoder
        for i in range(n_stacks - 1):
            self.encoder_modules.append(nn.Linear(self.enc_dims[i], self.enc_dims[i + 1]))
            self.encoder_modules.append(nn.ReLU())
        # encoded features layer. no activation.
        self.encoder_modules.append(nn.Linear(self.enc_dims[-2], self.enc_dims[-1]))
        # encoder model
        self.encoder_model = nn.Sequential(*(self.encoder_modules))
        self.encoder_model.apply(self.init_weights)

        # decoder
        # internal layers in decoder
        for i in range(n_stacks - 1, 0, -1):
            self.decoder_modules.append(nn.Linear(self.enc_dims[i + 1], self.enc_dims[i]))
            self.decoder_modules.append(nn.ReLU())
        # decoded output. no activation.
        self.decoder_modules.append(nn.Linear(self.enc_dims[1], self.enc_dims[0]))
        # decoder model
        self.decoder_model = nn.Sequential(*(self.decoder_modules))
        self.decoder_model.apply(self.init_weights)
        
        # pre_encoder
        n_stacks = len(self.enc_dims) - 1 # 4
        # internal layers in encoder
        for i in range(n_stacks - 1): # 3
            self.pre_encoder_modules.append(nn.Linear(self.enc_dims[i], self.enc_dims[i + 1]))
            self.pre_encoder_modules.append(nn.ReLU())
        # pre_encoder model
        self.pre_encoder_model = nn.Sequential(*(self.pre_encoder_modules))
        
        self.z_mean_fc = nn.Linear(self.enc_dims[-2], self.enc_dims[-1])
        self.z_log_var_fc = nn.Linear(self.enc_dims[-2], self.enc_dims[-1])
        
        # MLP
        self.mlp_dims[0] = self.mlp_dims[0] * 2
        m_stacks = len(self.mlp_dims) - 1
        for i in range(m_stacks - 1):
            self.mlp_modules.append(nn.Linear(self.mlp_dims[i], self.mlp_dims[i + 1]))
            self.mlp_modules.append(nn.ReLU())
            if dropout > 0:
                self.mlp_modules.append(nn.Dropout(p=dropout))
        # mlp output
        self.mlp_modules.append(nn.Linear(self.mlp_dims[-2], self.mlp_dims[-1]))
        self.mlp_modules.append(nn.Softmax(dim=1))
        self.mlp_model = nn.Sequential(*(self.mlp_modules))

        if self.verbose:
            print(self.encoder_model)
            print(self.decoder_model)
            print(self.pre_encoder_model)
            print(self.mlp_model)
        return

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        return
    
    def discrete_cmap(self, N, base_cmap=None):

        # Note that if base_cmap is a string or None, you can simply do
        #    return plt.cm.get_cmap(base_cmap, N)
        # The following works for string, None, or a colormap instance:

        base = plt.cm.get_cmap(base_cmap)
        color_list = base(np.linspace(0, 1, N))
        cmap_name = base.name + str(N)
        return base.from_list(cmap_name, color_list, N)
    
    
    def eval_plot(self, x_tensor_cuda, y_bin, epoch, args):
        with torch.no_grad():
            x_tensor = x_tensor_cuda
            #1
            _, _, _, _, x_encoded_tensor = self.encode(x_tensor, is_all_return = True)
            x_encoded = x_encoded_tensor.detach().cpu().float().numpy()

            N = len(np.unique(y_bin))
            
            for i in range(0, x_encoded.shape[1], 2):
                plt.scatter(x_encoded[:,i], x_encoded[:,i+1], c = y_bin, alpha=0.2, s=10, marker='o', edgecolor='none', cmap=self.discrete_cmap(N, 'jet'))
                #plt.colorbar(ticks=range(N))
                plt.xlabel('x'+str(i+1))
                plt.ylabel('x'+str(i+2))
                plt.colorbar()
                #plt.show()
                
                directory_path = "./maps/cade_kld/"+ str(x_encoded.shape[1]) + "/epoch_" + str(epoch)
                if not os.path.exists(directory_path):
                    os.makedirs(directory_path)
                
                file_name = directory_path + "/dim_" + str(i+1) + "-" + str(i+2) + "_for_x"
                plt.savefig(file_name)
                plt.clf()
                if i==14:
                    break
                
            self.pre_encoded = self.pre_encoder_model(x_tensor)
            
            z_mean = self.z_mean_fc(self.pre_encoded)
            z_log_var = self.z_log_var_fc(self.pre_encoded)
            
            z = self.reparameterize(z_mean, z_log_var)
            z = z.detach().cpu().float().numpy()

            #markers = ['o' if y == 0 else 'X' for y in y_bin]
            N = len(np.unique(y_bin))
            for i in range(0, z.shape[1], 2):
                plt.scatter(z[:,i], z[:,i+1], c = y_bin, alpha=0.2, s=10, marker='o', edgecolor='none', cmap=self.discrete_cmap(N, 'jet'))
                #plt.colorbar(ticks=range(N))
                plt.xlabel('z'+str(i))
                plt.ylabel('z'+str(i+1))
                plt.colorbar()
                #plt.show()
                
                file_name = directory_path + "/dim_" + str(i+1) + "-" + str(i+2) + "_for_z"
                plt.savefig(file_name)
                plt.clf()
                if i==14:
                    break
    
    
    def forward(self, x):
        kld_encoded, _, z_mean, z_log_var, c_encoded = self.encode(x, is_all_return = True)
        decoded = self.decoder_model(c_encoded)
        
        z_concat = torch.cat((c_encoded, kld_encoded), dim=1)
        self.out = self.mlp_model(z_concat)
        
        return kld_encoded, x, decoded, z_mean, z_log_var, c_encoded, self.out
    
    def predict_proba(self, x):
        kld_encoded, _, z_mean, z_log_var, c_encoded = self.encode(x, is_all_return = True)
        z_concat = torch.cat((c_encoded, kld_encoded), dim=1)
        mlp_out = self.mlp_model(z_concat)
        mlp_out = torch.clamp(mlp_out, min=1e-5, max=1 - 1e-5)
        return mlp_out
        
    
    def predict(self, x):
        self.out = self.predict_proba(x)
        preds = self.out.max(1)[1]
        return preds
    
    def reparameterize(self, z_mean, z_log_var):
        std = torch.exp(0.5*z_log_var)
        eps = torch.randn_like(std) * (math.sqrt(1/2) ** (self.two_x))
        eps = eps * self.kld_scale
        
        return z_mean + eps*std
    
    def encode_c(self, x):
        c_encoded = self.encoder_model(x)
        return c_encoded
    
    def encode_kld(self, x):
        pre_encoded = self.pre_encoder_model(x)
        z_mean = self.z_mean_fc(pre_encoded)
        z_log_var = self.z_log_var_fc(pre_encoded)
        kld_encoded = self.reparameterize(z_mean, z_log_var)

        return kld_encoded, z_mean, z_log_var
    
    
    def encode(self, x, is_all_return = False):
        
        c_encoded = self.encode_c(x)
        kld_encoded, z_mean, z_log_var = self.encode_kld(x)
        
        if is_all_return == False:
            return c_encoded
        
        return kld_encoded, x, z_mean, z_log_var, c_encoded
    
    
    
    
    
    
    
    

class CAEMlp(nn.Module):
    def __init__(self, enc_dims, mlp_dims, dropout=0.2, verbose=1):
        super().__init__()
        self.enc_dims = enc_dims
        self.mlp_dims = mlp_dims
        
        self.encoder_model = None
        self.decoder_model = None
        self.mlp_model = None
        
        self.encoded = None
        self.decoded = None
        
        self.encoder_modules = []
        self.decoder_modules = []
        self.mlp_modules = []
        
        self.verbose = verbose
        
        # encoder
        n_stacks = len(self.enc_dims) - 1
        # internal layers in encoder
        for i in range(n_stacks - 1):
            self.encoder_modules.append(nn.Linear(self.enc_dims[i], self.enc_dims[i + 1]))
            self.encoder_modules.append(nn.ReLU())
        # encoded features layer. no activation.
        self.encoder_modules.append(nn.Linear(self.enc_dims[-2], self.enc_dims[-1]))
        # encoder model
        self.encoder_model = nn.Sequential(*(self.encoder_modules))
        self.encoder_model.apply(self.init_weights)

        # decoder
        # internal layers in decoder
        for i in range(n_stacks - 1, 0, -1):
            self.decoder_modules.append(nn.Linear(self.enc_dims[i + 1], self.enc_dims[i]))
            self.decoder_modules.append(nn.ReLU())
        # decoded output. no activation.
        self.decoder_modules.append(nn.Linear(self.enc_dims[1], self.enc_dims[0]))
        # decoder model
        self.decoder_model = nn.Sequential(*(self.decoder_modules))
        self.decoder_model.apply(self.init_weights)
        
        # MLP
        #self.mlp_dims[0] = self.mlp_dims[0] * 2
        m_stacks = len(self.mlp_dims) - 1
        for i in range(m_stacks - 1):
            self.mlp_modules.append(nn.Linear(self.mlp_dims[i], self.mlp_dims[i + 1]))
            self.mlp_modules.append(nn.ReLU())
            if dropout > 0:
                self.mlp_modules.append(nn.Dropout(p=dropout))
        # mlp output
        self.mlp_modules.append(nn.Linear(self.mlp_dims[-2], self.mlp_dims[-1]))
        self.mlp_modules.append(nn.Softmax(dim=1))
        self.mlp_model = nn.Sequential(*(self.mlp_modules))

        if self.verbose:
            print(self.encoder_model)
            print(self.decoder_model)
            print(self.mlp_model)
        return

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        return
    
    def discrete_cmap(self, N, base_cmap=None):

        # Note that if base_cmap is a string or None, you can simply do
        #    return plt.cm.get_cmap(base_cmap, N)
        # The following works for string, None, or a colormap instance:

        base = plt.cm.get_cmap(base_cmap)
        color_list = base(np.linspace(0, 1, N))
        cmap_name = base.name + str(N)
        return base.from_list(cmap_name, color_list, N)
    
    
    def eval_plot(self, x_tensor_cuda, y_bin, epoch, args):
        with torch.no_grad():
            x_tensor = x_tensor_cuda
            #1
            x_encoded_tensor = self.encode(x_tensor, is_all_return = True)
            x_encoded = x_encoded_tensor.detach().cpu().float().numpy()

            N = len(np.unique(y_bin))
            
            for i in range(0, x_encoded.shape[1], 2):
                plt.scatter(x_encoded[:,i], x_encoded[:,i+1], c = y_bin, alpha=0.2, s=10, marker='o', edgecolor='none', cmap=self.discrete_cmap(N, 'jet'))
                #plt.colorbar(ticks=range(N))
                plt.xlabel('x'+str(i+1))
                plt.ylabel('x'+str(i+2))
                plt.colorbar()
                #plt.show()
                
                directory_path = "./maps/cade/"+ str(x_encoded.shape[1]) + "/epoch_" + str(epoch)
                if not os.path.exists(directory_path):
                    os.makedirs(directory_path)
                
                file_name = directory_path + "/dim_" + str(i+1) + "-" + str(i+2) + "_for_x"
                plt.savefig(file_name)
                plt.clf()
                if i==14:
                    break
                
    
    
    def forward(self, x):
        c_encoded = self.encode(x)
        decoded = self.decoder_model(c_encoded)
        self.out = self.mlp_model(c_encoded)
        
        return x, decoded, c_encoded, self.out
    
    def predict_proba(self, x):
        c_encoded = self.encode(x)
        mlp_out = self.mlp_model(c_encoded)
        mlp_out = torch.clamp(mlp_out, min=1e-5, max=1 - 1e-5)
        return mlp_out
        
    
    def predict(self, x):
        self.out = self.predict_proba(x)
        preds = self.out.max(1)[1]
        return preds
    
    
    def encode(self, x, is_all_return = False):
        self.encoded = self.encoder_model(x)
        if is_all_return == False:
            return self.encoded
        return None, x, None, None, self.encoded
    









class TripletMlp(nn.Module):
    def __init__(self, enc_dims, mlp_dims, dropout=0.2, verbose=1):
        super().__init__()
        self.enc_dims = enc_dims
        self.mlp_dims = mlp_dims
        
        self.encoder_model = None
        self.mlp_model = None
        
        self.encoded = None
        
        self.encoder_modules = []
        self.mlp_modules = []
        
        self.verbose = verbose
        
        # encoder
        n_stacks = len(self.enc_dims) - 1
        # internal layers in encoder
        for i in range(n_stacks - 1):
            self.encoder_modules.append(nn.Linear(self.enc_dims[i], self.enc_dims[i + 1]))
            self.encoder_modules.append(nn.ReLU())
        # encoded features layer. no activation.
        self.encoder_modules.append(nn.Linear(self.enc_dims[-2], self.enc_dims[-1]))
        # encoder model
        self.encoder_model = nn.Sequential(*(self.encoder_modules))
        self.encoder_model.apply(self.init_weights)
        
        # MLP
        #self.mlp_dims[0] = self.mlp_dims[0] * 2
        m_stacks = len(self.mlp_dims) - 1
        for i in range(m_stacks - 1):
            self.mlp_modules.append(nn.Linear(self.mlp_dims[i], self.mlp_dims[i + 1]))
            self.mlp_modules.append(nn.ReLU())
            if dropout > 0:
                self.mlp_modules.append(nn.Dropout(p=dropout))
        # mlp output
        self.mlp_modules.append(nn.Linear(self.mlp_dims[-2], self.mlp_dims[-1]))
        self.mlp_modules.append(nn.Softmax(dim=1))
        self.mlp_model = nn.Sequential(*(self.mlp_modules))

        if self.verbose:
            print(self.encoder_model)
            print(self.mlp_model)
        return

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        return
    
    def discrete_cmap(self, N, base_cmap=None):

        # Note that if base_cmap is a string or None, you can simply do
        #    return plt.cm.get_cmap(base_cmap, N)
        # The following works for string, None, or a colormap instance:

        base = plt.cm.get_cmap(base_cmap)
        color_list = base(np.linspace(0, 1, N))
        cmap_name = base.name + str(N)
        return base.from_list(cmap_name, color_list, N)
    
    
    def eval_plot(self, x_tensor_cuda, y_bin, epoch, args):
        with torch.no_grad():
            x_tensor = x_tensor_cuda
            #1
            x_encoded_tensor = self.encode(x_tensor, is_all_return = True)
            x_encoded = x_encoded_tensor.detach().cpu().float().numpy()

            N = len(np.unique(y_bin))
            
            for i in range(0, x_encoded.shape[1], 2):
                plt.scatter(x_encoded[:,i], x_encoded[:,i+1], c = y_bin, alpha=0.2, s=10, marker='o', edgecolor='none', cmap=self.discrete_cmap(N, 'jet'))
                #plt.colorbar(ticks=range(N))
                plt.xlabel('x'+str(i+1))
                plt.ylabel('x'+str(i+2))
                plt.colorbar()
                #plt.show()
                
                directory_path = "./maps/triplet/"+ str(x_encoded.shape[1]) + "/epoch_" + str(epoch)
                if not os.path.exists(directory_path):
                    os.makedirs(directory_path)
                
                file_name = directory_path + "/dim_" + str(i+1) + "-" + str(i+2) + "_for_x"
                plt.savefig(file_name)
                plt.clf()
                if i==14:
                    break
                
    
    
    def forward(self, x):
        c_encoded = self.encode(x)
        self.out = self.mlp_model(c_encoded)
        return x, c_encoded, self.out
    
    def predict_proba(self, x):
        c_encoded = self.encode(x)
        mlp_out = self.mlp_model(c_encoded)
        mlp_out = torch.clamp(mlp_out, min=1e-5, max=1 - 1e-5)
        return mlp_out
    
    def predict(self, x):
        self.out = self.predict_proba(x)
        preds = self.out.max(1)[1]
        return preds
    
    
    def encode(self, x, is_all_return = False):
        self.encoded = self.encoder_model(x)
        if is_all_return == False:
            return self.encoded
        return None, x, None, None, self.encoded
        





class TripletKldEnsembleMlp(nn.Module):
    def __init__(self, enc_dims, mlp_dims, kld_scale = 2.0, dropout=0.2, verbose=1):
        super().__init__()
        self.enc_dims = enc_dims
        self.mlp_dims = mlp_dims
        
        self.encoder_model = None
        self.pre_encoder_model = None
        self.mlp_model = None
        
        self.encoded = None
        
        self.encoder_modules = []
        self.pre_encoder_modules = []
        self.mlp_modules = []
        
        self.verbose = verbose
        
        self.kld_scale = kld_scale
        
        self.dim_last = enc_dims[-1]
        self.two_x = kld_func.find_exponent_of_two(self.dim_last)
        
        # encoder
        n_stacks = len(self.enc_dims) - 1
        # internal layers in encoder
        for i in range(n_stacks - 1):
            self.encoder_modules.append(nn.Linear(self.enc_dims[i], self.enc_dims[i + 1]))
            self.encoder_modules.append(nn.ReLU())
        # encoded features layer. no activation.
        self.encoder_modules.append(nn.Linear(self.enc_dims[-2], self.enc_dims[-1]))
        # encoder model
        self.encoder_model = nn.Sequential(*(self.encoder_modules))
        self.encoder_model.apply(self.init_weights)

        # pre_encoder
        n_stacks = len(self.enc_dims) - 1 # 4
        # internal layers in encoder
        for i in range(n_stacks - 1): # 3
            self.pre_encoder_modules.append(nn.Linear(self.enc_dims[i], self.enc_dims[i + 1]))
            self.pre_encoder_modules.append(nn.ReLU())
        # pre_encoder model
        self.pre_encoder_model = nn.Sequential(*(self.pre_encoder_modules))
        
        self.z_mean_fc = nn.Linear(self.enc_dims[-2], self.enc_dims[-1])
        self.z_log_var_fc = nn.Linear(self.enc_dims[-2], self.enc_dims[-1])
        
        # MLP
        self.mlp_dims[0] = self.mlp_dims[0] * 2
        m_stacks = len(self.mlp_dims) - 1
        for i in range(m_stacks - 1):
            self.mlp_modules.append(nn.Linear(self.mlp_dims[i], self.mlp_dims[i + 1]))
            self.mlp_modules.append(nn.ReLU())
            if dropout > 0:
                self.mlp_modules.append(nn.Dropout(p=dropout))
        # mlp output
        self.mlp_modules.append(nn.Linear(self.mlp_dims[-2], self.mlp_dims[-1]))
        self.mlp_modules.append(nn.Softmax(dim=1))
        self.mlp_model = nn.Sequential(*(self.mlp_modules))

        if self.verbose:
            print(self.encoder_model)
            print(self.pre_encoder_model)
            print(self.mlp_model)
        return

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        return
    
    def discrete_cmap(self, N, base_cmap=None):

        # Note that if base_cmap is a string or None, you can simply do
        #    return plt.cm.get_cmap(base_cmap, N)
        # The following works for string, None, or a colormap instance:

        base = plt.cm.get_cmap(base_cmap)
        color_list = base(np.linspace(0, 1, N))
        cmap_name = base.name + str(N)
        return base.from_list(cmap_name, color_list, N)
    
    
    def eval_plot(self, x_tensor_cuda, y_bin, epoch, args):
        with torch.no_grad():
            x_tensor = x_tensor_cuda
            #1
            _, _, _, _, x_encoded_tensor = self.encode(x_tensor, is_all_return = True)
            x_encoded = x_encoded_tensor.detach().cpu().float().numpy()

            N = len(np.unique(y_bin))
            
            for i in range(0, x_encoded.shape[1], 2):
                plt.scatter(x_encoded[:,i], x_encoded[:,i+1], c = y_bin, alpha=0.2, s=10, marker='o', edgecolor='none', cmap=self.discrete_cmap(N, 'jet'))
                #plt.colorbar(ticks=range(N))
                plt.xlabel('x'+str(i+1))
                plt.ylabel('x'+str(i+2))
                plt.colorbar()
                #plt.show()
                
                directory_path = "./maps/triplet_kld/"+ str(x_encoded.shape[1]) + "/epoch_" + str(epoch)
                # 디렉토리 생성
                if not os.path.exists(directory_path):
                    os.makedirs(directory_path)
                
                file_name = directory_path + "/dim_" + str(i+1) + "-" + str(i+2) + "_for_x"
                plt.savefig(file_name)
                plt.clf()
                if i==14:
                    break
                
            self.pre_encoded = self.pre_encoder_model(x_tensor)
            
            z_mean = self.z_mean_fc(self.pre_encoded)
            z_log_var = self.z_log_var_fc(self.pre_encoded)
            
            z = self.reparameterize(z_mean, z_log_var)
            z = z.detach().cpu().float().numpy()

            #markers = ['o' if y == 0 else 'X' for y in y_bin]
            N = len(np.unique(y_bin))
            for i in range(0, z.shape[1], 2):
                plt.scatter(z[:,i], z[:,i+1], c = y_bin, alpha=0.2, s=10, marker='o', edgecolor='none', cmap=self.discrete_cmap(N, 'jet'))
                #plt.colorbar(ticks=range(N))
                plt.xlabel('z'+str(i))
                plt.ylabel('z'+str(i+1))
                plt.colorbar()
                #plt.show()
                
                file_name = directory_path + "/dim_" + str(i+1) + "-" + str(i+2) + "_for_z"
                plt.savefig(file_name)
                plt.clf()
                if i==14:
                    break
    
    
    def forward(self, x):
        kld_encoded, _, z_mean, z_log_var, c_encoded = self.encode(x, is_all_return = True)
        
        z_concat = torch.cat((c_encoded, kld_encoded), dim=1)
        self.out = self.mlp_model(z_concat)
        
        return kld_encoded, x, z_mean, z_log_var, c_encoded, self.out
    
    def predict_proba(self, x):
        kld_encoded, _, z_mean, z_log_var, c_encoded = self.encode(x, is_all_return = True)
        z_concat = torch.cat((c_encoded, kld_encoded), dim=1)
        mlp_out = self.mlp_model(z_concat)
        mlp_out = torch.clamp(mlp_out, min=1e-5, max=1 - 1e-5)
        return mlp_out
        
    
    def predict(self, x):
        self.out = self.predict_proba(x)
        preds = self.out.max(1)[1]
        return preds
    
    def reparameterize(self, z_mean, z_log_var):
        std = torch.exp(0.5*z_log_var)
        eps = torch.randn_like(std) * (math.sqrt(1/2) ** (self.two_x))
        #eps = torch.randn_like(std)
        eps = eps * self.kld_scale
        
        return z_mean + eps*std
    
    def encode_c(self, x):
        c_encoded = self.encoder_model(x)
        return c_encoded
    
    def encode_kld(self, x):
        pre_encoded = self.pre_encoder_model(x)
        z_mean = self.z_mean_fc(pre_encoded)
        z_log_var = self.z_log_var_fc(pre_encoded)
        kld_encoded = self.reparameterize(z_mean, z_log_var)

        return kld_encoded, z_mean, z_log_var
    
    
    def encode(self, x, is_all_return = False):
        
        c_encoded = self.encode_c(x)
        kld_encoded, z_mean, z_log_var = self.encode_kld(x)
        
        if is_all_return == False:
            return c_encoded
        
        return kld_encoded, x, z_mean, z_log_var, c_encoded
    
    
