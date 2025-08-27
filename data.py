"""
data.py
~~~~~~~

Functions for loading data.

"""

import numpy as np
import os
import torch
import math
from collections import Counter



def load_range_dataset_w_benign(
    data_name, start_month, end_month, folder='data/'
):
    if start_month != end_month:
        dataset_name = f'{start_month}to{end_month}'
    else:
        dataset_name = f'{start_month}'
    saved_data_file = os.path.join(
        folder, data_name, f'{dataset_name}_selected.npz'
    )

    data = np.load(saved_data_file, allow_pickle=True)
    X_train = data['X_train']
    y_train = data['y_train']
    y_mal_family = data['y_mal_family']
        
    return X_train, y_train, y_mal_family



'''
def load_range_dataset_w_benign(data_name, start_month, end_month, folder='data/'):
    if start_month != end_month:
        dataset_name = f'{start_month}to{end_month}'
    else:
        dataset_name = f'{start_month}'
    saved_data_file = os.path.join(folder, data_name, f'{dataset_name}_selected.npz')
    data = np.load(saved_data_file, allow_pickle=True)
    X_train, y_train = data['X_train'], data['y_train']
    y_mal_family = data['y_mal_family']
    return X_train, y_train, y_mal_family
'''

from torch.utils.data import TensorDataset, DataLoader

class CustomDataset(TensorDataset):
    def __init__(self, *tensors, mids_for_y_class):

        super().__init__(*tensors)
        self.mids_for_y_class = mids_for_y_class

    def __getitem__(self, idx):
        items = super().__getitem__(idx)
        return (*items, self.mids_for_y_class[idx])

    def update_mids_for_y_class(self, mids_for_y_class: torch.Tensor):
        self.mids_for_y_class = mids_for_y_class
        
        
        
        
        
        
        
  
def find_exponent_of_two(enc_dims):
    if enc_dims <= 0:
        raise ValueError("Input must be a positive integer")
    
    n = math.log(enc_dims, 2)
    
    if n.is_integer():
        return int(n)
    else:
        raise ValueError(f"{enc_dims} is not a power of 2")
