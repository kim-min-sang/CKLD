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
    data_name, start_month, end_month, folder='data/', top_k=None
):
    """
    주어진 월 범위의 데이터셋을 불러온 후,
    y_train에서 샘플 수가 많은 상위 top_k 클래스만 선택하여 반환합니다.
    """
    # 1) 파일 경로 구성
    if start_month != end_month:
        dataset_name = f'{start_month}to{end_month}'
    else:
        dataset_name = f'{start_month}'
    saved_data_file = os.path.join(
        folder, data_name, f'{dataset_name}_selected.npz'
    )

    # 2) 데이터 로드
    data = np.load(saved_data_file, allow_pickle=True)
    X_train = data['X_train']
    y_train = data['y_train']
    y_mal_family = data['y_mal_family']

    if top_k != None:
        # 3) y_train에서 상위 top_k 클래스 선택
        counter = Counter(y_train)
        top_labels = [lbl for lbl, _ in counter.most_common(top_k)]
        
        # 4) 필터링 마스크 생성 및 적용
        mask = np.isin(y_train, top_labels)
        #mask_for_fam = np.isin(y_mal_family, top_labels)
        X_train = X_train[mask]
        y_train = y_train[mask]
        y_mal_family = y_mal_family[mask[:len(y_mal_family)]]
        
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
        """
        tensors: X, y, y_bin 등 기본 TensorDataset에 들어갈 텐서들
        mids: (N, D) 크기의 mids_for_each_sample 텐서
        """
        super().__init__(*tensors)
        self.mids_for_y_class = mids_for_y_class

    def __getitem__(self, idx):
        # 기본 TensorDataset 동작으로 먼저 X, y, y_bin 등을 가져오고
        items = super().__getitem__(idx)
        # 거기에 mids[idx]를 추가로 덧붙여 리턴
        return (*items, self.mids_for_y_class[idx])

    def update_mids_for_y_class(self, mids_for_y_class: torch.Tensor):
        # 새로운 mids_for_each_sample 텐서로 교체
        # new_mids.shape == (len(self), D) 여야 합니다.
        self.mids_for_y_class = mids_for_y_class
        
        
        
        
        
        
        
  
def find_exponent_of_two(enc_dims):
    if enc_dims <= 0:
        raise ValueError("Input must be a positive integer")
    
    # 로그를 이용하여 n 계산
    n = math.log(enc_dims, 2)
    
    # enc_dims 정확히 2의 거듭제곱인지 확인
    if n.is_integer():
        return int(n)
    else:
        raise ValueError(f"{enc_dims} is not a power of 2")
