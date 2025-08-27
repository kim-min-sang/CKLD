
import torch
import numpy as np
import os
import math

def get_kld_weights(z_hc, X_train_tensor, y_train_binary_tensor):
    ###
    # kld weights 구하기
    # 나중엔 이게 fam이 아니라 ben이면 재활용 할수도?
    y_bin_benign_indices = torch.where(y_train_binary_tensor[:,1] == 0)[0]
    y_bin_malware_indices = torch.where(y_train_binary_tensor[:,1] == 1)[0]
    
    '''
    import random
    r = random.randint(1,100)
    if r == 1:
        print(f"y_bin is {y_bin.shape}")
        print(f"y_fam is {y_fam}")
    '''
    
    
    #print(f"c_encoded_benigns is {c_encoded.shape}\n{c_encoded}")
    
    
    # hc_encoder 상에서의 잠재 변수 기준으로 kld의 평균을 구한다
    
    
    train_benigns = z_hc[y_bin_benign_indices]
    train_malwares = z_hc[y_bin_malware_indices]
    
    ben_means = train_benigns.mean(dim=0)
    mal_means = train_malwares.mean(dim=0)
    
    mean_gaps = torch.abs(ben_means - mal_means)
    
    kld_weights = (mean_gaps / torch.mean(mean_gaps))
    
    ###
    
    return kld_weights


'''

def get_mids_for_y_class(z_hc, mid_type, y_train_tensor, y_train_binary_tensor):
    ###
    # mids_for_y_fam 구하기
    
    if mid_type == "fam":
        y_class_tensor = y_train_tensor
    elif mid_type == "bin":
        y_class_tensor = y_train_binary_tensor[:,1]
    else:
        raise ValueError(f"Invalid mid_type: {args.mid_type!r}. Expected 'fam' or 'bin'.")
    
    y_class_unique, _ = torch.unique(y_class_tensor, return_inverse=True)
    
    mids_for_y_class = {}
    for label in y_class_unique:
        indices = torch.where(y_class_tensor == label)[0]  # label에 해당하는 인덱스들 추출
        mean_vector = z_hc[indices].mean(dim=0)          # 해당 인덱스들의 평균값 계산
        mids_for_y_class[int(label.item())] = mean_vector # Python의 기본 자료형인 int로 변환하여 딕셔너리에 저장
    ###
    return mids_for_y_class
'''



# mean
def get_mids_for_y_class(z_hc, mid_type, y_train_tensor, y_train_binary_tensor):
    # 1) 클래스별 레이블 결정
    if mid_type == "fam":
        y_cls = y_train_tensor
    elif mid_type == "bin":
        y_cls = y_train_binary_tensor[:,1]
    else:
        raise ValueError(f"Invalid mid_type: {mid_type!r}. Expected 'fam' or 'bin'.")
    
    # 2) GPU/CPU 일치시키기
    device = z_hc.device
    y_cls = y_cls.to(device)
    
    # 3) unique label + inverse index 추출
    y_unique, y_inv = torch.unique(y_cls, sorted=True, return_inverse=True)
    #   y_unique: (C,), y_inv: (N,) — y_inv[i] ∈ [0..C-1]
    
    # 4) 클래스별 합계 계산
    C, D = y_unique.size(0), z_hc.size(1)
    sums = torch.zeros(C, D, device=device)               # (C, D)
    sums = sums.index_add(0, y_inv, z_hc)                  # sums[c] += z_hc[i] where y_inv[i]==c
    
    # 5) 클래스별 개수 계산
    counts = torch.bincount(y_inv, minlength=C).unsqueeze(1)  # (C, 1)
    
    # 6) 평균값
    mids = sums / counts                                    # (C, D)
    
    #print(f"mids is {mids.shape}\n{mids}")
    
    mids_for_y_class = mids[y_inv]
    #print(f"mids_for_y_class is {mids_for_y_class.shape}\n{mids_for_y_class}")
    
    return mids_for_y_class, mids, y_unique


'''
# median
def get_mids_for_y_class(z_hc, mid_type, y_train_tensor, y_train_binary_tensor):
    # 1) 클래스별 레이블 결정
    if mid_type == "fam":
        y_cls = y_train_tensor
    elif mid_type == "bin":
        y_cls = y_train_binary_tensor[:,1]
    else:
        raise ValueError(f"Invalid mid_type: {mid_type!r}. Expected 'fam' or 'bin'.")

    # 2) GPU/CPU 일치
    device = z_hc.device
    y_cls = y_cls.to(device)

    # 3) unique + inverse index
    y_unique, y_inv = torch.unique(y_cls, sorted=True, return_inverse=True)
    C, D = y_unique.size(0), z_hc.size(1)

    # 4) 클래스별 중앙값(median) 계산
    mids = torch.zeros(C, D, device=device)
    for i, cls in enumerate(y_unique):
        group = z_hc[y_cls == cls]          # cls에 속하는 샘플들 (n_i, D)
        mids[i] = group.median(dim=0).values  # (D,)

    # 5) 원본 샘플 순서대로 재배치
    mids_for_y_class = mids[y_inv]          # (N, D)

    return mids_for_y_class, mids, y_unique

'''



def get_mids_for_y_class_batch(mids, y_unique, mid_type, y_train_batch_tensor, y_train_binary_batch_tensor):
    # 1) 클래스별 레이블 결정
    if mid_type == "fam":
        y_cls_batch = y_train_batch_tensor
    elif mid_type == "bin":
        y_cls_batch = y_train_binary_batch_tensor[:,1]
    else:
        raise ValueError(f"Invalid mid_type: {mid_type!r}. Expected 'fam' or 'bin'.")
    
    #    브로드캐스트 비교로 각 labels가 y_unique의 몇 번째인지 구하기  
    #    labels.unsqueeze(1): (N,1), y_unique.unsqueeze(0): (1,C)  
    eq   = y_cls_batch.unsqueeze(1) == y_unique.unsqueeze(0)  # (N, C)  
    idxs = eq.float().argmax(dim=1)                     # (N,)  

    # 3) mids에서 한 번에 인덱싱  
    mids_for_y_class_batch = mids[idxs]                                   # (N, D)
    
    return mids_for_y_class_batch
    
    
    
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
    
    
def get_kld_dev_scale(mid_type, enc_dim_last, kld_scale):
    
    two_x = find_exponent_of_two(enc_dim_last)
    kld_dev_scale_before = math.sqrt(1/2) ** (two_x) * kld_scale
    
    if mid_type == "bin":
        #kld_dev_scale = kld_dev_scale_before * y_fam_unique_num
        kld_dev_scale = kld_dev_scale_before
    elif mid_type == "fam":
        kld_dev_scale = kld_dev_scale_before
    else:
        raise ValueError(f"get_kld_dev_scale(), mid_type error")
    return kld_dev_scale
    