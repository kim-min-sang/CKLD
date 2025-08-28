
import torch
import numpy as np
import os
import math

def get_kld_weights(z_hc, X_train_tensor, y_train_binary_tensor):
    y_bin_benign_indices = torch.where(y_train_binary_tensor[:,1] == 0)[0]
    y_bin_malware_indices = torch.where(y_train_binary_tensor[:,1] == 1)[0]
    
    
    train_benigns = z_hc[y_bin_benign_indices]
    train_malwares = z_hc[y_bin_malware_indices]
    
    ben_means = train_benigns.mean(dim=0)
    mal_means = train_malwares.mean(dim=0)
    
    mean_gaps = torch.abs(ben_means - mal_means)
    
    kld_weights = (mean_gaps / torch.mean(mean_gaps))
    
    
    return kld_weights




def get_mids_for_y_class(z_hc, centroid_type, y_train_tensor, y_train_binary_tensor):
    if centroid_type == "fam":
        y_cls = y_train_tensor
    elif centroid_type == "bin":
        y_cls = y_train_binary_tensor[:,1]
    else:
        raise ValueError(f"Invalid centroid_type: {centroid_type!r}. Expected 'fam' or 'bin'.")
    
    device = z_hc.device
    y_cls = y_cls.to(device)
    
    y_unique, y_inv = torch.unique(y_cls, sorted=True, return_inverse=True)
    
    C, D = y_unique.size(0), z_hc.size(1)
    sums = torch.zeros(C, D, device=device)               # (C, D)
    sums = sums.index_add(0, y_inv, z_hc)                  # sums[c] += z_hc[i] where y_inv[i]==c
    
    counts = torch.bincount(y_inv, minlength=C).unsqueeze(1)  # (C, 1)
    
    mids = sums / counts                                    # (C, D)
    
    
    mids_for_y_class = mids[y_inv]
    
    return mids_for_y_class, mids, y_unique




def get_mids_for_y_class_batch(mids, y_unique, centroid_type, y_train_batch_tensor, y_train_binary_batch_tensor):
    if centroid_type == "fam":
        y_cls_batch = y_train_batch_tensor
    elif centroid_type == "bin":
        y_cls_batch = y_train_binary_batch_tensor[:,1]
    else:
        raise ValueError(f"Invalid centroid_type: {centroid_type!r}. Expected 'fam' or 'bin'.")
    
    eq   = y_cls_batch.unsqueeze(1) == y_unique.unsqueeze(0)  # (N, C)  
    idxs = eq.float().argmax(dim=1)                     # (N,)  

    mids_for_y_class_batch = mids[idxs]                                   # (N, D)
    
    return mids_for_y_class_batch
    
    
    
def find_exponent_of_two(enc_dims):
    if enc_dims <= 0:
        raise ValueError("Input must be a positive integer")
    
    n = math.log(enc_dims, 2)
    
    if n.is_integer():
        return int(n)
    else:
        raise ValueError(f"{enc_dims} is not a power of 2")
    
    
def get_kld_dev_scale(centroid_type, enc_dim_last, kld_scale):
    
    two_x = find_exponent_of_two(enc_dim_last)
    kld_dev_scale_before = math.sqrt(1/2) ** (two_x) * kld_scale
    
    if centroid_type == "bin":
        kld_dev_scale = kld_dev_scale_before
    elif centroid_type == "fam":
        kld_dev_scale = kld_dev_scale_before
    else:
        raise ValueError(f"get_kld_dev_scale(), centroid_type error")
    return kld_dev_scale
    
