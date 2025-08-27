import logging
import torch
import torch.nn as nn

import data
import math
import random
import gc








class TripletLoss(nn.Module):
    def __init__(self, reduce = 'mean'):
        """
        If reduce == False, we calculate sample loss, instead of batch loss.
        """
        super(TripletLoss, self).__init__()
        self.reduce = reduce

    def forward(self, features, labels = None, margin = 10.0,
                weight = None, split = None):
        """
        Triplet loss for model.

        Args:
            features: hidden vector of shape [bsz, feature_dim]. e.g., (512, 128)
            labels: ground truth of shape [bsz].
            weight: sample weights to adjust the sample loss values
            split: whether it is in test data
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        
        batch_size = features.shape[0]

        pass_size = batch_size // 3
        """
        three shares of pass_size
        1) training data sample
        2) positive samples
        3) negative samples
        """
        anchor = features[:pass_size]
        positive = features[pass_size:pass_size*2]
        negative = features[pass_size*2:]
        positive_losses = torch.maximum(torch.tensor(1e-10), torch.linalg.norm(anchor - positive, ord = 2, dim = 1))
        negative_losses = torch.maximum(torch.tensor(0), margin - torch.linalg.norm(anchor - negative, ord = 2, dim = 1))

        if weight is not None:
            anchor_weight = weight[:pass_size]
            positive_weight = weight[pass_size:pass_size*2]
            negative_weight = weight[pass_size*2:]
            positive_losses = positive_losses * anchor_weight * positive_weight
            negative_losses = negative_losses * positive_weight * negative_weight
        
        loss = positive_losses + negative_losses

        if self.reduce == 'mean':
            loss = loss.mean()

        return loss

class TripletMSELoss(nn.Module):
    def __init__(self, reduce = 'mean'):
        super(TripletMSELoss, self).__init__()
        # reduce: whether use 'mean' reduction or keep sample loss
        self.reduce = reduce

    def forward(self, cae_lambda,
            x, x_prime,
            features, labels = None,
            margin = 10.0,
            weight = None,
            split = None):
        """
        Args:
            cae_lambda: scale the CAE loss
            x: input to the Autoencoder
            x_prime: decoded x' from Autoencoder
            features: hidden vector of shape [bsz, feature_dim].
            labels: ground truth of shape [bsz].
            weight: sample weights to adjust the sample loss values
            split: whether it is in test data
        Returns:
            A loss scalar.
        """
        Triplet = TripletLoss(reduce = self.reduce)
        supcon_loss = Triplet(features, labels = labels, margin = margin, weight = weight, split = split)

        mse_loss = torch.nn.functional.mse_loss(x, x_prime, reduction = self.reduce)
        
        loss = cae_lambda * supcon_loss + mse_loss
        
        del Triplet
        torch.cuda.empty_cache()

        return loss, supcon_loss, mse_loss
    
    
    
    
    
    

    
class TripletMseKldEnsembleXentLoss(nn.Module):
    def __init__(self, reduce = 'mean', sample_reduce = 'mean'):
        super(TripletMseKldEnsembleXentLoss, self).__init__()
        # reduce: whether use 'mean' reduction or keep sample loss
        self.reduce = reduce
        self.sample_reduce = sample_reduce

    def forward(self, cae_lambda, xent_lambda, mse_lambda, 
            y_bin_pred, y_bin_batch,
            features, c_encoded, labels = None,
            x = None, x_recon = None,
            z_mean = None, z_log_var = None,
            margin = 10.0,
            margin_btw = 2.0,
            weight = None,
            split = None,
            is_for_selector = False,
            is_train_first = False, epoch = None, mids = None, kld_dev_scale = None, mids_for_y_class_batch = None):
        """
        Args:
            xent_lambda: scale the binary xent loss
            y_bin_pred: predicted MLP output
            y_bin_batch: binary one-hot encoded y
            features: hidden vector of shape [bsz, feature_dim]. # [1024, 128]
            labels: ground truth of shape [bsz]. # bsz means batch size
            margin: margin for HiDistanceLoss.
            weight: sample weights to adjust the sample loss values
            split: whether it is in test data, so we ignore these entries
        Returns:
            A loss scalar.
        """
        
        Triplet = TripletLoss(reduce = self.reduce)
        supcon_loss = Triplet(c_encoded, labels = labels, margin = margin, weight = weight, split = split)

        mse_loss = torch.nn.functional.mse_loss(x, x_recon, reduction = self.reduce)
        
        KldCustom = KldCustomEnsemble6Loss(reduce = self.reduce, sample_reduce = self.sample_reduce)

        is_z_mean_static = None
        
        if mids == False:
            if torch.equal(c_encoded, z_mean):
                is_z_mean_static = True
            else:
                is_z_mean_static = False
        
        
        kld_loss = KldCustom(z_mean, z_log_var, c_encoded, is_z_mean_static = is_z_mean_static, mids = mids, kld_dev_scale = kld_dev_scale, kld_weights = None, mids_for_y_class_batch = mids_for_y_class_batch)
        
        xent_bin_loss = torch.nn.functional.binary_cross_entropy(y_bin_pred[:, 1], y_bin_batch[:, 1], reduction = self.reduce, weight = weight)
        
        if self.reduce == 'mean':
            xent_bin_loss = xent_bin_loss.mean()
            
        Gap = GapLoss(reduce = self.reduce, sample_reduce = self.sample_reduce)
        gap_loss = Gap(c_encoded, y_bin_batch)
        
        try:
            if torch.isnan(gap_loss):
                gap_loss = torch.tensor(0.0, requires_grad=True).cuda()
        except Exception as e:
            pass
            
        try:
            if gap_loss == None:
                gap_loss = torch.tensor(0.0, requires_grad=True).cuda()
        except Exception as e:
            #print("if gap_loss == None error!!!")
            pass
        
        loss = cae_lambda * supcon_loss + mse_lambda * mse_loss + 1 * kld_loss + xent_lambda * xent_bin_loss + gap_loss
        
        
        
        
        del Triplet
        del KldCustom
        del Gap
        torch.cuda.empty_cache()
        
        return loss, supcon_loss, mse_loss, kld_loss, xent_bin_loss, gap_loss
    
    
    
    
    
class TripletMseXentLoss(nn.Module):
    def __init__(self, reduce = 'mean', sample_reduce = 'mean'):
        super(TripletMseXentLoss, self).__init__()
        # reduce: whether use 'mean' reduction or keep sample loss
        self.reduce = reduce
        self.sample_reduce = sample_reduce

    def forward(self, cae_lambda, xent_lambda, mse_lambda,
            y_bin_pred, y_bin_batch,
            c_encoded, labels = None,
            x = None, x_recon = None,
            margin = 10.0,
            margin_btw = 2.0,
            weight = None,
            split = None,
            is_for_selector = False,
            is_train_first = False, epoch = None):
        
        """
        Args:
            xent_lambda: scale the binary xent loss
            y_bin_pred: predicted MLP output
            y_bin_batch: binary one-hot encoded y
            features: hidden vector of shape [bsz, feature_dim]. # [1024, 128]
            labels: ground truth of shape [bsz]. # bsz means batch size
            margin: margin for HiDistanceLoss.
            weight: sample weights to adjust the sample loss values
            split: whether it is in test data, so we ignore these entries
        Returns:
            A loss scalar.
        """
        
        Triplet = TripletLoss(reduce = self.reduce)
        supcon_loss = Triplet(c_encoded, labels = labels, margin = margin, weight = weight, split = split)

        mse_loss = torch.nn.functional.mse_loss(x, x_recon, reduction = self.reduce)
        
        xent_bin_loss = torch.nn.functional.binary_cross_entropy(y_bin_pred[:, 1], y_bin_batch[:, 1], reduction = self.reduce, weight = weight)
        
        if self.reduce == 'mean':
            xent_bin_loss = xent_bin_loss.mean()
        
        loss = cae_lambda * supcon_loss + mse_lambda * mse_loss + xent_lambda * xent_bin_loss
        
        
        
        
        del Triplet
        torch.cuda.empty_cache()
        
        return loss, supcon_loss, mse_loss, xent_bin_loss
    
    
    
    
    

class TripletXentLoss(nn.Module):
    def __init__(self, reduce = 'mean', sample_reduce = 'mean'):
        super(TripletXentLoss, self).__init__()
        self.reduce = reduce
        self.sample_reduce = sample_reduce

    def forward(self, triplet_lambda, xent_lambda,
            y_bin_pred, y_bin_batch,
            c_encoded, labels = None,
            x = None,
            margin = 10.0,
            margin_btw = 2.0,
            weight = None,
            split = None,
            is_for_selector = False,
            is_train_first = False, epoch = None):
        """
        Args:
            xent_lambda: scale the binary xent loss
            y_bin_pred: predicted MLP output
            y_bin_batch: binary one-hot encoded y
            features: hidden vector of shape [bsz, feature_dim]. # [1024, 128]
            labels: ground truth of shape [bsz]. # bsz means batch size
            margin: margin for HiDistanceLoss.
            weight: sample weights to adjust the sample loss values
            split: whether it is in test data, so we ignore these entries
        Returns:
            A loss scalar.
        """
        
        Triplet = TripletLoss(reduce = self.reduce)
        supcon_loss = Triplet(c_encoded, labels = labels, margin = margin, weight = weight, split = split)

        xent_bin_loss = torch.nn.functional.binary_cross_entropy(y_bin_pred[:, 1], y_bin_batch[:, 1], reduction = self.reduce, weight = weight)
        
        if self.reduce == 'mean':
            xent_bin_loss = xent_bin_loss.mean()
        
        loss = triplet_lambda * supcon_loss + xent_lambda * xent_bin_loss
        
        del Triplet
        torch.cuda.empty_cache()
        
        return loss, supcon_loss, xent_bin_loss
    








class TripletKldEnsembleXentLoss(nn.Module):
    def __init__(self, reduce = 'mean', sample_reduce = 'mean'):
        super(TripletKldEnsembleXentLoss, self).__init__()
        # reduce: whether use 'mean' reduction or keep sample loss
        self.reduce = reduce
        self.sample_reduce = sample_reduce

    def forward(self, triplet_lambda, xent_lambda,
            y_bin_pred, y_bin_batch,
            features, c_encoded, labels = None,
            x = None, 
            z_mean = None, z_log_var = None,
            margin = 10.0,
            margin_btw = 2.0,
            weight = None,
            split = None,
            is_for_selector = False,
            is_train_first = False, epoch = None, mids = None, kld_dev_scale = None, mids_for_y_class_batch = None):
        """
        Args:
            xent_lambda: scale the binary xent loss
            y_bin_pred: predicted MLP output
            y_bin_batch: binary one-hot encoded y
            features: hidden vector of shape [bsz, feature_dim]. # [1024, 128]
            labels: ground truth of shape [bsz]. # bsz means batch size
            margin: margin for HiDistanceLoss.
            weight: sample weights to adjust the sample loss values
            split: whether it is in test data, so we ignore these entries
        Returns:
            A loss scalar.
        """
        
        Triplet = TripletLoss(reduce = self.reduce)
        supcon_loss = Triplet(c_encoded, labels = labels, margin = margin, weight = weight, split = split)

        KldCustom = KldCustomEnsemble6Loss(reduce = self.reduce, sample_reduce = self.sample_reduce)

        is_z_mean_static = None
        
        if mids == False:
            if torch.equal(c_encoded, z_mean):
                is_z_mean_static = True
            else:
                is_z_mean_static = False
        
        
        kld_loss = KldCustom(z_mean, z_log_var, c_encoded, is_z_mean_static = is_z_mean_static, mids = mids, kld_dev_scale = kld_dev_scale, kld_weights = None, mids_for_y_class_batch = mids_for_y_class_batch)
        
        xent_bin_loss = torch.nn.functional.binary_cross_entropy(y_bin_pred[:, 1], y_bin_batch[:, 1], reduction = self.reduce, weight = weight)
        
        if self.reduce == 'mean':
            xent_bin_loss = xent_bin_loss.mean()
            
        Gap = GapLoss(reduce = self.reduce, sample_reduce = self.sample_reduce)
        gap_loss = Gap(c_encoded, y_bin_batch)
        
        try:
            if torch.isnan(gap_loss):
                gap_loss = torch.tensor(0.0, requires_grad=True).cuda()
        except Exception as e:
            #print("if torch.isnan(gap_loss) error!!!")
            pass
            
        try:
            if gap_loss == None:
                gap_loss = torch.tensor(0.0, requires_grad=True).cuda()
        except Exception as e:
            #print("if gap_loss == None error!!!")
            pass
        
        loss = triplet_lambda * supcon_loss + 1 * kld_loss + xent_lambda * xent_bin_loss + gap_loss
        #loss = triplet_lambda * supcon_loss + 1 * kld_loss + xent_lambda * xent_bin_loss
        
        del Triplet
        del KldCustom
        del Gap
        torch.cuda.empty_cache()
        
        return loss, supcon_loss, kld_loss, xent_bin_loss, gap_loss
    
    
    
    

class HiDistanceLoss(nn.Module):
    def __init__(self, reduce = 'mean', sample_reduce='mean'):
        """
        If reduce == False, we calculate sample loss, instead of batch loss.
        """
        super(HiDistanceLoss, self).__init__()
        self.reduce = reduce
        self.sample_reduce = sample_reduce

    def forward(self, features, binary_cat_labels, labels = None, margin = 3.0, margin_btw = 2.0, weight = None, split = None):
        """
        Pair distance loss.

        Args:
            features: hidden vector of shape [bsz, feature_dim]. e.g., (512, 128) # in here, (1024, 128)
            binary_cat_labels: one-hot binary labels.
            labels: ground truth of shape [bsz].
            margin: margin for dissimilar distance.
            weight: sample weights to adjust the sample loss values
            split: whether it is in test data, so we ignore entries for these
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        # labels is 0 or 1
        if labels == None:
            raise ValueError('Need to define labels in DistanceLoss')

        batch_size = features.shape[0] # 1024
        
        labels = labels.contiguous().view(-1, 1) # (1024, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        
        # similar masks
        # mask_{i,j}=1 if sample j has the same class as sample i.
        binary_labels = binary_cat_labels[:, 1].view(-1, 1) # [1024, 1]
        # mask: both malware, or both benign # [1024, 1024]
        binary_mask = torch.eq(binary_labels, binary_labels.T).float().to(device)
        # multi_mask: same malware family, or benign # [1024, 1024]
        multi_mask = torch.eq(labels, labels.T).float().to(device)
        
        # malware but not the same family. does not have benign.
        other_mal_mask = binary_mask - multi_mask
        # both benign samples
        ben_labels = torch.logical_not(binary_labels).float().to(device)
        same_ben_mask = torch.matmul(ben_labels, ben_labels.T)
        # same malware family mask
        same_mal_fam_mask = multi_mask - same_ben_mask
        
        # logging.debug("=== new batch ===")
        # pseudo loss
        if self.reduce == 'none':
            tmp = other_mal_mask
            other_mal_mask = same_mal_fam_mask
            same_mal_fam_mask = tmp
            
        # dissimilar mask. malware vs benign binary labels
        binary_negate_mask = torch.logical_not(binary_mask).float().to(device)
        # multi_negate_mask = torch.logical_not(multi_mask).float().to(device)

        # mask-out self-contrast cases
        diag_mask = torch.logical_not(torch.eye(batch_size)).float().to(device)
        # similar mask
        binary_mask = binary_mask * diag_mask
        multi_mask = multi_mask * diag_mask
        other_mal_mask = other_mal_mask * diag_mask
        same_ben_mask = same_ben_mask * diag_mask
        same_mal_fam_mask = same_mal_fam_mask * diag_mask

        # adjust the masks based on test indices
        if split is not None: # split = None
            split_index = torch.nonzero(split, as_tuple=True)[0]
            # instance-level loss, paired with training samples, pseudo loss
            # logging.debug(f'split_index, {split_index}')
            binary_negate_mask[:, split_index] = 0
            # multi_negate_mask[:, split_index] = 0
            binary_mask[:, split_index] = 0
            multi_mask[:, split_index] = 0
            other_mal_mask[:, split_index] = 0
            same_ben_mask[:, split_index] = 0
            same_mal_fam_mask[:, split_index] = 0

        # reference: https://github.com/Lightning-AI/metrics/blob/master/src/torchmetrics/functional/pairwise/euclidean.py
        # not taking the sqrt for numerical stability
        x = features # [1024,128]
        y = features # y is why y=features? -> aha in order to make [1024, 1024] using x_norm, y_norm
        # norm means to do each x^2 and plus each x values and use root
        # resultly, norm means formula that get a distance betweend dimensons ex) 1:2:root3
        x_norm = x.norm(dim=1, keepdim=True) # [1024,1]
        y_norm = y.norm(dim=1).T # [1,1024] = [1024]
        
        """
        reduce: mean
        if reduce is mean, it's a training case.
        because we have to get mean value of all case
        
        reduce: none
        if reduce is none, it's a selector case.
        so we can select only lose[0]
        """
        # it's the Euclidean distance matrix
        distance_matrix = x_norm * x_norm + y_norm * y_norm - 2 * x.mm(y.T)
        distance_matrix = torch.maximum(torch.tensor(1e-10), distance_matrix)

        # default is to compute mean for these values per sample
        if self.sample_reduce == 'mean' or self.sample_reduce == None:
            if weight == None:
                # .sum(1) means if value == True, add the values
                sum_same_ben = torch.maximum(
                                    torch.sum(same_ben_mask * distance_matrix, dim=1) - \
                                            same_ben_mask.sum(1) * torch.tensor(margin),
                                    torch.tensor(0))
                sum_other_mal = torch.maximum(
                                    torch.sum(other_mal_mask * distance_matrix, dim=1) - \
                                            other_mal_mask.sum(1) * torch.tensor(margin),
                                    torch.tensor(0))
                sum_same_mal_fam = torch.sum(same_mal_fam_mask * distance_matrix, dim=1)
                sum_bin_neg = torch.maximum(
                                    binary_negate_mask.sum(1) * torch.tensor(margin_btw * margin) - \
                                            torch.sum(binary_negate_mask * distance_matrix,
                                                    dim=1),
                                    torch.tensor(0))

            else:
                weight_matrix = torch.matmul(weight.view(-1, 1), weight.view(1, -1)).to(device)
                sum_same_ben = torch.maximum(
                                    torch.sum(same_ben_mask * distance_matrix * weight_matrix, dim=1) - \
                                            same_ben_mask.sum(1) * torch.tensor(margin),
                                    torch.tensor(0))
                sum_other_mal = torch.maximum(
                                    torch.sum(other_mal_mask * distance_matrix * weight_matrix, dim=1) - \
                                            other_mal_mask.sum(1) * torch.tensor(margin),
                                    torch.tensor(0))
                sum_same_mal_fam = torch.sum(same_mal_fam_mask * distance_matrix * weight_matrix, dim=1)
                weight_prime = torch.div(1.0, weight)
                weight_matrix_prime = torch.matmul(weight_prime.view(-1, 1), weight_prime.view(1, -1)).to(device)
                sum_bin_neg = torch.maximum(
                                    binary_negate_mask.sum(1) * torch.tensor(2 * margin) - \
                                            torch.sum(binary_negate_mask * distance_matrix * weight_matrix_prime,
                                                    dim=1),
                                    torch.tensor(0))
            loss = sum_same_ben / torch.maximum(same_ben_mask.sum(1), torch.tensor(1)) + \
                    sum_other_mal / torch.maximum(other_mal_mask.sum(1), torch.tensor(1)) + \
                    sum_same_mal_fam / torch.maximum(same_mal_fam_mask.sum(1), torch.tensor(1)) + \
                    sum_bin_neg / torch.maximum(binary_negate_mask.sum(1), torch.tensor(1))
        elif self.sample_reduce == 'max':
            max_same_ben = torch.maximum(
                                torch.amax(same_ben_mask * distance_matrix, 1) - \
                                        torch.tensor(margin),
                                torch.tensor(0))
            max_other_mal = torch.maximum(
                                torch.amax(other_mal_mask * distance_matrix, 1) - \
                                        torch.tensor(margin),
                                torch.tensor(0))
            max_same_mal_fam = torch.amax(same_mal_fam_mask * distance_matrix, 1)
            max_bin_neg = torch.maximum(
                                torch.tensor(2 * margin) - \
                                        torch.amin(binary_negate_mask * distance_matrix, 1),
                                torch.tensor(0))
            loss = max_same_ben + max_other_mal + max_same_mal_fam + max_bin_neg
        else:
            raise Exception(f'sample_reduce = {self.sample_reduce} not implemented yet.')

        if self.reduce == 'mean':
            loss = loss.mean()
        
        return loss

class HiDistanceXentLoss(nn.Module):
    def __init__(self, reduce = 'mean', sample_reduce = 'mean'):
        super(HiDistanceXentLoss, self).__init__()
        # reduce: whether use 'mean' reduction or keep sample loss
        self.reduce = reduce
        self.sample_reduce = sample_reduce

    def forward(self, xent_lambda,
            y_bin_pred, y_bin_batch,
            features, labels = None,
            margin = 10.0,
            weight = None,
            split = None):
        """
        Args:
            xent_lambda: scale the binary xent loss
            y_bin_pred: predicted MLP output
            y_bin_batch: binary one-hot encoded y
            features: hidden vector of shape [bsz, feature_dim]. # [1024, 128]
            labels: ground truth of shape [bsz]. # bsz means batch size
            margin: margin for HiDistanceLoss.
            weight: sample weights to adjust the sample loss values
            split: whether it is in test data, so we ignore these entries
        Returns:
            A loss scalar.
        """
        # Hi means Hierarical
        Dist = HiDistanceLoss(reduce = self.reduce, sample_reduce = self.sample_reduce)
        # try not giving any weight to HiDistanceLoss
        # in this features, the features[0] is a test sample, the others are train samples
        supcon_loss = Dist(features, y_bin_batch, labels = labels, margin = margin, weight = None, split = split)
        
        # xent means cross entropy
        xent_bin_loss = torch.nn.functional.binary_cross_entropy(y_bin_pred[:, 1], y_bin_batch[:, 1],
                                                        reduction = self.reduce, weight = weight)
        if self.reduce == 'mean':
            xent_bin_loss = xent_bin_loss.mean()

        loss = supcon_loss + xent_lambda * xent_bin_loss
        
        del Dist
        torch.cuda.empty_cache()

        return loss, supcon_loss, xent_bin_loss

    
class KldCustomEnsemble6Loss(nn.Module):
    def __init__(self, reduce = 'mean', sample_reduce = 'mean'):
        super(KldCustomEnsemble6Loss, self).__init__()
        self.reduce = reduce
        self.sample_reduce = sample_reduce

    def forward(self, z_mean, z_log_var, c_encoded, is_z_mean_static = False, mids = None, kld_dev_scale = None, kld_beta = 1.0, kld_weights = None, mids_for_y_class_batch = None):
        
        if mids == False or mids == None:
            # KL divergence loss
            if is_z_mean_static == False:
                kl_loss = -0.5 * torch.sum(1 + z_log_var - (z_mean-c_encoded).pow(2) - kld_beta * torch.exp(z_log_var), dim=-1)
            else:
                kl_loss = -0.5 * torch.sum(1 + z_log_var - kld_beta * torch.exp(z_log_var), dim=-1)
        elif mids == True:
            kl_loss = -0.5 * torch.sum(1 + z_log_var - (z_mean-mids_for_y_class_batch).pow(2) - kld_beta * torch.exp(z_log_var), dim=-1)

        # Total loss
        loss = kl_loss
        

        if self.reduce == 'mean':
            loss = loss.mean()

        return loss
    
    
    
class GapLoss(nn.Module):
    def __init__(self, reduce = 'mean', sample_reduce = 'mean'):
        super(GapLoss, self).__init__()
        self.reduce = reduce
        self.sample_reduce = sample_reduce

    def forward(self, c_encoded, y_bin):
        
        try:
            y_bin_benign_indices = torch.where(y_bin[:,1] == 0)
            y_bin_malware_indices = torch.where(y_bin[:,1] == 1)
            
            ben_per_dim = c_encoded[y_bin_benign_indices]
            mal_per_dim = c_encoded[y_bin_malware_indices]
            
            ben_mean_per_dim = ben_per_dim.median(dim=0).values
            mal_mean_per_dim = mal_per_dim.median(dim=0).values
            
            gap_per_dim = torch.abs(ben_mean_per_dim - mal_mean_per_dim)
            
            gap = gap_per_dim.mean()
            
            mse = (gap_per_dim - gap) ** 2
            loss = torch.sum(mse)
            
            if self.reduce == 'mean':
                loss = loss.mean()
        except Exception as e:
            return None
        return loss
    
    
        
        
        

class HiDistanceKldCustomXentEnsemble6Loss(nn.Module):
    def __init__(self, reduce = 'mean', sample_reduce = 'mean'):
        super(HiDistanceKldCustomXentEnsemble6Loss, self).__init__()
        # reduce: whether use 'mean' reduction or keep sample loss
        self.reduce = reduce
        self.sample_reduce = sample_reduce

    def forward(self, xent_lambda,
            y_bin_pred, y_bin_batch,
            features, c_encoded, labels = None,
            x = None,
            z_mean = None, z_log_var = None,
            margin = 3.0,
            margin_btw = 2.0,
            weight = None,
            split = None,
            is_for_selector = False,
            is_train_first = False, epoch = None, mids = None, kld_dev_scale = None, kld_beta = None,
            kld_weights = None, mids_for_y_class_batch = None):
        """
        Args:
            xent_lambda: scale the binary xent loss
            y_bin_pred: predicted MLP output
            y_bin_batch: binary one-hot encoded y
            features: hidden vector of shape [bsz, feature_dim]. # [1024, 128]
            labels: ground truth of shape [bsz]. # bsz means batch size
            margin: margin for HiDistanceLoss.
            weight: sample weights to adjust the sample loss values
            split: whether it is in test data, so we ignore these entries
        Returns:
            A loss scalar.
        """
        # Hi means Hierarical
        Dist = HiDistanceLoss(reduce = self.reduce, sample_reduce = self.sample_reduce)
        # try not giving any weight to HiDistanceLoss
        # in this features, the features[0] is a test sample, the others are train samples
        supcon_loss = Dist(c_encoded, y_bin_batch, labels = labels, margin = margin, margin_btw = margin_btw, weight = None, split = split)
        
        KldCustom = KldCustomEnsemble6Loss(reduce = self.reduce, sample_reduce = self.sample_reduce)

        is_z_mean_static = None
        
        if mids == False:
            if torch.equal(c_encoded, z_mean):
                is_z_mean_static = True
            else:
                is_z_mean_static = False
        
        kld_loss = KldCustom(z_mean, z_log_var, c_encoded, is_z_mean_static = is_z_mean_static, mids = mids, kld_dev_scale = kld_dev_scale, kld_beta = kld_beta, kld_weights = kld_weights, mids_for_y_class_batch = mids_for_y_class_batch)
        
        
        xent_bin_loss = torch.nn.functional.binary_cross_entropy(y_bin_pred[:, 1], y_bin_batch[:, 1], reduction = self.reduce, weight = weight)
        if self.reduce == 'mean':
            xent_bin_loss = xent_bin_loss.mean()
        
        
        if is_for_selector:
            loss = supcon_loss + kld_loss + xent_lambda * xent_bin_loss
            gap_loss = None

            del Dist
            del KldCustom
            
        else:
            Gap = GapLoss(reduce = self.reduce, sample_reduce = self.sample_reduce)
            gap_loss = Gap(c_encoded, y_bin_batch)
            
            try:
                if torch.isnan(gap_loss):
                    gap_loss = torch.tensor(0.0, requires_grad=True).cuda()
            except Exception as e:
                #print("if torch.isnan(gap_loss) error!!!")
                pass
                
            try:
                if gap_loss == None:
                    gap_loss = torch.tensor(0.0, requires_grad=True).cuda()
            except Exception as e:
                #print("if gap_loss == None error!!!")
                pass
            
            loss = supcon_loss + kld_loss + (xent_lambda * xent_bin_loss) + gap_loss
            
            del Dist
            del KldCustom
            del Gap
        
        torch.cuda.empty_cache()
        
        return loss, supcon_loss, kld_loss, xent_bin_loss, gap_loss
