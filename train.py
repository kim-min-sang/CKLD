import logging
import numpy as np
import time
import torch
import torch.nn.functional as F
from collections import Counter
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from pytorch_metric_learning.samplers import MPerClassSampler

from common import to_categorical
from losses import TripletMSELoss, TripletMseKldEnsembleXentLoss, TripletMseXentLoss, TripletXentLoss, TripletKldEnsembleXentLoss
from losses import HiDistanceXentLoss, HiDistanceKldCustomXentEnsemble6Loss
from samplers import ProportionalClassSampler
from samplers import HalfSampler
from samplers import TripletSampler
from utils import AverageMeter
from utils import save_model
from utils import adjust_learning_rate
import utils


# custom
import data
import math
import gc
import kld_func



def pseudo_loss(args, encoder, X_train, y_train, y_train_binary, \
                X_test, y_test_pred, test_offset, total_epochs):
    X_tensor = torch.from_numpy(np.vstack((X_train, X_test))).float()
    y = np.concatenate((y_train_binary, y_test_pred), axis=0)

    # y_tensor is used for computing similarity matrix => supcon loss
    y_tensor = torch.from_numpy(y)
    
    device = (torch.device('cuda')
            if torch.cuda.is_available()
            else torch.device('cpu'))
    encoder = encoder.to(device)

    y_bin_cat_tensor = torch.from_numpy(to_categorical(y, num_classes=2)).float()

    split_tensor = torch.zeros(X_tensor.shape[0]).int()
    split_tensor[test_offset:] = 1
    index_tensor = torch.from_numpy(np.arange(y.shape[0]))

    all_data = TensorDataset(X_tensor, y_tensor, y_bin_cat_tensor, index_tensor, split_tensor)

    if args.sampler == 'mperclass':
        bsize = args.sample_per_class * len(np.unique(y))
        data_loader = DataLoader(dataset=all_data, batch_size=bsize, \
            sampler = MPerClassSampler(y, args.sample_per_class))
    elif args.sampler == 'proportional':
        if args.bsize is None:
            bsize = args.min_per_class * len(np.unique(y))
        else:
            bsize = args.bsize
        data_loader = DataLoader(dataset=all_data, batch_size=bsize, \
            sampler = ProportionalClassSampler(y, args.min_per_class, bsize))
    elif args.sampler == 'half':
        if args.plb == None:
            bsize = args.bsize
        else:
            bsize = args.plb
        data_loader = DataLoader(dataset=all_data, batch_size=bsize, \
            sampler = HalfSampler(y, bsize))
    elif args.sampler == 'random':
        bsize = args.bsize
        train_loader = DataLoader(dataset=all_data, batch_size=bsize, shuffle=True)
    else:
        raise Exception('Need to add a sampler here.')
    
    sample_num = y.shape[0]
    sum_loss = np.zeros([sample_num])
    cur_sample_loss = np.zeros([sample_num])
    for epoch in range(1, total_epochs + 1):
        # pseudo_loss goes through one epoch, loss for all samples
        time1 = time.time()
        sample_loss = pseudo_loss_one_epoch(args, encoder, data_loader, sample_num, epoch)
        time2 = time.time()
        if args.sample_reduce == 'mean':
            sum_loss += sample_loss
            # average the loss per sample, including both train and test
            cur_sample_loss = sum_loss / epoch
            # only print test sample cur_sample_loss
            logging.info('epoch {}, b {}, total time {:.2f}, (sorted avg loss)[:50] {}'.format(epoch, bsize, time2 - time1, sorted(cur_sample_loss[test_offset:], reverse=True)[:50]))
        else:
            # args.sample_reduce == 'max':
            cur_sample_loss = np.maximum(cur_sample_loss, sample_loss)
            # only print test sample cur_sample_loss
            logging.info('epoch {}, b {}, total time {:.2f}, (sorted max loss)[:50] {}'.format(epoch, bsize, time2 - time1, sorted(cur_sample_loss[test_offset:], reverse=True)[:50]))
    return cur_sample_loss

def pseudo_loss_one_epoch(args, encoder, data_loader, sample_num, epoch):
    """
    measure one epoch of pseudo loss for train + test samples.
    default data points number in an epoch: length_before_new_iter=100000
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    
    select_count = torch.zeros(sample_num, dtype=torch.float64)
    total_loss = torch.zeros(sample_num, dtype=torch.float64)
    # if args.sample_reduce == 'mean'
    sample_avg_loss = torch.zeros(sample_num, dtype=torch.float64)
    # if args.sample_reduce == 'max'
    sample_max_loss = torch.zeros(sample_num, dtype=torch.float64)

    pos_max_sim = torch.zeros(sample_num, dtype=torch.float64)
    neg_max_sim = torch.zeros(sample_num, dtype=torch.float64)

    idx = 0
    # average the loss for each index in batch_indices
    device = (torch.device('cuda')
                if torch.cuda.is_available()
                else torch.device('cpu'))
    
    for idx, (x_batch, y_batch, y_bin_batch, batch_indices, split_tensor) in enumerate(data_loader):
        data_time.update(time.time() - end)

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        y_bin_batch = y_bin_batch.to(device)
        
        if args.loss_func == 'triplet-mse':
            features, decoded = encoder(x_batch)

            TripletMSE = TripletMSELoss(reduce = args.reduce).to(device)
            loss, _, _ = TripletMSE(args.cae_lambda, \
                                x_batch, decoded, features, labels=y_batch, \
                                margin = args.margin, \
                                split = split_tensor)
            loss = loss.to('cpu').detach()
        elif args.loss_func == 'hi-dist-xent':
            _, features, y_pred = encoder(x_batch)
            HiDistanceXent = HiDistanceXentLoss(reduce = args.reduce).to(device)
            loss, _, _ = HiDistanceXent(args.xent_lambda, 
                                    y_pred, y_bin_batch,
                                    features, labels=y_batch,
                                    split = split_tensor)
            loss = loss.to('cpu').detach()
        else:
            raise Exception(f'pseudo loss for {args.loss_func} not implemented.')
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        select_count[batch_indices] = torch.add(select_count[batch_indices], 1)
        non_select_count = sample_num - torch.count_nonzero(select_count).item()
        # # update the loss values for batch_indices
        if args.sample_reduce == 'mean':
            total_loss[batch_indices] = torch.add(total_loss[batch_indices], loss)

        if args.sample_reduce == 'max':
            sample_max_loss[batch_indices] = torch.maximum(sample_max_loss[batch_indices], loss)
            
    # sample average loss
    if args.sample_reduce == 'mean':
        sample_avg_loss = torch.div(total_loss, select_count)
        return sample_avg_loss.numpy()
    else:
        # args.sample_reduce == 'max':
        return sample_max_loss.numpy()



def train_encoder(args, encoder, X_train, y_train, y_train_binary,
                optimizer, total_epochs, model_path,
                upsample = None, adjust = False, warm = False,
                save_best_loss = False,
                save_snapshot = False,
                pl_pretrain = False,
                weight = None, is_first_train = None):
    
    gc.collect()
    torch.cuda.empty_cache()
    
    device = (torch.device('cuda')
                if torch.cuda.is_available()
                else torch.device('cpu'))
    encoder = encoder.to(device)
    
    # construct the dataset loader
    # y_train is multi-class, y_train_binary is binary class
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).type(torch.int64)
    y_train_binary_cat_tensor = torch.from_numpy(to_categorical(y_train_binary)).float()
    
    if weight is None:
        weight_tensor = torch.ones(X_train.shape[0])
    else:
        weight_tensor = torch.from_numpy(weight).float()
    
    ckld_encoders = ['enc-kld-custom-mlp-ensemble6', 'cae-kld-ensemble-mlp', 'triplet-kld-ensemble-mlp']
    
    
    if args.encoder in ckld_encoders:
        train_data = data.CustomDataset(X_train_tensor, y_train_tensor, y_train_binary_cat_tensor, weight_tensor, mids_for_y_class=None)
    else:
        train_data = TensorDataset(X_train_tensor, y_train_tensor, y_train_binary_cat_tensor, weight_tensor)
    
    if args.sampler == 'mperclass':
        bsize = args.sample_per_class * len(np.unique(y_train))
        train_loader = DataLoader(dataset=train_data, batch_size=bsize, \
            sampler = MPerClassSampler(y_train, args.sample_per_class))
    elif args.sampler == 'proportional':
        if args.bsize is None:
            bsize = args.min_per_class * len(np.unique(y_train))
        else:
            bsize = args.bsize
        train_loader = DataLoader(dataset=train_data, batch_size=bsize, \
            sampler = ProportionalClassSampler(y_train, args.min_per_class, bsize))
    elif args.sampler == 'half':
        bsize = args.bsize
        train_loader = DataLoader(dataset=train_data, batch_size=bsize, \
            sampler = HalfSampler(y_train, bsize, upsample = upsample))
    elif args.sampler == 'triplet':
        bsize = args.bsize
        train_loader = DataLoader(dataset=train_data, batch_size=bsize, \
            sampler = TripletSampler(y_train, bsize))
    elif args.sampler == 'random':
        bsize = args.bsize
        train_loader = DataLoader(dataset=train_data, batch_size=bsize, shuffle=True)
    else:
        raise Exception(f'Sampler {args.sampler} not implemented yet.')
    best_loss = np.inf
    
    for epoch in range(1, total_epochs + 1):
        if adjust == True:
            new_lr = adjust_learning_rate(args, optimizer, epoch, warm = warm)
        else:
            for param_group in optimizer.param_groups:
                new_lr = param_group['lr']
                break
            
        # train one epoch
        time1 = time.time()
        if pl_pretrain == False:
            if args.encoder in ckld_encoders:
                
                if args.is_enc_kld_custom_mid:
                    mids = True
                    
                else:
                    mids = False
                    
                loss = train_encoder_one_epoch(args, encoder, train_loader, optimizer, epoch, X_train_tensor = X_train_tensor, y_train_tensor = y_train_tensor, y_train_binary_tensor = y_train_binary_cat_tensor, mids = mids, is_first_train = is_first_train, train_data = train_data)
            else:
                loss = train_encoder_one_epoch(args, encoder, train_loader, optimizer, epoch)
        else:
            loss = pl_train_encoder_one_epoch(args, encoder, train_loader, optimizer, epoch)
        time2 = time.time()
        logging.info('epoch {}, b {}, lr {}, loss {}, total time {:.2f}'.format(epoch, bsize, new_lr, loss, time2 - time1))

        if epoch >= total_epochs - 10:
            if save_best_loss == True:
                if loss < best_loss:
                    best_loss = loss
                    logging.info(f'Saving the best loss {loss} model from epoch {epoch}...')
                    save_model(encoder, optimizer, args, args.epochs, model_path)
    
        if save_snapshot == True and epoch % 50 == 0:
            save_path = model_path.replace("e%s" % total_epochs, "e%d" % epoch)
            logging.info(f'Saving the model from epoch {epoch} loss {loss} at {save_path}...')
            save_model(encoder, optimizer, args, args.epochs, save_path)
    return


def train_encoder_one_epoch(args, encoder, train_loader, optimizer, epoch, X_train_tensor = None, y_train_tensor = None, y_train_binary_tensor = None, mids = None, is_first_train = None, train_data = None):
    """ Train one epoch for the model """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    supcon_losses = AverageMeter()
    mse_losses = AverageMeter()
    xent_losses = AverageMeter()
    gap_losses = AverageMeter()
    kld_losses = AverageMeter()
    vae_losses = AverageMeter()
    xent_multi_losses = AverageMeter()
    xent_bin_losses = AverageMeter()
    end = time.time()

    device = (torch.device('cuda')
                if torch.cuda.is_available()
                else torch.device('cpu'))
    encoder = encoder.to(device)

    ckld_loss_funcs = ['hi-dist-kld-custom-xent-ensemble6', 'triplet-kld-ensemble-xent', 'triplet-mse-kld-ensemble-xent']
    
    if args.loss_func in ckld_loss_funcs:
        with torch.no_grad():
            X_train_tensor = X_train_tensor.to(device)
            y_train_tensor = y_train_tensor.to(device)
            y_train_binary_tensor = y_train_binary_tensor.to(device)
            z_hc = encoder.encode_c(X_train_tensor)
            mids_for_y_class, mids_y, y_unique = kld_func.get_mids_for_y_class(z_hc, args.mid_type, y_train_tensor, y_train_binary_tensor)
            encoder.mids = mids_y
            encoder.y_unique = y_unique
            train_data.update_mids_for_y_class(mids_for_y_class)
            
    idx = 0
    for idx, batch in enumerate(train_loader):
        
        if args.loss_func in ckld_loss_funcs:
            x_batch, y_batch, y_bin_batch, weight_batch, mids_for_y_class_batch = batch
        else:
            x_batch, y_batch, y_bin_batch, weight_batch = batch
            mids_for_y_class_batch = None
        
        data_time.update(time.time() - end)
        
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        y_bin_batch = y_bin_batch.to(device)
        weight_batch = weight_batch.to(device)
        
        bsz = y_batch.shape[0]
        
        if args.loss_func == 'triplet-mse':
            features, decoded = encoder(x_batch)

            TripletMSE = TripletMSELoss().cuda()
            loss, supcon_loss, mse_loss = TripletMSE(args.cae_lambda, \
                                x_batch, decoded, features, labels=y_batch, \
                                margin = args.margin, \
                                weight = weight_batch)
            
            # update metric
            losses.update(loss.item(), bsz)
            supcon_losses.update(supcon_loss.item(), bsz)
            mse_losses.update(mse_loss.item(), bsz)

            # SGD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print info, print every display_interval batches.
            if (idx + 1) % args.display_interval == 0:
                logging.info('Train: [{0}][{1}/{2}]\t'
                    'BT {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'DT {data_time.val:.3f} ({data_time.avg:.3f})  '
                    'loss {loss.val:.4f} ({loss.avg:.4f})  '
                    'triplet {supcon.val:.4f} ({supcon.avg:.4f})  '
                    'mse {mse.val:.4f} ({mse.avg:.4f})  '.format(
                    epoch, idx + 1, len(train_loader), batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    supcon=supcon_losses,
                    mse=mse_losses))
                
                
        elif args.loss_func == 'triplet-mse-kld-ensemble-xent':
            features, x_origin, x_recon, z_mean, z_log_var, c_encoded, y_pred = encoder(x_batch)
            # features: hidden vector of shape [bsz, n_feature_dim].
            # print(f"features shape {features.shape}") # why is this size 128? not 2? becuase enc-mlp returns (enc, enc, mlp)

            # 앙상블에서 가능한 편차 전해주기
            two_x = kld_func.find_exponent_of_two(c_encoded.shape[1])
            kld_dev_scale = math.sqrt(1/2) ** (two_x)
            
            # (2) calculate loss
            # Our own version of the supervised contrastive learning loss
            TripletMseKldEnsembleXent = TripletMseKldEnsembleXentLoss().cuda() # reduce = mean
            
            loss, supcon_loss, mse_loss, kld_loss, xent_loss, gap_loss = TripletMseKldEnsembleXent(args.cae_lambda, args.xent_lambda, args.mse_lambda, \
                                            y_pred, y_bin_batch, \
                                            features, c_encoded, labels = y_batch, \
                                            x = x_origin, x_recon = x_recon, \
                                            z_mean = z_mean, z_log_var = z_log_var, \
                                            margin = args.margin, \
                                            margin_btw = args.margin_between_b_and_m, \
                                            weight = weight_batch, \
                                            epoch = epoch, mids = mids, kld_dev_scale = kld_dev_scale, mids_for_y_class_batch = mids_for_y_class_batch)
            
            # update metric
            losses.update(loss.item(), bsz)
            supcon_losses.update(supcon_loss.item(), bsz)
            mse_losses.update(mse_loss.item(), bsz)
            kld_losses.update(kld_loss.item(), bsz)
            xent_losses.update(xent_loss.item(), bsz)
            gap_losses.update(gap_loss.item(), bsz)

            # (3) backward pass
            # SGD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print info, print every display_interval batches.
            if (idx + 1) == len(train_loader):
                logging.info('Train: [{0}][{1}/{2}]\t'
                    'BT {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'DT {data_time.val:.3f} ({data_time.avg:.3f})  '
                    'loss {loss.val:.5f} ({loss.avg:.5f})  '
                    'supcon {supcon.val:.5f} ({supcon.avg:.5f})  '
                    'mse {mse.val:.5f} ({mse.avg:.5f})  '
                    'kld {kld.val:.5f} ({kld.avg:.5f})  '
                    'xent {xent.val:.5f} ({xent.avg:.5f})   '
                    'gap {gap.val:.5f} ({gap.avg:.5f})  '.format(
                    epoch, idx + 1, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, supcon=supcon_losses, mse=mse_losses, kld=kld_losses,
                    xent=xent_losses,
                    gap=gap_losses))

                
                
                
              
              
        elif args.loss_func == 'triplet-mse-xent':
            x_origin, x_recon, c_encoded, y_pred = encoder(x_batch)
            TripletMseXent = TripletMseXentLoss().cuda() # reduce = mean
            
            loss, supcon_loss, mse_loss, xent_loss = TripletMseXent(args.cae_lambda, args.xent_lambda, args.mse_lambda,\
                                            y_pred, y_bin_batch, \
                                            c_encoded, labels = y_batch, \
                                            x = x_origin, x_recon = x_recon, \
                                            margin = args.margin, \
                                            margin_btw = args.margin_between_b_and_m, \
                                            weight = weight_batch, \
                                            epoch = epoch)
            
            # update metric
            losses.update(loss.item(), bsz)
            supcon_losses.update(supcon_loss.item(), bsz)
            mse_losses.update(mse_loss.item(), bsz)
            xent_losses.update(xent_loss.item(), bsz)

            # (3) backward pass
            # SGD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print info, print every display_interval batches.
            if (idx + 1) == len(train_loader):
                logging.info('Train: [{0}][{1}/{2}]\t'
                    'BT {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'DT {data_time.val:.3f} ({data_time.avg:.3f})  '
                    'loss {loss.val:.5f} ({loss.avg:.5f})  '
                    'supcon {supcon.val:.5f} ({supcon.avg:.5f})  '
                    'mse {mse.val:.5f} ({mse.avg:.5f})  '
                    'xent {xent.val:.5f} ({xent.avg:.5f})   '.format(
                    epoch, idx + 1, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, supcon=supcon_losses, mse=mse_losses,
                    xent=xent_losses))
                
                
                
                
                
                
                
        elif args.loss_func == 'triplet-xent':
            x_origin, c_encoded, y_pred = encoder(x_batch)
            TripletXent = TripletXentLoss().cuda() # reduce = mean
            
            loss, supcon_loss, xent_loss = TripletXent(args.triplet_lambda, args.xent_lambda, \
                                            y_pred, y_bin_batch, \
                                            c_encoded, labels = y_batch, \
                                            x = x_origin, \
                                            margin = args.margin, \
                                            margin_btw = args.margin_between_b_and_m, \
                                            weight = weight_batch, \
                                            epoch = epoch)
            
            # update metric
            losses.update(loss.item(), bsz)
            supcon_losses.update(supcon_loss.item(), bsz)
            xent_losses.update(xent_loss.item(), bsz)

            # (3) backward pass
            # SGD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print info, print every display_interval batches.
            if (idx + 1) == len(train_loader):
                logging.info('Train: [{0}][{1}/{2}]\t'
                    'BT {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'DT {data_time.val:.3f} ({data_time.avg:.3f})  '
                    'loss {loss.val:.5f} ({loss.avg:.5f})  '
                    'supcon {supcon.val:.5f} ({supcon.avg:.5f})  '
                    'xent {xent.val:.5f} ({xent.avg:.5f})   '.format(
                    epoch, idx + 1, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, supcon=supcon_losses,
                    xent=xent_losses))
                
                
        
        
        
        elif args.loss_func == 'triplet-kld-ensemble-xent':
            features, x_origin, z_mean, z_log_var, c_encoded, y_pred = encoder(x_batch)

            two_x = kld_func.find_exponent_of_two(c_encoded.shape[1])
            kld_dev_scale = math.sqrt(1/2) ** (two_x)
            
            TripletKldEnsembleXent = TripletKldEnsembleXentLoss().cuda() # reduce = mean
            
            loss, supcon_loss, kld_loss, xent_loss, gap_loss = TripletKldEnsembleXent(args.triplet_lambda, args.xent_lambda, \
                                            y_pred, y_bin_batch, \
                                            features, c_encoded, labels = y_batch, \
                                            x = x_origin, \
                                            z_mean = z_mean, z_log_var = z_log_var, \
                                            margin = args.margin, \
                                            margin_btw = args.margin_between_b_and_m, \
                                            weight = weight_batch, \
                                            epoch = epoch, mids = mids, kld_dev_scale = kld_dev_scale, mids_for_y_class_batch = mids_for_y_class_batch)
            
            # update metric
            losses.update(loss.item(), bsz)
            supcon_losses.update(supcon_loss.item(), bsz)
            kld_losses.update(kld_loss.item(), bsz)
            xent_losses.update(xent_loss.item(), bsz)
            gap_losses.update(gap_loss.item(), bsz)

            # (3) backward pass
            # SGD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print info, print every display_interval batches.
            if (idx + 1) == len(train_loader):
                logging.info('Train: [{0}][{1}/{2}]\t'
                    'BT {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'DT {data_time.val:.3f} ({data_time.avg:.3f})  '
                    'loss {loss.val:.5f} ({loss.avg:.5f})  '
                    'supcon {supcon.val:.5f} ({supcon.avg:.5f})  '
                    'kld {kld.val:.5f} ({kld.avg:.5f})  '
                    'xent {xent.val:.5f} ({xent.avg:.5f})   '
                    'gap {gap.val:.5f} ({gap.avg:.5f})  '.format(
                    epoch, idx + 1, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, supcon=supcon_losses, kld=kld_losses,
                    xent=xent_losses,
                    gap=gap_losses))
              
              
                
                
        
        elif args.loss_func == 'hi-dist-xent':
            _, cur_f, y_pred = encoder(x_batch)
        
            features = cur_f
            HiDistanceXent = HiDistanceXentLoss().cuda() # reduce = mean
            loss, supcon_loss, xent_loss = HiDistanceXent(args.xent_lambda, \
                                            y_pred, y_bin_batch, \
                                            features, labels = y_batch, \
                                            margin = args.margin, \
                                            weight = weight_batch)
            
            # update metric
            losses.update(loss.item(), bsz)
            supcon_losses.update(supcon_loss.item(), bsz)
            xent_losses.update(xent_loss.item(), bsz)

            # SGD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print info, print every display_interval batches.
            if (idx + 1) % args.display_interval == 0:
                logging.info('Train: [{0}][{1}/{2}]\t'
                    'BT {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'DT {data_time.val:.3f} ({data_time.avg:.3f})  '
                    'loss {loss.val:.5f} ({loss.avg:.5f})  '
                    'hidist {supcon.val:.5f} ({supcon.avg:.5f})  '
                    'xent {xent.val:.5f} ({xent.avg:.5f})'.format(
                    epoch, idx + 1, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, supcon=supcon_losses,
                    xent=xent_losses))
            
        elif args.loss_func == 'hi-dist-kld-custom-xent-ensemble6':
            
            features, x_origin, z_mean, z_log_var, c_encoded, y_pred = encoder(x_batch)
                
            HiDistanceKldCustomXentEnsemble6 = HiDistanceKldCustomXentEnsemble6Loss().cuda() # reduce = mean
            loss, supcon_loss, kld_loss, xent_loss, gap_loss = HiDistanceKldCustomXentEnsemble6(args.xent_lambda, \
                y_pred, y_bin_batch, \
                features, c_encoded, labels = y_batch, \
                x = x_origin, \
                z_mean = z_mean, z_log_var = z_log_var, \
                margin = args.margin, \
                margin_btw = args.margin_between_b_and_m, \
                epoch = epoch, mids = mids, kld_beta = args.kld_beta, \
                kld_weights = None, mids_for_y_class_batch = mids_for_y_class_batch,
                kld_dev_scale = encoder.kld_dev_scale)
            
            # update metric
            losses.update(loss.item(), bsz)
            supcon_losses.update(supcon_loss.item(), bsz)
            try:
                kld_losses.update(kld_loss.item(), bsz)
            except:
                print("kld_loss is not exist")
            
            try:
                gap_losses.update(gap_loss.item(), bsz)
            except:
                #print("gap_loss is not exist")
                pass
            
            xent_losses.update(xent_loss.item(), bsz)

            # (3) backward pass
            # SGD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print info, print every display_interval batches.
            if (idx + 1) == len(train_loader):
                logging.info('Train: [{0}][{1}/{2}]\t'
                    'BT {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'DT {data_time.val:.3f} ({data_time.avg:.3f})  '
                    'loss {loss.val:.5f} ({loss.avg:.5f})  '
                    'hidist {supcon.val:.5f} ({supcon.avg:.5f})  '
                    'kld {kld.val:.5f} ({kld.avg:.5f})  '
                    'xent {xent.val:.5f} ({xent.avg:.5f})  '
                    'gap {gap.val:.5f} ({gap.avg:.5f})'.format(
                    epoch, idx + 1, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, supcon=supcon_losses, kld=kld_losses,
                    xent=xent_losses, gap=gap_losses))
                
            del HiDistanceKldCustomXentEnsemble6
            del loss
            del supcon_loss
            del kld_loss
            del xent_loss
            try:
                del gap_loss
            except:
                print("gap_loss can't del")
            torch.cuda.empty_cache()
            
        else:
            raise Exception(f'The loss function {args.loss_func} for model ' \
                f'{args.encoder} is not supported yet.')
            
            
    
        
    return losses.avg
