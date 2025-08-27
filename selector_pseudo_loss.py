import logging
import operator
import time
import torch
import numpy as np
from collections import Counter, defaultdict
from sklearn import neighbors
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import v_measure_score
from scipy.spatial import KDTree

from common import to_categorical
from losses import HiDistanceXentLoss
from selector_def import Selector
from utils import AverageMeter
from train import pseudo_loss

# custom
from losses import HiDistanceKldCustomXentEnsemble6Loss
import gc
import kld_func

class LocalPseudoLossSelector(Selector):
    def __init__(self, encoder):
        self.encoder = encoder
        self.z_train = None
        self.z_test = None
        self.y_train = None
        return
    
    def select_samples(self, args, X_train, y_train, y_train_binary, \
                    X_test, y_test_pred, \
                    total_epochs, \
                    test_offset, \
                    all_test_family, \
                    total_count, \
                    y_test = None):
        
        X_train_tensor = torch.from_numpy(X_train).float().cuda()
        
        z_train = self.encoder.encode(X_train_tensor)
        logging.info(f'Normalizing z_train to unit length...')
        z_train = torch.nn.functional.normalize(z_train)
        z_train = z_train.cpu().detach().numpy()

        X_test_tensor = torch.from_numpy(X_test).float().cuda()
        z_test = self.encoder.encode(X_test_tensor)
        logging.info(f'Normalizing z_test to unit length...')
        z_test = torch.nn.functional.normalize(z_test)
        z_test = z_test.cpu().detach().numpy()
        
        self.z_train = z_train
        self.z_test = z_test
        self.y_train = y_train

        self.sample_indices = []
        self.sample_scores = []
        
        # build the KDTree
        logging.info(f'Building KDTree...')
        tree = KDTree(z_train)
        logging.info(f'Querying KDTree...')
        all_neighbors = tree.query(z_test, k=z_train.shape[0], workers=8) 
        logging.info(f'Finished querying KDTree...')
        
        """
        print(np.array(z_train.shape)) # [30533   128]
        print(np.array(z_test.shape)) # [4991  128]
        print(np.array(all_neighbors).shape) # (2, 4991, 30533)
        exit()
        """
        
        all_distances, all_indices = all_neighbors 

        batch_time = AverageMeter()
        data_time = AverageMeter()
        end = time.time()
        if args.plb == None:
            bsize = args.bsize
        else:
            bsize = args.plb

        sample_num = z_test.shape[0]
        for i in range(sample_num):
            
            if i % 500==0:
                print(f"sample_num i is {i} / {sample_num}")
            
            '''
            if not (i >= 0 and i <= 10000):
                continue
            '''
            
            test_sample = X_test_tensor[i:i+1] 
            batch_indices = all_indices[i][:bsize-1]
            x_train_batch = X_train_tensor[batch_indices] # on GPU
            x_batch = torch.cat((test_sample, x_train_batch), 0)
            y_train_batch = y_train_binary[batch_indices]
            y_batch_np = np.hstack((y_test_pred[i], y_train_batch))
            y_batch = torch.from_numpy(y_batch_np).cuda() # 0 or 1, but in train case, it has the family values
            y_bin_batch = torch.from_numpy(to_categorical(y_batch_np, num_classes=2)).float().cuda()
            
            data_time.update(time.time() - end)

            if args.loss_func == 'hi-dist-xent':
                _, features, y_pred = self.encoder(x_batch) # shapes are [1024,128] [1024,128] [1024,2]
                HiDistanceXent = HiDistanceXentLoss(reduce = args.reduce).cuda() # reduce = none
                
                loss, _, _ = HiDistanceXent(args.xent_lambda, 
                                        y_pred, y_bin_batch,
                                        features, labels=y_batch,
                                        margin = args.margin)
                loss = loss.to('cpu').detach().numpy()
                
            elif args.loss_func == 'hi-dist-kld-custom-xent-ensemble6':
                
                features, x_origin, z_mean, z_log_var, z_hc, y_pred = self.encoder(x_batch)
                
                mids_for_y_class_batch = kld_func.get_mids_for_y_class_batch(self.encoder.mids, self.encoder.y_unique, "bin", y_batch, y_bin_batch)
                
                HiDistanceKldCustomXentEnsemble6 = HiDistanceKldCustomXentEnsemble6Loss(reduce = args.reduce).cuda() # reduce = none
                
                loss, supcon_loss, vae_loss, xent_loss, gap_loss = HiDistanceKldCustomXentEnsemble6(args.xent_lambda, \
                    y_pred, y_bin_batch, \
                    features, z_hc, labels = y_batch, \
                    z_mean = z_mean, z_log_var = z_log_var, \
                    margin = args.margin, margin_btw = args.margin_between_b_and_m, \
                    mids = args.is_enc_kld_custom_mid, \
                    is_for_selector = True, \
                    kld_beta = args.kld_beta,
                    kld_weights = None, mids_for_y_class_batch = mids_for_y_class_batch, kld_dev_scale = self.encoder.kld_dev_scale)
                
                loss = loss.to('cpu').detach().numpy()
                supcon_loss = supcon_loss.to('cpu').detach().numpy()
                xent_loss = xent_loss.to('cpu').detach().numpy()
                
                del HiDistanceKldCustomXentEnsemble6
                torch.cuda.empty_cache()
                
            else:
                # other loss functions pending
                raise Exception(f'local pseudo loss for {args.loss_func} not implemented.')
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            self.sample_scores.append(loss[0])
        # enumerate() makes index to self.sample_socres
        sorted_sample_scores = list(sorted(list(enumerate(self.sample_scores)), key=lambda item: item[1], reverse=True))
        logging.info(f'sorted_sample_scores[:100]: {sorted_sample_scores[:100]}')
        sample_cnt = 0
        for idx, score in sorted_sample_scores:
            logging.info('Sample glb idx: %d, pred: %s, true: %s, ' \
                'score: %.2f\n' % \
                (test_offset+idx, y_test_pred[idx], all_test_family[idx], \
                score))
            self.sample_indices.append(idx)
            sample_cnt += 1
            if sample_cnt == total_count:
                break
        logging.info('Added %s samples...' % (len(self.sample_indices)))
        
        
        del X_train_tensor
        gc.collect()
        torch.cuda.empty_cache()
        
        return self.sample_indices, self.sample_scores # 10, 4991

