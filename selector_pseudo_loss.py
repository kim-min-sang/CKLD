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
        # z_train is latent space, but this method doesn't train reconstruction error
        # in this case, encode means just reduced space
        # I wonder what is the difference between this method and cade because I showed that there is a reconstruction error in cade #?
        
        z_train = self.encoder.encode(X_train_tensor)
        logging.info(f'Normalizing z_train to unit length...')
        # actually, I don't know normalizing is necessary after encoding
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
        # query all z_test up to a margin
        # z_test에 대해 가까운 z_train을 네이버로 설정한 것임. 모든 z_train에 대해서.
        all_neighbors = tree.query(z_test, k=z_train.shape[0], workers=8) # (2, 4991, 30533)
        logging.info(f'Finished querying KDTree...')
        
        """
        print(np.array(z_train.shape)) # [30533   128]
        print(np.array(z_test.shape)) # [4991  128]
        print(np.array(all_neighbors).shape) # (2, 4991, 30533)
        exit()
        """
        
        # I think all_indeces is index
        # it is order that is nearest to test samples
        all_distances, all_indices = all_neighbors 

        # each batch is to get one loss for one test sample (one test sample means a month?)
        batch_time = AverageMeter()
        data_time = AverageMeter()
        end = time.time()
        if args.plb == None:
            bsize = args.bsize
        else:
            bsize = args.plb

        # 새로이 들어오는 샘플에 대해서만 drifting 탐지를 해주는구나.
        sample_num = z_test.shape[0]
        for i in range(sample_num): # repeat 4991
            
            if i % 500==0:
                print(f"sample_num i is {i} / {sample_num}")
            
            '''
            if not (i >= 0 and i <= 10000):
                continue
            '''
            
            test_sample = X_test_tensor[i:i+1] # on GPU #[1,1159]
            # bsize-1 nearest neighbors of the test sample i
            # the reason why setting the bsize -1 is that it will add the test_sample to x_train_batch
            # it's enough to set bsize-1 because it's not necessary to use all distance list
            batch_indices = all_indices[i][:bsize-1]
            # x_batch
            x_train_batch = X_train_tensor[batch_indices] # on GPU
            # test sample 1개 + 가장 neighbor인거 1023개
            # 이렇게 하는 이유는 loss를 도출할때 비교 대상인 정상적인 neighbors가 있어야기 때문
            x_batch = torch.cat((test_sample, x_train_batch), 0)
            """
            print(f"test_sample {test_sample.shape}") # torch.Size([1, 1159])
            print(f"x_train_batch {x_train_batch.shape}") # torch.Size([1023, 1159])
            print(f"x_batch {x_batch.shape}") # torch.Size([1024, 1159])
            exit()
            """
            # y_batch
            y_train_batch = y_train_binary[batch_indices]
            y_batch_np = np.hstack((y_test_pred[i], y_train_batch))
            y_batch = torch.from_numpy(y_batch_np).cuda() # 0 or 1, but in train case, it has the family values
            # y_bin_batch
            y_bin_batch = torch.from_numpy(to_categorical(y_batch_np, num_classes=2)).float().cuda()
            # we don't need split_tensor. all samples are training samples
            # split_tensor = torch.zeros(x_batch.shape[0]).int().cuda()
            # split_tensor[test_offset:] = 1
            
            data_time.update(time.time() - end)

            # in the loss function, y_bin_batch is the categorical version
            # call the loss function once for every test sample
            if args.loss_func == 'hi-dist-xent':
                # what is mean about features? I think that means X passed the encoder
                _, features, y_pred = self.encoder(x_batch) # shapes are [1024,128] [1024,128] [1024,2]
                HiDistanceXent = HiDistanceXentLoss(reduce = args.reduce).cuda() # reduce = none
                
                # below "labels=y_batch" is just binary vlaues, not family in selector case
                # because "same_mal_fam_mask = multi_mask - same_ben_mask" equals 0 when "labels=y_batch" is just binary values
                loss, _, _ = HiDistanceXent(args.xent_lambda, 
                                        y_pred, y_bin_batch,
                                        features, labels=y_batch,
                                        margin = args.margin)
                loss = loss.to('cpu').detach().numpy()
                #print(f"features shape {features.shape}")
                #print(f"loss shape in selector {(loss).shape}") # 1024
                #quit()
                
            elif args.loss_func == 'hi-dist-kld-custom-xent-ensemble2':
                
                # what is mean about features? I think that means X passed the encoder
                features, x_origin, z_mean, z_log_var, pre_encoder_z, y_pred = self.encoder(x_batch)
                #print(f"when hi-dist-xent, y_pred is {y_pred}")
                HiDistanceKldCustomXentEnsemble2 = HiDistanceKldCustomXentEnsemble2Loss(reduce = args.reduce).cuda() # reduce = none
                
                # below "labels=y_batch" is just binary vlaues, not family in selector case
                # because "same_mal_fam_mask = multi_mask - same_ben_mask" equals 0 when "labels=y_batch" is just binary values
                loss, supcon_loss, vae_loss, xent_loss, gap_loss = HiDistanceKldCustomXentEnsemble2(args.xent_lambda, \
                                                y_pred, y_bin_batch, \
                                                features, pre_encoder_z, labels = y_batch, \
                                                z_mean = z_mean, z_log_var = z_log_var, \
                                                margin = args.margin, margin_btw = args.margin_between_b_and_m, \
                                                mids = args.is_enc_kld_custom_mid, \
                                                is_for_selector = True, kld_beta = args.kld_beta, mid_type = args.mid_type)
                
                
                loss = loss.to('cpu').detach().numpy()
                supcon_loss = supcon_loss.to('cpu').detach().numpy()
                #gap_loss = gap_loss.to('cpu').detach().numpy()
                #vae_loss = vae_loss.to('cpu').detach().numpy()
                xent_loss = xent_loss.to('cpu').detach().numpy()
                #gap_loss = gap_loss.to('cpu').detach().numpy()
                
                del HiDistanceKldCustomXentEnsemble2
                torch.cuda.empty_cache()
                
            elif args.loss_func == 'hi-dist-vae-custom-xent-ensemble3':
                
                # what is mean about features? I think that means X passed the encoder
                features, x_origin, x_recon, z_mean, z_log_var, pre_encoder_z, y_pred = self.encoder(x_batch)
                #print(f"when hi-dist-xent, y_pred is {y_pred}")
                HiDistanceVaeCustomXentEnsemble3 = HiDistanceVaeCustomXentEnsemble3Loss(reduce = args.reduce).cuda() # reduce = none
                
                # below "labels=y_batch" is just binary vlaues, not family in selector case
                # because "same_mal_fam_mask = multi_mask - same_ben_mask" equals 0 when "labels=y_batch" is just binary values
                loss, supcon_loss, vae_loss, xent_loss, gap_loss = HiDistanceVaeCustomXentEnsemble3(args.xent_lambda, \
                                                y_pred, y_bin_batch, \
                                                features, pre_encoder_z, labels = y_batch, \
                                                x = x_origin, x_recon = x_recon, \
                                                z_mean = z_mean, z_log_var = z_log_var, \
                                                margin = args.margin, \
                                                is_for_selector = True)
                
                
                loss = loss.to('cpu').detach().numpy()
                supcon_loss = supcon_loss.to('cpu').detach().numpy()
                #gap_loss = gap_loss.to('cpu').detach().numpy()
                #vae_loss = vae_loss.to('cpu').detach().numpy()
                xent_loss = xent_loss.to('cpu').detach().numpy()
                #gap_loss = gap_loss.to('cpu').detach().numpy()
                
                del HiDistanceVaeCustomXentEnsemble3
                torch.cuda.empty_cache()
                
            elif args.loss_func == 'hi-dist-kld-custom-xent-ensemble4':
                
                # what is mean about features? I think that means X passed the encoder
                features, x_origin, z_mean, z_log_var, pre_encoder_z, y_pred = self.encoder(x_batch)
                #print(f"when hi-dist-xent, y_pred is {y_pred}")
                HiDistanceKldCustomXentEnsemble4 = HiDistanceKldCustomXentEnsemble4Loss(reduce = args.reduce).cuda() # reduce = none
                
                # below "labels=y_batch" is just binary vlaues, not family in selector case
                # because "same_mal_fam_mask = multi_mask - same_ben_mask" equals 0 when "labels=y_batch" is just binary values
                loss, supcon_loss, vae_loss, xent_loss, gap_loss = HiDistanceKldCustomXentEnsemble4(args.xent_lambda, \
                                                y_pred, y_bin_batch, \
                                                features, pre_encoder_z, labels = y_batch, \
                                                z_mean = z_mean, z_log_var = z_log_var, \
                                                margin = args.margin, \
                                                is_for_selector = True)
                
                
                loss = loss.to('cpu').detach().numpy()
                supcon_loss = supcon_loss.to('cpu').detach().numpy()
                #gap_loss = gap_loss.to('cpu').detach().numpy()
                #vae_loss = vae_loss.to('cpu').detach().numpy()
                xent_loss = xent_loss.to('cpu').detach().numpy()
                #gap_loss = gap_loss.to('cpu').detach().numpy()
                
                del HiDistanceKldCustomXentEnsemble4
                torch.cuda.empty_cache()
                
            elif args.loss_func == 'hi-dist-kld-custom-xent-ensemble5':
                
                # what is mean about features? I think that means X passed the encoder
                features, x_origin, z_mean, z_log_var, z_kld_dev_scale, z_hc, y_pred = self.encoder(x_batch)
                
                mids_for_y_class_batch = kld_func.get_mids_for_y_class_batch(self.encoder.mids, self.encoder.y_unique, "bin", y_batch, y_bin_batch)
                
                #print(f"when hi-dist-xent, y_pred is {y_pred}")
                HiDistanceKldCustomXentEnsemble5 = HiDistanceKldCustomXentEnsemble5Loss(reduce = args.reduce).cuda() # reduce = none
                
                # below "labels=y_batch" is just binary vlaues, not family in selector case
                # because "same_mal_fam_mask = multi_mask - same_ben_mask" equals 0 when "labels=y_batch" is just binary values
                loss, supcon_loss, vae_loss, xent_loss, gap_loss = HiDistanceKldCustomXentEnsemble5(args.xent_lambda, \
                    y_pred, y_bin_batch, \
                    features, z_hc, labels = y_batch, \
                    z_mean = z_mean, z_log_var = z_log_var, z_kld_dev_scale = z_kld_dev_scale,\
                    margin = args.margin, margin_btw = args.margin_between_b_and_m, \
                    mids = args.is_enc_kld_custom_mid, \
                    is_for_selector = True, \
                    kld_dev_scale = self.encoder.kld_dev_scale, kld_beta = args.kld_beta,
                    kld_weights = self.encoder.kld_weights, mids_for_y_class_batch = mids_for_y_class_batch)
                
                loss = loss.to('cpu').detach().numpy()
                supcon_loss = supcon_loss.to('cpu').detach().numpy()
                #gap_loss = gap_loss.to('cpu').detach().numpy()
                #vae_loss = vae_loss.to('cpu').detach().numpy()
                xent_loss = xent_loss.to('cpu').detach().numpy()
                #gap_loss = gap_loss.to('cpu').detach().numpy()
                
                del HiDistanceKldCustomXentEnsemble5
                torch.cuda.empty_cache()
                
                
            elif args.loss_func == 'hi-dist-kld-custom-xent-ensemble6':
                
                # what is mean about features? I think that means X passed the encoder
                features, x_origin, z_mean, z_log_var, z_hc, y_pred = self.encoder(x_batch)
                
                mids_for_y_class_batch = kld_func.get_mids_for_y_class_batch(self.encoder.mids, self.encoder.y_unique, "bin", y_batch, y_bin_batch)
                
                #print(f"when hi-dist-xent, y_pred is {y_pred}")
                HiDistanceKldCustomXentEnsemble6 = HiDistanceKldCustomXentEnsemble6Loss(reduce = args.reduce).cuda() # reduce = none
                
                # below "labels=y_batch" is just binary vlaues, not family in selector case
                # because "same_mal_fam_mask = multi_mask - same_ben_mask" equals 0 when "labels=y_batch" is just binary values
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
                #gap_loss = gap_loss.to('cpu').detach().numpy()
                #vae_loss = vae_loss.to('cpu').detach().numpy()
                xent_loss = xent_loss.to('cpu').detach().numpy()
                #gap_loss = gap_loss.to('cpu').detach().numpy()
                
                del HiDistanceKldCustomXentEnsemble6
                torch.cuda.empty_cache()
                
            elif args.loss_func == 'hi-dist-kld-custom-xent-ensemble7':
                
                # what is mean about features? I think that means X passed the encoder
                features, x_origin, z_mean, z_log_var, z_hc, y_pred = self.encoder(x_batch)
                
                mids_for_y_class_batch = kld_func.get_mids_for_y_class_batch(self.encoder.mids, self.encoder.y_unique, "bin", y_batch, y_bin_batch)
                
                #print(f"when hi-dist-xent, y_pred is {y_pred}")
                HiDistanceKldCustomXentEnsemble7 = HiDistanceKldCustomXentEnsemble7Loss(reduce = args.reduce).cuda() # reduce = none
                
                # below "labels=y_batch" is just binary vlaues, not family in selector case
                # because "same_mal_fam_mask = multi_mask - same_ben_mask" equals 0 when "labels=y_batch" is just binary values
                loss, supcon_loss, vae_loss, xent_loss, gap_loss = HiDistanceKldCustomXentEnsemble7(args.xent_lambda, \
                    y_pred, y_bin_batch, \
                    features, z_hc, labels = y_batch, \
                    z_mean = z_mean, z_log_var = z_log_var, \
                    margin = args.margin, margin_btw = args.margin_between_b_and_m, \
                    mids = args.is_enc_kld_custom_mid, \
                    is_for_selector = True, \
                    kld_beta = args.kld_beta,
                    kld_weights = self.encoder.kld_weights, mids_for_y_class_batch = mids_for_y_class_batch)
                
                loss = loss.to('cpu').detach().numpy()
                supcon_loss = supcon_loss.to('cpu').detach().numpy()
                #gap_loss = gap_loss.to('cpu').detach().numpy()
                #vae_loss = vae_loss.to('cpu').detach().numpy()
                xent_loss = xent_loss.to('cpu').detach().numpy()
                #gap_loss = gap_loss.to('cpu').detach().numpy()
                
                del HiDistanceKldCustomXentEnsemble7
                torch.cuda.empty_cache()
                
                
                
            elif args.loss_func == 'hi-dist-kld-custom-xent-ensemble8':
                
                # what is mean about features? I think that means X passed the encoder
                features, x_origin, z_mean, z_log_var, z_hc, y_pred = self.encoder(x_batch)
                
                mids_for_y_class_batch = kld_func.get_mids_for_y_class_batch(self.encoder.mids, self.encoder.y_unique, "bin", y_batch, y_bin_batch)
                
                #print(f"when hi-dist-xent, y_pred is {y_pred}")
                HiDistanceKldCustomXentEnsemble8 = HiDistanceKldCustomXentEnsemble8Loss(reduce = args.reduce).cuda() # reduce = none
                
                # below "labels=y_batch" is just binary vlaues, not family in selector case
                # because "same_mal_fam_mask = multi_mask - same_ben_mask" equals 0 when "labels=y_batch" is just binary values
                loss, supcon_loss, vae_loss, xent_loss, gap_loss = HiDistanceKldCustomXentEnsemble8(args.xent_lambda, \
                    y_pred, y_bin_batch, \
                    features, z_hc, labels = y_batch, \
                    z_mean = z_mean, z_log_var = z_log_var,\
                    margin = args.margin, margin_btw = args.margin_between_b_and_m, \
                    mids = args.is_enc_kld_custom_mid, \
                    is_for_selector = True, \
                    kld_dev_scale = self.encoder.kld_dev_scale, kld_beta = args.kld_beta,
                    mids_for_y_class_batch = mids_for_y_class_batch)
                
                loss = loss.to('cpu').detach().numpy()
                supcon_loss = supcon_loss.to('cpu').detach().numpy()
                #gap_loss = gap_loss.to('cpu').detach().numpy()
                #vae_loss = vae_loss.to('cpu').detach().numpy()
                xent_loss = xent_loss.to('cpu').detach().numpy()
                #gap_loss = gap_loss.to('cpu').detach().numpy()
                
                del HiDistanceKldCustomXentEnsemble8
                torch.cuda.empty_cache()
                
                
            else:
                # other loss functions pending
                raise Exception(f'local pseudo loss for {args.loss_func} not implemented.')
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            # update the loss values for i
            # nn_loss[i] = loss[0]
            # loss[0] means [indice, score] #? NO!!!!
            
            # I think the reason why appending loss[0] is that [0] is a test_sample
            self.sample_scores.append(loss[0])
            #print(f"loss shape {np.array(loss).shape}") # 1024
        '''
            # only display the test samples
            if (i + 1) % (args.display_interval * 3) == 0:
                logging.debug('Train + Test: [0][{0}/{1}]\t'
                    'BT {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'DT {data_time.val:.3f} ({data_time.avg:.3f})  '
                    'i {i} loss {l}'.format(
                    i + 1, sample_num, batch_time=batch_time,
                    data_time=data_time, i=i, l=loss[0]))
        '''
        #print(f"self.sample_scores {self.sample_scores}")
        #print(f"self.sample_scores shape {np.array(self.sample_scores).shape}") # 4991
        
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

