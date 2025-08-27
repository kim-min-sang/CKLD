#! /usr/bin/env python

#from distutils import core
import os
import sys

import datetime as dt
import logging
import numpy as np
import time
import torch
from pprint import pformat
#from collections import Counter, defaultdict
from dateutil.relativedelta import relativedelta
#from pprint import pformat
#from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import LinearSVC
from collections import Counter, defaultdict
# local imports
import data
import utils
import kld_func
from common import get_model_stats
#from common import draw_tsne,draw_tsne_concat,draw_tsne_split
from model import SimpleEncClassifier
from model import EncKldCustomMlpEnsemble6, CAEKldEnsembleMlp, CAEMlp, TripletMlp, TripletKldEnsembleMlp

from selector_pseudo_loss import LocalPseudoLossSelector
from selector_cadeood import OODSelector, OODKldEnsembleSelector
from selector_simple import UncertainPredScoreSelector
from utils import save_model
from train import train_encoder

def eval_classifier(args, classifier, cur_month_str, X, y_binary, y_family, train_families, \
                        fout, fam_out, stat_out, gpu = False, multi = False):
    if gpu == True:
        X_tensor = torch.from_numpy(X).float()
        if torch.cuda.is_available():
            X_tensor = X_tensor.cuda()
            y_pred = classifier.cuda().predict(X_tensor)
            y_pred = y_pred.cpu().detach().numpy()
        else:
            y_pred = classifier.predict(X_tensor).numpy()
    else:
        y_pred = classifier.predict(X)
    
    if args.multi_class == True:
        y_pred_bin = np.where(y_pred == 0, 0, 1)
    else:
        y_pred_bin = y_pred

    tpr, tnr, fpr, fnr, acc, precision, f1 = get_model_stats(y_binary, y_pred_bin, multi_class = multi)
    fout.write('%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n' % \
                (cur_month_str, tpr, tnr, fpr, fnr, acc, precision, f1))
    fout.flush()
    if multi == False:
        tn, fp, fn, tp = confusion_matrix(y_binary, y_pred_bin).ravel()
        stat_out.write('%s\t%d\t%d\t%d\t%d\t%d\n' % \
                    (cur_month_str, X.shape[0], tp, tn, fp, fn))
        stat_out.flush()

    # check FNR within different families.
    family_cnt = defaultdict(lambda: 0)
    for idx, family in enumerate(y_family):
        family_cnt[family] += 1
    neg_by_fam = defaultdict(lambda: 0)
    family_to_idx = defaultdict(list)
    # y_family can be all_train_family since we only care abou False Negatives
    fn_indices = np.where((y_binary != y_pred_bin) & (y_binary != 0))[0]
    for idx in fn_indices:
        family = y_family[idx]
        neg_by_fam[family] += 1
        family_to_idx[family].append(idx)
    for family, neg_cnt in neg_by_fam.items():
        new = family not in train_families
        fam_total = family_cnt[family]
        fam_rate = neg_cnt / float(fam_total)
        fam_out.write('%s\t%s\t%s\t%s\t%d\n' % (cur_month_str, new, family, fam_rate, neg_cnt))
        fam_out.flush()
    return y_pred, neg_by_fam, family_to_idx





def main():
    """
    Step (0): Init log path and parse args.
    """
    args = utils.parse_args()

    # lr_decay_epochs don't be used
    start_epoch, end_epoch, step = args.lr_decay_epochs.split(',') 
    args.lr_decay_epochs = list([range(int(start_epoch), int(end_epoch), int(step))])

    log_file_path = args.log_path
    
    # verbose means too much talker lol
    if args.verbose == False:
        logging.basicConfig(filename=log_file_path,
                            filemode='a',
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            level=logging.INFO,
                            )
    else:
        logging.basicConfig(filename=log_file_path,
                            filemode='a',
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            level=logging.DEBUG,
                            )
    
    logging.info('Running with configuration:\n' + pformat(vars(args)))

    """
    Step (1): Prepare the training dataset. Load the feature vectors and labels.
    """
    logging.info(f'Loading {args.data} training dataset')
    if args.data.startswith('tesseract') or \
        args.data.startswith('gen_tesseract') or \
        args.data.startswith('fam_tesseract') or \
        args.data.startswith('emberv2'):
        X_train, y_train, all_train_family = data.load_range_dataset_w_benign(args.data, args.train_start, args.train_end)
    else:
        X_train, y_train, y_train_family = data.load_range_dataset_w_benign(args.data, args.train_start, args.train_end)

        ben_len = X_train.shape[0] - y_train_family.shape[0]
        y_ben_family = np.full(ben_len, 'benign')
        all_train_family = np.concatenate((y_train_family, y_ben_family), axis=0)
    train_families = set(all_train_family)
    
    counted_labels = Counter(y_train)
    logging.info(f'Loaded X_train: {X_train.shape}, {y_train.shape}')
    logging.info(f'y_train labels: {np.unique(y_train)}')
    logging.info(f'y_train: {Counter(y_train)}')
    
    new_y_mapping = {}
    for _, label in enumerate(np.unique(y_train)):
        new_y_mapping[label] = label

    """
    Step (2): Variable names and file names.
    """
    if args.train_start != args.train_end:
        train_dataset_name = f'{args.train_start}to{args.train_end}'
    else:
        train_dataset_name = f'{args.train_start}'

    SAVED_MODEL_FOLDER = 'models/'
    NUM_FEATURES = X_train.shape[1]
    NUM_CLASSES = len(np.unique(y_train))

    logging.info(f'Number of features: {NUM_FEATURES}; Number of classes: {NUM_CLASSES}')

    y_train_binary = np.array([1 if item != 0 else 0 for item in y_train])
    BIN_NUM_CLASSES = 2

    """
    Step (3): Train the encoder model.
    `encoder` needs to have the same APIs.
    If they don't have the required API, we could use a wrapper.
    """
    train_encoder_func = train_encoder
    if args.encoder == None:
        logging.info('Not using an encoder. Using the input features directly.')
    elif args.encoder == 'simple-enc-mlp':
        enc_dims = utils.get_model_dims('Encoder', NUM_FEATURES,
                            args.enc_hidden, NUM_CLASSES)
        mlp_dims = utils.get_model_dims('MLP', enc_dims[-1], args.mlp_hidden, BIN_NUM_CLASSES)
        encoder = SimpleEncClassifier(enc_dims, mlp_dims)
        encoder_name = 'simple_enc_classifier'
    elif args.encoder == 'enc-kld-custom-mlp-ensemble6':
        enc_dims = utils.get_model_dims('Encoder', NUM_FEATURES,
                            args.enc_hidden, NUM_CLASSES)
        mlp_dims = utils.get_model_dims('MLP', enc_dims[-1], args.mlp_hidden, BIN_NUM_CLASSES)
        
        kld_dev_scale = kld_func.get_kld_dev_scale(args.mid_type, enc_dims[-1], args.kld_scale)
        
        encoder = EncKldCustomMlpEnsemble6(enc_dims, mlp_dims, kld_scale = args.kld_scale, kld_dev_scale = kld_dev_scale)
        encoder_name = 'enc-kld-custom-mlp-ensemble6'
    elif args.encoder == 'cae-mlp':
        enc_dims = utils.get_model_dims('Encoder', NUM_FEATURES,
                            args.enc_hidden, NUM_CLASSES)
        mlp_dims = utils.get_model_dims('MLP', enc_dims[-1], args.mlp_hidden, BIN_NUM_CLASSES)
        encoder = CAEMlp(enc_dims, mlp_dims)
        encoder_name = 'cae-mlp'
    elif args.encoder == 'cae-kld-ensemble-mlp':
        enc_dims = utils.get_model_dims('Encoder', NUM_FEATURES,
                            args.enc_hidden, NUM_CLASSES)
        mlp_dims = utils.get_model_dims('MLP', enc_dims[-1], args.mlp_hidden, BIN_NUM_CLASSES)
        encoder = CAEKldEnsembleMlp(enc_dims, mlp_dims, kld_scale = args.kld_scale)
        encoder_name = 'cae-kld-ensemble-mlp'
    elif args.encoder == 'triplet-mlp':
        enc_dims = utils.get_model_dims('Encoder', NUM_FEATURES,
                            args.enc_hidden, NUM_CLASSES)
        mlp_dims = utils.get_model_dims('MLP', enc_dims[-1], args.mlp_hidden, BIN_NUM_CLASSES)
        encoder = TripletMlp(enc_dims, mlp_dims)
        encoder_name = 'triplet-mlp'
    elif args.encoder == 'triplet-kld-ensemble-mlp':
        enc_dims = utils.get_model_dims('Encoder', NUM_FEATURES,
                            args.enc_hidden, NUM_CLASSES)
        mlp_dims = utils.get_model_dims('MLP', enc_dims[-1], args.mlp_hidden, BIN_NUM_CLASSES)
        encoder = TripletKldEnsembleMlp(enc_dims, mlp_dims, kld_scale = args.kld_scale)
        encoder_name = 'triplet-kld-ensemble-mlp'
    else:
        raise Exception(f'The encoder {args.encoder} is not supported yet.')

    MODEL_DIR = os.path.join(SAVED_MODEL_FOLDER, train_dataset_name)
    utils.create_folder(MODEL_DIR)
    if args.optimizer == 'adam':
        optimizer_func = torch.optim.Adam
        optimizer = torch.optim.Adam(encoder.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer_func = torch.optim.SGD
        optimizer = torch.optim.SGD(encoder.parameters(), lr=args.learning_rate)
    else:
        raise Exception(f'The optimizer {args.optimizer} is not supported yet.')
    
    enc_dims_str = str(enc_dims).replace(' ', '').replace(',', '-').replace('[', '').replace(']', '') 
    
    ENC_MODEL_PATH = os.path.join(MODEL_DIR, f'{encoder_name}_{args.mid_type}_{enc_dims_str}_{args.loss_func}' + \
                                f'_xent{args.xent_lambda}' + \
                                f'_mselambda{args.mse_lambda}' + \
                                f'_caelambda{args.cae_lambda}' + \
                                f'_{args.optimizer}_{args.scheduler}' + \
                                f'_lr{args.learning_rate}_decay{args.lr_decay_rate}' + \
                                f'_{args.sampler}_b{args.bsize}_e{args.epochs}.pth')
    logging.info(f'Initial encoder model: ENC_MODEL_PATH {ENC_MODEL_PATH}')

    X_train_final = X_train
    y_train_final = y_train
    y_train_binary_final = y_train_binary
    upsample_values = None
    
    logging.info(f'upsample_values {upsample_values}')
    logging.info(f'X_train_final.shape: {X_train_final.shape}')
    logging.info(f'y_train_final.shape: {y_train_final.shape}')
    logging.info(f'y_train_binary_final.shape: {y_train_binary_final.shape}')
    logging.info(f'y_train_final labels: {np.unique(y_train_final)}')
    logging.info(f'y_train_final: {Counter(y_train_final)}')

    if args.encoder in ['simple-enc-mlp', 'enc-kld-custom-mlp-ensemble6'] == True:
        counted_y_train = Counter(y_train)
        singleton_families = [family for family, count in counted_y_train.items() if count == 1]
        logging.info(f'Singleton families: {singleton_families}')
        logging.info(f'Number of singleton families: {len(singleton_families)}')
        
        X_train_final = np.array([X_train[i] for i, family in enumerate(y_train) if family not in singleton_families])
        y_train_final = np.array([y_train[i] for i, family in enumerate(y_train) if family not in singleton_families])
        y_train_binary_final = np.array([y_train_binary[i] for i, family in enumerate(y_train) if family not in singleton_families])
        all_train_family = np.array([all_train_family[i] for i, family in enumerate(y_train) if family not in singleton_families])
        logging.info(f'After removing singleton families: X_train_final.shape, {X_train_final.shape}, y_train_final.shape, {y_train_final.shape}')
        logging.info(f'After removing singleton families: {Counter(y_train_final)}')

    try:
        if args.retrain_first == True or not os.path.exists(ENC_MODEL_PATH):
            s1 = time.time()
            train_encoder_func(args, encoder, X_train_final, y_train_final, y_train_binary_final, \
                            optimizer, args.epochs, ENC_MODEL_PATH, adjust = True, save_best_loss = False, \
                            save_snapshot = args.snapshot, is_first_train = True)
            e1 = time.time()
            logging.info(f'Training Encoder model time: {(e1 - s1):.3f} seconds')
            logging.info('Saving the model...')
            save_model(encoder, optimizer, args, args.epochs, ENC_MODEL_PATH)
            logging.info(f'Training Encoder model finished: {ENC_MODEL_PATH}')
        else:
            logging.info('Loading the model...')
            state_dict = torch.load(ENC_MODEL_PATH)
            encoder.load_state_dict(state_dict['model'])
    except Exception as e:
        print(e)
        import traceback
        traceback.print_exc()
        time.sleep(3)
        return False

    """
    Select the classifier model.
    """
    if args.cls_feat == 'encoded':
        X_train_tensor = torch.from_numpy(X_train).float()
        if torch.cuda.is_available():
            X_train_tensor = X_train_tensor.cuda()
            X_feat_tensor = encoder.cuda().encode(X_train_tensor)
            X_train_feat = X_feat_tensor.cpu().detach().numpy()
        else:
            X_train_feat = encoder.encode(X_train_tensor).numpy()
    else:
        X_train_feat = X_train

    if args.classifier == args.encoder:
        classifier = encoder
        CLS_MODEL_PATH = ENC_MODEL_PATH
        cls_gpu = True
    else:
        raise Exception(f'The classifier {args.classifier} is not supported yet.')

    # save training acc
    fout = open(args.result, 'w')
    fout.write('date\tTPR\tTNR\tFPR\tFNR\tACC\tPREC\tF1\n')
    fam_out = open(args.result.split('.csv')[0]+'_family.csv', 'w')
    fam_out.write('Month\tNew\tFamily\tFNR\tCnt\n')
    stat_out = open(args.result.split('.csv')[0]+'_stat.csv', 'w')
    stat_out.write('date\tTotal\tTP\tTN\tFP\tFN\n')
    eval_classifier(args, classifier, args.train_end, X_train_feat, y_train_binary, all_train_family, train_families, \
                    fout, fam_out, stat_out, gpu = cls_gpu, multi = args.eval_multi)
    sample_out = open(args.result.split('.csv')[0]+'_sample.csv', 'w')
    sample_out.write('date\tCount\tIndex\tTrue\tPred\tFamily\tScore\n')
    sample_out.flush()
    sample_explanation = open(args.result.split('.csv')[0]+'_sample_explanation.csv', 'w')
    sample_explanation.write('date\tCorrect\tWrong\tBenign\tMal\tNew_fam_cnt\tNew_fam\tUnique_fam\n')
    sample_explanation.flush()

    """
    Set up the selector.
    """
    if args.al == True:
        strategy = 'strategy'
        if args.rand == True: # False
            strategy += '_rand'
        if args.unc == True: # False
            strategy += '_unc'
            if args.multi_class == True:
                strategy += '_multi'
                selector = MultiUncertainPredScoreSelector(classifier)
            else:
                selector = UncertainPredScoreSelector(classifier)
        if args.ood == True: # False
            strategy += '_ood'
            selector = OODSelector(encoder)
        if args.ood_kld_ensemble == True: # False
            strategy += '_ood_kld_ensemble'
            selector = OODKldEnsembleSelector(encoder)
        if args.transcend == True: # False
            strategy += '_transcend'
            if args.criteria == 'cred':
                crit = 'cred'
            elif args.criteria == 'conf':
                crit = 'conf'
            else:
                # args.criteria == 'cred+conf'
                crit = 'cred+conf'
            selector = TranscendSelector(encoder, crit=crit)
        if args.local_pseudo_loss == True: # True
            strategy += '_local_pseudo_loss'
            strategy += f'_{args.reduce}'
            selector = LocalPseudoLossSelector(encoder) # this encoder was trained at above
        # I don't know why encoder will be retrained #?
        if args.encoder_retrain == True: # True
            strategy += '_encretrain'
        strategy += f'_warm_{args.al_optimizer}_wlr{args.al_epochs}_we{args.warm_learning_rate}'
        strategy += f'_count{args.count}'

        if args.encoder != None:
            NEW_ENC_MODEL_PATH = ENC_MODEL_PATH.split('.pth')[0] + f'_retrain.pth'
        if args.classifier != args.encoder:
            name, ext = CLS_MODEL_PATH.rsplit('.', 1)
            NEW_CLS_MODEL_PATH = f'{name}_retrain_{strategy}.{ext}'
    
    """
    Step (5): Go over each month in the test range.
    """
    start = dt.datetime.strptime(args.test_start, '%Y-%m')
    end = dt.datetime.strptime(args.test_end, '%Y-%m')
    valid_date = dt.datetime.strptime(args.valid_date, '%Y-%m')
    cur_month = start
    val_month = valid_date

    month_loop_cnt = 0
    prev_train_size = X_train.shape[0]
    cur_sample_indices = []
    
    while cur_month <= end:
        """
        Step (6): Load test data.
        """
        cur_month_str = cur_month.strftime('%Y-%m')

        if args.data.startswith('tesseract'):
            X_test, y_test, all_test_family = data.load_range_dataset_w_benign(args.data, cur_month_str, cur_month_str)
        else:
            X_test, y_test, y_test_family = data.load_range_dataset_w_benign(args.data, cur_month_str, cur_month_str)
            ben_test_len = X_test.shape[0] - y_test_family.shape[0]
            y_ben_test_family = np.full(ben_test_len, 'benign')
            all_test_family = np.concatenate((y_test_family, y_ben_test_family), axis=0)
        
        logging.info(f'X_test.shape {X_test.shape}')
        logging.info(f'y_test.shape {y_test.shape}')
        logging.info(f'y_test_family.shape {y_test_family.shape}')

        y_test_binary = np.array([1 if item != 0 else 0 for item in y_test])
        X_test_tensor = torch.from_numpy(X_test).float()
        if args.encoder != None:
            if torch.cuda.is_available():
                X_test_feat_tensor = encoder.cuda().encode(X_test_tensor.cuda())
                X_test_encoded = X_test_feat_tensor.cpu().detach().numpy()
            else:
                X_test_encoded = encoder.encode(X_test_tensor).numpy()
        
        if args.cls_feat == 'encoded':
            X_test_feat = X_test_encoded
        else:
            X_test_feat = X_test

        # Only month_loop_cnt == 0 will we update the accum data with new month data
        if args.accumulate_data == True and month_loop_cnt == 0:
            if cur_month_str == '2013-01':
                X_test_accum = X_test
                y_test_accum = y_test
                all_test_family_accum = all_test_family
                X_test_accum_feat = X_test_feat # for the classifier
            else:
                X_test_accum = np.concatenate((X_test_accum, X_test), axis=0)
                y_test_accum = np.concatenate((y_test_accum, y_test), axis=0)
                all_test_family_accum = np.concatenate((all_test_family_accum, all_test_family), axis=0)
                X_test_accum_feat = np.concatenate((X_test_accum_feat, X_test_feat), axis=0) # for the classifier
        elif month_loop_cnt == 0:
            X_test_accum = X_test
            y_test_accum = y_test
            all_test_family_accum = all_test_family
            X_test_accum_feat = X_test_feat # for the classifier
        
        y_test_binary_accum = np.array([1 if item != 0 else 0 for item in y_test_accum])
        
        """
        Evaluate the test performance.
        """
        logging.info(f'Testing on {cur_month_str}')
        
        y_test_pred, neg_by_fam, family_to_idx = eval_classifier(args, classifier, cur_month_str, X_test_feat, y_test_binary, all_test_family, train_families, \
                        fout, fam_out, stat_out, gpu = cls_gpu, multi = args.eval_multi)
        
        if args.accumulate_data == True and month_loop_cnt == 0:
            if cur_month_str == '2013-01':
                y_test_pred_accum = y_test_pred
            else:
                y_test_pred_accum = np.concatenate((y_test_pred_accum, y_test_pred), axis=0)
        elif month_loop_cnt == 0:
            y_test_pred_accum = y_test_pred
            
        if args.is_only_test_eval_without_al:
            cur_month += relativedelta(months=args.step_month)
            continue
        
        if torch.cuda.is_available():
            X_train_tensor = torch.from_numpy(X_train_final).float()
            X_train_tensor = X_train_tensor.cuda()
            kld_encoded_tensor, _, _, _, c_encoded_tensor = encoder.cuda().encode(X_train_tensor, is_all_return=True)
            c_encoded = c_encoded_tensor.cpu().detach().numpy()
            try:
                kld_encoded = kld_encoded_tensor.cpu().detach().numpy()
            except Exception as e:
                pass
            
        """
        Step (7): Pick samples. Expand the training set.
        """
        if args.al == True and cur_month != end:
            if cls_gpu == True:
                X_test_accum_feat_tensor = torch.from_numpy(X_test_accum_feat).float()
                if torch.cuda.is_available():
                    pred_scores_accum = classifier.cuda().predict_proba(X_test_accum_feat_tensor.cuda())
                    pred_scores_accum = pred_scores_accum.cpu().detach().numpy()
                else:
                    pred_scores_accum = classifier.predict_proba(X_test_accum_feat_tensor)
            else:
                pred_scores_accum = classifier.predict_proba(X_test_accum_feat)
            test_offset = prev_train_size
            
            if args.ood == True:
                sample_indices, sample_scores = selector.select_samples(X_train, y_train, \
                                                                X_test_accum, \
                                                                args.count)
            elif args.transcend == True:
                sample_indices, sample_scores = selector.select_samples(X_train, y_train, \
                                                                X_test_accum, \
                                                                args.count)
            elif args.unc == True:
                sample_indices, sample_scores = selector.select_samples(args, X_test_feat, y_test_pred_accum, args.count)
            elif args.local_pseudo_loss == True:
                total_epochs = 10
                sample_indices, sample_scores = selector.select_samples(args, \
                                                                X_train, y_train, y_train_binary, \
                                                                X_test_accum, y_test_pred_accum, \
                                                                total_epochs, \
                                                                test_offset, \
                                                                all_test_family_accum, \
                                                                args.count)
            else:
                raise ValueError('Unknown sampling method')
            
            """
            Step (8): expand the training set: X_train, y_train, etc.
            """
            cnt = 0
            for idx in sample_indices:
                try:
                    fam_label = all_test_family_accum[idx]
                except IndexError:
                    fam_label = 'benign'
                pred_label = int(y_test_pred_accum[idx])
                if args.classifier == 'gbdt':
                    sample_out.write('%s\t%d\t%d\t%s\t%.4f\t%s\t%.4f\n' % \
                                (cur_month_str, cnt, idx, y_test_binary_accum[idx], pred_scores_accum[idx], fam_label, sample_scores[idx]))
                else:
                    sample_out.write('%s\t%d\t%d\t%s\t%.4f\t%s\t%.4f\n' % \
                                (cur_month_str, cnt, idx, y_test_binary_accum[idx], pred_scores_accum[idx][pred_label], fam_label, sample_scores[idx]))
                cnt += 1
            sample_out.flush()
            
            correct_pred = 0
            wrong_pred = 0
            fam_dict = defaultdict(lambda: 0)
            
            for idx in sample_indices:
                try:
                    fam_label = all_test_family_accum[idx]
                except IndexError:
                    fam_label = 'benign'
                true_label = y_test_binary_accum[idx]
                pred_label = int(y_test_pred_accum[idx])
                logging.info(f'{idx}, {fam_label}, {true_label}, {pred_label}, {pred_scores_accum[idx][pred_label]}')
                if true_label == pred_label:
                    correct_pred += 1
                else:
                    wrong_pred += 1
                fam_dict[fam_label] += 1
            benign_num = fam_dict['benign']
            mal_num = cnt - benign_num
            new_families_lst = list(set(fam_dict.keys()) - set(all_train_family.flatten()))
            uniq_families_lst = list(fam_dict.keys())
            uniq_families = ",".join(uniq_families_lst)
            new_fam_cnt = 0
            for fam in new_families_lst:
                new_fam_cnt += fam_dict[fam]
            new_families_selected = ",".join(new_families_lst)
            sample_explanation.write('%s\t%d\t%d\t%d\t%d\t%d\t%s\t%s\n' % \
                (cur_month_str, correct_pred, wrong_pred, benign_num, mal_num, new_fam_cnt, new_families_selected, uniq_families))
            sample_explanation.flush()
            
            X_train = np.concatenate((X_train, X_test_accum[sample_indices]), axis=0)
            y_train_binary = np.concatenate((y_train_binary, y_test_binary_accum[sample_indices]), axis=0)
            original_y = y_test_accum[sample_indices] 
            new_y = np.copy(original_y)
            new_label = max(y_train) + 1
            for idx, label in enumerate(original_y):
                if new_y_mapping.get(label, None) != None:
                    new_y[idx] = new_y_mapping[label] # I think can remove
                else:
                    new_y_mapping[label] = new_label
                    new_y[idx] = new_label # I think can remove
                    new_label += 1
            y_train = np.concatenate((y_train, new_y), axis=0)
            logging.info(f'y_test_accum[sample_indices] {y_test_accum[sample_indices]}')
            logging.info(f'new_y {new_y}')
            
            all_train_family = np.concatenate((all_train_family, all_test_family_accum[sample_indices]), axis=0)
            
            X_test_accum = np.delete(X_test_accum, sample_indices, axis=0)
            X_test_accum_feat = np.delete(X_test_accum_feat, sample_indices, axis=0)
            y_test_accum = np.delete(y_test_accum, sample_indices, axis=0)
            all_test_family_accum = np.delete(all_test_family_accum, sample_indices, axis=0)
            y_test_pred_accum = np.delete(y_test_pred_accum, sample_indices, axis=0)

            X_train_final = X_train
            y_train_final = y_train
            y_train_binary_final = y_train_binary
            upsample_values = None

            logging.info(f'upsample_values {upsample_values}')
            logging.info(f'X_train_final.shape: {X_train_final.shape}')
            logging.info(f'y_train_final.shape: {y_train_final.shape}')
            logging.info(f'y_train_binary_final.shape: {y_train_binary_final.shape}')
            logging.info(f'y_train_final labels: {np.unique(y_train_final)}')
            logging.info(f'y_train_final: {Counter(y_train_final)}')

            """
            Step (9): Retrain the sample selection model, e.g., Enc + MLP.
            """
            
            if args.encoder_retrain == True:
                if args.al_optimizer == None:
                    logging.info(f'Active learning using optimizer {args.optimizer}')
                    pass
                elif args.al_optimizer == 'adam':
                    optimizer_func = torch.optim.Adam
                    logging.info(f'Active learning using optimizer {args.al_optimizer}')
                elif args.al_optimizer == 'sgd':
                    optimizer_func = torch.optim.SGD
                    logging.info(f'Active learning using optimizer {args.al_optimizer}')
                
                
                optimizer = optimizer_func(encoder.parameters(), lr=args.warm_learning_rate)
                al_total_epochs = args.al_epochs
                
                if args.encoder != None:
                    s2 = time.time()
                    logging.info('Training Encoder model...')
                    train_encoder_func(args, encoder, X_train_final, y_train_final, y_train_binary_final,
                                optimizer, al_total_epochs, NEW_ENC_MODEL_PATH,
                                weight = None,
                                adjust = True, warm = not args.cold_start, save_best_loss = False, is_first_train = False)
                    e2 = time.time()
                    logging.info(f'Training Encoder model time: {(e2 - s2):.3f} seconds')
                logging.info(f'Saving the model...')
                save_model(encoder, optimizer, args, args.epochs, NEW_ENC_MODEL_PATH)
                logging.info(f'Retraining Encoder model finished: {NEW_ENC_MODEL_PATH}')
            
            """
            Retrain the classifier if it's different from the encoder
            """
            # this is to retrain the classifier
            if args.cls_feat == 'encoded':
                X_train_tensor = torch.from_numpy(X_train).float()
                if torch.cuda.is_available():
                    X_train_tensor = X_train_tensor.cuda()
                    X_feat_tensor = encoder.cuda().encode(X_train_tensor)
                    X_train_feat = X_feat_tensor.cpu().detach().numpy()
                else:
                    X_train_feat = encoder.encode(X_train_tensor).numpy()
            else:
                X_train_feat = X_train
                logging.info('Classifier model is the same as the encoder...')
                NEW_CLS_MODEL_PATH = NEW_ENC_MODEL_PATH
        prev_train_size = X_train.shape[0]
        cur_month += relativedelta(months=1)
        
        if args.is_valid:
            if val_month < cur_month:
                break
    
    # finish writing the result file
    fout.close()
    fam_out.close()
    sample_out.close()
    stat_out.close()
    # sample_score_out.close()
    sample_explanation.close()
    return

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    logging.info(f'time elapsed: {end - start} seconds')
