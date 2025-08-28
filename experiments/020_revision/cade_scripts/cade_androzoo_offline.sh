#! /bin/bash

#SBATCH -t 04:00:00

#SBATCH -n 1

#SBATCH -c 8

SCH=step
DECAY=0.5
DATA=gen_androzoo_drebin
TRAIN_START=2019-01
TRAIN_END=2019-12
TEST_START=2020-01
TEST_END=2021-12
VALID_DATE=2021-02
RESULT_DIR=cade_results

modeldim="512-384-256-128"
S='triplet'
B=1536



###############################################################

OPT=adam
E=100
LR=0.00005

#ENCODER='cae-mlp'
#CLASSIFIER='cae-mlp'
ENCODER='cae-kld-ensemble-mlp'
CLASSIFIER='cae-kld-ensemble-mlp'

#LOSS='triplet-mse-xent'
LOSS='triplet-mse-kld-ensemble-xent'

CENTROID_TYPE=''
KLD_SCALE=3.0

CSV_NAME="1"

SLP=0

###############################################################


TS=$(date "+%m.%d-%H.%M.%S")

nohup python -u relabel.py	                                \
            --sleep ${SLP}                                  \
            --ood                                           \
            --retrain-first 1                               \
            --is-only-test-eval-without-al 1                \
            --margin 10                                     \
            --margin-between-b-and-m 2                      \
            --kld-scale ${KLD_SCALE}                        \
            --centroid-type ${CENTROID_TYPE}                          \
            --is-enc-kld-custom-mid 1                       \
            --is-valid 0                                    \
            --data ${DATA}                                  \
            --benign_zero                                   \
            --train_start ${TRAIN_START}                    \
            --train_end ${TRAIN_END}                        \
            --test_start ${TEST_START}                      \
            --test_end ${TEST_END}                          \
            --valid_date ${VALID_DATE}                      \
            --encoder ${ENCODER}                            \
            --classifier ${CLASSIFIER}                      \
            --loss_func ${LOSS}                             \
            --enc-hidden ${modeldim}                        \
            --mlp-hidden 100-100                            \
            --mlp-dropout 0.2                               \
            --sampler ${S}                                  \
            --bsize ${B}                                    \
            --optimizer ${OPT}                              \
            --scheduler ${SCH}                              \
            --learning_rate ${LR}                           \
            --lr_decay_rate ${DECAY}                        \
            --lr_decay_epochs "10,500,10"                   \
            --epochs ${E}                                   \
            --encoder-retrain                               \
            --cae-lambda 0.1                                \
            --xent-lambda 100                               \
            --display-interval 180                          \
            --al                                            \
            --reduce "none"                                 \
            --sample_reduce 'mean'                          \
            --result experiments/020_revision/${RESULT_DIR}/${ENCODER}_androzoo_${CENTROID_TYPE}_offline_lr${LR}_${OPT}_${SCH}_${DECAY}_e${E}_test_${TEST_START}_${TEST_END}${CSV_NAME}.csv \
            --log_path experiments/020_revision/${RESULT_DIR}/${ENCODER}_androzoo_${CENTROID_TYPE}_offline_lr${LR}_${OPT}_${SCH}_${DECAY}_e${E}_test_${TEST_START}_${TEST_END}_${TS}.log \
            >> experiments/020_revision/${RESULT_DIR}/${ENCODER}_androzoo_${CENTROID_TYPE}_offline_lr${LR}_${OPT}_${SCH}_${DECAY}_e${E}_test_${TEST_START}_${TEST_END}_${TS}.log 2>&1 &

wait