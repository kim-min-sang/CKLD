#! /bin/bash

#SBATCH -t 03:00:00

#SBATCH -n 1

#SBATCH -c 8

OPT=adam
SCH=step
DECAY=0.95
DATA=gen_apigraph_drebin
TRAIN_START=2012-01
TRAIN_END=2012-12
TEST_START=2013-01
TEST_END=2018-12
VALID_DATE=2018-12
RESULT_DIR=triplet_results

modeldim="512-384-256-128"
S='triplet'
B=1536


###############################################################

OPT=adam
E=150
LR=0.0005

ENCODER='triplet-mlp'
CLASSIFIER='triplet-mlp'
#ENCODER='triplet-kld-ensemble-mlp'
#CLASSIFIER='triplet-kld-ensemble-mlp'

LOSS='triplet-xent'
#LOSS='triplet-kld-ensemble-xent'

MID_TYPE='bin'
KLD_SCALE=2.0

CSV_NAME="_triplet_apigraph_1"

SLP=0

###############################################################






TS=$(date "+%m.%d-%H.%M.%S")

nohup python -u relabel.py	                                \
            --sleep ${SLP}                                  \
            --unc                                           \
            --retrain-first 1                               \
            --is-only-test-eval-without-al 1                \
            --margin 10                                     \
            --margin-between-b-and-m 2                      \
            --is-enc-kld-custom-mid 1                       \
            --mid-type ${MID_TYPE}                          \
            --kld-scale ${KLD_SCALE}                        \
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
            --cae-lambda 1                                \
            --triplet-lambda 1                              \
            --xent-lambda 100                               \
            --display-interval 180                          \
            --al                                            \
            --reduce "none"                                 \
            --sample_reduce 'mean'                          \
            --result experiments/020_revision/${RESULT_DIR}/triplet_apigraph_${MID_TYPE}_lr${LR}_${OPT}_${SCH}_${DECAY}_e${E}_test_${TEST_START}_${TEST_END}${CSV_NAME}.csv \
            --log_path experiments/020_revision/${RESULT_DIR}/triplet_apigraph_${MID_TYPE}_lr${LR}_${OPT}_${SCH}_${DECAY}_e${E}_test_${TEST_START}_${TEST_END}_${TS}.log \
            >> experiments/020_revision/${RESULT_DIR}/triplet_apigraph_${MID_TYPE}_lr${LR}_${OPT}_${SCH}_${DECAY}_e${E}_test_${TEST_START}_${TEST_END}_${TS}.log 2>&1 &

wait