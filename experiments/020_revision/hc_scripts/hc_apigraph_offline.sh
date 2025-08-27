#! /bin/bash

#SBATCH -t 03:00:00

#SBATCH -n 1

#SBATCH -c 8

SCH=step
DECAY=0.95

DATA=gen_apigraph_drebin
TRAIN_START=2012-01
TRAIN_END=2012-12
TEST_START=2013-01
TEST_END=2014-12
VALID_DATE=2013-06
RESULT_DIR=hc_results

modeldim="512-384-256-128"
S='half'
B=1024


###############################################################

OPT=adam
E=100
LR=0.0005

#LOSS='hi-dist-kld-custom-xent-ensemble6'
LOSS='hi-dist-xent'

#ENCODER='enc-kld-custom-mlp-ensemble6'
#CLASSIFIER='enc-kld-custom-mlp-ensemble6'
ENCODER='simple-enc-mlp'
CLASSIFIER='simple-enc-mlp'

MID_TYPE='bin'
KLD_SCALE=3.0

CSV_NAME="_hc_apigraph_1"

SLP=0

###############################################################

TS=$(date "+%m.%d-%H.%M.%S")

nohup python -u relabel.py	                                \
            --sleep ${SLP}                                  \
            --retrain-first 1                               \
            --is-only-test-eval-without-al 1                \
            --is-accum-samples-load 0                       \
            --is-accum-samples-save 0                       \
            --margin 10                                     \
            --margin-between-b-and-m 2                      \
            --kld-scale ${KLD_SCALE}                        \
            --is-enc-kld-custom-mid 1                       \
            --mid-type ${MID_TYPE}                          \
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
            --xent-lambda 100                               \
            --display-interval 180                          \
            --al                                            \
            --local_pseudo_loss                             \
            --reduce "none"                                 \
            --sample_reduce 'mean'                          \
            --result experiments/020_revision/${RESULT_DIR}/hc_apigraph_${MID_TYPE}_lr${LR}_${OPT}_${SCH}_${DECAY}_e${E}_test_${TEST_START}_${TEST_END}${CSV_NAME}.csv \
            --log_path experiments/020_revision/${RESULT_DIR}/hc_apigraph_${MID_TYPE}_lr${LR}_${OPT}_${SCH}_${DECAY}_e${E}_test_${TEST_START}_${TEST_END}_${TS}.log \
            >> experiments/020_revision/${RESULT_DIR}/hc_apigraph_${MID_TYPE}_lr${LR}_${OPT}_${SCH}_${DECAY}_e${E}_test_${TEST_START}_${TEST_END}_${TS}.log 2>&1 &

wait