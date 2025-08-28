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
VALID_DATE=2020-06
RESULT_DIR=hc_results
AL_OPT=adam

modeldim="512-384-256-128"

S='half'
B=1024

###############################################################

CNT=200

OPT=sgd

E=200
WE=50

LR=0.001
WLR=0.00001

LOSS='hi-dist-kld-custom-xent-ensemble6'
#LOSS='hi-dist-xent'

ENCODER='enc-kld-custom-mlp-ensemble6'
CLASSIFIER='enc-kld-custom-mlp-ensemble6'
#ENCODER='simple-enc-mlp'
#CLASSIFIER='simple-enc-mlp'

CENTROID_TYPE='fam'
KLD_SCALE=2.0

CSV_NAME="1"

SLP=0

###############################################################

TS=$(date "+%m.%d-%H.%M.%S")

nohup python -u relabel.py	                                \
            --retrain-first 1                               \
            --is-only-test-eval-without-al 0                \
            --is-accum-samples-load 0                       \
            --is-accum-samples-save 0                       \
            --sleep ${SLP}                                  \
            --margin 10                                     \
            --margin-between-b-and-m 2                      \
            --kld-scale ${KLD_SCALE}                        \
            --is-enc-kld-custom-mid 1                       \
            --centroid-type ${CENTROID_TYPE}                          \
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
            --al_optimizer ${AL_OPT}                        \
            --warm_learning_rate ${WLR}                     \
            --al_epochs ${WE}                               \
            --xent-lambda 100                               \
            --display-interval 180                          \
            --al                                            \
            --count ${CNT}                                  \
            --local_pseudo_loss                             \
            --reduce "none"                                 \
            --sample_reduce 'mean'                          \
            --result experiments/020_revision/${RESULT_DIR}/${ENCODER}_androzoo_${CENTROID_TYPE}_active_lr${LR}_${OPT}_${SCH}_${DECAY}_e${E}_${AL_OPT}_wlr${WLR}_we${WE}_test_${TEST_START}_${TEST_END}_cnt${CNT}${CSV_NAME}.csv \
            --log_path experiments/020_revision/${RESULT_DIR}/${ENCODER}_androzoo_${CENTROID_TYPE}_active_lr${LR}_${OPT}_${SCH}_${DECAY}_e${E}_${AL_OPT}_wlr${WLR}_we${WE}_test_${TEST_START}_${TEST_END}_cnt${CNT}_${TS}.log \
            >> experiments/020_revision/${RESULT_DIR}/${ENCODER}_androzoo_${CENTROID_TYPE}_active_lr${LR}_${OPT}_${SCH}_${DECAY}_e${E}_${AL_OPT}_wlr${WLR}_we${WE}_test_${TEST_START}_${TEST_END}_cnt${CNT}_${TS}.log 2>&1 &

wait