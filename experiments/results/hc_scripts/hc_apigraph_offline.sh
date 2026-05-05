
SCH=step
DECAY=0.95
DATA=apigraph

### Train ###
#TRAIN_START=2013-07
#TRAIN_END=2014-06

TRAIN_START=2015-01
TRAIN_END=2015-12

#TRAIN_START=2016-07
#TRAIN_END=2017-06

### Test ###
#TEST_START=2014-07
#TEST_END=2016-06

TEST_START=2016-01
TEST_END=2017-12

#TEST_START=2017-07
#TEST_END=2018-12

### Valid ###
VALID_DATE=2016-06
RESULT_DIR=A-AUT

modeldim="512-384-256-128"
S='half'
B=1024

###############################################################

OPT=adam
E=200
LR=0.001

# Encoder for contrastive-only (baseline)
#ENCODER='simple-enc-mlp'
#CLASSIFIER='simple-enc-mlp'

# Encoder for LCKLD-only
#ENCODER='enc-kld-custom-mlp-only6'
#CLASSIFIER='enc-kld-custom-mlp-only6'

# Encoder for CKLD
ENCODER='enc-kld-custom-mlp-ensemble6'
CLASSIFIER='enc-kld-custom-mlp-ensemble6'

# Loss for Contrastive-only
#LOSS='hi-dist-xent'

# Loss for LCKLD-only, CKLD
LOSS='hi-dist-kld-custom-xent-ensemble6'

# one of: '', 'bin', 'fam'
CENTROID_TYPE='fam'
# Set the beta (β)
KLD_SCALE=1.0

CSV_NAME="val_23-76"

SLP=0

###############################################################

TS=$(date "+%m.%d-%H.%M.%S")

nohup python -u relabel.py	                                \
            --sleep ${SLP}                                  \
            --retrain-first 1                               \
            --is-offline 1                                  \
            --is-accum-samples-load 0                       \
            --is-accum-samples-save 0                       \
            --margin 10                                     \
            --margin-between-b-and-m 2                      \
            --kld-scale ${KLD_SCALE}                        \
            --is-enc-kld-custom-mid 1                       \
            --centroid-type ${CENTROID_TYPE}                \
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
            --result experiments/results/${RESULT_DIR}/${ENCODER}_${DATA}_${CENTROID_TYPE}_offline_lr${LR}_${OPT}_${SCH}_${DECAY}_e${E}_test_${TEST_START}_${TEST_END}_${CSV_NAME}.csv \
            --log_path experiments/results/${RESULT_DIR}/${ENCODER}_${DATA}_${CENTROID_TYPE}_offline_lr${LR}_${OPT}_${SCH}_${DECAY}_e${E}_test_${TEST_START}_${TEST_END}_${TS}.log \
            >> experiments/results/${RESULT_DIR}/${ENCODER}_${DATA}_${CENTROID_TYPE}_offline_lr${LR}_${OPT}_${SCH}_${DECAY}_e${E}_test_${TEST_START}_${TEST_END}_${TS}.log 2>&1 &

wait