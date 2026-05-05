
C=0.1
DATA=apigraph

### Train ###
#TRAIN_START=2012-01
#TRAIN_END=2012-12

#TRAIN_START=2013-07
#TRAIN_END=2014-06

#TRAIN_START=2015-01
#TRAIN_END=2015-12

TRAIN_START=2016-07
TRAIN_END=2017-06

### Test ###
#TEST_START=2013-01
#TEST_END=2014-12

#TEST_START=2014-07
#TEST_END=2016-06

#TEST_START=2016-01
#TEST_END=2017-12

TEST_START=2017-07
TEST_END=2018-12

### Valid ###
VALID_DATE=2018-12

RESULT_DIR=A-AUT

CLASSIFIER='svm'

CSV_NAME="19"

SLP=0

TS=$(date "+%m.%d-%H.%M.%S")
nohup python -u relabel.py	                                \
            --sleep ${SLP}                                  \
            --is-offline 1                                  \
            --data ${DATA}                                  \
            --benign_zero                                   \
            --train_start ${TRAIN_START}                    \
            --train_end ${TRAIN_END}                        \
            --test_start ${TEST_START}                      \
            --test_end ${TEST_END}                          \
            --valid_date ${VALID_DATE}                      \
            --classifier svm                                \
            --svm-c ${C}                                    \
            --cls-retrain 1                                 \
            --al                                            \
            --unc                                           \
            --result experiments/results/${RESULT_DIR}/${CLASSIFIER}_${C}_${DATA}_offline_e${E}_test_${TEST_START}_${TEST_END}_${CSV_NAME}.csv \
            --log_path experiments/results/${RESULT_DIR}/${CLASSIFIER}_${C}_${DATA}_offline_e${E}_test_${TEST_START}_${TEST_END}_${TS}.log \
            >> experiments/results/${RESULT_DIR}/${CLASSIFIER}_${C}_${DATA}_offline_e${E}_test_${TEST_START}_${TEST_END}_${TS}.log 2>&1 &