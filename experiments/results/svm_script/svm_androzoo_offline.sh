
C=1
DATA=androzoo

TRAIN_START=2019-01
TRAIN_END=2019-12
TEST_START=2020-01
TEST_END=2021-12
VALID_DATE=2021-12

RESULT_DIR=A-AUT

CLASSIFIER='svm'

CSV_NAME="4"

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