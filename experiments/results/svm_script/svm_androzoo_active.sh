
C=0.01
DATA=androzoo

TRAIN_START=2019-01
TRAIN_END=2019-12
TEST_START=2020-01
TEST_END=2021-12
VALID_DATE=2021-12

RESULT_DIR=svm_results

CLASSIFIER='svm'

CSV_NAME="a_18"

CNT=600

SLP=600

TS=$(date "+%m.%d-%H.%M.%S")
nohup python -u relabel.py	                                \
            --sleep ${SLP}                                  \
            --is-offline 0                                  \
            --data ${DATA}                                  \
            --benign_zero                                   \
            --train_start ${TRAIN_START}                    \
            --train_end ${TRAIN_END}                        \
            --test_start ${TEST_START}                      \
            --test_end ${TEST_END}                          \
            --valid_date ${VALID_DATE}                      \
            --classifier ${CLASSIFIER}                      \
            --svm-c ${C}                                    \
            --cls-retrain 1                                 \
            --al                                            \
            --unc                                           \
            --cold-start                                    \
            --count ${CNT}                                  \
            --result experiments/results/${RESULT_DIR}/${CLASSIFIER}_${C}_${DATA}_active_e${E}_test_${TEST_START}_${TEST_END}_cnt${CNT}_${CSV_NAME}.csv \
            --log_path experiments/results/${RESULT_DIR}/${CLASSIFIER}_${C}_${DATA}_active_e${E}_test_${TEST_START}_${TEST_END}_cnt${CNT}_${TS}.log \
            >> experiments/results/${RESULT_DIR}/${CLASSIFIER}_${C}_${DATA}_active_e${E}_test_${TEST_START}_${TEST_END}_cnt${CNT}_${TS}.log 2>&1 &