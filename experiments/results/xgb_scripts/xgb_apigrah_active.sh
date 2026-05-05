
DEPTH=10
ROUND=60
ETA=0.3

DATA=apigraph

TRAIN_START=2012-01
TRAIN_END=2012-12
TEST_START=2013-01
TEST_END=2018-12
VALID_DATE=2013-06

RESULT_DIR=xgb_results

CLASSIFIER='xgb'

CSV_NAME='a_xgb_6'

CNT=800

SLP=0

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
            --max_depth ${DEPTH}                            \
            --num_round ${ROUND}                            \
            --eta ${ETA}                                    \
            --cls-retrain 1                                 \
            --al                                            \
            --unc                                           \
            --cold-start                                    \
            --count ${CNT}                                  \
            --result experiments/results/${RESULT_DIR}/${CLASSIFIER}_${DEPTH}_${ROUND}_${ETA}_${DATA}_active_e${E}_test_${TEST_START}_${TEST_END}_cnt${CNT}_${CSV_NAME}.csv \
            --log_path experiments/results/${RESULT_DIR}/${CLASSIFIER}_${DEPTH}_${ROUND}_${ETA}_${DATA}_active_e${E}_test_${TEST_START}_${TEST_END}_cnt${CNT}_${TS}.log \
            >> experiments/results/${RESULT_DIR}/${CLASSIFIER}_${DEPTH}_${ROUND}_${ETA}_${DATA}_active_e${E}_test_${TEST_START}_${TEST_END}_cnt${CNT}_${TS}.log 2>&1 &