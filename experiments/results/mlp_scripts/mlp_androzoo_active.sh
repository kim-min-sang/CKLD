
LR=0.0001
E=25
DATA=androzoo
TRAIN_START=2019-01
TRAIN_END=2019-12
TEST_START=2020-01
TEST_END=2021-12
VALID_DATE=2020-06
RESULT_DIR=mlp_results

CLASSIFIER='mlp'

CSV_NAME="a_mlp_155"
CNT=800
SLP=900

TS=$(date "+%m.%d-%H.%M.%S")

nohup python -u relabel.py	                                \
            --sleep ${SLP}                                  \
            --is-offline 0                                  \
            --data ${DATA}                                  \
            --train_start ${TRAIN_START}                    \
            --train_end ${TRAIN_END}                        \
            --test_start ${TEST_START}                      \
            --test_end ${TEST_END}                          \
            --valid_date ${VALID_DATE}                      \
            --classifier mlp                                \
            --cls-retrain 1                                 \
            --mlp-hidden 100-100                            \
            --mlp-dropout 0.2                               \
            --mlp-batch-size 32                             \
            --mlp-lr ${LR}                                  \
            --mlp-epochs ${E}                               \
            --al                                            \
            --unc                                           \
            --count ${CNT}                                  \
            --cold-start                                    \
            --result experiments/results/${RESULT_DIR}/${CLASSIFIER}_${DATA}_active_e${E}_test_${TEST_START}_${TEST_END}_cnt${CNT}_${CSV_NAME}.csv \
            --log_path experiments/results/${RESULT_DIR}/${CLASSIFIER}_${DATA}_active_e${E}_test_${TEST_START}_${TEST_END}_cnt${CNT}_${TS}.log \
            >> experiments/results/${RESULT_DIR}/${CLASSIFIER}_${DATA}_active_e${E}_test_${TEST_START}_${TEST_END}_cnt${CNT}_${TS}.log 2>&1 &

wait
