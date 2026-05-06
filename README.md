# CCR: Contrastive-based Centroid Regularization for Concept Drift Robustness in Android Malware Detection
This is the source code accompanying our paper
"CCR: Contrastive-based Centroid Regularization for Concept Drift Robustness in Android Malware Detection",
submitted to the Network and Distributed System Security Symposium (NDSS) 2027.


## Note on Repository Status
This repository is under active development and is not a final, camera-ready release. We will continue to make iterative updates for internal optimization (e.g., code refactoring, configuration cleanup, and reproducibility improvements).

That said, the current version is fully runnable for experiments. The provided scripts/configurations are sufficient to reproduce the main experimental pipeline, and ongoing updates are intended to improve clarity and robustness without breaking core experiment execution.


## Setup
The experiments were tested on Ubuntu 22.04 with an NVIDIA GPU.
To reproduce the results, please install the dependencies as follows:

```bash
conda env create -f environment.yml
conda activate ccr

# For CUDA 11.8
pip install torch==2.0.0 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.4
pip install torch==2.3.1 torchaudio==2.3.1
```

## Datasets
We provide a downloader script to fetch the datasets required to run CCR.
You can download, verify, and optionally extract the preprocessed feature archive with the following command:

```bash
python3 downloader.py --url "https://drive.google.com/file/d/1O0upEcTolGyyvasCPkZFY86FNclk29XO/view" --dst "data/dataset.zip" --sha256 "827f36fc5affd58dd31b6393042d92cbe3bae012290a03b6fcccc6355266384d" --extract
```

After the command finishes, the downloaded and extracted feature files used in our experiments are located under the following directories:

- `data/apigraph` : DREBIN features of the APIGraph dataset (2012–2018)  
- `data/androzoo` : DREBIN features of the AndroZoo dataset (2019–2021)  

**Note:** These processed Drebin feature datasets were obtained from the official repository of  
[Continuous Learning for Android Malware Detection (HC, USENIX Security 2023)](https://github.com/wagner-group/active-learning).  
The original raw datasets are APIGraph and AndroZoo, but we use the preprocessed versions released by HC for reproducibility.

### Generating A-AUT Training Splits
For A-AUT offline learning evaluation on APIGraph, first download and extract the APIGraph dataset. Then, run the following command:

```bash
python3 generate_a_aut_splits.py
```

This script creates temporal training splits for A-AUT based on the downloaded APIGraph feature files.

## Example Offline Learning Setting
We provide shell scripts under the `experiments/results` directory to reproduce our experiments.  
For example, to set **CCR applied on the Triplet baseline** on the APIGraph dataset under offline learning:

```bash
SCH=step
DECAY=0.95
DATA=apigraph
TRAIN_START=2012-01
TRAIN_END=2012-12
TEST_START=2013-01
TEST_END=2018-12
VALID_DATE=2013-06
RESULT_DIR=triplet_results

modeldim="512-384-256-128"
S='triplet'
B=1536

###############################################################

OPT=adam
E=150
LR=0.0005

# Encoder for contrastive-only (baseline)
ENCODER='triplet-mlp'
CLASSIFIER='triplet-mlp'

# Encoder for CR-only
#ENCODER='triplet-kld-only-mlp'
#CLASSIFIER='triplet-kld-only-mlp'

# Encoder for CCR
#ENCODER='triplet-kld-ensemble-mlp'
#CLASSIFIER='triplet-kld-ensemble-mlp'

# Loss for Contrastive-only
LOSS='triplet-xent'

# Loss for CR-only, CCR
#LOSS='triplet-kld-ensemble-xent'

# one of: '', 'bin', 'fam'
CENTROID_TYPE=''
# Set the beta (β)
KLD_SCALE=1.0

CSV_NAME="1"

SLP=0

###############################################################

TS=$(date "+%m.%d-%H.%M.%S")

nohup python -u relabel.py	                                \
            --sleep ${SLP}                                  \
            --unc                                           \
            --retrain-first 1                               \
            --is-offline 1                                  \
            --margin 10                                     \
            --margin-between-b-and-m 2                      \
            --is-enc-kld-custom-mid 1                       \
            --centroid-type ${CENTROID_TYPE}                \
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
            --triplet-lambda 1                              \
            --xent-lambda 100                               \
            --display-interval 180                          \
            --al                                            \
            --reduce "none"                                 \
            --sample_reduce 'mean'                          \
            --result experiments/results/${RESULT_DIR}/${ENCODER}_${DATA}_${CENTROID_TYPE}_offline_lr${LR}_${OPT}_${SCH}_${DECAY}_e${E}_test_${TEST_START}_${TEST_END}_${CSV_NAME}.csv \
            --log_path experiments/results/${RESULT_DIR}/${ENCODER}_${DATA}_${CENTROID_TYPE}_offline_lr${LR}_${OPT}_${SCH}_${DECAY}_e${E}_test_${TEST_START}_${TEST_END}_${TS}.log \
            >> experiments/results/${RESULT_DIR}/${ENCODER}_${DATA}_${CENTROID_TYPE}_offline_lr${LR}_${OPT}_${SCH}_${DECAY}_e${E}_test_${TEST_START}_${TEST_END}_${TS}.log 2>&1 &
wait
```

Since our framework is designed in an **end-to-end manner**, the encoder and classifier must be configured consistently.  
The loss function is also paired with the corresponding encoder/classifier setting.

- **Contrastive-only(Triplet)**
  ```ini
  ENCODER='triplet-mlp'
  CLASSIFIER='triplet-mlp'
  LOSS='triplet-xent'
  ```

- **CR-only(Triplet)**
  ```ini
  ENCODER='triplet-kld-only-mlp'
  CLASSIFIER='triplet-kld-only-mlp'
  LOSS='triplet-kld-ensemble-xent'
  ```

- **CCR(Triplet)**
  ```ini
  ENCODER='triplet-kld-ensemble-mlp'
  CLASSIFIER='triplet-kld-ensemble-mlp'
  LOSS='triplet-kld-ensemble-xent'
  ```

- **Contrastive-only(CADE)**
  ```ini
  ENCODER='cae-mlp'
  CLASSIFIER='cae-mlp'
  LOSS='triplet-mse-xent'
  ```

- **CR-only(CADE)**
  ```ini
  ENCODER='cae-kld-only-mlp'
  CLASSIFIER='cae-kld-only-mlp'
  LOSS='triplet-mse-kld-ensemble-xent'
  ```

- **CCR(CADE)**
  ```ini
  ENCODER='cae-kld-ensemble-mlp'
  CLASSIFIER='cae-kld-ensemble-mlp'
  LOSS='triplet-mse-kld-ensemble-xent'
  ```

- **Contrastive-only(HC)**
  ```ini
  ENCODER='simple-enc-mlp'
  CLASSIFIER='simple-enc-mlp'
  LOSS='hi-dist-xent'
  ```

- **CR-only(HC)**
  ```ini
  ENCODER='enc-kld-custom-mlp-only6'
  CLASSIFIER='enc-kld-custom-mlp-only6'
  LOSS='hi-dist-kld-custom-xent-ensemble6'
  ```

- **CCR(HC)**
  ```ini
  ENCODER='enc-kld-custom-mlp-ensemble6'
  CLASSIFIER='enc-kld-custom-mlp-ensemble6'
  LOSS='hi-dist-kld-custom-xent-ensemble6'
  ```


In addition, we use the following hyperparameters: 

- **E**  
  The number of training epochs.

- **LR**  
  The learning rate.

- **CENTROID_TYPE**  
  Sets the centroid type used in the paper (`bin` for binary label centroids, `fam` for family label centroids).

- **KLD_SCALE (β)**  
  Controls the degree of generalization in the KL divergence term, as described in the paper.

To enable offline learning, set `{--is-offline 1}`.

In the offline learning setting, you can train and test the model in a no-drift (i.e., without concept drift) scenario by setting `{--is-for-no-drift 1}`.


## Example Active Learning Setting

Following the offline learning setup, the additional configurations specific to **active learning** are as follows:

- **CNT**  
  Specifies the annotation budget, representing analyst effort as defined in the paper.

- **WE**  
  The number of epochs used for warm-start retraining.

- **WLR**  
  The learning rate used for warm-start retraining.

  To enable active learning, set `{--is-offline 0}`.


## Running Experiments

After setting the configuration as described above, you can run the experiments using the provided shell scripts.  

For example, to run **CCR applied on the Triplet baseline**:

- Dataset: **APIGraph**, Scenario: **offline**  
  ```bash
  sh experiments/results/triplet_scripts/triplet_apigraph_offline.sh
  ```

- Dataset: **AndroZoo**, Scenario: **offline**  
  ```bash
  sh experiments/results/triplet_scripts/triplet_androzoo_offline.sh
  ```

- Dataset: **APIGraph**, Scenario: **active**  
  ```bash
  sh experiments/results/triplet_scripts/triplet_apigraph_active.sh
  ```

- Dataset: **AndroZoo**, Scenario: **active**  
  ```bash
  sh experiments/results/triplet_scripts/triplet_androzoo_active.sh
  ```

Similarly:  
- For **CADE**, use the scripts in `experiments/results/cade_scripts/`.  
- For **HC**, use the scripts in `experiments/results/hc_scripts/`.  

The usage pattern is the same across Triplet, CADE, and HC.

## Results

We provide the experimental results generated by our framework.  
For the contrastive-based baselines and their CCR variants, the results are organized into separate directories:

- `experiments/results/triplet_results/`
  - `offline/` : Results for offline learning scenario
  - `active/`  : Results for active learning scenario

- `experiments/results/cade_results/`
  - `offline/` : Results for offline learning scenario
  - `active/`  : Results for active learning scenario

- `experiments/results/hc_results/`
  - `offline/` : Results for offline learning scenario
  - `active/`  : Results for active learning scenario

We also provide non-contrastive reference baselines:

- `experiments/results/svm_results/`
- `experiments/results/mlp_results/`
- `experiments/results/xgboost_results/`

These directories contain the CSV files used to reproduce the main quantitative tables reported in the paper.

