#!/bin/bash
TRAINING_DIR_GLOBAL="/opt/local/data/keith_azzopardi/demand_model_training"

# this script is meant to be run from a clone of the git
# repository inside the home directory.

# setting model/training parameters
#MODEL_NAME="5pc_40x10"
#DATASET_PATH="./od_matrix_5pc_40x10_uint8.npy"

MODEL_NAME="testing_tf2_100pc_10x3"
DATASET_PATH="./od_matrix_100pc_10x3.npy"


echo "Training model ${MODEL_NAME} using dataset ${DATASET_PATH}"
# additional configuration parameters
OUTPUT_PATH="${TRAINING_DIR_GLOBAL}/output/${MODEL_NAME}-$(date +'%F-%T')"
DIR_VENV="${TRAINING_DIR_GLOBAL}/training_env"
OVERWRITE_VENV="false"

# if it doesn't exist already, create the python environment in the specified folder
echo "~~ HANDLING PYTHON VENV ~~"
if [[ ! -d "${DIR_VENV}" || "${OVERWRITE_VENV}" == "true" ]]; then
    echo "creating venv..."
    mkdir -p "${DIR_VENV}"
    python3 -m venv "${DIR_VENV}"
    source "${DIR_VENV}/bin/activate"
    pip3 install -r ./requirements.txt
else
    echo "found venv..."
    source "${DIR_VENV}/bin/activate"
fi

echo "~~ STARTING MODEL TRAINING ~~"
mkdir -p "${TRAINING_DIR_GLOBAL}/output"
export HDF5_DISABLE_VERSION_CHECK=1
export PYTHONUNBUFFERED=1
python3 train.py --lr 0.001 \
                --batch_size 16 \
                --seq_len 5 \
                --num_days_test 60 \
                --dataset_path "${DATASET_PATH}" \
                --out_path "${OUTPUT_PATH}"



