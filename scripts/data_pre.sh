#!/bin/bash

python ${CDP_DIR}/data_processing.py split-image ${DATA_DIR}/train/x
python ${CDP_DIR}/data_processing.py split-image ${DATA_DIR}/test/x
python ${CDP_DIR}/data_processing.py merge-mask ${DATA_DIR}/train/y mask
python ${CDP_DIR}/data_processing.py merge-mask ${DATA_DIR}/test/y mask