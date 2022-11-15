#!/bin/bash

python /workspace/change_detection.pytorch/data_processing.py split-image /workspace/data/01_data/train/x
python /workspace/change_detection.pytorch/data_processing.py split-image /workspace/data/01_data/test/x
python /workspace/change_detection.pytorch/data_processing.py merge-mask /workspace/data/01_data/train/y mask
python /workspace/change_detection.pytorch/data_processing.py merge-mask /workspace/data/01_data/test/y mask