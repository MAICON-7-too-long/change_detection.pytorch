#!/bin/bash
    
# 가상환경 활성화
conda activate maicon

# 코드가 있는 디렉토리로 이동
cd $CDP_DIR

# train 및 test 데이터셋 전처리
python $CDP_DIR/data_processing.py split-image $DATA_DIR/train/x
python $CDP_DIR/data_processing.py split-image $DATA_DIR/test/x
python $CDP_DIR/data_processing.py merge-mask $DATA_DIR/train/y $DATA_DIR/train/mask