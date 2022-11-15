#!/bin/bash

# 가상환경 활성화
# conda activate maicon

# 코드가 있는 디렉토리로 이동
cd $CDP_DIR

# 모델 학습 수행
# 모델 1 학습
python $CDP_DIR/train.py $CDP_DIR/configs/model1.json -o model1
mv $CDP_DIR/checkpoints/model1_epoch_10.pth $CDP_DIR/checkpoints/model1.pth

python $CDP_DIR/train.py $CDP_DIR/configs/model2-1.json -l model1_last -o model2-1

# 모델 2 학습
python $CDP_DIR/train.py $CDP_DIR/configs/model2.json -l model2-1_last -o model2
mv $CDP_DIR/checkpoints/model2_epoch_4.pth $CDP_DIR/checkpoints/model2.pth

# 모델 3 학습
python $CDP_DIR/train.py $CDP_DIR/configs/model3.json -l model2_last -o model3
mv $CDP_DIR/checkpoints/model3_epoch_3.pth $CDP_DIR/checkpoints/model3.pth

python $CDP_DIR/train.py $CDP_DIR/configs/model4-1.json -o model4-1

# 모델 4, 5 학습
python $CDP_DIR/train.py $CDP_DIR/configs/model4-and-5.json -l model4-1_last -o model4-and-5
mv $CDP_DIR/checkpoints/model4-and-5_epoch_14.pth $CDP_DIR/checkpoints/model4.pth
mv $CDP_DIR/checkpoints/model4-and-5_epoch_28.pth $CDP_DIR/checkpoints/model5.pth

# 모델 6 학습
python $CDP_DIR/train.py $CDP_DIR/configs/model6.json -o model6
mv $CDP_DIR/checkpoints/model6_last.pth $CDP_DIR/checkpoints/model6.pth