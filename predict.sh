#!/bin/bash

# 가상환경 활성화
conda activate maicon

# 코드가 있는 디렉토리로 이동
cd /workspace/change_detection.pytorch

# 학습된 모델을 활용하여 예측 수행
python $CDP_DIR/predict.py model1
python $CDP_DIR/predict.py model2
python $CDP_DIR/predict.py model3
python $CDP_DIR/predict.py model4
python $CDP_DIR/predict.py model5
python $CDP_DIR/predict.py model6

# 생성된 결과를 후처리 진행
python $CDP_DIR/data_processing.py split-mask $CDP_DIR/infer_res/model1 $CDP_DIR/infer_res/model1_split
python $CDP_DIR/data_processing.py split-mask $CDP_DIR/infer_res/model2 $CDP_DIR/infer_res/model2_split
python $CDP_DIR/data_processing.py split-mask $CDP_DIR/infer_res/model3 $CDP_DIR/infer_res/model3_split
python $CDP_DIR/data_processing.py split-mask $CDP_DIR/infer_res/model4 $CDP_DIR/infer_res/model4_split
python $CDP_DIR/data_processing.py split-mask $CDP_DIR/infer_res/model5 $CDP_DIR/infer_res/model5_split
python $CDP_DIR/data_processing.py split-mask $CDP_DIR/infer_res/model5 $CDP_DIR/infer_res/model6_split

# 추론 결과 수합
mkdir $CDP_DIR/infer_res/submitted_mask
mv $CDP_DIR/infer_res/model1_split $CDP_DIR/infer_res/submitted_mask
mv $CDP_DIR/infer_res/model2_split $CDP_DIR/infer_res/submitted_mask
mv $CDP_DIR/infer_res/model3_split $CDP_DIR/infer_res/submitted_mask
mv $CDP_DIR/infer_res/model4_split $CDP_DIR/infer_res/submitted_mask
mv $CDP_DIR/infer_res/model5_split $CDP_DIR/infer_res/submitted_mask
mv $CDP_DIR/infer_res/model6_split $CDP_DIR/infer_res/submitted_mask

# 상기의 6가지 추론 결과를 Pixel-wise Averaging 처리하여 최종 detection 결과 생성
python $CDP_DIR/predict_ensemble.py $CDP_DIR/infer_res/submitted_mask $CDP_DIR/infer_res/final_mask