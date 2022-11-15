# 국방 AI 경진대회 코드 사용법
- 사이버전사22 팀, 김영준, 백두현, 신성욱, 이지성
- 닉네임 : acorn421, dudu, 신성욱신성욱, Irony


# 핵심 파일 설명
  - 학습 데이터 경로 : `/workspace/data/01_data/train`
  - 테스트 데이터 경로 : `/workspace/data/01_data/test`

  - 프로젝트 경로 : `/workspace/Final_Submission`

  - 데이터 전처리 스크립트 : `data_pre.sh`
  - 데이터 전후처리 메인 코드 : `data_processing.py`

  - Network 초기 값으로 사용한 공개된 Pretrained 파라미터:
    `./LaMa_models/big-lama-with-discr/models/best.ckpt`
  - 공개 Pretrained 모델 기반으로 Fine Tuning 학습을 한 모델 6개 : 
    - `./checkpoints/model1.pth`
    - `./checkpoints/model2.pth`
    - `./checkpoints/model3.pth`
    - `./checkpoints/model4.pth`
    - `./checkpoints/model5.pth`
    - `./checkpoints/model6.pth`

  - 학습 실행 스크립트: `train.sh`
  - 학습 메인 코드: `train.py`

  - 테스트 실행 스크립트: `predict.sh`
  - 테스트 메인 코드: `predict.py`


  - 테스트 결과 이미지 경로: `./infer_res/`
  - 최종 테스트 결과 이미지 경로 `./infer_res/final_mask`

  

## 코드 구조 설명

```
project
│   README.md                       : 프로젝트 설명 파일
│   file001.txt                     
│
└───folder1
│   │   file011.txt
│   │   file012.txt
│   │
│   └───subfolder1
│       │   file111.txt
│       │   file112.txt
│       │   ...
│   
└───folder2
    │   file021.txt
    │   file022.txt
```

## 코드 상세 설명

### 데이터 전처리
  - data_processing.py
    - split_image 함수 : 데이터셋 항공 이미지를 전/후 이미지로 분리
    - merge_mask 함수 : 모델 학습을 위해 데이터셋 mask 이미지를 하나의 이미지로 병합. 만약, 2번 label와 1,3번 label가 겹칠 경우 1,3번 label로 우선되게 설정
    - vis_mask 함수 : mask 이미지를 시각적으로 보기 좋게 변환
    - split_mask 함수 : 생성한 mask 이미지를 원래 mask 형식으로 분리
    - vis_result 함수 : 생성한 mask 이미지를 wandb에 업로드하여 항공 이미지와 겹쳐서 보이도록 변환

### 모델 학습
  - train.py

### 모델 추론
  - predict.py

### 모델 앙상블
  - 

### **최종 제출 파일 : submitted.zip**
### **학습된 가중치 파일 : training_results/submitted_model/iter_10000.pth**

## 주요 설치 library
- requirements.txt 참고

# 실행 환경 설정 방법
  - 소스 코드 및 conda 환경 설치
    ```bash
    cd /workspace
    unzip code.zip -d Final_Submission  # 코드 압축 해제

    echo "export CDP_DIR=/workspace/Final_Submission" >> ~/.bashrc  # 프로젝트 경로 환경변수 설정
    source ~/.bashrc

    cd $CDP_DIR

    conda env create -n maicon    # 가상환경 생성
    conda activate maicon         # 가상환경 활성화

    pip install -r requirements.txt   # 파이썬 패키지 설치

    wandb login # wandb login
    # 로그인 안내 창에서 다음과 같은 API key 입력 : d811788ed8439e74dd656fa7d663ae56a050a412
    ```

# 데이터 전처리 실행 방법
  - 데이터 경로 설정
    ```bash
    echo "export DATA_DIR=/workspace/data/01_data" >> ~/.bashrc # 데이터 경로 환경변수 설정
    # /workspace/data/01_data/train  : 학습 데이터 절대경로
    # /workspace/data/01_data/test   : 테스트 데이터 절대경로
    ```

  - 데이터 전처리 스크립트 실행
    ```bash
    ./data_pre.sh
    ```

  - 데이터 전처리 스크립트 내용
    ```bash
    #!/bin/bash
    
    # 가상환경 활성화
    conda activate maicon

    # 코드가 있는 디렉토리로 이동
    cd $CDP_DIR

    # train 및 test 데이터셋 전처리
    python $CDP_DIR/data_processing.py split-image $DATA_DIR/train/x
    python $CDP_DIR/data_processing.py split-image $DATA_DIR/test/x
    python $CDP_DIR/data_processing.py merge-mask $DATA_DIR/train/y $DATA_DIR/train/mask
    ```
    

# 모델 학습 실행 방법
  - 모델 학습 스크립트 실행
    ```bash
    ./train.sh
    ```
    
  - 모델 학습 스크립트 내용
    ```bash
    # 11_14-02_22_45 0~14 (11 : 10) model1.config /// mv model_epoch_10 /// model1.pth
    # 11_14-09_47_28 15~44  model2-1.config
    # 11_14-13_00_20 45~63 (5 : 49) model2.config /// model2_epoch_4.pth /// model2.pth
    # 11_14-15_09_51 64~67 (4 : 67) model3.config /// model3_epoch_3.pth // model3.pth

    # 11_14-02_14_48 0~10 model4-1.config
    # 11_14-09_37_15 11~39 (15 : 25, 29 : 39) 
              # model4-and-5.config /// model4-and-5_epoch_14.pth // model4.pth
              # model4-and-5.config /// model4-and-5_epoch_28.pth // model5.pth

    # 11_13-11_11_19 0~59 (60 : 59) model6.config /// model6_epoch_59.pth // model6.pth

    #!/bin/bash

    # 가상환경 활성화
    conda activate maicon

    # 코드가 있는 디렉토리로 이동
    cd /workspace/change_detection.pytorch

    # 첫번째 Model 설정 기반 학습: output1.pth 획득 (영준 필요)
    python train.py ./configs/MAICON_UnetPlusPlus_efficientnet_1.json -o output1

    # 두번째 Model 설정 기반 학습: output2.pth 획득
    python train.py ./configs/MAICON_UnetPlusPlus_efficientnet_2.json -o output2

    # 세번째, 네번째 Model 설정 기반 학습: output3.pth, output4.pth 획득
    python train.py ./configs/MAICON_UnetPlusPlus_efficientnet_3.json -o output3

    # 다섯번째 Model 설정 기반 학습: output5.pth 획득
    python train.py ./configs/MAICON_UnetPlusPlus_efficientnet_4.json -o output4
    python train.py ./configs/MAICON_UnetPlusPlus_efficientnet_5.json -o output5
    python train.py ./configs/MAICON_UnetPlusPlus_efficientnet_6.json -o output6

    # 여섯번째 Model 설정 기반 학습: output6.pth 획득
    python train.py ./configs/MAICON_UnetPlusPlus_efficientnet_7.json -o output7


# 모델 활용(모델 추론, 결과 후처리, 모델 앙상블) 실행 방법

  - 모델 활용 스크립트 실행
    ```bash
    ./predict.sh
    ```

  - 모델 활용 스크립트 내용
    ```bash
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
    ```
