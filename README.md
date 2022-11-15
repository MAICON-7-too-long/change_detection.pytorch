# 국방 AI 경진대회 코드 사용법
- 사이버전사22 팀, 김영준, 백두현, 신성욱, 이지성
- 닉네임 : acorn421, dudu, 신성욱신성욱, Irony


# 핵심 파일 설명
  - 학습 데이터 경로 : `/workspace/data/01_data/train`
  - 테스트 데이터 경로 : `/workspace/data/01_data/test`

  - 프로젝트 경로 : `/workspace/Final_Submission`

  - 데이터 전처리 스크립트 : `data_pre.sh`
  - 데이터 전후처리 메인 코드 : `data_processing.py`

  - Network 초기 값으로 사용한 공개된 Pretrained 파라미터 :
    - `./pretrained_model/tf_efficientnet_b5_ns-6f26d0cf.pth`
    - https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b5_ns-6f26d0cf.pth
    - `./pretrained_model/tf_efficientnet_b7_ns-1dbc32de.pth`
    - https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b7_ns-1dbc32de.pth

  - 공개 Pretrained 모델 기반으로 Fine Tuning 학습을 한 모델 6개 : 
    - `./checkpoints/model1.pth`
    - `./checkpoints/model2.pth`
    - `./checkpoints/model3.pth`
    - `./checkpoints/model4.pth`
    - `./checkpoints/model5.pth`
    - `./checkpoints/model6.pth`

  - 학습 실행 스크립트 : `train.sh`
  - 학습 메인 코드 : `train.py`

  - 테스트 실행 스크립트 : `predict.sh`
  - 테스트 메인 코드 : `predict.py`


  - 테스트 결과 이미지 경로 : `./infer_res/`
  - 최종 테스트 결과 이미지 경로 : `./infer_res/final_mask`


# 코드 구조 설명

```bash
📦change_detection.pytorch
 ┣ 📂change_detection_pytorch   # 메인 활용 오픈소스 라이브러리
 ┣ 📂checkpoints                # 학습된 모델 폴더
 ┃ ┣ 📜model1.pth
 ┃ ┣ 📜model2.pth
 ┃ ┣ 📜model3.pth
 ┃ ┣ 📜model4.pth
 ┃ ┣ 📜model5.pth
 ┃ ┗ 📜model6.pth
 ┣ 📂configs                    # 모델 학습 설정 파일 폴더
 ┃ ┣ 📜model1.json
 ┃ ┣ 📜model2-1.json
 ┃ ┣ 📜model2.json
 ┃ ┣ 📜model3.json
 ┃ ┣ 📜model4-1.json
 ┃ ┣ 📜model4-and-5.json
 ┃ ┗ 📜model6.json
 ┣ 📂debug_predict              # 디버그용 이미지 폴더
 ┣ 📂infer_res                  # 모델 추론 결과 마스크 폴더
 ┃ ┣ 📂model1
 ┃ ┃ ┣ 📜1000.png
 ┃ ┃ ┣ 📜1001.png
 ┃ ┃ ┣ ...
 ┃ ┃ ┣ 📜3336.png
 ┃ ┃ ┗ 📜3337.png
 ┃ ┣ 📂model2
 ┃ ┣ 📂model3
 ┃ ┣ 📂model4
 ┃ ┣ 📂model5
 ┃ ┣ 📂model6
 ┃ ┣ 📂final_mask               # 제출용 최종 추론 결과 폴더
 ┃ ┣ 📂submitted_mask           # 각 모델별 후처리 마스크 폴더
 ┃ ┃ ┣ 📂model1_split
 ┃ ┃ ┣ 📂model2_split
 ┃ ┃ ┣ 📂model3_split
 ┃ ┃ ┣ 📂model4_split
 ┃ ┃ ┣ 📂model5_split
 ┃ ┗ ┗ 📂model6_split
 ┣ 📂wandb                      # wandb 관련 로그 폴더
 ┣ 📜LICENSE
 ┣ 📜README.md
 ┣ 📜__init__.py
 ┣ 📜data_pre.sh                # 데이터 전처리 스크립트
 ┣ 📜train.sh                   # 모델 학습 스크립트
 ┣ 📜predict.sh                 # 모델 추론 스크립트
 ┣ 📜data_processing.py         # 데이터 처리 메인 코드
 ┣ 📜train.py                   # 모델 학습 메인 코드
 ┣ 📜predict.py                 # 모델 추론 메인 코드
 ┣ 📜predict_ensemble.py        # 모델 앙상블 메인 코드
 ┗ 📜requirements.txt           # 활용 파이썬 패키지 정보
```

# 코드 상세 설명

### 데이터 전처리
  - data_processing.py
    - split_image 함수 : 데이터셋 항공 이미지를 전/후 이미지로 분리
    - merge_mask 함수 : 모델 학습을 위해 데이터셋 mask 이미지를 하나의 이미지로 병합. 만약, 2번 label과 1,3번 label이 겹칠 경우 1,3번 label이 우선되게 설정
    - vis_mask 함수 : mask 이미지를 시각적으로 보기 좋게 변환
    - split_mask 함수 : 생성한 mask 이미지를 원래 mask 형식으로 분리
    - vis_result 함수 : 생성한 mask 이미지를 wandb에 업로드하여 항공 이미지와 겹쳐서 보이도록 변환

### 데이터셋 로더
  - change_detection_pytorch/datasets/MAICON.py
    - maicon 대회 데이터셋 활용 모듈
    - albumentations 패키지를 활용한 augmentation 기능 구현

### 모델 학습
  - train.py
    - 설정 파일을 기반으로 모델을 학습
    - 이미지 세그멘테이션 용으로 Unet++을 백엔드로 사용
    - Unet++의 인코더 네트워크로는 EfficientNet을 주로 활용
    - pretrained 된 weight를 기반으로 fine tuning 진행
    - 원활한 실험 진행을 위해 학습 과정을 wandb 서비스를 활용하여 저장 및 시각화

### 모델 추론
  - predict.py
    - 학습된 단일 모델을 사용하여 테스트 데이터의 마스크 결과를 생성
    - 생성된 마스크 결과의 시각화 기능도 구현

### 모델 앙상블
  - predict_ensemble.py
    - train 및 validation 데이터셋에 대해 성능이 좋은 6개의 모델을 활용하여 앙상블 수행
    - 32기가의 메모리 한계로 인해 성능이 좋은 6개의 모델만 선정
    - 6개의 모델이 예측한 테스트 이미지 데이터를 활용
    - 각 픽셀 마다 6개의 모델이 가장 많이 예측한 값으로 예측 수행

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
    #!/bin/bash

    # 가상환경 활성화
    conda activate maicon

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
    ```


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
