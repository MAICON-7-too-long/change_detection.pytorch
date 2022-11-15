# 국방 AI 경진대회 코드 사용법
- 사이버전사22 팀, 김영준, 백두현, 신성욱, 이지성
- 닉네임 : acorn421, dudu, 신성욱신성욱, Irony


# 핵심 파일 설명
  - 학습 데이터 경로: `./my_dataset` (두현)
  - Network 초기 값으로 사용한 공개된 Pretrained 파라미터: `./LaMa_models/big-lama-with-discr/models/best.ckpt` (영준)
  - 공개 Pretrained 모델 기반으로 추가 Fine Tuning 학습을 한 모델 6개
    - `./mymodel/models/last_v7.ckpt` (성욱)
    - `./mymodel/models/last_v10.ckpt` (성욱)
    - `./mymodel/models/last_v11.ckpt` (성욱)
  - 학습 실행 스크립트: `./train.sh` (성욱, 영준)
  - 학습 메인 코드: `./train.py` (영준)
  - 테스트 실행 스크립트: `./predict.sh` (성욱)
  - 테스트 메인 코드: `./predict.py` (성욱)
  - 테스트 이미지, 마스크 경로: `./Inpainting_Test_Raw_Mask` (두현)
  - 테스트 결과 이미지 경로: `./final_result/output_aipg` (성욱)

## 코드 구조 설명
- 데이터 생성 부분 (두현)
  - 
- train 모델 부분 (영준)
  - 
- 앙상블 부분 (성욱)
  - 

- **최종 제출 파일 : submitted.zip**
- **학습된 가중치 파일 : training_results/submitted_model/iter_10000.pth**

## 주요 설치 library
- requirements.txt

# 실행 환경 설정

  - 소스 코드 및 conda 환경 설치 (다 같이)
    ```
    unzip military_code.zip -d military_code
    cd ./military_code/detector

    conda env create -f conda_env.yml
    conda activate myenv
    conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y
    pip install pytorch-lightning==1.2.9
    '''

# 데이터 전처리 과정 (두현)
  - 데이터 경로 설정

  - 데이터 전처리 스크립트 실행

  - 데이터 전처리 스크립트 내용

# 학습 실행 방법 (영준 성욱)
  - 학습 스크립트 실행
    ```
    ./train.sh
    ```
    
  - 학습 스크립트 내용
    ```
    #!/bin/bash

    conda activate maicon

    # 코드가 있는 디렉토리로 이동
    cd /workspace/change_detection.pytorch

    # 첫번째 Model 설정 기반 학습: output1.pth 획득 (영준 필요)
    python train.py ./configs/MAICON_UnetPlusPlus_efficientnet_1.json

    # 두번째 Model 설정 기반 학습: output2.pth 획득
    python train.py ./configs/MAICON_UnetPlusPlus_efficientnet_2.json

    # 세번째, 네번째 Model 설정 기반 학습: output3.pth, output4.pth 획득
    python train.py ./configs/MAICON_UnetPlusPlus_efficientnet_3.json

    # 다섯번째 Model 설정 기반 학습: output5.pth 획득
    python train.py ./configs/MAICON_UnetPlusPlus_efficientnet_4.json
    python train.py ./configs/MAICON_UnetPlusPlus_efficientnet_5.json
    python train.py ./configs/MAICON_UnetPlusPlus_efficientnet_6.json

    # 여섯번째 Model 설정 기반 학습: output6.pth 획득
    python train.py ./configs/MAICON_UnetPlusPlus_efficientnet_7.json


# 테스트 실행 방법 (성욱)

  - 테스트 스크립트 실행
    ```
    ./predict.sh
    ```

  - 테스트 스크립트 내용
    ```
    #!/bin/bash

    conda activate maicon

    # 코드가 있는 디렉토리로 이동
    cd /workspace/change_detection.pytorch

    # 학습된 모델을 활용하여 예측 수행
    python predict.py output1
    python predict.py output2    
    python predict.py output3
    python predict.py output4
    python predict.py output5
    python predict.py output6
    python data_processing.py split-mask ./infer_res/output1/ ./infer_res/output1_split
    python data_processing.py split-mask ./infer_res/output2/ ./infer_res/output2_split
    python data_processing.py split-mask ./infer_res/output3/ ./infer_res/output3_split
    python data_processing.py split-mask ./infer_res/output4/ ./infer_res/output4_split
    python data_processing.py split-mask ./infer_res/output5/ ./infer_res/output5_split
    python data_processing.py split-mask ./infer_res/output5/ ./infer_res/output6_split

    # 상기의 3가지 추론 결과를 Pixel-wise Averaging 처리하여 최종 detection 결과 생성
    python predict_ensemble.py
    ```
