# 국방 AI 경진대회 코드 사용법
- 사이버전사22 팀, 김영준, 백두현, 신성욱, 이지성
- 닉네임 : acorn421, dudu, 신성욱신성욱, Irony


# 핵심 파일 설명
  - 데이터 전처리 스크립트: `./data_pre.sh` (두현)
  - 학습 데이터 경로: `/workspace/data/01_data/train` (두현)
  - Network 초기 값으로 사용한 공개된 Pretrained 파라미터: `./LaMa_models/big-lama-with-discr/models/best.ckpt` (영준)
  - 공개 Pretrained 모델 기반으로 추가 Fine Tuning 학습을 한 모델 6개
    - `./mymodel/models/last_v7.ckpt` (성욱)
    - `./mymodel/models/last_v10.ckpt` (성욱)
    - `./mymodel/models/last_v11.ckpt` (성욱)
  - 학습 실행 스크립트: `./train.sh` (성욱, 영준)
  - 학습 메인 코드: `./train.py` (영준)
  - 테스트 실행 스크립트: `./predict.sh` (성욱)
  - 테스트 메인 코드: `./predict.py` (성욱)
  - 테스트 이미지, 마스크 경로: `/workspace/data/01_data/test` (두현)
  - 테스트 결과 이미지 경로: `./final_result/output_aipg` (성욱)

## 코드 구조 설명
- 데이터 생성 부분 (두현)
  - data_processing.py
  - split_image 함수 : 데이터셋 항공 이미지를 전/후 이미지로 분리
  - merge_mask 함수 : 모델 학습을 위해 데이터셋 mask 이미지를 하나의 이미지로 병합. 만약, 2번 label와 1,3번 label가 겹칠 경우 1,3번 label로 우선되게 설정
  - vis_mask 함수 : mask 이미지를 시각적으로 보기 좋게 변환
  - split_mask 함수 : 생성한 mask 이미지를 원래 mask 형식으로 분리
  - vis_result 함수 : 생성한 mask 이미지를 wandb에 업로드하여 항공 이미지와 겹쳐서 보이도록 변환
  - 
- train 모델 부분 (영준)
  - 
- 앙상블 부분 (성욱)
  - 

- **최종 제출 파일 : submitted.zip**
- **학습된 가중치 파일 : training_results/submitted_model/iter_10000.pth**

## 주요 설치 library
- requirements.txt 참고

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
    - /workspace/data/01_data/train  # 학습 데이터 절대경로
    - /workspace/data/01_data/test   # 테스트 데이터 절대경로

  - 데이터 전처리 스크립트 실행
    ```bash
    - ./data_pre.sh
    ```

  - 데이터 전처리 스크립트 내용
    ```bash
    #!/bin/bash
    
    python /workspace/change_detection.pytorch/data_processing.py split-image /workspace/data/01_data/train/x
    python /workspace/change_detection.pytorch/data_processing.py split-image /workspace/data/01_data/test/x
    python /workspace/change_detection.pytorch/data_processing.py merge-mask /workspace/data/01_data/train/y mask
    python /workspace/change_detection.pytorch/data_processing.py merge-mask /workspace/data/01_data/test/y mask
    ```
    

# 학습 실행 방법 (영준 성욱)
  - 학습 스크립트 실행
    ```bash
    ./train.sh
    ```
    
  - 학습 스크립트 내용
    ```bash
    # 11_14-02_22_45 (10) 
    # 11_14-09_47_28
    # 11_14-13_00_20 (4) 
    # 11_14-15_09_51 (3) 

    # 11_14-02_14_48
    # 11_14-09_37_15 [10 dead] (14, 28) 

    # 11_13-11_11_19 (last)

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


# 테스트 실행 방법 (성욱)

  - 테스트 스크립트 실행
    ```bash
    ./predict.sh
    ```

  - 테스트 스크립트 내용
    ```bash
    #!/bin/bash

    # 가상환경 활성화
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

    # 생성된 결과를 후처리 진행
    python data_processing.py split-mask ./infer_res/output1/ ./infer_res/output1_split
    python data_processing.py split-mask ./infer_res/output2/ ./infer_res/output2_split
    python data_processing.py split-mask ./infer_res/output3/ ./infer_res/output3_split
    python data_processing.py split-mask ./infer_res/output4/ ./infer_res/output4_split
    python data_processing.py split-mask ./infer_res/output5/ ./infer_res/output5_split
    python data_processing.py split-mask ./infer_res/output5/ ./infer_res/output6_split

    # 상기의 6가지 추론 결과를 Pixel-wise Averaging 처리하여 최종 detection 결과 생성
    python predict_ensemble.py
    ```
