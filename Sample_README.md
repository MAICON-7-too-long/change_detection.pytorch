# 국방 AI 경진대회 코드 사용법
- 사이버전사22 팀, 김영준, 백두현, 신성욱, 이지성
- 닉네임 : acorn421, dudu, 신성욱신성욱, Irony


# 핵심 파일 설명
  - 학습 데이터 경로: `./my_dataset` (두현)
  - Network 초기 값으로 사용한 공개된 Pretrained 파라미터: `./LaMa_models/big-lama-with-discr/models/best.ckpt` (영준)
  - 공개 Pretrained 모델 기반으로 추가 Fine Tuning 학습을 한 파라미터 3개
    - `./mymodel/models/last_v7.ckpt` (성욱)
    - `./mymodel/models/last_v10.ckpt` (성욱)
    - `./mymodel/models/last_v11.ckpt` (성욱)
  - 학습 실행 스크립트: `./train.sh` (성욱, 영준)
  - 학습 메인 코드: `./bin/train.py` (영준)
  - 테스트 실행 스크립트: `./inference.sh` (성욱)
  - 테스트 메인 코드: `./bin/predict.py` (성욱)
  - 테스트 이미지, 마스크 경로: `./Inpainting_Test_Raw_Mask` (성욱)
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

  - 학습 데이터 경로 설정
    - `./configs/training/location/my_dataset.yaml` 내의 경로명을 실제 학습 환경에 맞게 수정
      ```
      data_root_dir: /home/user/detection/my_dataset/  # 학습 데이터 절대경로명
      out_root_dir: /home/user/detection/experiments/  # 학습 결과물 절대경로명
      tb_dir: /home/user/detection/tb_logs/  # 학습 로그 절대경로명
      ```

  - 학습 스크립트 실행
    ```
    ./train.sh
    ```
    
  - 학습 스크립트 내용
    ```
    #!/bin/bash

    export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)

    # 첫번째 Model 설정 기반 학습: last_v7__0daee5c4615df5fc17fb1a2f6733dfc1.ckpt, last_v10__dfcb68d46a9604de3147f9ead394f179.ckpt 획득
    python bin/train.py -cn big-lama-aigp-1 location=my_dataset data.batch_size=5

    # 두번째 Model 설정 기반 학습: last_v11__cdb2dc80b605a5e59d234f2721ff80ea.ckpt 획득
    python bin/train.py -cn big-lama-aigp-2 location=my_dataset data.batch_size=5
    ```

# 테스트 실행 방법 (성욱)

  - 테스트 스크립트 실행
    ```
    ./inference.sh
    ```

  - 테스트 스크립트 내용
    ```
    #!/bin/bash

    export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)

    install -d ./final_result/output_aipg

    python bin/predict.py 


    # 상기의 3가지 추론 결과를 Pixel-wise Averaging 처리하여 최종 detection 결과 생성
    python ensemble_avg.py
    ```
