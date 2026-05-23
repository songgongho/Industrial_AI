# KAMP Sound CNN AutoEncoder 이상탐지

- 목표
  - MIMII Dataset 기반 멜 스펙트로그램 입력으로 CNN 오토인코더를 학습해 F1-Score 0.75 이상을 목표로 이상탐지를 수행
  - 정상과 비정상의 재구성 오차 분포를 분리하고 threshold 기반 판정을 적용

- 핵심 개선 포인트
  - Encoder와 Decoder에 Batch Normalization, Dropout 적용
  - latent dimension을 하이퍼파라미터로 조절 가능하게 구성
  - 손실 함수는 MSE와 SSIM을 결합한 `ReconstructionLoss` 사용
  - 평가 시 F1-Score, Confusion Matrix, reconstruction score 분포를 시각화
     - PR curve 기반으로 F1-Score가 최대가 되는 threshold를 자동 탐색
     - `confidence`와 `confidence_band`로 95% 이상 확신 구간과 70~95% 불확실 구간을 분리 관리

- 전처리 기준
  - sampling rate: 16000 Hz
  - mel bins: 128
  - n_fft: 1024
  - hop_length: 512
  - 입력 크기: 128x128
  - min-max normalization: 0~1
  - baseline 단계에서는 augmentation 미사용

## 실행 순서

```bash
python -m pip install -r requirements.txt
```

```bash
python step1_eda.py
python step2_train.py
python step3_evaluate.py
python step4_inference.py
```

## 출력 파일

- `artifacts/cnn_autoencoder.pt`
  - 학습된 모델 checkpoint
- `artifacts/train_history.csv`
  - epoch별 train/val loss
- `artifacts/training_summary.json`
  - threshold와 학습 요약 정보
- `artifacts/test_manifest.csv`
  - 최종 평가용 샘플 목록
- `artifacts/evaluation/evaluation_scores.csv`
  - 샘플별 reconstruction score와 예측 결과
- `artifacts/evaluation/confusion_matrix.png`
  - Confusion Matrix 시각화
- `artifacts/evaluation/score_distribution.png`
  - 정상/비정상 score 분포 시각화
- `artifacts/evaluation/reconstruction_examples/`
  - 입력, 복원 결과, 차이 맵 예시

## 보고서 / PPT 구성 예시

- 1장 문제 정의
  - MIMII valve 음향 이상탐지 목표와 F1-Score 0.75 이상 달성 필요성
- 2장 Baseline 한계 분석
  - 정상과 비정상의 MSE 분포 겹침
  - 단순 재구성 오차만으로는 경계가 불분명함
- 3장 개선 방법론
  - BatchNorm, Dropout, latent dimension 조절
  - MSE + SSIM 결합 손실
  - PR curve 기반 threshold 탐색과 confidence band 관리
- 4장 실험 설정
  - 전처리 파라미터와 데이터 분할
- 5장 결과 비교
  - Baseline vs 개선 모델의 F1, Precision, Recall, Confusion Matrix
- 6장 해석 및 한계
  - 왜 오탐이 줄었는지, 어떤 샘플에서 여전히 실패하는지
- 7장 결론
  - 최종 성능과 향후 개선 방향

## 실습 노트

- 현재 코드는 `data/train/normal`, `data/test/normal`, `data/test/abnormal` 구조를 우선 탐색
- 해당 구조가 없으면 기존 `FAN_sound_OK`, `FAN_sound_error` 폴더를 자동 사용
- `step2_train.py` 실행 후 `step3_evaluate.py`에서 최종 평가 수행
- `step4_inference.py` 결과는 `normal_probability`, `anomaly_probability`, `confidence_band`, `business_action` 기준으로 해석

