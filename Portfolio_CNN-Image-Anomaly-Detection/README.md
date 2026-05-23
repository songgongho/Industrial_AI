# 🔍 CNN 기반 이미지 이상탐지 (CNN Image Anomaly Detection)

MVTec 데이터셋과 제조 현장 이미지 데이터를 활용하여 CNN(Convolutional Neural Network) 기반의 이미지 이상탐지 시스템을 구현한 프로젝트입니다.

---

## 📌 프로젝트 개요

- **분야**: 컴퓨터 비전 / 이상탐지
- **데이터셋**: MVTec Anomaly Detection Dataset (산업 표면 결함 데이터)
- **모델**: CNN Autoencoder 기반 비지도 이상탐지

---

## 🛠️ 주요 내용

- Step 1 (`step1_data_analysis.py`): MVTec 데이터셋 EDA 및 정상/비정상 이미지 분포 분석
- Step 2 (`step2_train.py`): CNN Autoencoder 학습 (정상 이미지만으로 재구성 오차 학습)
- Step 3 (`step3_evaluation.py`): 재구성 오차 기반 이상 점수 계산 및 임계값 설정
- Step 4 (`step4_inference.py`): 신규 이미지 입력에 대한 실시간 이상 판정

---

## 📂 파일 구조

```
Portfolio_CNN-Image-Anomaly-Detection/
├── src/
│   ├── cnn_anomaly.py          # CNN Autoencoder 모델 정의
│   ├── step1_data_analysis.py  # EDA 및 데이터 분포 분석
│   ├── step2_train.py          # 모델 학습
│   ├── step3_evaluation.py     # 성능 평가 및 임계값 결정
│   └── step4_inference.py      # 추론 및 이상 판정
└── results/
    └── 결과보고서.pptx          # 최종 결과 보고서
```

---

## 🔑 핵심 기술

- **CNN Autoencoder**: 정상 이미지 재구성 학습 → 재구성 오차로 이상 탐지
- **MVTec Dataset**: 15종 산업 표면 결함 벤치마크 데이터셋 활용
- **임계값 기반 판정**: 재구성 손실(MSE)을 기준으로 정상/비정상 분류

---

## 📊 결과 요약

- 정상 이미지와 비정상 이미지 간 재구성 오차 분포가 명확히 분리됨
- CNN Autoencoder가 정상 패턴을 효과적으로 학습하여 결함 탐지 성능 확보
- 산업 현장 적용 가능한 비지도 이상탐지 파이프라인 구현

---

## 📝 사용 기술 스택

- Python, PyTorch, OpenCV
- MVTec Anomaly Detection Dataset
- Matplotlib, NumPy
