# 딥러닝 모델을 활용한 전력 수요 예측 모델 비교 및 구현

프랑스 가정의 4년간 전력 사용량 데이터를 활용하여, 다양한 딥러닝 시계열 모델(LSTM, GRU, Bi-LSTM, CNN-LSTM)을 구현하고 성능을 비교 분석하는 프로젝트입니다.

<br>

## 📂 프로젝트 개요

* [cite_start]**목표**: 실제 시계열 데이터를 기반으로 다양한 딥러닝 예측 모델을 구축하고, 예측 정확도(MAE, RMSE)와 학습 시간을 기준으로 각 모델의 성능을 정량적으로 평가하여 최적의 모델을 도출합니다. [cite: 65, 67, 68, 69]
* **데이터셋**: [UCI Individual Household Electric Power Consumption Dataset](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
* [cite_start]**핵심 변수**: `Global_active_power` 컬럼을 1일 단위 평균값으로 재구성하여 사용합니다. [cite: 67, 93]

<br>

## 🛠️ 개발 환경

* [cite_start]**언어**: Python 3.9+ [cite: 75]
* [cite_start]**핵심 라이브러리**: PyTorch 2.x, Pandas, NumPy, Scikit-learn, Matplotlib [cite: 75]
* [cite_start]**실행 환경**: Anaconda 또는 Jupyter Notebook 환경 [cite: 79]

<br>

## ⚙️ 실행 방법

1.  이 레퍼지토리를 로컬 환경에 복제(Clone)합니다.
2.  필요한 라이브러리를 설치합니다.
    ```bash
    pip install torch pandas numpy scikit-learn matplotlib
    ```
3.  메인 파이썬 스크립트를 실행합니다. (예: `main.py`)
    ```bash
    python main.py
    ```

<br>

## 📖 주요 구현 내용

#### 1. 데이터 전처리
* [cite_start]1분 단위 데이터를 1일 단위 평균값으로 리샘플링하고, 결측치를 처리합니다. [cite: 92, 93]
* [cite_start]Min-Max 정규화를 통해 데이터 스케일을 0과 1 사이로 조정합니다. [cite: 100]

#### 2. 시계열 시퀀스 생성
* [cite_start]과거 30일간의 데이터를 기반으로 다음 날의 전력 사용량을 예측하는 시퀀스 데이터(X, y)를 생성합니다. [cite: 101]

#### 3. 모델 아키텍처
[cite_start]PyTorch를 사용하여 4가지 딥러닝 모델 구조를 정의하고 구현했습니다. [cite: 109, 140]
* **LSTM (Long Short-Term Memory)**
* **GRU (Gated Recurrent Unit)**
* **Bi-LSTM (Bidirectional LSTM)**
* **CNN-LSTM (CNN + LSTM Hybrid)**

<br>

## 📊 성능 비교 결과

각 모델의 예측 성능(MAE, RMSE) 및 학습 시간을 비교한 결과는 다음과 같습니다.

| 모델 (Model) | MAE | RMSE | 학습 시간 (Time) |
| :--- | :--- | :--- | :--- |
| LSTM | 0.190 | 0.252 | 1.00s |
| GRU | 0.180 | 0.244 | 3.21s |
| **Bi-LSTM** | **0.178** | **0.242** | **1.64s ✅** |
| CNN-LSTM| 0.187 | 0.248 | 0.99s |

[cite_start]*성능 비교 결과, **Bi-LSTM** 모델이 가장 낮은 MAE와 RMSE를 기록하여 최적 모델로 선정되었습니다.* [cite: 130, 151]

#### 결과 시각화 (Bi-LSTM)

가장 우수한 성능을 보인 Bi-LSTM 모델의 실제값과 예측값 비교 그래프입니다.

![BiLSTM 예측 결과](https://github.com/songgongho/Industrial_AI/assets/174919318/25a21074-d2e0-4742-b677-2f15e8b44fd6)


<br>

## 💡 결론

* [cite_start]**Bi-LSTM 모델**이 과거와 미래의 시점 정보를 모두 활용하는 양방향 구조 덕분에 단방향 모델보다 더 정확한 예측 결과를 보였습니다. [cite: 152]
* 이를 통해 복잡한 시계열 데이터 예측 시, 데이터의 양방향 맥락을 모두 고려하는 모델의 중요성을 확인할 수 있었습니다.
