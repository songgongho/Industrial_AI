# 다변량 시계열 데이터를 이용한 공기질 예측 모델 개발

대기 오염 측정소에서 수집된 여러 환경 변수(오염물질 농도, 기상 데이터 등)를 동시에 고려하여, 특정 오염물질의 미래 농도를 예측하는 다변량 시계열 분석 프로젝트입니다.

<br>

## 📂 프로젝트 개요

* **목표**: 다변량 시계열 데이터의 복잡한 상관관계를 학습하여 단일 변수 모델보다 더 정확한 공기질 예측 모델을 구축하고, 딥러닝 모델(LSTM, GRU)의 성능을 비교 분석합니다.
* **데이터셋**: [UCI Air Quality Dataset](https://archive.ics.uci.edu/dataset/360/air+quality)
* **핵심 변수**: 여러 오염물질(`CO`, `NMHC`, `NOx` 등)과 환경 변수(`T`, `RH`)를 입력으로 사용하여 특정 오염물질의 농도를 예측합니다.

<br>

## 🛠️ 개발 환경

* **언어**: Python
* **핵심 라이브러리**: TensorFlow, Keras, Pandas, NumPy, Scikit-learn, Matplotlib

<br>

## ⚙️ 실행 방법

1.  이 레퍼지토리를 로컬 환경에 복제(Clone)합니다.
2.  필요한 라이브러리를 설치합니다.
    ```bash
    pip install tensorflow pandas numpy scikit-learn matplotlib
    ```
3.  메인 파이썬 스크립트를 실행합니다. (예: `air_quality_main.py`)
    ```bash
    python air_quality_main.py
    ```

<br>

## 📖 주요 구현 내용

#### 1. 데이터 전처리
* 날짜와 시간 컬럼을 통합하여 Datetime 인덱스를 생성합니다.
* 결측치를 이전 시점의 값으로 채우고(forward fill), 다변량 데이터셋을 구축합니다.
* Min-Max 정규화를 통해 모든 변수의 스케일을 0과 1 사이로 조정합니다.

#### 2. 다변량 시계열 시퀀스 생성
* 과거 24시간 동안의 모든 변수 데이터를 기반으로 다음 시간의 특정 오염물질 농도를 예측하는 시퀀스 데이터(X, y)를 생성합니다.

#### 3. 모델 아키텍처
TensorFlow/Keras를 사용하여 2가지 딥러닝 모델 구조를 정의하고 구현했습니다.
* **LSTM (Long Short-Term Memory)**
* **GRU (Gated Recurrent Unit)**

<br>

## 📊 성능 비교 결과

각 모델의 예측 성능을 손실 함수(MSE) 값으로 비교한 결과는 다음과 같습니다.

| 모델 (Model) | Validation Loss (MSE) |
| :--- | :--- |
| **LSTM** | **0.00063 ✅** |
| GRU | 0.00067 |

*두 모델 모두 낮은 손실 값을 보였으나, **LSTM** 모델이 미세하게 더 나은 성능을 기록하여 최적 모델로 선정되었습니다.*

#### 결과 시각화 (LSTM)

가장 우수한 성능을 보인 LSTM 모델의 실제값과 예측값 비교 그래프입니다.

![LSTM 예측 결과](https://github.com/songgongho/Industrial_AI/assets/174919318/1e17e3cd-f8d1-4203-aa6a-f0f8d18b2098)


<br>

## 💡 결론

* 다변량 시계열 데이터 분석에서 LSTM과 GRU는 모두 효과적인 성능을 보였습니다.
* LSTM은 복잡한 장기 의존성 패턴을 학습하는 데 약간의 강점을 보여, 여러 환경 변수가 복합적으로 작용하는 공기질 예측 문제에 더 적합할 수 있음을 확인했습니다.
