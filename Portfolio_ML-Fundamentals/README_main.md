# ML Learning Portfolio 🚀

기계학습의 기초부터 심화까지 학습하며 구현한 프로젝트 포트폴리오입니다. 선형 회귀, 로지스틱 회귀, 신경망 등 다양한 알고리즘을 직접 구현하고 비교 분석한 결과물입니다.

## 📚 포트폴리오 개요

이 저장소는 머신러닝의 핵심 알고리즘들을 단계적으로 학습하고 실제 데이터에 적용해본 결과를 담고 있습니다.

| 주제 | 주요 내용 | 파일 |
|------|---------|------|
| **선형 회귀** | 단순/다중 회귀, 특성 공학 | `01_linear_regression/` |
| **로지스틱 회귀** | 이진 분류, 최적화 알고리즘 비교 | `02_logistic_regression/` |
| **신경망** | 다층 신경망, MNIST 분류 | `03_neural_networks/` |

---

## 🎯 주요 프로젝트

### 1️⃣ 선형 회귀 (Linear Regression)

#### 1-1. 기본 선형 회귀
- **목표**: 선형 회귀의 기초 이해
- **구성**: 데이터 생성 → 모델 훈련 → 성능 평가
- **파일**: `01_linear_regression/simple_linear_regression.ipynb`

#### 1-2. 캘리포니아 주택 가격 예측 (1개 특성)
- **데이터셋**: California Housing Dataset
- **특성**: 1개 특성 사용 (실내 중앙값)
- **성능**: MSE, MAE, R² Score 평가
- **파일**: `01_linear_regression/california_housing_feat1.ipynb`

#### 1-3. 캘리포니아 주택 가격 예측 (8개 특성)
- **데이터셋**: California Housing Dataset
- **특성**: 8개 특성 (위치, 방 개수, 인구 등)
- **특성 공학**: 정규화, 스케일링 적용
- **성능**: 다중 회귀를 통한 성능 향상 검증
- **파일**: `01_linear_regression/california_housing_feat8.ipynb`

---

### 2️⃣ 로지스틱 회귀 (Logistic Regression)

#### 2-1. 기본 로지스틱 회귀
- **목표**: 이진 분류 문제 해결
- **알고리즘**: 로지스틱 회귀 (Sigmoid 활성화함수)
- **최적화**: SGD (확률적 경사 하강법)
- **파일**: `02_logistic_regression/logistic_regression_basic.ipynb`

#### 2-2. 경사 하강법 비교 분석
- **비교 대상**: 3가지 경사 하강법 구현
  - **Batch GD**: 전체 데이터로 한 번에 업데이트
  - **SGD (Stochastic GD)**: 샘플 1개씩 업데이트
  - **Mini-batch GD**: 배치 크기 32로 업데이트

- **데이터셋**: Make-moons (2D 분류 문제)
- **평가 메트릭**: Loss 곡선, 수렴 속도, 정확도
- **주요 발견**: 
  - Batch GD: 안정적이지만 느린 수렴
  - SGD: 빠르지만 진동이 심함
  - Mini-batch GD: 안정성과 수렴 속도의 균형

- **파일**: `02_logistic_regression/logistic_regression_gd_comparison.ipynb`
- **성능 비교**:
  ```
  Batch GD 정확도:        84.5%
  Stochastic GD 정확도:   86.5%
  Mini-batch GD 정확도:   85.0%
  ```

---

### 3️⃣ 신경망 (Neural Networks)

#### MNIST 손글씨 분류
- **데이터셋**: MNIST (70,000개 샘플, 28x28 픽셀)
- **아키텍처**: 다층 신경망
  - 입력층: 784개 뉴런 (28×28)
  - 은닉층: ReLU 활성화함수
  - 출력층: Softmax (10개 클래스)

- **하이퍼파라미터**:
  - Batch size: 32
  - Epochs: 20
  - Learning rate: 0.01
  - Optimizer: Adam

- **성능**: 99%+ 정확도 달성
- **파일**: `03_neural_networks/mnist_nn_classification.ipynb`

---

## 📊 성능 비교 요약

### 모델별 성능 메트릭

| 모델 | 데이터셋 | 정확도 | MSE/Loss | 주요 특징 |
|------|---------|--------|---------|---------|
| 선형 회귀 (1 feat) | California Housing | - | MSE: 1.33 | 기본 모델 |
| 선형 회귀 (8 feat) | California Housing | - | MSE: 0.73 | 특성 증가 → 성능 향상 |
| Logistic Regression | Make-moons | 85.0% | 0.35 | 안정적 수렴 |
| Neural Network | MNIST | 99%+ | 0.01 | 고정확 분류 |

---

## 🛠 설치 및 실행

### 필수 요구사항
```bash
python >= 3.8
```

### 패키지 설치
```bash
# requirements.txt로 설치
pip install -r requirements.txt

# 또는 개별 설치
pip install tensorflow scikit-learn numpy pandas matplotlib
```

### Jupyter Notebook 실행
```bash
jupyter notebook

# 원하는 폴더의 .ipynb 파일 실행
# 예: 01_linear_regression/simple_linear_regression.ipynb
```

---

## 📂 파일 구조

```
ML-Learning-Portfolio/
├── README.md                          # 메인 설명 문서
├── requirements.txt                   # 패키지 의존성
│
├── 01_linear_regression/              # 선형 회귀
│   ├── README.md
│   ├── simple_linear_regression.ipynb
│   ├── california_housing_feat1.ipynb
│   └── california_housing_feat8.ipynb
│
├── 02_logistic_regression/            # 로지스틱 회귀
│   ├── README.md
│   ├── logistic_regression_basic.ipynb
│   └── logistic_regression_gd_comparison.ipynb
│
├── 03_neural_networks/                # 신경망
│   ├── README.md
│   └── mnist_nn_classification.ipynb
│
└── docs/                              # 상세 문서
    ├── RESULTS.md                     # 성능 결과 분석
    └── MODEL_COMPARISON.md            # 모델 상세 비교
```

---

## 🔍 각 섹션 상세 설명

### 1️⃣ Linear Regression 폴더

**주요 학습 포인트**:
- 선형 회귀의 수학적 기초 (Normal Equation)
- 단순 vs 다중 회귀의 차이
- 특성 공학 (Feature Engineering)의 중요성
- 회귀 평가 메트릭 (MSE, MAE, R²)

**기술 스택**:
- scikit-learn
- NumPy, Pandas
- Matplotlib, Seaborn

---

### 2️⃣ Logistic Regression 폴더

**주요 학습 포인트**:
- 분류 문제의 개념
- Sigmoid 활성화함수
- 3가지 경사 하강법의 비교 분석:
  - Batch Gradient Descent (배치 GD)
  - Stochastic Gradient Descent (확률적 GD)
  - Mini-batch Gradient Descent (미니배치 GD)
- 수렴 속도와 안정성의 트레이드오프

**실험 결과**:
- Loss 곡선 비교를 통한 수렴 특성 분석
- 각 방법의 장단점 정량화

---

### 3️⃣ Neural Networks 폴더

**주요 학습 포인트**:
- 신경망의 기본 구조 (입력층, 은닉층, 출력층)
- 활성화함수 (ReLU, Softmax)
- 역전파(Backpropagation) 알고리즘
- TensorFlow/Keras 사용법
- 하이퍼파라미터 조정

**성과**:
- MNIST 데이터셋에서 99%+ 정확도 달성

---

## 📈 학습 경로

```
기초 수학
    ↓
선형 회귀 (회귀 문제 이해)
    ↓
로지스틱 회귀 (분류 문제 이해)
    ↓
최적화 알고리즘 (경사 하강법 비교)
    ↓
신경망 (복잡한 패턴 학습)
    ↓
고급 주제 (CNN, RNN 등)
```

---

## 💡 주요 인사이트

1. **특성의 중요성**: 특성 개수가 증가하면 모델 성능이 향상됨 (feat1 vs feat8)

2. **최적화 알고리즘 선택**: 
   - 배치 크기가 작을수록 수렴 과정이 진동
   - 미니배치는 안정성과 속도의 균형 제공

3. **신경망의 강력함**: 
   - 복잡한 비선형 패턴을 자동으로 학습
   - MNIST에서 99%+ 정확도 달성

---

## 📝 결과 분석 문서

더 자세한 성능 분석과 비교는 다음을 참고하세요:
- [`docs/RESULTS.md`](./docs/RESULTS.md) - 전체 결과 요약
- [`docs/MODEL_COMPARISON.md`](./docs/MODEL_COMPARISON.md) - 상세 모델 분석

---

## 🎓 학습 출처 및 참고

- TensorFlow Official Documentation
- scikit-learn Documentation
- 머신러닝 온라인 강의 자료
- 개인 구현 및 실험

---

## 🚀 향후 계획

- [ ] CNN (Convolutional Neural Networks) 구현
- [ ] RNN (Recurrent Neural Networks) 구현
- [ ] 시계열 예측 모델
- [ ] 앙상블 방법 (Random Forest, Gradient Boosting)
- [ ] Hyperparameter Tuning 자동화
- [ ] 모델 해석가능성(Explainability) 분석

---

## 👨‍💻 작성자

**이름**: Song Gong Ho  
**학번**: 2025254010  
**진행 기간**: 2025년 하반기

---

## 📧 문의 및 피드백

이슈 또는 개선 사항이 있으시면 GitHub Issues를 통해 연락주세요.

---

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 공개됩니다.

---

**마지막 업데이트**: 2025년 12월 28일

