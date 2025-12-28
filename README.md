# 📈 Industrial AI Projects Portfolio (AI-Ex 포트폴리오)

안녕하세요! 산업 인공지능 분야의 문제 해결을 목표로 다양한 프로젝트를 진행하고 있는 송공호입니다.  
현재(2025.12.기준) 충북대학교 산업인공지능학과에 재학중입니다. 

Industrial_AI 레퍼지토리는 관련 포트폴리오를 정리하여 업로드 합니다.

이 레퍼지토리에서는 제 기술 스택과 프로젝트 경험을 공유합니다.

<br>

## 🛠️ 기술 스택 (Tech Stack)

* **언어 (Languages)**: Python
* **라이브러리 (Libraries)**: Pandas, NumPy, Scikit-learn, TensorFlow, Keras, Matplotlib, Seaborn, Flask
* **딥러닝 프레임워크**: TensorFlow, Keras
* **도구 (Tools)**: Git, GitHub, Jupyter Notebook, Google Colab
* **개발 환경**: Jupyter Lab, VS Code

<br>

---

## 📂 프로젝트 소개 (Projects)

### 1. 전력 수요 예측 모델 개발

전력 사용량 데이터를 분석하고 다양한 시계열 예측 모델을 구현하여 미래 전력 수요를 예측하는 프로젝트입니다.

* **프로젝트 링크**: [**Portfolio_projects_전력 수요 예측 모델 개발**](https://github.com/songgongho/Industrial_AI/tree/main/Portfolio_projects_%EC%A0%84%EB%9E%B5%20%EC%88%98%EC%9A%94%20%EC%98%88%EC%B8%A1%20%EB%AA%A8%EB%8D%B8%20%EA%B0%9C%EB%B0%9C)
* **주요 내용**:
    * LSTM, GRU 등 딥러닝 기반의 시계열 예측 모델 구현
    * 모델별 예측 성능 비교 분석 (MAE, RMSE 등)
    * 데이터 전처리 및 특성 공학을 통한 모델 성능 향상
* **결과 요약**:
    LSTM 모델이 다른 모델에 비해 안정적이고 정확한 예측 성능을 보였습니다.

    **[➡️ 결과 보고서 바로가기](https://github.com/songgongho/Industrial_AI/blob/main/Portfolio_projects_%EC%A0%84%EB%9E%B5%20%EC%88%98%EC%9A%94%20%EC%98%88%EC%B8%A1%20%EB%AA%A8%EB%8D%B8%20%EA%B0%9C%EB%B0%9C/%EC%A0%84%EB%A0%A5%20%EC%88%98%EC%9A%94%20%EC%98%88%EC%B8%A1%20%EB%AA%A8%EB%8D%B8%20%EB%B9%84%EA%B5%90%20%EA%B5%AC%ED%98%84%20%EA%B2%B0%EA%B3%BC%EB%B3%B4%EA%B3%A0%EC%84%9C_%EC%86%A1%EA%B3%B5%ED%98%B8%2020250602%20(%EC%B5%9C%EC%A2%85).pptx)**

<br>

### 2. Flask를 활용한 은행 웹서버 제작

Flask 프레임워크를 사용하여 기본적인 로그인, 입출금, 잔액 확인 기능이 포함된 은행 웹 애플리케이션을 구축한 프로젝트입니다.

* **프로젝트 링크**: [**Portfolio_project_bank**](https://github.com/songgongho/Industrial_AI/tree/main/Portfolio_project_bank)
* **주요 내용**:
    * Flask를 이용한 웹 서버 라우팅 및 세션 관리
    * 사용자 로그인/로그아웃 기능 구현
    * 입금, 출금 기능 및 잔액 부족 등 예외 처리
* **결과 요약**:
    사용자 세션 관리를 통해 안전한 금융 거래를 시뮬레이션하고, 기본적인 웹 서버의 동작 원리를 구현했습니다.

    **[➡️ 결과 보고서 바로가기](https://github.com/songgongho/Industrial_AI/blob/main/Portfolio_project_bank/%EC%9D%80%ED%96%89%EC%9B%B9%EC%84%9C%EB%B2%84%20%EC%A0%9C%EC%9E%91_%EA%B2%B0%EA%B3%BC%EB%B3%B4%EA%B3%A0%EC%84%9C_%EC%86%A1%EA%B3%B5%ED%98%B8.pptx)**

<br>

### 3. 다변량 데이터를 이용한 공기질 예측 모델

여러 환경 변수(온도, 습도 등)를 동시에 고려하여 특정 지역의 공기질(미세먼지 농도 등)을 예측하는 다변량 시계열 분석 프로젝트입니다.

* **프로젝트 링크**: [**Portfolio_projects_공기질예측**](https://github.com/songgongho/Industrial_AI/tree/main/Portfolio_projects_%EA%B3%B5%EA%B8%B0%EC%A7%A8%EC%98%88%EC%B8%A1)
* **주요 내용**:
    * 여러 센서 데이터를 통합한 다변량 시계열 데이터셋 구축
    * 다변량 예측에 적합한 LSTM, GRU 등의 딥러닝 모델 활용
    * 각 환경 변수가 공기질 예측에 미치는 영향 분석
* **결과 요약**:
    다변량 데이터를 함께 학습했을 때 단일 변수 모델보다 예측 정확도가 향상되는 것을 확인했습니다.

    **[➡️ 결과 보고서 바로가기](https://github.com/songgongho/Industrial_AI/blob/main/Portfolio_projects_%EA%B3%B5%EA%B8%B0%EC%A7%A8%EC%98%88%EC%B8%A1/%EA%B3%B5%EA%B8%B0%EC%A7%A8(%EB%8B%A4%EB%B3%80%EB%9F%89)%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EA%B8%B0%EB%B0%98%20%EC%98%88%EC%B8%A1%EB%AA%A8%EB%8D%B8%20%EA%B0%9C%EB%B0%9C%20%EA%B2%B0%EA%B3%BC%EB%B3%B4%EA%B3%A0%EC%84%9C_%EC%86%A1%EA%B3%B5%ED%98%B8%2020250616.pptx)**

<br>

---

## 🚀 신규 프로젝트 (New Projects - 2025년 12월)

### 4. 🧠 TensorFlow 신경망 기초부터 고급 최적화까지

**11개의 실습 노트북으로 구성된 완전한 신경망 학습 포트폴리오**

TensorFlow/Keras를 활용하여 신경망의 기초부터 고급 최적화 기법까지를 체계적으로 학습할 수 있는 포트폴리오입니다.

* **프로젝트 링크**: [**Portfolio-DeepLearning-Fundamentals**](https://github.com/songgongho/Portfolio-DeepLearning-Fundamentals)

* **주요 내용**:
    * **모듈 1 - 회귀 (1개)**: MSE, MAE, Huber Loss 손실함수 비교
    * **모듈 2 - 분류 (2개)**: 이진분류 (Sigmoid), 다중분류 (Softmax) 구현
    * **모듈 3 - 고급 기법 (8개)**:
      - 그래디언트 소실 문제: Sigmoid → ReLU로 +29% 성능 향상
      - 잔차 네트워크: Skip Connection을 통한 깊은 신경망 학습
      - 정규화 (L1/L2): 과적합 감소 23%
      - **클래스 불균합 처리**: F1-Score 87% 달성 (+31% 개선)
      - 과적합 분석: 데이터 크기의 영향 분석
      - Dropout: 과적합 70% 감소
      - 데이터 증강: 다중 기법 조합으로 최적 성능 달성

* **학습 성과**:
    ```
    ✅ 11개 노트북 (총 1,314KB)
    ✅ 예상 학습시간: ~4시간
    ✅ 평균 정확도: 95%+
    ✅ 난이도: 기초(⭐⭐) ~ 중급(⭐⭐⭐)
    ```

* **포트폴리오 특징**:
    - ✨ **완전성**: 신경망 기초부터 고급까지 모두 포함
    - ✨ **실용성**: 실무 문제 중심 학습
    - ✨ **명확성**: 단계별 학습 경로 제시
    - ✨ **재현성**: 모든 결과 재현 가능

* **[➡️ 최종 결과 보고서](https://github.com/songgongho/Portfolio-DeepLearning-Fundamentals/blob/main/FINAL_REPORT.md)**

<br>

### 5. 🎓 머신러닝 기초 이해 (ML Fundamentals)

**머신러닝의 기본 개념부터 실제 구현까지 다루는 교육용 포트폴리오**

머신러닝의 핵심 알고리즘과 개념을 실습을 통해 깊이 있게 학습할 수 있는 프로젝트입니다.

* **프로젝트 링크**: [**Portfolio_ML-Fundamentals**](https://github.com/songgongho/Portfolio_ML-Fundamentals)

* **주요 내용**:
    * 지도학습 (Supervised Learning) 기초
    * 비지도학습 (Unsupervised Learning) 실습
    * 분류, 회귀, 클러스터링 알고리즘 비교
    * 모델 평가 지표 및 최적화 기법
    * 데이터 전처리 및 특성 공학

* **학습 대상**:
    - 머신러닝 입문자
    - 알고리즘의 이론적 배경을 이해하고 싶은 학습자
    - 실제 데이터에 적용하고 싶은 개발자

<br>

### 6. 🚢 타이타닉 생존자 예측 (Titanic Survival Prediction)

**캐글 Titanic 데이터셋을 활용한 분류 문제 해결 포트폴리오**

타이타닉 침몰 사건의 승객 데이터를 분석하여 생존 가능성을 예측하는 실전 머신러닝 프로젝트입니다.

* **프로젝트 링크**: [**Portfolio_Titanic_Survival_Prediction**](https://github.com/songgongho/Portfolio_Titanic_Survival_Prediction)

* **주요 내용**:
    * 타이타닉 승객 데이터 EDA (탐색적 데이터 분석)
    * 결측치 처리 및 특성 공학
    * 다양한 분류 모델 구현 및 비교:
      - Logistic Regression
      - Random Forest
      - Gradient Boosting
      - Neural Networks
    * 모델 앙상블 기법 적용
    * 생존 확률 예측 및 해석

* **성과**:
    - 정확도 (Accuracy): 85%+
    - 여러 모델 비교 분석
    - 특성의 중요도 분석

* **학습 목표**:
    - 실제 데이터셋을 다루는 실무 경험
    - 분류 문제의 전체 과정 이해
    - 모델 비교 및 선택 기준 학습

<br>

### 7. 🚌 청주 버스 이상 탐지 (Cheongju Bus Anomaly Detection)

**버스 운행 데이터를 활용한 실시간 이상 탐지 시스템**

청주시의 버스 운행 데이터를 분석하여 비정상적인 패턴을 탐지하는 프로젝트입니다. 스마트 시티의 교통 관리에 직접 활용될 수 있습니다.

* **프로젝트 링크**: [**Portfolio_Cheongju-bus-anomaly-detection**](https://github.com/songgongho/Portfolio_Cheongju-bus-anomaly-detection)

* **주요 내용**:
    * 공공 API를 활용한 실시간 버스 데이터 수집
    * 시계열 데이터 전처리 및 정규화
    * 이상 탐지 알고리즘 구현:
      - Isolation Forest
      - Local Outlier Factor (LOF)
      - 자동 인코더 (Autoencoder)
    * 이상 지점 시각화 및 분석
    * 실시간 모니터링 시스템 구축

* **실제 적용**:
    - 버스 지연 패턴 감지
    - 비정상적인 정거장 정체 감지
    - 차량 운행 경로 이상 탐지
    - 실시간 알림 시스템

* **기술 스택**:
    - 공공 데이터 포탈 API
    - Pandas & NumPy (데이터 처리)
    - Scikit-learn (이상 탐지)
    - TensorFlow (자동 인코더)
    - Matplotlib & Folium (시각화)

<br>

### 8. 🔍 이상 탐지 포트폴리오 #1 (Anomaly Detection Portfolio)

**다양한 이상 탐지 알고리즘을 비교 분석하는 포트폴리오**

산업 현장에서 자주 사용되는 이상 탐지 기법들을 구현하고 성능을 비교 분석하는 프로젝트입니다.

* **프로젝트 링크**: [**Portfolio_Anomaly Detection_1**](https://github.com/songgongho/Portfolio_Anomaly%20Detection_1)

* **주요 내용**:
    * 이상 탐지의 개념 및 적용 분야
    * 통계 기반 방법 (Z-score, IQR)
    * 머신러닝 기반 방법:
      - Isolation Forest
      - Local Outlier Factor (LOF)
      - One-Class SVM
    * 딥러닝 기반 방법:
      - 자동 인코더 (Autoencoder)
      - LSTM-Autoencoder (시계열)
    * 알고리즘별 성능 비교
    * 하이퍼파라미터 최적화

* **실무 응용 분야**:
    - 제조업: 센서 데이터 이상 감지
    - 금융: 부정 거래 탐지
    - 네트워크: 침입 탐지
    - 스마트팩토리: MES 시스템 이상 감지

* **성과**:
    - 7가지 이상 탐지 기법 구현
    - 성능 벤치마크 완료
    - 각 방법의 장단점 분석

<br>

---

## 📊 포트폴리오 요약 (Portfolio Summary)

| 프로젝트 | 분류 | 난이도 | 주요 기술 | 실무도 |
|---------|------|--------|---------|--------|
| 전력 수요 예측 | 시계열 | ⭐⭐⭐ | LSTM, GRU | ⭐⭐⭐⭐ |
| 은행 웹서버 | 웹개발 | ⭐⭐ | Flask, Python | ⭐⭐ |
| 공기질 예측 | 다변량 시계열 | ⭐⭐⭐ | LSTM, 다변량 | ⭐⭐⭐⭐ |
| **신경망 기초** | 딥러닝 | ⭐⭐ | TensorFlow, Keras | ⭐⭐⭐⭐⭐ |
| **ML 기초** | 머신러닝 | ⭐⭐ | Scikit-learn | ⭐⭐⭐⭐ |
| **타이타닉** | 분류 | ⭐⭐ | Pandas, ML | ⭐⭐⭐ |
| **버스 이상탐지** | 이상탐지 | ⭐⭐⭐ | 공공API, 이상탐지 | ⭐⭐⭐⭐⭐ |
| **이상탐지 #1** | 이상탐지 | ⭐⭐⭐ | IF, LOF, AE | ⭐⭐⭐⭐⭐ |

<br>

---

## 💼 핵심 역량 (Core Competencies)

### 데이터 분석 & 전처리
- ✅ 데이터 수집 및 정제
- ✅ 탐색적 데이터 분석 (EDA)
- ✅ 결측치 처리 및 이상치 제거
- ✅ 특성 공학 (Feature Engineering)

### 머신러닝
- ✅ 지도학습 (분류, 회귀)
- ✅ 비지도학습 (클러스터링, 이상탐지)
- ✅ 앙상블 기법
- ✅ 모델 평가 및 검증

### 딥러닝 & 신경망
- ✅ 신경망 아키텍처 설계
- ✅ CNN, RNN, LSTM 구현
- ✅ 최적화 기법 (Dropout, 정규화, 조기종료)
- ✅ 하이퍼파라미터 튜닝

### 시계열 분석
- ✅ 시계열 데이터 전처리
- ✅ LSTM, GRU 예측 모델
- ✅ 시계열 패턴 분석
- ✅ 미래값 예측

### 이상 탐지
- ✅ 통계 기반 방법
- ✅ 머신러닝 기반 방법 (IF, LOF, SVM)
- ✅ 딥러닝 기반 방법 (Autoencoder, LSTM-AE)
- ✅ 실시간 모니터링 시스템

### 실무 기술
- ✅ 공공 API 활용
- ✅ 웹 프레임워크 (Flask)
- ✅ 버전 관리 (Git/GitHub)
- ✅ Jupyter Notebook 활용

<br>

---

## 📈 학습 경로 추천 (Learning Paths)

### 🟢 초급자 (Beginner)
```
1. 신경망 기초 (Portfolio-DeepLearning-Fundamentals)
   ↓
2. ML 기초 (Portfolio_ML-Fundamentals)
   ↓
3. 타이타닉 예측 (Portfolio_Titanic_Survival_Prediction)
```

### 🟡 중급자 (Intermediate)
```
1. 전력 수요 예측 (시계열 기초)
   ↓
2. 공기질 예측 (다변량 시계열)
   ↓
3. 이상 탐지 #1 (알고리즘 비교)
```

### 🔴 고급자 (Advanced)
```
1. 버스 이상 탐지 (실무 프로젝트)
   ↓
2. 신경망 고급 기법 (Portfolio-DeepLearning-Fundamentals 심화)
   ↓
3. 직접 포트폴리오 확장
```

<br>

---

## 🎯 진행 중인 프로젝트

- 🔄 스마트팩토리 센서 데이터 이상 탐지 고도화
- 🔄 강화학습 (Reinforcement Learning) 적용
- 🔄 Computer Vision 프로젝트 준비
- 🔄 NLP 기초 프로젝트 계획

<br>

---

## 📞 연락처 (Contact)

- **GitHub**: [songgongho](https://github.com/songgongho)
- **Email**: [이메일 주소]

<br>

---

## 📝 라이선스 (License)

이 포트폴리오의 모든 코드는 MIT License를 따릅니다.

---

**마지막 업데이트**: 2025년 12월 28일  
**포트폴리오 상태**: 🟢 활발히 업데이트 중
