# 📈 Industrial AI Projects Portfolio (AI-Ex 포트폴리오)

안녕하세요! 산업 인공지능 분야의 문제 해결을 목표로 다양한 프로젝트를 진행하고 있는 송공호입니다
현재(2026년 04월 기준) 충북대학교 산업인공지능학과에 재학중입니다

Industrial_AI 레퍼지토리는 관련 포트폴리오를 정리하여 업로드 합니다

이 레퍼지토리에서는 제 기술 스택과 프로젝트 경험을 공유합니다

<br>

## 🛠️ 기술 스택 (Tech Stack)

* **언어 (Languages)**: Python, HTML, CSS, JavaScript
* **라이브러리 (Libraries)**: **OpenCV**, Pandas, NumPy, Scikit-learn, PyTorch, TensorFlow, Keras, Matplotlib, Seaborn, Flask, **asyncua**
* **컴퓨터 비전 (Computer Vision)**: **OpenCV, MediaPipe, Azure AI Vision**
* **데이터 엔지니어링 (Data Engineering)**: **OPC UA**
* **딥러닝 프레임워크**: PyTorch, TensorFlow, Keras
* **도구 (Tools)**: Git, GitHub, Jupyter Notebook, Google Colab
* **개발 환경**: Jupyter Lab, VS Code

<br>

***

## 📂 프로젝트 소개 (Projects)

### 1. 전력 수요 예측 모델 개발

전력 사용량 데이터를 분석하고 다양한 시계열 예측 모델을 구현하여 미래 전력 수요를 예측하는 프로젝트입니다

* **프로젝트 링크**: [**Portfolio_projects_전력 수요 예측 모델 개발**](https://github.com/songgongho/Industrial_AI/tree/main/Portfolio_projects_%EC%A0%84%EB%9E%B5%20%EC%88%98%EC%9A%94%20%EC%98%88%EC%B8%A1%20%EB%AA%A8%EB%8D%B8%20%EA%B0%9C%EB%B0%9C)
* **주요 내용**:
    * LSTM, GRU 등 딥러닝 기반의 시계열 예측 모델 구현
    * 모델별 예측 성능 비교 분석 (MAE, RMSE 등)
    * 데이터 전처리 및 특성 공학을 통한 모델 성능 향상
* **결과 요약**:
    * LSTM 모델이 다른 모델에 비해 안정적이고 정확한 예측 성능을 보임

    **[➡️ 결과 보고서 바로가기](https://github.com/songgongho/Industrial_AI/blob/main/Portfolio_projects_%EC%A0%84%EB%9E%B5%20%EC%88%98%EC%9A%94%20%EC%98%88%EC%B8%A1%20%EB%AA%A8%EB%8D%B8%20%EB%B9%84%EA%B5%90%20%EA%B5%AC%ED%98%84%20%EA%B2%B0%EA%B3%BC%EB%B3%B4%EA%B3%A0%EC%84%9C_%EC%86%A1%EA%B3%B5%ED%98%B8%2020250602%20(%EC%B5%9C%EC%A2%85).pptx)**

<br>

### 2. Flask를 활용한 은행 웹서버 제작

Flask 프레임워크를 사용하여 기본적인 로그인, 입출금, 잔액 확인 기능이 포함된 은행 웹 애플리케이션을 구축한 프로젝트입니다

* **프로젝트 링크**: [**Portfolio_project_bank**](https://github.com/songgongho/Industrial_AI/tree/main/Portfolio_project_bank)
* **주요 내용**:
    * Flask를 이용한 웹 서버 라우팅 및 세션 관리
    * 사용자 로그인/로그아웃 기능 구현
    * 입금, 출금 기능 및 잔액 부족 등 예외 처리
* **결과 요약**:
    * 사용자 세션 관리를 통해 안전한 금융 거래를 시뮬레이션하고 웹 서버의 기본 동작 원리를 구현함

    **[➡️ 결과 보고서 바로가기](https://github.com/songgongho/Industrial_AI/blob/main/Portfolio_project_bank/%EC%9D%80%ED%96%89%EC%9B%B9%EC%84%9C%EB%B2%84%20%EC%A0%9C%EC%9E%91_%EA%B2%B0%EA%B3%BC%EB%B3%B4%EA%B3%A0%EC%84%9C_%EC%86%A1%EA%B3%B5%ED%98%B8.pptx)**

<br>

### 3. 다변량 데이터를 이용한 공기질 예측 모델

여러 환경 변수(온도, 습도 등)를 동시에 고려하여 특정 지역의 공기질(미세먼지 농도 등)을 예측하는 다변량 시계열 분석 프로젝트입니다

* **프로젝트 링크**: [**Portfolio_projects_공기질예측**](https://github.com/songgongho/Industrial_AI/tree/main/Portfolio_projects_%EA%B3%B0%EA%B8%B0%EC%A7%A8%EC%98%88%EC%B8%A1)
* **주요 내용**:
    * 여러 센서 데이터를 통합한 다변량 시계열 데이터셋 구축
    * 다변량 예측에 적합한 LSTM, GRU 등의 딥러닝 모델 활용
    * 각 환경 변수가 공기질 예측에 미치는 영향 분석
* **결과 요약**:
    * 다변량 데이터를 함께 학습했을 때 단일 변수 모델보다 예측 정확도가 향상됨을 확인

    **[➡️ 결과 보고서 바로가기](https://github.com/songgongho/Industrial_AI/blob/main/Portfolio_projects_%EA%B3%B0%EA%B8%B0%EC%A7%A8%EC%98%88%EC%B8%A1/%EA%B3%B0%EA%B8%B0%EC%A7%A8(%EB%8B%A4%EB%B3%80%EB%9F%89)%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EA%B8%B0%EB%B0%98%20%EC%98%88%EC%B8%A1%EB%AA%A8%EB%8D%B8%20%EA%B0%9C%EB%B0%9C%20%EA%B2%B0%EA%B3%BC%EB%B3%B4%EA%B3%A0%EC%84%9C_%EC%86%A1%EA%B3%B5%ED%98%B8%2020250616.pptx)**

<br>

***

## 🚀 실무 자동화 프로젝트 (Automation Projects)

### 4. 헬스동호회 정산 및 결과보고서 자동 생성기

**네이버폼 데이터를 활용한 업무 자동화 및 A4 결과보고서 생성 웹 애플리케이션**

매월 발생하는 복잡한 동호회비 수기 정산 과정과 문서 작업 시간을 단축하기 위해 개발한 순수 프론트엔드 기반 자동화 프로젝트입니다

* **프로젝트 링크**: [**Club-Report-Auto-Generator**](https://github.com/songgongho/Industrial_AI/tree/main/Club-Report-Auto-Generator)
* **주요 내용**:
    * 네이버폼 엑셀 데이터 파싱을 통한 개인별 물품/헬스장 구매비 자동 합산
    * 보조비 기준 부족분 발생 시 1,000원 단위 올림 스마트 공동구매 분배 알고리즘 적용
    * 공동구매 내역 확인 및 금액 직접 수정이 가능한 인터랙티브 UI 구현
    * 직관적인 영수증/활동 사진 첨부 및 A4 규격 인쇄 레이아웃 최적화
* **결과 요약**:
    * 수기 계산 오류를 차단하고 서류 작성 시간을 대폭 단축하여 업무 효율성을 증대시킴

    **[➡️ 2월 결과 보고서 예시 바로가기](https://github.com/songgongho/Industrial_AI/blob/main/Portfolio-Club-Report-Auto-Generator.v10/2F%20%EB%A7%88%EB%8F%99%EC%84%9D%20%ED%97%AC%EC%8A%A4%EB%AA%A8%EC%9E%84%20%EB%8F%99%ED%98%B8%ED%9A%8C%EC%A7%80%EC%9B%90%EA%B8%88%EC%8B%A0%EC%B2%AD%EC%84%9C%2C%20%EB%8F%99%ED%98%B8%ED%9A%8C%ED%99%9C%EB%8F%99%EA%B2%B0%EA%B3%BC%EB%B3%B4%EA%B3%A0%EC%84%9C_2%EC%9B%94-%EC%95%95%EC%B6%95%EB%90%A8.pdf)**

<br>

***

## 🚀 심화 분석 프로젝트 (Advanced Projects)

### 5. 🧠 TensorFlow 신경망 기초부터 고급 최적화까지

**11개의 실습 노트북으로 구성된 완전한 신경망 학습 포트폴리오**

* **프로젝트 링크**: [**Portfolio-DeepLearning-Fundamentals**](https://github.com/songgongho/Industrial_AI/tree/main/Portfolio-DeepLearning-Fundamentals)
* **주요 내용**: MSE/MAE 손실함수 비교, 이진/다중 분류 구현, Residual Network 및 정규화 기법 최적화
* **[➡️ 최종 결과 보고서](https://github.com/songgongho/Industrial_AI/blob/main/Portfolio-DeepLearning-Fundamentals/FINAL_REPORT.md)**

<br>

### 6. 🎓 머신러닝 기초 이해 (ML Fundamentals)

* **프로젝트 링크**: [**Portfolio_ML-Fundamentals**](https://github.com/songgongho/Industrial_AI/tree/main/Portfolio_ML-Fundamentals)
* **주요 내용**: 지도/비지도 학습 알고리즘 비교, 모델 평가 지표 및 특성 공학 실습
* **[➡️ 최종 결과 보고서](https://github.com/songgongho/Industrial_AI/blob/main/Portfolio_ML-Fundamentals/docs/RESULTS_REPORT.md)**

<br>

### 7. 🚢 타이타닉 생존자 예측 (Titanic Survival Prediction)

* **프로젝트 링크**: [**Portfolio_Titanic_Survival_Prediction**](https://github.com/songgongho/Industrial_AI/tree/main/Portfolio_Titanic_Survival_Prediction)
* **주요 내용**: EDA를 통한 데이터 인사이트 도출, 앙상블 기법(Random Forest, Gradient Boosting) 적용
* **[➡️ 최종 결과 보고서](https://github.com/songgongho/Industrial_AI/blob/main/Portfolio_Titanic_Survival_Prediction/titanic_report_%EA%B2%B0%EA%B3%BC%EB%B3%B4%EA%B3%A0%EC%84%9C.html)**

<br>

### 8. 🚌 청주 버스 이상 탐지 (Cheongju Bus Anomaly Detection)

* **프로젝트 링크**: [**Portfolio_Cheongju_bus_anomaly_detection_project**](https://github.com/songgongho/Industrial_AI/tree/main/Portfolio_Cheongju_bus_anomaly_detection_project)
* **주요 내용**: 실시간 공공 API 데이터 수집, Isolation Forest 및 Autoencoder 기반 이상 탐지
* **[➡️ 최종 결과 보고서](https://github.com/songgongho/Industrial_AI/blob/main/Portfolio_Cheongju_bus_anomaly_detection_project/report/%EC%B2%AD%EC%A3%BC%20%EB%B2%84%EC%8A%A4%20%EC%9D%B4%EC%83%81%20%EA%B0%90%EC%A7%80%20%EC%8B%9C%EC%8A%A4%ED%85%9C_2025254010_%EC%86%A1%EA%B3%B5%ED%98%B8.html)**

<br>

### 9. 🔍 이상 탐지 포트폴리오 #1 (Anomaly Detection Portfolio)

* **프로젝트 링크**: [**Portfolio_Anomaly Detection_1**](https://github.com/songgongho/Industrial_AI/tree/main/Portfolio_Anomaly%20Detection_1)
* **주요 내용**: 통계적 기법(Z-score) 및 딥러닝 기반(LSTM-Autoencoder) 이상 탐지 알고리즘 비교 분석
* **[➡️ 최종 결과 보고서](https://github.com/songgongho/Industrial_AI/blob/main/Portfolio_Anomaly%20Detection_1/Anomaly%20Detection%20Results%20Report_SONGGONGHO.pdf)**

<br>

### 10. 👁️ OpenCV 기반 영상 처리 심화 및 실습

**컴퓨터 비전 기초부터 인터랙티브 파라미터 최적화까지의 영상 처리 프로세스 구현**

* **프로젝트 링크**: [**Portfolio-OpenCV**](https://github.com/songgongho/Industrial_AI/tree/main/Portfolio-OpenCV)
* **주요 실습 내용**:
    * **Ex 01/03/05 (Color Analysis)**: BGR, HSV, YUV 색 공간 변환 및 휘도 채널 통계 분석을 통한 조명 강건성 확보
    * **Ex 02 (Preprocessing)**: 효율적인 연산을 위한 Grayscale 변환 및 데이터 입출력 최적화
    * **Ex 04 (Interactive Tuning)**: 트랙바 GUI를 활용한 실시간 이진화 임계값 탐색기 구현
    * **시스템 설계**: argparse 모듈을 통한 GUI/CLI 실행 모드 지원 및 예외 처리 로직 적용
* **결과 요약**:
    * 영상의 수치적 특성 분석과 동적 파라미터 튜닝을 통해 산업 현장의 비전 시스템 구축 핵심 역량 확보

* **[➡️ 최종 결과](https://github.com/songgongho/Industrial_AI/tree/main/Portfolio-OpenCV/results)**

<br>

### 11. ⚙️ AI4I 2020 예지보전(Predictive Maintenance) 및 F1 개선

**불균형 제조업 데이터를 활용한 MLP 베이스라인 구축 및 고장 예측 성능 최적화**

* **프로젝트 링크**: [**Portfolio_AI4I_Predictive_Maintenance**](https://github.com/songgongho/Industrial_AI/tree/main/Portfolio-ai4i_2020_predictive_maintenance)
* **주요 내용**:
    * PyTorch 환경 기반의 다층 퍼셉트론(MLP) 예지보전 베이스라인 모델 학습 및 평가지표 도출
    * Class Weight, SMOTE, Threshold Tuning 및 MLP 하이퍼파라미터 튜닝을 통한 데이터 불균형 완화
    * 방법론별 성능 지표(Accuracy, Precision, Recall, F1) 비교 그래프 시각화
    * 실험 결과 및 성능 분석 내용을 한국어 마크다운 PPT 보고서 형태로 자동 생성하는 파이프라인 구축
* **결과 요약**:
    * 데이터 불균형 개선 기법을 적용하여 모델의 고장 예측 목표 성능인 Failure F1 Score 0.70 이상을 달성함
    * 
   **[➡️ 최종 결과 보고서 바로가기](https://github.com/songgongho/Industrial_AI/blob/main/Portfolio-ai4i_2020_predictive_maintenance/%EA%B2%B0%EA%B3%BC%EB%B3%B4%EA%B3%A0%EC%84%9C_%EC%A0%9C%EC%A1%B0%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EB%B6%84%EC%84%9D%EA%B3%BC%20%EC%B5%9C%EC%A0%81%ED%99%94%20%EA%B3%BC%EC%A0%9C_2025254010%20%EC%86%A1%EA%B3%B5%ED%98%B8.pdf)**

<br>

### 12. 🏭 OPC UA 기반 제조 데이터 통합 파이프라인 구축

**가상 설비 서버 연동 및 정제, 라벨링, 품질 검증을 아우르는 데이터 파이프라인 자동화**

* **프로젝트 링크**: [**Portfolio_Manufacturing_Data_Pipeline**](https://github.com/songgongho/Industrial_AI/tree/main/Portfolio-%EC%A0%9C%EC%A1%B0%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EB%B6%84%EC%84%9D%20%EC%B5%9C%EC%A0%81%ED%99%94)
* **주요 내용**:
    * `asyncua` 라이브러리를 활용한 가상 제조 설비 서버 및 센서 데이터 수집 클라이언트 구현
    * 수집된 원시 데이터(`raw_sensor_dataset.csv`)의 결측치 대치 및 정제 자동화
    * 분석 목적에 맞춘 데이터 라벨링 및 품질 검증 모듈을 통합하여 `lifecycle_optimized_dataset.csv` 산출
    * 통합 실행기(`main.py`)를 개발하여 기초 데이터 발생부터 심화 수집 시나리오까지 모듈식 실행 환경 제공
* **결과 요약**:
    * 산업 표준인 OPC UA 통신 기반의 데이터 수집부터 최종 품질 점검까지의 프로세스를 자동화하여 현장 적용 가능한 데이터 엔지니어링 역량을 확보함

    **[➡️ 최종 결과 보고서 바로가기](https://github.com/songgongho/Industrial_AI/blob/main/Portfolio-%EC%A0%9C%EC%A1%B0%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EB%B6%84%EC%84%9D%20%EC%B5%9C%EC%A0%81%ED%99%94/result_report.md)**

<br>

***

## 📊 포트폴리오 요약 (Portfolio Summary)

| 프로젝트 | 분류 | 난이도 | 주요 기술 | 실무도 |
|---------|------|--------|---------|--------|
| 전력 수요 예측 | 시계열 | ⭐⭐⭐ | LSTM, GRU | ⭐⭐⭐⭐ |
| 은행 웹서버 | 웹개발 | ⭐⭐ | Flask, Python | ⭐⭐ |
| 공기질 예측 | 다변량 시계열 | ⭐⭐⭐ | LSTM, 다변량 | ⭐⭐⭐⭐ |
| **동호회 자동 보고서** | 웹자동화 | ⭐⭐ | HTML, CSS, JS | ⭐⭐⭐⭐⭐ |
| **신경망 기초** | 딥러닝 | ⭐⭐ | TensorFlow, Keras | ⭐⭐⭐⭐⭐ |
| **ML 기초** | 머신러닝 | ⭐⭐ | Scikit-learn | ⭐⭐⭐⭐ |
| **타이타닉** | 분류 | ⭐⭐ | Pandas, ML | ⭐⭐⭐ |
| **버스 이상탐지** | 이상탐지 | ⭐⭐⭐ | 공공API, 이상탐지 | ⭐⭐⭐⭐⭐ |
| **이상탐지 #1** | 이상탐지 | ⭐⭐⭐ | IF, LOF, AE | ⭐⭐⭐⭐⭐ |
| **OpenCV 심화** | **컴퓨터비전** | **⭐⭐⭐** | **OpenCV, Python** | **⭐⭐⭐⭐** |
| **AI4I 예지보전** | **예측모델링** | **⭐⭐⭐** | **PyTorch, SMOTE, MLP**| **⭐⭐⭐⭐⭐**|
| **제조 데이터 파이프라인** | **데이터엔지니어링** | **⭐⭐⭐** | **OPC UA, asyncua** | **⭐⭐⭐⭐⭐** |

<br>

***

## 💼 핵심 역량 (Core Competencies)

### 데이터 수집 & 파이프라인 구축
* ✅ 산업 표준 통신(OPC UA) 기반 실시간 제조 설비 및 센서 데이터 수집 아키텍처 구현
* ✅ 데이터 정제, 라벨링, 품질 검증에 이르는 End-to-End 자동화 파이프라인 통합

### 데이터 분석 & 전처리
* ✅ 데이터 수집 및 정제 / 탐색적 데이터 분석 (EDA)
* ✅ 결측치 처리, 데이터 불균형 해소(SMOTE 등) 및 특성 공학 (Feature Engineering)

### 머신러닝 & 딥러닝
* ✅ 지도/비지도 학습 및 이상 탐지 모델 구현 (IF, LOF, Autoencoder)
* ✅ 신경망 아키텍처 설계 (MLP, CNN, RNN, LSTM) 및 평가지표(F1 Score 등) 기반 하이퍼파라미터 최적화

### 컴퓨터 비전 (Computer Vision)
* ✅ OpenCV를 활용한 영상 필터링 및 색 공간 분석
* ✅ 트랙바 기반 인터랙티브 영상 전처리 도구 개발
* ✅ MediaPipe 및 AI 비전 API를 활용한 특징점 검출

### 실무 기술
* ✅ 분석 실험 결과 마크다운 자동 생성 및 실행 파이프라인 구축
* ✅ 프론트엔드 업무 자동화 및 Flask 웹 서버 연동
* ✅ Git/GitHub 버전 관리 및 협업 도구 활용

<br>

***

## 📈 학습 경로 추천 (Learning Paths)

### 🟢 초급자 (Beginner)
* 신경망 기초 (Portfolio-DeepLearning-Fundamentals)
* ML 기초 (Portfolio_ML-Fundamentals)
* 타이타닉 예측 (Portfolio_Titanic_Survival_Prediction)

### 🟡 중급자 (Intermediate)
* 전력 수요 예측 (시계열 기초)
* OpenCV 심화 실습 (컴퓨터 비전 기초)
* AI4I 2020 예지보전 프로젝트 (데이터 불균형 개선 실습)

### 🔴 고급자 (Advanced)
* 버스 이상 탐지 (실무 프로젝트)
* OPC UA 제조 데이터 파이프라인 구축 (데이터 엔지니어링 응용)
* 동호회 정산 자동 생성기 (업무 자동화 응용)

<br>

***

## 🎯 진행 중인 프로젝트

* 🔄 스마트팩토리 센서 데이터 이상 탐지 고도화
* 🔄 강화학습 적용 및 시뮬레이션
* 🔄 CNN 기반 영상 분류 프로젝트 구현
* 🔄 실시간 비전 인식 시스템 프로토타입 제작

<br>

***

## 📞 연락처 (Contact)

* **GitHub**: [songgongho](https://github.com/songgongho)
* **Email**: [song5@kakao.com]

<br>

***

## 📝 라이선스 (License)

이 포트폴리오의 모든 코드는 MIT License를 따름

***

**마지막 업데이트**: 2026년 04월
**포트폴리오 상태**: 🟢 활발히 업데이트 중
