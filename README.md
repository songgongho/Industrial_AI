# 📈 Industrial AI Projects Portfolio

안녕하세요! 산업 인공지능 분야의 문제 해결을 목표로 다양한 프로젝트를 진행하고 있는 송공호입니다.  
현재(2025.07.기준) 충북대학교 산업인공지능학과에 재학중입니다. 

Industrial_AI 레퍼지토리는 관련 포트폴리오를 정리하여 업로드 합니다.

이 레퍼지토리에서는 제 기술 스택과 프로젝트 경험을 공유합니다.


<br>

## 🛠️ 기술 스택 (Tech Stack)

* **언어 (Languages)**: Python
* **라이브러리 (Libraries)**: Pandas, NumPy, Scikit-learn, TensorFlow, Keras, Matplotlib, Flask
* **도구 (Tools)**: Git, GitHub, Jupyter Notebook

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
    LSTM 모델이 다른 모델에 비해 안정적이고 정확한 예측 성능을 보였습니다. 상세 내용은 아래 링크에서 전체 결과 보고서를 확인하실 수 있습니다.

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
    사용자 세션 관리를 통해 안전한 금융 거래를 시뮬레이션하고, 기본적인 웹 서버의 동작 원리를 구현했습니다. 상세 내용은 아래 링크에서 전체 결과 보고서를 확인하실 수 있습니다.

    **[➡️ 결과 보고서 바로가기](https://github.com/songgongho/Industrial_AI/blob/main/Portfolio_project_bank/%EC%9D%80%ED%96%89%EC%9B%B9%EC%84%9C%EB%B2%84%20%EC%A0%9C%EC%9E%91_%EA%B2%B0%EA%B3%BC%EB%B3%B4%EA%B3%A0%EC%84%9C_%EC%86%A1%EA%B3%B5%ED%98%B8.pptx)**

<br>

### 3. 다변량 데이터를 이용한 공기질 예측 모델

여러 환경 변수(온도, 습도 등)를 동시에 고려하여 특정 지역의 공기질(미세먼지 농도 등)을 예측하는 다변량 시계열 분석 프로젝트입니다.

* **프로젝트 링크**: [**Portfolio_projects_공기질예측**](https://github.com/songgongho/Industrial_AI/tree/main/Portfolio_projects_%EA%B3%B5%EA%B8%B0%EC%A7%88%EC%98%88%EC%B8%A1)
* **주요 내용**:
    * 여러 센서 데이터를 통합한 다변량 시계열 데이터셋 구축
    * 다변량 예측에 적합한 LSTM, GRU 등의 딥러닝 모델 활용
    * 각 환경 변수가 공기질 예측에 미치는 영향 분석
* **결과 요약**:
    다변량 데이터를 함께 학습했을 때 단일 변수 모델보다 예측 정확도가 향상되는 것을 확인했습니다. 상세 내용은 아래 링크에서 전체 결과 보고서를 확인하실 수 있습니다.

    **[➡️ 결과 보고서 바로가기](https://github.com/songgongho/Industrial_AI/blob/main/Portfolio_projects_%EA%B3%B5%EA%B8%B0%EC%A7%88%EC%98%88%EC%B8%A1/%EA%B3%B5%EA%B8%B0%EC%A7%88(%EB%8B%A4%EB%B3%80%EB%9F%89)%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EA%B8%B0%EB%B0%98%20%EC%98%88%EC%B8%A1%EB%AA%A8%EB%8D%B8%20%EA%B0%9C%EB%B0%9C%20%EA%B2%B0%EA%B3%BC%EB%B3%B4%EA%B3%A0%EC%84%9C_%EC%86%A1%EA%B3%B5%ED%98%B8%2020250616.pptx)**

<br>

---
