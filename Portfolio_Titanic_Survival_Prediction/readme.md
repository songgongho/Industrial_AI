# 🚢 Titanic Survival Prediction: End-to-End ML Pipeline

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Kaggle](https://img.shields.io/badge/Kaggle-Top%2030%25-blueviolet)

## 📌 Project Overview
Kaggle의 대표적인 입문 과제인 **Titanic - Machine Learning from Disaster**의 생존자 예측 프로젝트입니다.  
데이터 전처리(Pre-processing), 탐색적 데이터 분석(EDA), 피처 엔지니어링(Feature Engineering), 모델 튜닝 및 교차 검증(Cross Validation)까지의 **전체 머신러닝 파이프라인**을 구축했습니다.

- **Goal**: 탑승객의 신상 정보를 기반으로 생존 여부(0: 사망, 1: 생존)를 이진 분류(Binary Classification)
- **Result**: Kaggle Leaderboard Score **0.77033** (SVM Model)

## 🛠️ Tech Stack
- **Language**: Python
- **Data Manipulation**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Missingno
- **Machine Learning**: Scikit-learn (SVM, RandomForest, KNN, DecisionTree)

## 📊 Key Workflow & Analysis

### 1. Exploratory Data Analysis (EDA)
데이터의 분포와 생존율 간의 상관관계를 분석하여 주요 인사이트를 도출했습니다.
- **성별(Sex)**: 여성이 남성보다 생존율이 압도적으로 높음 (상관계수 -0.54)
- **객실 등급(Pclass)**: 1등급 > 2등급 > 3등급 순으로 생존율이 높음
- **운임(Fare)**: 운임이 높을수록 생존할 가능성이 높음

### 2. Feature Engineering
모델 성능 향상을 위해 기존 변수를 가공하여 새로운 파생 변수를 생성했습니다.

| Feature | Engineering Method | Description |
|:---:|---|---|
| **AgeGroup** | Binning | 나이를 10세 단위로 구간화하여 범주형 변수로 변환 |
| **Title** | String Extraction | 이름(Name)에서 `Mr`, `Mrs`, `Miss` 등의 호칭 추출 및 희귀 호칭 통합 |
| **Cabin** | String Extraction | 객실 번호의 첫 글자(Deck)만 추출, 결측치는 'U'(Unknown) 처리 |
| **FamilySize** | Combination | `SibSp`(형제자매) + `Parch`(부모자녀) + 1(본인) |
| **Fare** | Quantile Binning | 운임 요금을 4분위수로 나누어 범주화 |

### 3. Model Selection & Evaluation
과적합(Overfitting)을 방지하고 일반화 성능을 높이기 위해 **5-Fold Stratified Cross Validation**을 수행했습니다.

| Model | Train Accuracy | Validation Accuracy | Note |
|---|---|---|---|
| KNN | 0.845 | 0.796 | - |
| Decision Tree | **0.920** | 0.813 | 과적합 경향 보임 |
| Random Forest | 0.903 | 0.813 | Feature Importance 분석용으로 활용 |
| **SVM** | 0.836 | **0.829** | **최종 모델로 선정 (가장 안정적)** |

> **Insight**: Random Forest의 Feature Importance 분석 결과, **Sex(성별)** 변수가 예측에 가장 큰 영향을 미치는 것으로 나타났습니다.

## 🚀 How to Run

### Prerequisites
필요한 라이브러리를 설치합니다.

