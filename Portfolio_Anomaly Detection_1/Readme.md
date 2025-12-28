Portfolio: 이상 탐지 (Anomaly Detection)
클래스 불균형, Isolation Forest, AutoEncoder를 활용한 반도체 제조 공정 이상 탐지. 스마트 팩토리 MES 데이터 기반 비지도 학습 적용.
​

📋 목차
📊 데이터셋

🎯 문제 정의

🔧 방법론

📈 결과 요약

💻 실행 방법

📁 파일 구조

📊 데이터셋
SECom (반도체 제조 공정): 590 특성, 1567 샘플, 이상 비율 6.7% (104/1567)
​

Credit Card Fraud: 30 특성, 고도로 불균형 (0.17% 이상)
​

text
label 0 (정상): 1463개 (93.3%)
label 1 (이상): 104개   (6.7%)
![SECom 데이터 분포](results/figures/secom_class_distribution.png

🎯 문제 정의
도전 과제: 클래스 불균형(93:7), 레이블 부족 → 비지도 학습 적용

text
정상(Normal) vs 이상(Abnormal)
• 점 이상(Point): 개별 데이터 포인트 이상
• 맥락 이상(Context): 시계열/공정 맥락상 이상 [file:1]
🔧 방법론
단계	모델	핵심 기술
1단계	Class Weight	balanced, F1-score 0.32↑ 
​
2단계	Isolation Forest	contamination=0.071, 깊이 기반 이상 스코어 
​
3단계	AutoEncoder	MSE reconstruction error, F1 최적 threshold 
​
![메소드 파이프라인](results/figures/method_pipeline 요약

모델	데이터셋	F1-Score	Precision	Recall	Threshold
Class Weight	SECom	0.32	0.91	0.20	- 
​
Isolation Forest	SECom	0.28	0.85	0.19	contamination=0.071 
​
AutoEncoder	Credit Card	0.75	0.82	0.69	MSE=0.01 
​
![모델 비교](results/figures/model_comparison사이트**:

text
✅ AutoEncoder > Isolation Forest > Class Weight (F1-score)
✅ Reconstruction Error 기반 threshold 최적화 효과적
✅ 스마트팩토리 실시간 이상 탐지 적용 가능 [memory:11]
💻 실행 방법
1. 환경 설정
bash
git clone https://github.com/songgongho/Portfolio_AnomalyDetection_1.git
cd Portfolio_AnomalyDetection_1
pip install -r requirements.txt
2. Jupyter 실행
bash
jupyter notebook notebooks/
3. 순서대로 실행
text
01-Class-Imbalance_Class-Weight.ipynb  → 클래스 불균형 해결
02-isolation-forest.ipynb             → Isolation Forest
03-AutoEncoder.ipynb                  → 딥러닝 AutoEncoder
requirements.txt:

text
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
tensorflow==2.13.0
matplotlib==3.7.2
seaborn==0.12.2
jupyter==1.0.0
📁 파일 구조
text
Portfolio_AnomalyDetection_1/
├── README.md                    # 📄 이 문서
├── notebooks/                   # 🧪 Jupyter 노트북 (3개)
│   ├── 01-Class-Imbalance_Class-Weight.ipynb
│   ├── 02-isolation-forest.ipynb
│   └── 03-AutoEncoder.ipynb
├── data/                        # 📊 원본 데이터셋
│   └── raw/
│       ├── secom.csv
│       └── creditcard.csv
├── results/                     # 📊 결과물
│   ├── figures/                 # 시각화 이미지
│   └── metrics/                 # 성능 메트릭 CSV
├── report/                      # 📋 상세 보고서
│   └── anomaly_detection_report.md
└── requirements.txt             # 📦 의존성
🚀 스마트 팩토리 적용
text
MES + IoT 센서 데이터 → 실시간 이상 탐지
• 생산 라인 정지 예측 (예방 정비)
• 품질 불량 원인 자동 탐지
• 공정 최적화 ROI 분석 [memory:17]
