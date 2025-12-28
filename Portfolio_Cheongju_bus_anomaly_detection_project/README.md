# Cheongju-bus-anomaly-detection

청주시 TAGO 버스 운행 데이터와 날씨 정보를 활용해
KMeans + IsolationForest + LSTM-AE로 이상 운행 패턴과 폭설 시 이상 상황을 감지하는 프로젝트입니다.

## 1. 프로젝트 개요
- 목표: 폭설/이용량 급변 등으로 인한 버스 이상 운행을 자동 감지하고 운영 의사결정을 지원
- 대상: 청주시 115개 노선, 30일 × 24시간 운행 패턴

## 2. 데이터셋
- `data/cheongju_bus_routes-2.csv`: TAGO API로 수집한 청주 버스 노선 메타데이터
- `data/res_insight.csv`: 노선-일자별 클러스터 / 이상점수 / 인사이트
- `data/daily_anomalies.csv`: 일별 평균 이상점수 및 이상 노선 수
- `data/res_insight_snowfall.csv`: 폭설(12-04) 시 실험 결과

## 3. 사용 알고리즘
- KMeans (n_clusters=5): 운행 패턴(출근형, 저수요형, 균등형, 저녁형, 복합형) 자동 분류
- IsolationForest (contamination=0.1): 노선-일자 단위 이상 탐지
- LSTM-AE: 24시간 시계열 재구성 기반 정교한 이상도 계산

## 4. 코드 구조
- `src/`: TAGO API 수집, 시뮬레이션, 피처 생성, 모델 학습·평가 스크립트
- `notebook/`: 실험 노트북(R01~R03)
- `figures/`: 논문/발표용 시각화 이미지
- `ppt/`: 최종 발표 자료 (PPTX)

## 5. 실행 방법
git clone https://github.com/<username>/cheongju-bus-anomaly-detection.git
cd cheongju-bus-anomaly-detection
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate
pip install -r requirements.txt

python src/paeteoninsig_ceongju_beoseunoseon_isangbalsaeng_gamji_silseub_r03-2.py


## 6. 결과
예시 그래프는 `figures/` 폴더를 참고하세요.
- 이상점수 분포 히스토그램
- 상위 10개 이상 노선 바 차트
- 클러스터별 24시간 평균 운행 패턴
- 12월 4일(정상) vs 5일(폭설) 운행 비교 그래프

## 7. 라이선스
MIT License
