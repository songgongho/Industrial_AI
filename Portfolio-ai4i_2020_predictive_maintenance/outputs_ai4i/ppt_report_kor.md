# 슬라이드 1: 과제 및 목표 요약
- 과제: MLP 베이스라인 모델의 취약점 분석 및 성능 개선
- 목표: Test 데이터 기준 F1-Score 0.70 이상 달성
- 접근: 불균형 처리(Class Weight/SMOTE), 임계값 최적화, MLP 구조/학습률 튜닝을 단계적으로 비교

# 슬라이드 2: 데이터셋 개요
AI4I 2020 Predictive Maintenance 데이터는 제조 공정 센서 기반 이진 분류 문제로, 타깃은 `Machine failure`입니다.
주요 피처는 공기온도, 공정온도, 회전속도, 토크, 공구마모 시간과 제품 타입(Type)입니다.
고장 관련 세부 레이블(TWF/HDF/PWF/OSF/RNF)은 타깃과 직접 연관된 누수 가능성이 있어 학습 피처에서 제외했습니다.
전체 데이터에서 고장 비율이 낮아 클래스 불균형이 존재하며, 이는 Accuracy 중심 평가의 한계를 유발합니다.

# 슬라이드 3: 베이스라인 모델 구조 및 성능
- 모델: 2-hidden MLP(64, 64), ReLU, BatchNorm, Dropout
- 학습: BCEWithLogitsLoss, Adam, stratified split(60/20/20)
- Baseline Test 성능: Accuracy=0.977, F1-macro=0.749, F1-failure=0.511
- 평가지표 불균형: Accuracy 대비 Failure F1 격차가 존재하여 소수 클래스 검출 한계 확인

# 슬라이드 4: 성능 개선 방법론(1)
- 클래스 가중치: 소수 클래스 오분류에 더 큰 손실을 부여하여 failure 검출 민감도 강화
- 임계값 튜닝: 고정 0.5 대신 검증셋에서 F1 최대 임계값 선택
- 수치 데이터 해석 관점: 손실 함수/결정 임계값을 목표 지표(F1)에 정렬해 불균형 편향 완화

# 슬라이드 5: 성능 개선 방법론(2)
- SMOTE: 훈련셋의 소수 클래스 분포를 확장해 경계학습 안정화
- SMOTEENN/Focal Loss/Ensemble: 경계 정리 + 난이도 기반 손실 + 다중 시드 평균으로 failure 검출 안정성 향상
- 해석 관점: 데이터 레벨 + 손실 레벨 + 의사결정 레벨을 동시에 보정해 failure F1 향상 유도

# 슬라이드 6: 방법론별 평가지표 변화 추이
- 최고 성능 방법: +SMOTEENN+ThresholdTuning
- 최고 성능 지표: Accuracy=0.981, F1-macro=0.867, F1-failure=0.743
- 목표(0.70) 달성 여부: 달성
- Baseline 대비 Failure F1 변화: 0.511 -> 0.743 (+0.233p)

# 슬라이드 7: 최종 모델 및 결론
- 최종 선정: +SMOTEENN+ThresholdTuning
- 핵심 성과: Baseline 대비 failure 클래스 검출 성능(F1) 정량 개선
- 해석: Accuracy는 일부 변동 가능하나, 고장 탐지 품질 관점에서는 F1 개선이 더 중요한 의사결정 근거
- 한계/향후: 임계값의 운영 환경 재보정, 시간축/설비별 드리프트 반영, 비용민감 학습 추가 검토

# 부록: Baseline 취약점 자동 분석 문구
### Baseline 취약점 요약
- Accuracy와 failure F1의 격차: **0.466p** (Accuracy=0.977, Failure F1=0.511)
- Failure(1) Recall=0.353, Precision=0.923로, 소수 클래스 검출 성능이 제한됨
- Normal(0) Recall=0.999로 다수 클래스 중심의 예측 경향이 존재
- 고정 임계값 0.5는 F1 최적점과 불일치할 수 있어 재현율/정밀도 균형이 깨짐

정확도는 다수 클래스(정상)가 매우 많은 데이터에서 쉽게 높아질 수 있으므로, 실제 고장 검출 품질을 충분히 반영하지 못합니다.
데이터 불균형 상황에서는 BCE 손실이 전체 오류를 평균화하면서 소수 클래스의 오분류 비용을 상대적으로 작게 학습할 수 있습니다.
또한 임계값 0.5는 운영 목적(F1 최대화)과 다를 수 있어 failure recall이 낮아지고, 그 결과 failure F1이 accuracy 대비 낮게 나타납니다.

# 방법별 결과 테이블

| method | accuracy | precision_macro | recall_macro | f1_macro | f1_positive | roc_auc | meets_goal |
| --- | --- | --- | --- | --- | --- | --- | --- |
| +SMOTEENN+ThresholdTuning | 0.9810 | 0.8404 | 0.8979 | 0.8667 | 0.7432 | 0.9790 | True |
| +SMOTE+ThresholdTuning | 0.9815 | 0.8494 | 0.8840 | 0.8658 | 0.7413 | 0.9768 | True |
| +Ensemble3(+SMOTEENN+ThresholdTuning) | 0.9815 | 0.8520 | 0.8769 | 0.8640 | 0.7376 | 0.9772 | True |
| +FocalLoss+SMOTEENN+Threshold | 0.9810 | 0.8472 | 0.8767 | 0.8613 | 0.7324 | 0.9751 | True |
| +FocalLoss+SMOTE+ThresholdTuning | 0.9785 | 0.8201 | 0.8966 | 0.8539 | 0.7190 | 0.9741 | True |
| +ClassWeight+ThresholdTuning | 0.9770 | 0.8143 | 0.8675 | 0.8386 | 0.6892 | 0.9723 | False |
| +UltraFocal+SMOTE_US | 0.9765 | 0.8118 | 0.8601 | 0.8340 | 0.6803 | 0.9678 | False |
| +FocalLoss(g1.5,a0.85)+SMOTE+Threshold | 0.9805 | 0.8799 | 0.7984 | 0.8338 | 0.6777 | 0.9738 | False |
| +FocalLoss+SMOTE_US+ThresholdTuning | 0.9740 | 0.7938 | 0.8518 | 0.8199 | 0.6533 | 0.9672 | False |
| +TunedMLP(ClassWeight+SMOTE+Threshold) | 0.9735 | 0.8076 | 0.7593 | 0.7812 | 0.5760 | 0.9648 | False |
| Baseline | 0.9770 | 0.9504 | 0.6760 | 0.7494 | 0.5106 | 0.9761 | False |
| +ClassWeight | 0.9160 | 0.6354 | 0.9069 | 0.6877 | 0.4207 | 0.9723 | False |
