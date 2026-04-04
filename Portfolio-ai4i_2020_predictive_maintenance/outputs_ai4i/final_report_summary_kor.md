# AI4I 결과 요약 (제출용 1페이지)

## 1) 과제 목표
- 과제: 베이스라인 MLP의 지표 불균형(Accuracy 대비 failure F1 저하) 분석 및 개선
- 목표: 테스트셋 기준 `F1-failure >= 0.70`

## 2) 핵심 문제 진단
- 데이터 불균형: 정상 96.61%, 고장 3.39%
- Baseline 성능: Accuracy=0.978, F1-macro=0.794, F1-failure=0.600
- 해석: Accuracy는 높지만 failure Recall이 낮아 실제 고장 검출 품질이 부족

## 3) 적용한 개선 방법
- 불균형 보정: Class Weight, SMOTE, SMOTEENN
- 의사결정 보정: Validation 기반 Threshold Tuning
- 학습 보정: Focal Loss, 구조/정규화 튜닝
- 안정화: Soft-Voting Ensemble(다중 시드 평균)

## 4) 최종 성능 및 결론
- 최종 최고 모델: `+FocalLoss+SMOTE+ThresholdTuning`
- 최종 성능: Accuracy=0.982, F1-macro=0.869, F1-failure=0.746
- 개선 폭: `0.600 -> 0.746` (**+0.146p**)
- 목표 달성 여부: **달성**

## 5) 실무 적용 관점 코멘트
- 본 과제에서는 Accuracy보다 F1-failure를 핵심 의사결정 지표로 사용
- 고장 검출 누락 비용이 큰 환경에서 임계값 최적화와 불균형 보정 조합이 유효
- 운영 배포 시에는 설비/기간별 데이터 드리프트를 고려해 임계값을 주기적으로 재보정 권장

