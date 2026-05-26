# 4단계 | 목표 결과 및 검증 전략

## 정량 목표

### 성능 지표 (synthpress 합성 데이터 기반)

| 지표 | 목표값 | 기준 베이스라인 | 비고 |
|-----|-------|---------------|------|
| **AUROC** | ≥ 0.95 | IsolationForest: 0.87 | 이상 탐지 표준 |
| | | 단순 LSTM: 0.89 | |
| **FAR@Recall=0.95** | < 5% | Random Forest: 8% | 불량 95% 탐지 시 오탐 최소 |
| | | CNN-AE: 6% | |
| **Cost-Aware Score** | 최소화 | XGBoost (단일 모달): 0.15 | FN:FP=20:1 비용 구조 반영 |
| **F1 Score** | ≥ 0.85 | SECOM 기존 기법: 0.78 | 정확도와 재현율 균형 |

### 세부 목표

**클래스별 성능:**
```
정상 (Negative, 95%):
- Specificity ≥ 0.95 (오탐 최소)

불량 (Positive, 5%):
- Sensitivity(Recall) ≥ 0.95 (미탐 최소)
- Precision ≥ 0.80 (검사 비용 합리적 수준)
```

**모달별 기여도:**
```
시계열 센서 기여도: 60~70%
이벤트 로그 기여도: 30~40%

(상대적 기여도, Cross-modal Attention가중치 기반)
```

---

## 검증 설계

### Ablation Study 1: 단일 모달 vs 이중 모달

**목표**: Cross-modal Attention의 기여도 정량화

**대조군:**
```
Model A (Baseline): 시계열 센서만
├─ LSTM → Dense → Output
└─ 결과: AUROC 0.89

Model B (Proposed: Full PressFuse)
├─ Cross-modal Attention(TS + Event)
└─ 결과: AUROC 0.95+

기대 개선: +6% (0.89 → 0.95)
```

**실험 설정:**
- 합성 데이터: 1,000개 사이클 (불량 10%)
- 학습:검증:테스트 = 60:20:20
- Epoch: 100, Patience: 20
- optimizer: Adam, lr=1e-3

**수집 지표:**
- AUROC, F1, Precision, Recall
- 모달 생략 시 성능 저하 곡선

---

### Ablation Study 2: Cross-modal Attention 유무

**목표**: Attention 메커니즘의 필요성 확인

**대조군:**
```
Model C (No Attention): 단순 concatenation
├─ TS_encoded ⊕ Event_encoded → Dense
└─ AUROC: 0.91

Model B (Cross-modal Attention)
├─ scaled_dot_product_attention(Q, K, V)
└─ AUROC: 0.95+

기대 개선: +4% (0.91 → 0.95)
```

**분석 항목:**
- Attention weight의 분포 (어느 모달에 집중되는가?)
- 시간대별 attention 패턴 (어느 시각대가 중요인가?)

---

### Ablation Study 3: 비용민감 손실함수 유무

**목표**: Cost-aware 손실함수의 효과 검증

**대조군:**
```
Model D (Standard BCE):
├─ Loss = BCE(y, pred)
├─ FN:FP Cost 무시
└─ 결과: FAR@Recall=0.95 = 8%

Model B (Cost-weighted CE):
├─ Loss = BCE + λ × cost_aware_penalty
├─ FN_cost=20, FP_cost=1
└─ 결과: FAR@Recall=0.95 = < 5%

기대 개선: FAR 감소 (8% → <5%)
```

**추적 지표:**
- ROC curve (threshold별 TPR vs FPR)
- Cost-savings curve (real operations)
- Threshold 움직임 패턴

---

### 시나리오별 성능 비교 (Scenario Analysis)

**목표**: 다양한 불량 유형에 따른 성능 검증

**6종 단일 이상:**
```
P013-001 (Pressure Drop): 감지율 __% 
P013-002 (Pressure Spike): 감지율 __% 
P013-003 (Temp Overshoot): 감지율 __% 
P013-004 (Temp Fail to Rise): 감지율 __% 
P013-005 (Vacuum Leak): 감지율 __% 
P013-006 (Program Fault): 감지율 __% 

기대: 모두 ≥ 90%
```

**4종 Cascade 이상:**
```
P013-001 + P013-004 (Pressure + Temp): 감지율 __% 
P013-005 + P013-002 (Vacuum + Pressure): 감지율 __% 
P013-003 + P013-002 (Temp + Pressure): 감지율 __% 
Gradual Degradation (장기 열화): 감지율 __% 

기대: 모두 ≥ 85% (복잡도로 인한 감소 허용)
```

**분석:**
- 단순 이상 vs Cascade 이상 감지 능력 비교
- 어느 유형이 더 어려운지 파악
- 모델 개선 방향 도출

---

## 교차 검증 전략

### Hold-out Test Set
```
Train: 60% (600 cycles, 정상:불량 = 9:1)
Valid: 20% (200 cycles, same ratio)
Test:  20% (200 cycles, same ratio)

불균형 처리: 
- class weight 적용 (loss 함수)
- stratified split 사용
```

### k-Fold Cross-Validation (k=5)
```
제한: 합성 데이터는 반복 생성 가능
      (stochasticity 고려)

매 fold마다:
╔═══════════════════════════════════╗
║ fold 1: 80% train, 20% test      ║
║ fold 2: 80% train, 20% test      ║
║ fold 3: 80% train, 20% test      ║
║ fold 4: 80% train, 20% test      ║
║ fold 5: 80% train, 20% test      ║
╚═══════════════════════════════════╝

결과: mean ± std
```

---

## 베이스라인 모델 (Baseline Comparison)

| Baseline | 특징 | 기대 AUROC |
|----------|-----|-----------|
| **IsolationForest** | 비지도, 단순 | 0.87 |
| **LSTM (단일 모달)** | 시계열 학습 | 0.89 |
| **CNN-AE** | 시계열 + 재구성 | 0.90 |
| **Random Forest** | 앙상블, 해석 가능 | 0.88 |
| **XGBoost** | 그래디언트 부스팅 | 0.91 |
| **PressFuse (본 연구)** | 멀티모달 + 비용민감 | **0.95+** |

### 베이스라인 코드 예시

```python
# IsolationForest
iso = IsolationForest(contamination=0.05)
y_pred = iso.fit_predict(X)

# LSTM
lstm_model = LSTM(units=64, dropout=0.2)
lstm_model.compile(loss='binary_crossentropy', optimizer='adam')

# CNN-AE
encoder = Conv1D(32, 3, activation='relu')
decoder = Conv1DTranspose(32, 3, activation='relu')

# Random Forest
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced')

# XGBoost
xgb = XGBClassifier(scale_pos_weight=20, max_depth=7, learning_rate=0.1)
```

---

## 실험 일정

```
Week 1-2:
├─ 베이스라인 모델 학습 및 평가
├─ PressFuse 학습 (기본 설정)
└─ 초기 결과 수집

Week 3-4:
├─ Ablation Study 1: 모달 분리
├─ Ablation Study 2: Attention 유무
├─ Ablation Study 3: 비용민감 함수

Week 5-6:
├─ 시나리오별 성능 분석
├─ 하이퍼파라미터 튜닝
└─ 최종 모델 확정

Week 7-8:
├─ 교차 검증 (5-fold)
├─ 설명 가능성 검증 (SHAP + Attention)
└─ 최종 리포트 작성
```

---

## 성공 기준 (Success Criteria)

### 필수 조건 (Must-have)
- [ ] AUROC ≥ 0.95
- [ ] FAR@Recall=0.95 < 5%
- [ ] 3가지 Ablation 완료 및 개선량 정량화

### 권장 조건 (Nice-to-have)
- [ ] F1 ≥ 0.85
- [ ] Scenario별 감지율 ≥ 85%
- [ ] SHAP + Attention 설명 시각화
- [ ] 5-fold CV 확인

### 논문 게재 기준
- ✅ 위 필수 조건 모두 달성
- ✅ 베이스라인 대비 명확한 개선 (≥ 4%)
- ✅ 통계적 유의성 검증 (p < 0.05)

---

## 다음 단계: 논문 서술 계획
👉 [STAGE_5_PAPER_OUTLINE.md](STAGE_5_PAPER_OUTLINE.md) 참조

