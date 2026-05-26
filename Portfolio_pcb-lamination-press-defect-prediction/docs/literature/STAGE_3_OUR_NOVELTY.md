# 3단계 | 본 연구의 차별점 (Novelty & Contribution)

## 연구 제목 (안)

**"PressFuse: Cost-Sensitive Multimodal Fusion with Cross-Modal Attention for In-Process Defect Prediction in PCB MLB Lamination Press"**

---

## 차별점 4가지: C1 ~ C4

### C1: Press 도메인 특화 멀티모달 융합

#### 본 연구
```
시계열 센서        이벤트 로그
(19개 P013변수)  (알람, 설비상태)
     ↓                ↓
   Encoder          Encoder
     ↓                ↓
    Q,K,V          Q,K,V
     ↓________________↓
        Cross-modal Attention
        (learned weights)
             ↓
    모달 간 상호 정보 교환
             ↓
        Fusion Representation
             ↓
    불량 분류 + 신뢰도
```

**특징:**
- ✅ Cross-modal Attention: 두 모달 간 **동적 가중치** 학습
- ✅ Press 공정의 물리 제약조건 반영
  - 압력과 온도의 강한 양의 상관
  - 진공과 온도의 약한 음의 상관
  - 알람과 센서 이상의 시간 누적 효과
- ✅ 사전학습(pretrain) 가능 → fine-tune으로 실공정 적응

#### 선행 연구 (MFGAN, 2024)
```
- 범용 멀티모달 아키텍처
- 데이터셋: 합성 데이터 또는 벤치마크
- 도메인 물리 법칙 미반영
```

**차이점:**
| 항목 | 본 연구 | 선행 연구 |
|-----|-------|---------|
| 도메인 | PCB Press 특화 | 범용 |
| 모달 | 센서 + 이벤트 로그 | 센서만 또는 범용 |
| 물리 제약 | 반영 | 미반영 |
| 설명 가능성 | 높음 (attention weights) | 낮음 |

---

### C2: 합성 데이터 기반 사전학습 (Physics-informed Synthetic Data)

#### 본 연구: synthpress 생성기

**물리 기반 시나리오 (10종):**

```python
# 6종 단일 이상 (P013-001 ~ P013-006)
1. Pressure Drop: 목표 압력 도달 실패
2. Pressure Spike: 급격한 압력 상승
3. Temperature Overshoot: 온도 과승
4. Temperature Fail to Rise: 온도 상승 지연
5. Vacuum Leak: 진공 누설
6. Program Fault: 프로그램 오류 (급정지)

# 4종 cascade 이상 (연쇄 이상, 현실적)
7. Pressure Drop + Temperature Lag
8. Vacuum Leak + Pressure Fluctuation
9. Temperature Overshoot + Pressure Spike
10. Gradual Degradation (장기 열화)
```

**생성 알고리즘:**
```
사이클 특성:
├─ 설정값 (Set Temp, Set Pressure, Set Vacuum)
├─ 정상 경로 (시뮬레이션 기반)
└─ 이상 주입 (시나리오별 합리적 왜곡)

모달별 출력:
├─ 센서 시계열(192개 시점 × 19변수)
├─ 이벤트/알람 추적
├─ 라벨 (불량 유형 및 시작 시점)
└─ 메타데이터 (cycle_id, anomaly_type, severity)
```

**성과 기대:**
- ✅ 라벨 희소 문제 해결 (합성 1000+ 샘플)
- ✅ 실공정 적응 가능성 (물리 기반 → fine-tune)
- ✅ 다양한 부패 경로 학습 가능

#### 선행 연구

**GAN 기반 범용 합성 (MFGAN, 2024 등):**
- 데이터 분포 학습 → 새 샘플 생성
- **문제**: 도메인 물리 법칙 미반영
- **결과**: 현실성 낮은 이상 시나리오

**SECOM 데이터 증강 (2022~2024):**
- SMOTE, 회전/스케일링 증강
- **문제**: 라벨된 불량 샘플 기반만 가능
- **한계**: 새로운 이상 유형 생성 불가

**차이점:**

| 항목 | 본 연구 (synthpress) | 선행 연구 (GAN/SMOTE) |
|-----|-------------------|-------------------|
| 원리 | **물리 기반** 시뮬레이션 | 통계 기반 |
| 현실성 | 높음 (도메인 제약 준수) | 낮음 (확률적) |
| 새 시나리오 | ✅ 생성 가능 | ❌ 기존 데이터 재조합만 |
| 라벨 정보 | 풍부 (시작시점, 심각도) | 단순 (정상/불량만) |
| 공정 이전성 | ✅ 높음 | ❌ 낮음 |

---

### C3: 비용민감 이중 지표 (Cost-Aware Dual Metrics)

#### 본 연구

**지표 1: cost_aware_score**
```python
cost_aware_score = 
  (FN_count × FN_cost + FP_count × FP_cost) 
  / (Total_Negatives × FN_cost + Total_Positives × FP_cost)

# Press 공정 기본 설정:
FN_cost = 20  (불량 폐기/리콜 손실)
FP_cost = 1   (정상 제품 재검 또는 재작업)
```

**지표 2: FAR@Recall=0.95**
```
Recall = 0.95: 불량의 95%는 반드시 탐지
                (5% 미탐 허용 최소한)

FAR (False Alarm Rate) 최소화:
FAR = FP / (FP + TN)

의미: Recall 고정 → FP를 최소화
      생산 중단(연쇄 손실) 방지
```

**손실함수 설계:**
```python
Loss = BCE(y, pred) + λ × cost_aware_penalty

cost_aware_penalty = 
  FN_rate × FN_cost + 
  FP_rate × FP_cost × α(temporal_context)
```

**이중 최적화:**
- **학습**: cost_aware_score 최소화 (배치 단위)
- **검증**: FAR@Recall=0.95로 임계값 결정
- **배포**: Recall ≥ 0.95 조건 하에서 FAR 최소화

#### 선행 연구

**Cost-sensitive Learning (2021, ESA):**
- 이론적 틀 제시 (비용 비대칭 인식)
- **문제**: 구현 예시 거의 없음
- **한계**: 멀티모달 + 시계열 맥락에서의 실제 적용 사례 부족

**SECOM 기반 연구들:**
- Threshold tuning (단순 ROC curve)
- **문제**: 비용 구조 반영 안 함
- **결과**: 산업 손실 구조와 무관

**차이점:**

| 항목 | 본 연구 | 선행 연구 |
|-----|-------|---------|
| 지표 | 비용민감 이중 (cost aware + FAR) | AUROC 단일 |
| 의도 | 현장 손실 최소화 | 분류 정확도 |
| 임계값 | 동적 (cost 기반) | 고정 (0.5) |
| 멀티모달 | ✅ 적용 | 대부분 미적용 |
| 실시간 성능 모니터링 | ✅ (비용 누적 추적) | ❌ |

---

### C4: SHAP + Attention 융합 설명 (Fused Explainability)

#### 본 연구

**3계층 설명 아키텍처:**

```
Level 1: 변수 기여도 (Feature Attribution)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SHAP Shapley values for each sensor
└─ "온도(PT7), 압력(6HPPRESSPV) 이 불량 기여도 상위"

Level 2: 시간대 중요도 (Temporal Attribution)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Cross-modal Attention weights over timeline
└─ "가압 초기(0~30초) vs 보온 중기(60~120초) 중 어느 구간이 결정적?"

Level 3: 물리적 맥락 (Physical Context)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Event log alignment
└─ "알람 발생 직후 온도 이상" vs 
   "설비 유지보수 후 압력 부조화"
```

**출력 예시 (cycle 단위):**
```
사이클 #1042:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
예측: 불량 (확률 0.87)
근거:
 1️⃣  온도 변수
    ├─ SHAP기여도: +0.35
    ├─ 시간: 90~120초 구간
    └─ 맥락: 베이킹 후기중 목표값 도달 실패
    
 2️⃣  압력 변수
    ├─ SHAP 기여도: +0.28
    ├─ 시간: 30~60초 구간
    └─ 맥락: 가압 초기 목표값 오버슈트
    
 3️⃣  알람 이벤트
    ├─ Alarm_code: 0x102 (High Temp Warning)
    ├─ 시간: 115초
    └─ 연관성: 온도 이상의 직접 결과로 추정

💡 추천 조치: 가열기 교정 + 온도 센서 재검
```

#### 선행 연구

**SHAP 단독 (2023, IEEE Trans. IIE):**
- 기변수 기여도 → 어떤 변수가 중요한가
- **한계**: 시간축 정보 부족
- 예: "온도가 중요하다" 만 알 수 있음
  → 사이클의 어느 시점이 결정적인지 불명

**Attention Visualization 단독 (2024~2025):**
- 시간대 / 모달별 가중치 시각화
- **한계**: 물리적 인과성 약함
- 예: "구간 A의 가중치 0.8" 
  → 왜 중요한지 현장 운영자가 이해 어려움

**DAG 기반 인과 분석 (2026, GACRI):**
- 변수 간 인과 경로 파악
- **한계**: 멀티모달 설명 안 함
- 예: "온도→압력" 경로 강조
  → 이벤트 로그의 역할 미제시

**차이점:**

| 설명 항목 | 본 연구 | SHAP | Attention | DAG |
|---------|-------|------|-----------|-----|
| **어떤 변수** | ✅ (SHAP) | ✅ | ❌ | ✅ |
| **언제 (시간대)** | ✅ (Attention) | ❌ | ✅ | ❌ |
| **왜 (물리 맥락)** | ✅ (Event Log) | ❌ | ❌ | ~ |
| **멀티모달 통합** | ✅ | ❌ | 부분 | ❌ |
| **cycle 레벨 리포팅** | ✅ | ✅ | ❌ | ❌ |
| **현장 활용도** | 높음 | 중간 | 낮음 | 낮음 |

---

## 종합 차별점 표

| 차별점 | 핵심 기여 | 선행 연구 한계 | 본 연구 해결책 |
|-------|---------|-------------|------------|
| **C1** | Press 특화 멀티모달 | 범용/도메인 미반영 | Cross-modal Attention + 물리 제약 |
| **C2** | 합성 데이터 기반 PdM | 라벨 희소 미해결 | synthpress: 10종 물리 기반 시나리오 |
| **C3** | 비용민감 이중 지표 | AUROC 단일/현장 손실 구조 무시 | cost_aware + FAR@Recall 동시 최적화 |
| **C4** | 멀티모달 설명 | 개별 설명 기법 / 통합 부재 | SHAP + Attention + Event Log 융합 |

---

## 논문의 기대 영향도

### 학술적 기여 (Academic)
1. **멀티모달 + 인과 + 비용민감 통합 프레임워크** 최초 제시
2. **도메인 특화 합성 데이터 생성기** 방법론
3. **비용민감 멀티모달 XAI** 파이프라인

### 산업 기여 (Industry)
1. **In-process 불량 예방**: 공정 중단 없이 사전 조치
2. **원가 절감**: FN 비용 감소 (폐기/리콜 방지)
3. **신뢰도 향상**: XAI로 운영자 의사결정 지원
4. **모델 일반화**: 다른 Press 공정/반도체 제조에 확산 가능

---

## 다음 단계: 검증 설계
👉 [STAGE_4_VALIDATION_STRATEGY.md](STAGE_4_VALIDATION_STRATEGY.md) 참조

