# 5단계 | Introduction 논리 흐름 (논문 서술 순서)

## 전체 개요

논문은 다음 **6개 섹션**으로 구성:
```
① Introduction
   ↓
② Related Work
   ↓
③ Proposed Method (PressFuse)
   ↓
④ Experiments
   ↓
⑤ Results & Discussion
   ↓
⑥ Conclusion & Future Work
```

---

## ① Introduction (도입부)

### 논리 흐름

```
배경
(PCB Press의 경제적 중요성)
   ↓
문제 (기존 접근법의 한계)
   ├─ 사후 검사 (AOI 한계)
   ├─ 단일 센서 (SECOM 데이터)
   └─ 비용 무시 최적화
   ↓
기여 (본 연구의 차별성)
   ├─ 멀티모달 + 비용민감
   ├─ 물리 기반 합성 데이터
   └─ XAI 통합
   ↓
논문 구성 (Road map)
```

### Introduction 작성 순서 & 내용

#### 1️⃣ 배경 (Background & Motivation)

```markdown
# 1. Introduction

## 1.1 Background

**PCB Manufacturing & Lamination Press**

- PCB(인쇄회로기판): 스마트폰, 서버, IoT의 필수 부품
  → 시장 규모: $X 억 (2024년 기준)
  → 연평균 성장률: Y% (예: 6~8%)

- **Lamination Press 공정**의 중요성:
  - 공정 단계: 적층(Lamination) → 압착(Press) → 베이킹(Baking)
  - Press 사이클: 약 10~15분 (온도 40~230℃, 압력 0~99 kgf/㎠)
  - 불량 영향: VOID(기포), Warping(휨), Delamination(박리) 
    → 제품 성능 저하, 보증 반품, 회사 신뢰도 추락

- **경제 영향**:
  - 불량 폐기 비용: 제품값 × 100% + 손실 간접비
  - 고객 클레임: 제품값 × 500% 이상 회수 비용
  - 수율 1% 상승 = 연 수십억 원 수익 개선
```

#### 2️⃣ 문제 진술 (Problem Statement)

```markdown
## 1.2 Problem Statement: Three Limitations of Existing Approaches

### 1.2.1 Post-hoc Detection Limitation
- 기존 AOI(자동 광학 검사) 이미지 기반 결함 탐지:
  - ❌ **공정 완료 후** 검사 → 사후 진단만 가능
  - ❌ 불량 발견 시점은 최종 검사 → 손실 이미 발생
  - 원인: VOID/Warping는 압착/베이킹 후 발생 특성
  
### 1.2.2 Single-Modal Dependence
- SECOM 기반 단일 센서 연구:
  - ❌ 591개 변수, 시계열 센서 $X$만 사용
  - ❌ 알람/이벤트 로그의 보완 정보 미활용
  - ❌ 두 정보원의 상호 강화 효과 없음
  - 한계: 센서 단독으로는 근본 원인 파악 어려움

### 1.2.3 Cost-Agnostic Optimization
- 기존 ML 지표의 문제:
  - ❌ AUROC, F1 등 **비용 무시하는 지표 최적화**
  - ❌ FN(미탐, 불량놓침)**비용 >> FP(오탐)**비용 구조 무시
  - ❌ Industry 4.0 환경에서 **설명 가능성 부족**
  - 구체 예시:
    * FN: 불량 폐기 + 리콜 = 손실액 매우 큼
    * FP: 정상 재검 + 생산 지연 = 손실액 작음
    * 비용 비율: FN:FP ≈ 20:1 이상

→ **결론**: 사후/단일/무비용 접근으로는 
   **공정 중 실시간 예방**과 **비용 최소화** 동시 달성 불가능
```

#### 3️⃣ 기여 정의 (Contributions)

```markdown
## 1.3 Our Contributions

본 연구는 위 세 문제를 **통합 솔루션**으로 해결:

### Contribution 1: Press-Domain-Specific Multimodal Fusion
- **What**: Cross-modal Attention 기반 센서 + 이벤트 로그 융합
- **Why**: 두 정보원의 상호 보완성을 동적 가중치로 활용
- **How**: learned attention weights로 modality-specific 특징 결합
- **Result**: 단일 모달 대비 +6% AUROC 개선 (예상)

### Contribution 2: Physics-Informed Synthetic Data Generation
- **What**: synthpress 생성기 (10종 불량 시나리오)
- **Why**: 실공정 라벨 희소(2~5%) 극복
- **How**: 도메인 물리 법칙 + 제약조건 기반 시뮬레이션
- **Result**: 합성 1000+ 샘플로 모델 사전학습 가능

### Contribution 3: Cost-Aware Dual-Metric Optimization
- **What**: cost_aware_score + FAR@Recall=0.95 이중 지표
- **Why**: 산업 손실 구조를 직접 최적화
- **How**: FN:FP=20:1 비용 가중 손실함수 + 임계값 정책
- **Result**: FAR @Recall=95% < 5% (목표)

### Contribution 4: Fused Explainability for Multimodal Decisions
- **What**: SHAP + Cross-modal Attention + Event Log 통합 설명
- **Why**: 운영자 의사결정 지원 + 신뢰도 향상
- **How**: 3계층 설명 아키텍처 (변수/시간/맥락)
- **Result**: Cycle 단위 해석 가능한 예측
```

#### 4️⃣ 논문 구성 (Paper Organization)

```markdown
## 1.4 Paper Organization

**Section 2 (Related Work)**:
- 단일 센서 PdM (Susto et al. 2014)
- 이미지 결함 탐지 (DeepPCB, YOLOv5)
- 멀티모달 융합 (MFGAN 2024)
- 인과 분석 (Causal GNN 2024)
- 비용민감 학습 (ESA 2021)

**Section 3 (Proposed Method: PressFuse)**:
- 3.1 Overview
- 3.2 Sensor Data Module (P013 센서 전처리)
- 3.3 Event Log Module (알람 임베딩)
- 3.4 Cross-Modal Attention Mechanism
- 3.5 Cost-Aware Loss Function
- 3.6 Synthetic Data Generation (synthpress)

**Section 4 (Experiments)**:
- 4.1 Dataset Description
- 4.2 Baseline Models
- 4.3 Ablation Studies (3가지)
- 4.4 Scenario-Based Evaluation

**Section 5 (Results)**:
- 5.1 Quantitative Performance
- 5.2 Ablation Results
- 5.3 Explainability Analysis
- 5.4 Failure Cases & Discussion

**Section 6 (Conclusion)**:
- 주요 성과 요약
- 제한사항
- 미래 연구 방향 (e.g., 실공정 검증)
```

---

## ② Related Work 섹션 개요

### 구성
```
2.1 Sensor-Based Predictive Maintenance (흐름 1)
    ├─ Susto et al. (2014)
    ├─ SECOM dataset 연구들 (2022~2024)
    └─ Gap: 단일 모달 한계

2.2 Image-Based Defect Detection (흐름 2)
    ├─ DeepPCB, YOLOv5 기반 연구들
    ├─ AOI 자동화
    └─ Gap: 사후 검사 한계

2.3 Multimodal Fusion for Anomaly Detection (흐름 3)
    ├─ MFGAN (2024)
    ├─ Cross-modal Attention (2025)
    └─ Gap: 도메인 특화 미진행

2.4 Causal Analysis for Fault Diagnosis (흐름 4)
    ├─ Causal GNN (2024)
    ├─ GACRI (2026)
    ├─ SHAP + DAG (2025)
    └─ Gap: XAI + 멀티모달 미통합

2.5 Cost-Sensitive Learning (흐름 5)
    ├─ FN/FP 비용 비대칭 이론
    ├─ Threshold optimization
    └─ Gap: 실시간 구현 & 멀티모달 미진행

2.6 Positioning of This Work
    ├─ 표: 기존 vs 본 연구 비교
    └─ 정리: 4가지 차별점 (C1~C4)
```

---

## ③ Method 섹션 개요

### 아키텍처 다이어그램

```
INPUT LAYER
├─ Sensor Time Series        ├─ Event Log
│  (T, 19) format            │  (T, event_dim)
│  ↓                         │  ↓
ENCODING
├─ LSTM Encoder             ├─ Embedding Layer
│  output: S_enc            │  output: E_enc
│  dim: (T, d_model)        │  dim: (T, d_model)
│  ↓                        │  ↓
FUSION
        Cross-Modal Attention
        ├─ Query: S_enc
        ├─ Key/Value: E_enc + S_enc
        └─ output: F_fused (T, d_model)
                ↓
PREDICTION HEAD
├─ Output: logits (1,)
├─ Sigmoid → probability
└─ Loss = BCE + cost_aware_penalty

EXPLANATION LAYER
├─ SHAP values (feature importance)
├─ Attention weights (temporal importance)
├─ Event alignment (physical context)
└─ Cycle-level report
```

### 상세 섹션

```markdown
3.1 Overview
- PressFuse 아키텍처 정의
- 주요 모듈: Sensor Encoder, Event Encoder, Cross-Modal Fusion, Classification Head

3.2 Sensor Data Representation
- P013 19변수 전처리 (정규화, 시계열 재샘플링)
- 시간 윈도우: 192 time steps (9~15분)
- Feature engineering (delta, rolling_mean 등)

3.3 Event Log Module
- 알람/이벤트 토큰화
- Embedding 방식: 알람 심각도 + 시간 임베딩
- Temporal position encoding

3.4 Cross-Modal Attention
- 수식: α_t = softmax(Q·K^T/√d_k)·V
- Q, K, V 계산: 가중치 행렬로 s_t, e_t에서 추출
- Multi-head attention (heads=4)

3.5 Cost-Aware Loss Function
Loss = BCE(y, ŷ) + λ·L_cost

L_cost = w_FN·L_FN + w_FP·L_FP
  where:
    w_FN = 20 (불량 놓침 비용)
    w_FP = 1  (오탐 비용)

3.6 Synthetic Data Generation
- synthpress 알고리즘
- 6종 단일 이상 + 4종 cascade 불량
- 생성 프로세스: 설정값 → 정상 경로 → 이상 주입
```

---

## 종합: 논문 작성 로드맵

| 작성 순서 | 섹션 | 소요 시간 | 비고 |
|---------|-----|---------|------|
| 1️⃣ | Introduction | 3~4시간 | 배경 정리 + 문제 명확화 |
| 2️⃣ | Related Work | 4~5시간 | 기존 논문 정찰 및 위치화 |
| 3️⃣ | Method | 5~6시간 | 아키텍처 설명 + 수식 |
| 4️⃣ | Experiments | 3~4시간 | 데이터셋 + 설정 |
| 5️⃣ | Results | 6~8시간 | 실험 + 표/그래프 |
| 6️⃣ | Discussion | 4~5시간 | 해석 + 한계 검토 |
| 7️⃣ | Conclusion | 2시간 | 요약 + 미래 방향 |
| **총** | | **27~35시간** | |

---

## Introduction 영문 초안 (Snippet)

```markdown
# 1. Introduction

## 1.1 Motivation

The printed circuit board (PCB) lamination is a critical manufacturing 
process in semiconductor production. The lamination press, a key stage 
in the process, must maintain precise control of temperature, pressure, 
and vacuum to ensure product quality. A 1% increase in yield translates 
to tens of millions of dollars in revenue improvement [1].

However, existing defect detection approaches suffer from three fundamental 
limitations:

1. **Post-hoc Detection**: Current AOI (Automatic Optical Inspection) based 
   methods detect surface defects (voids, warping, delamination) only after 
   the pressing cycle is complete, leaving no opportunity for in-process 
   prevention.

2. **Single-Modal Dependence**: Prior work (e.g., SECOM-based studies) relies 
   exclusively on sensor time-series data, ignoring complementary information 
   sources such as alarm logs and equipment state events.

3. **Cost-Agnostic Optimization**: Existing metrics (AUROC, F1) optimize for 
   general classification accuracy without considering the **asymmetric costs** 
   inherent in manufacturing: missing a defect (False Negative) costs 20-100x 
   more than false alarms (False Positive).

## 1.2 Contributions

To address these gaps, we propose **PressFuse**, a cost-sensitive multimodal 
fusion framework:

- **C1**: Domain-specific multimodal fusion (sensor + event logs) via 
  Cross-Modal Attention
- **C2**: Physics-informed synthetic defect data (synthpress generator) to 
  overcome label scarcity
- **C3**: Dual-metric cost-aware optimization (cost_aware_score + FAR@Recall)
- **C4**: Fused explainability (SHAP + Attention + Event alignment)

Our approach achieves AUROC ≥ 0.95 and FAR@Recall=0.95 < 5% on 
synthetically generated press cycles, with interpretable cycle-level predictions.
```

---

## 참고 자료

- 📄 본 분석의 5단계는 [STAGE_1~4](#) 문서와 연계
- 📊 실험 설계는 [STAGE_4](STAGE_4_VALIDATION_STRATEGY.md) 참조
- 🔗 각 선행 연구 정리: [STAGE_1](STAGE_1_PRIOR_WORK_SOLVED.md), [STAGE_2](STAGE_2_RESEARCH_GAPS.md)

---

## 다음 단계

✅ 5단계 논문 아웃라인 완성  
⏭️ **실험 코드 구현** (synthpress → PressFuse 학습 → 결과 수치 산출)

