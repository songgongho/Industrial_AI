# 🔍 논문 완성 가능성 냉정 리뷰 & 리스크 분석
## MS-CDPNet: 2026년 10월 이전 완료 시나리오

**작성일**: 2026년 5월 25일  
**논문 제출 기한**: 2026년 10월 31일 (기존 2027년 2월 28일에서 단축)  
**변경 배경**: 학위 논문 조기 완료 (1차 회의 기하급수적 단축)

---

## A. 6개 기준별 Red/Yellow/Green 평가

### 평가 체계
- 🟢 **GREEN**: 문제없음 / 실현 가능 (80% 이상 확도)
- 🟡 **YELLOW**: 주의 필요 / 조건부 가능 (50-79% 확도)
- 🔴 **RED**: 심각한 우려 / 원점 재검토 필요 (50% 미만 확도)

### 1️⃣ 연구 범위 적절성 (Scope Adequacy)

| 항목 | 평가 | 근거 | 개선 방안 |
|------|------|------|---------|
| **Causal DAG + GNN** | 🟡 YELLOW | 4-5개월에 PCMCI 튜닝 + GNN 구현은 매우 촉박 | 단순 PCMCI만 사용 (GNN은 미래 work) |
| **XAI 통합** | 🟡 YELLOW | SHAP + Attention은 복잡, 둘 중 하나만 집중 | SHAP 그래디언트에만 집중 |
| **공정 최적화** | 🔴 RED | 4-5개월 내에 추가 불가능 | **삭제** → 논문 제목에서 "최적화" 제거 |
| **멀티모달 (이미지)** | 🔴 RED | PressFuse의 이미지 모달 구현할 시간 없음 | 시계열 + 범주형만 사용 (이미지 제거) |
| **Domain Adaptation** | 🟡 YELLOW | SECOM만 가능, TEP는 불가능 | 단일 데이터셋 (SECOM) 벤치마크만 |
| **Overall Scope** | 🟡 YELLOW | 3개 Tier1만 가능, Tier2 대부분 불가능 | 핵심 3개만 Focus |

**결론**: 🟡 현재 범위는 **과도함**. **50% 축소 필요**.

---

### 2️⃣ 구현 난이도 (Implementation Difficulty)

| 구성요소 | 난도 | 평가 | 위험도 | 예상 시간 |
|---------|------|------|--------|----------|
| **PCMCI 기본 구현** | ★★ (쉬움) | 🟢 GREEN | 낮음 | 20h |
| **PCMCI 정확도 > 80%** | ★★★★ (매우 어려움) | 🔴 RED | 매우 높음 | 60-100h |
| **PyG GNN 기본** | ★★★ (중간) | 🟡 YELLOW | 중간 | 40h |
| **SHAP integration** | ★★ (쉬움) | 🟢 GREEN | 낮음 | 20h |
| **Attention visualization** | ★★ (쉬움) | 🟢 GREEN | 낮음 | 15h |
| **Auto-Report PDF** | ★★ (쉬움) | 🟢 GREEN | 낮음 | 15h |
| **웹 UI (7 screens)** | ★★★★ (매우 어려움) | 🔴 RED | 높음 | 100-150h |

**결론**: 🔴 **UI/웹 개발을 하면 불가능**. → Streamlit 단순화만 필요.

**권장 난이도 조정**:
```
현재: PCMCI 정확도 튜닝(100h) + GNN(40h) + UI(150h) = 290h (불가능)

권장: PCMCI 기본(20h) + 정확도 튜닝(40h) + SHAP(20h) + 
      Attention(15h) + Streamlit UI(30h) = 125h (가능)
```

---

### 3️⃣ 데이터셋 현실성 (Dataset Realism)

| 데이터셋 | 사용성 | 평가 | 비고 |
|---------|--------|------|------|
| **합성 데이터** | Ground truth DAG | 🟢 GREEN | 이미 구현 완료, 즉시 사용 가능 |
| **SECOM** | 벤치마크용 | 🟡 YELLOW | 공개 데이터 (시계열만, 공정 다름) |
| **DeepPCB** | 이미지용 | 🔴 RED | 멀티모달 시간 없음, 사용 불필요 |
| **TEP** | 추가 벤치마크 | 🔴 RED | 시간 없음, 생략 |
| **현실성 평가** | | 🟡 YELLOW | 합성 데이터 + SECOM만으로 충분하지만, 현실성 논증 약할 수 있음 |

**실제 문제**:
- 합성 데이터만 사용하면 "진짜 공정에서 작동하나?" 의문 제기 가능
- SECOM은 다른 공정이라 "우리 공정에 특화되지 않음" 비판 가능
- **해결책**: 합성 데이터의 physics 타당성을 논문에서 강조

---

### 4️⃣ 실험 재현 가능성 (Experiment Reproducibility)

| 항목 | 현황 | 평가 | 비고 |
|------|------|------|------|
| **코드 버전 관리** | Git + CI/CD | 🟢 GREEN | GitHub Actions 준비됨 |
| **의존성 명시** | requirements.txt | 🟢 GREEN | 관리됨 |
| **데이터 재현** | DVC + 합성 데이터 | 🟡 YELLOW | 합성 데이터 시드 설정 필요 |
| **모델 체크포인트** | MLflow 로깅 | 🟡 YELLOW | MLflow 구성은 되어 있으나 사용 미흡 |
| **하이퍼파라미터** | Hydra config | 🟢 GREEN | 준비됨 |
| **논문 보충 자료 (Appendix)** | 미계획 | 🔴 RED | 모든 실험 설정 공개 필요 |

**결론**: 🟡 기본은 있으나, **논문 제출까지 Appendix 정리 필수**.

---

### 5️⃣ 논문 기여도 명확성 (Paper Contribution Clarity)

| 기여도 항목 | 강도 | 평가 | 위험도 |
|-----------|------|------|--------|
| **1. Problem Definition** | 중간 | 🟡 YELLOW | 공정 도메인 잘 정의되어 있음 (강점) |
| **2. Causal Discovery (PCMCI)** | 낮음 | 🔴 RED | PCMCI는 이미 알려진 기법, "우리가 뭘 새로 했나?" 질문 예상 |
| **3. GNN propagation** | 중간-높음 | 🟡 YELLOW | 인과 DAG + GNN은 신선하나, 평가 지표 부족 (propagation accuracy 정의 필요) |
| **4. XAI (SHAP + Attention)** | 중간 | 🟡 YELLOW | SHAP/Attention은 표준 기법, 통합이 주 기여 |
| **5. Cost-sensitive metrics** | 낮음 | 🟡 YELLOW | 이미 metrics.py에 구현됨, 신규 기여 아님 |
| **Overall Novelty** | | 🟡 YELLOW | **"인과 그래프 + GNN을 공정 불량 예측에 처음 적용"이 주 기여** (이미 많은 논문에서 유사 아이디어 있음) |

**문제점**:
- 명확한 "우리만의 혁신" 이 부족
- PCMCI, GNN, SHAP 모두 기존 기법
- "처음 조합했다" 정도의 기여는 학위논문에는 약할 수 있음

**개선책**:
- **새로운 인과 속성 정의** (e.g., "공정 변수 간 인과 강도의 시간 변화 추적")
- **새로운 평가 지표** (e.g., "불량 전파 정확도를 Causal intervention으로 평가")
- **실무 적용 사례** (e.g., "3개월간 실제 공정에 적용하여 수율 5% 개선")

---

### 6️⃣ 시각화/UI 데모 차별성 (Visualization & Demo Differentiation)

| 항목 | 현황 | 평가 | 비고 |
|------|------|------|------|
| **DAG 시각화** | Networkx + Plotly | 🟡 YELLOW | 표준. 더 있어야 함: "DAG 신뢰도" 시각화 (엣지 색상 = confidence) |
| **SHAP 시각화** | SHAP force plot | 🟡 YELLOW | 표준. 새로운 아이디어: "인과 DAG 상에 SHAP 값 오버레이" |
| **Attention 히트맵** | Plotly heatmap | 🟡 YELLOW | 표준. 시간축 특수성 강조 필요 |
| **웹 UI/대시보드** | Streamlit (계획) | 🔴 RED | Streamlit은 좋지만, 7개 화면 개발 = 4-5개월 소요 → 불가능 |
| **차별성 평가** | | 🔴 RED | 대부분 표준 시각화, 혁신적인 UI 없음 |

**권장 개선**:
1. **Streamlit Simple** (3개 탭만: Upload/Causal DAG/Predictions)
2. **DAG 신뢰도 표시** (엣지 색상, 숫자 레이블 = p-value 또는 confidence)
3. **"Causal SHAP"**: DAG 경로 위에 SHAP 값 오버레이 (새로운 아이디어!)
4. **Jupyter 데모 Notebook** (상세 분석용, UI 대신)

---

## B. 평가 결과 요약표

```
┌─────────────────────┬────────┬───────────────────┬──────────┐
│ 평가 기준            │ 평가   │ 신뢰도            │ 개선필요 │
├─────────────────────┼────────┼───────────────────┼──────────┤
│ 연구 범위 적절성     │ 🟡    │ 60% (과도함)      │ YES     │
│ 구현 난이도          │ 🟡    │ 55% (UI 제거 필요) │ YES     │
│ 데이터셋 현실성      │ 🟡    │ 65% (다만 논증 약) │ YES     │
│ 실험 재현 가능성     │ 🟡    │ 70% (Appendix 필요)│ YES    │
│ 논문 기여도 명확성   │ 🟡    │ 50% (혁신 약함)    │ **YES**  │
│ 시각화/UI 차별성     │ 🔴    │ 40% (표준만)       │ **YES**  │
├─────────────────────┼────────┼───────────────────┼──────────┤
│ **전체 평가**        │ 🟡    │ **57% (조건부 가능)│ 긴급    │
└─────────────────────┴────────┴───────────────────┴──────────┘
```

**최종 평가**: 
🟡 **"현재 범위와 구성으로는 불가능에 가까움. 범위 50% 축소 + 기여도 명확화 필수"**

---

## C. 가장 위험한 TOP 5 리스크 & 대응책

### 리스크 평가 기준
- **발생확률**: Low (< 30%) / Medium (30-60%) / High (> 60%)
- **영향도**: Low (지연만) / Medium (부분 손상) / Critical (프로젝트 실패)
- **Risk Score** = 발생확률 × 영향도 (최대 10)

### TOP 5 위험도 순서

| 순위 | 리스크 명 | 발생확률 | 영향도 | Score | 현재 상태 | 대응책 |
|------|---------|--------|--------|-------|---------|--------|
| **1** | **PCMCI 정확도 < 70%** (인과 학습 실패) | **HIGH (70%)** | **CRITICAL** | **9.8/10** | 🔴 매우 높음 | 1) Week 1: 즉시 프로토타입 (합성 DAG 테스트) 2) 정확도 < 75%면 Method 변경 (FCI, LiNGAM) 검토 3) 최악: PCMCI 정확도로 평가하지 말고 "interpretability" 강조로 전환 |
| **2** | **시간 부족** (4개월 → 3개월로 단축) | **MEDIUM-HIGH (65%)** | **CRITICAL** | **9.3/10** | 🔴 매우 높음 | 1) 범위 50% 즉시 축소 (공정 최적화, 이미지, TEP 제거) 2) 기능별 우선순위 엄격히 (RED 항목만) 3) 주당 2주에 1 산출물 강제 (스프린트 엄격화) |
| **3** | **GNN 학습 수렴 실패** (Propagation accuracy < 60%) | **MEDIUM (50%)** | **CRITICAL** | **7.5/10** | 🟡 중간 | 1) GNN 필수 아님 (Tier2로 강등) 2) 단순 rule-based propagation으로 대체 가능 (if PCMCI DAG, then BFS) 3) 최악: GNN 삭제, PCMCI DAG만 논문화 |
| **4** | **논문 기여도 낮음** (Reviewer 지적: "기존 기법 조합일 뿐") | **HIGH (60%)** | **CRITICAL** | **8.1/10** | 🔴 높음 | 1) **새로운 인과 속성 정의** (e.g., "temporal causal strength") 있어야 함 2) 새로운 평가 지표 제안 (e.g., "propagation F1-score") 3) 실무 임팩트 강조 (if 실제 공정 데이터 있으면) 4) 마지막 수단: "Industrial AI application to semiconductor" 각도로 포지셔닝 |
| **5** | **합성 데이터 신뢰도 낮음** (Reviewer: "Toy data로는 실용성 증명 안 됨") | **HIGH (60%)** | **MEDIUM** | **6.8/10** | 🟡 높음 | 1) 합성 데이터 물리 타당성 명시 (논문 Section 3에 상세 기술) 2) real SECOM data 추가로 검증 3) 불가능하면: "이 방법론은 다양한 센서 config에 적용 가능함" 강조 4) 향후 연구 (future work): "실제 제조 환경에서 검증" 명시 |

---

## D. 상위 3개 리스크에 대한 상세 대응 계획

### 🔴 리스크 1: PCMCI 정확도 문제 (최우선)

**심각도**: 🔴 **CRITICAL** (논문 전체 기여도 좌우)

**상황 분석**:
- PCMCI는 매개변수 민감함 (tau, alpha, method 등)
- 합성 DAG가 단순하면 PCMCI가 쉽게 100% 찾음 (의미 없음)
- 합성 DAG가 복잡하면 80% 이상 정확도 달성 어려움
- 실제 공정은 더 복잡 → 50-60% 정도만 예상

**Week 1-2 집중 액션**:

| 일정 | 액션 | 목표 | 성공 기준 |
|------|------|------|---------|
| **Day 1-2** | 합성 DAG 복잡도 에스컬레이션 | Simple → Complex 단계별 테스트 | 각 단계별 정확도 기록 |
| **Day 3-5** | PCMCI 매개변수 스윕 | tau, alpha, condind_run_mode 조합 테스트 | 최적 정확도 79% 이상 찾기 |
| **Day 6-7** | 대체 방법 사전 준비 | FCI, LiNGAM, VARLiNGAM 기본 구현 | 각 방법 1시간 테스트 |
| **Day 8-10** | 정확도 평가 기준 재정의 | Precision, Recall, F1 계산 (adjacency + edge direction) | 성능 기록 |
| **Day 11-14** | Contingency Plan 결정 | PCMCI 정확도 목표 재설정 (80% vs 75% vs 65% vs "interpretability만") | 모든 경우의 논문 기여도 재평가 |

**Contingency 시나리오**:

```
Scenario A (Best Case): PCMCI precision/recall > 80%
    → 논문 제목 유지 (MS-CDPNet)
    → GNN propagation 구현 진행

Scenario B (Good Case): PCMCI precision/recall 70-80%
    → 논문: "Causal discovery에서 trade-off: more edges false positive BUT fewer false negative"로 정당화
    → 논문 제목 유지

Scenario C (Acceptable Case): PCMCI precision/recall 60-70%
    → 논문 위치: "interpretability" 강조 (정확도 대신 설명 가능성)
    → 제목 변경 고려: "Interpretable Defect Propagation via Causal Discovery"

Scenario D (Worst Case): PCMCI precision/recall < 60%
    → GNN 삭제, PCMCI DAG 시각화만 제시
    → 제목 변경: "Causal Structure Learning for PCB Process Anomaly Detection"
    → 기여도: "처음 시도한 공정-특화 인과 탐색" 강조
```

---

### 🔴 리스크 2: 시간 부족 (3개월 대폭 단축)

**심각도**: 🔴 **CRITICAL** (전체 일정 불가능)

**시간 분석**:

```
기존 계획 (9개월):
- Sprint 1: 110시간 (8주)
- Sprint 2: 180시간 (12주)
- Sprint 3: 140시간 (10주)
  Total: 430시간

새 계획 (4개월 = 16주):
- 총 가능 시간: 16주 × 8시간/주 = 128시간
- 필수 시간: Sprint 1 핵심만 = 약 100시간
- 여유: 28시간

현실: 128시간은 이상적 가정 (실제 50-60시간 정도만 가능)
```

**대응책 (범위 감축)**:

| 삭제 대상 | 기존 시간 | 삭제 이유 | 대체 방안 |
|---------|---------|---------|---------|
| **공정 최적화 (Surrogate model)** | 60h | 구현 복잡, 논문 핵심 아님 | Future work로 동연 |
| **이미지 멀티모달 (PressFuse)** | 50h | 데이터 없음, 시간 없음 | 시계열만 사용 |
| **GNN propagation** | 80h | 시간 부족 시 | 단순 rule-based propagation (10h) |
| **웹 UI (7 screens)** | 120h | 구현 시간 너무 많음 | Streamlit 2-3 탭으로 단순화 (20h) |
| **Domain Adaptation (TEP)** | 40h | 시간 부족 | 단일 dataset (SECOM) 컷 |
| **Advanced visualization** | 30h | 시간 부족 | 기본 Plotly만 사용 |
| **XAI 비교 (SHAP vs LIME)** | 40h | 시간 부족 | SHAP만 구현 |
| **합계 삭제** | **420시간** | | **필수만 100시간 내** |

**새로운 일정** (4개월 = 17주):

```
2026년 6월 1주 ~ 10월 4주

Week 1-3 (6월 1-21): Sprint 1-A (PCMCI + EDA) = 40시간
├─ PCMCI 프로토타입 (20h)
├─ 합성 데이터 검증 (10h)
└─ Baseline model 학습 (10h)

Week 4-6 (6월 22-7월 12): Sprint 1-B (SHAP + Attention) = 30시간
├─ SHAP 그래디언트 (15h)
├─ Attention viz (10h)
└─ Streamlit UI 기본 (5h)

Week 7-10 (7월 13-8월 9): Sprint 2 (GNN 또는 Rule-based) = 25시간
├─ GNN / Rule-based propagation (20h)
└─ 통합 파이프라인 (5h)

Week 11-14 (8월 10-9월 6): Sprint 3-A (실험 + 벤치마크) = 20시간
├─ SECOM 벤치마크 (10h)
├─ Ablation study (5h)
└─ 결과표 정리 (5h)

Week 15-17 (9월 7-28): Sprint 3-B (논문 작성) = 15시간
├─ 논문 Methods (5h)
├─ Results + Discussion (5h)
├─ Appendix (3h)
└─ 최종 교정 (2h)

Week 18-19 (9월 29-10월 11): 버퍼 + 최후 점검 = 10시간

총 시간: 130시간 (가능 범위 내)
```

---

### 🔴 리스크 3: 논문 기여도 제한 (혁신 약함)

**심각도**: 🔴 **CRITICAL** (심사위원 지적 가능)

**근본 원인**:
- PCMCI = 기존 기법
- GNN = 기존 기법
- SHAP = 기존 기법
- 조합만 새로움 → 학위 논문 수준으로는 약함

**개선책** (3가지 축):

#### 축 1: 새로운 인과 평가 지표 제안

**아이디어**: "Causal Motif" 개념 도입

```
기존: "DAG precision/recall만으로 평가"

새로운: "Causal Motif 발생 빈도로 평가"

Causal Motif = 공정 상황에서 반복되는 인과 구조 패턴
예시:
  - Motif 1: Pressure ↓ → Vacuum ↑ → PT ↓ (cascade)
  - Motif 2: Temp ↓ → Temp unstable → Defect (direct)
  - Motif 3: Equipment fault → All variables affected (global)

논문 기여:
  1) Motif 분류체계 제안 (새로운 개념!)
  2) 각 motif의 불량 영향도 정량화
  3) 공정 특성에 맞는 motif 우선순위
```

#### 축 2: 새로운 평가 메트릭 제안

**아이디어**: "Causal Intervention F1-score"

```
기존 평가:
  - AUROC, FAR@Recall (binary classification metrics)

새로운 평가 (論文에서 처음 제안):
  - "Estimated Causation" (EC) Score
  - 공식: EC = Σ(propagated_defect - actual_defect)^2 최소화
  - 물리적 의미: DAG 기반 전파 예측이 실제 불량과 얼마나 일치하는가?

논문 기여:
  1) Industrial-specific metric 제안
  2) 이 metric을 통해 기존 metrics보다 더 타당한 평가 가능
  3) "Causal propagation"을 직접 평가하는 유일한 metric
```

#### 축 3: 실무 적용 임팩트 강조

**만약 실제 공정 데이터 있으면**:

```
Case A (현재): 합성 데이터만
  → 논문: "Proof-of-concept로서 synthetic dataset에서 method 검증"
  → 약점: "Real deployment는 어떨지 모름"

Case B (이상적): 1개월 실제 공정 데이터 통합
  → 논문: 
    Section 1: Synthetic experiment (검증)
    Section 2: Real SECOM data (일반화)
    Section 3: 공정 특화 분석 (novelty)
  → 강점: "이미 2개 데이터셋에서 검증됨"
```

**현실 대안**: 가능하면 SECOM과 합성 데이터 **정량적 비교** 표 추가

```
Table: Comparison of Synthetic vs SECOM
┌─────────────────┬──────────┬──────────┐
│ Property        │ Synthetic│ SECOM    │
├─────────────────┼──────────┼──────────┤
│ Sample count    │ 10K      │ 1.6K     │
│ Features        │ 19       │ 590      │
│ Defect rate     │ 3%       │ ~5%      │
│ Temporal length │ 192      │ Variable │
│ AUROC (our     │ 0.98     │ 0.92     │
│  method)        │          │          │
└─────────────────┴──────────┴──────────┘

(이것만으로도 "method가 다양한 데이터에서 동작함" 증명)
```

---

## E. 새로운 일정표 (4개월: 2026년 6월 ~ 10월)

### 개요

```
┌─────────────────────────────────────────────────────────────┐
│           4개월 Compressed Sprint Schedule                  │
│                2026년 6월 1 ~ 10월 31                       │
└─────────────────────────────────────────────────────────────┘

2026-06        2026-07        2026-08        2026-09        2026-10
  │              │              │              │              │
  ├──Sprint 1A───┤──Sprint 1B───┤──Sprint 2 ────┤──Sprint 3────┤
  │ PCMCI        │ SHAP+Attent  │GNN/Propagation│Experiments   │
  │ + EDA        │ + StreamUI   │ + Integration │+ 논문 작성   │
  │ (40h)        │ (30h)        │ (25h)         │ (35h)        │
  │              │              │               │              │
  ✅ DAG         ✅ Figure      ✅ Propagation ✅ Complete     → 📄
  ✅ Results     ✅ UI          ✅ Pipeline      Draft         2026
  ✅ 실험 기초   ✅ SHAP plots  ✅ Benchmark
```

### 상세 마일스톤 (주 단위)

| 주 | 기간 | Sprint | 주요 산출물 | 예상 시간 | Go/No-Go |
|----|------|--------|-----------|---------|---------|
| **1-3** | 06/01-06/21 | **1A** | PCMCI 프로토타입 + DAG 시각화 + 합성 EDA | 40h | 🟢 GO |
| **4-6** | 06/22-07/12 | **1B** | SHAP 그래디언트 + Attention viz + Streamlit기본 | 30h | 🟢 GO |
| **7-10** | 07/13-08/09 | **2** | GNN 또는 Rule-based propagation + 통합 | 25h | 🟢 GO |
| **11-14** | 08/10-09/06 | **3-A** | SECOM 벤치마크 + Ablation + 결과표 | 20h | 🟡 CHECK |
| **15-17** | 09/07-09/28 | **3-B** | 논문 Methods+Results draft | 15h | 🟡 CHECK |
| **18-19** | 09/29-10/11 | **Finalize** | 버퍼 + 최종 교정 + Appendix | 10h | 🟡 CHECK |
| **20** | 10/12-10/31 | **Reserved** | 예비 시간 (발생 이슈 대응) | - | 🔴 BUFFER |

---

### 주별 상세 태스크 (Week 1-3: PCMCI 집중)

**Week 1 (6월 1-7)**

실제 작업: Mon-Fri 08:00-18:00

```
Day 1 (Mon, 6/1):
  ✓ PCMCI 라이브러리 최종 선택 (causalml, DoWhy, castle 비교 완료)
  ✓ 합성 DAG 최종 정의 (variables, edges)
  ✓ GitHub repo branch 생성: feature/causal-discovery

Day 2 (Tue, 6/2):
  ✓ PCMCI wrapper 기본 구조 (src/causal/pcmci_wrapper.py skeleton)
  ✓ 테스트 케이스 3개 작성

Day 3 (Wed, 6/3):
  ✓ PCMCI 매개변수 기본값 실험
  ✓ 첫번째 정확도 기록 (baseline)

Day 4 (Thu, 6/4):
  ✓ 정확도 < 75%면 매개변수 조정 시작
  ✓ 3가지 설정 비교

Day 5 (Fri, 6/5):
  ✓ Week 1 정리 + Friday review meeting
  ✓ 정확도 best 기록 저장
  ✓ Week 2 계획 확정
```

---

## F. 현실적 완성 시나리오 (3가지 경로)

### Scenario 1: BEST CASE (현재 계획 대로)

**조건**:
- PCMCI 정확도 ≥ 75% 달성
- GNN 학습 수렴
- 시간 예상 대로

**결과**:
- 논문 페이지: 6-7 pages
- Figure: 12-15개
- Tables: 8-10개
- 기여도: "Causal + GNN + XAI 통합"
- 심사: 중상 수준
- **가능성: 35%**

---

### Scenario 2: REALISTIC CASE (범위 축소, 혁신 강조)

**조건**:
- PCMCI 정확도 60-75% (충분히 좋음)
- GNN 삭제 또는 Rule-based로 대체
- 새로운 평가 지표 추가 (Causal Motif, EC score)

**결과**:
- 논문 페이지: 5-6 pages
- Figure: 10-12개
- Tables: 6-8개
- 기여도: "공정 특화 인과 탐색 + 새로운 평가 메트릭"
- 심사: 중 수준 (하지만 완성도 높음)
- **가능성: 55%** ← 현실적 목표

---

### Scenario 3: WORST CASE (최소 기준)

**조건**:
- PCMCI 정확도 < 60% (해석 가능성만 강조)
- GNN, 공정 최적화 모두 삭제
- 기존 LSTM 벤치마크 추가

**결과**:
- 논문 페이지: 4-5 pages
- Figure: 8-10개
- Tables: 4-6개
- 기여도: "공정 불량 예측에 인과 탐색 적용 (첫 시도)"
- 심사: 중-하 수준 (하지만 완성됨)
- **가능성: 85%** ← 최악의 경우도 대비하자는 뜻

---

## G. 즉시 실행 최후 체크리스트 (이번 주 필수)

### RED ZONE: 반드시 이번 주 안에 결정해야 할 것

```
[ ] 1. 논문 제목 최종 확정
      기존: "Multi-Stage Causal Defect Propagation Network"
      후보: "Causal Discovery and Explainable Anomaly Detection in PCB Process"
      선택: ________________

[ ] 2. Tier 1 (필수) vs Tier 2 (선택) 최종 재분류
      필수: PCMCI + SHAP + Streamlit basic
      선택: GNN (고려중) / 공정 최적화 (삭제) / 이미지 (삭제)

[ ] 3. 새로운 기여도 정의
      추가 아이디어 1: "Causal Motif" / "EC score" / 기타?
      추가 아이디어 2: ________________

[ ] 4. 시간 할당 최종 (130시간 배분)
      PCMCI + EDA: ___h
      SHAP + Attention: ___h
      GNN / Propagation: ___h
      Experiments: ___h
      Paper writing: ___h

[ ] 5. 지도교수 회의 일정
      제안 날짜: 2026년 6월 1일 10:00?
      또는: ________________

[ ] 6. 위험 대응 담당자
      PCMCI 정확도 모니터링: [You]
      시간 관리: [You]
      기여도 논증: [You] + 지도교수 자문
```

---

## H. 최종 결론

### 냉정한 평가

**현 상황**:
- 🟡 조건부 가능 (55% 가능성)
- 🔴 5개 심각한 리스크 (1주 안에 대응책 구현 필수)
- 🟡 기여도 명확화 필수 (새로운 아이디어 추가)
- 🟡 범위 50% 축소 필수

**성공 확률**:

```
최선:  80% (Scenario 1, 모든 것이 계획대로)
현실:  55% (Scenario 2, 범위 축소 + 혁신 추가) ← 목표
최악:  25% (Scenario 3도 실패, 논문 미완성)
```

### 권고사항

1. **긴급** (이번 주):
   - PCMCI 프로토타입 (Week 1-3)
   - 새로운 기여도 아이디어 2-3개 구체화
   - 범위 최종 결정

2. **중요** (이달):
   - 주 단위 진도 관리 (엄격)
   - 리스크 1-3 집중 모니터링
   - Contingency plan 준비

3. **시스템**:
   - 주간 진도 회의 (매주 금요일)
   - 월간 위험 검토 (매월 첫 월요일)
   - 산출물 제출 일정 고정

---

**최종 판정**: 🟡 **"가능하지만, 완벽한 실행 필수. 지금부터 한 주라도 낭비하면 불가능"**


