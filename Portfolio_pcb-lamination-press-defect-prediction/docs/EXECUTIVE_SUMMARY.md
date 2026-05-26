# 📋 실행 요약 (Executive Summary)

## 현 상황 진단

### 한 줄 평가
**원점-확인 단계(POC 완성), 논문 핵심인 인과 그래프 + XAI 심화 + 공정 최적화 로직이 부재하므로, 9개월간 3개 영역에 집중하되, 2주마다 논문용 아티팩트를 생산하는 "스프린트 중심" 개발 필수**

---

## 강점 ✅

| # | 항목 | 상태 | 설명 |
|---|------|------|------|
| 1 | PyTorch/Lightning 기초 | ✅ 완성 | 5/5 (프로덕션 레벨) |
| 2 | 시계열 모델 (PressFuse) | ✅ 완성 | 4/5 (Cross-modal attention 있음) |
| 3 | 합성 데이터 생성 | ✅ 완성 | 4/5 (6종 + 4종 multi-anomaly) |
| 4 | 평가 지표 (FAR@Recall) | ✅ 완성 | 4/5 (도메인 특화 지표) |
| 5 | 웹 UI (Streamlit) | 🟡 Proto | 2/5 (기본 구조만) |
| 6 | 코드 품질 (tests, type hints) | ✅ 우수 | 75% test coverage, Black+Ruff |

---

## 약점 🔴

| # | 영역 | 현황 | 위험도 | 논문 영향도 |
|---|------|------|--------|-----------|
| 1 | **인과 그래프 학습** (PCMCI/NOTEARS) | 0% | 🔴 매우 높음 | 🔴 핵심 |
| 2 | **XAI 구현** (SHAP + Attention) | 5% (스켈레톤) | 🔴 높음 | 🔴 핵심 |
| 3 | **공정 최적화** (Differentiable surrogate) | 0% | 🟠 중간 | 🟠 중요 |
| 4 | **자동 리포트** (PDF 생성) | 0% | 🟡 낮음 | 🟠 중요 |
| 5 | **고급 UI/UX** (React/Dash) | 0% | 🟡 낮음 | 🟡 보조 |

---

## 논문 시간과 리소스 계획

### 마일스톤 (2026.6 ~ 2027.2, 9개월)

```
2026-06     2026-09     2026-12     2027-02
  │            │            │          │
  ├──Sprint 1──┤  ├──Sprint 2──┤  ├──Sprint 3──┤
  │  POC검증   │  │  핵심기여  │  │ 최적화+논문│
  │  (3개월)   │  │  (3개월)   │  │  (3개월)   │
  │            │  │            │  │            │
✅ PCMCI선택  ✅ GNN구현   ✅ Benchmark  → 📄 논문제출
✅ DAG학습   ✅ SHAP+Att  ✅ 다중공정    2027.02.28
✅ 모델학습   ✅ 설명파이프 ✅ XAI비교
✅ EDA완성    ✅ 자동보고   ✅ 최적화완성
```

### 인력 및 시간 분배 (혼자 진행 시)

| Sprint | 총 시간 | 핵심 작업 | 기술 리스크 |
|--------|---------|---------|-----------|
| **Sprint 1** | ~110시간 (3주) | PCMCI, EDA, 시각화 | 낮음 (라이브러리 선택) |
| **Sprint 2** | ~180시간 (5주) | GNN, SHAP, 최적화 | 높음 (튜닝 곡선) |
| **Sprint 3** | ~140시간 (4주) | 벤치마크, 논문 마무리 | 중간 (시간 압박) |
| **합계** | **~430시간** | | **평균 2주당 3-4 full days** |

---

## 즉시 실행 항목 (이번 주)

### Top 5 액션 (2주 안에 완료해야 함)

| 우선 | 작업 | 예상 시간 | 완료 기준 |
|------|------|---------|---------|
| 🔴 1순위 | PCMCI 라이브러리 선택 (3개 비교) | 4시간 | 선택 결정 + 테스트 |
| 🔴 2순위 | ground truth DAG 정의 | 6시간 | YAML 파일 + 검증 |
| 🟠 3순위 | `src/causal/` 폴더 구축 | 3시간 | GitHub commit |
| 🟠 4순위 | PCMCI wrapper 구현 | 8시간 | `tests/test_pcmci.py` pass |
| 🟡 5순위 | 논문 Introduction draft | 4시간 | 1.5페이지 |

**이주 목표**: PCMCI 프로토타입 동작 확인 + GitHub commit 3개 이상

---

## 시스템 아키텍처 한 줄 요약

```
React Frontend → FastAPI Backend → PyTorch/PyG ML Core → PostgreSQL+S3 Storage
│데이터 시각화   │REST API+아키텍처 │모델/인과/설명      │메타+체크포인트
└─→ 7개 웹 화면 (Upload, Preprocess, Causal DAG, Predictions, Explanations, Propagation, Reports)
```

### 기술스택 선택 근거

| 계층 | 추천 | 이유 | 대안 |
|------|------|------|------|
| **Frontend** | React + TypeScript | 풍부한 생태계, 회사 표준 | Vue, Svelte |
| **Backend** | FastAPI + Pydantic | 빠른 개발, 자동 문서화, async 네이티브 | Django, Flask |
| **ML Core** | PyTorch + PyG | GNN 연구 표준 (SOTA 논문들) | DGL, Graph Nets |
| **XAI** | SHAP + Captum | 산업계 신뢰도, 논문 기준 | LIME, Integrated Grad |
| **Causal** | causalml/DoWhy | 풍부한 알고리즘, 문서화 | castle, networkx |
| **DB** | PostgreSQL + Redis | 신뢰성, 복잡 쿼리, 오픈소스 | MongoDB, MySQL |

---

## 논문 섹션별 진도 예測

| 섹션 | 현재 | Sprint 1 | Sprint 2 | Sprint 3 | 최종 | 난도 |
|------|------|---------|---------|---------|------|------|
| 1. Introduction | 0% | 30% | 70% | 95% | 100% | ★ |
| 2. Related Work | 0% | 10% | 50% | 100% | 100% | ★ |
| 3. Method: Causal DAG | **0%** | **30%** | **80%** | **100%** | **100%** | ★★★★★ |
| 4. Method: GNN Propagation | **0%** | **0%** | **70%** | **100%** | **100%** | ★★★★ |
| 5. Method: XAI | **5%** | **15%** | **80%** | **100%** | **100%** | ★★★ |
| 6. Experiments | 0% | 20% | 60% | 100% | 100% | ★★★ |
| 7. Results & Ablation | 0% | 5% | 40% | 100% | 100% | ★★ |
| 8. Discussion | 0% | 0% | 10% | 80% | 100% | ★★ |
| 9. Conclusion | 0% | 0% | 0% | 50% | 100% | ★ |
| **전체 논문** | **0.5%** | **12%** | **50%** | **95%** | **100%** | |

**목표**: 2027년 1월 31일 전체 draft 완성 → 2월 교수 피드백 반영 → 2월 말 제출

---

## 논문 기여도 분류

### Tier 1: 논문 제목에 직결 (반드시 구현) 🔴

1. **Multi-Stage Causal DAG** (PCMCI/NOTEARS)
2. **Defect Propagation Prediction** (GNN)
3. **XAI: SHAP + Attention Integration**
4. **Cost-Sensitive Evaluation** (FAR@Recall)

### Tier 2: 논문 강점 추가 (여유 있으면) 🟠

1. XAI 비교연구 (SHAP vs LIME vs IG)
2. Foundation Model 벤치마크 (Chronos)
3. Advanced Causal Methods (DoWhy)
4. Domain Adaptation (SECOM/TEP)

### Tier 3: 포트폴리오/후속 (논문 이후) 🟡

1. Digital Twin (SimPy)
2. ONNX + TensorRT (Edge AI)
3. Federated Learning PoC
4. VLM 기반 이미지 분석

---

## UI/UX 화면 설계 개요

### 7개 핵심 화면

```
1️⃣ Data Upload
   ↓ 파일 선택 + 스키마 매핑
   
2️⃣ Preprocessing
   ↓ 정규화 + 특성 선택
   
3️⃣ Causal DAG
   ↓ PCMCI 학습 + 시각화 (Cytoscape)
   
4️⃣ Predictions Dashboard
   ↓ 모델 예측 + 임계값 조절 (ag-Grid)
   
5️⃣ SHAP/Attention Explanation
   ↓ 불량 원인 설명 (Plotly heatmap)
   
6️⃣ Propagation Tracer
   ↓ 인과 경로 추적 + 시나리오 시뮬레이션
   
7️⃣ Auto-Report Generator
   ↓ PDF/HTML/XLSX 다운로드
```

각 화면별:
- 입력/출력 명확
- 백엔드 API 초안
- 추천 라이브러리 (Plotly, ag-Grid, Cytoscape, Recharts)
- UX 주의사항

**총 개발 일정**: Sprint 2-3에서 진행 (현재 Sprint 1은 ML 핵심에 집중)

---

## 위험 관리 매트릭스

### 기술적 리스크

| 리스크 | 발생확률 | 영향도 | 방어 전략 | 담당 | 검토일 |
|--------|-------|--------|---------|------|--------|
| PCMCI 정확도 < 70% | 40% | 높음 | 조기 프로토타입 (Week 1) | [You] | 2026.06.15 |
| GNN 학습 수렴 실패 | 25% | 높음 | 베이스라인 구축 (LSTM만?) | [You] | 2026.09.15 |
| SHAP 계산 시간 > 5초/cycle | 20% | 중간 | GPU 최적화, 캐싱 | [You] | 2026.10.15 |
| 데이터 불균형 > 99:1 | 60% | 중간 | Weighted loss + SMOTE | [You] | 2026.07.01 |
| 시간 부족 (430시간) | 35% | 높음 | 우선순위 엄격히, 자동화 | [You] | 매월 |

### 관리 및 커뮤니케이션 리스크

| 리스크 | 방지책 |
|--------|--------|
| 지도교수 피드백 지연 | 월 1회 정기 미팅 일정 고정 |
| 요구사항 변경 (scope creep) | 각 Sprint 시작 전 명세서 확정 |
| 팀원 이탈 (혼자 진행 시) | Self-motivation, 커뮤니티 참여 (GitHub Discussions) |
| 환경 설정 문제 | Docker 자동화, requirements.txt 명확화 |

---

## 성공 지표 (Key Metrics)

### 분기별 성공 기준

**분기 1 (6월-8월)**
- [ ] PCMCI 모듈 단위 테스트 pass rate ≥ 90%
- [ ] 합성 데이터 KS test p-value > 0.05
- [ ] PressFuse AUROC ≥ 0.95 (합성 데이터)
- [ ] 논문 Introduction + Related Work draft 완성

**분기 2 (9월-11월)**
- [ ] GNN 불량 전파 정확도 ≥ 85%
- [ ] SHAP + Attention 통합 파이프라인 동작
- [ ] 논문 Figure 15개 이상 완성
- [ ] Methods 섹션 draft (3-4 페이지)

**분기 3 (12월-2월)**
- [ ] 3개 모델 벤치마크 완성
- [ ] AUROC ≥ 0.98, FAR < 5% (recall=0.95)
- [ ] 논문 전체 초안 완성
- [ ] 논문 제출 및 방어 준비

---

## 다음 회의 아젠다

### Sprint 1 킥오프 미팅 (예정: 2026.06.01)

**10분 발표**:
1. 로드맵 개요 설명
2. 우선순위 Tier 설명
3. Sprint 1 목표 (8주, 110시간)

**15분 토론**:
- PCMCI 라이브러리 선택안 논의
- 위험 항목 협의 (GNN 시간 vs 논문 일정)
- 주간 보고 형식 정의

**5분 확정**:
- 첫 주 작업 일정 고정
- 지도교수 미팅 일정 확정

---

## 참고: 상세 문서 위치

생성된 종합 문서: **`docs/RESEARCH_ROADMAP_2026.md`**

내용:
- A. 한 줄 총평
- B. 현재 준비 수준 진단 (SWOT 분석 포함)
- C. 논문 직결 우선순위 표
- D. 3단계 로드맵 (3개월 x 3단계, 50+ 액션 아이템)
- E. 개발 백로그 (High/Med/Low 총 15개 항목)
- F. UI/UX 화면 설계 (7개 화면, 각 100+ 줄)
- G. 시스템 아키텍처 (Frontend/Backend/ML/Storage 상세)
- H. 다음 스프린트 Top 10 액션 (체크리스트 포함)
- I. 결론 및 권고사항

---

**작성**: 2026년 5월 25일  
**버전**: 1.0 (Initial Draft)  
**상태**: 🔴 피드백 대기 중 (지도교수 검토)  
**다음 업데이트**: 2026년 8월 31일 (Sprint 1 종료)

