# 📅 UPDATE-2026-10-31 버전 통합 로드맵

## 🚨 주요 변경사항 (2026년 2월 → 2026년 10월)

| 항목 | 기존 | 변경 후 | 비고 |
|------|------|--------|------|
| **논문 제출 기한** | 2027년 2월 28일 | **2026년 10월 31일** | -4개월 (단축) |
| **전체 기간** | 9개월 | **5개월** | 급속도 진행 |
| **총 개발 시간** | 430시간 | **130시간** | 70% 감축 |
| **범위** | 전체 (15개 항목) | **필수만 (5개 항목)** | 50% 축소 |
| **목표 타입** | 포괄적 | **REALISTIC** | 완성도 우선 |

---

## 📊 4개월 압축 로드맵

### Overview Timeline

```
2026년 6월 1주 ~ 10월 4주 (17주)

📍 Phase 1: Foundation (Week 1-6, 6월 1-7월 12)
   ├─ Sprint 1A: PCMCI + EDA (Week 1-3, 40h)
   ├─ Sprint 1B: SHAP + Attention + StreamUI (Week 4-6, 30h)
   └─ 완료 기준: PCMCI ≥ 75%, Figure 10개

📍 Phase 2: Core (Week 7-10, 7월 13-8월 9)
   ├─ Sprint 2: GNN/Rule-based + Integration (Week 7-10, 25h)
   └─ 완료 기준: Propagation accuracy ≥ 60%, Pipeline complete

📍 Phase 3: Delivery (Week 11-17, 8월 10-10월 11)
   ├─ Sprint 3A: Experiments + Benchmark (Week 11-14, 20h)
   ├─ Sprint 3B: Paper Writing (Week 15-17, 15h)
   └─ 완료 기준: 논문 5-6 pages, Appendix 포함

📍 Buffer & Final (Week 18-20, 10월 12-31)
   ├─ 예비 시간 활용
   └─ 최종 교정 & 제출 준비
```

---

## ✅ MUST-DO 필수 항목 (Tier 1: 논문 통과 필수)

| # | 작업 | 난도 | 기한 | 시간 | 산출물 |
|----|------|------|------|------|--------|
| **M1** | Causal DAG Learning (PCMCI) | ★★★★ | 7월 12일 | 40h | DAG 시각화 + 정확도 결과 |
| **M2** | SHAP + Attention XAI | ★★ | 8월 9일 | 30h | Force plot + Heatmap figure (5개) |
| **M3** | Anomaly Detection Baseline | ★ | 7월 12일 | 20h | AUROC ≥ 0.95 (합성) |
| **M4** | Streamlit Demo (2-3 탭) | ★★ | 8월 9일 | 10h | 동작하는 UI |
| **M5** | SECOM Benchmark | ★★ | 9월 6일 | 10h | 성능 테이블 |
| **M6** | 논문 작성 (5-6p) | ★★★ | 10월 11일 | 15h | 완성 논문 draft |
| **M7** | Appendix (실험 설정) | ★ | 10월 11일 | 5h | 재현 가능한 상세 기술 |
| **합계** | | | | **130h** | 7개 산출물 |

---

## 🗓️ 상세 주간 계획 (Week-by-week)

### 📍 Phase 1-A: PCMCI + EDA (Week 1-3, 6월 1-21)

**목표**: PCMCI 정확도 ≥ 75% 달성 또는 Contingency plan 확정

#### Week 1 (6월 1-7)

| 일 | 작업 | 예상 시간 | 산출물 |
|----|------|---------|--------|
| Mon 6/1 | PCMCI 라이브러리 최종 선택 (3개 비교 + 테스트) | 4h | 선택 결정 문서 |
| Tue 6/2 | 합성 DAG 최종 정의 + ground truth yaml | 3h | configs/ground_truth_dag.yaml |
| Wed 6/3 | PCMCI wrapper 기본 구조 (skeleton) | 4h | src/causal/pcmci_wrapper.py |
| Thu 6/4 | PCMCI 첫번째 실행 + baseline 정확도 기록 | 4h | 정확도 결과 (예: precision=0.65) |
| Fri 6/5 | Week 1 리뷰 + Week 2 계획 확정 | 2h | 진도 리포트 |
| Sat-Sun | 예비 | +4h | |
| **합계** | | **21h** | **5개 산출물** |

#### Week 2 (6월 8-14)

| 일 | 작업 | 예상 시간 | 산출물 |
|----|------|---------|--------|
| Mon 6/8 | PCMCI 매개변수 스윕 (tau, alpha, method) | 6h | 3-5개 설정 비교 |
| Tue 6/9 | 정확도 개선 시도 (조건 독립성 테스트 변경) | 4h | 개선 기록 |
| Wed 6/10 | 정확도 목표 진단 (75% 달성 가능성 평가) | 2h | Go/No-go 판단 |
| Thu 6/11 | Contingency 준비 (LiNGAM 또는 FCI 백업) | 2h | 백업 방법 프로토타입 |
| Fri 6/12 | 테스트 코드 작성 (tests/test_pcmci.py) | 3h | 단위 테스트 |
| Sat-Sun | 예비 | +2h | |
| **합계** | | **19h** | **5개 산출물** |

#### Week 3 (6월 15-21)

| 일 | 작업 | 예상 시간 | 산출물 |
|----|------|---------|--------|
| Mon 6/15 | 합성 데이터 EDA (분포, 이상치, 결측) | 4h | 4개 visualization |
| Tue 6/16 | 데이터 통계 분석 (기술 통계표) | 3h | summary_statistics.csv |
| Wed 6/17 | PCMCI 최종 결과 정리 및 시각화 | 4h | DAG graph image + table |
| Thu 6/18 | 논문 Section 3 (Method: Causal DAG) 초안 작성 | 3h | 1-1.5 페이지 |
| Fri 6/19 | Phase 1-A 완료 리뷰 + Phase 1-B 킥오프 | 2h | 종료 리포트 |
| Sat-Sun | 예비 | +4h | |
| **합계** | | **20h** | **5개 산출물** |

**Phase 1-A 완료 기준**:
- ✅ PCMCI 정확도 기록 (목표 ≥ 75%, 최소 ≥ 65%)
- ✅ 합성 데이터 EDA 완료
- ✅ 테스트 코드 pass
- ✅ 논문 Method 섹션 초안

---

### 📍 Phase 1-B: SHAP + Attention + StreamUI (Week 4-6, 6월 22-7월 12)

**목표**: SHAP/Attention 통합 + 기본 Streamlit UI

#### Week 4 (6월 22-28)

| 일 | 작업 | 예상 시간 | 산출물 |
|----|------|---------|--------|
| Mon 6/22 | SHAP 그래디언트 계산 구현 | 6h | src/explain/shap_gradient_impl.py |
| Tue 6/23 | SHAP Force plot visualization | 4h | 3개 사례 figure |
| Wed 6/24 | Feature importance ranking | 2h | top_10_features.csv |
| Thu 6/25 | Attention map 추출 및 시각화 | 4h | attention_heatmap.py |
| Fri 6/26 | 통합 테스트 (SHAP + Attention) | 3h | integration_test.py |
| Sat-Sun | 예비 | +3h | |
| **합계** | | **22h** | **6개 산출물** |

#### Week 5 (6월 29-7월 5)

| 일 | 작업 | 예상 시간 | 산출물 |
|----|------|---------|--------|
| Mon 6/29 | Streamlit 기본 프레임워크 구축 (3개 탭) | 3h | scripts/ui_v2.py sketch |
| Tue 6/30 | Tab 1: Data Upload | 3h | 동작하는 파일 업로드 |
| Wed 7/1 | Tab 2: PCMCI DAG 시각화 | 4h | Cytoscape 연동 |
| Thu 7/2 | Tab 3: Predictions + SHAP 설명 | 3h | 예측 결과 + 설명 표시 |
| Fri 7/3 | UI 통합 테스트 | 2h | 전체 플로우 테스트 |
| Sat-Sun | 예비 | +4h | |
| **합계** | | **19h** | **4개 산출물** |

#### Week 6 (7월 6-12)

| 일 | 작업 | 예상 시간 | 산출물 |
|----|------|---------|--------|
| Mon 7/6 | Streamlit 배포 (로컬 테스트) | 2h | 동작 확인 |
| Tue 7/7 | 논문 Figure 최종화 (SHAP, Attention 5개) | 4h | 5개 figure |
| Wed 7/8 | 논문 Section 4 (Results: Causal part) 초안 | 3h | 0.5-1 페이지 |
| Thu 7/9 | 논문 Section 5 (XAI part) 초안 | 3h | 0.5-1 페이지 |
| Fri 7/10 | Phase 1-B 최종 리뷰 + Phase 2 준비 | 2h | 리뷰 리포트 |
| Sat-Sun | 예비 | +4h | |
| **합계** | | **18h** | **4개 산출물** |

**Phase 1-B 완료 기준**:
- ✅ SHAP 그래디언트 정상 작동
- ✅ Attention 시각화 완료
- ✅ Streamlit UI 기본 동작
- ✅ 논문 Methods/Results (Part 1-2) 초안 완료

---

### 📍 Phase 2: GNN/Rule-based + Integration (Week 7-10, 7월 13-8월 9)

**목표**: Propagation 모델 구현 또는 Rule-based 대체

#### 시나리오 A: GNN 구현 (시간 있을 시)

| Week | 초점 | 예상 시간 | 산출물 |
|------|------|---------|--------|
| 7 | PyG 기본 + Graph 변환 | 8h | DAG → PyG graph |
| 8 | GNN 모델 설정 (GAT or GCN) | 8h | model.py |
| 9 | 학습 + 평가 | 6h | accuracy result |
| 10 | 통합 파이프라인 | 3h | propagation_pipeline.py |
| **합계** | | **25h** | **완전 구현** |

#### 시나리오 B: Rule-based 대체 (시간 부족 시)

| Week | 초점 | 예상 시간 | 산출물 |
|------|------|---------|--------|
| 7-10 | BFS 또는 Influence score 기반 전파 | 15h | propagation_rule_based.py |
| | | 10h | 평가 및 검증 |
| **합계** | | **25h** | **충분한 구현** |

**Phase 2 완료 기준**:
- ✅ Propagation 모델 구현 완료
- ✅ 파이프라인 완전 통합
- ✅ 정성적 예시 (case study) 2-3개

---

### 📍 Phase 3-A: Experiments + Benchmark (Week 11-14, 8월 10-9월 6)

#### Week 11 (8월 10-16)

| 일 | 작업 | 예상 시간 | 산출물 |
|----|------|---------|--------|
| Mon 8/10 | SECOM 데이터 로드 + 전처리 | 3h | secom_processed.parquet |
| Tue 8/11 | 모델 학습 (SECOM) | 3h | model_secom.ckpt |
| Wed 8/12 | 평가 (AUROC, FAR, Cost-aware) | 2h | secom_results.csv |
| Thu 8/13 | 합성 vs SECOM 비교표 | 2h | comparison_table.csv |
| Fri 8/14 | Ablation study 계획 | 1h | ablation_design.txt |
| Sat-Sun | 예비 | +4h | |
| **합계** | | **15h** | **4개 산출물** |

#### Week 12 (8월 17-23)

| 일 | 작업 | 예상 시간 | 산출물 |
|----|------|---------|--------|
| Mon 8/17 | Ablation 1: "Without Causal DAG" | 2h | 결과 |
| Tue 8/18 | Ablation 2: "Without XAI" | 2h | 결과 |
| Wed 8/19 | Ablation 3: "LSTM baseline" | 3h | 결과 |
| Thu 8/20 | 모든 결과 통합 테이블 작성 | 2h | ablation_table.csv |
| Fri 8/21 | 정성 분석 (figure + 텍스트) | 2h | analysis.md |
| Sat-Sun | 예비 | +3h | |
| **합계** | | **14h** | **5개 산출물** |

#### Week 13-14 (8월 24-9월 6)

| 작업 | 예상 시간 | 산출물 |
|------|---------|--------|
| 부실 분석 및 재실험 | 5h | refinement results |
| 최종 메트릭 정리 | 3h | final_metrics_table.xlsx |
| 논문 Results 섹션 최종화 | 3h | Results section draft |
| **합계** | **11h** | **3개 산출물** |

**Phase 3-A 완료 기준**:
- ✅ SECOM 벤치마크 완료 (≥ 0.90 AUROC)
- ✅ Ablation study 3개 완료
- ✅ 최종 메트릭테이블 (합성 + SECOM + Ablation)
- ✅ 논문 Results 섹션 완성도 80%

---

### 📍 Phase 3-B: Paper Writing (Week 15-17, 9월 7-28)

#### Week 15 (9월 7-13)

| 일 | 작업 | 예상 시간 | 산출물 |
|----|------|---------|--------|
| Mon 9/7 | 논문 전체 구조 확정 (목차 최종) | 1h | outline.md |
| Tue 9/8 | Introduction 재작성 (최종) | 2h | intro_final.md |
| Wed 9/9 | Related Work 재작성 (최종) | 2h | related_final.md |
| Thu 9/10 | Methods 통합 (Causal + GNN + XAI) | 2h | methods_final.md |
| Fri 9/11 | Results + Discussion 작성 | 2h | results_discussion_draft.md |
| Sat-Sun | 예비 | +3h | |
| **합계** | | **12h** | **5개 산출물** |

#### Week 16 (9월 14-20)

| 일 | 작업 | 예상 시간 | 산출물 |
|----|------|---------|--------|
| Mon 9/14 | 모든 figure/table 최종 통합 | 2h | figures_final/ |
| Tue 9/15 | 논문 통독 + 1차 교정 | 2h | draft_v1.pdf |
| Wed 9/16 | 지도교수 피드백 통합 | 2h | draft_v2.pdf |
| Thu 9/17 | Appendix 작성 (실험 설정, 하이퍼파라미터) | 2h | appendix.md |
| Fri 9/18 | 최종 교정 | 1h | draft_v3.pdf |
| Sat-Sun | 예비 | +3h | |
| **합계** | | **12h** | **5개 산출물** |

#### Week 17 (9월 21-28)

| 일 | 작업 | 예상 시간 | 산출물 |
|----|------|---------|--------|
| Mon 9/21 | 요약(Abstract) 최종화 | 1h | abstract_final.md |
| Tue 9/22 | 참고문헌 최종화 (bibtex) | 1h | references.bib |
| Wed 9/23 | 최종 논문 생성 (PDF) | 1h | thesis_final_2026-09-23.pdf |
| Thu 9/24 | 최종 검토 (format, fonts, margins) | 1h | 체크리스트 완료 |
| Fri 9/25 | 최종 제출 준비 + 예비 | 2h | 제출 ready |
| Sat-Sun | | | |
| **합계** | | **6h** | **5개 산출물** |

**Phase 3-B 완료 기준**:
- ✅ 논문 완전 draft (5-6 pages)
- ✅ 모든 figure/table 최종 (15-20개)
- ✅ Appendix 포함 (재현 가능한 상세 기술)
- ✅ 교수 피드백 반영 완료

---

### 📍 Finalization & Reserve (Week 18-20, 10월 1-31)

| Week | 목표 | 예상 시간 |
|------|------|---------|
| 18-19 (10/1-18) | 예비 시간 (혹시 모를 보완) | 10-15h |
| 20 (10/19-31) | 최종 제출 + 방어 준비 | 5-10h |
| **합계** | | **15-25h** |

---

## 📋 필수 산출물 체크리스트 (최종)

### 📄 논문

```
✅ 논문 전문 (5-6 pages, PDF)
   ├─ Abstract (0.3p)
   ├─ Introduction (1p)
   ├─ Related Work (1p)
   ├─ Methods (1-1.5p)
   │  ├─ Causal Discovery
   │  ├─ GNN/Rule-based Propagation
   │  └─ XAI Integration
   ├─ Results (1p)
   ├─ Discussion (0.5p)
   └─ Conclusion (0.3p)

✅ Appendix (2-3p)
   ├─ Implementation details
   ├─ Hyperparameters
   ├─ Additional figures
   └─ Code availability

✅ References (bibtex format)
```

### 📊 데이터 & 코드

```
✅ GitHub repository
   ├─ src/causal/pcmci_wrapper.py (PCMCI)
   ├─ src/models/propagation.py (GNN/Rule-based)
   ├─ src/explain/shap_xai.py (SHAP)
   ├─ scripts/ui_v2.py (Streamlit)
   ├─ tests/test_*.py (Unit tests)
   └─ configs/ (Hydra configs)

✅ Reproducible experiments
   ├─ configs/experiment/final_config.yaml
   ├─ scripts/reproduce_results.py
   └─ requirements.txt + dvc.yaml
```

### 📈 도출 산출물

```
✅ Figures (15-20개)
   ├─ DAG visualization (2-3)
   ├─ SHAP plots (3-4)
   ├─ Attention heatmaps (2-3)
   ├─ Benchmark tables (4-5)
   ├─ Ablation study (2-3)
   └─ Misc. (2-3)

✅ Performance tables
   ├─ Synthetic SECOM 비교
   ├─ Ablation study 결과
   ├─ Baseline comparison
   └─ Statistical significance
```

### 🖥️ 데모 & UI

```
✅ Streamlit application (3개 탭)
   ├─ Data Upload
   ├─ Causal DAG Viewer
   └─ Predictions + Explanations

✅ Jupyter Notebook (선택)
   ├─ Case study 1-3
   ├─ Detailed analysis
   └─ Community reproducibility
```

---

## 🎯 주요 초점 (핵심 성공 장치)

### 1️⃣ PCMCI 정확도 문제 대비

```
Goal: Precision/Recall ≥ 75%

If achieved       → 그대로 진행
If 65-75%        → 논문에서 trade-off 설명
If 55-65%        → "Interpretability focus" 피벗
If < 55%         → Contingency: 백업 방법 (FCI, LiNGAM) 사용
```

### 2️⃣ 시간 관리 (매주 체크)

```
매주 금요일 저녁 점검:
  ✓ 주간 예상 시간: 8시간 이상 확보했나?
  ✓ 산출물 생성: 지난 주 예상 산출물 완료했나?
  ✓ 위험도 변화: 새로운 리스크 없나?
  ✓ 일정 조정: 내주 계획 수정 필요한가?
```

### 3️⃣ 논문 기여도 명확화

```
최소 1가지는 반드시 추가:
  ☑ Option A: "Temporal Causal Motif" 개념 (새로운 아이디어)
  ☑ Option B: "Causal Intervention F1-score" 평가지표 (새로운 지표)
  ☑ Option C: "Real-world applicability study" (임팩트)
  
선택 기한: 6월 15일까지
```

---

## 📌 최상 시나리오 vs 현실 시나리오

### Best Case (35% 가능성)
```
PCMCI: 85% accuracy
GNN: Convergence success
Time: 정확히 계획대로
Result: 6-7 pages 완성도 높은 논문
```

### Realistic Case (55% 가능성) ← 목표
```
PCMCI: 70-75% accuracy
GNN: Rule-based로 대체
Time: 약간의 압박 있음
Result: 5-6 pages 완성도 중상의 논문 (충분히 통과)
```

### Worst Case (25% 미만)
```
PCMCI: < 60% accuracy
GNN: 삭제
Time: 심각한 지연
Result: 4-5 pages 기본만 충족
```

---

## 🔥 가장 중요한 3줄 요약

1. **이번 주가 결정적**: Week 1-3 (6월 1-21)에 PCMCI 정확도 ≥ 75% 달성 못하면 논문 기여도 급락
2. **범위 축소는 성공의 열쇠**: 필수 5개 항목만 완벽하게 → 모든 항목 중간 수준의 100배 낫다
3. **매주 산출물 생성 필수**: "진행 중" 상태는 불가능, 매주 GitHub commit + figure/table 1개 이상 필수

---

**최종 일정 버전**: 2.0 (2026년 10월 31일 기한)  
**승인**: 🟡 조건부 통과 (범위 축소 + 리스크 대응 완료 시 가능)  
**다음 리뷰**: 2026년 6월 15일 (Mid-week 체크포인트)


