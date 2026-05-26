# PCB Press Defect Prediction - 논문 연구개발 로드맵 (2026-2027)

**작성일**: 2026년 5월 25일  
**저자**: GitHub Copilot (논문 전략 분석)  
**목표 제출일**: 2027년 2월  
**논문명**: MS-CDPNet (Multi-Stage Causal Defect Propagation Network)

---

## A. 한 줄 총평

**현재는 원점-확인 단계(POC 완성), 논문 핵심인 인과 그래프 + XAI 심화 + 공정 최적화 로직이 부재하므로, 9개월간 3개 영역(인과추론, 설명가능성, 공정최적화)에 집중하되, 2주마다 논문용 아티팩트를 생산하는 "스프린트 중심" 개발 아젠다가 필수.**

---

## B. 현재 준비 수준 진단

### B.1 기술 스택 현황

| 항목 | 상태 | 평가 | 비고 |
|------|------|------|------|
| **Python/PyTorch 기초** | ✅ 완성 | 5/5 | PyTorch Lightning, PyTorch 2.2 구성 완료 |
| **시계열 모델 (PressFuse)** | ✅ 완성 | 4/5 | Cross-modal attention 기본 구조 있음, 실험 결과화 필요 |
| **합성 데이터 생성** | ✅ 완성 | 4/5 | 6가지 I anomaly type + 4가지 multi-anomaly 균형잡힘 |
| **기본 평가 지표** | ✅ 완성 | 4/5 | FAR@Recall, Cost-aware score 구현됨 |
| **웹 UI (Streamlit)** | ✅ Proto | 2/5 | 기본 구조만 있음, 인터랙티브 분석 기능 미충분 |
| **인과 그래프 학습 (PCMCI/NOTEARS)** | ❌ 부재 | 0/5 | **[최우선 개발 필요]** 논문 핵심 |
| **XAI 심화 (SHAP + Attention)** | 🟡 스켈레톤 | 1/5 | shap_grad.py, attention_viz.py 파일만 있음, 실제 구현 부재 |
| **공정 최적화 (Differentiable surrogate)** | ❌ 부재 | 0/5 | **[핵심 기여]** 향후 개발 |
| **Digital Twin / OPC UA** | ❌ 부재 | 0/5 | 중기 목표 |
| **Foundation Model (Chronos)** | ❌ 부재 | 0/5 | 비교 실험용 (중기) |

### B.2 코드 품질 현황

| 하위 항목 | 현황 | 평가 |
|----------|------|------|
| 테스트 커버리지 | 75% (보고됨) | ✅ 양호 |
| 타입 힌트 | Python 3.11+ (PEP 604) | ✅ 우수 |
| 문서화 | README, ARCHITECTURE 기본 | 🟡 보통 (API 레퍼런스 부족) |
| 코드 스타일 | Black + Ruff 자동화 | ✅ 우수 |
| CI/CD | GitHub Actions 구성 중 | 🟡 진행 중 |
| 의존성 관리 | requirements.txt 명시 | ✅ 명확 |

### B.3 데이터 준비 현황

| 항목 | 상태 | 평가 |
|------|------|------|
| 합성 데이터 생성 파이프라인 | ✅ 동작 | 즉시 사용 가능 |
| SECOM Dataset 통합 | ✅ 베이스라인 준비 | 시계열 기본 학습용 |
| DeepPCB 고려 | 🟡 고려 중 | 이미지 멀티모달 학습용 |
| 실제 제조 데이터 | ❌ NDA 비공개 | 합성 데이터로 충분 |
| DVC 파이프라인 | ✅ dvc.yaml 존재 | 재현성 확보 |

### B.4 논문 구조 준비 현황

| 섹션 | 예상 기여도 | 현재 진도 | 리스크 |
|------|----------|---------|--------|
| 1. Introduction | 논문 기본 | 30% | 낮음 |
| 2. Related Work | 논문 기본 | 20% | 낮음 |
| 3. Problem Definition | 논문 핵심 | 60% | 낮음 |
| 4. Proposed Method (캐주얼 DAG) | **논문 핵심** | 5% | **🔴 매우 높음** |
| 5. Causal Discovery (NOTEARS/PCMCI) | **논문 핵심** | 0% | **🔴 매우 높음** |
| 6. XAI (SHAP + Attention) | 논문 핵심 | 10% | **🔴 높음** |
| 7. 공정 최적화 (Differentiable surrogate) | 논문 기여 | 0% | **🔴 높음** |
| 8. Experiments (Synthetic + SECOM) | 논문 핵심 | 40% | 중간 |
| 9. Results & Discussion | 논문 핵심 | 0% | 중간 |

### B.5 강점 (Strengths)

1. **도메인 이해**: PCB Press 공정, P013/P019 불량 유형, 시간 지연 문제 명확히 정의
2. **합성 데이터**: 6가지 I anomaly + 4가지 multi-anomaly → 재현 가능, 데이터 부족 극복
3. **기본 ML 파이프라인**: PyTorch Lightning, MLflow, Hydra 모두 갖춤
4. **비용 민감 평가**: FAR@Recall, Cost-aware 지표 구현 (도메인 특화)
5. **커뮤니티 준비**: Fork 최적화, GitHub Pages, CI/CD 자동화 완료

### B.6 약점 (Weaknesses)

1. **인과 추론 부재**: 논문의 "MS-CDPNet" 이름 자체가 인과 기반인데, PCMCI/NOTEARS 구현 없음
2. **XAI 구현 부실**: shap_grad.py, attention_viz.py 파일만 있고 실제 기능 없음 (스켈레톤 상태)
3. **공정 최적화 미계획**: "공정 조건 최적화" 목표 있지만 구현 전무
4. **UI/UX 초라**: Streamlit 기본 구조만 있음, 엔지니어가 필요한 대시보드 기능 부족
5. **논문 양적 부족**: 현재 코드량 ~3,000 LOC → 논문 포함 최종 ~10,000 LOC 예상
6. **일정 리스크**: 9개월 안에 3개 핵심 영역 모두 구현 + 실험 + 논문 작성 → 촉박함

### B.7 기회 (Opportunities)

1. **GNN 확대**: PyG (PyTorch Geometric) 활용하면 인과 그래프 자연스럽게 구현 가능
2. **재사용 가능 패턴**: SECOM/TEP 베이스라인 → 다른 공정에도 일반화 가능
3. **커뮤니티 영향**: 합성 데이터 + 오픈소스 → 인성 강려쓰 논문화
4. **XAI 확대**: SHAP + Attention + GNN을 조합하면 "멀티 레이어 설명" 가능

### B.8 위협 (Threats)

1. **논문 데드라인**: 2027년 2월까지 9개월 → 스프린트 실패 시 심각한 지연
2. **인과 학습의 어려움**: PCMCI/NOTEARS 튜닝 복잡함, 심플한 DAG만으로는 논문 기여도 낮음
3. **멀티모달 오버엔지니어링**: 이미지 데이터 없으면 PressFuse 복잡도만 증가, 논문 기여도 미흡
4. **실험 재현성**: 합성 데이터 → 실제 공정 성능 차이 설명하기 어려움
5. **의존성 변경**: PyTorch 업데이트, Transformers 호환성 문제 (빈번함)

---

## C. 논문 직결 우선순위 표

### 우선순위 판단 기준

1. **논문 핵심**: 제목 "MS-CDPNet"에 직접 포함되는 내용
2. **포트폴리오 강화**: 논문 이후 취업/진학에 도움되는 기술
3. **장기 탐색**: 논문 제출 이후 하지만 시간 있으면 선행 연구

### 최우선 순위 (P0: 논문 제출까지 필수)

| 번호 | 작업명 | 목적 | 목표치 | 담당자 | 완료 기한 | 논문 핵심도 |
|------|--------|------|--------|--------|----------|-----------|
| **P0-1** | **Causal Discovery (PCMCI/NOTEARS)** | 공정 변수 간 인과 DAG 학습 | DAG precision ≥ 80% (합성 기준) | [구현 필요] | 2026년 8월 | 🔴 핵심 |
| **P0-2** | **GNN-based Defect Propagation** | PCMCI DAG를 PyG 그래프로 변환 후 인과 전파 예측 | 불량 전파 경로 정확도 ≥ 85% | [구현 필요] | 2026년 9월 | 🔴 핵심 |
| **P0-3** | **SHAP + Attention Visualization** | SHAP 그래디언트와 어텐션 맵 통합 후 가시화 | 논문용 Fig 3~5 생성, 엔지니어 이해도 ≥ 80% | [구현 필요] | 2026년 9월 | 🔴 핵심 |
| **P0-4** | **Auto-Report Pipeline** | 불량 메커니즘 자동 보고서 생성 (PDF + JSON) | ReportLab 기반 템플릿, 2~3 사례 | [구현 필요] | 2026년 10월 | 🟠 중요 |
| **P0-5** | **Synthetic Data Validation** | 합성 데이터 분포 vs 실제 공정 데이터 통계 검증 | KS 검정 p-value > 0.05, 분포 매칭율 ≥ 90% | 부분 완료 | 2026년 6월 | 🟠 중요 |
| **P0-6** | **Cost-Sensitive Evaluation + Benchmarks** | FAR@Recall, Cost-aware score 상세 분석 | AUROC ≥ 0.98, FAR < 5% (recall=0.95) | 부분 완료 | 2026년 7월 | 🟠 중요 |

### 우선 순위 높음 (P1: 논문에 강점 추가)

| 번호 | 작업명 | 목적 | 목표치 | 완료 기한 | 기여도 |
|------|--------|------|--------|----------|--------|
| **P1-1** | Foundation Model 비교 (Chronos) | 기존 LSTM vs Chronos 시계열 예측 정확도 비교 | Chronos RMSE vs LSTM RMSE 비교표 | 2026년 10월 | 포트폴리오 |
| **P1-2** | Explainability 비교 (SHAP vs LIME vs Integrated Gradients) | 3가지 XAI 방법론 비교 (정성/정량) | Table 비교, Fig 6~8 생성 | 2026년 10월 | 포트폴리오 |
| **P1-3** | Advanced Causal Methods (DoWhy, CausalML) | PCMCI + 추가 인과 학습 기법 비교 | DAG precision 비교표 | 2026년 11월 | 포트폴리오 |
| **P1-4** | Domain Adaptation (다른 공정으로 전이) | SECOM/TEP 추가 데이터셋 성능 검증 | 3개 공정 이상 AUROC 기록 | 2026년 11월 | 포트폴리오 |
| **P1-5** | Interactive Web Dashboard (React 또는 Dash) | 현재 Streamlit 대신 고급 UI/UX | 논문용 데모 영상, 사용자 만족도 ≥ 80% | 2026년 12월 | UI/UX |

### 중간 우선순위 (P2: 시간 남으면 추진)

| 번호 | 작업명 | 목적 | 완료 기한 | 기여도 |
|------|--------|------|----------|--------|
| **P2-1** | Digital Twin Prototype (SimPy) | 공정 시뮬레이션과 모델 학습 루프 통합 | 2026년 12월 이후 | 포트폴리오 |
| **P2-2** | ONNX Export + TensorRT 최적화 | 경량화 및 실시간 추론 | 2026년 12월 | Edge AI |
| **P2-3** | Federated Learning PoC (Flower) | 분산 학습 기본 구조 | 2027년 1월 | 포트폴리오 |
| **P2-4** | VLM 기반 불량 이미지 분석 (LLaVA) | 이미지 설명 + 좌표 출력 | 2027년 1월 | 포트폴리오 |

### 낮은 우선순위 (P3: 논문 이후 장기 과제)

| 참고 항목 | 설명 |
|----------|------|
| Edge AI deployment | 논문 이후 실무 응용 |
| Federated Learning | 제조 데이터 보안 대응 |
| VLM integration | AOI 시스템과 연계 |

---

## D. 연구개발 로드맵 (3개월 × 3단계, 2026.6 ~ 2027.2)

### 로드맵 개요

```
2026년 6월   2026년 9월   2026년 12월   2027년 2월
 │            │             │             │
 ├────────┤  ├────────┤    ├────────┤    └─ 논문 제출
 Sprint 1  Sprint 2  Sprint 3
 (POC검증)  (핵심기여)  (실험+최적화)
```

### Sprint 1 (2026.6.1 ~ 2026.8.31) : POC 검증 & 기초 다지기

**슬로건**: "인과 그래프 학습 준비 + 기본 인프라 구축"

#### 목표
- PCMCI/NOTEARS 여건 조사 및 프로토타입 구현
- 합성 데이터 품질 검증
- 논문용 아티팩트(Fig, Table) 초안 생성
- 팀 내 정렬 → 2차 개발 상세 기획

#### 산출물

| 주제 | 산출물 | 형식 | 완료일 |
|------|--------|------|--------|
| **Causal Discovery Baseline** | PCMCI 기본 구현 + 합성 DAG 로드 | Python module (`src/causal/pcmci_wrapper.py`) | 2026.07.15 |
| **Data Validation** | 합성 데이터 통계 vs 실제 분포 비교 리포트 | Jupyter + HTML Report | 2026.07.20 |
| **Synthetic Data EDA** | 6가지 anomaly 시각화 (60 figure) | PNG + Plotly HTML | 2026.07.31 |
| **Baseline Model Training** | LSTM/PressFuse 기본 학습 + 성능표 | CSV (metric benchmark) | 2026.08.10 |
| **Causal DAG Visualization** | Networkx 기반 DAG 시각화 + 엣지 중요도 | PNG + Interactive HTML (Plotly) | 2026.08.20 |
| **Sprint 1 Review Document** | 다음 sprint 상세 기획서 | Markdown (이 문서 업데이트) | 2026.08.31 |

#### 주요 작업 항목 (2주 단위 태스크)

**Week 1-2 (6월 1-15)**
- [ ] PCMCI 라이브러리 선택 (causalml vs DoWhy vs castle)
- [ ] PCMCI 기본 매개변수 이해 및 테스트
- [ ] 합성 데이터 기반 ground truth DAG 정의

**Week 3-4 (6월 16-30)**
- [ ] `src/causal/` 폴더 생성 및 기본 모듈 구조화
- [ ] PCMCI 래퍼 함수 작성 (`learn_causal_dag()`)
- [ ] 테스트 코드 작성 (`tests/test_pcmci.py`)

**Week 5-6 (7월 1-15)**
- [ ] 합성 데이터 통계 분석 및 분포 검증
- [ ] SECOM 데이터셋 통계와 합성 데이터 비교
- [ ] 논문 시각화 초안 5개 생성

**Week 7-8 (7월 16-31)**
- [ ] LSTM/PressFuse 기본 모델 학습 완료
- [ ] AUROC, FAR@Recall 성능표 생성
- [ ] MLflow 로깅 정상화 및 대시보드 구성

**Week 9-10 (8월 1-15)**
- [ ] PCMCI DAG 학습 및 시각화
- [ ] Graph structure learning 성능 분석 (precision/recall)
- [ ] Causal structure 품질 평가 지표 정의

**Week 11-12 (8월 16-31)**
- [ ] Sprint 1 리뷰 및 다음 단계 기획
- [ ] 논문 draft Introduction + Related Work 작성
- [ ] Sprint 2 상세 기획서 완성

#### 팀 역할 분담 (혼자 진행 시)

| 작업 | 우선순위 | 소요시간 |
|------|---------|--------|
| PCMCI 구현 | 🔴 최우선 | 40시간 |
| 데이터 검증 | 🟠 중요 | 20시간 |
| 시각화 | 🟠 중요 | 30시간 |
| 논문 draft | 🟡 보조 | 20시간 |
| **합계** | | **110시간 (약 3주)** |

#### 성공 기준

- [ ] PCMCI 모듈 테스트 pass rate ≥ 90%
- [ ] 합성 데이터 KS test p-value > 0.05
- [ ] 논문용 Figure 5개 이상 초안 생성
- [ ] AUROC ≥ 0.95 (합성 데이터 기준)

---

### Sprint 2 (2026.9.1 ~ 2026.11.30) : 핵심 기여 구현

**슬로건**: "GNN 인과 전파 + SHAP/Attention XAI + 공정 최적화 기본"

#### 목표

1. PCMCI DAG → PyG 그래프로 변환 및 GNN 기반 불량 전파 예측
2. SHAP 그래디언트와 Attention 맵 통합 XAI 구현
3. 공정 변수 최적화를 위한 미분 가능 서로게이트 모델 기초
4. 논문 Methods 섹션 초안 완성
5. 첫번째 투고용 preliminary results 생성

#### 산출물

| 주제 | 산출물 | 형식 | 완료일 |
|------|--------|------|--------|
| **GNN Causal Propagation** | PyG 기반 불량 전파 모델 | Python module (`src/models/causal_gnn.py`) | 2026.09.30 |
| **SHAP Integration** | SHAP 그래디언트 계산 및 가시화 | Python module + Visualization | 2026.10.10 |
| **Attention Visualization** | Cross-modal attention 히트맵 | PNG + Interactive HTML | 2026.10.15 |
| **Surrogate Model (Process Opt.)** | 미분 가능 서로게이트 기본 구조 | Python module skeleton | 2026.10.20 |
| **Unified Explainability Pipeline** | SHAP + Attention + Causal 통합 파이프라인 | Python script + Test | 2026.10.31 |
| **Preliminary Results Table** | 메인 실험 결과표 (AUROC, FAR, DAG precision) | CSV + LaTeX Table | 2026.11.15 |
| **Methods & Results Draft** | 논문 Methods 섹션 + Results 단락 | Markdown/LaTeX | 2026.11.30 |

#### 주요 작업 항목 (2주 단위)

**Week 1-2 (9월 1-15)**
- [ ] PyTorch Geometric 학습 및 기본 구조 이해
- [ ] PCMCI DAG → PyG Graph 변환 함수 작성
- [ ] GNN 모든 아키텍처 (GAT, GCN, GraphSAGE) 중 선택

**Week 3-4 (9월 16-30)**
- [ ] GNN 기반 불량 전파 모델 구현
- [ ] 학습 루프 및 evaluation 메트릭 정의
- [ ] 불량 전파 정확도 벤치마크

**Week 5-6 (10월 1-15)**
- [ ] SHAP 라이브러리 통합 및 그래디언트 계산
- [ ] Attention 맵 추출 및 시각화 구현
- [ ] 통합 explanation pipeline 설계

**Week 7-8 (10월 16-31)**
- [ ] 공정 최적화 서로게이트 모델 기초 구현
- [ ] Inverse problem (target defect rate → 최적 pressure/temp) 포뮬레이션
- [ ] Differentiable loss function 정의

**Week 9-10 (11월 1-15)**
- [ ] 메인 실험 결과 취합 및 표 생성
- [ ] Ablation studies (with/without causal DAG, XAI)
- [ ] Preliminary results 논문 draft

**Week 11-12 (11월 16-30)**
- [ ] 논문 Methods 섹션 최종 작성
- [ ] Results discussion 작성
- [ ] Figure 10개 이상 최종 생성

#### 성공 기준

- [ ] GNN 불량 전파 정확도 ≥ 85%
- [ ] SHAP + Attention 통합 파이프라인 동작
- [ ] 논문용 Figure 15개 이상 완성
- [ ] Preliminary results table 완성 (AUROC, FAR, DAG precision, Explanation quality)
- [ ] Methods 섹션 draft (3-4 페이지)

---

### Sprint 3 (2026.12.1 ~ 2027.2.28) : 실험 최적화 & 논문 최종화

**슬로건**: "벤치마크 완성 + 비교 방법론 + 논문 마무리"

#### 목표

1. 공정 최적화 기능 완성 및 검증
2. 비교 방법론 성능 비교 (LSTM vs PressFuse vs GNN, SHAP vs LIME vs IG)
3. 추가 공정/데이터셋 (SECOM, TEP) 위에서 일반화 검증
4. 논문 전체 완성 및 제출
5. 최종 코드 정리 및 배포 준비

#### 산출물

| 주제 | 산출물 | 형식 | 완료일 |
|------|--------|------|--------|
| **Process Optimization Complete** | 공정 변수 최적화 모듈 완성 | Python module (`src/models/process_optimizer.py`) | 2026.12.20 |
| **Benchmark Comparison** | 3개 모델 × 3개 공정 성능 비교표 | CSV + LaTeX Table | 2027.01.10 |
| **XAI Method Comparison** | SHAP vs LIME vs IG 정성/정량 비교 | Figure + Table | 2027.01.15 |
| **SECOM/TEP Validation** | 추가 공개 데이터셋 성능 | CSV + Figure | 2027.01.15 |
| **Complete Thesis Draft** | 논문 완전 초안 (all sections) | PDF (6-8 pages) | 2027.01.30 |
| **Final Code & Docs** | 정리된 소스 코드 + API 문서 | GitHub repo | 2027.02.10 |
| **Thesis Submission** | 최종 논문 제출 | PDF + Supporting Materials | **2027.02.28** |

#### 주요 작업 항목 (2주 단위)

**Week 1-2 (12월 1-15)**
- [ ] 공정 최적화 로직 완성 (Lagrange 또는 Gradient-based)
- [ ] 최적화 validation (논문 Figure 생성)
- [ ] 성능 개선 검증

**Week 3-4 (12월 16-31)**
- [ ] LSTM vs PressFuse vs GNN 벤치마크 실행
- [ ] 3개 공정 (합성, SECOM, TEP) 위에서 성능 테스트
- [ ] 결과표 생성

**Week 5-6 (1월 1-15)**
- [ ] SHAP vs LIME vs Integrated Gradients 비교 분석
- [ ] 정성 (Figure) + 정량 (metric) 평가
- [ ] 종합 discussion 작성

**Week 7 (1월 16-22)**
- [ ] 논문 전체 draft 작성 완료
- [ ] 지도교수 피드백 반영 1차
- [ ] 그림/표 최종 검증

**Week 8 (1월 23-29)**
- [ ] 논문 피드백 반영 2차
- [ ] 최종 교정 및 포맷팅
- [ ] Supporting materials (code, appendix) 최종화

**Week 9 (1월 30 ~ 2월 10)**
- [ ] 논문 최종 검토
- [ ] 제출 체크리스트 완료
- [ ] GitHub repo 최종 정리

**Week 10 (2월 11-28)**
- [ ] 논문 제출 및 방어 준비
- [ ] 예비 발표 자료 준비
- [ ] 추가 질문 대비

#### 성공 기준

- [ ] 공정 최적화 기능 정상 동작 (Proof-of-concept)
- [ ] 3개 모델 벤치마크 완성
- [ ] AUROC ≥ 0.98, FAR < 5% (recall=0.95)
- [ ] 논문 6-8 페이지 완성
- [ ] 논문 제출 완료

---

## E. 추가 개발 백로그 표

### 백로그 정의 기준

- **High**: 논문 제출 전 반드시 구현해야 할 기능
- **Medium**: 논문에 추가 강점을 주지만, 시간 부족 시 생략 가능
- **Low**: 포트폴리오 또는 후속 연구용

### 백로그 상세 정보

#### High Priority

| ID | 작업명 | 목적 | 입력 | 예상 산출물 | 기술스택 | 완료 기준 | 논문 기여도 | 완료예상 |
|-----|--------|------|------|-----------|---------|----------|-----------|---------|
| **BL-H1** | Causal Discovery Baseline (PCMCI) | 공정 변수 간 인과 관계 학습 | 합성 시계열 (B, T, D) | DAG 인접 행렬 + edge weights | causalml, DoWhy, castle | DAG precision ≥ 80% | 🔴 핵심 | 2026.08 |
| **BL-H2** | PyTorch Geometric GNN Wrapper | PyG 기반 그래프 신경망 설계 및 학습 | PCMCI DAG 구조 | Trained GNN model (*.ckpt) | PyTorch Geometric, Lightning | Defect propagation accuracy ≥ 85% | 🔴 핵심 | 2026.09 |
| **BL-H3** | SHAP Gradient Integration | 입력 특성에 대한 모델 기여도 계산 | Model + input (B, T, D) | SHAP values + force plot | shap, torch.autograd | SHAP value 정상 계산 | 🔴 핵심 | 2026.10 |
| **BL-H4** | Attention Map Visualization | Cross-modal attention 가중치 시각화 | Attention weights (num_heads, T, T) | Heatmap PNG + interactive HTML | matplotlib, plotly | 5개 사례 시각화 | 🔴 핵심 | 2026.10 |
| **BL-H5** | Unified Explainability Pipeline | SHAP + Attention + Causal 통합 | 모든 explanation 산출물 | Unified JSON report | Python wrapper | 파이프라인 테스트 pass | 🔴 핵심 | 2026.10 |
| **BL-H6** | Process Optimization (Surrogate) | 공정 변수 최적화를 위한 미분 가능 서로게이트 | Trained PressFuse model | Optimized (pressure, temp, vacuum) set | PyTorch autodiff | 최적화 valid 결과 | 🔴 핵심 | 2026.12 |
| **BL-H7** | Auto-Report Generation (PDF) | 불량 메커니즘을 자동 보고서로 생성 | 예측 + explanation 정보 | PDF 보고서 (예시 3개) | ReportLab, Jinja2 | Template 정상 렌더링 | 🟠 중요 | 2026.11 |
| **BL-H8** | Evaluation Metrics Refinement | 비용 민감 평가 + 불량 유형별 세부 지표 | y_true, y_pred (B,) | Detailed metric table | sklearn, numpy | FAR@Recall, Cost-aware 검증 | 🟠 중요 | 2026.08 |
| **BL-H9** | Comprehensive Benchmark Suite | LSTM vs PressFuse vs GNN 성능 비교 | 3개 모델 × 3개 데이터셋 | Benchmark table (AUROC, FAR, DAG precision) | PyTorch Lightning | 모든 benchmark 완료 | 🟠 중요 | 2027.01 |

#### Medium Priority

| ID | 작업명 | 목적 | 완료 기한 | 기여도 |
|-----|--------|------|----------|--------|
| **BL-M1** | XAI 비교연구 (SHAP vs LIME vs Integrated Gradients) | 설명 방법론 정량 비교 | 2026.11 | 포트폴리오 |
| **BL-M2** | Foundation Model 비교 (Chronos, MOMENT) | 최신 시계열 모델 벤치마크 | 2026.11 | 포트폴리오 |
| **BL-M3** | Domain Adaptation (SECOM/TEP/신규 공정) | 다른 공정으로 일반화 검증 | 2027.01 | 포트폴리오 |
| **BL-M4** | Interactive Web Dashboard (React/Dash) | 기존 Streamlit 대신 고급 UI | 2026.12 | UI/UX |
| **BL-M5** | Advanced Causal Methods (ADMGs, FCI) | PCMCI 외 추가 인과 학습 기법 | 2026.11 | 포트폴리오 |
| **BL-M6** | Anomaly Detection Baselines (Isolation Forest, LOF) | ML 베이스라인과 DL 모델 비교 | 2026.09 | 포트폴리오 |

#### Low Priority

| ID | 작업명 | 목적 | 완료 기한 | 기여도 |
|-----|--------|------|----------|--------|
| **BL-L1** | Digital Twin Prototype (SimPy) | PCB Press 공정 시뮬레이션 | 2027.01 | 포트폴리오 |
| **BL-L2** | ONNX Export + TensorRT | 경량화 및 실시간 추론 | 2026.12 | Edge AI |
| **BL-L3** | Federated Learning PoC (Flower) | 분산 학습 기본 구조 | 2027.01 | 포트폴리오 |
| **BL-L4** | VLM 기반 불량 이미지 분석 (LLaVA) | 이미지 설명 + 좌표 출력 | 2027.01 | 포트폴리오 |
| **BL-L5** | OPC UA Data Collection | 실제 설비 데이터 수집 | 2027.02 이후 | 장기 |

---

## F. UI/UX 기반 데이터 분석 시각화 방향

### 총론: 사용자 중심 웹 기반 분석 플랫폼

**핵심 원칙**:
1. **연구자**: 모델 개발, 실험 관리, 논문 결과 추출
2. **엔지니어**: 공정 모니터링, 불량 원인분석, 개선 조치 제안
3. **관리자**: 고급 대시보드, 트렌드 분석, KPI 추적

### F.1 화면 1: 데이터 업로드/선택

#### 목적
제조 데이터 또는 공개 데이터셋(SECOM, TEP) 로드

#### 핵심 기능
- **데이터 업로드**: CSV/Parquet 끌어놓기 또는 파일 선택
- **데이터셋 선택**: 프리로드된 공개 데이터셋 (SECOM, TEP, 합성 데이터)
- **데이터 미리보기**: 상위 20행 표시 + 기본 통계
- **스키마 매핑**: 자동 감지 또는 수동 설정 (P013_*, P019_* 변수)

#### 주요 위젯/컴포넌트

| 컴포넌트 | 타입 | 목적 |
|----------|------|------|
| **File Uploader** | Input | CSV/Parquet 파일 선택 |
| **Dataset Dropdown** | Select | SECOM/TEP/Synthetic 선택 |
| **Data Preview Table** | DataTable | 상위 50행 + 스크롤 |
| **Summary Statistics** | Metric/Card | Row count, Missing %, Feature count |
| **Schema Mapping Form** | Form | 변수명 → P013_* 매핑 |
| **Next Button** | Button | 다음 화면으로 진행 |

#### 입력/출력

| 항목 | 형식 | 설명 |
|------|------|------|
| 입력 | CSV/Parquet file | (N, D) 시계열 데이터 |
| 출력 | DataFrame + schema dict | 매핑된 데이터 + 변수 정의 |

#### 사용자 액션 플로우

```
1. 파일 끌어놓기 OR 선택
   ↓
2. 자동 스키마 감지
   ↓
3. 미리보기 확인 + 통계 보기
   ↓
4. 필요시 수동 매핑
   ↓
5. [Next] → 화면 2
```

#### 백엔드 API 초안

```python
# POST /api/upload
{
  "file": <binary>,  # CSV/Parquet
  "infer_schema": true,
  "override_schema": {
    "HPPRESSPV": "P013_pressure",
    "PT1": "P013_pt1",
    ...
  }
}

# Response
{
  "data_id": "uuid-xxx",
  "shape": [1000, 19],
  "columns": [...],
  "dtypes": {...},
  "missing_pct": 0.02,
  "schema": {...}
}
```

#### 추천 시각화 라이브러리

| 기능 | 추천 | 대안 |
|------|------|------|
| 테이블 표시 | ag-Grid | Tanstack Table |
| 통계 카드 | React + Tailwind | Material-UI |
| 파일 업로드 | Dropzone.js | React-dropzone |
| 스키마 매핑 폼 | React Hook Form | Formik |

#### UX 주의사항

- 파일 크기 > 500MB 시 경고
- 스키마 자동 감지 성공률 표시
- 매핑 오류 시 예시 제공 (e.g., "PT1_LEFT" → "P013_pt1")
- 모바일 화면에서도 드래그 & 드롭 가능

---

### F.2 화면 2: 전처리 및 변수 선택

#### 목적
이상탐지 및 인과학습 전에 데이터 정제, 정규화, 특성 선택

#### 핵심 기능
- **결측치 처리**: 제거/보간 옵션
- **정규화**: Min-Max, Z-score, Robust 선택
- **이상치 처리**: IQR, Z-score 기반 마킹
- **특성 선택**: 모든 변수 또는 상관도 기반 선택
- **시간 윈도우 설정**: 시계열 길이 (T) 설정

#### 주요 위젯/컴포넌트

| 컴포넌트 | 타입 | 목적 |
|----------|------|------|
| **Missing Data Handler** | Radio/Select | 전략 선택 (remove/interpolate/forward_fill) |
| **Normalization Method** | Radio | Min-Max/Z-score/Robust 선택 |
| **Outlier Detection** | Checkbox + Slider | IQR/Z-score + threshold |
| **Feature Selection** | Checkbox group | 변수별 포함/제외 선택 |
| **Time Window Slider** | Slider | T = 100-500 설정 |
| **Preview After Preprocessing** | Line chart | 전/후 데이터 비교 |

#### 입력/출력

| 항목 | 형식 | 설명 |
|------|------|------|
| 입력 | DataFrame + schema | 원본 데이터 + 변수 정의 |
| 출력 | Preprocessed DataFrame | (N', T, D') 정규화된 시계열 |

#### 사용자 액션 플로우

```
1. 결측치 처리 선택
   ↓
2. 정규화 방법 선택
   ↓
3. 이상치 처리 옵션 설정
   ↓
4. 특성 선택 (checkbox)
   ↓
5. 시간 윈도우 설정 (T)
   ↓
6. [Preview] → 그래프 비교
   ↓
7. [Apply] → 화면 3
```

#### 백엔드 API 초안

```python
# POST /api/preprocess
{
  "data_id": "uuid-xxx",
  "missing_strategy": "interpolate",  # remove/interpolate/forward_fill
  "normalization": "z_score",  # min_max/z_score/robust
  "outlier_method": "iqr",  # iqr/z_score
  "outlier_threshold": 3.0,
  "selected_features": ["P013_pressure", "P013_pt1", ...],
  "time_window": 192
}

# Response
{
  "preprocessed_data_id": "uuid-yyy",
  "shape": [950, 192, 15],  # After removing N-missing rows + time window
  "stats_before": {...},
  "stats_after": {...},
  "removed_rows": 50,
  "removed_features": 4
}
```

#### 추천 시각화 라이브러리

| 기능 | 추천 | 대안 |
|------|------|------|
| 선 그래프 | Plotly | Recharts |
| 히스토그램 | Plotly | Visx |
| 슬라이더 | rc-slider | React Slider |

#### UX 주의사항

- 결측치 비율에 따라 전략 추천 (>30% → 제거 권장)
- 정규화 전/후 분포도 표시
- 선택된 특성 개수 표시 (e.g., "15개 선택")
- 이상치 제거 후 데이터 손실률 경고

---

### F.3 화면 3: 인과 그래프 시각화 & 분석

#### 목적
PCMCI/NOTEARS로 학습한 인과 DAG를 시각화하고 변수 간 관계 분석

#### 핵심 기능
- **DAG 시각화**: 노드 (변수), 엣지 (인과 관계) 시각화
- **엣지 필터링**: 인과도 threshold (p-value, confidence) 조절
- **노드 정보**: 클릭 시 변수 통계 + 영향도 표시
- **경로 분석**: 특정 노드에서 시작한 인과 경로 추적
- **엣지 가중치**: 엣지 굵기/색상으로 인과도 표현

#### 주요 위젯/컴포넌트

| 컴포넌트 | 타입 | 목적 |
|----------|------|------|
| **DAG Visualization** | Interactive Graph | Cytoscape / Force Layout |
| **Threshold Slider** | Slider | p-value 또는 confidence threshold |
| **Edge List Table** | DataTable | edge (source → target) 목록 + 가중치 |
| **Node Info Panel** | Collapsible | 선택 노드의 상세 정보 |
| **Path Trace** | Network animation | 영향 경로 시각화 |
| **Export Button** | Button | DAG 이미지/JSON 다운로드 |

#### 입력/출력

| 항목 | 형식 | 설명 |
|------|------|------|
| 입력 | Preprocessed DataFrame | (N, T, D) 정규화 시계열 |
| 출력 | DAG JSON + Figure | 인과 관계 구조 + PNG 시각화 |

#### 사용자 액션 플로우

```
1. 전처리 데이터로부터 PCMCI 학습 [Backend]
   ↓
2. DAG 시각화
   ↓
3. [Threshold Slider] 조절 → 엣지 필터
   ↓
4. [노드 클릭] → 노드 정보 패널 표시
   ↓
5. [Path Trace] 버튼 → 특정 경로 강조
   ↓
6. [Export] → DAG 다운로드 또는 화면 4로 진행
```

#### 백엔드 API 초안

```python
# POST /api/learn_causal_dag
{
  "preprocessed_data_id": "uuid-yyy",
  "method": "pcmci",  # pcmci/notears/fci
  "ci_test": "time_series_based_correlation",  # or other CIT
  "alpha": 0.05  # significance threshold
}

# Response
{
  "dag_id": "uuid-zzz",
  "adjacency_matrix": [[0, 1, 0], [0, 0, 1], ...],  # (D, D)
  "edge_weights": [[0.0, 0.85, 0.0], ...],  # confidence scores
  "edge_pvalues": [[1.0, 0.001, 1.0], ...],
  "features": ["P013_pressure", "P013_pt1", ...],
  "graph_json": {...},  # Cytoscape format
  "precision": 0.82,  # vs ground truth (if available)
  "recall": 0.78
}

# GET /api/dag/{dag_id}/export
# 반환: PNG image (DAG 그래프)
```

#### 추천 시각화 라이브러리

| 기능 | 추천 | 대안 |
|------|------|------|
| 그래프 레이아웃 | Cytoscape.js | vis.js, D3.js |
| 교차형 레이아웃 | Cytoscape Hierarchy | Dagre |
| 애니메이션 | Framer Motion | React Spring |
| 테이블 | ag-Grid | Tanstack Table |

#### UX 주의사항

- 초기 로딩 시간 길 수 있음 (PCMCI 계산 시간) → 진행바 표시
- 무수히 많은 엣지 시 threshold 자동 조정 제안
- 노드 라벨 겹침 방지 (자동 정렬)
- 모바일 환경: 터치 제스처 지원 (확대/축소)

---

### F.4 화면 4: 이상탐지 결과 대시보드

#### 목적
학습된 모델로 이상탐지, 실시간 또는 배치 예측 결과 표시

#### 핵심 기능
- **예측 결과 테이블**: Cycle ID, 불량 확률, 불량 유형, 신뢰도
- **임계값 조절**: Risk threshold 슬라이더 (동적 업데이트)
- **성능 지표**: AUROC, Precision, Recall, FAR@Recall, Cost-aware score
- **혼동 행렬**: True Positive/False Positive 등 분류 결과
- **시간별 추이**: 시간/날짜에 따른 불량율 트렌드

#### 주요 위젯/컴포넌트

| 컴포넌트 | 타입 | 목적 |
|----------|------|------|
| **Predictions Table** | DataTable | Cycle 별 예측 결과 (정렬, 필터 가능) |
| **Threshold Slider** | Slider | 불량 확률 threshold (0~1) |
| **Metric Cards** | Card Grid | AUROC, Precision, Recall, FAR, Cost |
| **Confusion Matrix Heatmap** | Heatmap | TP/FP/TN/FN 분포 |
| **Trend Chart** | Line Chart | 시간별 불량율 추이 |
| **Defect Type Distribution** | Bar Chart | 불량 유형별 건수 |
| **Alert List** | Collapsible List | 미검출/과검출 사례 강조 |

#### 입력/출력

| 항목 | 형식 | 설명 |
|------|------|------|
| 입력 | Test DataFrame + Trained Model | (N_test, T, D) |
| 출력 | Predictions JSON | Cycle-wise predictions + metrics |

#### 사용자 액션 플로우

```
1. 모델 선택 (pre-trained or custom)
   ↓
2. 테스트 데이터 예측 실행 [Backend]
   ↓
3. 결과 테이블 표시
   ↓
4. [Threshold Slider] 조절 → metrics 실시간 업데이트
   ↓
5. [Cycle 클릭] → 화면 5 (설명 화면)로 이동
   ↓
6. [Export Predictions] → JSON/CSV 다운로드
```

#### 백엔드 API 초안

```python
# POST /api/predict
{
  "preprocessed_data_id": "uuid-yyy",
  "model_id": "pressfuse_v1",  # or gnn_causal_v1
  "threshold": 0.5  # optional, defaults to 0.5
}

# Response
{
  "predictions_id": "uuid-aaa",
  "cycle_ids": [1, 2, 3, ...],
  "defect_proba": [0.1, 0.87, 0.2, ...],  # (N_test,)
  "defect_type": [null, "pressure_drop", null, ...],  # (N_test,)
  "anomaly_confidence": [0.3, 0.92, 0.1, ...],  # (N_test,)
  "metrics": {
    "auroc": 0.9823,
    "precision": 0.94,
    "recall": 0.96,
    "far_at_recall_0.95": 0.023,
    "cost_aware_score": 0.91
  },
  "confusion_matrix": [[TN, FP], [FN, TP]],
  "alerts": [
    {"cycle_id": 42, "issue": "high_false_negative_risk", "proba": 0.1},
    ...
  ]
}

# GET /api/predictions/{predictions_id}/metrics
# 임계값 변경 시 metrics 재계산
```

#### 추천 시각화 라이브러리

| 기능 | 추천 | 대안 |
|------|------|------|
| 테이블 | ag-Grid | Tanstack Table |
| 히트맵 | Plotly | Visx |
| 라인 차트 | Recharts | Chart.js |
| 슬라이더 | rc-slider | React Slider |

#### UX 주의사항

- 대량 예측 > 10K cycles: 페이지네이션 필수
- Threshold 조절 시 실시간 metrics 업데이트 (debounce 필요)
- 빨간색으로 고위험 항목 강조
- "임계값 최적화" 자동 제안 (FAR < 5% 달성)

---

### F.5 화면 5: SHAP/Attention 설명 화면

#### 목적
특정 Cycle의 불량 원인을 SHAP + Attention으로 설명

#### 핵심 기능
- **SHAP Force Plot**: 각 변수가 불량 확률에 미친 영향
- **Attention Heatmap**: 시간 단계별 어텐션 가중치
- **Feature Importance Bar**: 가장 영향 큰 변수 TOP 10
- **시간대별 분석**: 어느 시점에 문제 신호 나타났는지 표시
- **비교 분석**: 정상 Cycle과 불량 Cycle 비교

#### 주요 위젯/컴포넌트

| 컴포넌트 | 타입 | 목적 |
|----------|------|------|
| **SHAP Force Plot** | Interactive Plot | 각 변수의 기여도 시각화 |
| **Feature Importance Bar** | Bar Chart | Top 10 features by SHAP |
| **Attention Heatmap** | 2D Heatmap | Time × Attention weights |
| **Time Series Overlay** | Line Chart | Original signal + highlight critical zone |
| **Comparison Selector** | Dropdown | 정상 Cycle 선택 후 비교 |
| **Explanation Text** | Markdown | 자동 생성 설명 문구 |

#### 입력/출력

| 항목 | 형식 | 설명 |
|------|------|------|
| 입력 | Cycle ID + Model | 특정 cycle의 입력 시계열 + 훈련된 모델 |
| 출력 | Explainability JSON | SHAP values, Attention weights, Text explanation |

#### 사용자 액션 플로우

```
1. 예측 대시보드에서 Cycle 클릭
   ↓
2. 설명 화면 로드 [Backend SHAP/Attention 계산]
   ↓
3. Force plot, importance bar, heatmap 표시
   ↓
4. [시간대별 분석] 클릭 → 특정 구간 강조
   ↓
5. [비교] → 정상 cycle 선택
   ↓
6. [설명 보고서] → 화면 6 (자동 리포트) 이동
```

#### 백엔드 API 초안

```python
# POST /api/explain
{
  "cycle_id": 42,
  "preprocessed_data_id": "uuid-yyy",
  "model_id": "pressfuse_v1",
  "explain_method": "shap"  # or attention, or both
}

# Response
{
  "explanation_id": "uuid-bbb",
  "cycle_id": 42,
  "prediction": {
    "defect_proba": 0.87,
    "defect_type": "pressure_drop"
  },
  "shap_values": {  # (T, D) SHAP values
    "P013_pressure": [...],
    "P013_pt1": [...],
    ...
  },
  "shap_base_value": 0.15,  # baseline probability
  "attention_weights": {...},  # (num_heads, T, T)
  "top_features": [  # sorted by |SHAP|
    {"feature": "P013_pressure", "shap_mean": 0.42},
    {"feature": "P013_vacuum", "shap_mean": 0.35},
    ...
  ],
  "critical_time_range": [100, 150],  # zero-indexed
  "explanation_text": "불량 원인: 압력 저하(T=100-150)로 인한 진공 불안정"
}
```

#### 추천 시각화 라이브러리

| 기능 | 추천 | 대안 |
|------|------|------|
| SHAP Plot | shap (Python) + custom React wrapper | LIME, IntGrad |
| Heatmap | Plotly | Visx, react-grid-heatmap |
| 라인 차트 | Recharts | Chart.js |
| 텍스트 강조 | React + CSS | Highlight.js |

#### UX 주의사항

- SHAP 계산 시간 길 수 있음 (GPU 필요) → 진행바 표시
- Attention 많은 헤드 (num_heads > 8): 평균 또는 선택 가능하게
- 색상: 빨간색 (불량 기여), 파란색 (정상 기여)
- 시간 축 레이블: UTC 또는 local 시간 (공정 맥락에 맞게)
- 비교 모드: Side-by-side 또는 overlay 선택 가능

---

### F.6 화면 6: 불량 전파 경로 추적 화면

#### 목적
인과 DAG 기반으로 불량이 어떻게 전파되는지 단계별 추적

#### 핵심 기능
- **경로 애니메이션**: DAG 상에서 불량의 인과 전파 시뮬레이션
- **단계별 설명**: 각 변수 간 전파 메커니즘 설명
- **역추적 (Root Cause)**: 불량 아래로부터 원인 역추적
- **영향 범위**: 특정 변수 변화의 하류 영향 계산
- **시나리오 분석**: "만약 변수 X가 정상이었다면?" 시뮬레이션

#### 주요 위젯/컴포넌트

| 컴포넌트 | 타입 | 목적 |
|----------|------|------|
| **Causality DAG** | Interactive Graph | 인과 구조 + 경로 강조 |
| **Path Animation** | SVG Animation | 불량 전파 애니메이션 |
| **Root Cause Panel** | Collapsible | 원인 변수 리스트 (역추적) |
| **Impact Analysis** | Heatmap/Bar | Variable intervention 효과 |
| **Scenario Simulator** | Form + Output | "If X fixed, defect prob = ?" |
| **Recommendation Box** | Alert/Card | 개선 제안 (어떤 변수 조정?) |

#### 입력/출력

| 항목 | 형식 | 설명 |
|------|------|------|
| 입력 | Cycle + DAG + Model | 특정 cycle의 상황 + 인과 구조 |
| 출력 | Propagation Path JSON | 경로 정보 + 시나리오 결과 |

#### 사용자 액션 플로우

```
1. 불량 Cycle 설정 (기본값: 이전 화면 cycle)
   ↓
2. DAG 시각화 + 불량 경로 강조
   ↓
3. [Root Cause Analysis] → 역추적 결과 표시
   ↓
4. [Scenario If...] 폼 입력
   ↓
5. [Simulate] → "변수 X를 Y로 조정 시 불량확률 Z%로 감소"
   ↓
6. [권장 조정 값] 자동 계산 및 제안
   ↓
7. [보고서 생성] → 화면 7
```

#### 백엔드 API 초안

```python
# POST /api/causal_propagation
{
  "cycle_id": 42,
  "dag_id": "uuid-zzz",
  "model_id": "pressfuse_v1",
  "original_values": {  # 실제 cycle 값
    "P013_pressure": 85.5,
    "P013_pt1": 180.2,
    ...
  }
}

# Response
{
  "propagation_id": "uuid-ccc",
  "original_defect_proba": 0.87,
  "root_causes": [  # 역추적 결과
    {"variable": "P013_pressure", "influence_score": 0.65, "direction": "decreased"},
    {"variable": "P013_vacuum", "influence_score": 0.42, "direction": "increased"},
    ...
  ],
  "propagation_path": [  # Forward path
    {"step": 1, "variable": "P013_pressure", "change": "↓", "defect_delta": 0.25},
    {"step": 2, "variable": "P013_vacuum", "change": "↑", "defect_delta": 0.18},
    ...
  ],
  "scenario_results": [
    {
      "scenario": "If P013_pressure increased by 5%",
      "new_defect_proba": 0.72,
      "improvement": "↓ 15%"
    },
    ...
  ],
  "optimal_adjustment": {
    "variable": "P013_pressure",
    "target_value": 90.0,
    "expected_defect_proba": 0.15,
    "feasibility": "high"
  }
}

# POST /api/intervention_simulation
{
  "cycle_id": 42,
  "dag_id": "uuid-zzz",
  "model_id": "pressfuse_v1",
  "intervention": {  # Causal intervention (do-calculus)
    "P013_pressure": 90.0,
    "P013_temp_setpoint": 185  # Optional
  }
}

# Response
{
  "simulated_defect_proba": 0.15,
  "improvement_from_original": 0.72,  # 100% × (0.87 - 0.15) / 0.87
  "causal_effect": {
    "direct_effect": 0.68,
    "indirect_effect": 0.19,
    "total_effect": 0.87
  }
}
```

#### 추천 시각화 라이브러리

| 기능 | 추천 | 대안 |
|------|------|------|
| 그래프 + 애니메이션 | Cytoscape + Framer Motion | D3, Visx |
| Heatmap | Plotly | react-grid-heatmap |
| 폼 입력 | React Hook Form | Formik |
| 시뮬레이션 결과 | Recharts + Cards | Chart.js |

#### UX 주의사항

- 애니메이션 너무 빠르지 않게 (2-3초/단계)
- 경로 여러 개 시: 중요도 순서로 표시 또는 선택 가능
- 시나리오 입력: 실 공정 범위 내로 제약 (e.g., pressure 60-99 kgf/cm²)
- 최적값 제안: confidence interval 또는 feasibility score 함께 표시

---

### F.7 화면 7: 자동 리포트 생성 화면

#### 목적
분석 전체를 PDF/HTML 보고서로 자동 생성

#### 핵심 기능
- **리포트 템플릿 선택**: 논문 스타일, 엔지니어 스타일, 관리자 대시보드
- **섹션 선택**: 포함할 내용 체크박스 (Executive Summary, Technical Details, Recommendations)
- **커스터마이징**: 로고, 제목, 회사명 등 메타데이터 입력
- **다운로드**: PDF (ReportLab), Excel (XLSX), HTML (Jinja2)
- **공유**: 이메일 또는 클라우드 링크 생성

#### 주요 위젯/컴포넌트

| 컴포넌트 | 타입 | 목적 |
|----------|------|------|
| **Template Selector** | Radio / Card Grid | 3-4가지 리포트 템플릿 |
| **Section Checklist** | Checkbox Group | 포함할 섹션 선택 |
| **Metadata Form** | Form | 회사명, 엔지니어 이름, 날짜 등 |
| **Preview Panel** | Collapsible | 최종 리포트 미리보기 |
| **Download Buttons** | Button Group | PDF / XLSX / HTML 다운로드 |
| **Share Options** | Radio/Input | 이메일 / 클라우드 링크 |

#### 입력/출력

| 항목 | 형식 | 설명 |
|------|------|------|
| 입력 | 분석 전체 메타데이터 | Cycle, predictions, explanations, DAG, scenarios |
| 출력 | PDF / XLSX / HTML | 최종 보고서 (다운로드 또는 이메일) |

#### 사용자 액션 플로우

```
1. [리포트 생성] 클릭
   ↓
2. 템플릿 선택 (논문/엔지니어/관리자)
   ↓
3. 포함할 섹션 선택
   ↓
4. 메타데이터 입력 (회사, 엔지니어, 날짜)
   ↓
5. [미리보기] 클릭
   ↓
6. PDF/XLSX/HTML 다운로드 선택
   ↓
7. [생성 및 다운로드] OR [이메일로 전송]
```

#### 백엔드 API 초안

```python
# POST /api/generate_report
{
  "template": "engineer",  # academic / engineer / manager
  "sections": ["summary", "predictions", "explanations", "recommendations"],
  "cycle_ids": [42, 100, 234],  # If multiple cycles
  "metadata": {
    "company_name": "OO Semi",
    "engineer_name": "Kim, John",
    "process_line": "Press Line #1",
    "report_date": "2026-05-25",
    "logo_url": "https://..."
  },
  "output_format": "pdf"  # pdf / xlsx / html
}

# Response (streaming or long-polling)
{
  "report_id": "uuid-ddd",
  "status": "completed",  # processing / completed / error
  "download_url": "https://api.example.com/reports/uuid-ddd.pdf",
  "file_size_mb": 2.5,
  "pages": 8,
  "generated_at": "2026-05-25T14:30:00Z"
}

# 대안: 스트리밍 PDF 생성
# GET /api/reports/{report_id}/download
# 반환: application/pdf (stream)
```

#### 추천 시각화 라이브러리

| 기능 | 추천 | 대안 |
|------|------|------|
| PDF 생성 | ReportLab (Python) + Weasyprint | pdfkit |
| Excel 생성 | openpyxl or xlsxwriter | pandas + ExcelWriter |
| HTML 템플릿 | Jinja2 | Mako |

#### UX 주의사항

- 리포트 생성 2-5초 소요 가능 → 진행바 표시
- 대용량 리포트 (>50 pages): 비동기 생성 후 이메일 링크
- 템플릿별 미리보기 제공 (첫 페이지만)
- 다운로드 구간: 파일 무결성 체크 (MD5)

---

## G. 추천 시스템 아키텍처

### G.1 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend Layer                          │
│  React (TypeScript) + TailwindCSS + Plotly + ag-Grid        │
│  - 7개 화면 (Data Upload, Preprocessing, Causal DAG, etc)   │
└────────────────────────┬────────────────────────────────────┘
                         │ REST API + WebSocket
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                     Backend API Layer                        │
│     FastAPI (Python) + Pydantic + SQLAlchemy                │
│  - Upload/Preprocess API                                    │
│  - Model Training/Inference API                             │
│  - Causal discovery API (PCMCI/NOTEARS)                     │
│  - Explanation API (SHAP/Attention)                         │
│  - Report generation API                                    │
└──┬─────────────────┬──────────────────┬──────────────────┬──┘
   │                 │                  │                  │
   ↓                 ↓                  ↓                  ↓
┌────────┐   ┌─────────────┐   ┌──────────────┐   ┌───────────┐
│ML Core │   │Experiments  │   │Data Storage  │   │Task Queue │
│Serv.   │   │Management   │   │(DB + Cache)  │   │(Async)    │
└────────┘   └─────────────┘   └──────────────┘   └───────────┘
   │              │                   │                  │
   ├─→ PyTorch    ├─→ MLflow         ├─→ PostgreSQL    ├─→ Celery
   ├─→ PyG        ├─→ DVC            ├─→ Redis         │
   ├─→ SHAP       ├─→ Weights&Biases │                 └─→ RabbitMQ
   ├─→ Captum     │                   │
   └─→ Scikit     │                   │
                  │                   │
                  └─ MLops Infra
```

### G.2 아키텍처 컴포넌트 상세

#### Frontend (React TypeScript)

**역할**: 사용자 인터페이스, 데이터 시각화, 웹 소켓 통신

**기술 스택**
- **Framework**: React 18 + TypeScript
- **State Management**: Zustand (Redux 대신 경량)
- **Styling**: Tailwind CSS + shadcn/ui
- **Charts**: Plotly.js, Recharts
- **Tables**: ag-Grid (Pro, for sorting/filtering)
- **Network**: axios + React Query (react-query)
- **Build**: Vite (faster than CRA)
- **Deployment**: Vercel 또는 Netlify

**구조**
```
frontend/
├── src/
│   ├── components/
│   │   ├── DataUpload.tsx
│   │   ├── Preprocessing.tsx
│   │   ├── CausalDAG.tsx
│   │   ├── PredictionDashboard.tsx
│   │   ├── ExplanationPanel.tsx      # SHAP + Attention
│   │   ├── PropagationTracer.tsx
│   │   └── ReportGenerator.tsx
│   ├── hooks/
│   │   ├── useDataUpload.ts
│   │   ├── usePredictions.ts
│   │   ├── useExplanations.ts
│   │   └── useWebSocket.ts
│   ├── store/                       # Zustand state
│   │   ├── dataStore.ts
│   │   ├── modelStore.ts
│   │   └── uiStore.ts
│   ├── utils/
│   │   ├── api.ts                   # Axios instances
│   │   ├── formatters.ts
│   │   └── validators.ts
│   └── pages/
│       ├── Dashboard.tsx             # Main layout
│       └── DetailPanel.tsx           # Detail screens
├── public/
├── vite.config.ts
├── tsconfig.json
├── tailwind.config.js
└── package.json
```

#### Backend API (FastAPI)

**역할**: 데이터 처리, 모델 학습/추론, 인과 학습, 설명 생성

**기술 스택**
- **Framework**: FastAPI + uvicorn
- **Data Validation**: Pydantic v2
- **ORM**: SQLAlchemy 2.0 (async support)
- **Database**: PostgreSQL + psycopg3 (async)
- **Caching**: Redis (for quick responses)
- **Documentation**: Swagger UI (자동 생성)
- **Monitoring**: Prometheus + Grafana (optional)

**구조**
```
backend/
├── app/
│   ├── main.py                       # FastAPI app
│   ├── config.py                     # Settings (env variables)
│   ├── dependencies.py               # Shared dependencies (DB, auth)
│   ├── api/
│   │   ├── v1/
│   │   │   ├── routers/
│   │   │   │   ├── data.py           # POST /upload, PUT /preprocess
│   │   │   │   ├── causal.py         # POST /learn_dag
│   │   │   │   ├── models.py         # POST /predict
│   │   │   │   ├── explain.py        # POST /explain (SHAP/Attention)
│   │   │   │   ├── propagation.py    # POST /propagation, /scenario
│   │   │   │   └── reports.py        # POST /report/generate
│   │   │   └── schemas.py            # Pydantic models
│   │   └── health.py                 # /health endpoint
│   ├── models/
│   │   ├── ml_models.py              # PressFuse, GNN wrapper
│   │   ├── causal_models.py          # PCMCI/NOTEARS wrapper
│   │   └── explanation_models.py     # SHAP/Attention wrapper
│   ├── services/
│   │   ├── data_service.py           # Data loading, preprocessing
│   │   ├── causal_service.py         # DAG learning
│   │   ├── prediction_service.py     # Model prediction
│   │   ├── explanation_service.py    # SHAP/Attention
│   │   ├── propagation_service.py    # Causal propagation
│   │   └── report_service.py         # PDF/HTML generation
│   ├── db/
│   │   ├── base.py                   # Base model
│   │   ├── models/
│   │   │   ├── dataset.py            # Dataset metadata
│   │   │   ├── causal_graph.py       # DAG storage
│   │   │   ├── predictions.py        # Predictions
│   │   │   └── explanations.py       # Explanations
│   │   ├── session.py                # Database session
│   │   └── migrations/               # Alembic migrations
│   └── utils/
│       ├── logging.py
│       ├── exceptions.py
│       └── constants.py
├── tests/
├── requirements.txt
└── docker/
    ├── Dockerfile
    └── .dockerignore
```

**API Endpoints (sketch)**
```
POST   /api/v1/data/upload              # 파일 업로드
PUT    /api/v1/data/{data_id}/preprocess # 전처리
GET    /api/v1/data/{data_id}           # 메타데이터 조회

POST   /api/v1/causal/learn_dag         # PCMCI 학습
GET    /api/v1/causal/dag/{dag_id}      # DAG 조회
GET    /api/v1/causal/dag/{dag_id}/export # DAG 시각화

POST   /api/v1/models/predict           # 예측 실행
GET    /api/v1/models/predictions/{pred_id} # 예측 결과

POST   /api/v1/explain                  # SHAP/Attention 계산
GET    /api/v1/explain/{explain_id}

POST   /api/v1/propagation/analyze      # 인과 전파 분석
POST   /api/v1/propagation/simulate     # 시나리오 시뮬레이션

POST   /api/v1/reports/generate         # 리포트 생성
GET    /api/v1/reports/{report_id}/download
```

#### ML Core Service

**역할**: 모델 학습, 추론, 설명 생성 (compute-heavy)

**기술 스택**
- **Framework**: PyTorch 2.2 + PyTorch Lightning
- **Graph Neural Networks**: PyTorch Geometric (PyG)
- **Causal Discovery**: causalml, DoWhy, castle
- **Explainability**: SHAP, Captum
- **Data Processing**: pandas, polars, numpy

**구조** (별도 Python 프로세스 또는 컨테이너)
```
ml_service/
├── src/
│   ├── data/
│   │   ├── loaders.py                # PyTorch Dataset
│   │   ├── preprocess.py
│   │   └── synthpress.py
│   ├── models/
│   │   ├── pressfuse.py              # Main model
│   │   ├── causal_gnn.py             # GNN
│   │   └── heads.py
│   ├── training/
│   │   ├── trainer.py                # Lightning trainer wrapper
│   │   ├── callbacks.py
│   │   └── loss_functions.py
│   ├── causal/
│   │   ├── pcmci_wrapper.py
│   │   ├── notears_wrapper.py
│   │   └── evaluation.py             # DAG precision/recall
│   ├── explain/
│   │   ├── shap_wrapper.py
│   │   ├── attention_viz.py
│   │   ├── captum_wrapper.py
│   │   └── explanation_aggregator.py
│   ├── propagation/
│   │   ├── causal_propagation.py     # GNN forward
│   │   └── intervention_simulator.py # do-calculus
│   └── utils/
│       ├── model_io.py               # Load/save checkpoints
│       └── metrics.py                # AUROC, FAR, etc
├── notebooks/                        # Jupyter experiments
├── tests/
├── requirements-ml.txt
└── config/                           # Hydra configs
    ├── experiment/
    ├── data/
    └── model/
```

#### Data Storage

**역할**: 학습 데이터, 모델 체크포인트, 메타데이터 저장

**기술 스택**
- **Database**: PostgreSQL (메타데이터, 구조화 데이터)
- **Cache**: Redis (hot data, session)
- **File Storage**: MinIO 또는 AWS S3 (모델, 대용 데이터)
- **Data Versioning**: DVC (experiment tracking)

**저장소 구조**
```
PostgreSQL 스키마:
├── datasets (업로드된 데이터 메타 정보)
├── causal_graphs (PCMCI DAG 결과)
├── models (체크포인트 경로, 훈련 파라미터)
├── predictions (예측 결과)
├── explanations (SHAP/Attention)
├── reports (생성된 리포트 메타)
└── experiments (MLflow 연계)

S3/MinIO 버킷:
├── /datasets/
│   ├── raw/
│   └── processed/
├── /models/
│   ├── /checkpoints/
│   └── /artifacts/
├── /reports/
│   ├── /pdf/
│   └── /html/
└── /experiments/
    └── /dvc_outputs/
```

#### Task Queue & Async Workers

**역할**: 비동기 작업 (장시간 계산) 처리

**기술 스택**
- **Queue**: Celery + RabbitMQ (또는 Redis)
- **Monitoring**: Flower (Celery monitoring web UI)
- **Scheduling**: APScheduler (주기적 작업)

**작업 큐**
```
High Priority:
├── predict_batch (< 1 min)
└── explain_cycle (< 2 min)

Medium Priority:
├── learn_causal_dag (5-30 min)
├── generate_report (1-5 min)
└── propagation_analysis (2-10 min)

Low Priority:
├── data_audit (long-running)
├── benchmark_experiments (hours)
└── model_retraining (hours)
```

#### Experiments & Monitoring

**역할**: 모델 성능 추적, 재현성 관리

**기술 스택**
- **Experiment Tracking**: MLflow + Weights & Biases
- **Data Versioning**: DVC
- **Monitoring**: Prometheus + Grafana
- **Logging**: ELK Stack (Elasticsearch, Logstash, Kibana) 또는 Loki

**MLflow 통합**
```
runs/
├── run1: baseline_lstm (date)
├── run2: pressfuse_v1 (date)
├── run3: pressfuse_with_causal (date)
├── run4: gnn_propagation (date)
└── ... (생산성 높은 실험 관리)

각 run에 기록:
├── params (학습률, 배치크기, 모델 아키텍처)
├── metrics (AUROC, FAR, Precision, Recall)
├── artifacts (모델, 그래프, 테이블)
└── environment (dependencies, 환경변수)
```

### G.3 배포 전략

#### 개발 환경 (로컬)

```bash
# 1. Frontend
npm install && npm run dev      # Vite dev server (port 3000)

# 2. Backend
python -m uvicorn app.main:app --reload --port 8000

# 3. ML Service (별도 프로세스)
python -m ml_service.worker    # Celery worker

# 4. Database + Redis (Docker Compose)
docker-compose -f docker-compose.dev.yml up
```

#### 테스트 환경 (Staging)

```
Frontend: Vercel preview deployment (PR당 자동)
Backend: Docker container on AWS ECS
ML Service: Separate GPU instance
Database: RDS PostgreSQL
Cache: ElastiCache Redis
Monitoring: CloudWatch + DataDog
```

#### 프로덕션 환경 (after thesis)

```
Frontend: Vercel or CloudFront + S3
Backend: ECS + ALB (auto-scaling)
ML Service: SageMaker or k8s with GPU nodes
Database: RDS Multi-AZ
Cache: ElastiCache Cluster
Monitoring: DataDog + PagerDuty alerts
```

### G.4 기술 선택 근거

| 기술 | 선택 이유 | 대안 |
|------|---------|------|
| **React** | 풍부한 생태계, 성숙한 커뮤니티, 회사 표준 | Vue, Svelte |
| **FastAPI** | 빠른 개발, 자동 Swagger 문서, async 네이티브 | Django, Flask |
| **PyTorch Geometric** | GNN 연구용으로 최고 (SOTA 논문들), Easy API | DGL, Graph Nets |
| **SHAP + Captum** | XAI 표준화, 산업계 신뢰도 높음 | LIME, Saliency maps |
| **PostgreSQL** | 신뢰성, 복잡한 쿼리 지원, 오픈소스 | MongoDB, MySQL |
| **Celery** | 분산 작업 처리 표준, 스케일링 용이 | Huey, RQ |
| **MLflow + DVC** | 재현성, 논문 요구사항 충족 | W&B (유료), Neptune |

---

## H. 바로 다음 스프린트 액션 아이템 (Sprint 1 첫 2주)

### 우선순위 Top 10 액션 아이템 (2주 안에 완료)

| # | 액션 | 담당 | 예상 시간 | 결과물 | 완료 기준 |
|---|------|------|---------|--------|----------|
| **1** | PCMCI 라이브러리 비교 및 선택 (causalml vs castle vs DoWhy) | [You] | 4시간 | 선택 보고서 + 테스트 코드 | 3개 과 모두 테스트 후 우수 선택 |
| **2** | 합성 데이터 기반 ground truth DAG 정의 | [You] | 6시간 | `configs/data/ground_truth_dag.yaml` | 6개 변수 × 엣지 강도 정의 |
| **3** | `src/causal/` 폴더 및 기본 모듈 구조화 | [You] | 3시간 | 디렉토리 구조 + 의존성 | GitHub push |
| **4** | PCMCI 래퍼 함수 작성 (`learn_causal_dag()`) | [You] | 8시간 | `src/causal/pcmci_wrapper.py` | 단위 테스트 3개 pass |
| **5** | `tests/test_pcmci.py` 단위 테스트 작성 | [You] | 5시간 | 테스트 코드 + 커버리지 리포트 | Coverage ≥ 80% |
| **6** | 합성 데이터 기본 통계 분석 스크립트 | [You] | 4시간 | `scripts/analyze_synthetic_distribution.py` | 5개 분포도 생성 |
| **7** | PyTorch Lightning 모델 학습 파이프라인 검증 | [You] | 5시간 | MLflow run 기록 | AUROC ≥ 0.90 달성 |
| **8** | 논문 Introduction 1차 draft (1.5 페이지) | [You] | 4시간 | `paper/draft/introduction.md` | 지도교수 검토 대비 |
| **9** | 시각화 템플릿 (Plotly) 5개 기본 구성 | [You] | 5시간 | `notebooks/viz_templates.ipynb` | Cycle anomaly, DAG, Feature importance 시각화 |
| **10** | 다음 Sprint 상세 기획서 작성 | [You] | 3시간 | 이 문서 업데이트 | Backlog refinement 준비 |

### 체크리스트 형식 (Daily 추진)

**Week 1 (6월 1-7)**
- [ ] Day 1: PCMCI 비교 분석 완료 + 선택 보고서 작성
- [ ] Day 2-3: 합성 DAG 정의 + 테스트
- [ ] Day 4-5: `src/causal/` 구조화 + PCMCI wrapper 초안
- [ ] Day 6-7: 단위 테스트 + 첫 커밋 (GitHub)

**Week 2 (6월 8-14)**
- [ ] Day 8: 데이터 통계 분석 완료
- [ ] Day 9-10: 모델 학습 파이프라인 검증
- [ ] Day 11-12: 논문 Introduction draft 작성
- [ ] Day 13-14: 시각화 + 다음 기획서 최종화

### 성공 지표

- ✅ 모든 Action Item이 완료되고 GitHub에 커밋됨
- ✅ PCMCI 기본 구현이 단위 테스트를 통과함
- ✅ 논문 Introduction이 1.5 페이지 이상 작성됨
- ✅ Sprint 2 상세 기획서가 준비됨

---

## I. 결론 및 권고사항

### I.1 현재 상황 요약

본 프로젝트는 **POC 완성 단계 → 논문 핵심 기여 구현 단계**로 전환해야 합니다.

**강점**:
- 도메인 공정 이해, 합성 데이터, 기본 ML 파이프라인 완성
- 팀 문화 및 개발 인프라 우수 (pytest, MLflow, DVC)

**약점**:
- 논문 제목 "MS-CDPNet"의 핵심인 **인과 추론 부재**
- **XAI 구현 미흡** (스켈레톤 파일만 존재)
- 논문 마감까지 9개월이므로 **가장 복잡한 부분부터 병행 개발 필수**

### I.2 3가지 핵심 리스크

1. **인과 학습의 복잡성**: PCMCI는 매개변수 튜닝이 어렵고, DAG 정확도 예측 어려움 → **위크 1-2에 조기 프로토타입 필수**
2. **시간 부족**: 9개월 안에 인과+XAI+최적화+실험+논문 모두 → **스프린트별 명확한 산출물 강제**
3. **검증 데이터 부재**: 실제 제조 데이터 없음 → **합성 데이터 품질 검증 중요** (Week 1-2에 집중)

### I.3 권고 다음 액션

1. **긴급**: 이 문서를 지도교수/팀과 공유하고 **Sprint 1 계획 확정**
2. **긴급**: PCMCI 라이브러리 선택 및 **프로토타입 1개 구현**
3. **주간**: 매주 월요일 스프린트 목표 검토 + 금요일 진도 확인
4. **월간**: 월 말 새로운 아티팩트 (논문 Figure 또는 실험 결과)
5. **최종**: 2027년 1월 중순 논문 드래프트 완성 후 교수 피드백 반영 기간 확보

---

## 부록: 추가 참고 자료

### 추천 논문 & 도구

| 주제 | 논문/도구 | 링크 |
|------|---------|------|
| **Causal Discovery** | Weiss et al. "Rethinking Causal Discovery" (2021) | arXiv:2107.02737 |
| **PCMCI** | Runge et al. "Identifying causal gateways and mediators" (2015) | Nature Communications |
| **GNN** | Kipf & Welling "Semi-Supervised Classification with GCNs" (2016) | ICLR |
| **SHAP** | Lundberg & Lee "A Unified Approach to Interpreting Model Predictions" (2017) | NIPS |
| **PyG** | Fey & Lenssen "Fast Graph Representation Learning with PyTorch Geometric" (2019) | arXiv:1903.02428 |

### 관련 충북대학교 교수진

- 산업인공지능학과: 지도교수 연락처
- 반도체 관련 네트워크: 산학협력처

### 오픈소스 참고 프로젝트

- **CausalNex** (QuantumBlack/McKinsey): DAG 시각화 및 분석
- **DoWhy** (Microsoft): 인과 추론 라이브러리
- **Alibi** (Seldon): XAI 도구 모음

---

**문서 버전**: 1.0  
**작성일**: 2026년 5월 25일  
**다음 검토 예정일**: 2026년 8월 31일 (Sprint 1 종료)  
**피드백 담당**: [지도교수명]


