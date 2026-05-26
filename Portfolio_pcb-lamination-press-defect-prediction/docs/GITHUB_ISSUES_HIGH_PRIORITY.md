# GitHub Issues: PCB Press Defect Prediction High Priority Backlog

**Project**: MS-CDPNet (2026년 10월 31일 제출)  
**Created**: 2026년 5월 25일  
**Total High Priority Issues**: 8개  
**Total Estimated Effort**: ~160시간

---

## 📌 발행 순서 (우선순위)

### ISSUE #1: PCMCI 인과 탐색 기본 구현

```
Title: [Core] Implement PCMCI Causal Discovery Baseline with Synthetic DAG Validation

Background:
- PCB Press 공정의 변수 간 인과 관계 학습이 논문의 핵심
- PCMCI는 시계열 조건부 독립성 테스트 기반 인과 탐색 알고리즘
- 합성 데이터의 ground truth DAG와 비교하여 정확도 검증 필수
- 目標 대체율: Precision ≥ 75%, Recall ≥ 75%

Task:
1. 라이브러리 선택 (causalml vs DoWhy vs castle)
   - 각 라이브러리 기본 기능 테스트 (코드 1-2시간)
   - 성능 비교표 작성
   - 최적 선택 근거 문서화

2. 합성 데이터 기반 ground truth DAG 정의
   - 공정 변수 간 인과 관계 메커니즘 설명
   - configs/ground_truth_dag.yaml 작성 (변수 × 엣지 강도)
   - DAG 시각화 (Networkx)

3. PCMCI 래퍼 함수 구현 (src/causal/pcmci_wrapper.py)
   - learn_causal_dag(data, method='pcmci', ci_test='...')
   - 반환: adjacency matrix, edge weights, p-values
   - 매개변수 (tau, alpha, condind_run_mode) 설정 가능

4. 단위 테스트 작성 (tests/test_pcmci.py)
   - Ground truth DAG 로드 → PCMCI 학습 → 정확도 계산
   - 테스트 케이스 3개 minimum

5. 정확도 분석 리포트
   - Precision, Recall, F1 계산 (adjacency matrix 기준)
   - Edge direction 정확도 별도 계산
   - Precision/Recall trade-off 시각화

Acceptance Criteria:
- [ ] 라이브러리 선택 결정 및 근거 문서 완료
- [ ] config/ground_truth_dag.yaml 생성 (≥ 10개 변수, ≥ 15개 엣지)
- [ ] src/causal/pcmci_wrapper.py 구현 완료
- [ ] tests/test_*.py 3개 이상 pass rate ≥ 90%
- [ ] 정확도 결과표 생성 (precision/recall/f1)
- [ ] GitHub commit 후 PR 생성

Technical Notes:
- PCMCI 성능 최적화: tau 값 (typical: 1-3), alpha (typical: 0.01-0.05)
- CI test 선택: 'time_series_based_correlation' vs 'linear_regresssion' vs ...
- 계산 복잡도: O(D^2 * T) where D=num_variables, T=time_steps
  → 합성 데이터 (D=19, T=192)에서 ~1분 예상
- Ground truth DAG는 실제 물리 메커니즘에서 도출 (documentation 필수)

Dependencies:
- causalml, DoWhy 또는 castle (선택할 라이브러리)
- src/data/synthpress.py (합성 데이터 생성)
- numpy, pandas, scikit-learn (이미 설치)

Estimated Effort: **Large (40 hours)**

Breaking Down:
- Library selection & test: 4h
- Ground truth DAG: 6h
- PCMCI wrapper: 12h
- Unit tests: 8h
- Analysis & reporting: 10h

Timeline:
- Week 1-2 (6월 1-14): 완료 목표
- Critical path item (리스크 1)

Related Issues:
- #2 (SHAP Gradient Integration - dependency)
- #5 (Unified Explainability Pipeline - dependency)
```

---

### ISSUE #2: SHAP 그래디언트 적분 및 시각화

```
Title: [XAI] Implement SHAP Gradient Integration for Feature Importance

Background:
- 모델이 각 입력 특성(변수)에 대해 예측을 어떻게 결정했는지 설명 필요
- SHAP (SHapley Additive exPlanations)은 게임 이론 기반 설명 가능 AI
- 그래디언트 기반 SHAP은 신경망 모델에 직접 적용 가능
- 목표: 불량 확률에 대한 각 변수의 기여도 정량화

Task:
1. SHAP 라이브러리 통합
   - src/explain/shap_gradient_impl.py 작성
   - KernelExplainer 또는 GradientExplainer 선택
   - Batch 처리 지원

2. 예측 결과별 SHAP 값 계산
   - input: model, data batch (B, T, D)
   - output: SHAP values (B, T, D), base_value
   - 시간 축 aggregation (mean, max, std)

3. SHAP Force plot 생성
   - 각 샘플별 force plot (individual prediction 설명)
   - 정상 cycle 1개, 불량 cycle 3개 = 4개 plot 생성
   - PNG 저장

4. Feature Importance ranking
   - Mean absolute SHAP (변수별 평균 영향도)
   - Top 10 features 추출 및 테이블
   - 정상/불량별 상위 features 차이 분석

5. Temporal analysis
   - SHAP values를 시간축으로 분해
   - 어느 시점에서 중요한 변수가 바뀌는가?
   - 시간대별 top features 시각화

Acceptance Criteria:
- [ ] src/explain/shap_gradient_impl.py 구현 완료
- [ ] 단위 테스트 pass (정상/불량 샘플 각 1개)
- [ ] Force plot 4개 생성 및 저장
- [ ] Feature importance 테이블 생성 (CSV)
- [ ] 시간대별 분석 figure 생성
- [ ] 소요시간: SHAP 계산 < 5초/sample (성능 기준)

Technical Notes:
- GradientExplainer의 수렴성: 안정적인 그래디언트 계산을 위해
  - Baseline을 데이터 평균으로 설정 (random이 아님)
  - num_samples (GradientExplainer) = 50-100 충분
- SHAP 값의 합: Σ(SHAP_i) = model(x) - base_value (검증 용이)
- Numerical precision: float32 vs float64 (메모리 vs 정확도 trade-off)
- 시간 축 aggregation: mean이 기본, 필요시 weighted (by attention)

Dependencies:
- shap >= 0.45
- torch, torch.autograd
- src/models/pressfuse.py (모델)
- src/data/loaders.py (데이터)

Estimated Effort: **Medium (30 hours)**

Breaking Down:
- SHAP 래퍼 구현: 8h
- Force plot 생성: 5h
- Feature importance: 6h
- Temporal analysis: 7h
- Testing & refinement: 4h

Timeline:
- Week 4-5 (6월 22-7월 5): 완료 목표
- Sprint 1B의 핵심

Related Issues:
- #1 (PCMCI - 함께 결과 비교)
- #4 (Attention Viz - 통합)
- #5 (Unified Pipeline - dependency)
```

---

### ISSUE #3: Cross-Modal Attention 맵 시각화

```
Title: [Visualization] Implement Cross-Modal Attention Heatmap Extraction and Plotting

Background:
- PressFuse 모델의 Cross-modal Attention은 어느 시간 단계가 중요한지 표시
- Attention weights (num_heads, T, T)를 2D heatmap으로 가시화
- 불량 탐지와 attention pattern의 상관성 분석 필요

Task:
1. Attention weights 추출 함수 (src/explain/attention_extraction.py)
   - Forward pass 중 attention weights 캡처
   - 모든 head별로 저장 또는 average
   - output: attention tensor (num_heads, T, T)

2. Heatmap 시각화 생성 (src/explain/attention_viz.py)
   - Plotly heatmap: x-axis=key_time_step, y-axis=query_time_step
   - Color: attention weight [0, 1]
   - Title/labels: 시간단계 의미 포함 (vacuum, hot_press, cooling)
   - Annotation: 값 표시 (선택)

3. Multi-head aggregation
   - Head별 개별 heatmap 4개 + 평균 heatmap
   - 또는 평균 heatmap만 (간단)
   - 각 head마다 다른 패턴 있는지 분석

4. 시간대별 주석 추가
   - X축/Y축에 단계 표시:
     - [0-180]: Vacuum phase
     - [180-10140]: Hot press phase
     - [10140-13140]: Cooling phase
     - [13140-13200]: Release phase

5. Cycle 비교
   - 정상 cycle 2개 + 불량 cycle 2개 (총 4개 heatmap)
   - Side-by-side 배치 (2x2 subplot)
   - Attention 패턴의 차이 해석

Acceptance Criteria:
- [ ] Attention weights 추출 함수 작성 완료
- [ ] Heatmap figure (4개) 생성 및 PNG 저장
- [ ] 사용자 해석 가능한 주석/레이블 포함
- [ ] Figure quality: dpi ≥ 150, 글꼴 가독성 확인
- [ ] 소요시간: heatmap 생성 < 1초

Technical Notes:
- Attention weights 저장: torch.autograd.profiler로 hook 또는
  model.forward() 수정 시 weights return
- Heatmap 색상: viridis (상업) vs RdYlGn (흥미) vs coolwarm
  권장: 'Blues' (높을수록 진함) 또는 'RdYlBu_r'
- 정규화: attention weights는 이미 [0,1]이므로 추가 정규화 불필요
- 해석: attention이 높은 구간 = 예측에 중요한 구간
  (예: "불량인 경우 100-150 구간의 attention이 높다" → "압력 강하 감지")

Dependencies:
- torch (모델 inference)
- plotly (visualization)
- numpy (배열 처리)
- src/models/pressfuse.py (모델)
- src/data/loaders.py (데이터)

Estimated Effort: **Medium (25 hours)**

Breaking Down:
- Attention extraction: 6h
- Heatmap generation: 8h
- Multi-head handling: 4h
- Annotations & labels: 4h
- Testing & refinement: 3h

Timeline:
- Week 5-6 (6월 29-7월 12): 완료 목표
- Sprint 1B의 마무리

Related Issues:
- #2 (SHAP - 함께 설명 파이프라인 구성)
- #5 (Unified Pipeline - 통합)
```

---

### ISSUE #4: 통합 설명 가능성 파이프라인

```
Title: [Integration] Build Unified Explainability Pipeline (SHAP + Attention + Causal)

Background:
- 단일 cycle의 불량 예측을 여러 각도에서 설명
- PCMCI DAG + SHAP + Attention을 통합하여 종합 설명 제공
- 제조 엔지니어가 "왜 이 cycle이 불량인가?"에 대해 쉽게 이해
- Output: JSON 보고서 + visualization figures

Task:
1. Unified Explanation JSON 구조 설계 (src/explain/explanation_schema.py)
   ```json
   {
     "cycle_id": 42,
     "prediction": {
       "defect_probability": 0.87,
       "anomaly_type": "pressure_drop"
     },
     "causal_analysis": {
       "root_causes": [...]
     },
     "shap_analysis": {
       "top_features": [...]
     },
     "attention_analysis": {
       "critical_time_range": [100, 150]
     }
   }
   ```

2. Pipeline 함수 구현 (src/explain/pipeline.py)
   - explain(cycle_data, model, dag, threshold) → JSON
   - 각 component 통합 호출
   - 시간 계측 (performance tracking)

3. JSON → Report 변환
   - Markdown 형식으로 자동 생성
   - 각 section: Causal / SHAP / Attention
   - 자동 텍스트 요약 (e.g., "Variable X가 불량 신호, 이는 변수 Y에 영향")

4. 다중 cycle 배치 처리
   - explain_batch(cycle_ids, model, dag)
   - Parallel 처리 옵션 (torch DataLoader)
   - Progress bar

5. Unit & Integration Test
   - 각 component별 output 검증
   - 종합 pipeline 테스트 (cycle 1개)
   - 성능 테스트 (< 3초/cycle)

Acceptance Criteria:
- [ ] Schema 설계 및 validation 구현
- [ ] Pipeline function 완성
- [ ] JSON 생성 및 구조 검증
- [ ] Markdown 변환 동작 (3개 예제)
- [ ] 배치 처리 구현 (≥ 10 cycles 병렬)
- [ ] 모든 테스트 pass
- [ ] 소요시간: < 3초/cycle (single), near-linear with batch

Technical Notes:
- Causal analysis: PCMCI DAG에서 BFS로 영향 경로 추적
- SHAP-Causal fusion: "SHAP top features"를 DAG 구조에 매핑
  → "Variable A의 SHAP 값이 높은 이유는 Variable B → A의 인과 경로" 설명
- Attention-Causal fusion: Critical time range + DAG pattern
  → "이 시기의 pressure drop은 vacuum 저하로 인한 cascade" 추론
- Performance: 병렬 처리 위해 torch.nn.DataParallel 또는
  multiprocessing.Pool 활용

Dependencies:
- #1, #2, #3 완료 필수
- json, markdown libraries
- torch (parallel processing)

Estimated Effort: **Large (35 hours)**

Breaking Down:
- Schema design: 4h
- Integration function: 10h
- JSON generation: 6h
- Markdown export: 5h
- Batch processing: 6h
- Testing: 4h

Timeline:
- Week 6 (7월 6-12): 완료 목표
- Sprint 1B 마무리 ~ Sprint 2 준비

Related Issues:
- #1, #2, #3 (dependencies)
- #7 (Auto-Report와 연계)
```

---

### ISSUE #5: 불량 전파 모델 (GNN 또는 Rule-based)

```
Title: [Model] Implement Defect Propagation Predictor (GNN or Rule-based)

Background:
- PCMCI DAG에서 학습한 인과 구조를 이용하여 불량 전파 경로 예측
- 두 가지 접근법:
  1. GNN (그래프 신경망): 보다 강력, 구현 복잡도 높음
  2. Rule-based (BFS + influence score): 간단, 해석 용이
- 선택은 #1 (PCMCI 정확도) 결과에 따라 결정
- 목표: Propagation accuracy ≥ 60% (synthetic) 또는 ≥ 50% (real)

Task (Option A: GNN):
1. PyTorch Geometric (PyG) 래퍼
   - DAG adjacency matrix → DGL/PyG Graph 변환
   - GraphAttention (GAT) 또는 GraphConv (GCN) 모델 선택
   - 불량 확률 예측 head 추가

2. 전파 모델 학습 (src/models/propagation_gnn.py)
   - Input: PCMCI DAG structure, 변수별 이상드 신호
   - Output: 각 변수의 예상 영향도 (0-1)
   - Loss: MSE 또는 BCE (depends on task)

3. Inference & evaluation
   - Propagation path visualization
   - Accuracy metric: RMSE 또는 F1

Task (Option B: Rule-based):
1. BFS 기반 전파 알고리즘 (src/models/propagation_rule_based.py)
   - Root cause (high anomaly score) 식별
   - PCMCI DAG에서 BFS로 downstream 노드 탐색
   - Edge weight를 이용한 influence score 계산

2. Influence score 정의
   - score(node_i) = Σ(parent_node_j.anomaly_score × edge_weight_ji)
   - Recursive: 사이클 없으므로 topological order로 계산

3. Evaluation
   - Synthetic 불량 전파 경로와 비교
   - 예측 경로의 정확도, recall

Acceptance Criteria (공통):
- [ ] 방법 선택 (GNN vs Rule-based) 결정 및 근거
- [ ] 모델 구현 완료
- [ ] 학습 또는 파라미터 설정 완료
- [ ] 예제 3개 성공적 전파 예측
- [ ] 평가 지표 계산 (정확도 > 60% 목표)
- [ ] Visualization: 전파 경로 다이어그램

Technical Notes (GNN):
- Graph structure: DAG이므로 레이어별 구성 가능 (acyclic → easy propagation)
- Input features: 변수별 anomaly score, 과거 values
- Output: node-level prediction (각 노드의 예상 상태)
- Training data: synthetic cycles with propagated anomalies labeled

Technical Notes (Rule-based):
- Topological sort: anomaly detection 후 순서대로 계산
- Edge weight normalization: PCMCI의 p-value 또는 confidence 사용
- Threshold: influence score > 0.3이면 "significant propagation"

Dependencies:
- #1 (PCMCI DAG) 완료 필수
- GNN 선택시: torch-geometric, dgl
- Rule-based: networkx, numpy

Estimated Effort:
- GNN: **Large (50 hours)**
- Rule-based: **Medium (20 hours)**

Timeline:
- Week 7-10 (7월 13-8월 9): 완료 목표
- Sprint 2

Related Issues:
- #1 (dependency)
- #6 (Metrics - propagation accuracy)
```

---

### ISSUE #6: 평가 지표 정제 및 확장

```
Title: [Metrics] Refine and Extend Evaluation Metrics (FAR, Cost-Aware, Propagation)

Background:
- 기존 metrics.py에 FAR@Recall, cost_aware_score 있음
- 추가 필요 지표:
  1. Propagation accuracy (불량 전파 경로 정확도)
  2. Causal DAG 평가 (precision, recall, F1)
  3. Defect type 분류 성능 (per-class metrics)
- 통합된 평가 벤치마크 테이블 생성 필요

Task:
1. Causal DAG 평가 함수 (src/eval/causal_metrics.py)
   - evaluate_dag(predicted_dag, ground_truth_dag)
   - Metrics: edge precision, edge recall, edge F1
   - 또한: directed edge accuracy (방향성 올바름)
   - Output: dict with all metrics

2. Propagation accuracy 함수
   - ground truth: 인위적으로 cascade 불량 생성
   - Prediction: 모델이 예측한 영향 범위
   - Metric: path-level accuracy, node-level recall

3. Per-class defect metrics
   - Defect type별 precision, recall, F1
   - Confusion matrix per defect type
   - Worst-performing class 식별

4. Unified evaluation report generator
   - CSV 또는 JSON 으로 모든 지표 출력
   - 테이블 형태: 모델별, dataset별, metric별

5. Ablation study helper
   - compare_models(models_list, datasets_list)
   - 자동으로 모든 모델/데이터셋 조합 평가
   - 결과표 생성

Acceptance Criteria:
- [ ] Causal DAG metric 함수 구현 (3가지 이상)
- [ ] Propagation accuracy 함수 구현
- [ ] Per-class metrics 계산
- [ ] Unified report generator 완성
- [ ] 예예제: 3개 모델 × 2개 dataset 평가 테이블 생성
- [ ] 모든 metric 검증 (단위 테스트)

Technical Notes:
- DAG precision/recall:
  - Edge-level: predicted edge가 실제 edge와 일치하는 비율
  - Direction-level: edge 방향까지 올바른 비율 (별도 계산)
- Propagation accuracy:
  - Node-level: 예측 propagated nodes ⊆ ground truth nodes 확인
  - Path-level: 방향성 경로 일치 확인 (더 엄격)
- Cost-aware score 재검토: FN cost, FP cost 설정 (기존: 100, 5)

Dependencies:
- #1 (DAG), #5 (Propagation)
- numpy, sklearn.metrics

Estimated Effort: **Medium (20 hours)**

Breaking Down:
- DAG metrics: 6h
- Propagation metrics: 5h
- Per-class metrics: 4h
- Report generator: 3h
- Integration & testing: 2h

Timeline:
- Week 8-9 (7월 20-8월 2): 완료 목표

Related Issues:
- #1, #5 (dependencies)
- #9 (벤치마크에서 사용)
```

---

### ISSUE #7: Streamlit 기본 UI 구현

```
Title: [UI] Implement Basic Streamlit Dashboard (Data Upload + Causal DAG + Predictions)

Background:
- 복잡한 웹 프레임워크 대신 Streamlit으로 빠른 프로토타입
- 3개 탭: 데이터 업로드, 인과 DAG 시각화, 예측 결과
- 모바일 미사용, 데스크톱만 지원 (시간 절약)

Task:
1. Streamlit 프로젝트 초기화 (scripts/ui_v2.py)
   - 3개 탭 레이아웃 설정
   - 기본 CSS 스타일 (선택)

2. Tab 1: Data Upload
   - 파일 업로드 위젯 (CSV/Parquet)
   - 자동 스키마 인식 (또는 수동 선택)
   - 데이터 미리보기 (상위 50행)
   - 기본 통계 (row count, missing %, feature count)

3. Tab 2: Causal DAG
   - 데이터 업로드 후 PCMCI 학습 버튼
   - 결과 DAG를 Plotly/Networkx로 시각화
   - Threshold slider (p-value 또는 confidence)
   - 엣지 리스트 테이블 (source → target, weight)

4. Tab 3: Predictions & Explanations
   - 모델 선택 (dropdown: PressFuse, LSTM, ...)
   - 예측 실행 버튼
   - 결과 테이블 (cycle_id, defect_prob, defect_type, confidence)
   - Threshold slider (동적 업데이트)
   - 메트릭 카드 (AUROC, FAR, Precision, Recall)

5. 기본 기능
   - Session state 관리 (캐싱)
   - 에러 핸들링 (파일 형식 오류, 계산 실패 등)
   - 진행 상황 표시 (progress bar, spinner)

Acceptance Criteria:
- [ ] Streamlit 앱 구동 성공 (streamlit run scripts/ui_v2.py)
- [ ] 3개 탭 모두 기능 동작
- [ ] Tab 1: 파일 업로드 → 데이터 미리보기 성공
- [ ] Tab 2: PCMCI 학습 → DAG 시각화 성공
- [ ] Tab 3: 모델 로드 → 예측 → 메트릭 표시 성공
- [ ] 로컬 호스트에서 부하 테스트 (10 cycles 예측 < 3초)
- [ ] 사용자 경험 테스트 (직관적 UI)

Technical Notes:
- Session state: @st.cache_resource로 모델/DAG 캐싱
- File upload: st.file_uploader (자동 임시 저장)
- Visualization:
  - DAG: Plotly Figure로 interactive
  - Metrics: st.metric() 위젯
  - Table: st.dataframe or st.table (ag-Grid 미사용, 단순화)
- Performance:
  - 대용량 데이터 (> 100K rows) 시 샘플링
  - 모델 로드 시 캐싱 필수
- Styling: Streamlit 기본 light/dark theme 사용 (CSS override 최소)

Dependencies:
- streamlit >= 1.24
- plotly, pandas, numpy
- #1 (PCMCI), 학습된 모델 체크포인트

Estimated Effort: **Medium (25 hours)**

Breaking Down:
- Layout setup: 3h
- Tab 1 (Upload): 5h
- Tab 2 (DAG): 8h
- Tab 3 (Predictions): 6h
- Error handling & UX: 3h

Timeline:
- Week 4-5 (6월 22-7월 5): 완료 목표
- Sprint 1B

Related Issues:
- #1 (데이터/DAG), #5 (설명)
- 앞으로 모든 feature의 UI화는 이 기본 위에서
```

---

### ISSUE #8: 자동 보고서 생성 (PDF/HTML)

```
Title: [Report] Implement Auto-Report Generation Pipeline (PDF + HTML)

Background:
- 분석 결과를 제조 엔지니어가 쉽게 읽을 보고서로 변환
- 제조데이터 분석 표준 형식
- 반복 실행 가능: 새 데이터 → 자동 보고서

Task:
1. 보고서 템플릿 설계
   - Executive Summary (결과 요약)
   - Dataset Info (공정 정보)
   - Model Performance (메트릭 테이블)
   - Causal Analysis (DAG + key insights)
   - Predictions (불량 사례 3개)
   - Explanations (SHAP + Attention)
   - Recommendations (개선 제안)

2. ReportLab 기반 PDF 생성 (src/reporting/pdf_generator.py)
   - Paragraph, Table, Image, PageBreak 조합
   - 폰트, 색상, 레이아웃 설정
   - A4 페이지 × 4-5장

3. Jinja2 기반 HTML 생성 (src/reporting/html_generator.py)
   - HTML 템플릿 설계
   - 스타일: CSS embedded
   - 특성: 인터랙티브 요소 (collapsible, tabs)

4. Report generation 주함수 (src/reporting/report.py)
   - generate_report(cycles, model, dag, metrics, output_format='pdf'/'html'/'both')
   - Input: cycle IDs, 분석 결과
   - Output: report file (PDF 또는 HTML)

5. 예제 보고서 생성
   - 합성 데이터 기반 3개 예제
   - 각 형식 (PDF, HTML) 테스트

Acceptance Criteria:
- [ ] 보고서 템플릿 설계 완료
- [ ] PDF 생성 함수 구현 (ReportLab)
- [ ] HTML 생성 함수 구현 (Jinja2)
- [ ] 예제 보고서 3개 생성
- [ ] 품질 검증: PDF 정상 열림, 레이아웃 깔끔
- [ ] 성능: < 5초/보고서 생성

Technical Notes:
- ReportLab: 오픈소스, 무료, PDF 생성 표준
- Styles: Arial 또는 Helvetica (유니버설)
- Table: 데이터 많으면 여러 페이지 자동 분할
- Image 삽입: 150 DPI, PNG 형식 권장
- HTML: Bootstrap 또는 inline CSS

Dependencies:
- reportlab (PDF)
- jinja2 (HTML)
- #2, #3, #4 (설명 components)
- #6 (메트릭)

Estimated Effort: **Medium (30 hours)**

Breaking Down:
- Template design: 5h
- PDF generator: 10h
- HTML generator: 8h
- Main function & integration: 4h
- Testing & examples: 3h

Timeline:
- Week 9-10 (7월 27-8월 9): 완료 목표
- Sprint 2 마무리

Related Issues:
- #2-4 (설명 components input)
- #6 (메트릭 input)
```

---

### ISSUE #9: 종합 벤치마크 슈트

```
Title: [Experiment] Run Comprehensive Benchmark Suite (Synthetic + SECOM)

Background:
- 여러 모델/데이터셋을 체계적으로 평가
- 논문 결과표 생성의 기반
- Ablation study 포함 (with/without causal, XAI 등)

Task:
1. 벤치마크 시나리오 정의
   - Model: LSTM baseline, PressFuse (without causal), PressFuse (with causal)
   - Dataset: Synthetic, SECOM
   - 각 조합 3번 실행 (random seed 변경)

2. Ablation study 설계
   - All features
   - Without causal DAG (feature로만 예측)
   - Without XAI (설명 없이 예측만)
   - LSTM only (baseline)

3. 평가 지표 자동 수집
   - 각 run마다: AUROC, FAR@0.95, Precision, Recall, F1, Cost-aware
   - DAG 평가 (if applicable): precision, recall
   - 실행 시간, 메모리 사용량

4. 결과 저장 및 통계 계산
   - 모든 결과 CSV 저장 (runs_results.csv)
   - 평균, 표준편차 계산
   - 통계 유의성 검증 (t-test)

5. 최종 벤치마크 테이블 생성
   - Table: Model × Metric (논문 Figure로 사용 가능)
   - Ablation table: Feature set × Metric

Acceptance Criteria:
- [ ] 벤치마크 시나리오 정의 완료
- [ ] 모든 모델 × dataset 조합 실행 성공 (총 ≥ 9 run)
- [ ] 각 run 메트릭 수집 완료
- [ ] CSV 파일 생성
- [ ] 최종 벤치마크 테이블 생성 (LaTeX 형식 선택)
- [ ] 통계 유의성 보고 (where applicable)
- [ ] 시간: 총 실행 시간 < 4시간 (병렬 가능)

Technical Notes:
- Random seed: torch.manual_seed, numpy.random.seed 고정 (재현성)
- Running 병렬화: GPU 수에 따라 여러 프로세스
- 결과 저장: 각 run마다 checkpoint + metrics.json
- 중간 모니터링: Tensorboard 또는 로그 파일

Dependencies:
- 모든 모델 구현 완료 (#1-5)
- #6 (평가 지표)
- torch, pytorch-lightning, scikit-learn

Estimated Effort: **Large (30 hours)**

Breaking Down:
- Scenario setup: 4h
- Ablation design: 3h
- Automated pipeline: 10h
- Running experiments: 8h
- Analysis & table generation: 5h

Timeline:
- Week 11-14 (8월 10-9월 6): 완료 목표
- Sprint 3-A

Related Issues:
- #1-6 (완료 필수)
- 논문 Results 섹션 작성의 기반
```

---

## 📊 요약 표

| Issue # | Title | Effort | Timeline | Status | Dependencies |
|---------|-------|--------|----------|--------|--------------|
| #1 | PCMCI Baseline | Large (40h) | W1-2 | 🔴 CRITICAL | - |
| #2 | SHAP Gradient | Medium (30h) | W4-5 | 🔴 CRITICAL | #1 |
| #3 | Attention Viz | Medium (25h) | W5-6 | 🔴 CRITICAL | - |
| #4 | Unified Pipeline | Large (35h) | W6 | 🟠 IMPORTANT | #1,#2,#3 |
| #5 | Propagation Model | Large (50h) or Medium (20h) | W7-10 | 🟠 IMPORTANT | #1 |
| #6 | Extended Metrics | Medium (20h) | W8-9 | 🟠 IMPORTANT | #1,#5 |
| #7 | Streamlit UI | Medium (25h) | W4-5 | 🟠 IMPORTANT | #1,#2 |
| #8 | Auto-Report | Medium (30h) | W9-10 | 🟡 NICE-TO-HAVE | #2,#3,#4,#6 |
| #9 | Benchmark Suite | Large (30h) | W11-14 | 🟠 IMPORTANT | #1-8 |

**총 예상 시간**: 160-185시간 (GNN vs Rule-based 선택에 따라)

---

## 🎯 진행 전략

1. **Critical path**: #1 → #2 → #3 → #4 (순차적, 병렬 불가)
   - 이 경로상 아이템 중 하나 지연 = 전체 지연

2. **병렬 가능한 항목**:
   - #3 (Attention)과 #7 (UI)는 #1과 독립적으로 진행 가능
   - #6 (Metrics)는 #1, #5 완료 후

3. **Go/No-go 포인트**:
   - Week 3 끝 (#1 검증): PCMCI 정확도 ≥ 75% 달성 확인
   - 실패 시: Contingency plan 실행 (백업 알고리즘으로 변경)

4. **Weekly Sync**:
   - 매주 금요일: 진도 리뷰 + 블로커 확인
   - 지난주 예상 vs 실제 시간 기록

---

**문서 생성**: 2026년 5월 25일  
**형식**: GitHub Issues 템플릿  
**다음 단계**: 각 이슈를 GitHub repo에 등록 (label: High Priority, 마일스톤 설정)


