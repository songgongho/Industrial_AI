# 🚀 GitHub Issues 빠른 시작 가이드

**생성일**: 2026년 5월 25일  
**대상**: 9개 High Priority Issues (160-185시간)  
**용도**: 개발팀 협업 / 진도 추적 / 스프린트 계획

---

## 📌 이슈 구조 한눈에 보기

```
┌─────────────────────────────────────────────────────────┐
│        Critical Path (순차적 진행 필수)                  │
├─────────────────────────────────────────────────────────┤
│  #1: PCMCI Baseline         (40h, Week 1-2) ← 가장 중요!
│    ↓ (dependency)
│  #2: SHAP Gradient          (30h, Week 4-5)
│    ↓ (dependency)
│  #3: Attention Viz          (25h, Week 5-6) [병렬 가능]
│    ↓ (dependency)
│  #4: Unified Pipeline       (35h, Week 6)
│    ↓ (dependency)
│  #9: Benchmark Suite        (30h, Week 11-14)
└─────────────────────────────────────────────────────────┘

병렬 진행 가능:
- #5 (Propagation): Week 7-10 (GNN 50h or Rule-based 20h)
- #6 (Metrics): Week 8-9 (20h)
- #7 (StreamUI): Week 4-5 (25h) ← #1과 독립
- #8 (Report): Week 9-10 (30h)
```

---

## 🎯 일주일별 이슈 로드맵

### Week 1-2 (6월 1-14): #1 PCMCI 기본
```
🟢 Target: PCMCI 정확도 결정 (Go/No-go 포인트)
🟢 Owner: [You]
🟢 Effort: 40 hours

Daily Breakdown:
  Day 1-2 (Mon-Tue): Library selection (4h)
  Day 3-5 (Wed-Fri): PCMCI implementation (12h)
  Day 6-10 (W2): Parameter tuning (16h)
  Day 11-14 (W2): Testing & reporting (8h)

Success Criteria:
  ✅ PCMCI test pass rate ≥ 90%
  ✅ Accuracy ≥ 75% (goal) or ≥ 65% (acceptable)
  ✅ GitHub commit + PR created
  ✅ Decision: Continue full scope or Pivot to contingency
```

### Week 3 (6월 15-21): 병렬 준비
```
🟡 #1 최종 마무리 (review & merge)
🟡 #3 Attention 시작 (parallel to #2 준비)
🟡 #7 StreamUI 시작 가능
```

### Week 4-6 (6월 22-7월 12): Sprint 1B
```
🟢 #2 SHAP (30h, W4-5)
🟢 #3 Attention (25h, W5-6) [overlap possible]
🟢 #7 StreamUI (25h, W4-5) [parallel]

Expected Output:
  ✅ SHAP force plots (4개)
  ✅ Attention heatmaps (4개)
  ✅ Streamlit demo (3 탭 동작)
  ✅ 논문 Figure 10개
```

### Week 6 (7월 6-12): 통합
```
🟢 #4 Unified Pipeline (35h)

Expected Output:
  ✅ Explanation JSON schema
  ✅ Markdown report generator
  ✅ Integration test pass
```

### Week 7-10 (7월 13-8월 9): Sprint 2
```
🟢 #5 Propagation Model (50h GNN 또는 20h Rule-based)
🟡 #6 Extended Metrics (20h, W8-9)
🟡 #8 Auto-Report (30h, W9-10)

Decision Point (Week 7):
  - GNN 구현 성공 → 계속
  - GNN 어려움 → Rule-based로 pivot
```

### Week 11-14 (8월 10-9월 6): Sprint 3-A
```
🟢 #9 Benchmark Suite (30h)

Expected Output:
  ✅ 3 models × 2 datasets 벤치마크 완료
  ✅ 메트릭 테이블 (논문 Table 1)
  ✅ Ablation study 결과
```

---

## 📊 이슈 우선순위 판정 기준

### 🔴 CRITICAL (즉시 시작)
- **#1**: PCMCI 정확도가 전체 논문 기여도 결정
- **Go/No-go 포인트** (Week 3 끝)

### 🟠 IMPORTANT (Week 4 시작)
- **#2, #3**: XAI 통합의 기초
- **#4**: 설명 파이프라인 완성
- **#5**: 불량 전파 모델
- **#9**: 벤치마크 (논문 결과표)

### 🟡 NICE-TO-HAVE (시간 남을 경우)
- **#8**: Auto-Report (웹 UI나 기본 스크립트로 대체 가능)

---

## 💻 GitHub 등록 방법

### Step 1: Issue 생성
```bash
cd your-repo
gh issue create --title "[Core] Implement PCMCI..." \
  --body "$(cat template.md)" \
  --label "High Priority" \
  --label "Core ML" \
  --milestone "Sprint 1 (June)"
```

### Step 2: Label 설정
```
Labels:
  - High Priority (🔴 P0)
  - Core ML (기본 모델)
  - Visualization (시각화)
  - Research (연구 컴포넌트)
  - Experiment (실험)
  
Milestones:
  - Sprint 1-A (June 1-21)
  - Sprint 1-B (June 22-July 12)
  - Sprint 2 (July 13-August 9)
  - Sprint 3-A (August 10-Sept 6)
```

### Step 3: 담당자 할당
```
모두 [You] 담당 (혼자 프로젝트)
```

### Step 4: 진도 추적
```
Project Board Setup:
  Columns: [Backlog] → [In Progress] → [Review] → [Done]
  
각 이슈:
  - 시작하면: "In Progress"로 이동
  - 완료하면: PR 링크 추가 → "Review"로 이동
  - Merge 후: "Done"으로 이동
```

---

## 🔍 이슈별 체크리스트

### #1: PCMCI (40h)
```
Pre-requisite:
  [ ] Python 3.11+ 환경 설정
  [ ] causalml / DoWhy / castle 설치

Development:
  [ ] Library comparison script 작성 및 테스트
  [ ] Synthetic DAG ground truth 정의 (config yaml)
  [ ] PCMCI wrapper 기본 구조
  [ ] 첫 실행 정확도 측정
  [ ] 매개변수 튜닝 (tau, alpha)
  [ ] Unit test 3개 작성
  [ ] 결과 리포트 작성

Testing:
  [ ] pytest tests/test_pcmci.py pass_rate >= 90%
  [ ] 정상/불량 데이터 각 1개 정확도 확인

Submission:
  [ ] GitHub commit: src/causal/pcmci_wrapper.py
  [ ] GitHub commit: tests/test_pcmci.py
  [ ] GitHub commit: configs/ground_truth_dag.yaml
  [ ] PR 생성 및 리뷰 요청
  [ ] Go/No-go 결정도 PR description에 포함
```

### #2: SHAP (30h)
```
Pre-requisite:
  [ ] #1 완료 및 merged
  [ ] 학습된 모델 checkpoint 준비
  
Development:
  [ ] SHAP 라이브러리 통합 결정 (KernelExplainer vs GradientExplainer)
  [ ] Batch SHAP 계산 함수
  [ ] Force plot 생성 (4개: 정상 1, 불량 3)
  [ ] Feature importance ranking
  [ ] Temporal SHAP 분석

Testing:
  [ ] SHAP 값이 합산되어 모델 예측과 일치하는지 검증
  [ ] Figure 품질 확인 (가독성, 색상)

Output:
  [ ] src/explain/shap_gradient_impl.py
  [ ] Figure: shap_analysis/*.png (4개)
  [ ] CSV: top_features.csv
```

### #3: Attention (25h)
```
Pre-requisite:
  [ ] PressFuse 모델이 attention weights 반환하도록 수정
  
Development:
  [ ] Attention extraction hook 구현
  [ ] Heatmap generation (Plotly)
  [ ] Multi-head aggregation
  [ ] 시간대별 주석 추가
  [ ] 정상/불량 비교 (2x2 subplot)

Output:
  [ ] src/explain/attention_extraction.py
  [ ] src/explain/attention_viz.py
  [ ] Figure: attention_heatmap/*.png (4개)
```

### #4: Unified Pipeline (35h)
```
Pre-requisite:
  [ ] #1, #2, #3 모두 완료
  
Development:
  [ ] Explanation schema 설계 (JSON)
  [ ] Pipeline function (explain_single, explain_batch)
  [ ] JSON → Markdown 변환
  [ ] Integration test

Output:
  [ ] src/explain/explanation_schema.py
  [ ] src/explain/pipeline.py
  [ ] Test: 3개 cycle 예제 JSON 생성
```

### #5: Propagation Model (50h 또는 20h)
```
Decision Point: Week 7 (GNN vs Rule-based)

Option A (GNN, 50h):
  [ ] PyG graph 변환 함수
  [ ] GAT/GCN 모델 구현
  [ ] 학습 루프
  [ ] 평가 (accuracy ≥ 60%)

Option B (Rule-based, 20h):
  [ ] BFS + influence score 구현
  [ ] Topological sort
  [ ] 평가 (accuracy ≥ 60%)

Output (공통):
  [ ] src/models/propagation_*.py
  [ ] Visualization: propagation_path.png (3개 예제)
```

### #6: Extended Metrics (20h)
```
Pre-requisite:
  [ ] #1, #5 완료
  
Development:
  [ ] Causal DAG metrics (precision, recall, F1)
  [ ] Propagation accuracy metrics
  [ ] Per-class defect metrics
  [ ] Unified evaluation report generator

Output:
  [ ] src/eval/causal_metrics.py
  [ ] src/eval/propagation_metrics.py
  [ ] CSV: benchmark_results.csv (3 models × 2 datasets)
```

### #7: Streamlit UI (25h)
```
Pre-requisite:
  [ ] streamlit >= 1.24 설치
  [ ] #1 완료
  
Development:
  [ ] 3-tab 레이아웃
  [ ] Tab 1: 파일 업로드 + 미리보기
  [ ] Tab 2: PCMCI DAG 시각화
  [ ] Tab 3: 예측 결과 + 메트릭
  [ ] Session caching (성능)

Testing:
  [ ] streamlit run scripts/ui_v2.py 정상 구동
  [ ] 각 탭 기능 테스트
  [ ] 부하 테스트 (10 cycles < 3초)

Output:
  [ ] scripts/ui_v2.py
```

### #8: Auto-Report (30h)
```
Pre-requisite:
  [ ] #2, #3, #4, #6 완료
  
Development:
  [ ] Report template 설계
  [ ] ReportLab PDF generator
  [ ] Jinja2 HTML generator
  [ ] 예제 보고서 생성

Output:
  [ ] src/reporting/pdf_generator.py
  [ ] src/reporting/html_generator.py
  [ ] Example reports: reports/example_*.pdf (3개)
```

### #9: Benchmark Suite (30h)
```
Pre-requisite:
  [ ] #1-8 모두 완료 (또는 #1, #5, #6)
  
Development:
  [ ] Benchmark 시나리오 정의
  [ ] 자동 실행 스크립트
  [ ] 결과 저장 및 통계
  [ ] 최종 벤치마크 테이블

Output:
  [ ] scripts/run_benchmark.py
  [ ] results/benchmark_results.csv
  [ ] Table: model_comparison.tex (논문용)
```

---

## 📈 진도 추적 템플릿

### 주간 진도 리포트 (매주 금요일)

```markdown
## Week N Progress Report

### Completed Issues
- [x] #1 (PCMCI) - 40/40h
  - commit hash: abc1234
  - Status: Merged

### In Progress
- [ ] #2 (SHAP) - 18/30h
  - Last update: 2026-06-07
  - Blocker: 없음
  
### Planned for Next Week
- [ ] #3 (Attention) - 시작 예정

### Risk & Mitigation
- PCMCI 정확도가 예상 75%에 미치지 못하면:
  → LiNGAM으로 pivot (백업 준비)

### Time Tracking
- Planned: 40h
- Actual: 38h
- Variance: -2h (좋음)

### Confidence Score
- 일정 준수: 95% (on track)
- 품질: 85% (acceptable)
```

---

## 🎓 Issue 작성 예시

### Template for All Issues

```markdown
# [Label] Title

## Background
- 왜 필요한가?
- 현재 상태는?
- 목표는?

## Task
1. 첫번째 작업
   - 세부 항목
2. 두번째 작업

## Acceptance Criteria
- [ ] Criterion 1
- [ ] Criterion 2

## Technical Notes
- 주요 알고리즘
- 성능 고려사항

## Dependencies
- #issueN
- library >= version

## Estimated Effort
- Size: S/M/L
- Hours: Xh
- Timeline: WeekN

## Related Issues
- Related to #issueN
```

---

## ⚡ 빠른 명령어

### 로컬 사본 만들기
```bash
# 이슈 내용 저장
git checkout -b issue/1-pcmci-baseline
```

### PR 생성
```bash
gh pr create --title "Fix #1: Implement PCMCI baseline" \
  --body "Closes #1" \
  --label "Core ML" \
  --milestone "Sprint 1-A"
```

### 이슈 닫기 (PR 병합 후)
```bash
gh issue close 1 --comment "Fixed in PR #123"
```

### 진도 조회
```bash
gh issue list --label "High Priority" --state open
```

---

## 📌 주의사항

### ❌ 하지 말 것
```
1. 이슈 생성 후 PR 없이 오래 두기
   → 주 1회 이상 commit 필수

2. 하나의 이슈에서 너무 많은 작업
   → 이슈 분해 (epic 활용)

3. 테스트 없이 merge
   → All AC + test pass 확인 후

4. 문서 미기록
   → README, docstring 함께 작성
```

### ✅ 할 것
```
1. 매일 진도 기록
   → 15분 이상 작업하면 commit

2. 블로커 조기 보고
   → 문제 감지 시 즉시 issue 코멘트

3. 주간 리뷰
   → 금요일 오후 1시간 정리

4. Acceptance Criteria 엄격히
   → 모두 만족할 때까지 PR 오픈 금지
```

---

## 🏁 최종 체크리스트 (전체 이슈 완료 후)

```
[ ] 모든 9개 이슈 closed (상태: Done)
[ ] 총 시간: 160-185h 범위 내
[ ] GitHub commit 총 N개
[ ] 논문 Figure 15-20개 생성
[ ] 메트릭 테이블 3개 이상
[ ] 코드 테스트 커버리지 ≥ 80%
[ ] README.md 업데이트 (실행 방법)
[ ] 논문 draft 완성도 80% 이상
```

---

**생성일**: 2026년 5월 25일  
**버전**: v1.0  
**다음 스텝**: GitHub repo에 실제 이슈 등록 (6월 1일 오전)


