# Product Requirements Document (PRD)
## PCB Press Anomaly Detection & Explainability Demo System

**문서 버전**: 1.0  
**작성일**: 2026년 5월 25일  
**대상**: MS-CDPNet 논문 연구용 프로토타입  
**타겟 배포**: 2026년 10월 31일

---

## 1. Executive Summary

### 1.1 제품명
**"PCB Press Anomaly Detection & Explainability Demo"** (이하 "PressXAI")

### 1.2 제품 개요
PressXAI는 반도체 PCB 적층(Lamination) 공정의 불량을 **실시간으로 탐지**하고, **인과 그래프 기반으로 설명**하는 웹 기반 분석 시스템입니다. 제조 엔지니어와 연구자가 공정 불량의 원인을 신속하게 파악할 수 있도록 지원합니다.

### 1.3 핵심 가치 제안
```
❌ 문제:
  - 불량 원인을 찾는 데 며칠 소요
  - AI 모델의 "블랙박스" 결정으로 엔지니어 신뢰도 낮음
  - 공정 변수 간 인과관계 파악 어려움

✅ 솔루션 (PressXAI):
  - 1단계 분석으로 불량 원인 파악 (< 5초)
  - 그래프 기반 인과 경로 시각화
  - SHAP + Attention으로 "왜" 불량인지 설명
  - 엔지니어가 신뢰하고 행동 가능한 인사이트 제공
```

---

## 2. 제품 목적 & 비전

### 2.1 짧은 목적 (Elevator Pitch)
"공정 엔지니어가 불량을 **즉시 원인 분석**하고 **개선 조치를 신속하게 실행**할 수 있도록 돕는 설명 가능한 AI 대시보드"

### 2.2 중기 비전 (12개월)
```
Phase 1 (현재, 2026-10):
  ✅ 논문 및 데모 시스템 완성
  ✅ 학술 발표 & 업계 인정

Phase 2 (미래, 2027+):
  🔄 실제 제조 현장 파일럿 (3개월)
  🔄 Digital Twin 연계
  🔄 모바일 앱 출시

Vision:
  "모든 반도체 공정에 설명 가능한 AI를 보급"
```

### 2.3 전략적 초점
1. **설명 가능성 (Explainability)**: AI 의사결정 투명화
2. **인과성 (Causality)**: 단순 상관관계가 아닌 진정한 원인 파악
3. **실무성 (Practicality)**: 엔지니어가 당장 쓸 수 있음
4. **재현성 (Reproducibility)**: 완전 공개 소스, GitHub 배포

---

## 3. 사용자 유형 (Personas)

### Persona 1: 공정 엔지니어 (Process Engineer)
```
이름: "현장 김엔지니어"
역할: PCB Press Line 담당, 5년 경력
목표:
  - "이 cycle이 왜 불량인지 30초 안에 알고 싶어"
  - "압력이 문제? 온도가 문제? 진공? 한눈에 봤으면"
  
행동:
  - 매일 아침 결과 리포트 확인
  - 불량 발생 시 즉시 원인 분석
  - 유선 또는 모바일로 접근 (데스크톱 선호)

Pain Points:
  - 현재: 엑셀 + 경험으로만 분석
  - 데이터 기반 의사결정 어려움
  - AI 결과를 신뢰하기 어려움 ("왜 이렇게?" 질문)

Success:
  - 불량 원인 명확히 파악
  - 대응 시간 < 30초
  - 정확도 ≥ 80%
```

### Persona 2: 연구자 / AI 엔지니어 (Researcher)
```
이름: "박사 이연구자"
역할: 대학 산업 AI 학과, 논문 작성 중
목표:
  - 인과 탐색 + GNN 성능 검증
  - XAI 방법론 비교 (SHAP vs LIME)
  - 논문 Figure & Table 생성

행동:
  - 주간 1-2회 상세 분석
  - 모델 파라미터 변경 후 재실험
  - 코드 수정 및 테스트

Pain Points:
  - 성능 재현 어려움
  - 설정 파일 복잡함
  - Jupyter로 수동 분석 중

Success:
  - 재현 가능성 100% (GitHub)
  - 모든 실험 결과 관리 (MLflow)
  - 논문 품질 높음
```

### Persona 3: 관리자 / 운영 담당자 (Operations)
```
이름: "부장 정관리자"
역할: 공장 품질 관리 부서 장
목표:
  - "월별 수율 목표 달성"
  - "불량 트렌드 파악"
  - "팀의 성과 관리"

행동:
  - 주간 1회 대시보드 조회
  - 월간 리포트 생성 및 상보

Pain Points:
  - 데이터 시각화 부족
  - KPI 추적 어려움
  - 자동 리포트 필요

Success:
  - 월별 불량 추이 그래프 제공
  - 주요 불량 원인 TOP 3
  - 개선 효과 정량 보고
```

---

## 4. 핵심 사용 시나리오 (Use Cases)

### UC-1: 불량 Cycle 원인 분석 (Primary)

**시나리오**:
```
시간: 2026-08-15 09:30 (Press Line #1)

상황:
  - PCB 100개 생산 중
  - Cycle #4237에서 불량 감지 (AOI 검사)
  - 긴급 분석 필요

사용자: 공정 엔지니어 김oo

Step 1: UI 접속
  └─ [Upload] 탭 → "cycle_4237.csv" 파일 선택
  
Step 2: 자동 분석
  └─ [Causal DAG] 탭 → "PCMCI 학습" 버튼 클릭
  └─ 1초 내 DAG 시각화 완료

Step 3: 투명 설명
  └─ [Predictions] 탭 → Cycle #4237 선택
  └─ 결과: DEFECT PROBABILITY = 0.87 (확실)
  └─ ANOMALY TYPE = "Pressure Drop" (90% 신뢰)
  
Step 4: 근본 원인 파악
  └─ [Explanations] 탭 → SHAP + Attention 분석
  └─ "변수 HPPRESSPV가 -15% 감소"
  └─ "이로 인해 VACUUM이 쉬프트 (cascade)"
  └─ "T=100-150초 구간에서 critical"

Step 5: 액션 결정
  └─ 엔지니어: "아, 압력 센서가 문제겠네!"
  └─ 즉시 기술팀 호출 → 센서 교체
  └─ 생산 재개 (15분 내 해결)

결과: ✅ 수율 유지, 근본 원인 즉시 파악
```

### UC-2: 인과 구조 학습 & 시각화

**시나리오**:
```
시간: 2026-09-01 14:00 (연구 중)

사용자: 연구자 박oo
목적: 논문 Figure 3 (Causal DAG) 생성

Step 1: 데이터 준비
  └─ [Upload] → 합성 데이터 1000 cycles 로드
  
Step 2: DAG 학습
  └─ [Causal DAG] → PCMCI 실행
  └─ 매개변수: tau=2, alpha=0.05

Step 3: 결과 검증
  └─ DAG precision = 0.82 (합성 ground truth vs)
  └─ DAG recall = 0.78

Step 4: 시각화 및 저장
  └─ [Export] → PNG + SVG 다운로드
  └─ Networkx 그래프 (20개 노드, 35개 엣지)

Step 5: 논문 기재
  └─ "공정 변수 간 인과 구조를 PCMCI로 학습"
  └─ Figure 해석: "HPPRESSPV ← Equipment control"
  
Activity: 1시간 작업 완료, Figure 건설적으로 생성
```

### UC-3: 모델 성능 벤치마크 & 비교

**시나리오**:
```
시간: 2026-09-10 10:00 (실험 중)

사용자: 연구자 박oo
목적: LSTM vs PressFuse vs PressFuse+Causal 비교

Step 1: 벤치마크 데이터 준비
  └─ Synthetic dataset 500 cycles
  └─ SECOM public dataset

Step 2: 모델 로드 및 평가
  └─ [UI] → "Benchmark" 섹션
  └─ 3개 모델 자동 로드
  
Step 3: 메트릭 계산
  └─ AUROC, FAR@Recall=0.95, Cost-aware score
  └─ 결과를 실시간으로 Table로 노출
  
Step 4: 결과 다운로드
  └─ 벤치마크_results.csv
  └─ 모든 실험이 MLflow에 기록

Activity: 시간 절약, 논문 Table 1 자동 생성
```

### UC-4: 자동 리포트 생성 & 배포

**시나리오**:
```
시간: 2026-09-20 18:00 (주간 사무 자동화)

사용자: 관리자 정oo
목적: 주간 불량 분석 리포트 생성 & 팀에 배포

Step 1: 주간 데이터 취합
  └─ [Upload] → 월-금 모든 cycle 데이터
  
Step 2: 자동 분석 & 보고서 생성
  └─ [Report] 탭 → "Auto-Generate Weekly Report"
  └─ 형식: PDF (Executive Summary 포함)
  
Step 3: 주요 findings 자동 추출
  └─ "Top 3 불량 원인"
  └─ "월대비 수율 변화"
  └─ "권장 개선 조치"
  
Step 4: 배포
  └─ PDF 자동 생성 (< 30초)
  └─ 이메일 자동 발송 (관리층)
  
Result: ✅ 수작업 제거, 일관된 보고, 시간 절약
```

---

## 5. 기능 요구사항 (Functional Requirements)

### 5.1 Core Features

#### F-1: 데이터 관리 (Data Management)

**F-1.1 데이터 업로드**
```
Requirements:
  - 지원 형식: CSV, Parquet, Excel (.xls, .xlsx)
  - 파일 크기: 최대 500MB
  - 자동 인코딩 감지 (UTF-8, EUC-KR, GB2312)
  - 스키마 자동 인식 또는 수동 설정
  
Input Validation:
  - 빈 값 감지 및 고지
  - 데이터 타입 검증 (숫자, 범주형)
  - 결측치 비율 표시 (> 30% 경고)
  
Output:
  - Preprocessed DataFrame
  - 데이터 통계: 행 수, 열 수, 결측비율, 분포
  - 다음 단계 추천
```

**F-1.2 데이터 전처리**
```
Requirements:
  - 정규화: Min-Max, Z-score, Robust scaling
  - 결측치 처리: 제거, 보간, forward/backward fill
  - 이상치 처리: IQR, Z-score 기반 마킹
  - 특성 선택: UI에서 체크박스로 선택/제외
  - 시간 윈도우 설정: T=64~512 범위
  
Output:
  - 전처리된 시계열 (B, T, D)
  - 통계 요약: 전/후 비교
```

**F-1.3 데이터 저장 & 재사용**
```
Requirements:
  - 전처리된 데이터 캐싱 (로컬 또는 서버)
  - 데이터셋 관리: 이름, 설명, 태그
  - 버전 관리: 각 전처리 config 저장
  - 재실행 가능성: 동일 config로 재생성
```

---

#### F-2: 인과 탐색 (Causal Discovery)

**F-2.1 PCMCI DAG 학습**
```
Requirements:
  - 알고리즘: PCMCI (시계열 조건부 독립성 기반)
  - 입력: 전처리된 시계열 (B, T, D)
  - 매개변수 (UI slider):
    - tau (time lag): 1-5 (default 2)
    - alpha (significance): 0.001-0.1 (default 0.05)
    - CI test: 'time_series_based_correlation' (고정)
  
  - 계산 시간: < 1분 (합성 데이터, D=19)
  - Output: Adjacency matrix (D×D), edge weights, p-values
```

**F-2.2 DAG 시각화**
```
Requirements:
  - 시각화 라이브러리: Plotly (상호작용)
  - 노드: 변수명 (크기 = 연결도)
  - 엣지: 방향성 (두께 = 강도, 색상 = p-value)
  - 인터랙션:
    - 마우스 호버: 엣지 정보 (source, target, weight, p-value)
    - 클릭: 노드 정보 패널 표시
  - 필터: Threshold slider (p-value)로 엣지 수동 조절
  - 레이아웃: Hierarchical 또는 Force-directed
```

**F-2.3 DAG 품질 평가**
```
Requirements (합성 데이터에만):
  - Ground truth DAG와 비교
  - 평가 지표:
    - Edge precision: 예측 엣지 중 정답 비율
    - Edge recall: 정답 엣지 중 감지 비율
    - Direction accuracy: 방향성 정확도
  - 결과 표시: 백분율 + 해석 텍스트
  
예시: "21개 엣지 중 17개 정확 (81% precision), 
        실제 25개 중 17개 감지 (68% recall)"
```

---

#### F-3: 불량 탐지 (Anomaly Detection)

**F-3.1 모델 예측**
```
Requirements:
  - 입력: 전처리된 cycle 데이터 (또는 배치 다중 cycle)
  - 모델: PressFuse (또는 대체: LSTM)
  - 출력 (각 cycle별):
    - defect_probability: [0, 1] (sigmoid)
    - anomaly_type: categorical (P013-001, ..., Normal)
    - anomaly_confidence: [0, 1]
    - prediction_latency: < 100ms (CPU), < 10ms (GPU)
  
  - 배치 처리: 최대 1000 cycles 동시 예측
```

**F-3.2 결과 테이블**
```
Requirements:
  - 예측 결과 DataTable:
    Columns: [cycle_id | defect_prob | anomaly_type | confidence | timestamp]
  - 인터랙션:
    - 정렬 (클릭 헤더)
    - 필터 (범주형, 수치형)
    - 페이지네이션 (1000+ rows)
    - 개별 row 클릭 → Details panel (설명 이동)
```

**F-3.3 임계값 조절**
```
Requirements:
  - Threshold slider: 0.0 ~ 1.0 (0.5 default)
  - 슬라이더 움직임 → 실시간 메트릭 업데이트:
    - Confusion matrix (TP, FP, FN, TN)
    - Precision, Recall, F1 동적 계산
    - FAR (False Alarm Rate) @ Recall=0.95
    - Cost-aware score
  - 추천 임계값 자동 제시 (FAR < 5% 달성하는 지점)
```

---

#### F-4: 설명 가능성 (Explainability)

**F-4.1 SHAP 그래디언트**
```
Requirements:
  - 입력: 예측된 1개 cycle, 학습된 모델
  - 계산: 각 변수의 기여도 (SHAP values)
  - 시각화:
    * Force plot: 각 변수가 기여하는 정도 (빨강=+, 파랑=-)
    * Bar chart: Top 10 features (절대값 정렬)
    * 시간대별 분석: 시간 축으로 SHAP aggregation
  
  - 다중 cycle 분석:
    * 정상 cycle 2개 + 불량 cycle 3개
    * Side-by-side SHAP comparison
    * 해석: "불량은 정상 대비 어느 변수가 크게 편차?"
  
  - 성능: < 3초/cycle (GPU)
```

**F-4.2 Attention 맵**
```
Requirements:
  - 입력: 1개 cycle의 attention weights (num_heads, T, T)
  - 시각화: 2D heatmap
    * X축: Key 시간 단계 (0-192)
    * Y축: Query 시간 단계 (0-192)
    * 색상: Attention weight [0, 1]
    * 주석: 시간대별 phase (Vacuum, Hot-press, Cooling, Release)
  
  - Multi-head 처리:
    * 개별 head별 heatmap 4개 (관심 있으면)
    * 또는 평균 heatmap만 (간단)
  
  - 정상/불량 비교: 2×2 subplot
  - 해석: "불량은 T=100-150에서 높은 attention"
```

**F-4.3 인과 경로 추적**
```
Requirements:
  - 입력: 1개 불량 cycle, PCMCI DAG
  - 분석:
    * Root cause 역추적: "변수 A의 anomaly → 변수 B 전파"
    * Forward propagation: "최상류에서 하류까지"
    * Influence score: 각 노드의 영향도 계산
  
  - 시각화: 그래프에서 경로 강조 (빨간색)
  - 텍스트 설명: "HPPRESSPV 저하 → VACUUM 상승 → PT 저하"
```

**F-4.4 통합 설명 리포트**
```
Requirements:
  - 자동 생성: SHAP + Attention + Causal 통합
  - 형식: JSON (프로그래밍용) + Markdown (사람용)
  - 내용:
    * Executive summary (1문장)
    * Root causes (Top 3)
    * Contributing factors (각 변수별)
    * Temporal analysis (언제 발생)
    * Recommendations (개선 제안)
  
  - 성능: < 5초/cycle
```

---

#### F-5: 대시보드 & 시각화 (Dashboard)

**F-5.1 메인 대시보드**
```
Requirements (Tab 기반 UI):
  Tab 1: Data Overview
    - Upload widget
    - Data statistics card (rows, features, missing %)
    - Distribution plots (select feature)
  
  Tab 2: Causal Analysis
    - DAG 시각화 (Plotly)
    - Threshold slider
    - Edge list table
    - Export button (PNG, SVG, JSON)
  
  Tab 3: Predictions
    - 모델 선택 (Dropdown)
    - 예측 결과 DataTable
    - Threshold slider + 실시간 메트릭 업데이트
    - 개별 cycle 상세보기 링크
  
  Tab 4: Explanations
    - Cycle ID 선택
    - SHAP force plot
    - Attention heatmap
    - Causal pathway diagram
    - Text explanation
  
  Tab 5: Reports (선택 기능)
    - 데이터 범위 선택
    - "Generate Report" 버튼
    - PDF/HTML 다운로드
```

**F-5.2 성능 지표 카드**
```
Requirements:
  - 실시간 메트릭 표시:
    * AUROC (model performance)
    * Precision, Recall, F1 (at current threshold)
    * FAR (False Alarm Rate)
    * Cost-aware score
  
  - 형식: 큰 숫자 + 색상 코드 (🟢 Good, 🟡 Acceptable, 🔴 Poor)
  
  - 히스토리: 최근 10개 실행 통계 (평균, 표준편차)
```

---

### 5.2 Advanced Features (Optional, Low Priority)

#### F-6: 자동 리포트 생성 (Auto-Report)

**F-6.1 주간/월간 리포트**
```
Requirements:
  - 데이터 범위 입력 (시작일~종료일)
  - 자동 분석:
    * Top 3 불량 원인
    * 수율 추이 그래프
    * 변수별 이상 패턴
  - 형식: PDF (Executive Summary)
  - 시간: < 1분 생성
```

#### F-7: 시나리오 시뮬레이션 (Scenario What-If)

**F-7.1 인과 개입 (Causal Intervention)**
```
Requirements:
  - 입력: 1개 cycle + "만약 변수 X를 Y로 조정한다면?"
  - 계산: DAG 구조에 따른 downstream 영향 추정
  - 결과: "조정 후 예상 defect_probability = 0.15 (↓ 0.72)"
  
Status: Nice-to-have, 시간 부족 시 생략
```

---

## 6. 비기능 요구사항 (Non-Functional Requirements)

### 6.1 성능 (Performance)

| 항목 | 요구사항 | 목표 |
|------|---------|------|
| **데이터 업로드** | 500MB 파일 < 30초 | Latency |
| **PCMCI 실행** | 합성 19변수, 1000 cycles < 1분 | Throughput |
| **모델 예측** | 1 cycle < 100ms (CPU), < 10ms (GPU) | Latency |
| **SHAP 계산** | 1 cycle < 3초 (GPU) | Latency |
| **Attention 시각화** | 생성 < 1초 | Latency |
| **메트릭 업데이트** | Threshold 슬라이더 실시간 (< 500ms) | Responsiveness |
| **리포트 생성** | PDF < 1분 | Latency |

### 6.2 유용성 (Usability)

```
- 사용자 교육 시간: < 10분 (새로운 사용자)
- 작업 완료 시간: 불량 분석 < 2분
- 에러율: < 1% (잘못된 조작)
- 만족도 (SUS score): ≥ 70/100
```

### 6.3 신뢰성 (Reliability)

```
- 가용성: 99.5% (시스템 정상 운영 시간)
- MTBF (Mean Time Between Failures): > 1000시간
- 복구 시간: < 30분 (에러 발생 시)
- 데이터 무결성: 손실 없음 (자동 백업)
```

### 6.4 확장성 (Scalability)

```
- 동시 사용자: 최소 5명 (프로토타입)
- 데이터셋 크기: 최대 500MB (메모리)
- 모델 추가 가능성: 새 모델 추가 < 2시간
- API 래이트 리미트: 100 req/min (필요시)
```

### 6.5 보안 (Security)

```
- 데이터 암호화: 전송(HTTPS), 저장소(AES-256) - 향후
- 접근 제어: 현재는 로컬 접근만 (프로토타입)
- 감사 로깅: 모든 작업 기록 (CSV 로그)
- 입력 검증: SQL injection, XSS 방어
```

### 6.6 유지보수성 (Maintainability)

```
- 코드 구조: Modular (src/data, src/models, src/explain, ...)
- 문서화: 모든 함수 docstring + API 문서
- 테스트 커버리지: ≥ 80% (pytest)
- 버전 관리: Git + Semantic versioning
- 로깅: DEBUG, INFO, WARNING, ERROR 수준
```

### 6.7 배포 (Deployment)

```
- 개발 환경: Python 3.11+ (Conda 추천)
- 프로덕션: Docker container (선택 사항)
- CI/CD: GitHub Actions (자동 테스트 & 배포)
- 문서: README.md, SETUP.md, API.md
```

---

## 7. 화면 목록 (Screens/Views)

### 7.1 화면 맵

```
Main Dashboard (Streamlit)
├── Tab 1: Data Upload & Overview
│   ├── File uploader (CSV/Parquet/Excel)
│   ├── Data statistics card
│   ├── Distribution plot
│   └── Next button → Tab 2
│
├── Tab 2: Causal Analysis
│   ├── PCMCI learner button
│   ├── DAG graph (Plotly interactive)
│   ├── Threshold slider
│   ├── Edge list table
│   └── Export options (PNG/SVG/JSON)
│
├── Tab 3: Predictions & Anomaly Detection
│   ├── Model selector
│   ├── Results DataTable
│   ├── Threshold slider
│   ├── Metrics cards (AUROC, FAR, F1, etc.)
│   └── Row → Details link
│
├── Tab 4: Explanations (Detail View)
│   ├── Cycle ID display
│   ├── Prediction result
│   ├── SHAP force plot
│   ├── Attention heatmap
│   ├── Causal pathway diagram
│   └── Text summary
│
└── Tab 5: Reports (Optional)
    ├── Date range picker
    ├── "Generate Report" button
    ├── Report type selector (PDF/HTML)
    └── Preview & Download
```

### 7.2 주요 화면 상세

**화면 1: 데이터 업로드 & 선택**

```
┌─────────────────────────────────────────────────┐
│  PCB Press Anomaly Detection & Explainability   │
├─────────────────────────────────────────────────┤
│                                                 │
│  📁 Upload Data (Drag & Drop or Browse)         │
│    [Choose File] (CSV, Parquet, Excel)          │
│                                                 │
│  ┌─ Data Preview ────────────────────────────┐  │
│  │ 1000 rows × 19 features                    │  │
│  │ Missing: 0.5% | Timestamp: 2026-08-15    │  │
│  └────────────────────────────────────────────┘  │
│                                                 │
│  Preprocessing Options:                         │
│    Normalization: [Z-score ▼] [Min-Max]        │
│    Time Window: [192 ◄─ Slider ──► 512]        │
│                                                 │
│                        [Preprocess] → Tab 2    │
│                                                 │
└─────────────────────────────────────────────────┘
```

**화면 2: 인과 그래프 시각화**

```
┌─────────────────────────────────────────────────┐
│ Causal Discovery (PCMCI DAG Learning)           │
├─────────────────────────────────────────────────┤
│                                                 │
│ Parameters:                                     │
│  tau (time lag): 2 ◄──────► 5                   │
│  alpha (p-value): 0.05 ◄──────►                 │
│                                                 │
│  [🔄 Learn DAG]                                 │
│                                                 │
│  ┌─ DAG Graph ────────────────────────────────┐ │
│  │                                             │ │
│  │    HPPRESSPV ──→ VACUUM ──→ PT1            │ │
│  │       ↓              ↓                     │ │
│  │    FHPPRESSPV      DEFECT                  │ │
│  │                                             │ │
│  │  [Interactive - hover for details]         │ │
│  └─────────────────────────────────────────────┘ │
│                                                 │
│ Threshold: 0.05 ◄──────────────►                │
│ Edges shown: 21 / 35                            │
│                                                 │
│ ┌─ Edge Table ───────────────────────────────┐ │
│ │ Source    Target      Weight   P-value      │ │
│ │ HPPRESSPV VACUUM      0.82     0.001 ✓     │ │
│ │ VACUUM    DEFECT      0.65     0.01  ✓     │ │
│ └─────────────────────────────────────────────┘ │
│                                                 │
│ [📥 Export PNG] [📥 Export JSON]                │
│                                                 │
└─────────────────────────────────────────────────┘
```

**화면 3: 예측 결과 및 임계값 조절**

```
┌─────────────────────────────────────────────────┐
│ Predictions & Anomaly Detection                 │
├─────────────────────────────────────────────────┤
│ Model: [PressFuse ▼]  [Make Predictions]        │
│                                                 │
│ ┌─ Metrics (Live Update) ────────────────────┐ │
│ │  AUROC: 0.98  │ PRECISION: 0.94            │ │
│ │  RECALL: 0.96 │ F1: 0.95                   │ │
│ │  FAR@0.95: 0.023 │ Cost-Aware: 0.91        │ │
│ └─────────────────────────────────────────────┘ │
│                                                 │
│ Threshold: 0.5 ◄────────────────────────────►   │
│                                                 │
│ ┌─ Confusion Matrix ────────────────────────┐  │
│ │        Predicted:                          │  │
│ │        Positive  Negative                  │  │
│ │  Real Pos:  96       4                     │  │
│ │      Neg:   5      895                     │  │
│ └─────────────────────────────────────────────┘ │
│                                                 │
│ ┌─ Prediction Results (Scroll enabled) ──────┐ │
│ │ Cycle │ Defect│ Type        │ Confidence   │ │
│ │ 4237  │ 0.87  │ Pressure ↓  │ 0.90  🔴 [+]│ │
│ │ 4238  │ 0.12  │ Normal      │ 0.95  🟢     │ │
│ │ 4239  │ 0.65  │ Vacuum ↑    │ 0.78  🟡 [+]│ │
│ └─────────────────────────────────────────────┘ │
│                                                 │
│ [Click row → Details & Explanation]             │
│                                                 │
└─────────────────────────────────────────────────┘
```

**화면 4: SHAP + Attention 설명**

```
┌─────────────────────────────────────────────────┐
│ Explanations: Cycle #4237 (DEFECT=0.87)        │
├─────────────────────────────────────────────────┤
│                                                 │
│ === SHAP Feature Importance ===                 │
│ [Force Plot]                                    │
│ base_value: 0.15  →  model_output: 0.87        │
│ HPPRESSPV:  +0.35 (most important)             │
│ VACUUM:     +0.22                              │
│ PT1:        +0.15                              │
│ ...                                             │
│                                                 │
│ === Attention Heatmap ===                      │
│ ┌─────────────────────────────────────────┐   │
│ │ [T=0-180]      [T=180-10140]  [Cooling] │   │
│ │ Vacuum Phase   Hot Press      Phase     │   │
│ │                    ████████ ← High attention in Hot Press
│ │                    Vacuum ↑ occurs here
│ └─────────────────────────────────────────┘   │
│                                                 │
│ === Causal Pathway ===                         │
│ Root: HPPRESSPV ↓ (T=100)                      │
│  ↓ (DAG edge: 0.82)                            │
│ VACUUM ↑ (T=110)                               │
│  ↓ (DAG edge: 0.65)                            │
│ DEFECT → HIGH RISK                             │
│                                                 │
│ === Recommendation ===                         │
│ 📌 Check HIGH PRESSURE SENSOR                  │
│ 📌 Verify vacuum pump calibration              │
│                                                 │
│ [📥 Export Explanation]                        │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

## 8. API 개요 (API Specification)

### 8.1 Backend REST API (FastAPI, 선택사항)

```
┌─────────────────────────────────────────────────┐
│          FastAPI Backend (선택 기능)             │
│  (현재 Streamlit으로 범용, 향후 확장용)         │
└─────────────────────────────────────────────────┘
```

**데이터 관리**

```python
# Upload & Preprocess
POST /api/v1/data/upload
  Input:
    - file: MultipartFile (CSV/Parquet)
    - infer_schema: bool
  Response:
    {
      "data_id": "uuid-xxx",
      "shape": [1000, 19],
      "stats": {...},
      "schema": {...}
    }

PUT /api/v1/data/{data_id}/preprocess
  Input:
    - normalization: "z_score" | "min_max" | "robust"
    - outlier_method: "iqr" | "z_score"
    - time_window: 192
    - selected_features: ["HPPRESSPV", ...]
  Response:
    {
      "preprocessed_data_id": "uuid-yyy",
      "shape": [950, 192, 15]
    }
```

**인과 탐색**

```python
POST /api/v1/causal/learn_dag
  Input:
    - preprocessed_data_id: "uuid-yyy"
    - method: "pcmci"
    - tau: 2
    - alpha: 0.05
  Response:
    {
      "dag_id": "uuid-zzz",
      "adjacency_matrix": [...],
      "edge_weights": [...],
      "edge_pvalues": [...],
      "precision": 0.82,
      "recall": 0.78
    }

GET /api/v1/causal/dag/{dag_id}
  Response: DAG metadata

GET /api/v1/causal/dag/{dag_id}/export
  Params: format="png" | "svg" | "json"
  Response: File (PNG/SVG) 또는 JSON
```

**예측**

```python
POST /api/v1/models/predict
  Input:
    - preprocessed_data_id: "uuid-yyy"
    - model_id: "pressfuse_v1"
    - threshold: 0.5
  Response:
    {
      "predictions_id": "uuid-aaa",
      "cycle_ids": [1, 2, ...],
      "defect_proba": [0.87, 0.12, ...],
      "defect_type": ["pressure_drop", "normal", ...],
      "metrics": {
        "auroc": 0.98,
        "precision": 0.94
      }
    }
```

**설명**

```python
POST /api/v1/explain
  Input:
    - cycle_id: 42
    - data_id: "uuid-yyy"
    - model_id: "pressfuse_v1"
    - explain_method: "shap" | "attention" | "both"
  Response:
    {
      "explanation_id": "uuid-bbb",
      "shap_values": [...],
      "top_features": [...],
      "attention_heatmap": [...],
      "causal_pathway": [...]
    }

GET /api/v1/explain/{explanation_id}
  Response: 상세 설명 데이터
```

**리포트**

```python
POST /api/v1/reports/generate
  Input:
    - data_id: "uuid-yyy"
    - format: "pdf" | "html"
    - date_range: {"start": "2026-08-01", "end": "2026-08-31"}
  Response:
    {
      "report_id": "uuid-ccc",
      "download_url": "https://..."
    }

GET /api/v1/reports/{report_id}/download
  Response: PDF/HTML 파일
```

---

## 9. 성공 지표 (Success Metrics & KPIs)

### 9.1 기술 성과 지표 (Tech KPIs)

```
┌─────────────────────────────────────────────┐
│ Metric                  │ Target │ Current  │
├─────────────────────────┼────────┼──────────┤
│ 모델 AUROC              │ ≥ 0.98 │ TBD      │
│ FAR @ Recall=0.95       │ < 5%   │ TBD      │
│ PCMCI DAG Precision     │ ≥ 75%  │ TBD      │
│ PCMCI DAG Recall        │ ≥ 75%  │ TBD      │
│ SHAP 계산 시간/cycle   │ < 3s   │ TBD      │
│ 예측 레이턴시           │ < 100ms│ TBD      │
│ 서버 가용성             │ 99.5%  │ TBD      │
│ 테스트 커버리지         │ ≥ 80%  │ TBD      │
│ API 응답 시간 (p95)     │ < 500ms│ TBD      │
└─────────────────────────────────────────────┘
```

### 9.2 비즈니스 성과 지표 (Business KPIs)

```
┌─────────────────────────────────────────────┐
│ 지표                    │ Target │ Timeline │
├─────────────────────────┼────────┼──────────┤
│ 논문 투고               │ 1편    │ 10월 31일│
│ 논문 격수 (Impact)      │ Top 20% (estimated)
│ GitHub Stars            │ ≥ 20   │ 6개월   │
│ 커뮤니티 활동           │ ≥ 3 issues/month
│ 사용자 만족도 (SUS)     │ ≥ 70   │ 초기 평가
│ 대학 협력 제안          │ ≥ 1    │ 1년    │
│ 산업 파일럿 기회        │ ≥ 1    │ 2년    │
└─────────────────────────────────────────────┘
```

### 9.3 사용자 경험 지표 (UX KPIs)

```
┌─────────────────────────────────────────────┐
│ 지표                    │ Target │ 측정 방식│
├─────────────────────────┼────────┼──────────┤
│ 첫 사용 학습 시간       │ < 15분│ User test
│ 불량 분석 시간          │ < 2분  │ Workflow test
│ 에러율                  │ < 1%   │ 사용 로그
│ 만족도 (NPS)            │ ≥ 50   │ 설문
│ 재사용 의향             │ ≥ 80%  │ 설문
└─────────────────────────────────────────────┘
```

### 9.4 연구 영향 지표 (Research Impact KPIs)

```
┌─────────────────────────────────────────────┐
│ 지표                    │ 정의                │
├─────────────────────────┼──────────────────────┤
│ 논문 인용도             │ 1년 내 ≥ 5 citations│
│ 컨퍼런스 발표           │ ≥ 3 presentations   │
│ 학계 협력 제안          │ ≥ 1 collaboration   │
│ 후속 연구 착수          │ ≥ 2 follow-up works│
│ 코드 포크/기여          │ ≥ 5 forks           │
└─────────────────────────────────────────────┘
```

---

## 10. 제외 범위 (Out of Scope)

### 10.1 명시적 제외항목

```
❌ 불포함 기능:

1. 모바일 네이티브 앱
   - Reason: Streamlit 웹 앱으로 충분, 시간 제약
   - 향후: React Native / Flutter로 확장

2. 사용자 인증 & 권한 관리
   - Reason: 프로토타입, 로컬 접근만
   - 향후: OAuth 2.0 구현

3. 실시간 스트리밍 데이터 처리
   - Reason: 배치 처리 충분
   - 향후: Kafka/Redis 연계

4. Digital Twin 연계
   - Reason: 별도 대규모 프로젝트
   - 향후: SimPy 기반 시뮬레이션

5. VLM 기반 이미지 분석
   - Reason: AOI 이미지 데이터 없음
   - 향후: LLaVA 통합

6. 재정 분석 (ROI 계산)
   - Reason: 과도한 scope
   - 향후: 비용-편익 분석 추가

7. 다국어 지원
   - Reason: Korean/English 기본
   - 향후: i18n 라이브러리 추가

8. 대규모 분산 처리
   - Reason: 단일 서버 충분
   - 향후: Spark / Dask 고려
```

### 10.2 명시적 포함 예정

```
✅ 포함 (Phase 1, 10월까지):

1. 단일 공정 (PCB Press)만 대상
   - 향후: 다른 공정으로 확장

2. 합성 + SECOM 데이터셋만
   - 향후: 실제 공정 데이터 추가

3. 영어 + 한국어 기본 UI
   - 향후: 다국어 충분

4. Streamlit 웹 앱
   - 향후: FastAPI + React 고도화

5. CPU/GPU 여유 기반 단순 배포
   - 향후: Docker, K8s 지원
```

### 10.3 경계선상의 항목 (Contingent)

```
🟡 조건부 (시간 여유 있으면):

1. Auto-Report Generation (PDF/HTML)
   - 현재: 스켈레톤
   - 조건: 시간 > 150시간이면 Full 구현

2. What-If Scenario (Causal Intervention)
   - 현재: 백로그 low priority
   - 조건: Jupyter notebook으로라도 제공

3. 추가 벤치마크 (LSTM vs PressFuse 비교)
   - 현재: 최소 2개 모델
   - 조건: 시간 남으면 5개 모델까지
```

---

## 11. 타임라인 및 마일스톤

### 11.1 개발 타임라인 (2026년)

```
Jun 1   |━━ Sprint 1-A: Core (PCMCI) ━━|
        └─ Week 1-3: 40시간
        
Jun 22  |━━ Sprint 1-B: XAI (SHAP/Att) ━━|
        └─ Week 4-6: 55시간
        
Jul 13  |━━ Sprint 2: Propagation ━━|
        └─ Week 7-10: 30시간
        
Aug 10  |━━ Sprint 3-A: Experiments ━━|
        └─ Week 11-14: 30시간
        
Sep 7   |━━ Sprint 3-B: Paper ━━|
        └─ Week 15-17: 20시간
        
Oct 1   |━━ Finalization ━━|
        └─ Week 18-20: 15시간
        
Oct 31  📄 THESIS SUBMISSION
```

### 11.2 주요 마일스톤

```
✅ Mile-stone 1 (Jun 21): MVP 완성
   - PCMCI 정확도 ≥ 75% 확정
   - 기본 Streamlit UI 동작
   - 첫 번째 Figure 5개 생성

✅ Mile-stone 2 (Jul 31): XAI 완성
   - SHAP + Attention 작동
   - 통합 설명 파이프라인
   - Figure 15개

✅ Mile-stone 3 (Sep 6): 실험 완료
   - 벤치마크 테이블 완성
   - 논문 Methods + Results draft 80%
   - 모든 코드 테스트 pass

✅ Mile-stone 4 (Oct 11): Thesis Ready
   - 논문 완전 draft (95%)
   - 교수 피드백 반영 완료
   - GitHub repo 최종 정리

✅ Mile-stone 5 (Oct 31): SUBMIT ✨
   - 논문 최종 제출
   - 방어 준비 완료
```

---

## 12. 위험 및 완화 전략

### 12.1 기술적 위험

```
┌─────────────────────────────────────────────────────┐
│ 위험                  │ 확률 │ 영향 │ 완화 전략      │
├─────────────────────────────────────────────────────┤
│ PCMCI 정확도 < 60%   │ 고   │ 심각│ 백업: FCI/LiN  │
│ GNN 수렴 실패         │ 중   │ 높음│ Rule-based 전환│
│ 시간 부족            │ 중   │ 심각│ Scope 축소     │
│ 데이터 불균형        │ 중   │ 보통│ SMOTE/WeightedLoss
│ 의존성 호환성        │ 낮   │ 보통│ Docker 캡슐화  │
└─────────────────────────────────────────────────────┘
```

### 12.2 관리적 위험

```
┌─────────────────────────────────────────────────────┐
│ 위험                  │ 원인  │ 완화 전략           │
├─────────────────────────────────────────────────────┤
│ 스코프 크리프        │ 끝없는 요청 추가 | 엄격한 백로그 관리
│ 지연된 피드백         │ 지도교수 바쁨 | 주간 정기 미팅 확정
│ 참고 논문 부족       │ 시간 부족 | 사전 준비 완료
│ 팀 커뮤니케이션       │ 혼자 진행 | 주간 진도 기록
└─────────────────────────────────────────────────────┘
```

---

## 13. 성공 시나리오 vs 실패 시나리오

### 13.1 성공 시나리오

```
✅ 최선의 경우 (확률 35%):

Oct 31일:
  ✓ 논문 6-7 페이지 완성도 높음
  ✓ Figure 15-20개, Table 8-10개
  ✓ GitHub Stars ≥ 30
  ✓ 학위 심사 합격
  ✓ 국제 저널 투고 준비 중

Impact:
  - 반도체 업계 주목
  - 대학 모사 프로젝트 2-3개
  - 산업 협력 문의 다수
```

### 13.2 현실 시나리오 (가장 확률 높음)

```
🟡 현실적 경우 (확률 55%):

Oct 31일:
  ✓ 논문 5-6 페이지 완성도 중상
  ✓ Figure 12-15개, Table 6-8개
  ✓ GitHub Stars ≥ 15
  ✓ 학위 심사 통과 (가능)
  ✓ 학술 학회에 발표 예한

Impact:
  - 충북대 우수 논문상
  - 후속 연구비 신청 가능
  - 산업 관심 (middle)
```

### 13.3 실패 시나리오

```
❌ 최악의 경우 (확률 < 10%):

Oct 31일:
  ✗ 논문 미완성 또는 4 페이지 이하
  ✗ Figure 부족, 실험 미흡
  ✗ 학위 심사 불합격
  ✗ 재투고 또는 내년도 재도전

원인:
  - PCMCI 정확도 너무 낮음
  - 시간 초과 (scope 축소 실패)
  - 블로킹된 문제 미해결

해결:
  - 6개월 추가 작업
  - 혹은 scope 대폭 축소 (DAG만 제시)
```

---

## 14. 승인 및 이해 관계자

### 14.1 서명 및 승인

| 역할 | 이름 | 서명 | 날짜 |
|------|------|------|------|
| Product Owner | 송공호 | __ | 2026-05-25 |
| Technical Lead | (자신) | __ | 2026-05-25 |
| Academic Advisor | (지도교수) | __ | 2026-06-01 |

### 14.2 이해 관계자

```
Primary:
  - 송공호 (Product Owner, 학생)
  - 지도교수 (Academic Advisor)

Secondary:
  - 대학원 학위 심사위원 (3명)
  - 학과 행정팀

Tertiary:
  - 산업계 협력사 (향후)
  - 오픈소스 커뮤니티
```

---

## 15. 승인 기준 (Definition of Done)

### 15.1 전체 프로젝트 완료 기준

```
✅ Paper Ready:
  - [ ] 논문 5-6 페이지 작성 완료
  - [ ] Figure 15개 이상
  - [ ] Table 6개 이상
  - [ ] References ≥ 30개
  - [ ] Appendix (코드, 설정) 포함

✅ Code Quality:
  - [ ] Test coverage ≥ 80%
  - [ ] All tests pass
  - [ ] Docstring 100%
  - [ ] Type hints 100%
  - [ ] No security warnings (bandit)

✅ Reproducibility:
  - [ ] README.md (한글 + 영문)
  - [ ] SETUP.md (설치 가이드)
  - [ ] requirements.txt 명시
  - [ ] dvc.yaml (데이터 재현)
  - [ ] GitHub commit log 명확

✅ System Functionality:
  - [ ] Streamlit UI 3개 탭 모두 동작
  - [ ] PCMCI DAG 정확도 ≥ 65%
  - [ ] 모델 AUROC ≥ 0.95
  - [ ] SHAP 계산 < 3초/cycle
  - [ ] 에러 < 1%

✅ Deployment:
  - [ ] GitHub repo public
  - [ ] README 조회 시 즉시 이해 가능
  - [ ] "Quick Start" < 5분
  - [ ] 첫 실행 성공률 > 90%
```

### 15.2 각 이슈별 완료 기준

**#1 PCMCI**: 기술 문서 참고 (GITHUB_ISSUES_HIGH_PRIORITY.md)

---

## 부록 A: 용어 정의 (Glossary)

```
PCMCI
  Momentary conditional independence (MCI)를 바탕한
  시계열 인과 탐색 알고리즘

DAG (Directed Acyclic Graph)
  방향성 순환 없는 그래프 (인과 구조 표현)

SHAP (SHapley Additive exPlanations)
  게임 이론 기반 설명 가능 AI 방법

XAI (Explainable Artificial Intelligence)
  해석 가능한 인공지능

AUROC (Area Under the Receiver Operating Characteristic Curve)
  이진 분류 성능 지표

FAR (False Alarm Rate)
  거짓 양성 비율 (오경보율)

Press Cycle
  PCB 적층 공정의 한 번의 사이클 (약 13,000초)

Defect
  공정 또는 제품 불량

Anomaly
  공정 변수의 정상 범위 벗어남
```

---

**문서 버전**: 1.0  
**작성일**: 2026년 5월 25일  
**마지막 수정**: 2026년 5월 25일  
**상태**: ✅ 승인 대기 (지도교수 검토 후 확정)


