# 📚 Literature Review & Research Organization

## 개요

본 폴더는 **PressFuse** 연구의 선행 연구 분석 및 논문 작성을 위한 체계적인 문헌 관리 구조입니다.

---

## 📂 폴더 구조

```
docs/literature/
├── 📖 마크다운 문서 (학술적 분석)
│   ├── LITERATURE_MASTER_INDEX.md        👈 이곳부터 시작
│   ├── STAGE_1_PRIOR_WORK_SOLVED.md      (선행 연구 5개 흐름)
│   ├── STAGE_2_RESEARCH_GAPS.md          (공통 한계 3가지 Gap)
│   ├── STAGE_3_OUR_NOVELTY.md            (본 연구 차별점 4가지)
│   ├── STAGE_4_VALIDATION_STRATEGY.md    (검증 설계 & 실험 계획)
│   └── STAGE_5_PAPER_OUTLINE.md          (논문 작성 아웃라인)
│
├── 📊 데이터 분석 (엑셀)
│   analysis/
│   ├── research_summary.xlsx             (선행 연구 10개 요약)
│   ├── novelty_comparison.xlsx           (차별점 C1~C4 비교표)
│   └── experiment_tracking.xlsx          (실험 진행 추적)
│
└── 📥 외부 자료 (업로드용)
    raw_materials/
    ├── pdf/                              (논문, 인더스트리 리포트)
    ├── xlsx/                             (선행 연구 데이터, 결과표)
    └── docx/                             (요약 노트, 인터뷰 기록)
```

---

## 🎯 사용 가이드

### 1️⃣ 처음 접하는 사용자

```
1. LITERATURE_MASTER_INDEX.md 읽기
   ↓
2. STAGE_2_RESEARCH_GAPS.md (문제 이해)
   ↓
3. STAGE_3_OUR_NOVELTY.md (본 연구 솔루션)
   ↓
4. STAGE_1_PRIOR_WORK_SOLVED.md (깊이 있는 학습)
```

### 2️⃣ 논문 작성자

```
1. STAGE_5_PAPER_OUTLINE.md (구조 파악)
   ↓
2. STAGE_3_OUR_NOVELTY.md (Introduction 작성)
   ↓
3. STAGE_1 & 2 (Related Work 작성)
   ↓
4. analysis/*.xlsx (표/그래프 참고)
```

### 3️⃣ 선행 연구 분석가

```
1. research_summary.xlsx 열기
   ↓
2. STAGE_1_PRIOR_WORK_SOLVED.md 참고
   ↓
3. 각 논문의 PDF를 raw_materials/pdf/에 저장
```

### 4️⃣ 실험 담당자

```
1. experiment_tracking.xlsx 열기
   ↓
2. STAGE_4_VALIDATION_STRATEGY.md로 설계 이해
   ↓
3. 실험별로 상태 & 결과 업데이트
```

---

## 📋 각 마크다운 파일의 역할

| 파일명 | 내용 | 분량 | 작성 완료 |
|--------|------|------|---------|
| **LITERATURE_MASTER_INDEX** | 5단계 종합 인덱스 | 200줄+ | ✅ |
| **STAGE_1** | 선행 연구 5개 흐름 정리 | 300줄+ | ✅ |
| **STAGE_2** | 공통 한계 3가지(Gap A/B/C) | 350줄+ | ✅ |
| **STAGE_3** | 본 연구 차별점 C1~C4 | 400줄+ | ✅ |
| **STAGE_4** | 검증 설계 & Ablation | 300줄+ | ✅ |
| **STAGE_5** | 논문 작성 로드맵 & 초안 | 350줄+ | ✅ |

---

## 📊 각 엑셀 파일의 용도

### research_summary.xlsx
```
10개 선행 연구 요약표

컬럼:
- 연구ID (S01~S10)
- 주요저자 및 년도
- 학술지 및 제목
- 주요 기여도
- 방법론 요약
- 평가 지표
- 사용 데이터셋
- 연구 한계
- 인용도

용도: 논문 Related Work 섹션 작성 시 참고
```

### novelty_comparison.xlsx
```
본 연구의 4가지 차별점 비교표

행:
- C1: Press 특화 멀티모달
- C2: 물리 기반 합성 데이터
- C3: 비용민감 이중 지표
- C4: 멀티모달 XAI 통합

컬럼:
- 차별점 정의
- 본 연구 기여
- 핵심 기법
- 기대 효과 (선행 대비)
- 선행 연구 한계
- 학술적 기여 수준

용도: Contribution 강조, Introduction 작성, 리뷰어 질의 대응
```

### experiment_tracking.xlsx
```
6개 실험 계획 및 진행 상태

컬럼:
- 실험 ID (EXP001~006)
- 제목 및 설명
- 상태 (Planned/Running/Completed)
- 예상 성능 (AUROC, FAR@R95)
- 담당자, 시작일, 완료일
- 비고

용도: 실험 일정 관리, 진행 상황 공유, 결과 추적
```

---

## 🔄 데이터 입출력 흐름

```
raw_materials/ (외부 자료 수집)
        ↓
        ├─ pdf/: 선행 연구 논문 저장
        ├─ xlsx/: 외부 데이터, 협력사 자료
        └─ docx/: 요약 노트, 인터뷰 기록
        ↓
분석 & 통합 (본 폴더의 마크다운 문서)
        ↓
analysis/ (요약 통계 & 추적표)
        ├─ research_summary.xlsx (정렬된 선행 연구)
        ├─ novelty_comparison.xlsx (차별점 강조)
        └─ experiment_tracking.xlsx (실험 진행)
        ↓
논문 & 발표 (STAGE_5 → 실제 작성)
        ↓
outputs/papers/ (최종 결과물)
```

---

## ✅ 필수 액션 아이템

### 즉시 (1주일 내)
- [ ] `raw_materials/pdf/` 폴더에 선행 연구 논문 다운로드
- [ ] `raw_materials/xlsx/` 폴더에 고객사 제공 데이터 복사
- [ ] 각 마크다운 파일 정독 및 이해

### 단기 (2주일 내)
- [ ] STAGE_5 기반 Introduction 영문 초안 작성
- [ ] research_summary.xlsx 보충 (추가 논문 발견 시)
- [ ] experiment_tracking.xlsx에 담당자/일정 입력

### 중기 (1개월 내)
- [ ] Related Work 섹션 완성
- [ ] 실험 실행 및 결과 기록
- [ ] novelty_comparison.xlsx 업데이트 (실제 성능 기반)

---

## 📖 논문 작성 심화 구조

```
Introduction (STAGE_5 참고)
    ├─ 배경: PCB Press 경제성 (Gap B와 연결)
    ├─ 문제: 사후/단일/무비용 3가지 한계 (Gap A/B/C)
    ├─ 기여: C1, C2, C3, C4 명시
    └─ 구성: 섹션별 로드맵

Related Work (STAGE_1 & 2, research_summary.xlsx)
    ├─ 2.1 Sensor-based PdM
    ├─ 2.2 Image-based Defect Detection
    ├─ 2.3 Multimodal Fusion
    ├─ 2.4 Causal Analysis
    ├─ 2.5 Cost-Sensitive Learning
    └─ 2.6 Positioning (차별점 강조)

Method (STAGE_3, STAGE_5 Method 섹션)
    ├─ Overview
    ├─ Sensor Module
    ├─ Event Module
    ├─ Cross-Modal Attention
    ├─ Cost-Aware Loss
    └─ Synthetic Data Generation

Experiments (STAGE_4, experiment_tracking.xlsx)
    ├─ Dataset & Baseline
    ├─ Ablation Setup
    └─ Scenario Analysis

Results (실험 완료 후 작성)
    ├─ Quantitative (표 & 그래프)
    ├─ Ablation Impact
    ├─ Explainability (SHAP + Attention)
    └─ Failure Cases

Discussion & Conclusion
    ├─ Key Findings
    ├─ Limitations
    └─ Future Work
```

---

## 🎓 인용 스타일

### 마크다운 내 인용
```markdown
[Susto et al., 2014] → @article{Susto2014, ...}
[MFGAN, 2024] → @article{MFGAN2024, ...}
```

### BibTeX 형식 (STAGE_1부터 수집)
```bibtex
@article{Key2024,
  title={...},
  author={...},
  journal={...},
  year={2024}
}
```

---

## 💡 팁 & 주의사항

### Do's ✅
- 각 단계별 마크다운을 먼저 읽고 엑셀을 사용
- 실험 진행 중 experience_tracking.xlsx를 실시간 업데이트
- raw_materials에 다운로드한 PDF의 메타정보를 research_summary.xlsx에 기록

### Don'ts ❌
- 마크다운 문서를 Word로 변환하지 말 것 (git 추적 손실)
- 엑셀을 지역 저장만 하지 말 것 (버전 관리 필요)
- 새로운 선행 연구 발견 시 파일 생성 대신 기존 파일 업데이트

---

## 📞 연락처 & 피드백

- 선행 연구 분석 오류 발견 → STAGE_1~2 마크다운 업데이트
- 실험 결과가 예상과 다름 → STAGE_4 Validation Strategy 검토
- 논문 작성 중 구조 변경 필요 → STAGE_5 Paper Outline 논의

---

## 버전 관리

| 버전 | 날짜 | 변경 사항 |
|-----|------|---------|
| 1.0 | 2026-05-26 | 초판 작성 (5단계 + 3 엑셀) |
| 1.1 | (예정) | 선행 연구 논문 추가 다운로드 |
| 2.0 | (예정) | 실험 결과 반영 & 최종 논문 |

---

**마지막 업데이트**: 2026-05-26  
**관리자**: Industrial AI Lab  
**상태**: ✅ 완성 & 즉시 사용 가능

