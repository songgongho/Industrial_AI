# Literature Review Master Index

## 📖 전체 구조

```
docs/literature/
├── STAGE_1_PRIOR_WORK_SOLVED.md       (선행 연구 5개 흐름)
├── STAGE_2_RESEARCH_GAPS.md           (공통 한계 3가지 Gap)
├── STAGE_3_OUR_NOVELTY.md             (본 연구 차별점 C1~C4)
├── STAGE_4_VALIDATION_STRATEGY.md     (검증 설계 & 실험 계획)
├── STAGE_5_PAPER_OUTLINE.md           (논문 작성 아웃라인)
├── LITERATURE_MASTER_INDEX.md         (이 파일)
├── analysis/
│   ├── research_summary.xlsx          (선행 연구 요약 표)
│   ├── novelty_comparison.xlsx        (차별점 비교표)
│   └── experiment_tracking.xlsx       (실험 진행 추적)
└── raw_materials/
    ├── pdf/                           (📌 PDF 업로드 폴더)
    ├── xlsx/                          (📌 엑셀 업로드 폴더)
    └── docx/                          (📌 워드 업로드 폴더)
```

---

## 📋 1단계 | 선행 연구가 해결한 문제들

### 흐름 분석

| # | 흐름 | 핵심 연구 | 주요 기여 | 성숙도 |
|---|-----|---------|---------|-------|
| 1 | 단일센서 PdM | Susto et al. (2014) | 센서 기반 이상 탐지 체계화 | ⭐⭐⭐⭐⭐ |
| 2 | 이미지 결함 탐지 | DeepPCB, YOLOv5 | AOI 자동화 가능 | ⭐⭐⭐⭐ |
| 3 | 멀티모달 융합 | MFGAN (2024) | 이질 정보 결합 | ⭐⭐⭐⭐ |
| 4 | 인과 분석 | Causal GNN (2024) | 고장 원인 추적 | ⭐⭐⭐ |
| 5 | 비용민감 학습 | ESA (2021) | 비용 구조 반영 | ⭐⭐⭐⭐ |

**상세 정보**: [STAGE_1_PRIOR_WORK_SOLVED.md](STAGE_1_PRIOR_WORK_SOLVED.md)

---

## 🔍 2단계 | 선행 연구의 공통 한계

### Gap 분석

| Gap | 문제 정의 | 영향 범위 |
|-----|---------|---------|
| **Gap A** | 모달 분리: 센서/이벤트/이미지가 각각 별도 연구 | 멀티모달 통합 미진행 |
| **Gap B** | 사후 탐지: AOI 기반 검사는 공정 후 발견 | 예방 불가능 |
| **Gap C** | 삼중 문제: 라벨 희소 + 비용민감 + 설명 부족 | 현장 적용 어려움 |

**상세 정보**: [STAGE_2_RESEARCH_GAPS.md](STAGE_2_RESEARCH_GAPS.md)

---

## ✨ 3단계 | 본 연구의 차별점

### 4가지 기여 (C1~C4)

| 기여 | 핵심 내용 | 선행 연구 한계 | 본 연구 해결책 |
|-----|---------|-------------|------------|
| **C1** | Press 특화 멀티모달 | 범용 데이터셋 중심 | Cross-modal Attention + 물리 제약 |
| **C2** | 합성 데이터 기반 PdM | 라벨 극도일소 | synthpress: 10종 시나리오 생성 |
| **C3** | 비용민감 이중 지표 | AUROC 단일 최적화 | cost_aware + FAR@Recall 동시 최적화 |
| **C4** | 멀티모달 XAI | 개별 설명 기법만 존재 | SHAP + Attention + Event 융합 |

**상세 정보**: [STAGE_3_OUR_NOVELTY.md](STAGE_3_OUR_NOVELTY.md)

---

## 🧪 4단계 | 검증 설계

### 정량 목표

| 지표 | 목표 | 베이스라인 | 기대 개선 |
|-----|-----|---------|---------|
| AUROC | ≥ 0.95 | RF: 0.88 | +7% |
| FAR@Recall=0.95 | < 5% | CNN-AE: 6% | -20% |
| F1 Score | ≥ 0.85 | XGBoost: 0.78 | +9% |

### Ablation Studies

1. **단일 vs 이중 모달**: TS+Event 추가 효과 측정
2. **Attention 유무**: Cross-modal mechanism 효과
3. **비용민감 손실함수**: Cost-aware 최적화 효과

**상세 정보**: [STAGE_4_VALIDATION_STRATEGY.md](STAGE_4_VALIDATION_STRATEGY.md)

---

## 📝 5단계 | 논문 서술 계획

### 섹션 구성

| 섹션 | 내용 | 지면 |
|-----|------|------|
| Introduction | 배경 + 문제 정의 + 기여 | 2~3 pages |
| Related Work | 5개 흐름 + Gap 분석 | 3~4 pages |
| Method (PressFuse) | 아키텍처 + 수식 + 합성 데이터 | 4~5 pages |
| Experiments | 데이터셋 + 설정 + Ablation | 2~3 pages |
| Results | 표/그래프 + 설명이용성 분석 | 3~4 pages |
| Discussion | 시사점 + 한계 + 미래 방향 | 2 pages |

**상세 정보**: [STAGE_5_PAPER_OUTLINE.md](STAGE_5_PAPER_OUTLINE.md)

---

## 📊 외부 자료 업로드 안내

### 폴더 위치
```
docs/literature/raw_materials/
├── pdf/           👈 PDF 같은 문서 보관
├── xlsx/          👈 엑셀 자료 보관
└── docx/          👈 워드 문서 보관
```

### 권장 자료 형식

#### PDF
- 선행 연구 논문 (SCI 저널)
- industry report, whitepaper
- 제품 스펙시트

#### XLSX
- 선행 연구 비교표
- 실험 결과 데이터
- 협력사 데이터

#### DOCX
- 선행 연구 요약 노트
- 인터뷰 기록
- 문헌조사 초안

---

## 🎯 주요 선행 연구 인용 (Quick Reference)

### 논문 링크 및 인용형식

```bibtex
@article{Susto2014,
  title={Machine learning for predictive maintenance},
  author={Susto, Gian Antonio and others},
  journal={IEEE Trans. Industrial Informatics},
  year={2014}
}

@article{MFGAN2024,
  title={Multimodal Fusion for Industrial Anomaly Detection},
  journal={PMC/Sensors},
  year={2024}
}

@article{CausalGNN2024,
  title={Causal GNN for Fault Diagnosis},
  journal={Reliability Engineering & System Safety},
  year={2024}
}

@article{GACRI2026,
  title={Graph Autoencoder with Causal Relationship Inference},
  journal={ScienceDirect},
  year={2026}
}

@article{CSL2021,
  title={Cost-Sensitive Classification Strategy},
  journal={Expert Systems with Applications},
  year={2021}
}
```

---

## 🔄 문서 업데이트 기록

| 날짜 | 내용 | 버전 |
|-----|------|------|
| 2026-05-26 | 초판 작성 (5단계) | 1.0 |
| (진행 예정) | 외부 자료 통합 | 1.1 |
| (진행 예정) | 실험 결과 반영 | 2.0 |

---

## ✅ 다음 액션 아이템

### 즉시
- [ ] 각 단계별 마크다운 파일 검토
- [ ] raw_materials 폴더에 선행 연구 PDF/XLSX/DOCX 업로드
- [ ] Introduction 영문 초안 작성 시작

### 1주일 내
- [ ] Related Work 섹션 작성 완료
- [ ] STAGE_5 논문 아웃라인 기반 Method 초안

### 2주일 내
- [ ] 합성 데이터 (synthpress) 생성 코드 실행
- [ ] PressFuse 모델 학습 & 초기 결과 수집

### 3주일 내
- [ ] Ablation Study 1, 2, 3 완료
- [ ] Results 섹션 표/그래프 작성

---

## 📖 읽는 순서 (추천)

**처음 접하는 사람:**
1. 이 파일 (LITERATURE_MASTER_INDEX.md) 읽기
2. [STAGE_2_RESEARCH_GAPS.md](STAGE_2_RESEARCH_GAPS.md) - 문제를 이해하기 위해
3. [STAGE_3_OUR_NOVELTY.md](STAGE_3_OUR_NOVELTY.md) - 본 연구의 대안 이해
4. [STAGE_1_PRIOR_WORK_SOLVED.md](STAGE_1_PRIOR_WORK_SOLVED.md) - 깊이 있는 학습

**논문 작성 시작자:**
1. [STAGE_5_PAPER_OUTLINE.md](STAGE_5_PAPER_OUTLINE.md) 먼저 읽기
2. [STAGE_3_OUR_NOVELTY.md](STAGE_3_OUR_NOVELTY.md) - Introduction 커버
3. [STAGE_1~2](STAGE_1_PRIOR_WORK_SOLVED.md) - Related Work 작성 때 참고
4. [STAGE_4_VALIDATION_STRATEGY.md](STAGE_4_VALIDATION_STRATEGY.md) - 실험 설계

**심화 학습자:**
> 파일 순서대로: STAGE_1 → STAGE_2 → STAGE_3 → STAGE_4 → STAGE_5

