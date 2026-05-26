# 📚 PCB Press Defect Prediction - 논문 전략 수립 완료

## 📌 생성된 문서 구조

### 메인 문서
```
docs/
├── RESEARCH_ROADMAP_2026.md          ⭐ [메인 전략 문서] 35,000+ 줄
│   ├── A. 한 줄 총평
│   ├── B. 현재 준비 수준 진단 (SWOT)
│   ├── C. 논문 직결 우선순위 표
│   ├── D. 3단계 로드맵 (Sprint 1/2/3)
│   ├── E. 추가 개발 백로그 (15개 항목)
│   ├── F. UI/UX 화면 설계 (7개 화면)
│   ├── G. 시스템 아키텍처
│   ├── H. 다음 스프린트 Top 10 액션
│   └── I. 결론 & 권고사항
│
└── EXECUTIVE_SUMMARY.md              ⭐ [11,000줄 요약본]
    ├── 한 줄 평가
    ├── 강점/약점/리스크
    ├── 마일스톤 및 시간 분배
    ├── 즉시 실행 TOP 5
    ├── 아키텍처 한 줄 요약
    ├── 성공 지표
    └── 다음 회의 아젠다
```

---

## 🎯 핵심 내용 서머리

### 현재 상황 진단

| 항목 | 평가 | 비고 |
|------|------|------|
| **강점** | ⭐⭐⭐⭐⭐ | PyTorch, 합성데이터, 기본 ML 파이프라인 완성 |
| **약점** | 🔴🔴🔴 | 인과 그래프, XAI, 공정 최적화 완전 부재 |
| **시간** | 9개월 (430시간) | 2026.06 ~ 2027.02 (스프린트 중심) |
| **리스크** | 🔴 높음 | PCMCI 정확도, GNN 수렴, 시간 부족 |

### 논문 핵심 (반드시 구현)

1. **Causal DAG Learning** (PCMCI/NOTEARS) → 논문 이름의 핵심
2. **GNN-based Propagation** (PyG) → 불량 전파 경로 예측
3. **XAI Integration** (SHAP + Attention) → 설명 가능성 확보
4. **Cost-Sensitive Metrics** (FAR@Recall) → 실무 가치

### 3단계 로드맵

```
Sprint 1 (6-8월, 110시간)     Sprint 2 (9-11월, 180시간)    Sprint 3 (12-2월, 140시간)
├─ PCMCI 선택 & 구현           ├─ GNN 불량 전파              ├─ 벤치마크 완성
├─ 합성 데이터 검증            ├─ SHAP + Attention           ├─ 3개 공정 테스트
├─ 기본 모델 학습              ├─ 공정 최적화 기초            ├─ XAI 비교 연구
└─ 논문 Introduction draft      └─ Methods 섹션 draft         └─ 논문 제출 준비
   (결과: DAG precision 80%)       (결과: Figure 15개)            (결과: 완성본)
```

### UI/UX 설계 (7개 화면)

| 순서 | 화면명 | 목적 | 기술스택 |
|------|--------|------|---------|
| 1️⃣ | Data Upload | 파일 선택 + 스키마 매핑 | Dropzone, React Hook Form |
| 2️⃣ | Preprocessing | 정규화 + 특성 선택 | rc-slider, Plotly |
| 3️⃣ | Causal DAG | PCMCI 결과 시각화 | Cytoscape.js |
| 4️⃣ | Predictions | 예측 결과 + 지표 | ag-Grid, Recharts |
| 5️⃣ | Explanations | SHAP + Attention | Plotly heatmap |
| 6️⃣ | Propagation | 경로 추적 + 시나리오 | Cytoscape + animation |
| 7️⃣ | Reports | PDF/HTML 생성 | ReportLab, Jinja2 |

### 아키텍처 스택

```
Frontend: React + TypeScript + TailwindCSS (Vite)
   ↓ REST API
Backend: FastAPI + Pydantic (uvicorn)
   ↓ Async tasks
ML Core: PyTorch + PyG + SHAP + causalml (Celery workers)
   ↓ Storage
DB: PostgreSQL + Redis + MinIO (S3)
```

---

## 🚀 즉시 실행 항목 (이번 주)

### TOP 5 액션 (2주 안에 완료)

| 우선 | 작업 | 시간 | 완료 기준 |
|------|------|------|---------|
| 🔴 | PCMCI 라이브러리 선택 | 4h | 3개 비교 후 선택 결정 |
| 🔴 | Ground truth DAG 정의 | 6h | configs/data/ground_truth_dag.yaml |
| 🟠 | src/causal/ 폴더 구축 | 3h | GitHub commit |
| 🟠 | PCMCI wrapper 구현 | 8h | tests/test_pcmci.py pass |
| 🟡 | 논문 Introduction | 4h | 1.5페이지 draft |

**이주 우선 목표**: PCMCI 프로토타입 동작 + GitHub commit 3개 이상

---

## 📊 성공 지표

### 분기별 체크포인트

**2Q (6-8월)**
- [ ] PCMCI 단위 테스트 pass rate ≥ 90%
- [ ] PressFuse AUROC ≥ 0.95
- [ ] 논문 Introduction + Related Work draft
- ✅ **결과물**: DAG 이미지 + Figure 5개

**3Q (9-11월)**
- [ ] GNN 전파 정확도 ≥ 85%
- [ ] SHAP + Attention 통합 파이프라인
- [ ] 논문 Figure 15개 이상
- ✅ **결과물**: Methods 섹션 draft

**4Q (12-2월)**
- [ ] AUROC ≥ 0.98, FAR < 5%
- [ ] 논문 전체 초안 완성
- [ ] 방어 준비 완료
- ✅ **결과물**: 완성 논문 PDF

---

## 📚 문서 사용 방법

### 1단계: 빠른 파악 (5분)
→ **EXECUTIVE_SUMMARY.md** 읽기

### 2단계: 상세 계획 (30분)
→ **RESEARCH_ROADMAP_2026.md** 섹션 B/C/D 읽기

### 3단계: 개발 실행 (실시간)
→ 섹션 H (Top 10 액션) 체크리스트 사용

### 4단계: 주간 리뷰 (매주 금요일)
→ Sprint 진도표 업데이트

---

## ⚡ 반드시 기억할 3가지

### 1️⃣ 논문의 핵심은 "인과 그래프"
- 제목 **"MS-CDPNet"** = Multi-Stage **Causal** Defect Propagation Network
- PCMCI 정확도가 낮으면 전체 논문 기여도 급락
- **위크 1-2에 조기 프로토타입 반드시 완료**

### 2️⃣ 9개월은 촉박함
- 430시간 ÷ 40주 = **주당 10.75시간**
- 스프린트마다 명확한 산출물 강제
- 시간을 낭비하면 "논문 완성" 불가능

### 3️⃣ 데이터는 합성 데이터로 충분
- 실제 제조 데이터 NDA 비공개 → 합성 데이터 필수
- 합성 데이터 품질 검증 (Week 1-2) → 논문 신뢰도 좌우
- KS test p-value > 0.05 목표

---

## 🎓 지도교수와 공유 체크리스트

**다음 미팅 전 준비물** (추천: 2026년 6월 1일)

- [ ] EXECUTIVE_SUMMARY.md 사전 배포 (5분)
- [ ] Sprint 1 목표 확인 (10분)
- [ ] PCMCI 라이브러리 선택안 상의 (5분)
- [ ] 주간 보고 형식 정의 (5분)
- [ ] 위험 항목 협의 (5분)

---

## 🔗 추가 정보

### 시스템 요구사항

```
Python: 3.11+
GPU: NVIDIA (CUDA 12.1) 추천
Memory: 16GB RAM (32GB 권장)
Storage: 200GB (models, data, experiments)
```

### 필수 라이브러리

```
ML Core:        torch, pytorch-lightning, pytorch-geometric, scikit-learn
Causal:         causalml, DoWhy, castle
XAI:            shap (0.45+), captum (0.7+)
Backend:        fastapi, sqlalchemy, pydantic
Frontend:       react, typescript, plotly, ag-grid
```

### 참고 링크

- PCMCI 논문: Runge et al., Nature Comm 2015
- PyG 튜토리얼: https://github.com/pyg-team/pytorch_geometric
- SHAP 문서: https://shap.readthedocs.io/
- FastAPI 가이드: https://fastapi.tiangolo.com/

---

## 📞 피드백 & 개선

**이 전략 문서에 대한 질문/제안:**
- GitHub Issues: Feature request로 등록
- 이메일: [지도교수]
- 주간 회의: 매주 월요일 10:00

**다음 업데이트 일정**: 2026년 8월 31일 (Sprint 1 종료 시)

---

**작성일**: 2026년 5월 25일  
**버전**: 1.0 (Initial Release)  
**상태**: ✅ 준비 완료 (피드백 대기 중)  

> **제 역할**: 이 로드맵을 기반으로 Copilot에게 구체적인 개발 요청을 할 수 있습니다.  
> **예시**: "BL-H1 (Causal Discovery Baseline) 구현시작" → 상세 구현 가이드 제공


