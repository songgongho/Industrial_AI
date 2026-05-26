# 멀티모달 융합 기반 PCB MLB Press 공정 이상·불량 예측 및 공정 조건 최적화

**프로젝트명**: MS-CDPNet (Multi-Stage Causal Defect Propagation Network)  
**부제**: 변수-결함-출하 다단계 인과 그래프 학습 기반 반도체 PCB 적층 공정 불량 예측 및 설명

| 항목 | 내용 |
|------|------|
| 작성자 | 송공호 (2025254010) |
| 소속 | 충북대학교 산업인공지능학과 |
| 라이선스 | MIT |

---

## 목차

- [연구 개요](#연구-개요)
- [핵심 기능](#핵심-기능)
- [빠른 시작](#빠른-시작)
- [프로젝트 구조](#프로젝트-구조)
- [사용 예시](#사용-예시)
- [결과 및 데모](#결과-및-데모)
- [데이터 스키마](#데이터-스키마)
- [배포 방법](#배포-방법)
- [개발 환경](#개발-환경)
- [인용](#인용)
- [라이선스 및 주의사항](#라이선스-및-주의사항)

---

## 연구 개요

### 문제 정의

**반도체 PCB 적층(MLB Press) 공정**은 다음과 같은 수율 문제를 가집니다.

- **공정 불량**: 10분 이상 소요되는 사이클 내 온도 이상, 압력 손실, 진공 누출 발생
- **후공정 불량**: VOID(기포), 휨(Warping), 층간 박리(Delamination)는 고가의 검사 후에야 발견
- **시간 지연**: 압착 공정(P013)과 출하 검사(P019) 사이의 간격으로 인해 근본 원인 분석 어려움
- **비대칭 비용**: 미검출(FN)은 대규모 불량 비용, 과검출(FP)은 생산 중단 야기

### 해결 방법

**멀티모달 + 설명 가능 불량 예측 시스템**:

1. 시계열 + 이벤트 로그 + AOI 이미지를 활용한 **실시간 Press 이상 탐지**
2. 인과 DAG 학습(PCMCI, NOTEARS)을 통한 **하류 불량 전파 예측**
3. 어텐션 맵 및 SHAP 그래디언트를 통한 **불량 메커니즘 설명**
4. 모델을 미분 가능한 대리 함수로 활용한 **공정 조건 최적화**

### 목표 지표

| 지표 | 목표값 |
|------|--------|
| AUROC (P013 이상 탐지) | ≥ 0.98 |
| FAR @ Recall=0.95 | < 5% (FN 가중치 = FP × 100) |
| 인과 엣지 정확도 | ≥ 85% (합성 정답 DAG 기준) |

---

## 핵심 기능

### 1. 합성 데이터 생성기

```python
from src.data.synthpress import generate_press_cycle

frame, label, metadata = generate_press_cycle(
    cycle_id=1,
    panel_id=1001,
    anomaly_type="pressure_drop",  # P013-002
    anomaly_prob=0.5
)
```

- P013 단일 이상 시나리오 6종
- 현실적인 연쇄 이상 패턴 4종
- 도메인 제약조건 기반 검증

### 2. 멀티모달 융합 모델 (PressFuse)

- Cross-modal Attention + Temporal Convolution
- 입력: 시계열(압력, 온도, 진공) + 범주형 이벤트 + AOI 이미지
- 출력: 이진 불량 분류 + 다중 불량 유형 분류 + 이상 신뢰도

### 3. 비용 민감 평가 지표

```python
from src.eval.metrics import cost_aware_score, far_at_recall

# FN 비용 = 100, FP 비용 = 5
score = cost_aware_score(y_true=y_test, y_pred_proba=y_prob, fn_cost=100, fp_cost=5)
far = far_at_recall(y_true, y_pred_proba, recall_threshold=0.95)
```

### 4. 설명 가능성

- **어텐션 시각화**: Cross-modal 어텐션 맵
- **SHAP 그래디언트**: 특성 중요도 및 모델 행동 분석
- **인과 DAG**: 변수 → 결함 → 수율 경로 시각화

### 5. 웹 대시보드 (선택)

- Streamlit 기반 실시간 데모 UI
- 데이터 업로드 및 인터랙티브 분석
- 사전 계산된 결과 조회

---

## 빠른 시작

### 환경 설치

```bash
# 가상환경 생성 (Python 3.11 이상)
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/Mac

# 의존성 설치
pip install --upgrade pip
pip install -r requirements.txt

# 개발 도구 (선택)
pip install -r requirements-dev.txt
```

### 30초 데모

```bash
# 1. 합성 데이터셋 생성
python scripts/generate_demo_data.py --output data/demo/sample.parquet

# 2. 모델 학습 (합성 데이터, 빠른 실행)
python scripts/train.py --synthetic-cycles 100 --batch-size 16 --epochs 5 --fast-dev-run

# 3. 평가
python scripts/predict.py \
  --data data/demo/sample.parquet \
  --checkpoint outputs/model.ckpt \
  --output reports/predictions.json

# 4. 결과 리포트 생성
python scripts/generate_html_report.py \
  --predictions reports/predictions.json \
  --output app/data/dashboard-results.html
```

### 대시보드 실행 방법

#### 방법 1: 더블클릭으로 실행 (가장 간단) ⭐

**Windows 사용자 추천:**
1. 프로젝트 폴더에서 `run_dashboard.py` 파일을 더블클릭
2. 자동으로 Streamlit 서버가 시작되고 브라우저가 열립니다
3. `http://localhost:8501` 에서 대시보드 확인

> ✅ **장점**: 터미널 명령어 없이 원클릭으로 실행, 완벽한 사용자 경험

#### 방법 2: 터미널에서 실행

```bash
# 프로젝트 루트에서
python run_dashboard.py
```

또는 직접 Streamlit 실행:
```bash
streamlit run app/streamlit_app.py
```

접속: `http://localhost:8501`

#### 방법 3: PyInstaller로 .exe 생성 (고급)

나중에 배포용 `.exe` 파일을 만들려면:

```bash
# 1. PyInstaller 설치
pip install pyinstaller

# 2. run_dashboard.py를 .exe로 변환
pyinstaller --onefile --windowed run_dashboard.py

# 3. 생성된 .exe는 dist/ 폴더에서 확인
# dist/run_dashboard.exe 를 더블클릭하면 대시보드 실행
```

---

#### 대시보드 기능

- **Overview**: 프로젝트 현황, 마일스톤 진척도, 고객 요청 데이터 항목 요약
- **Data**: 데모 데이터 생성 및 업로드
- **Train**: 모델 학습 및 하이퍼파라미터 조정
- **Predict**: 예측 결과 및 성능 메트릭
- **Explain**: SHAP, Attention 시각화
- **Causal**: 센서 간 인과관계 분석
- **Report**: 종합 리포트 다운로드

---

## 프로젝트 구조

```
Portfolio_pcb-lamination-press-defect-prediction/
├── src/
│   ├── data/
│   │   ├── loaders.py          # 데이터 로딩 및 전처리
│   │   ├── schema.py           # 도메인 스키마 (P013/P019)
│   │   ├── synthpress.py       # 합성 데이터 생성기
│   │   └── audit.py            # 데이터 검증 및 리포트
│   ├── models/
│   │   ├── pressfuse.py        # 멀티모달 융합 모델
│   │   ├── heads.py            # 태스크별 출력 헤드
│   │   └── baselines/          # 참조 베이스라인 모델
│   ├── training/
│   │   ├── module.py           # PyTorch Lightning 학습 모듈
│   │   └── callbacks.py        # 커스텀 콜백 (지표, 로깅)
│   ├── eval/
│   │   └── metrics.py          # 비용 민감 평가 지표
│   └── explain/
│       ├── attention_viz.py    # 어텐션 맵 시각화
│       └── shap_grad.py        # SHAP 그래디언트
│
├── scripts/
│   ├── train.py                # 학습 CLI
│   ├── eval.py                 # 평가 CLI
│   ├── predict.py              # 추론 CLI
│   ├── secom_baseline.py       # SECOM 베이스라인 실행
│   ├── generate_demo_data.py   # 데모 데이터 생성
│   ├── generate_html_report.py # 정적 리포트 생성
│   └── ui.py                   # Streamlit 대시보드
│
├── tests/                      # pytest 단위 테스트
├── configs/                    # Hydra 실험 설정
│   ├── experiment/
│   ├── data/
│   └── model/
├── data/
│   ├── raw/                    # 원본 데이터 (공개 데이터셋)
│   ├── processed/              # 전처리 완료 데이터
│   └── demo/                   # 빠른 시작용 데모 데이터
├── notebooks/                  # Jupyter 분석 노트북
├── docs/                       # 상세 문서
├── app/                        # 정적 대시보드 (GitHub Pages)
├── paper/                      # 논문 참고자료 및 노트
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── dvc.yaml                    # DVC 파이프라인
└── .gitignore
```

---

## 사용 예시

### 합성 데이터로 학습

```bash
python scripts/train.py \
  --synthetic-cycles 500 \
  --batch-size 32 \
  --epochs 10 \
  --learning-rate 1e-3 \
  --output-dir outputs/v1
```

### 라벨 데이터로 평가

```bash
python scripts/eval.py \
  --data-dir data/processed/my_dataset \
  --labels-path data/processed/my_dataset/labels.csv \
  --checkpoint outputs/v1/model.ckpt \
  --metrics auroc,far_at_recall,cost_aware
```

### SECOM 공개 데이터셋 베이스라인

```bash
python scripts/secom_baseline.py \
  --data-dir data/raw/secom \
  --target-length 192 \
  --model logistic_regression
```

### 전체 테스트 실행

```bash
# 전체 테스트
pytest tests/ -v

# 특정 모듈
pytest tests/test_synthpress.py -v

# 커버리지 포함
pytest tests/ --cov=src --cov-report=html
```

---

## 결과 및 데모

### 예측 출력 형식

```json
{
  "cycle_id": 42,
  "panel_id": 1042,
  "predictions": {
    "defect_probability": 0.87,
    "anomaly_type": "pressure_drop",
    "confidence": 0.92
  },
  "metrics": {
    "auroc": 0.9823,
    "precision": 0.94,
    "recall": 0.96
  },
  "explanation": {
    "top_features": ["HPPRESSPV", "PT1", "VACUUM"],
    "attention_weights": {}
  }
}
```

### 대시보드 HTML

```bash
python scripts/generate_html_report.py \
  --predictions outputs/predictions.json \
  --output app/data/dashboard.html
```

생성 후 `app/index.html`을 브라우저에서 열어 확인합니다.

---

## 데이터 스키마

### Press 공정 변수 (P013)

| 변수명 | 형식 | 범위 | 단위 | 설명 |
|--------|------|------|------|------|
| `HPPRESSPV` | Float | 0–99 | kgf/㎠ | 압력 (측정값) |
| `HPPRESSV` | Float | 0–99 | kgf/㎠ | 압력 (설정값) |
| `FHPPRESSPV` | Float | 0–45 | kgf/㎠ | 최종 압력 (측정값) |
| `VACUUM` | Float | 0–764 | mmHg | 진공 수준 |
| `HPTEMPSV` | Float | 40–230 | ℃ | 온도 설정값 |
| `PT1`–`PT9` | Float | 20–230 | ℃ | 플레이트 온도 (9채널) |

### 불량 유형 (P019)

| 코드 | 불량 유형 | 분류 | 비고 |
|------|-----------|------|------|
| P019-013 | VOID | 불량 | 수분 트래핑 |
| P019-014 | 외곽 VOID | 불량 | 엣지 박리 |
| P019-028 | Press 트러블 스크랩 | 설비 | 압착 공정 내 탐지 |
| P019-036 | 표면 VOID 스크랩 | 불량 | 베이킹 후 표면 불량 |
| P019-037 | 휨(Warping) 스크랩 | 불량 | XY 변형 |
| P019-0XX | (총 37가지) | 혼합 | `src/data/schema.py` 참고 |

---

## 배포 방법

### GitHub Pages (정적 사이트)

```bash
# 1. 정적 리포트 생성
python scripts/generate_html_report.py \
  --predictions outputs/predictions.json \
  --output app/index.html

# 2. GitHub 저장소 설정 → Pages → Branch: main / Folder: app

# 3. 커밋 후 푸시
git add app/ && git commit -m "docs: 대시보드 업데이트" && git push
```

### Streamlit Cloud

```bash
# 1. GitHub에 푸시 후
# 2. https://streamlit.io/cloud 접속
#    → New app → 저장소 선택 → "scripts/ui.py" 지정
```

### Docker

```bash
# 이미지 빌드
docker build -t pcb-press-pred:latest .

# 학습
docker run --gpus all pcb-press-pred:latest python scripts/train.py

# 추론 서비스
docker run -p 8080:8080 pcb-press-pred:latest python scripts/predict.py
```

자세한 배포 가이드: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)

---

## 개발 환경

### 개발 환경 설정

```bash
# 개발 의존성 설치
pip install -r requirements-dev.txt

# pre-commit 훅 설정
pre-commit install

# 코드 스타일 검사
ruff check src/
black --check src/

# 자동 포맷
black src/ && ruff check src/ --fix
```

### 코드 스타일

- Python 3.11+ 문법 (PEP 604: `X | None`)
- 모든 함수에 타입 힌트 필수
- Black 포맷 (88자 기준)
- Ruff 린트
- Google 스타일 독스트링

### 다음 단계

1. `data/raw/`에 SECOM 등 공개 데이터셋 배치
2. `src/data/synthpress.py`로 합성 Press 사이클 검증
3. `python scripts/secom_baseline.py --data-dir data/raw/secom` 로 베이스라인 평가
4. `python scripts/train.py --fast-dev-run --epochs 1 --batch-size 2`
5. `streamlit run scripts/ui.py` 로 진행 상황 모니터링

---

## 관련 문서

| 문서 | 설명 |
|------|------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 시스템 아키텍처 및 설계 결정 |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | 배포 절차 (GitHub Pages, Docker, Cloud) |
| [docs/API.md](docs/API.md) | 주요 모듈 API 레퍼런스 |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 기여 가이드라인 |
| [CHANGELOG.md](CHANGELOG.md) | 버전 이력 |

---

## 인용

이 프로젝트를 연구에 활용하시는 경우 아래와 같이 인용해 주세요.

```bibtex
@thesis{song2027cdpnet,
  author = {Song, Gong-Ho},
  title  = {Multi-Stage Causal Defect Propagation Network for PCB Lamination Press Anomaly Detection},
  school = {Chungbuk National University, Department of Industrial Artificial Intelligence},
  year   = {2027},
  month  = {February}
}
```

---

## 라이선스 및 주의사항

**라이선스**: MIT License ([LICENSE](LICENSE) 참고)

1. **원본 데이터 비공개**: 실제 제조 데이터는 기밀이며 이 저장소에 포함되지 않습니다.
   - 합성 데이터 및 공개 데이터셋(SECOM, DeepPCB)만 제공됩니다.
   - 실제 데이터 연동은 `src/data/schema.py`의 스키마를 참고하세요.

2. **연구 목적**: 학술 연구, 학위논문, 방법론 검증 및 산업 응용 베이스라인 비교를 위해 제작되었습니다.

3. **상업적 활용**: 원본 데이터 공유, 기술 이전, 상용 배포 관련 문의는 충북대학교로 연락 바랍니다.

---

## 연락처

**작성자**: 송공호  
**소속**: 충북대학교 산업인공지능학과  
**이메일**: songgongho@gmail.com  
**GitHub Issues**: 버그 보고 및 기능 제안 환영

---

**최종 수정**: 2026년 5월  
**상태**: 개발 중 (학위논문 제출 목표 2026년 10월)
