# 멀티모달 융합 기반 PCB MLB Press 공정 이상·불량 예측 및 공정 조건 최적화

영문명: **PCB MLB Press Defect Prediction — Multimodal Fusion & Causal Analysis**

프로젝트명: **MS-CDPNet** (Multi-Stage Causal Defect Propagation Network)  
부제: 변수-결함-출하 다단계 인과 그래프 학습 기반 반도체 PCB 적층 공정 불량 예측 및 설명

**Author**: Song Gong-Ho (송공호), Student ID: 2025254010  
**Institution**: Chungbuk National University, Department of Industrial Artificial Intelligence  
**Project Type**: M.S. Thesis + Industrial Collaboration  
**Thesis Target**: February 2027  
**License**: MIT (see [LICENSE](LICENSE))  
**Repository**: `pcb-lamination-press-defect-prediction`

---

## 📋 Table of Contents
- [Overview](#overview)
- [Problem Definition](#problem-definition)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage Examples](#usage-examples)
- [Results & Demo](#results--demo)
- [Data Schema](#data-schema)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [Citation](#citation)
- [License & Disclaimer](#license--disclaimer)

---

## 🎯 Overview

### Research Problem

**Semiconductor PCB lamination (MLB press) processes** face critical yield challenges:
- **Press defects** (temperature anomalies, pressure loss, vacuum leaks) occur during the 10+ minute cycle
- **Downstream failures** (voids, warping, delamination) are detected only after expensive testing
- **Time lag** between Press (P013) and Outgoing Quality (P019) inspection makes root cause analysis difficult
- **Cost asymmetry**: False-negatives cause massive warranty costs; false-positives halt production

### Solution Approach

This project builds a **multimodal, explainable defect predictor** that:
1. Detects **Press anomalies in real-time** using time-series + event logs + AOI images
2. Predicts **downstream defect propagation** using causal DAG learning (PCMCI, NOTEARS)
3. Explains **defect mechanisms** via attention maps and SHAP gradients
4. Optimizes **process conditions** using the model as a differentiable surrogate

### Target Metric

- **AUROC ≥0.98** on P013 press anomaly detection
- **FAR@Recall=0.95** < 5% (cost-aware: FN weight = 100x FP weight)
- **Causal edge accuracy**: ≥85% on synthetic ground-truth DAG

---

## 🔬 Key Features

### 1. Synthetic Data Generator
```python
from src.data.synthpress import generate_press_cycle

frame, label, metadata = generate_press_cycle(
    cycle_id=1,
    panel_id=1001,
    anomaly_type="pressure_drop",  # P013-002
    anomaly_prob=0.5
)
```
- 6개 P013 단일 이상 시나리오
- 4가지 다중 이상 (realistic cascade patterns)
- 도메인 제약조건 기반 검증

### 2. Multimodal Fusion
- **PressFuse** model: Cross-modal attention + temporal convolution
- Inputs: Time-series (pressure, temperature, vacuum) + Categorical events + AOI images
- Outputs: Binary defect + Multitype classification + Anomaly confidence

### 3. Cost-Aware Metrics
```python
from src.eval.metrics import cost_aware_score, far_at_recall

# Cost matrix: FN cost = 100, FP cost = 5
score = cost_aware_score(
    y_true=y_test,
    y_pred_proba=y_prob,
    fn_cost=100,
    fp_cost=5
)
far = far_at_recall(y_true, y_pred_proba, recall_threshold=0.95)
```

### 4. Explainability
- **Attention visualization**: Cross-modal attention maps
- **SHAP gradients**: Feature importance & model behavior
- **Causal DAG**: Variable→Defect→Yield paths

### 5. Web Dashboard (Optional)
- Real-time Streamlit UI for demo
- Interactive data upload & analysis
- Pre-computed result inspection

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/username/pcb-lamination-press-defect-prediction.git
cd pcb-lamination-press-defect-prediction

# Create virtual environment (Python 3.11+)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# (Optional) Install development tools
pip install -r requirements-dev.txt
```

### 30-Second Demo

```bash
# Generate synthetic dataset
python scripts/generate_demo_data.py --output data/demo/sample.parquet

# Train model (synthetic, fast)
python scripts/train.py \
  --synthetic-cycles 100 \
  --batch-size 16 \
  --epochs 5 \
  --fast-dev-run

# Evaluate
python scripts/predict.py \
  --data data/demo/sample.parquet \
  --checkpoint outputs/model.ckpt \
  --output reports/predictions.json

# View results
python scripts/generate_html_report.py \
  --predictions reports/predictions.json \
  --output app/data/dashboard-results.html
```

### Web Dashboard (Streamlit)

```bash
streamlit run scripts/ui.py
# Opens: http://localhost:8501
```

---

## 📁 Project Structure

```
pcb-lamination-press-defect-prediction/
├── src/
│   ├── data/
│   │   ├── loaders.py       # Data loading & preprocessing
│   │   ├── schema.py        # Domain schema (P013/P019)
│   │   ├── synthpress.py    # Synthetic data generator
│   │   └── audit.py         # Data validation & reporting
│   ├── models/
│   │   ├── pressfuse.py     # Multi-modal fusion model
│   │   ├── heads.py         # Task-specific heads
│   │   └── baselines/       # Reference models
│   ├── training/
│   │   ├── module.py        # PyTorch Lightning module
│   │   └── callbacks.py     # Custom callbacks (metrics, logging)
│   ├── eval/
│   │   └── metrics.py       # Cost-aware evaluation metrics
│   ├── explain/
│   │   ├── attention_viz.py # Attention map visualization
│   │   └── shap_grad.py     # SHAP gradient integration
│   └── utils/
│       ├── config.py        # Configuration management
│       ├── logging.py       # Logging setup
│       └── paths.py         # Path utilities
│
├── scripts/
│   ├── train.py                     # Training CLI
│   ├── eval.py                      # Evaluation CLI
│   ├── predict.py                   # Inference CLI
│   ├── secom_baseline.py            # Reference baseline (SECOM dataset)
│   ├── generate_demo_data.py        # Demo data generation
│   ├── generate_html_report.py      # Static report generation
│   ├── ui.py                        # Streamlit dashboard
│   └── admin/                       # Maintenance scripts
│
├── tests/
│   ├── test_synthpress.py
│   ├── test_loaders.py
│   ├── test_metrics.py
│   ├── test_model.py
│   ├── test_training.py
│   └── fixtures/
│
├── configs/
│   ├── experiment/               # Experiment configurations
│   ├── data/                     # Data loading configs
│   └── model/                    # Model configs
│
├── data/
│   ├── raw/
│   │   ├── sample_synthetic.parquet  # Always included
│   │   ├── secom/                    # Public dataset
│   │   └── deeppcb/                  # Public dataset
│   ├── processed/                    # Preprocessed datasets
│   └── demo/                         # Demo data for quick start
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_model_analysis.ipynb
│   └── 03_deployment.ipynb
│
├── docs/
│   ├── DEPLOYMENT.md
│   ├── API.md
│   ├── DATA_SCHEMA.md
│   ├── ARCHITECTURE.md
│   └── REFACTORING.md
│
├── app/
│   ├── index.html                # Static dashboard entry
│   ├── css/style.css
│   ├── js/data-loader.js
│   └── data/
│
├── paper/
│   ├── references.bib
│   └── notes/
│
├── .github/
│   ├── workflows/
│   │   ├── tests.yml
│   │   └── pages.yml
│   ├── ISSUE_TEMPLATE/
│   └── PULL_REQUEST_TEMPLATE.md
│
├── .gitignore
├── .pre-commit-config.yaml
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── README.md                    # (You are here)
├── SETUP.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── CHANGELOG.md
└── LICENSE
```

---

## 📖 Usage Examples

### Example 1: Train on Synthetic Data

```bash
python scripts/train.py \
  --synthetic-cycles 500 \
  --batch-size 32 \
  --epochs 10 \
  --learning-rate 1e-3 \
  --output-dir outputs/v1
```

### Example 2: Evaluate with Domain Labels

```bash
python scripts/eval.py \
  --data-dir data/processed/my_dataset \
  --labels-path data/processed/my_dataset/labels.csv \
  --checkpoint outputs/v1/model.ckpt \
  --metrics auroc,far_at_recall,cost_aware
```

### Example 3: Run SECOM Baseline

```bash
python scripts/secom_baseline.py \
  --data-dir data/raw/secom \
  --target-length 192 \
  --model logistic_regression
```

### Example 4: Generate Predictions

```bash
python scripts/predict.py \
  --data data/demo/sample.parquet \
  --checkpoint outputs/v1/model.ckpt \
  --batch-size 128 \
  --output-format json
```

### Example 5: Test with Pytest

```bash
# Run all tests
pytest tests/ -v

# Run specific module
pytest tests/test_synthpress.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 📊 Results & Demo

### Sample Output

#### Prediction JSON
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
    "attention_weights": {...}
  }
}
```

#### Dashboard HTML
Visit `app/index.html` after running:
```bash
python scripts/generate_html_report.py \
  --predictions outputs/predictions.json \
  --output app/data/dashboard.html
```

Then open `app/index.html` in your browser.

---

## 📡 Data Schema

### Press Process Variables (P013)
| Variable | Type | Range | Unit | Domain |
|----------|------|-------|------|--------|
| `HPPRESSPV` | Float | 0-99 | kgf/㎠ | Pressure (measured) |
| `HPPRESSV` | Float | 0-99 | kgf/㎠ | Pressure (setpoint) |
| `FHPPRESSPV` | Float | 0-45 | kgf/㎠ | Final pressure (measured) |
| `VACUUM` | Float | 0-764 | mmHg | Vacuum level |
| `HPTEMPSV` | Float | 40-230 | ℃ | Setpoint temperature |
| `PT1`-`PT9` | Float | 20-230 | ℃ | Plate temperatures (9 channels) |

### Defect Labels (P019)
| Code | Defect Type | Category | Notes |
|------|-------------|----------|-------|
| P019-013 | VOID | Defect | Moisture trapping |
| P019-014 | Outer edge VOID | Defect | Edge delamination |
| P019-028 | Press trouble scrapped | Equipment | Detected at press |
| P019-036 | Surface VOID scrapped | Defect | Post-bake surface |
| P019-037 | Warping scrapped | Defect | XY deformation |
| P019-0XX | (37 types total) | Mixed | Refer to `src/data/schema.py` |

---

## 🌐 Deployment

### Option 1: GitHub Pages (Static Site)

```bash
# 1. Generate static report
python scripts/generate_html_report.py \
  --predictions outputs/predictions.json \
  --output app/index.html

# 2. Enable GitHub Pages in repo settings
#    → Source: "Deploy from a branch"
#    → Branch: "main", Folder: "app"

# 3. Push to GitHub
git add app/
git commit -m "docs: update dashboard"
git push

# 4. View at https://username.github.io/pcb-lamination-press-defect-prediction
```

### Option 2: Streamlit Cloud

```bash
# 1. Push to GitHub
git push origin main

# 2. Go to https://streamlit.io/cloud
#    → New app → select repository & "scripts/ui.py"

# 3. App runs automatically on updates
```

### Option 3: Docker (Production)

```bash
# Build image
docker build -t pcb-press-pred:latest .

# Run model training
docker run --gpus all pcb-press-pred:latest python scripts/train.py

# Run inference service
docker run -p 8080:8080 pcb-press-pred:latest python scripts/predict.py
```

Full deployment guide: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)

---

## 🔧 Development

### Setup Development Environment

See [SETUP.md](SETUP.md) for detailed instructions.

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Enable pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v

# Lint code
ruff check src/
black --check src/

# Auto-format
black src/
ruff check src/ --fix
```

### Code Style

- **Python 3.11+** syntax (PEP 604: `X | None`)
- **Type hints required** on all functions
- **Black** formatting (88 char line length)
- **Ruff** linting
- **Docstrings** in Google style

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| [SETUP.md](SETUP.md) | Installation & environment setup |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Deployment procedures (GitHub Pages, Docker, Cloud) |
| [docs/API.md](docs/API.md) | API reference for key modules |
| [docs/DATA_SCHEMA.md](docs/DATA_SCHEMA.md) | Complete data schema & mappings |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture & design decisions |
| [docs/REFACTORING.md](docs/REFACTORING.md) | Code quality improvements (backlog) |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution guidelines |
| [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) | Community guidelines |
| [CHANGELOG.md](CHANGELOG.md) | Version history |

---

## 🎓 Citation

If you use this project in your research, please cite:

```bibtex
@thesis{song2027cdpnet,
  author = {Song, Gong-Ho},
  title = {Multi-Stage Causal Defect Propagation Network for PCB Lamination Press Anomaly Detection},
  school = {Chungbuk National University, Department of Industrial Artificial Intelligence},
  year = {2027},
  month = {February}
}
```

---

## 📄 License & Disclaimer

**License**: MIT License (see [LICENSE](LICENSE))

### Important Notes

1. **Proprietary Data**: Original manufacturing data is confidential and NOT included.
   - Only synthetic and public datasets (SECOM, DeepPCB) are provided.
   - To integrate with your manufacturing data, follow the schema in `src/data/schema.py`.

2. **Research Purpose**: This code is intended for:
   - Academic research and thesis publication
   - Methodology validation and reproduction
   - Baseline comparison for industrial applications

3. **Commercial Use**: Contact Chungbuk National University for:
   - Proprietary data sharing agreements
   - Technology transfer licenses
   - Production deployment support

---

## 📞 Contact & Support

**Author**: Song Gong-Ho (송공호)  
**Email**: contact@example.com  
**Institution**: Chungbuk National University  
**Department**: Industrial Artificial Intelligence  

### Getting Help

- **Issues/Bugs**: [GitHub Issues](https://github.com/username/pcb-lamination-press-defect-prediction/issues)
- **Discussions**: [GitHub Discussions](https://github.com/username/pcb-lamination-press-defect-prediction/discussions)
- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md)

---

## 🔮 Future Work

- [ ] Integration with real-time MES data streaming
- [ ] Multi-equipment causal learning (Press units 3-6)
- [ ] Reinforcement learning for condition optimization
- [ ] Mobile app for in-plant monitoring
- [ ] Mixed-signal anomaly detection (voltage, current monitoring)

---

**Last Updated**: May 2026  
**Status**: 🚀 Active Development (Thesis Submission Feb 2027)


### 완료

- 프로젝트 기본 폴더 구조 생성
- `requirements.txt`, `.gitignore`, `.pre-commit-config.yaml`, `pyproject.toml` 구성
- `src/data/`
  - `synthpress.py`: 합성 Press 사이클 생성기
  - `audit.py`: 데이터 감사 및 마크다운 리포트
  - `schema.py`: Press 스키마 및 split spec
  - `loaders.py`: Press 로더, 리샘플링, group-aware split
- `src/eval/`
  - `metrics.py`: 비용가중 지표, 분류 리포트
- `src/models/`
  - `pressfuse.py`: cross-modal attention + 멀티태스크 모델
  - `heads.py`: binary / defect / anomaly heads
  - `baselines/secom.py`: SECOM용 sklearn 베이스라인
- `src/training/`
  - `module.py`: PyTorch Lightning 학습 모듈
- `scripts/`
  - `train.py`: synthetic 또는 실데이터용 학습 CLI
  - `eval.py`: 라벨된 파일 또는 synthetic 데모 평가 CLI
  - `secom_baseline.py`: SECOM 데이터셋 베이스라인 실행 CLI
  - `audit.py`: 데이터 감사 CLI
- `tests/`: 핵심 기능 단위 테스트

### 진행 중 / 다음 단계

- 실제 반도체 PCB 적층 공정 원본 데이터셋 연결
- Hydra 기반 실험 설정 확장
- MLflow/DVC 연동 강화
- AOI 이미지/이벤트 스트림 모듈 추가

### 웹 대시보드

- 실행: `streamlit run scripts/ui.py`
- Windows 보조 실행: `.\scripts\run_ui.ps1`
- 왼쪽 페이지 메뉴:
  - 대시보드
  - 연구 방향
  - 참고 자료
  - 용어 사전
  - 보안 / 운영 원칙
- 표시 항목: 개발 목표, 진행률, 데이터/리포트 상태, 다음 작업, 데이터셋 업로드/폴더 분석

#### 파일 업로드 설정

- **최대 업로드 크기**: 10GB (기본값 200MB → 상향 조정)
- **설정 적용 방법** (3단계 모두 자동 적용됨):
  1. **환경변수**: `STREAMLIT_SERVER_MAX_UPLOAD_SIZE=10000` (PowerShell 스크립트에 포함)
  2. **설정 파일**: `~/.streamlit/config.toml` 또는 `.streamlit/config.toml`
     ```toml
     [server]
     maxUploadSize = 10000
     ```
  3. **CLI 옵션**: `--server.maxUploadSize=10000`
- **더 큰 파일 분석**: 로컬 폴더 경로로 분석하면 크기 제한 없음
  - UI에서 "또는 로컬 폴더 경로" 입력 후 "분석 시작"
- **설정 확인**: `python check_streamlit_config.py`

### Docker / 배포 전략 검토

- 현재 단계에서는 `모델마다 Dockerfile을 완전히 분리`하기보다, `공통 베이스 이미지 1개 + 역할별 실행 명령`으로 시작하는 편이 유지보수에 유리합니다.
- 권장 구조
  - `ui`: Streamlit 대시보드
  - `train`: 학습/실험용 GPU 이미지
  - `eval` / `baseline`: 평가 및 베이스라인 실행용 이미지
- 이렇게 하는 이유
  - 공통 의존성(Python, pandas, sklearn, PyTorch, Streamlit)을 한 번만 관리할 수 있습니다.
  - 모델이 늘어나도 `entrypoint`나 `command`만 바꿔 재사용하기 쉽습니다.
  - 개발 초기에는 모델마다 의존성이 크게 갈리지 않으므로, Dockerfile이 너무 많아지는 것을 막을 수 있습니다.
- 모델별로 분리하는 것이 좋은 경우
  - 모델마다 CUDA / PyTorch / OpenCV 등 런타임이 크게 다를 때
  - 모델을 서로 다른 스케줄러/서버에 독립 배포해야 할 때
  - 추론 API가 모델별로 완전히 분리되어야 할 때
- 결론: 지금은 `1개 공통 이미지 + 역할별 컨테이너`를 추천하고, 실제 서빙 단계에서만 모델별 분리를 검토하는 것이 좋습니다.

### 데이터셋 분석 CLI

- 실행: `python scripts/analyze_dataset.py --source data/raw/secom --output-dir reports/analysis`
- 저장물: 분석 리포트 `.md`, 미리보기 `.csv`, 자료형 `.csv`, 수치형 요약 `.csv`

### 데이터셋 전처리 CLI

- 실행: `python scripts/preprocess_dataset.py --source data/raw/secom --output-dir data/processed/secom`
- 저장물: 정제된 `.csv`, `.parquet`, 전처리 리포트 `.md`, 메타데이터 `.json`

### SECOM 베이스라인

- 실행: `python scripts/secom_baseline.py --data-dir data/raw/secom --target-length 128`
- 모델: `StandardScaler + LogisticRegression(class_weight="balanced")`
- 출력: AUROC, FAR@Recall, 비용가중 점수, train/test 크기

### 기획 메모

- 연구 방향은 "멀티모달 불량 전파 예측 + 설명 가능성 + 비용민감 평가"를 기본축으로 둡니다.
- 보안 원칙은 원본 데이터 비외부화, 비밀정보 비저장, 업로드 파일 최소 보관입니다.

## 빠른 시작 (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pre-commit install
pytest
```

## 포함된 기본 구조

- `src/data/`: 데이터 로더, audit, 합성 시뮬레이터
- `src/eval/`: 평가지표
- `configs/`: Hydra 설정
- `paper/`: 논문 정리 노트, 참고문헌
- `scripts/`: 실행용 진입점
- `tests/`: 최소 단위 테스트

## 다음 단계

1. `data/raw/`에 SECOM 등 공개 데이터셋 배치
2. `src/data/synthpress.py`로 합성 Press 사이클 검증
3. `src/eval/metrics.py`를 기준으로 베이스라인 평가 시작
4. `python scripts/secom_baseline.py --data-dir data/raw/secom --target-length 128`
5. `python scripts/train.py --fast-dev-run --epochs 1 --batch-size 2`
6. `python scripts/eval.py --synthetic-cycles 12`
7. `streamlit run scripts/ui.py`로 진행 상황을 상시 모니터링

## 추가 도구 및 재현성

- EDA 복구 스크립트:
  - `python scripts/eda_secom.py`  # SECOM EDA 리포트 생성
  - `python scripts/eda_deeppcb.py` # DeepPCB EDA 리포트 생성

- 자동 전처리 템플릿 생성:
  - `python scripts/generate_preprocess_template.py --source data/raw/secom --output-dir configs/preprocess_templates`

- MLflow 연동 (학습 추적):
  - 빠른 실행: `python scripts/train.py --fast-dev-run --use-mlflow`
  - 기본 실험 로컬 저장소: `./mlruns`

- Hydra 기반 재현 예시:
  - `python scripts/train_hydra.py`  # baseline 설정으로 빠른 실행

