# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project aims to adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Synthetic Press cycle generator with 6 P013 anomaly types
- Cost-aware evaluation metrics (FAR@Recall, cost-sensitive F1)
- PyTorch Lightning training module with MLflow integration
- Streamlit dashboard for interactive analysis
- SECOM baseline implementation (sklearn + Transformer-AE)
- Data schema validation and audit tooling
- Type hints and comprehensive docstrings

### Fixed
- Data loader handles missing values gracefully
- Synthetic data respects domain constraints (pressure, temperature ranges)

### Planned
- [ ] Causal DAG learning (PCMCI, NOTEARS)
- [ ] Propagation GNN for multi-stage defect prediction
- [ ] Integration with real manufacturing MES data
- [ ] Multi-GPU training (DistributedDataParallel)
- [ ] ONNX model export for edge deployment

---

## [0.2.0-beta] - 2026-05-23

**Status**: Release Beta for GitHub  
**Focus**: GitHub Public Release Preparation

### Added
- Complete GitHub documentation structure:
  - `README.md` with comprehensive project overview
  - `SETUP.md` for installation guidance
  - `CONTRIBUTING.md` for contribution workflow
  - `CODE_OF_CONDUCT.md` for community standards
  - `.github/workflows/` for CI/CD automation
  - `docs/DEPLOYMENT.md`, `docs/API.md`, `docs/ARCHITECTURE.md`
- Static HTML dashboard generator (`scripts/generate_html_report.py`)
- Demo data generation script (`scripts/generate_demo_data.py`)
- Prediction CLI (`scripts/predict.py`)
- Improved `.gitignore` with comprehensive rules
- `requirements-dev.txt` for development dependencies
- Test fixtures and conftest setup (`tests/conftest.py`)

### Changed
- Reorganized project structure for open-source standards
- Enhanced `src/utils/` with config and path utilities
- Improved error handling in data loaders
- Updated documentation for public usage

### Removed
- Sensitive data paths from documentation
- Hardcoded local paths (replaced with utilities)
- Unnecessary .xlsx and .docx files (converted to markdown)

---

## [0.1.0] - 2026-05-17

**Status**: Initial Scaffold Release  
**Focus**: Core Infrastructure & Baselines

### Added
- Project scaffolding and folder structure
- Core data module:
  - `src/data/schema.py`: P013/P019 domain definitions
  - `src/data/synthpress.py`: Synthetic Press cycle generator
  - `src/data/loaders.py`: Data loading and preprocessing
  - `src/data/audit.py`: Data validation and reporting
- Core models module:
  - `src/models/pressfuse.py`: Multimodal fusion architecture
  - `src/models/heads.py`: Multi-task prediction heads
  - `src/models/baselines/secom.py`: Reference baseline
- Training module:
  - `src/training/module.py`: PyTorch Lightning trainer
  - `src/training/callbacks.py`: Custom callbacks
- Evaluation module:
  - `src/eval/metrics.py`: Cost-aware evaluation metrics
- Explanation module:
  - `src/explain/attention_viz.py`: Attention map visualization
  - `src/explain/shap_grad.py`: SHAP integration
- CLI scripts:
  - `scripts/train.py`: Training entry point
  - `scripts/eval.py`: Evaluation entry point
  - `scripts/secom_baseline.py`: SECOM baseline runner
  - `scripts/ui.py`: Streamlit dashboard
- Configuration framework:
  - `configs/experiment/`, `configs/data/`, etc. (Hydra YAML)
- Testing infrastructure:
  - `tests/` with unit tests for key modules
  - `pyproject.toml` with pytest configuration
- Documentation:
  - `paper/references.bib`: Bibliography for thesis
  - `paper/notes/`: Research paper summaries
- Version control:
  - `.gitignore` for Python/ML projects
  - `.pre-commit-config.yaml` for code quality hooks

### Configuration
- Black code formatter (88 char line length)
- Ruff linter with PEP8 compliance
- MLflow for experiment tracking
- DVC for data version control (optional)

---

## [Planned Releases]

### v0.3.0 (Q3 2026)
- Real manufacturing data integration
- Causal DAG learning implementation
- Enhanced model evaluation on production data

### v1.0.0 (Q4 2026 / Q1 2027)
- Thesis publication ready
- Production-grade documentation
- CI/CD pipeline (GitHub Actions)
- PyPI package release

### v1.1.0+ (Post-Thesis)
- Commercial licensing options
- Enterprise support
- Advanced optimization features

---

## Notes for Maintainers

### Version Increment Guide
- **MAJOR** (X.0.0): Breaking API changes, major research contribution
- **MINOR** (0.X.0): New features, backward compatible
- **PATCH** (0.0.X): Bug fixes, documentation, minor improvements

### Before Each Release
- [ ] Update version in `pyproject.toml`
- [ ] Update this CHANGELOG.md
- [ ] Run full test suite: `pytest tests/ -v`
- [ ] Verify documentation is up-to-date
- [ ] Create git tag: `git tag vX.Y.Z`
- [ ] Push tag: `git push origin vX.Y.Z`

---

## Contributors Acknowledged

*(To be updated with each release)*

**v0.2.0-beta**:
- Song Gong-Ho (primary author)

**v0.1.0**:
- Song Gong-Ho (primary author)

---

**Last Updated**: May 2026  
**Maintainer**: Song Gong-Ho  
**Thesis Submission Target**: February 2027

