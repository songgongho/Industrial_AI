# Refactoring & Code Quality Checklist

This document tracks code improvements and technical debt items.

## Priority 1 (Must Fix Before Thesis)

### Module: src/data/loaders.py
- [ ] Remove hardcoded `target_length=192` → make configurable parameter
- [ ] Add validation for missing file paths (raises FileNotFoundError with helpful message)
- [ ] Add logging for data loading progress (tqdm or logging.info)
- [ ] Type hint for all function parameters and returns
- [ ] Add docstring examples for PressDataset usage
- [ ] Handle edge case: empty dataframe input

### Module: src/models/pressfuse.py
- [ ] Split forward() into _encode_timeseries(), _encode_categorical(), _fuse() (single responsibility)
- [ ] Add attention_weights return to forward() for explainability
- [ ] Matrix dimensions validation (assert shapes match)
- [ ] Remove magic numbers (64, 12, 4) → use class attributes
- [ ] Add comprehensive docstring with input/output shapes

### Module: src/training/module.py
- [ ] Add training step logging (lr, loss, batch_size)
- [ ] Add learning rate scheduler (ReduceLROnPlateau or CosineAnneal)
- [ ] Remove hardcoded fn_cost=100, fp_cost=5 → make configurable
- [ ] Add checkpoint save logic (best model by AUROC)
- [ ] Add early stopping (if validation metric plateaus)

### Module: src/eval/metrics.py
- [ ] Validate input shapes and types (raise ValueError if invalid)
- [ ] Add edge case handling:
  - [ ] Empty y_true
  - [ ] All samples same class
  - [ ] All probabilities identical
  - [ ] Probability outside [0, 1]
- [ ] Add docstring examples for each metric
- [ ] Add sklearn equivalence tests (cost_aware_score vs manual)

### Module: scripts/train.py, scripts/eval.py
- [ ] Remove `sys.path.append()` hack → use proper package import
- [ ] Add argument validation (e.g., epochs > 0, batch_size > 0)
- [ ] Add error handling for missing data files
- [ ] Add progress bar (tqdm) for long operations
- [ ] Add summary of results to stdout/log file

### Module: scripts/ui.py
- [ ] Remove absolute paths (e.g., `C:/Users/...` paths)
- [ ] Use `Path` from pathlib for all file operations
- [ ] Add error handling for file upload failures
- [ ] Add logging across UI interactions
- [ ] Performance: cache data loading results (st.cache)

### Global: Codebase
- [ ] Remove all Windows-specific paths (hardcoded `C:\...`)
- [ ] Replace print() with logging.info() / logging.warning()
- [ ] Add type hints to ALL functions (target: 100%)
- [ ] Add docstrings to ALL public functions (target: 100%)
- [ ] Remove commented-out code (commits are for history, not comments)
- [ ] Extract magic numbers → constants file (config.py)

---

## Priority 2 (Nice to Have, Before Release)

### Refactoring: Simplification
- [ ] Create `src/utils/config.py` for centralized config (paths, constants, ranges)
  ```python
  # config.py
  PRESSURE_RANGE = (0, 99)
  VACUUM_RANGE = (0, 764)
  TEMP_RANGE = (20, 230)
  DEFAULT_BATCH_SIZE = 32
  FN_COST_DEFAULT = 100
  FP_COST_DEFAULT = 5
  ```
- [ ] Create `src/utils/logging.py` for consistent logging setup
  ```python
  # logging.py
  def get_logger(name: str) -> logging.Logger:
      """Get configured logger instance."""
  ```
- [ ] Create `src/data/constants.py` for P013/P019 label mappings
  ```python
  P013_LABELS = {
      'P013-001': 'Temperature Anomaly',
      'P013-002': 'Pressure Drop',
      ...
  }
  ```

### Testing: Expand Coverage
- [ ] Add conftest.py with common fixtures (sample_data, sample_model, tmp_dir)
- [ ] Add edge case tests:
  - [ ] Empty dataset
  - [ ] Single sample
  - [ ] All zeros / all ones
  - [ ] NaN values
  - [ ] Inf values
  - [ ] Wrong shape input
- [ ] Add integration tests (end-to-end: load → train → eval → predict)
- [ ] Add performance tests (batch processing speed benchmark)
- [ ] Target coverage: 80%+ (currently ~60%)

### Documentation
- [ ] Add type stub files (.pyi) for complex types
- [ ] Add examples to docstrings (every function has Example section)
- [ ] Add architecture diagrams (ASCII or image)
- [ ] Add data flow diagrams
- [ ] Add model capacity analysis (params, FLOPs)
- [ ] Add performance benchmarks (training time, inference latency)

### Dependencies
- [ ] Audit each dependency for security vulnerabilities
  ```bash
  pip audit
  ```
- [ ] Set version bounds in requirements.txt (currently too loose)
  - [ ] `torch>=2.2,<3.0` instead of `torch>=2.2`
  - [ ] Similar for other ML packages
- [ ] Create requirements-gpu.txt with CUDA-specific versions

### Performance
- [ ] Profile data loading (find bottlenecks)
  ```bash
  py-spy record -o profile.svg -- python scripts/train.py
  ```
- [ ] Optimize hot paths (likely: data encoding, distance computation)
- [ ] Consider using TorchScript for inference speedup
- [ ] Benchmark inference latency on different hardware

### Deployment
- [ ] Create Dockerfile (Docker support)
- [ ] Create docker-compose.yml (for dev environment)
- [ ] Add CI/CD tests for Docker builds
- [ ] Create k8s manifests (if deploying to cloud)

---

## Priority 3 (Post-Thesis, If Time Allows)

### Advanced Features
- [ ] Multi-GPU training support (DistributedDataParallel)
- [ ] Mixed precision training (torch.cuda.amp)
- [ ] ONNX export for edge deployment
- [ ] TorchScript compilation for inference
- [ ] Quantization support (int8) for model compression

### Monitoring & Observability
- [ ] Prometheus metrics export
- [ ] Weights & Biases (wandb) integration
- [ ] Integration tests with real manufacturing data
- [ ] Production ML monitoring (data drift detection)

### Research Direction
- [ ] Causal DAG learning (PCMCI, NOTEARS)
- [ ] Causal Forest for heterogeneous treatment effects
- [ ] Active learning for label efficiency
- [ ] Few-shot adaptation to new equipment

---

## Long-Term Tech Debt

### Architectural
- [ ] Separate data preprocessing pipeline (separate CLI)
- [ ] Separate model serialization (SavedModel format, not just .ckpt)
- [ ] API service layer (FastAPI for inference)
- [ ] Async data loading / prefetching

### Code Organization
- [ ] Move constants to separate files (less magic numbers)
- [ ] Extract utility functions (reduce duplication)
- [ ] Standardize error handling (custom exceptions)
- [ ] Add retry logic for flaky I/O operations

---

## Completed Refactorings ✅

- [x] Initial scaffold structure creation
- [x] Cost-aware metrics implementation
- [x] Type hints on core modules (partial)
- [x] Docstrings on public APIs (partial)
- [x] Synthetic data generator
- [x] GitHub documentation structure
- [x] CI/CD pipeline setup

---

## Code Quality Metrics

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Type Hint Coverage | 100% | ~70% | ⚠️ In Progress |
| Docstring Coverage | 100% | ~75% | ⚠️ In Progress |
| Test Coverage | 80%+ | ~60% | ⚠️ Improving |
| Cyclomatic Complexity | < 10 | ~8 (avg) | ✅ Good |
| Lines per Function | < 30 | ~25 (avg) | ✅ Good |
| Ruff Score | 0 errors | 0 errors | ✅ Pass |
| Black Format | Pass | Pass | ✅ Pass |

---

## How to Help with Refactoring

1. Pick an item from Priority 1
2. Create branch: `refactor/descriptive-name`
3. Make changes + add tests
4. Submit PR referencing this checklist
5. Update status here (check the box)

Example PR title: `[refactor] Extract magic numbers from data loader`

---

## Code Review Checklist

Before submitting any PR, verify:

- [ ] Ruff passes: `ruff check src/`
- [ ] Black passes: `black --check src/`
- [ ] Tests pass: `pytest tests/ -v`
- [ ] Coverage > 80%: `pytest --cov-report=term-missing`
- [ ] No hardcoded paths
- [ ] No print() statements (use logging)
- [ ] Type hints on all public functions
- [ ] Docstrings on all public functions
- [ ] No commented-out code
- [ ] No new dependencies added without justification

---

**Last Updated**: May 2026  
**Maintainer**: Song Gong-Ho  
**Next Review**: Before thesis submission (Feb 2027)

