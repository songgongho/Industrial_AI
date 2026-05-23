# Contributing to MS-CDPNet

Thanks for your interest in contributing! This guide explains how to report issues, suggest improvements, and submit pull requests.

## Code of Conduct

This project aims to be inclusive and welcoming. Please read our [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

---

## How to Contribute

### 1. Reporting Bugs

Found a bug? Please report it as an issue on [GitHub Issues](https://github.com/your-username/pcb-lamination-press-defect-prediction/issues).

**Use the bug report template:**
- Describe what happened
- Steps to reproduce
- Expected vs. actual behavior
- Environment (OS, Python version, torch version)
- Error traceback (if applicable)

Example:
```
**Bug**: Model training crashes with CUDA OOM on batch size 32
**Steps**:
1. `python scripts/train.py --batch-size 32`
2. Wait 5 minutes...
3. See "RuntimeError: CUDA out of memory"

**Environment**:
- OS: Ubuntu 22.04
- Python: 3.11.6
- torch: 2.2.1+cu121
- GPU: RTX 3090 (24GB)
```

### 2. Requesting Features

Have an idea? Open a [Feature Request](https://github.com/your-username/pcb-lamination-press-defect-prediction/issues).

**Describe**:
- What's the use case?
- Why is it useful?
- Proposed approach (optional)
- Related research/papers (optional)

Example:
```
**Feature**: Support multi-GPU training with DistributedDataParallel

**Use case**: Speed up training on multi-GPU systems (DDP commonly used in PyTorch Lightning)

**Proposed**: Switch from DataParallel to DDP in src/training/module.py
```

### 3. Discussing Ideas

Want to discuss before coding? Use [GitHub Discussions](https://github.com/your-username/pcb-lamination-press-defect-prediction/discussions).

---

## Development Workflow

### Step 1: Fork & Clone

```bash
# Fork on GitHub (click "Fork" button)

# Clone your fork
git clone https://github.com/YOUR-USERNAME/pcb-lamination-press-defect-prediction.git
cd pcb-lamination-press-defect-prediction

# Add upstream remote
git remote add upstream https://github.com/original-author/pcb-lamination-press-defect-prediction.git
```

### Step 2: Create Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-number
```

**Naming conventions:**
- Features: `feature/descriptive-name`
- Bugfixes: `bugfix/issue-number` or `bugfix/descriptive-name`
- Documentation: `docs/topic-name`

### Step 3: Setup Development Environment

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
pip install -r requirements-dev.txt
pre-commit install
```

### Step 4: Make Changes

**Code style**:
- Follow PEP 8 with Black (88 char line length)
- Use type hints: `def func(x: int) -> str:`
- Write docstrings (Google style):
  ```python
  def example_function(param1: int) -> bool:
      """One-line description.
      
      Longer description if needed.
      
      Args:
          param1: Description of param1
          
      Returns:
          Description of return value
          
      Raises:
          ValueError: When X happens
          
      Example:
          >>> example_function(5)
          True
      """
  ```

**Commit messages**:
- Use imperative mood: "Add feature" not "Added feature"
- Keep first line under 50 characters
- Reference issues: "Fix issue #42: ..."
- Example:
  ```
  [src] Add cost-aware evaluation metric
  
  - Implement cost_aware_score() with configurable FN/FP weights
  - Add tests with 3 scenarios (cost matrix, unbalanced data)
  - Update docs/API.md with usage example
  
  Closes #42
  ```

### Step 5: Write/Update Tests

Add tests for new code:

```bash
# Pytest automatically discovers files matching test_*.py

# Test structure
tests/
├── test_your_feature.py
├── fixtures/
│   └── sample_data.py          # Reusable test data
└── integration/
    └── test_your_integration.py
```

**Example test**:
```python
import pytest
from src.eval.metrics import cost_aware_score

@pytest.fixture
def sample_labels():
    return [0, 1, 0, 1, 1]

@pytest.fixture
def sample_predictions():
    return [0.2, 0.8, 0.3, 0.7, 0.9]

def test_cost_aware_score(sample_labels, sample_predictions):
    score = cost_aware_score(
        y_true=sample_labels,
        y_pred_proba=sample_predictions,
        fn_cost=100,
        fp_cost=5
    )
    assert 0 <= score <= 1, "Score must be between 0 and 1"
    assert score > 0.8, "Good predictions should score high"
```

**Run tests locally**:
```bash
pytest tests/ -v                           # All tests
pytest tests/test_your_feature.py -v       # Single file
pytest tests/ --cov=src --cov-report=html  # With coverage
```

All tests **must pass** before submitting PR.

### Step 6: Format & Lint

```bash
# Auto-format
black src/ tests/

# Check for issues
ruff check src/ tests/ --fix

# (Optional) Type checking
mypy src/
```

The pre-commit hook will run these automatically on commit.

### Step 7: Push & Create Pull Request

```bash
git add .
git commit -m "[module] Your meaningful commit message"
git push origin feature/your-feature-name
```

Then on GitHub:
1. Go to your fork
2. Click "Compare & pull request"
3. **Fill out PR template** (see below)

**PR Template (auto-loaded from `.github/PULL_REQUEST_TEMPLATE.md`):**
```markdown
## Description
Brief description of changes

## Related Issue(s)
Fixes #42

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Performance improvement

## Testing
- [ ] Added/updated tests
- [ ] All tests pass locally
- [ ] Tested on edge cases

## Checklist
- [ ] Code follows style guide (black, ruff)
- [ ] Docstrings added/updated
- [ ] Tests written and passing
- [ ] Type hints included
- [ ] No breaking changes to public API
```

### Step 8: Review & Iterate

- Maintainer reviews your code
- Address feedback (push additional commits, don't force-push)
- Once approved, maintainer merges the PR

---

## Code Review Checklist

Before submitting, self-review against:

### Correctness
- [ ] Logic is sound (test with edge cases)
- [ ] Handles errors gracefully
- [ ] No hardcoded paths or secrets
- [ ] Works on Windows/macOS/Linux

### Quality
- [ ] Follows style guide (Black, Ruff)
- [ ] Type hints on all functions
- [ ] Docstrings present
- [ ] No unused imports
- [ ] Meaningful variable names

### Testing
- [ ] Unit tests written
- [ ] Edge cases covered (empty input, NaN, shape mismatch)
- [ ] At least 80% test coverage
- [ ] `pytest tests/ -v` passes locally

### Documentation
- [ ] README updated (if user-facing)
- [ ] Docstrings explain unusual behavior
- [ ] CHANGELOG.md updated (for releases)
- [ ] Examples provided (for new features)

### Performance
- [ ] No unnecessary loops
- [ ] Uses vectorized ops (NumPy, PyTorch) when possible
- [ ] Memory usage is reasonable

---

## Contribution Guidelines

### Good PRs

✅ **Address a single issue**
- Not mixing refactoring with features
- Focused scope

✅ **Well-tested**
- New tests for new code
- All tests passing
- >80% coverage

✅ **Well-documented**
- Docstrings with examples
- README/API docs updated
- Commit messages explain *why*

✅ **Backward compatible**
- No breaking API changes in minor versions
- If necessary, include migration guide

### PRs We May Reject

❌ **Large refactoring without discussion**
- Open an issue first for big changes

❌ **Missing tests**
- We enforce pytest coverage

❌ **No docstrings or type hints**
- Required for all new code

❌ **Hardcoded paths or credentials**
- Never commit secrets
- Use `src/utils/paths.py` utilities

---

## Domain-Specific Guidelines

### Working with Press Data

When modifying data loaders or metrics:
1. Verify variable ranges match domain spec in `CLAUDE.md`
   - `HPPRESSPV`: 0-99 kgf/㎠
   - `VACUUM`: 0-764 mmHg
   - Temperatures: 20-230℃
2. Test with synthetic data: `python scripts/generate_demo_data.py`
3. Ensure labels match P013/P019 definitions

### Adding New Metrics

1. Add to `src/eval/metrics.py`
2. Document cost assumptions (if any)
3. Include example usage
4. Add unit test + integration test
5. Update `docs/API.md`

### Adding New Models

1. Add to `src/models/`
2. Inherit from PyTorch + Lightning when possible
3. Support `forward()` and `configure_optimizers()`
4. Test with synthetic data (see `test_model.py`)
5. Document in `docs/ARCHITECTURE.md`

---

## Release Process

(Maintainers only)

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md` with features/fixes
3. Create tag: `git tag v1.0.0`
4. Push: `git push origin v1.0.0`
5. GitHub Actions auto-publishes to PyPI (if configured)

---

## Getting Help

- **Setup issues**: See [SETUP.md](SETUP.md)
- **API questions**: Check [docs/API.md](docs/API.md)
- **Design discussions**: [GitHub Discussions](https://github.com/your-username/pcb-lamination-press-defect-prediction/discussions)
- **Chat**: Optionally set up Discord/Slack (not yet available)

---

## Recognition

Contributors will be acknowledged in:
- Commit history (GitHub)
- [CHANGELOG.md](CHANGELOG.md) (for releases)
- Your GitHub profile (via commits)

---

**Thank you for contributing! 🎉**

