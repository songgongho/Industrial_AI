# API Reference

Complete reference for key modules and functions.

## Table of Contents
- [Data Module](#data-module)
- [Models Module](#models-module)
- [Training Module](#training-module)
- [Evaluation Module](#evaluation-module)
- [Explain Module](#explain-module)
- [CLI Scripts](#cli-scripts)

---

## Data Module

### `src.data.schema`

Domain-specific schema definitions for Press process.

#### `PressFeatureSpec`
```python
class PressFeatureSpec:
    """Press process feature definitions."""
    
    feature_columns: list[str]  # All numerical feature names
    categorical_columns: list[str]  # Event/category columns
    
    # Ranges
    PRESSURE_MIN, PRESSURE_MAX = 0, 99
    VACUUM_MIN, VACUUM_MAX = 0, 764
    TEMP_MIN, TEMP_MAX = 20, 230
```

### `src.data.synthpress`

Synthetic Press cycle generator.

#### `generate_press_cycle()`
```python
def generate_press_cycle(
    cycle_id: int,
    panel_id: int,
    anomaly_type: AnomalyType | None = None,
    anomaly_prob: float = 0.3,
    seed: int | None = None,
) -> tuple[pd.DataFrame, int, dict[str, Any]]:
    """Generate synthetic Press cycle with optional anomalies.
    
    Args:
        cycle_id: Unique cycle identifier
        panel_id: Panel ID for traceability
        anomaly_type: One of ["temp_offset", "pressure_drop", "vacuum_leak", ...]
        anomaly_prob: Probability of anomaly injection (0-1)
        seed: Random seed for reproducibility
        
    Returns:
        (frame, label, metadata): DataFrame with features, binary label, metadata
        
    Example:
        >>> frame, label, meta = generate_press_cycle(1, 1001, anomaly_type="pressure_drop")
        >>> assert frame.shape[0] > 10  # Multiple time steps
        >>> assert label in [0, 1]
    """
```

**Anomaly Types**:
- `"temp_offset"`: Temperature deviation (P013-001)
- `"pressure_drop"`: Pressure loss (P013-002)
- `"vacuum_leak"`: Vacuum instability (P013-003)
- `"equipment_fault"`: Equipment malfunction (P013-004)
- `"power_loss"`: Brief power loss with gaps (P013-005)
- `"program_mismatch"`: Setpoint vs actual mismatch (P013-006)

### `src.data.loaders`

Data loading and batching.

#### `PressDataset`
```python
class PressDataset(Dataset):
    """PyTorch Dataset for Press cycles."""
    
    def __init__(
        self,
        data_dir: str | Path,
        labels_path: str | Path | None = None,
        target_length: int = 192,
        mapping_path: str | Path | None = None,
    ):
        """Load Press dataset.
        
        Args:
            data_dir: Directory containing parquet/CSV files
            labels_path: Path to labels mapping file
            target_length: Pad/truncate frames to this length
            mapping_path: Path to cycle-to-label mapping
        """
        
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]: ...
```

### `src.data.audit`

Data validation and reporting.

#### `audit_dataset()`
```python
def audit_dataset(
    data_path: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Validate dataset structure and generate audit report.
    
    Returns:
        Dictionary with statistics like:
        {
            'total_samples': 1250,
            'missing_values': {'HPPRESSPV': 0.2, ...},
            'feature_ranges': {'HPPRESSPV': (0, 99), ...},
            'label_distribution': {0: 1200, 1: 50},
            'report_path': 'path/to/audit.md'
        }
    """
```

---

## Models Module

### `src.models.pressfuse`

Multimodal fusion model for Press defect prediction.

#### `PressFuse`
```python
class PressFuse(pl.LightningModule):
    """Multi-modal fusion model with attention.
    
    Inputs:
        - Time series (pressure, temperature, vacuum): (B, T, 12)
        - Categorical features (events): (B, 4)
        - (Optional) AOI image: (B, 3, 256, 256)
        
    Outputs:
        - Binary defect probability: (B,)
        - Defect type logits: (B, 37) [P019 categories]
        - Anomaly confidence: (B,)
    """
    
    def __init__(
        self,
        time_series_dim: int = 12,
        categorical_dim: int = 4,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
    ):
        """Initialize PressFuse model."""
        
    def forward(
        self,
        ts: torch.Tensor,  # (B, T, 12)
        cat: torch.Tensor,  # (B, 4)
        img: torch.Tensor | None = None,  # (B, 3, 256, 256)
    ) -> dict[str, torch.Tensor]:
        """Forward pass.
        
        Returns:
            {
                'defect_prob': torch.Tensor,
                'defect_type': torch.Tensor,
                'anomaly_conf': torch.Tensor,
                'attention_weights': torch.Tensor,
            }
        """
```

### `src.models.heads`

Task-specific prediction heads.

#### `BinaryDefectHead`
```python
class BinaryDefectHead(nn.Module):
    """Binary defect detection head (0=normal, 1=defective)."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return probability of defect (B,)."""
```

#### `MultitypeHead`
```python
class MultitypeHead(nn.Module):
    """Multi-class defect type prediction (37 P019 categories)."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for each defect type (B, 37)."""
```

---

## Training Module

### `src.training.module`

PyTorch Lightning training wrapper.

#### `PCBStackLightningModule`
```python
class PCBStackLightningModule(pl.LightningModule):
    """Lightning module for Press defect prediction."""
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        fn_cost: float = 100.0,
        fp_cost: float = 5.0,
    ):
        """Initialize training module.
        
        Args:
            model: Backbone model (e.g., PressFuse)
            learning_rate: Adam learning rate
            fn_cost: Cost of false negatives (high = penalize missing defects)
            fp_cost: Cost of false positives (low = tolerate false alarms)
        """
        
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """Compute training loss with cost weighting."""
        
    def validation_step(self, batch, batch_idx) -> None:
        """Compute validation metrics."""
        
    def configure_optimizers(self) -> dict:
        """Return optimizer config."""
```

### `src.training.callbacks`

Custom PyTorch Lightning callbacks.

#### `MetricsCallback`
```python
class MetricsCallback(Callback):
    """Log cost-aware metrics on validation."""
    
    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        """Compute and log FAR, AUROC, cost-aware score."""
```

---

## Evaluation Module

### `src.eval.metrics`

Cost-aware evaluation metrics.

#### `cost_aware_score()`
```python
def cost_aware_score(
    y_true: np.ndarray | list[int],
    y_pred_proba: np.ndarray | list[float],
    fn_cost: float = 100.0,
    fp_cost: float = 5.0,
    threshold: float = 0.5,
) -> float:
    """Compute cost-aware evaluation score.
    
    Cost = FN_count * fn_cost + FP_count * fp_cost
    Score = 1 - (Cost / max_possible_cost)
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities [0, 1]
        fn_cost: Cost of missing a defect
        fp_cost: Cost of false alarm
        threshold: Classification threshold
        
    Returns:
        Score in range [0, 1] (higher is better)
        
    Example:
        >>> score = cost_aware_score([0, 1, 1], [0.2, 0.9, 0.3])
        >>> assert 0 <= score <= 1
    """
```

#### `far_at_recall()`
```python
def far_at_recall(
    y_true: np.ndarray | list[int],
    y_pred_proba: np.ndarray | list[float],
    recall_threshold: float = 0.95,
) -> float:
    """Compute False Alarm Rate at a target Recall.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        recall_threshold: Target recall level (e.g., 0.95)
        
    Returns:
        False alarm rate (lower is better, range [0, 1])
        
    Example:
        >>> far = far_at_recall([0, 1, 1], [0.2, 0.9, 0.3], recall_threshold=0.95)
        >>> assert 0 <= far <= 1
    """
```

#### `auroc_score()`
```python
def auroc_score(
    y_true: np.ndarray | list[int],
    y_pred_proba: np.ndarray | list[float],
) -> float:
    """Area Under ROC Curve (sklearn wrapper).
    
    Returns:
        AUROC score in [0, 1] (0.5=random, 1.0=perfect)
    """
```

---

## Explain Module

### `src.explain.attention_viz`

Attention-based explanations.

#### `visualize_attention()`
```python
def visualize_attention(
    attention_weights: torch.Tensor,  # (num_heads, T, T)
    feature_names: list[str],
    save_path: str | Path | None = None,
) -> str:
    """Visualize attention weights as heatmap.
    
    Returns:
        Path to saved figure
    """
```

### `src.explain.shap_grad`

SHAP gradient-based explanations.

#### `compute_shap_values()`
```python
def compute_shap_values(
    model: nn.Module,
    data: torch.Tensor,
    target_class: int = 1,
) -> np.ndarray:
    """Compute SHAP values for input features.
    
    Args:
        model: PyTorch model
        data: Input batch (B, T, D)
        target_class: Which output to explain (e.g., 1=defect)
        
    Returns:
        SHAP values shape (B, T, D)
    """
```

---

## CLI Scripts

### `scripts/train.py`

Training entry point.

```bash
python scripts/train.py \
  --synthetic-cycles 500 \
  --batch-size 32 \
  --epochs 10 \
  --learning-rate 1e-3 \
  --output-dir outputs/v1
```

**Arguments**:
- `--synthetic-cycles`: Number of synthetic samples (default: 100)
- `--batch-size`: Batch size (default: 32)
- `--epochs`: Number of training epochs (default: 10)
- `--learning-rate`: Learning rate (default: 1e-3)
- `--output-dir`: Output directory for checkpoints (default: outputs/)
- `--use-mlflow`: Enable MLflow logging (flag)
- `--fast-dev-run`: Quick 1-epoch test (flag)

### `scripts/eval.py`

Evaluation entry point.

```bash
python scripts/eval.py \
  --data data/processed/dataset.parquet \
  --checkpoint outputs/v1/model.ckpt \
  --metrics auroc,far_at_recall,cost_aware
```

### `scripts/predict.py`

Inference entry point.

```bash
python scripts/predict.py \
  --data data/demo/sample.parquet \
  --checkpoint outputs/v1/model.ckpt \
  --batch-size 128 \
  --output-format json
```

### `scripts/secom_baseline.py`

Reference baseline on SECOM dataset.

```bash
python scripts/secom_baseline.py \
  --data-dir data/raw/secom \
  --target-length 192 \
  --model logistic_regression
```

---

## Utility Functions

### `src.utils.paths`

Path utilities for cross-platform compatibility.

```python
from src.utils.paths import get_project_root, get_data_dir

root = get_project_root()  # Project root directory
data_dir = get_data_dir()  # data/ directory
```

### `src.utils.config`

Configuration management.

```python
from src.utils.config import load_config

cfg = load_config("configs/experiment/baseline.yaml")
# cfg.model.embedding_dim
# cfg.training.learning_rate
```

### `src.utils.logging`

Logging setup.

```python
from src.utils.logging import get_logger

logger = get_logger(__name__)
logger.info("Training started")
```

---

## Type Hints Reference

Key type aliases used throughout:

```python
from typing import Literal

AnomalyType = Literal[
    "temp_offset",
    "pressure_drop",
    "vacuum_leak",
    "equipment_fault",
    "power_loss",
    "program_mismatch",
]

DefectType = Literal[
    "void",
    "warping",
    "delamination",
    "surface_defect",
    "edge_defect",
]
```

---

## Common Recipes

### Recipe 1: Train Model on Synthetic Data

```python
from src.data.synthpress import generate_press_cycle
from src.data.loaders import PressDataset
from src.models.pressfuse import PressFuse
from src.training.module import PCBStackLightningModule
import pytorch_lightning as pl

# Generate synthetic dataset
xs, ys = [], []
for i in range(100):
    frame, label, _ = generate_press_cycle(i, 1000+i)
    xs.append(frame)
    ys.append(label)

# Create DataLoader
from torch.utils.data import DataLoader, TensorDataset
import torch
dataset = TensorDataset(torch.stack(...), torch.tensor(ys))
loader = DataLoader(dataset, batch_size=32)

# Train
model = PressFuse()
module = PCBStackLightningModule(model, learning_rate=1e-3)
trainer = pl.Trainer(max_epochs=10)
trainer.fit(module, train_dataloaders=loader)
```

### Recipe 2: Evaluate Model

```python
from src.eval.metrics import auroc_score, far_at_recall, cost_aware_score
import numpy as np

y_true = np.array([0, 1, 0, 1, 1])
y_pred_proba = np.array([0.2, 0.9, 0.3, 0.7, 0.95])

auroc = auroc_score(y_true, y_pred_proba)
far = far_at_recall(y_true, y_pred_proba, recall_threshold=0.95)
cost = cost_aware_score(y_true, y_pred_proba, fn_cost=100, fp_cost=5)

print(f"AUROC: {auroc:.4f}")
print(f"FAR@Recall=0.95: {far:.4f}")
print(f"Cost-aware score: {cost:.4f}")
```

### Recipe 3: Make Predictions

```python
from src.models.pressfuse import PressFuse
import torch

model = PressFuse()
model.load_state_dict(torch.load("outputs/model.ckpt")["state_dict"])
model.eval()

# Prepare input
ts = torch.randn(1, 192, 12)  # (batch, time, features)
cat = torch.randint(0, 5, (1, 4))  # (batch, categories)

# Predict
with torch.no_grad():
    outputs = model(ts, cat)
    defect_prob = outputs["defect_prob"].item()
    print(f"Defect probability: {defect_prob:.2%}")
```

---

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError: No module named 'src'`:
- Make sure you're running from project root
- Check `__init__.py` files exist in `src/` subdirectories

### Type Checking

Run mypy for type errors:
```bash
mypy src/ --ignore-missing-imports
```

### Function Not Found

- Check the module name matches import: `from src.eval.metrics import cost_aware_score`
- Verify function signature in docstring before calling

---

**Last Updated**: May 2026  
**Version**: 0.2.0-beta

