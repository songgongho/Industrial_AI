# System Architecture

High-level overview of the MS-CDPNet system design.

## 1. Data Flow

```
Raw Press Data (MDB/CSV)
  ↓
[Data Loader] (src/data/loaders.py)
  ∟ Preprocessing: scaling, imputation, padding
  ↓
Time-Series Tensor (B, T, D)
  │
  ├─→ [Categorical Encoder] → (B, C)
  │
  ├─→ [Time-Series Encoder] → Features
  │
  └─→ [Attention Fusion] → Multimodal Representation
  ↓
[Task Head 1] → Binary Defect (0/1)
[Task Head 2] → Defect Type (P019 multi-class)
[Task Head 3] → Anomaly Confidence
  ↓
Predictions + Attention Weights
  ↓
[Evaluation] (src/eval/metrics.py)
  ∟ Cost-aware score, FAR@Recall, AUROC
  ↓
[Explanation] (src/explain/)
  ∟ Attention visualization
  ∟ SHAP gradient integration
```

## 2. Module Hierarchy

```
src/
├── data/
│   ├── schema.py          ← Domain definitions (P013, P019)
│   ├── synthpress.py      ← Synthetic data generator
│   ├── loaders.py         ← PyTorch Dataset + DataLoader
│   ├── preprocess.py      ← Feature engineering
│   ├── audit.py           ← Data validation
│   └── dataset_inspector.py
│
├── models/
│   ├── pressfuse.py       ← Main multimodal model
│   ├── heads.py           ← Task-specific prediction heads
│   └── baselines/
│       ├── secom.py       ← SECOM sklearn baseline
│       └── deep_pcb.py    ← DeepPCB baseline
│
├── training/
│   ├── module.py          ← PyTorch Lightning wrapper
│   └── callbacks.py       ← Metrics, logging callbacks
│
├── eval/
│   └── metrics.py         ← Cost-aware evaluation
│
├── explain/
│   ├── attention_viz.py  ← Attention map visualization
│   └── shap_grad.py      ← SHAP gradient computation
│
└── utils/
    ├── config.py          ← YAML config loading
    ├── logging.py         ← Logging setup
    └── paths.py           ← Cross-platform path utilities
```

## 3. PressFuse Model Architecture

```
                Input (multimodal)
                      │
        ┌─────────────┼─────────────┐
        │             │             │
   Time-series    Categorical    (Optional)
   [B, T, 12]     [B, 4]        Image [B, 3, 256, 256]
        │             │             │
   Embed →      Embed →         ViT →
   [B, T, 64]   [B, 64]       [B, 49, 64]
        │             │             │
        └─────────────┼─────────────┘
                      │
            Cross-Modal Attention
           (Transformer encoder)
                      │
            [B, T, 64] fusion output
                      │
        ┌─────────────┼─────────────┐
        │             │             │
    Binary Head   Multitype Head  Anomaly Head
    (sigmoid)     (softmax)       (sigmoid)
        │             │             │
    [B, 1]      [B, 37]        [B, 1]
  (0 to 1)     (logits)       (0 to 1)
```

### Key Components

#### 1. Time-Series Encoder
- Conv1D blocks with residual connections
- ∟ Captures local temporal patterns
- Output: [B, T, 64] hidden state

#### 2. Categorical Encoder
- Embedding layer + Dense FC
- ∟ Encodes discrete variables (equipment, line, etc.)
- Output: [B, 64]

#### 3. Cross-Modal Attention
- Multi-head self-attention
- ∟ Learns which time steps/features are important
- Weights shape: [num_heads, T, T]
- ∟ Can be visualized for explainability

#### 4. Task Heads
- Binary defect classifier (P013 presence)
- Multi-class defect type (P019, 37 categories)
- Anomaly confidence (early warning)

## 4. Training Pipeline

```
┌─────────────────────────────┐
│  Data Loading               │
│  (SyntheticData/RealData)   │
└──────────────┬──────────────┘
               │
               ↓
┌─────────────────────────────┐
│  Train/Val/Test Split       │
│  (Group-aware, stratified)  │
└──────────────┬──────────────┘
               │
               ↓
┌─────────────────────────────┐
│  Lightning Trainer          │
│  ├─ Forward pass             │
│  ├─ Loss computation         │
│  ├─ Backward pass            │
│  └─ Optimizer step           │
└──────────────┬──────────────┘
               │
               ↓
┌─────────────────────────────┐
│  Validation per Epoch       │
│  ├─ Metrics (AUROC, FAR)    │
│  ├─ MLflow logging          │
│  └─ Checkpoint saving       │
└──────────────┬──────────────┘
               │
               ↓
         [Trained Model]
```

### Loss Function

```
Total Loss = α * Binary_Loss + β * Multitype_Loss + γ * Anomaly_Loss

Binary_Loss = WeightedBCE(y_true, y_pred_defect)
  ∟ Weight = fn_cost / (fn_cost + fp_cost) if y_true=1, else (fp_cost / ...)

Multitype_Loss = CrossEntropyLoss(y_true_type, logits_type)

Anomaly_Loss = BCELoss(y_true_anomaly, y_pred_anomaly)

Weights (α, β, γ) are configurable via Hydra configs
```

## 5. Evaluation Strategy

### Metrics Used

1. **AUROC** (Area Under ROC Curve)
   - Threshold-independent
   - Good for imbalanced data (0.03% defect rate)
   - Target: ≥ 0.98

2. **FAR@Recall=0.95**
   - False Alarm Rate at 95% recall
   - Operators want to catch 95% of defects
   - Acceptable FAR: < 5%
   - Formula:
     ```
     FAR = FP / (TN + FP)  where threshold = percentile_95(y_pred_proba)
     ```

3. **Cost-Aware Score**
   - Incorporates business costs (FN weight >> FP weight)
   - FN cost: 100 (warranty loss if defect goes undetected)
   - FP cost: 5 (production line stoppage cost)
   - Formula:
     ```
     Cost = FN_count * 100 + FP_count * 5
     Score = 1 - (Cost / max_possible_cost)
     Range: [0, 1], higher is better
     ```

### Validation Flow

```
┌────────────┐
│ Validation │
│ Dataset    │
└─────┬──────┘
      │
      ↓
 Model Inference
 ├─ y_pred_proba = model(x)
 └─ attention_weights
      │
      ↓
┌────────────────────────┐
│ Compute Metrics        │
│ ├─ AUROC (sklearn)     │
│ ├─ FAR@Recall (custom) │
│ ├─ Cost-Aware (custom) │
│ └─ Attention summary   │
└────────────────────────┘
      │
      ↓
MLflow Log → Model Registry
```

## 6. Explainability

### Method 1: Attention Visualization

```
Attention Weights [num_heads, T, T]
  ↓
Average over heads: [T, T]
  ↓
Heatmap visualization
  ├─ X-axis: Time step (input)
  ├─ Y-axis: Time step (query)
  └─ Color: Attention weight [0, 1]
  
→ Shows which time periods are important for prediction
```

### Method 2: SHAP Gradients

```
Input Features [B, T, D]
        │
        ↓
  Model Forward
        │
        ↓
  Backprop to input
        │
        ↓
  Gradient magnitude = feature importance
        │
        ↓
  SHAP values
        │
        ↓
  Force plot showing which features push toward defect
```

### Method 3: Saliency Maps (Temporal)

```
Feature importance per time step
[T,] importance vector
  │
  ├─ T=0 (vacuum phase): low
  ├─ T=50 (pressure ramp): high ← typically where issues show
  ├─ T=100-150 (hot press): highest ← critical zone
  └─ T=192 (cooling): medium
```

## 7. Inference Pipeline

```
New Press Cycle
      │
      ↓
[Preprocess]
├─ Normalize features
├─ Pad/truncate to T=192
└─ Embed categoricals
      │
      ↓
   Model
   [PressFuse]
    Forward(x)
      │
      ↓
Predictions
├─ defect_prob: [0, 1]
├─ defect_type: logits [0, 37]
├─ anomaly_conf: [0, 1]
└─ attention_weights: [num_heads, T, T]
      │
      ↓
[Decision Logic]
if defect_prob > threshold:
  ├─ Log alert
  ├─ Explain (show attention + SHAP)
  └─ (Optional) Trigger control action
      │
      ↓
Output (JSON/API)
{
  "cycle_id": 42,
  "defect_probability": 0.87,
  "anomaly_type": "pressure_drop",
  "confidence": 0.92,
  "explanation": {...}
}
```

## 8. Future Extensions: Causal DAG

*Planned for future versions*

```
Causal Graph Discovery (PCMCI / NOTEARS)

Variables: HPPRESSPV, PT1, VACUUM, etc.

Learned DAG:
  HPPRESSPV → VACUUM (pressure affects vacuum)
  VACUUM → PT1 (vacuum affects cooling)
  HPTEMPSV → PT1 (setpoint affects plate temp)
  {PRESSURE, VACUUM} → DEFECT (causality bottleneck)
  DEFECT → P019_VOID (defect → downstream yield)

Uses: Root cause analysis, intervention recommendations
```

## 9. Codebase Conventions

### Naming
- Models: CamelCase (PressFuse, BinaryDefectHead)
- Functions: snake_case (cost_aware_score, generate_press_cycle)
- Constants: UPPER_SNAKE_CASE (PRESSURE_MAX=99)
- Private: Leading underscore (_build_dataset)

### Type Hints (Python 3.11)
```python
def func(x: int | str, y: list[float] | None = None) -> dict[str, Any]:
    """All functions must have type hints on params and return."""
    pass
```

### Docstrings (Google Style)
```python
def example(param1: int) -> bool:
    """One-line summary.
    
    Longer explanation if needed across multiple lines,
    explaining the method's purpose and behavior.
    
    Args:
        param1: Description of param1
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When X happens
        
    Example:
        >>> example(5)
        True
    """
```

## 10. Performance Characteristics

### Training
- **GPU Memory**: ~6-8 GB for batch_size=32, model_dim=64
- **Speed**: ~300 ms per batch on RTX 3090
- **Convergence**: ~10 epochs for synthetic data (< 5 minutes on GPU)

### Inference
- **Latency**: ~50 ms per cycle (CPU)
- **Latency**: ~10 ms per cycle (GPU batch processing)
- **Throughput**: 100-1000 cycles/second depending on hardware

### Data
- **Synthetic data generation**: ~100 cycles/second
- **Data loading**: ~1000 samples/second from disk
- **Feature extraction**: negligible (< 1 ms per sample)

---

**Last Updated**: May 2026  
**Version**: 0.2.0-beta

