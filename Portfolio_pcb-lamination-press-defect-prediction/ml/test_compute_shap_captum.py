import json, sys, os
from src.explain.shap_wrapper import compute_and_save_shap
from src.data.dataset import SyntheticPressDataset, collate_fn
from src.models.pressfuse import CrossModalAttentionConfig, PressFuseModel
import torch

checkpoint = 'outputs/experiments/exp06/model_mvp.ckpt'
output_dir = 'outputs/experiments/exp06'

# prepare tiny dataset and model load similar to predict_and_explain
try:
    dataset = SyntheticPressDataset(n_cycles=128, n_points=64, anomaly_prob=0.15, seed=42)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state = torch.load(checkpoint, map_location=device)
    inferred_d_model = None
    for key in ['ts_encoder.0.weight','event_encoder.0.weight']:
        if key in state:
            inferred_d_model = int(state[key].shape[0]); break
    if inferred_d_model is None:
        inferred_d_model = CrossModalAttentionConfig().d_model
    cfg = CrossModalAttentionConfig(ts_input_dim=dataset[0][0].shape[1], d_model=inferred_d_model)
    model = PressFuseModel(config=cfg)
    model.load_state_dict(state)
    model.to(device)
    print('Model loaded, device=', device)
    # call compute_and_save_shap using captum
    res = compute_and_save_shap(model, loader, output_dir, method='captum', subsample=16, baseline='mean', device=str(device))
    print('SHAP result:', res)
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)

