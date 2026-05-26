# -*- coding: utf-8 -*-
"""
PressXAI MVP Streamlit Dashboard
PCB 적층 공정 불량 예측 및 분석 대시보드

기능:
- Data 탭: 데모 데이터 생성 및 업로드
- Train 탭: 모델 학습
- Predict 탭: 예측 결과 및 메트릭
- Explain 탭: SHAP 및 Attention 시각화
- Causal 탭: 인과관계 분석
- Report 탭: 종합 리포트 다운로드

실행: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import tempfile
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

# Try to import local dataset generator
try:
    from src.data.synthpress import generate_press_cycle_multi
    from src.data.dataset import SyntheticPressDataset
except Exception:
    generate_press_cycle_multi = None
    SyntheticPressDataset = None

# Paths
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(ROOT, "outputs", "sample_run")
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEMO_PATH = os.path.join(OUTPUT_DIR, "demo_data.csv")
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics.json")
PRED_PATH = os.path.join(OUTPUT_DIR, "predictions.csv")
HIST_PATH = os.path.join(OUTPUT_DIR, "train_history.csv")
ATTN_NPY = os.path.join(OUTPUT_DIR, "attention.npy")
DAG_JSON = os.path.join(OUTPUT_DIR, "dag.json")
SHAP_NPY = os.path.join(OUTPUT_DIR, "shap_values.npy")

st.set_page_config(page_title="PressXAI MVP", layout="wide")

# Sidebar
st.sidebar.title("PressXAI MVP")
mode = st.sidebar.selectbox("Mode", ["Demo", "Upload Data"])
run_actual_training = st.sidebar.checkbox("Run actual training (ml/train_mvp.py)", value=False)
n_cycles = st.sidebar.number_input("# cycles (demo)", value=128, min_value=8, max_value=2048, step=8)

# Tabs (add Overview tab to show data request & project progress)
tabs = st.tabs(["Overview", "Data", "Train", "Predict", "Explain", "Causal", "Report"])

# Utility functions

def generate_demo_csv(out_path: str, n_cycles: int = 128, n_points: int = 192, anomaly_prob: float = 0.15):
    """Generate demo CSV by concatenating cycles from synthpress generator."""
    if generate_press_cycle_multi is None:
        raise RuntimeError("synthpress not available in environment")

    rows = []
    for i in range(n_cycles):
        df, label, atype = generate_press_cycle_multi(i, 1000 + i % 10, n_points=n_points, anomaly_prob=anomaly_prob, seed=i)
        # add cycle index
        df_copy = df.copy()
        df_copy["cycle_row"] = np.arange(len(df_copy))
        rows.append(df_copy)
    big = pd.concat(rows, ignore_index=True)
    big.to_csv(out_path, index=False)
    return out_path


def load_demo_dataframe(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        return None


def read_metrics(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def read_text_file(path: str) -> str:
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except Exception:
        return ""


def extract_markdown_section(md_text: str, header: str, max_lines: int = 14) -> str:
    """Extract a short preview block under a markdown header."""
    if not md_text:
        return ""
    lines = md_text.splitlines()
    start_idx = -1
    for i, line in enumerate(lines):
        if line.strip().lower().startswith(header.lower()):
            start_idx = i
            break
    if start_idx < 0:
        return ""

    block = []
    for line in lines[start_idx + 1 :]:
        if line.startswith("## "):
            break
        if line.strip():
            block.append(line)
        if len(block) >= max_lines:
            break
    return "\n".join(block).strip()


# Overview Tab
with tabs[0]:
    st.header("Overview & Project Status")
    left, right = st.columns([2, 1])
    with left:
        framework_path = os.path.join(ROOT, "docs", "DATA_ANALYSIS_FRAMEWORK.md")
        framework_md = read_text_file(framework_path)

        st.subheader("Auto summary from DATA_ANALYSIS_FRAMEWORK.md")
        if framework_md:
            sec_req = extract_markdown_section(framework_md, "## Section 1", max_lines=8)
            sec_plan = extract_markdown_section(framework_md, "## Section 2", max_lines=8)
            sec_status = extract_markdown_section(framework_md, "## Section 5", max_lines=8)

            col_a, col_b = st.columns(2)
            with col_a:
                st.caption("Section 1 preview")
                st.markdown(sec_req if sec_req else "- Not found")
                st.caption("Section 2 preview")
                st.markdown(sec_plan if sec_plan else "- Not found")
            with col_b:
                st.caption("Section 5 preview")
                st.markdown(sec_status if sec_status else "- Not found")
                with st.expander("Open full framework markdown"):
                    st.markdown(framework_md)
        else:
            st.info("`docs/DATA_ANALYSIS_FRAMEWORK.md` not found yet.")

        st.markdown("---")
        st.subheader("Customer data request (requested items)")
        st.markdown("""
1. Daily/Weekly/Monthly quality status (defect counts, PPM, by shift)
2. Equipment run/stop history and per-process availability (OEE inputs)
3. PRESS alarm history and recipe/parameter change logs
4. LOT / PANEL / CYCLE mapping keys (traceability)
5. Maintenance / calibration history and operator/shift info
6. Process parameter time-series (temperature, pressure, time, flow, humidity)
7. Column definitions, data timestamps and sampling rates
""")
        st.markdown("**Please provide column definitions and sampling rates when sending data.**")
        st.markdown("---")
        st.subheader("What we'll deliver after data reception")
        st.markdown("- Data validation & synchronization (master timeseries)")
        st.markdown("- EDA and root-cause analysis (PCMCI / NOTEARS + SHAP)")
        st.markdown("- Anomaly detection (LSTM AE / IsolationForest / TFT)")
        st.markdown("- GNN-based defect propagation model (MS-CDPNet)")
        st.markdown("- Process optimization suggestions and auto-reporting pipeline")
    with right:
        st.subheader("Project progress")
        # compute milestone progress dynamically (heuristic)
        def file_exists(relpath: str) -> bool:
            return os.path.exists(os.path.join(ROOT, relpath))

        milestones = {}
        # POC & repo setup: README, pyproject, src/
        poc = 100 if (file_exists('README.md') and file_exists('pyproject.toml') and os.path.isdir(os.path.join(ROOT, 'src'))) else 20
        milestones['POC & Repo Setup'] = poc

        # Literature review: presence of docs/literature files
        lit_dir = os.path.join(ROOT, 'docs', 'literature')
        lit_prog = 100 if os.path.isdir(lit_dir) and len(os.listdir(lit_dir)) > 0 else 0
        milestones['Literature Review (STAGE 1-5)'] = lit_prog

        # Data request sent: client request doc exists
        dr = 100 if file_exists(os.path.join('docs', 'CLIENT_DATA_REQUEST.md')) else 0
        milestones['Data Request Sent'] = dr

        # Data validation framework: docs + validation script
        dv = 0
        if file_exists(os.path.join('docs', 'DATA_ANALYSIS_FRAMEWORK.md')):
            dv += 60
        if file_exists(os.path.join('scripts', 'validate_customer_data.py')):
            dv += 30
        milestones['Data Validation Framework'] = min(dv, 100)

        # Causal discovery: mention in docs or runner script
        cd = 0
        if file_exists(os.path.join('docs', 'DATA_ANALYSIS_FRAMEWORK.md')):
            try:
                with open(os.path.join(ROOT, 'docs', 'DATA_ANALYSIS_FRAMEWORK.md'), 'r', encoding='utf-8') as _fh:
                    txt = _fh.read()
                if 'PCMCI' in txt or 'NOTEARS' in txt:
                    cd = max(cd, 10)
            except Exception:
                pass
        if file_exists(os.path.join('ml', 'run_pcmci_discovery.py')):
            cd = max(cd, 40)
        milestones['Causal Discovery (PCMCI/NOTEARS)'] = cd

        # GNN model: check core model file
        gnn = 30 if file_exists(os.path.join('src', 'models', 'pressfuse.py')) else 0
        milestones['GNN Model (MS-CDPNet)'] = gnn

        # XAI: check explain wrappers
        xai = 0
        if file_exists(os.path.join('src', 'explain', 'shap_wrapper.py')) or file_exists(os.path.join('src', 'explain', 'attention_viz.py')):
            xai = 30
        milestones['XAI (SHAP/Attention)'] = xai

        # Auto-report pipeline: docs or existing script
        ar = 0
        if file_exists(os.path.join('docs', 'EXECUTION_CHECKLIST.md')):
            ar += 10
        if file_exists(os.path.join('scripts', 'generate_html_report.py')):
            ar += 20
        milestones['Auto-report Pipeline'] = min(ar, 100)

        for name, pct in milestones.items():
            st.markdown(f"**{name}**")
            st.progress(int(pct))
        # show latest commit info
        try:
            git_info = subprocess.run(["git", "log", "-n", "1", "--pretty=format:%h %s"], capture_output=True, text=True)
            latest_commit = git_info.stdout.strip() if git_info.returncode == 0 else "N/A"
        except Exception:
            latest_commit = "N/A"
        st.markdown("---")
        st.write(f"Latest commit: {latest_commit}")
        st.write("Last update: 2026-05-26")
        st.markdown("---")
        st.subheader("Runtime artifact status")
        validation_report_path = os.path.join(ROOT, 'data', 'customer', 'validation_report.json')
        sync_path = os.path.join(ROOT, 'data', 'customer', 'processed', 'master_synchronized.parquet')
        eda_path = os.path.join(ROOT, 'outputs', 'eda', 'eda_report.json')
        pcmci_path = os.path.join(ROOT, 'outputs', 'pcmci_result.json')

        artifact_checks = [
            ('Validation report', validation_report_path),
            ('Synchronized master data', sync_path),
            ('EDA report', eda_path),
            ('PCMCI result', pcmci_path),
            ('Training metrics', METRICS_PATH),
            ('Predictions', PRED_PATH),
        ]
        for label, path in artifact_checks:
            ok = os.path.exists(path)
            st.write(f"{'OK' if ok else 'MISSING'} - {label}")

        # If metrics exist, expose a tiny KPI strip in Overview
        kpis = read_metrics(METRICS_PATH)
        if kpis:
            st.caption('Latest training KPIs')
            kpi_cols = st.columns(min(4, len(kpis)))
            for i, (k, v) in enumerate(kpis.items()):
                if i >= len(kpi_cols):
                    break
                try:
                    kpi_cols[i].metric(k, f"{float(v):.3f}")
                except Exception:
                    kpi_cols[i].metric(k, str(v))

        st.markdown("---")
        st.subheader("Quick actions")
        run_col_a, run_col_b = st.columns([2, 1])
        with run_col_a:
            if st.button("Run data validation (scripts/validate_customer_data.py)"):
                try:
                    out = subprocess.run(["python", os.path.join("scripts", "validate_customer_data.py")], capture_output=True, text=True, check=False)
                    st.code(out.stdout + ("\nErrors:\n" + out.stderr if out.stderr else ""))
                    vr = os.path.join(ROOT, 'data', 'customer', 'validation_report.json')
                    if os.path.exists(vr):
                        st.success(f"Validation report saved: {vr}")
                        with open(vr, 'rb') as fh:
                            st.download_button("Download validation_report.json", data=fh.read(), file_name='validation_report.json')
                except Exception as e:
                    st.error(f"Validation run failed: {e}")
            if st.button("Run synchronization (scripts/synchronize_customer_data.py)"):
                try:
                    out = subprocess.run(["python", os.path.join("scripts", "synchronize_customer_data.py")], capture_output=True, text=True, check=False)
                    st.code(out.stdout + ("\nErrors:\n" + out.stderr if out.stderr else ""))
                    ms = os.path.join(ROOT, 'data', 'customer', 'processed', 'master_synchronized.parquet')
                    if os.path.exists(ms):
                        st.success(f"Master synchronized data saved: {ms}")
                except Exception as e:
                    st.error(f"Synchronization failed: {e}")
            if st.button("Run EDA (scripts/eda_customer_data.py)"):
                try:
                    out = subprocess.run(["python", os.path.join("scripts", "eda_customer_data.py")], capture_output=True, text=True, check=False)
                    st.code(out.stdout + ("\nErrors:\n" + out.stderr if out.stderr else ""))
                    eda_json = os.path.join(ROOT, 'outputs', 'eda', 'eda_report.json')
                    if os.path.exists(eda_json):
                        st.success(f"EDA report: {eda_json}")
                        with open(eda_json, 'rb') as fh:
                            st.download_button("Download eda_report.json", data=fh.read(), file_name='eda_report.json')
                except Exception as e:
                    st.error(f"EDA failed: {e}")
        with run_col_b:
            if st.button("Refresh progress"):
                st.experimental_rerun()

# Data Tab
with tabs[1]:
    st.header("1. Data")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Generate or Upload")
        if st.button("Generate demo data"):
            try:
                with st.spinner("Generating demo data..."):
                    generate_demo_csv(DEMO_PATH, n_cycles=int(n_cycles))
                st.success(f"Demo data saved to {DEMO_PATH}")
            except Exception as e:
                st.error(f"Demo generation failed: {e}")

        uploaded_file = st.file_uploader("Upload CSV (cycle time-series format)", type=["csv"])
        if uploaded_file is not None:
            # save to outputs folder
            save_to = os.path.join(OUTPUT_DIR, "uploaded_data.csv")
            with open(save_to, "wb") as fh:
                fh.write(uploaded_file.getbuffer())
            st.success(f"Saved uploaded data to {save_to}")

        # Determine active data path
        data_path = None
        if uploaded_file is not None:
            data_path = os.path.join(OUTPUT_DIR, "uploaded_data.csv")
        elif os.path.exists(DEMO_PATH):
            data_path = DEMO_PATH

        if data_path:
            df = load_demo_dataframe(data_path)
            if df is not None:
                st.write(f"Data path: {data_path}")
                st.write(f"Shape: {df.shape}")
                if "label" in df.columns:
                    # label distribution per cycle (we aggregate by cycle_id)
                    try:
                        label_counts = df.groupby("cycle_id")["label"].first().value_counts().to_dict()
                        st.write("Label distribution (by cycle):")
                        st.json(label_counts)
                    except Exception:
                        st.write("No per-cycle labels found")
                st.subheader("Preview (first cycle)")
                try:
                    first_cycle_id = df["cycle_id"].iloc[0]
                    preview = df[df["cycle_id"] == first_cycle_id].head(20)
                    st.dataframe(preview)
                except Exception:
                    st.dataframe(df.head(20))
        else:
            st.info("No data available. Generate demo data or upload a CSV.")

    with col2:
        st.subheader("Quick actions")
        if st.button("Clear outputs/sample_run"):
            if os.path.exists(OUTPUT_DIR):
                try:
                    for f in os.listdir(OUTPUT_DIR):
                        fp = os.path.join(OUTPUT_DIR, f)
                        if os.path.isfile(fp):
                            os.remove(fp)
                    st.success("Cleared outputs/sample_run")
                except Exception as e:
                    st.error(f"Failed to clear: {e}")
            else:
                st.info("No outputs to clear")

# Train Tab
with tabs[2]:
    st.header("2. Train")
    cols = st.columns(3)
    with cols[0]:
        epochs = st.number_input("Epochs", min_value=1, max_value=50, value=3)
        batch_size = st.number_input("Batch size", min_value=1, max_value=128, value=16)
    with cols[1]:
        lr = st.number_input("Learning rate", value=1e-3, format="%.6f")
        d_model = st.number_input("d_model", value=64)
    with cols[2]:
        num_heads = st.number_input("num_heads", min_value=1, max_value=8, value=4)
        seed = st.number_input("Seed", value=42)

    run_col1, run_col2 = st.columns([2, 1])
    with run_col1:
        if st.button("Run Training"):
            # decide whether to call actual training script
            if run_actual_training:
                st.info("Running actual training (this may take time). Check console for logs.")
                cmd = ["python", os.path.join("ml", "train_mvp.py"),
                       "--n-cycles", str(n_cycles),
                       "--n-points", "192",
                       "--batch-size", str(batch_size),
                       "--epochs", str(epochs),
                       "--lr", str(lr),
                       "--d-model", str(d_model),
                       "--num-heads", str(num_heads),
                       "--output-dir", OUTPUT_DIR,
                       "--seed", str(seed)]
                try:
                    with st.spinner("Training (ml/train_mvp.py)... this may take several minutes"):
                        subprocess.run(cmd, check=True)
                    st.success("Training finished (ml/train_mvp.py)")
                except subprocess.CalledProcessError as e:
                    st.error(f"Training failed: {e}")
            else:
                st.info("Running dummy training (fast) and creating placeholder outputs")
                # dummy training: simulate progress and write placeholder outputs
                progress = st.progress(0)
                history = []
                for e in range(1, epochs + 1):
                    progress.progress(int(100 * e / epochs))
                    # fake metrics
                    train_loss = float(max(0.01, 1.0 / (e + 1)))
                    val_loss = float(max(0.01, 1.2 / (e + 1)))
                    val_acc = float(min(0.99, 0.5 + 0.1 * e))
                    val_precision = float(min(0.99, 0.5 + 0.08 * e))
                    val_recall = float(min(0.99, 0.4 + 0.09 * e))
                    val_f1 = float(max(0.0, 2 * (val_precision * val_recall) / max(1e-6, (val_precision + val_recall))))
                    val_auc = float(min(0.99, 0.6 + 0.08 * e))
                    history.append({
                        "epoch": e,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "val_precision": val_precision,
                        "val_recall": val_recall,
                        "val_f1": val_f1,
                        "val_auc": val_auc,
                    })
                # save history and fake metrics/preds
                import json
                import csv

                os.makedirs(OUTPUT_DIR, exist_ok=True)
                with open(METRICS_PATH, "w", encoding="utf-8") as fh:
                    fake_metrics = history[-1]
                    json.dump({
                        "accuracy": fake_metrics["val_acc"],
                        "precision": fake_metrics["val_precision"],
                        "recall": fake_metrics["val_recall"],
                        "f1": fake_metrics["val_f1"],
                        "roc_auc": fake_metrics["val_auc"],
                    }, fh, indent=2)
                # fake predictions: if demo data exists load cycle ids
                preds = []
                if os.path.exists(DEMO_PATH):
                    df_all = pd.read_csv(DEMO_PATH)
                    cycle_ids = df_all["cycle_id"].unique().tolist()
                    for cid in cycle_ids:
                        # random prob biased by label if available
                        lab = df_all[df_all["cycle_id"] == cid]["label"].iloc[0] if "label" in df_all.columns else 0
                        prob = float(min(0.99, max(0.01, np.random.rand() * 0.3 + lab * 0.6)))
                        preds.append((int(cid), int(lab), prob, int(prob >= 0.5)))
                else:
                    for i in range(min(128, n_cycles)):
                        p = float(np.random.rand())
                        preds.append((i, 0, p, int(p >= 0.5)))
                with open(PRED_PATH, "w", newline="", encoding="utf-8") as fh:
                    writer = csv.writer(fh)
                    writer.writerow(["cycle_id", "label", "pred_proba", "pred_label"])
                    for row in preds:
                        writer.writerow(row)
                # save history csv
                import pandas as pd

                pd.DataFrame(history).to_csv(HIST_PATH, index=False)
                st.success("Dummy training completed: outputs written to outputs/sample_run/")

    with run_col2:
        st.write("Training artifacts")
        if os.path.exists(METRICS_PATH):
            st.write("metrics.json present")
        if os.path.exists(PRED_PATH):
            st.write("predictions.csv present")
        if os.path.exists(HIST_PATH):
            st.write("train_history.csv present")

    # show training history if exists
    if os.path.exists(HIST_PATH):
        try:
            hist = pd.read_csv(HIST_PATH)
            st.subheader("Training History")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['train_loss'], mode='lines+markers', name='train_loss'))
            fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['val_loss'], mode='lines+markers', name='val_loss'))
            fig.update_layout(title='Loss over epochs', xaxis_title='epoch')
            st.plotly_chart(fig, use_container_width=True)
            # metrics
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=hist['epoch'], y=hist['val_auc'], mode='lines+markers', name='val_auc'))
            fig2.update_layout(title='Validation AUC', xaxis_title='epoch')
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to plot history: {e}")

# Predict Tab
with tabs[3]:
    st.header("3. Predict")
    if st.button("Load predictions from outputs/sample_run"):
        if os.path.exists(PRED_PATH):
            preds = pd.read_csv(PRED_PATH)
            st.subheader("Predictions")
            st.dataframe(preds.head(200))
            # compute metrics if label present
            if set(["label", "pred_proba"]) <= set(preds.columns):
                ytrue = preds['label'].tolist()
                yprob = preds['pred_proba'].tolist()
                ypred = [1 if p >= 0.5 else 0 for p in yprob]
                try:
                    acc = accuracy_score(ytrue, ypred)
                    prec = precision_score(ytrue, ypred, zero_division=0)
                    rec = recall_score(ytrue, ypred, zero_division=0)
                    f1 = f1_score(ytrue, ypred, zero_division=0)
                    auc = roc_auc_score(ytrue, yprob) if len(set(ytrue)) > 1 else 0.0
                    st.metric("Accuracy", f"{acc:.3f}")
                    st.metric("Precision", f"{prec:.3f}")
                    st.metric("Recall", f"{rec:.3f}")
                    st.metric("F1", f"{f1:.3f}")
                    st.metric("ROC AUC", f"{auc:.3f}")
                    # confusion matrix
                    cm = confusion_matrix(ytrue, ypred)
                    fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues')
                    fig.update_layout(title='Confusion Matrix', xaxis_title='Predicted', yaxis_title='True')
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to compute metrics: {e}")
        else:
            st.info("No predictions found. Run training or generate dummy outputs in Train tab.")

# Explain Tab
with tabs[4]:
    st.header("4. Explain (SHAP / Attention)")
    st.write("SHAP and attention visualizations. If real files exist in outputs/sample_run they will be used; otherwise placeholders are shown.")

    # Load SHAP: either numpy array or summary CSV
    shap_arr = None
    shap_summary = None
    if os.path.exists(SHAP_NPY):
        try:
            shap_arr = np.load(SHAP_NPY)
            st.success("Loaded SHAP values from shap_values.npy")
        except Exception as e:
            st.error(f"Failed to load shap npy: {e}")
    elif os.path.exists(os.path.join(OUTPUT_DIR, "shap_summary.csv")):
        try:
            shap_summary = pd.read_csv(os.path.join(OUTPUT_DIR, "shap_summary.csv"))
            st.success("Loaded SHAP summary CSV")
        except Exception as e:
            st.error(f"Failed to load shap_summary.csv: {e}")
    else:
        st.info("No SHAP files found (shap_values.npy or shap_summary.csv)")

    # Load attention if present
    attn = None
    if os.path.exists(ATTN_NPY):
        try:
            attn = np.load(ATTN_NPY)
            st.success("Loaded attention weights (attention.npy)")
        except Exception as e:
            st.error(f"Failed to load attention.npy: {e}")
    else:
        st.info("No attention file found (attention.npy)")

    col1, col2 = st.columns([2, 3])
    # Feature importance bar chart and top-k table
    with col1:
        st.subheader("Feature importance (SHAP)")
        if shap_summary is not None:
            # Expect columns: feature, importance
            try:
                df_feat = shap_summary.sort_values("importance", ascending=False)
            except Exception:
                df_feat = shap_summary
        elif shap_arr is not None:
            # shap_arr could be (N,T,D) or (T,D) or (D,)
            arr = shap_arr
            if arr.ndim == 3:
                arr_t_d = arr.mean(axis=0)
            elif arr.ndim == 2:
                arr_t_d = arr
            elif arr.ndim == 1:
                fi = np.abs(arr)
                features = [f"f{i}" for i in range(1, fi.size + 1)]
                df_feat = pd.DataFrame({"feature": features, "importance": fi})
                arr_t_d = None
            else:
                arr_t_d = None

            if shap_arr is not None and 'df_feat' not in locals():
                if arr_t_d is not None:
                    fi = np.mean(np.abs(arr_t_d), axis=0)  # mean over time -> (D,)
                    features = [f"f{i}" for i in range(1, fi.size + 1)]
                    df_feat = pd.DataFrame({"feature": features, "importance": fi})
        else:
            # placeholder
            features = [f"f{i}" for i in range(1, 20)]
            vals = np.abs(np.random.randn(len(features)))
            df_feat = pd.DataFrame({"feature": features, "importance": vals})

        # show bar chart and top-k table
        try:
            fig = px.bar(df_feat.sort_values('importance', ascending=False).head(30), x='importance', y='feature', orientation='h')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("**Top-k important features**")
            st.dataframe(df_feat.sort_values('importance', ascending=False).head(10).reset_index(drop=True))
        except Exception as e:
            st.error(f"Failed to render feature importance: {e}")

    # Time-feature heatmap and attention
    with col2:
        st.subheader("Time × Feature heatmap (SHAP)")
        if shap_arr is not None and shap_arr.ndim >= 2:
            try:
                if shap_arr.ndim == 3:
                    arr_t_d = shap_arr.mean(axis=0)  # (T,D)
                elif shap_arr.ndim == 2:
                    arr_t_d = shap_arr
                else:
                    arr_t_d = None

                if arr_t_d is not None:
                    # transpose to show features on y axis
                    fig_h = px.imshow(np.abs(arr_t_d).T, labels=dict(x="time_step", y="feature_index", color="|SHAP|"))
                    fig_h.update_layout(height=400)
                    st.plotly_chart(fig_h, use_container_width=True)
                else:
                    st.info("SHAP array has unexpected shape; cannot render heatmap")
            except Exception as e:
                st.error(f"Failed to render SHAP heatmap: {e}")
        else:
            st.info("No SHAP time-feature data; showing placeholder")
            mat = np.abs(np.random.randn(32, 19))
            fig_h = px.imshow(mat, labels=dict(x="time_step", y="feature_index"))
            st.plotly_chart(fig_h, use_container_width=True)

        st.subheader("Attention heatmap (average)")
        if attn is not None:
            try:
                a = attn
                if a.ndim == 4:
                    # (B, H, T, T) -> take first sample
                    a = a[0]
                if a.ndim == 3:
                    avg = a.mean(axis=0)
                    fig2 = px.imshow(avg, color_continuous_scale='Viridis')
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("Unexpected attention shape")
            except Exception as e:
                st.error(f"Failed to render attention heatmap: {e}")
        else:
            st.info("No attention file; placeholder below")
            mat = np.random.rand(32, 32)
            fig2 = px.imshow(mat, color_continuous_scale='Viridis')
            st.plotly_chart(fig2, use_container_width=True)

# Causal Tab
with tabs[5]:
    st.header("5. Causal")
    st.write("Load adjacency/edge-scores CSV files to visualize causal adjacency and top edges.")

    adj_path = os.path.join(OUTPUT_DIR, "adjacency.csv")
    edges_path = os.path.join(OUTPUT_DIR, "edge_scores.csv")

    if os.path.exists(adj_path):
        try:
            adj_df = pd.read_csv(adj_path, index_col=0)
            st.subheader("Adjacency matrix")
            fig = px.imshow(adj_df.values, x=adj_df.columns.tolist(), y=adj_df.index.tolist(), color_continuous_scale='RdBu_r')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to load adjacency.csv: {e}")
    else:
        st.info("No adjacency.csv found; showing dummy adjacency")
        D = 19
        mat = np.random.rand(D, D)
        np.fill_diagonal(mat, 0.0)
        fig = px.imshow(mat, color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)

    # Edge table and top-k causal edges
    st.subheader("Edge table / Top-k causal edges")
    if os.path.exists(edges_path):
        try:
            edges_df = pd.read_csv(edges_path)
            # Expect columns: source,target,weight (or score)
            st.dataframe(edges_df.head(200))
            if 'weight' in edges_df.columns:
                sort_col = 'weight'
            elif 'score' in edges_df.columns:
                sort_col = 'score'
            else:
                sort_col = edges_df.columns[-1]
            topk = st.number_input("Top-k edges", min_value=1, max_value=50, value=10)
            top_edges = edges_df.sort_values(sort_col, ascending=False).head(int(topk))
            st.markdown("**Top causal edges**")
            st.dataframe(top_edges.reset_index(drop=True))
        except Exception as e:
            st.error(f"Failed to load edge_scores.csv: {e}")
    else:
        st.info("No edge_scores.csv found; showing dummy edges")
        cols = ["source", "target", "weight"]
        rows = [(f"f{i}", f"f{j}", float(np.random.rand())) for i in range(1, 10) for j in range(1, 10) if i != j]
        df_edge = pd.DataFrame(rows, columns=cols).sort_values('weight', ascending=False).head(20)
        st.dataframe(df_edge)

    # Optional: network graph visualization
    if st.checkbox("Show network graph (interactive)"):
        try:
            import networkx as nx

            if os.path.exists(edges_path):
                G = nx.DiGraph()
                for _, r in edges_df.iterrows():
                    G.add_edge(str(r['source']), str(r['target']), weight=float(r.get('weight', 1.0)))
            else:
                G = nx.DiGraph()
                for _, r in df_edge.iterrows():
                    G.add_edge(r['source'], r['target'], weight=r['weight'])

            pos = nx.spring_layout(G, seed=42)
            edge_x = []
            edge_y = []
            for u, v in G.edges():
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]
            edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#888'), hoverinfo='none', mode='lines')
            node_x = []
            node_y = []
            node_text = []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(str(node))
            node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=node_text)
            fig = go.Figure(data=[edge_trace, node_trace])
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to render network graph: {e}")

# Report Tab
with tabs[6]:
    st.header("6. Report")
    st.write("Summary of key metrics, params and predictions. Download combined report JSON.")

    metrics = read_metrics(METRICS_PATH)
    params = None
    params_path = os.path.join(OUTPUT_DIR, "params.json")
    if os.path.exists(params_path):
        try:
            with open(params_path, 'r', encoding='utf-8') as fh:
                params = json.load(fh)
        except Exception as e:
            st.error(f"Failed to read params.json: {e}")

    preds_df = None
    if os.path.exists(PRED_PATH):
        try:
            preds_df = pd.read_csv(PRED_PATH)
        except Exception as e:
            st.error(f"Failed to read predictions.csv: {e}")

    # display metrics
    if metrics:
        st.subheader("Key Metrics")
        cols = st.columns(len(metrics))
        for (k, v), c in zip(metrics.items(), cols):
            try:
                c.metric(k, f"{v:.3f}")
            except Exception:
                c.metric(k, str(v))
    else:
        st.info("No metrics.json found")

    # display params
    if params:
        st.subheader("Parameters (params.json)")
        st.json(params)
    else:
        st.info("No params.json found")

    # display predictions summary
    if preds_df is not None:
        st.subheader("Predictions sample")
        st.dataframe(preds_df.head(200))
        st.markdown(f"Total predictions: {len(preds_df)}")
    else:
        st.info("No predictions.csv found")

    # Combined report download
    report = {"metrics": metrics or {}, "params": params or {}, "predictions_sample": preds_df.head(20).to_dict(orient='records') if preds_df is not None else []}
    report_json = json.dumps(report, indent=2)
    st.download_button("Download combined report (JSON)", data=report_json, file_name="report_summary.json", mime="application/json")

    st.markdown("---")
    st.write("Design notes: Replace metrics.json, params.json, predictions.csv with experiment outputs to generate the report. Later this JSON can be rendered to PDF/HTML for formal reports.")
