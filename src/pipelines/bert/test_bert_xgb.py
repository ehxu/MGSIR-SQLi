# src/pipelines/bert/test_bert_xgb.py
import sys
import os
import time
import joblib
import argparse
import numpy as np
import xgboost as xgb
import torch
import io
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# === 环境配置 ===
current_file = Path(__file__).resolve()
project_root = current_file.parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config.paths import get_dataset_paths, get_pipeline_paths, append_global_result
from src.utils.data_utils import load_test_csv
from src.utils.math_ops import sigmoid
from src.utils.logger import setup_logger
from src.visualization.metric_plots import (
    plot_metrics_single,
    compute_plot_auc,
    save_confusion_matrix,
)
from src.utils.runtime_env import configure_xgboost_booster_threads
from src.features.bert_extractor import (
    preprocess_text,
    load_bert_model,
    bert_encode_batch,
)

def get_pytorch_model_size_mb(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_bytes = buffer.getbuffer().nbytes
    return size_bytes / (1024 * 1024)

def load_bert_artifacts(pipeline_name, logger=None):
    paths = get_pipeline_paths(pipeline_name)
    model_path = paths.model_file

    if not model_path.exists():
        raise FileNotFoundError(f"XGB Model not found: {model_path}")

    # 加载 XGBoost (~0.17 MB)
    xgb_model = joblib.load(model_path)
    xgb_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    # BERT 总是实时加载预训练权重
    tokenizer, bert_model, device = load_bert_model()

    # 计算 BERT 大小 (~420 MB)
    bert_size_mb = get_pytorch_model_size_mb(bert_model)

    # 4. 计算总大小
    model_size_mb = xgb_size_mb + bert_size_mb

    if logger:
        logger.info(f"[Artifacts] Model loaded: {model_size_mb:.2f} MB")
        logger.info(f"[Artifacts] BERT Model loaded on {device}")

    return xgb_model, tokenizer, bert_model, device, model_size_mb


def run_bert_testing(
    dataset_name="dataset1", feature_name="bert_xgb", test_csv_path=None
):
    # 1. Logger
    log_dir = project_root / "results" / "logs"
    suffix = f"_{Path(test_csv_path).stem}" if test_csv_path else ""
    logger = setup_logger(
        f"test_{feature_name}{suffix}",
        log_dir / f"test_{feature_name}_{dataset_name}{suffix}.log",
    )

    logger.info(f"{'='*60}")
    logger.info(f"[Start] BERT Testing: {feature_name} | Dataset: {dataset_name}")
    logger.info(f"{'='*60}")

    # 2. Data
    if test_csv_path:
        df_test = load_test_csv(test_csv_path)
    else:
        paths = get_dataset_paths(dataset_name)
        df_test = load_test_csv(paths.test_csv)

    queries = df_test["Query"].astype(str).tolist()
    labels = df_test["Label"].values

    # 3. Artifacts
    xgb_model, tokenizer, bert_model, device, model_size_mb = load_bert_artifacts(
        feature_name, logger
    )
    paths = get_pipeline_paths(feature_name)

    plots_save_path = str(paths.plots_file)
    if test_csv_path:
        custom_name = Path(test_csv_path).stem
        plot_dir = Path(paths.plots_file).parent
        plots_save_path = str(plot_dir / f"plots_{feature_name}_{custom_name}.png")

    # ==========================================
    # 4. Efficiency Benchmark (Latency & QPS)
    # ==========================================
    logger.info("[Benchmark] Running Efficiency Test...")
    booster = xgb_model.get_booster()
    configure_xgboost_booster_threads(booster)
    latencies = []

    # BERT 比较慢，为了不跑太久，我们只测前 500 条
    bench_queries = queries[:500] if len(queries) > 500 else queries

    # Warm up (exclude from measurement)
    try:
        warm_q = bench_queries[0] if bench_queries else "select * from admin"
        warm_clean = preprocess_text(warm_q)
        vec = bert_encode_batch(
            [warm_clean], tokenizer, bert_model, device, batch_size=1
        )
        _ = booster.predict(xgb.DMatrix(vec))
    except Exception:
        pass

    for q in bench_queries:
        t0 = time.perf_counter()

        # A. Preprocess
        q_clean = preprocess_text(q)

        # B. BERT Encode (Single)
        # 必须模拟单条推理，所以 batch_size=1
        vec = bert_encode_batch([q_clean], tokenizer, bert_model, device, batch_size=1)

        # C. Predict
        d_mat = xgb.DMatrix(vec)
        _ = booster.predict(d_mat)

        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    latency_avg = np.mean(latencies)
    latency_p99 = np.percentile(latencies, 99)
    qps = 1000.0 / latency_avg if latency_avg > 0 else 0

    logger.info(
        f"[Benchmark] Avg: {latency_avg:.4f} ms | P99: {latency_p99:.4f} ms | QPS: {qps:.2f}"
    )

    # ==========================================
    # 5. Full Prediction
    # ==========================================
    logger.info("[Predict] Processing full test set (Batch Mode)...")

    # 批量处理以提高评估速度
    queries_clean = [preprocess_text(q) for q in queries]
    X_test = bert_encode_batch(
        queries_clean, tokenizer, bert_model, device, batch_size=32
    )
    feature_dim = X_test.shape[1]

    dtest = xgb.DMatrix(X_test)
    raw_scores = booster.predict(dtest, output_margin=True)
    proba = sigmoid(raw_scores)
    preds = (proba >= 0.5).astype(int)

    # ==========================================
    # 6. Metrics & WAF Robustness
    # ==========================================
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    tnr = tn / (fp + tn) if (fp + tn) > 0 else 0

    # ROC & WAF Metrics
    auc = 0.0
    rec_at_1, rec_at_01 = -1.0, -1.0

    if len(np.unique(labels)) > 1:
        # Plotting Linear & Log ROC
        auc, fpr_arr, tpr_arr = compute_plot_auc(
            labels,
            proba,
            model_name=f"{feature_name}",
            plot_path=plots_save_path.replace(".png", "_roc.png"),
        )

        # Calculate Recall at Low FPR
        rec_at_1 = np.interp(0.01, fpr_arr, tpr_arr)
        rec_at_01 = np.interp(0.001, fpr_arr, tpr_arr)

    logger.info("-" * 30)
    logger.info(f"Recall  : {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    logger.info(
        f"Rec@1%: {rec_at_1:.4f} | Rec@0.1%: {rec_at_01:.4f}"
    )
    logger.info("-" * 30)

    # ==========================================
    # 7. Save Results
    # ==========================================
    method_name = feature_name
    note_str = ""
    if test_csv_path:
        custom_suffix = Path(test_csv_path).stem.replace("test_", "")
        method_name = f"{feature_name}_{custom_suffix}"
        note_str = f"Custom: {custom_suffix}"

    append_global_result(
        method=method_name,
        dataset=dataset_name,
        acc=f"{acc:.4f}",
        prec=f"{prec:.4f}",
        rec=f"{rec:.4f}",
        f1=f"{f1:.4f}",
        auc=f"{auc:.4f}",
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        fnr=f"{fnr:.4f}",
        fpr=f"{fpr:.4f}",
        tnr=f"{tnr:.4f}",
        rec_at_fpr_1=f"{rec_at_1:.4f}",
        rec_at_fpr_01=f"{rec_at_01:.4f}",
        latency_avg_ms=f"{latency_avg:.4f}",
        latency_p99_ms=f"{latency_p99:.4f}",
        qps=f"{qps:.2f}",
        model_size_mb=f"{model_size_mb:.2f}",
        feature_dim=feature_dim,
        notes=note_str,
    )

    # Save Plot
    if plots_save_path:
        cm_path = plots_save_path.replace(".png", "_cm.png")
        save_confusion_matrix(labels, preds, cm_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dataset1")
    parser.add_argument("--test_file", type=str, default=None)
    args = parser.parse_args()

    run_bert_testing(args.dataset, test_csv_path=args.test_file)
