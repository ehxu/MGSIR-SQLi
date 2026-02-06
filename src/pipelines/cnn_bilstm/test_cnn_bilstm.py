# src/pipelines/deep/test_cnn_bilstm.py
import sys
import os
import time
import pickle
import argparse
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
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
from src.utils.logger import setup_logger
from src.visualization.metric_plots import compute_plot_auc, save_confusion_matrix
from src.utils.runtime_env import configure_tensorflow_runtime

# 参数必须与训练时一致
MAX_SEQUENCE_LENGTH = 200


def load_deep_artifacts(pipeline_name, logger=None):
    paths = get_pipeline_paths(pipeline_name)
    # 这里的模型文件名要和训练保存的一致
    model_path = paths.model_dir / f"model_{pipeline_name}.h5"
    tok_path = paths.feature_dir / f"tokenizer_{pipeline_name}.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Deep Model not found: {model_path}")
    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tok_path}")

    model = load_model(str(model_path))
    with open(tok_path, "rb") as f:
        tokenizer = pickle.load(f)

    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    if logger:
        logger.info(f"[Artifacts] Deep Model loaded: {model_size_mb:.2f} MB")

    return model, tokenizer, model_size_mb


def run_cnn_bilstm_testing(
    dataset_name="dataset1", feature_name="cnn_bilstm", test_csv_path=None
):
    configure_tensorflow_runtime(tf)
    # 1. Logger
    log_dir = project_root / "results" / "logs"
    suffix = f"_{Path(test_csv_path).stem}" if test_csv_path else ""
    logger = setup_logger(
        f"test_{feature_name}{suffix}",
        log_dir / f"test_{feature_name}_{dataset_name}{suffix}.log",
    )

    logger.info(f"{'='*60}")
    logger.info(f"[Start] Deep Testing: {feature_name}")
    logger.info(f"{'='*60}")

    # 2. Data
    if test_csv_path:
        df_test = load_test_csv(test_csv_path)
    else:
        paths = get_dataset_paths(dataset_name)
        df_test = load_test_csv(paths.test_csv)

    # 确保转为字符串
    queries = df_test["Query"].astype(str).tolist()
    labels = df_test["Label"].values

    # 3. Artifacts
    model, tokenizer, model_size_mb = load_deep_artifacts(feature_name, logger)
    paths = get_pipeline_paths(feature_name)

    plots_save_path = str(paths.plots_file)
    if test_csv_path:
        custom_name = Path(test_csv_path).stem
        plot_dir = Path(paths.plots_file).parent
        plots_save_path = str(plot_dir / f"plots_{feature_name}_{custom_name}.png")

    # ==========================================
    # 4. Efficiency Benchmark (Latency & QPS)
    # ==========================================
    logger.info("[Benchmark] Running Efficiency Test (Single Inference)...")
    latencies = []

    # Deep Model 很慢，如果是 GPU 可能需要 warm up，这里简单测前 500 条
    bench_queries = queries[:500] if len(queries) > 500 else queries

    # Warm up (exclude from measurement): run full preprocess + predict once
    try:
        warm_q = queries[0] if queries else "select * from admin"
        seq = tokenizer.texts_to_sequences([warm_q])
        pad = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
        _ = model.predict(pad, verbose=0)
    except Exception:
        _ = model.predict(np.zeros((1, MAX_SEQUENCE_LENGTH)), verbose=0)

    for q in bench_queries:
        t0 = time.perf_counter()

        # A. Preprocess (Text -> Seq -> Pad)
        # 单条处理
        seq = tokenizer.texts_to_sequences([q])
        pad = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)

        # B. Predict
        _ = model.predict(pad, verbose=0)

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

    # 批量预处理
    seqs = tokenizer.texts_to_sequences(queries)
    X_test = pad_sequences(seqs, maxlen=MAX_SEQUENCE_LENGTH)
    feature_dim = MAX_SEQUENCE_LENGTH

    # 批量预测 (Softmax output: [prob_class0, prob_class1])
    # batch_size 设大一点提高 GPU 利用率
    y_pred_prob = model.predict(X_test, batch_size=64, verbose=1)

    # 取正类概率 (Class 1)
    proba = y_pred_prob[:, 1]
    preds = np.argmax(y_pred_prob, axis=1)  # 0 or 1

    # ==========================================
    # 6. Metrics & WAF Robustness
    # ==========================================
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    tnr = tn / (fp + tn) if (fp + tn) > 0 else 0

    auc = 0.0
    rec_at_1, rec_at_01 = -1.0, -1.0

    if len(np.unique(labels)) > 1:
        # 传递 proba (正类概率) 给 AUC 计算
        auc, fpr_arr, tpr_arr = compute_plot_auc(
            labels,
            proba,
            model_name=f"{feature_name}",
            plot_path=plots_save_path.replace(".png", "_roc.png"),
        )

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

    if plots_save_path:
        cm_path = plots_save_path.replace(".png", "_cm.png")
        save_confusion_matrix(labels, preds, cm_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dataset1")
    parser.add_argument("--test_file", type=str, default=None)
    args = parser.parse_args()

    run_cnn_bilstm_testing(args.dataset, test_csv_path=args.test_file)
