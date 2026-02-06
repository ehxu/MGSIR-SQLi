# src/pipelines/mgsir/test_core.py
import sys
import os
import time
import pickle
import joblib
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from typing import Callable, List, Optional
from scipy.sparse import csr_matrix
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# === 设置项目根路径 ===
current_file = Path(__file__).resolve()
project_root = current_file.parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# === 导入通用工具 ===
from src.config.paths import get_dataset_paths, get_pipeline_paths, append_global_result
from src.utils.data_utils import load_test_csv
from src.utils.math_ops import sigmoid
from src.visualization.metric_plots import (
    plot_metrics_single,
    compute_plot_auc,
    save_confusion_matrix,
)
from src.utils.logger import setup_logger
from src.utils.runtime_env import configure_xgboost_booster_threads


def load_artifacts(pipeline_name: str, logger=None, sub_dir: Optional[str] = None):
    """加载模型和 Scaler"""
    paths = get_pipeline_paths(pipeline_name, sub_dir=sub_dir)
    scaler_path = str(paths.scaler_file)
    model_path = str(paths.model_file)

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"[Error] 缺少数值特征 Scaler: {scaler_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[Error] 缺少模型文件: {model_path}")

    with open(scaler_path, "rb") as fs:
        numeric_scaler = pickle.load(fs)

    loaded_model = joblib.load(model_path)

    # 计算模型大小
    try:
        model_size_bytes = os.path.getsize(model_path)
        model_size_mb = model_size_bytes / (1024 * 1024)
    except:
        model_size_mb = 0.0

    msg = f"[Artifacts] 成功加载模型: {model_path} ({model_size_mb:.2f} MB) | Scaler: {scaler_path}"
    if logger:
        logger.info(msg)

    return numeric_scaler, loaded_model, model_size_mb


def run_mgsir_testing_pipeline(
    dataset_name: str,
    feature_name: str,
    extractor_func: Callable[[str], List[float]],
    sub_dir: Optional[str] = None,
    test_csv_path: Optional[str] = None,
):
    """
    通用测试流水线 (包含极致性能测试)
    """
    # 1. 初始化 Logger
    log_dir = project_root / "results" / "logs"
    log_suffix = ""
    if test_csv_path:
        custom_name = Path(test_csv_path).stem
        log_suffix = f"_{custom_name}"

    log_file = log_dir / f"test_{feature_name}_{dataset_name}{log_suffix}.log"
    logger = setup_logger(f"test_{feature_name}{log_suffix}", log_file)

    logger.info(f"{'='*60}")
    logger.info(f"[Start] 开始测试 pipeline: {feature_name} | 数据集: {dataset_name}")

    # 2. 加载测试集
    if test_csv_path:
        logger.info(f"[Data] Custom Test CSV: {test_csv_path}")
        if not os.path.exists(test_csv_path):
            raise FileNotFoundError(f"Custom test file not found: {test_csv_path}")
        df_test = load_test_csv(test_csv_path)
    else:
        dataset_paths = get_dataset_paths(dataset_name)
        logger.info(f"[Data] Test CSV: {dataset_paths.test_csv}")
        df_test = load_test_csv(dataset_paths.test_csv)

    test_queries = df_test["Query"].astype(str).tolist()
    test_labels = df_test["Label"].values

    # 3. 加载 Artifacts
    numeric_scaler, loaded_model, model_size_mb = load_artifacts(
        feature_name, logger, sub_dir=sub_dir
    )

    # 准备保存路径
    paths = get_pipeline_paths(feature_name, sub_dir=sub_dir)
    plots_save_path = str(paths.plots_file)
    pred_csv_path = str(paths.predictions_csv)

    if test_csv_path:
        custom_name = Path(test_csv_path).stem
        pred_dir = Path(paths.predictions_csv).parent
        pred_csv_path = str(pred_dir / f"predictions_{feature_name}_{custom_name}.csv")
        plot_dir = Path(paths.plots_file).parent
        plots_save_path = str(plot_dir / f"plots_{feature_name}_{custom_name}.png")

    # ======================================================
    # [Optimized] Step 3.5: Efficiency Benchmark (Latency & QPS)
    # 模拟真实 WAF 引擎：移除 Sklearn 开销，使用纯 NumPy
    # ======================================================
    logger.info("[Benchmark] 开始执行极致推理效率测试 (Efficiency Test)...")

    benchmark_samples = test_queries[:500] if len(test_queries) > 500 else test_queries
    latencies = []

    # [优化 1] 预取 XGBoost Booster (跳过 sklearn wrapper 检查)
    # 这一步直接获取 C++ 核心对象
    booster = loaded_model.get_booster()
    configure_xgboost_booster_threads(booster)

    # [优化 2] 预取 Scaler 参数 (跳过 scaler.transform 的 overhead)
    # 这一步将单次归一化耗时从 0.1ms 压到 0.01ms
    scaler_mean = numeric_scaler.mean_.astype(np.float32)
    scaler_scale = numeric_scaler.scale_.astype(np.float32)

    # [优化 3] 预热 (Warm-up)
    # 激活 CPU 缓存、JIT 和 Python 解释器状态，确保测得的是稳定性能
    print("[Benchmark] Warming up CPU/Cache (20 iterations)...")
    try:
        dummy_feat = extractor_func("select * from admin")  # 提取一次作为模板
        dummy_vec = (
            np.array(dummy_feat, dtype=np.float32) - scaler_mean
        ) / scaler_scale
        dummy_d = xgb.DMatrix(dummy_vec.reshape(1, -1))
        for _ in range(20):
            _ = booster.predict(dummy_d)
    except Exception as e:
        logger.warning(f"[Benchmark] Warmup failed: {e}")

    # --- 正式测试循环 ---
    for q in benchmark_samples:
        t0 = time.perf_counter()

        # A. 特征提取 (极速版 extract_struct_features_single)
        # 耗时约 0.02ms ~ 0.04ms (取决于 hdcan.py 是否优化到位)
        f_raw = extractor_func(q)

        # B. 预处理 (Manual Scale with NumPy)
        # 耗时约 0.01ms (原 scaler.transform 需 0.08ms+)
        f_np = np.array(f_raw, dtype=np.float32)
        f_scaled = (f_np - scaler_mean) / scaler_scale

        # C. 构造 DMatrix & 预测 (Direct Booster)
        # 耗时约 0.08ms (Python DMatrix 构造无法完全避免，但在 C++ 部署时会消失)
        # 注意：reshape(1, -1) 确保是 2D 数组
        d_single = xgb.DMatrix(f_scaled.reshape(1, -1))
        _ = booster.predict(d_single)

        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    latency_avg = np.mean(latencies)
    latency_p99 = np.percentile(latencies, 99)
    qps = 1000.0 / latency_avg if latency_avg > 0 else 0

    logger.info(f"[Benchmark] Avg Latency: {latency_avg:.4f} ms")
    logger.info(f"[Benchmark] P99 Latency: {latency_p99:.4f} ms")
    logger.info(f"[Benchmark] QPS: {qps:.2f} req/sec")

    # ======================================================
    # Step 4: 批量特征提取 (为了计算 Metrics 还是走批量流程更稳)
    # ======================================================
    logger.info("[Feature] 开始提取全量测试集特征 (Batch Mode)...")
    feats_all = []
    valid_idx = []

    for i, raw_q in enumerate(test_queries):
        try:
            feats = extractor_func(raw_q)
            feats_all.append(feats)
            valid_idx.append(i)
        except Exception as e:
            logger.warning(f"Query {i} 解析失败: {e}")
            continue

    if len(valid_idx) == 0:
        logger.error("[Error] 全部 Query 无效, 终止测试.")
        raise ValueError(
            "All queries failed feature extraction. Cannot proceed with testing."
        )

    # 转换格式
    arr_struct = np.array(feats_all, dtype=np.float64)
    feature_dim = arr_struct.shape[1]

    if arr_struct.shape[1] != numeric_scaler.n_features_in_:
        error_msg = f"Feature dimension mismatch! Model expects {numeric_scaler.n_features_in_}, got {arr_struct.shape[1]}"
        logger.error(f"[Error] {error_msg}")
        raise ValueError(error_msg)

    # 批量 Scale (这里可以用 sklearn 原生方法，因为批量处理效率很高)
    arr_struct_scaled = numeric_scaler.transform(arr_struct)
    X_final = csr_matrix(arr_struct_scaled)

    # ======================================================
    # Step 5: 执行批量预测 & 评估
    # ======================================================
    logger.info("[Model] 正在执行全量预测...")
    dtest = xgb.DMatrix(X_final)

    # 获取原始 margin 和 概率
    raw_margin = booster.predict(dtest, output_margin=True)
    proba_raw = sigmoid(raw_margin)
    pred_default = (proba_raw >= 0.5).astype(int)

    # 结果对齐 (防止前面有 Query 解析失败导致长度不一致)
    n_total = len(df_test)
    final_pred_default = np.full(n_total, fill_value=-1, dtype=int)
    final_raw_margin = np.full(n_total, fill_value=np.nan, dtype=float)
    final_proba_raw = np.full(n_total, fill_value=np.nan, dtype=float)

    for i, idx in enumerate(valid_idx):
        final_pred_default[idx] = pred_default[i]
        final_raw_margin[idx] = raw_margin[i]
        final_proba_raw[idx] = proba_raw[i]

    # 仅评估有效样本
    mask_valid = final_pred_default != -1
    sub_labels = test_labels[mask_valid]
    sub_def = final_pred_default[mask_valid]
    sub_proba = final_proba_raw[mask_valid]

    if not np.issubdtype(sub_labels.dtype, np.integer):
        sub_labels = sub_labels.astype(int)

    # 计算基础指标
    acc_def = accuracy_score(sub_labels, sub_def)
    prec_def = precision_score(sub_labels, sub_def, zero_division=0)
    rec_def = recall_score(sub_labels, sub_def, zero_division=0)
    f1_def = f1_score(sub_labels, sub_def, zero_division=0)

    # 绘图
    metrics_default = {"acc": acc_def, "prec": prec_def, "rec": rec_def, "f1": f1_def}
    plot_metrics_single(metrics_dict=metrics_default)

    # AUC
    auc_def = 0.0
    FPR_def = 0.0
    TNR_def = 0.0
    FNR_def = 0.0
    fpr_arr, tpr_arr = None, None

    try:
        if len(np.unique(sub_labels)) > 1:
            auc_def, fpr_arr, tpr_arr = compute_plot_auc(
                y_true=sub_labels,
                y_score=sub_proba,
                model_name=f"{feature_name}_Default0.5",
                plot_path=plots_save_path.replace(".png", "_default_roc.png"),
            )
    except Exception as e:
        logger.error(f"[Error] AUC calculation failed: {e}")

    # Low FPR Metrics (关键 WAF 指标)
    rec_at_1 = -1.0
    rec_at_01 = -1.0

    if fpr_arr is not None and tpr_arr is not None:
        rec_at_1 = np.interp(0.01, fpr_arr, tpr_arr)
        rec_at_01 = np.interp(0.001, fpr_arr, tpr_arr)

    # 混淆矩阵
    try:
        cm_default = confusion_matrix(sub_labels, sub_def, labels=[0, 1])
        TN_def, FP_def, FN_def, TP_def = cm_default.ravel()
        if (FP_def + TN_def) > 0:
            FPR_def = FP_def / (FP_def + TN_def)
            TNR_def = TN_def / (FP_def + TN_def)
        if (FN_def + TP_def) > 0:
            FNR_def = FN_def / (FN_def + TP_def)
    except:
        TN_def, FP_def, FN_def, TP_def = 0, 0, 0, 0

    # 日志输出
    logger.info("\n" + "=" * 30 + " TEST RESULTS " + "=" * 30)
    logger.info(f"   TP: {TP_def:<5} | FN: {FN_def:<5}")
    logger.info(f"   FP: {FP_def:<5} | TN: {TN_def:<5}")
    logger.info("-" * 45)
    logger.info(f" Accuracy  : {acc_def:.4f}")
    logger.info(f" Recall    : {rec_def:.4f}")
    logger.info(f" F1-Score  : {f1_def:.4f}")
    logger.info(f" AUC       : {auc_def:.4f}")
    logger.info(f" FPR       : {FPR_def:.4f}")
    logger.info("-" * 45)
    logger.info(f" Rec@FPR1% : {rec_at_1:.4f}")
    logger.info(f" Rec@FPR0.1%: {rec_at_01:.4f}")
    # Rec@0FP removed per evaluation protocol
    logger.info("=" * 74)

    # 保存结果
    method_name = feature_name
    notes = ""
    if test_csv_path:
        custom_suffix = Path(test_csv_path).stem.replace("test_", "")
        method_name = f"{feature_name}_{custom_suffix}"
        notes = f"Custom: {custom_suffix}"

    append_global_result(
        method=method_name,
        dataset=dataset_name,
        acc=f"{acc_def:.4f}",
        prec=f"{prec_def:.4f}",
        rec=f"{rec_def:.4f}",
        f1=f"{f1_def:.4f}",
        auc=f"{auc_def:.4f}",
        tp=TP_def,
        tn=TN_def,
        fp=FP_def,
        fn=FN_def,
        fnr=f"{FNR_def:.4f}",
        fpr=f"{FPR_def:.4f}",
        tnr=f"{TNR_def:.4f}",
        rec_at_fpr_1=f"{rec_at_1:.4f}",
        rec_at_fpr_01=f"{rec_at_01:.4f}",
        latency_avg_ms=f"{latency_avg:.4f}",
        latency_p99_ms=f"{latency_p99:.4f}",
        qps=f"{qps:.2f}",
        model_size_mb=f"{model_size_mb:.2f}",
        feature_dim=feature_dim,
        notes=notes,
    )

    if plots_save_path:
        cm_plot_path = plots_save_path.replace(".png", "_cm.png")
        save_confusion_matrix(sub_labels, sub_def, cm_plot_path, "(Default=0.5)")

    df_test["RawMargin"] = final_raw_margin
    df_test["Prob_Raw"] = final_proba_raw
    df_test["Pred_Default"] = final_pred_default
    df_test.to_csv(pred_csv_path, index=False, encoding="utf-8")

    logger.info(f"[Done] 测试完成。")
