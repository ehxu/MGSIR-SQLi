# src/config/paths.py

"""
Unified Pipeline Path System + Global Result Paths
--------------------------------------------------

Supports:
✔ Multiple feature types (bow, tfidf, mgsir, w2v, bert, fasttext, ...)
✔ Multiple models (xgb, lstm, cnn, transformer, bert, ...)
✔ Unlimited pipeline combinations such as:
  bow_xgb, tfidf_xgb, w2v_cnn, bert_lstm, ...
"""

import csv
from pathlib import Path
from dataclasses import dataclass, field


# ============================================================
# 🌟 Utility Functions
# ============================================================


def _ensure_dirs(*dirs: Path):
    """
    Ensure that multiple directories exist.
    If a directory does not exist, it will be created automatically.
    """
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


# ============================================================
# 🌟 Base Paths
# ============================================================

# Project root: ../../ (two levels above this file)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Standardized directory structure
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
GLOBAL_METRICS_DIR = RESULTS_DIR / "metrics"

# Create these directories if missing
_ensure_dirs(DATA_DIR, RESULTS_DIR, CHECKPOINT_DIR, GLOBAL_METRICS_DIR)

# Global CSV file that aggregates evaluation results from all pipelines
GLOBAL_RESULTS_CSV = GLOBAL_METRICS_DIR / "all_results.csv"


# ============================================================
# 🌟 Global Metrics Writing (Better Structured)
# ============================================================

# CSV header definition for all global evaluation results
# [Updated] Expanded metrics for CCF-B/A requirements (Efficiency, Robustness, Deployment)
GLOBAL_HEADER = [
    "method",
    "dataset",  # [New] To track cross-dataset validation
    "acc",
    "prec",
    "rec",
    "f1",
    "auc",
    "tp",
    "tn",
    "fp",
    "fn",
    "fnr",  # False Negative Rate (Miss Rate)
    "fpr",  # False Positive Rate (Fall-out)
    "tnr",  # True Negative Rate (Specificity)
    "rec_at_fpr_1",  # [New] Recall @ FPR = 1%
    "rec_at_fpr_01",  # [New] Recall @ FPR = 0.1% (Critical for WAF)
    "latency_avg_ms",  # [New] Inference Latency (Feature extraction + Prediction)
    "latency_p99_ms",  # [New] P99 Latency (Tail Latency)
    "qps",  # [New] Queries Per Second
    "train_time_sec",  # [New] Training Duration
    "model_size_mb",  # [New] Model Checkpoint Size
    "feature_dim",  # [New] Feature Dimension Count
    "notes",  # [New] Any additional remarks (e.g. "Adv Set A")
]


def _is_valid_metric_value(val):
    """
    判断指标值是否有效
    有效值: 非None、非空字符串、且数值 >= 0
    """
    if val is None or val == "":
        return False
    try:
        num_val = float(val)
        return num_val >= 0
    except (ValueError, TypeError):
        return False


def append_global_result(
    method: str,
    acc: float,
    prec: float,
    rec: float,
    f1: float,
    auc: float,
    tp: int,
    tn: int,
    fp: int,
    fn: int,
    # === [New] Added arguments ===
    fnr: float = -1.0,
    fpr: float = -1.0,
    tnr: float = -1.0,
    # =============================
    dataset: str = "dataset1",
    rec_at_fpr_1: float = -1.0,
    rec_at_fpr_01: float = -1.0,
    latency_avg_ms: float = -1.0,
    latency_p99_ms: float = -1.0,
    qps: float = -1.0,
    train_time_sec: float = -1.0,
    model_size_mb: float = -1.0,
    feature_dim: int = -1,
    notes: str = "",
    csv_path: Path = GLOBAL_RESULTS_CSV,
):
    """
    Append or UPDATE a result entry in the global metrics CSV file.
    """
    # 1. 构造当前传入的新数据字典
    new_data = {
        "method": str(method),
        "dataset": str(dataset),
        "acc": str(acc),
        "prec": str(prec),
        "rec": str(rec),
        "f1": str(f1),
        "auc": str(auc),
        "tp": str(tp),
        "tn": str(tn),
        "fp": str(fp),
        "fn": str(fn),
        # === [New] Map new fields ===
        "fnr": str(fnr),
        "fpr": str(fpr),
        "tnr": str(tnr),
        # ============================
        "rec_at_fpr_1": str(rec_at_fpr_1),
        "rec_at_fpr_01": str(rec_at_fpr_01),
        "latency_avg_ms": str(latency_avg_ms),
        "latency_p99_ms": str(latency_p99_ms),
        "qps": str(qps),
        "train_time_sec": str(train_time_sec),
        "model_size_mb": str(model_size_mb),
        "feature_dim": str(feature_dim),
        "notes": str(notes),
    }

    rows = []
    updated = False

    # 2. 读取并尝试更新现有行
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("method") == method and row.get("dataset") == dataset:
                    merged_row = row.copy()
                    for key, new_val in new_data.items():
                        # 使用改进的验证逻辑
                        if _is_valid_metric_value(new_val):
                            merged_row[key] = new_val
                        # 如果新值无效但旧值有效，保留旧值
                        elif key in merged_row and _is_valid_metric_value(merged_row[key]):
                            continue
                    rows.append(merged_row)
                    updated = True
                else:
                    rows.append(row)

    # 3. 如果没找到匹配行，则追加新行
    if not updated:
        rows.append(new_data)

    # 4. 写回文件
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=GLOBAL_HEADER)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 🌟 Dataset Paths
# ============================================================


@dataclass(frozen=True)
class DatasetPaths:
    """
    Container for paths related to a given dataset.
    Each dataset has its own raw/processed/train/val/test files.
    """

    name: str
    base_dir: Path
    raw_dir: Path
    processed_dir: Path
    train_csv: Path
    val_csv: Path
    test_csv: Path


def get_dataset_paths(dataset: str = "dataset1") -> DatasetPaths:
    """
    Return all required paths for a dataset.
    Automatically creates raw/processed directories if missing.
    """
    base = DATA_DIR / dataset
    raw = base / "raw"
    processed = base / "processed"

    _ensure_dirs(raw, processed)

    return DatasetPaths(
        name=dataset,
        base_dir=base,
        raw_dir=raw,
        processed_dir=processed,
        train_csv=processed / "train.csv",
        val_csv=processed / "val.csv",
        test_csv=processed / "test.csv",
    )


# ============================================================
# 🌟 Pipeline Path System (Core Component)
# ============================================================


@dataclass(frozen=True)
class PipelinePaths:
    """
    Unified container for all paths used by a pipeline.

    A pipeline = feature_extractor + classifier
    For example:
        bow_xgb, tfidf_svm, bert_lstm, w2v_cnn ...

    Automatically handles:
    - model storage
    - feature storage
    - scalers
    - embeddings
    - logs
    - figures
    - predictions
    """

    pipeline_name: str
    base_dir: Path

    # Sub-directories
    model_dir: Path
    feature_dir: Path
    scaler_dir: Path
    # embeddings_dir: Path
    # logs_dir: Path
    figures_dir: Path
    predictions_dir: Path

    # Standard files inside the pipeline
    model_file: Path
    thr_file: Path
    scaler_file: Path
    plots_file: Path
    predictions_csv: Path

    def file(self, dirname: str, filename: str) -> Path:
        """
        Helper to retrieve a file inside a pipeline sub-directory.
        Example:
            paths.file("features", "vector.npy")
        """
        return (self.base_dir / dirname) / filename


def get_pipeline_paths(pipeline_name: str, sub_dir: str = None) -> PipelinePaths:
    """
    Generate and return all paths for a given pipeline.

    This function:
    - Creates a standard directory structure for the pipeline
    - Generates file paths for model, scaler, plots, predictions, etc.
    """

    if sub_dir:
        base = CHECKPOINT_DIR / sub_dir / pipeline_name
    else:
        base = CHECKPOINT_DIR / pipeline_name

    # Standardized pipeline sub-directory structure
    dirs = {
        "model_dir": base / "model",
        "feature_dir": base / "features",
        "scaler_dir": base / "scaler",
        # "embeddings_dir": base / "embeddings",
        # "logs_dir": base / "logs",
        "figures_dir": base / "figures",
        "predictions_dir": base / "predictions",
    }

    # Ensure all dirs exist
    _ensure_dirs(base, *dirs.values())

    # Build the PipelinePaths data object
    return PipelinePaths(
        pipeline_name=pipeline_name,
        base_dir=base,
        **dirs,
        model_file=dirs["model_dir"] / f"{pipeline_name}_model.pkl",
        thr_file=dirs["model_dir"] / f"{pipeline_name}_threshold.pkl",
        scaler_file=dirs["scaler_dir"] / f"scaler_for_numeric_{pipeline_name}.pkl",
        plots_file=dirs["figures_dir"] / f"plots_{pipeline_name}.png",
        predictions_csv=dirs["predictions_dir"] / f"predictions_{pipeline_name}.csv",
    )
