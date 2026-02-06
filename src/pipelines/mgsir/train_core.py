# src/pipelines/mgsir/train_core.py
import sys
from pathlib import Path
from typing import Callable, Optional

# === 导入项目通用工具 ===
current_file = Path(__file__).resolve()
project_root = current_file.parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# [Updated] Import append_global_result to save train metrics
from src.config.paths import get_pipeline_paths, append_global_result
from src.utils.data_utils import resolve_dataset, load_data_from_csv
from src.models.traditional.trainer_xgb import train_and_save_xgb_model
from src.config.train_config import TRAIN_CONFIG
from src.config.model_config import MODELS_CONFIG
from src.utils.logger import setup_logger


def run_mgsir_training_pipeline(
    dataset_name: str,
    feature_name: str,
    extractor_func: Callable,
    train_csv_path: Optional[str] = None,
    val_csv_path: Optional[str] = None,
    sub_dir: Optional[str] = None,
):
    """
    通用的 mgsir 训练流水线函数。

    Args:
        dataset_name: 数据集名称
        feature_name: 特征名称 (如 mgsir_xgb, mgsir_xgb_en)
        extractor_func: 特征提取函数 (依赖注入，传入中文或英文的处理函数)
        train_csv_path: 自定义训练集路径
        val_csv_path: 自定义验证集路径
    """

    # 1. 初始化 Logger
    log_dir = project_root / "results" / "logs"
    log_file = log_dir / f"train_{feature_name}_{dataset_name}.log"
    logger = setup_logger(f"train_{feature_name}", log_file)

    logger.info(f"{'='*60}")
    logger.info(f"[Start] 开始训练 pipeline: {feature_name}")
    logger.info(f"[Mode]  使用特征提取器: {extractor_func.__name__}")
    logger.info(f"{'='*60}")

    # 2. 路径解析与数据加载
    train_csv, val_csv = resolve_dataset(dataset_name, train_csv_path, val_csv_path)
    logger.info(f"[Path] Train CSV: {train_csv}")
    logger.info(f"[Path] Val CSV:   {val_csv}")

    train_df, val_df = load_data_from_csv(train_csv, val_csv)
    logger.info(f"[Load] 数据加载完成. Train: {train_df.shape}, Val: {val_df.shape}")

    logger.info("[Dataset] Train/Val sample distribution:")
    logger.info(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

    if "Label" in train_df.columns:
        logger.info("[Train Label Distribution]")
        for label, cnt in train_df["Label"].value_counts().items():
            logger.info(f"  Label {label}: {cnt}")

    if "Label" in val_df.columns:
        logger.info("[Val Label Distribution]")
        for label, cnt in val_df["Label"].value_counts().items():
            logger.info(f"  Label {label}: {cnt}")

    # 3. 准备 Pipeline 路径
    pipeline_paths = get_pipeline_paths(feature_name, sub_dir=sub_dir)
    pipeline_paths.feature_dir.mkdir(parents=True, exist_ok=True)

    # 4. 特征提取 (关键点：调用传入的 extractor_func)
    logger.info(f"[Feature] 正在处理特征: {feature_name}...")

    features_dict = extractor_func(
        train_df, val_df, feature_name, pipeline_paths.base_dir
    )

    y_train = features_dict["y_train"]
    y_val = features_dict["y_test"]
    train_feat, val_feat = features_dict["num_features"]

    # [New] Get feature dimension
    feature_dim = train_feat.shape[1]

    logger.info("-" * 30)
    logger.info(f"训练特征维度: {train_feat.shape} | 训练标签: {len(y_train)}")
    logger.info(f"验证特征维度: {val_feat.shape} | 验证标签: {len(y_val)}")
    logger.info("-" * 30)

    # 5. 模型训练
    logger.info("[Model] 开始训练模型 (XGBoost)...")
    model_name = TRAIN_CONFIG["model_name"]
    model_cfg = MODELS_CONFIG[model_name]

    results = train_and_save_xgb_model(
        train_x=train_feat,
        val_x=val_feat,
        train_y=y_train,
        val_y=y_val,
        model_cfg=model_cfg,
        pipeline_name=feature_name,
        logger=logger,
        sub_dir=sub_dir,
    )

    # 6. 结果摘要 & 保存到 Global CSV
    metrics = results.get("metrics", {})
    val_f1 = metrics.get("f1_val_best_threshold", metrics.get("val_default_f1", "N/A"))

    # [New] Extract efficiency metrics from trainer results
    train_time = results.get("train_time_sec", -1.0)
    model_size = results.get("model_size_mb", -1.0)

    # [New] Append Training Artifact Info to global CSV
    # 只记录训练资源消耗，分类指标设为 -1 (由 Test 阶段负责)
    # [修改为]：直接使用 feature_name，确保和 Test 阶段的 method 一致
    train_method_name = feature_name

    append_global_result(
        method=train_method_name,
        dataset=dataset_name,
        # Performance metrics are NOT written (set to -1)
        acc="-1",
        prec="-1",
        rec="-1",
        f1="-1",
        auc="-1",
        tp=-1,
        tn=-1,
        fp=-1,
        fn=-1,
        fnr="-1",
        fpr="-1",
        tnr="-1",
        # ===================================
        # Efficiency/Resource metrics ARE written
        train_time_sec=f"{train_time:.2f}",
        model_size_mb=f"{model_size:.2f}",
        feature_dim=feature_dim,
        notes="",
    )

    logger.info(f"{'='*60}")
    logger.info(f"[Done] 训练流程结束")
    logger.info(f"[Result] 最终验证集 F1: {val_f1}")
    logger.info(
        f"[Efficiency] 训练耗时: {train_time:.2f}s | 模型大小: {model_size:.2f}MB | 特征维度: {feature_dim}"
    )
    logger.info(f"[Path] 模型已保存至: {results['model_path']}")
    logger.info(f"{'='*60}")
