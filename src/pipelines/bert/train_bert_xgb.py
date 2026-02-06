# src/pipelines/bert/train_bert_xgb.py
import sys
import argparse
from pathlib import Path

# === 环境配置 ===
current_file = Path(__file__).resolve()
project_root = current_file.parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config.paths import get_pipeline_paths, append_global_result
from src.utils.data_utils import resolve_dataset, load_data_from_csv
from src.utils.logger import setup_logger
from src.config.model_config import MODELS_CONFIG

# 复用现有的 XGB Trainer
from src.models.traditional.trainer_xgb import train_and_save_xgb_model

# 导入 BERT Extractor
from src.features.bert_extractor import process_bert_features


def run_bert_training(dataset_name="dataset1", feature_name="bert_xgb"):
    # 1. 初始化 Logger
    log_dir = project_root / "results" / "logs"
    logger = setup_logger(
        f"train_{feature_name}", log_dir / f"train_{feature_name}_{dataset_name}.log"
    )

    logger.info(f"{'='*60}")
    logger.info(f"[Start] BERT Training Pipeline: {feature_name}")
    logger.info(f"{'='*60}")

    # 2. 加载数据
    train_csv, val_csv = resolve_dataset(dataset_name)
    train_df, val_df = load_data_from_csv(train_csv, val_csv)

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

    # 3. 准备路径
    paths = get_pipeline_paths(feature_name)

    # 4. 特征提取 (BERT)
    data_dict = process_bert_features(train_df, val_df, feature_name, paths.base_dir)

    # 获取 BERT 大小
    bert_backbone_size = data_dict.get("bert_size_mb", 0.0)

    # 5. 模型训练
    model_cfg = MODELS_CONFIG["xgb"]

    results = train_and_save_xgb_model(
        train_x=data_dict["x_train"],
        train_y=data_dict["y_train"],
        val_x=data_dict["x_val"],
        val_y=data_dict["y_val"],
        model_cfg=model_cfg,
        pipeline_name=feature_name,
        logger=logger,
    )

    # 6. 记录训练产物信息到 Global CSV
    # 保持 method 名一致，只填资源指标，Metrics 填 -1
    train_time = results.get("train_time_sec", -1.0)
    xgb_model_size = results.get("model_size_mb", -1.0)
    feature_dim = data_dict["feature_dim"]

    # === 核心修改：累加大小 ===
    total_model_size = bert_backbone_size + xgb_model_size

    logger.info(f"[Size Info] BERT Backbone: {bert_backbone_size:.2f} MB")
    logger.info(f"[Size Info] XGB Classifier: {xgb_model_size:.2f} MB")
    logger.info(f"[Size Info] Total to Report: {total_model_size:.2f} MB")

    append_global_result(
        method=feature_name,
        dataset=dataset_name,
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
        train_time_sec=f"{train_time:.2f}",
        model_size_mb=f"{total_model_size:.2f}",
        feature_dim=feature_dim,
        notes="",
    )

    logger.info(f"[Done] BERT Training Finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dataset1")
    args = parser.parse_args()

    run_bert_training(args.dataset)
