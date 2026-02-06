# src/pipelines/deep/train_textcnn.py
import sys
import argparse
import time
import os
from pathlib import Path
import tensorflow as tf

# === 环境配置 ===
current_file = Path(__file__).resolve()
project_root = current_file.parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config.paths import get_pipeline_paths, append_global_result
from src.utils.data_utils import resolve_dataset, load_data_from_csv
from src.utils.logger import setup_logger

# [Change] 导入 TextCNN 模型
from src.models.deep.textcnn import build_textcnn_model
from src.features.deep.base_deep_extractor import process_deep_features
from src.utils.runtime_env import configure_tensorflow_runtime


def run_textcnn_training(dataset_name="dataset1", feature_name="textcnn"):
    configure_tensorflow_runtime(tf)
    # 1. Logger
    log_dir = project_root / "results" / "logs"
    logger = setup_logger(
        f"train_{feature_name}", log_dir / f"train_{feature_name}_{dataset_name}.log"
    )

    logger.info(f"{'='*60}")
    logger.info(f"[Start] TextCNN Training: {feature_name}")
    logger.info(f"{'='*60}")

    # 2. Data
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

    # 3. Features
    paths = get_pipeline_paths(feature_name)
    data_dict = process_deep_features(train_df, val_df, feature_name, paths.base_dir)

    # 4. Model Build [Change]
    model = build_textcnn_model(
        vocab_size=data_dict["vocab_size"], max_len=data_dict["max_len"]
    )
    # model.summary(print_fn=logger.info)

    # 5. Training
    logger.info("[Training] Start fitting model...")
    t0 = time.time()

    history = model.fit(
        data_dict["x_train"],
        data_dict["y_train"],
        validation_data=(data_dict["x_val"], data_dict["y_val"]),
        epochs=5,
        batch_size=32,
        verbose=1,
    )

    train_time = time.time() - t0
    logger.info(f"[Training] Finished in {train_time:.2f}s")

    # 6. Save Model
    model_path = paths.model_dir / f"model_{feature_name}.h5"
    model.save(model_path)
    logger.info(f"[Save] Model saved to {model_path}")

    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    feature_dim = data_dict["max_len"]

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
        model_size_mb=f"{model_size_mb:.2f}",
        feature_dim=feature_dim,
        notes="TextCNN Baseline",
    )

    logger.info(f"[Done] TextCNN Training Finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dataset1")
    args = parser.parse_args()

    run_textcnn_training(args.dataset)
