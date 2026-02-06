# scripts/plots/analyze_mgsir_shap.py
import sys
import os
import argparse
import joblib
import pickle
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# === 1. 环境路径配置 ===
# 确保可以引用 src 包
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config.paths import get_pipeline_paths, get_dataset_paths
from src.utils.data_utils import load_test_csv
from src.features.mgsir.mgsif import extract_struct_features
from src.features.mgsir.hdcan import advanced_preprocess
from src.features.mgsir.extractor import get_ablation_features


def main():
    parser = argparse.ArgumentParser(description="XGBoost SHAP Analysis")
    parser.add_argument("--dataset", type=str, default="dataset1", help="Dataset name")
    parser.add_argument(
        "--feature",
        type=str,
        default="mgsir_xgb",
        help="Model name (folder name)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="Full",
        help="Feature mode used in training (Full, L1, L1_L2, etc.)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Number of samples to analyze (SHAP is slow)",
    )
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"🕵️‍♂️  Starting SHAP Analysis")
    print(f"    Model:   {args.feature}")
    print(f"    Dataset: {args.dataset}")
    print(f"    Mode:    {args.mode}")
    print(f"{'='*60}")

    # === 2. 加载路径与工件 ===
    # 使用项目现有的路径管理系统
    paths = get_pipeline_paths(args.feature)

    model_path = paths.model_file
    scaler_path = paths.scaler_file

    # 兼容 ablation 子路径
    # if not model_path.exists():
    #     # 尝试查找 ablation 子目录 (兼容 ablation 训练路径)
    #     paths = get_pipeline_paths(args.feature, sub_dir="ablation")
    #     model_path = paths.model_file
    #     scaler_path = paths.scaler_file

    if not model_path.exists():
        raise FileNotFoundError(f"❌ Model not found at: {model_path}")

    print(f"[Load] Loading Model and Scaler...")
    model = joblib.load(model_path)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # === 3. 加载并准备数据 ===
    # 加载测试集
    dataset_paths = get_dataset_paths(args.dataset)
    df_test = load_test_csv(dataset_paths.test_csv)

    # 为了速度，只取一部分数据进行解释
    if args.limit > 0 and len(df_test) > args.limit:
        # 保持正负样本比例 Stratified Sample
        df_test = df_test.groupby("Label", group_keys=False).apply(
            lambda x: x.sample(int(args.limit * len(x) / len(df_test)), random_state=42)
        )
        print(f"[Data] Subsampled to {len(df_test)} samples for SHAP analysis.")

    # === 4. 特征工程 (关键步骤) ===
    # 必须完全复现训练时的特征提取流程
    print("[Feature] Extracting features...")

    # 4.1 获取对应模式的特征列名
    feature_names = get_ablation_features(args.mode)
    print(f"[Feature] Active Features ({len(feature_names)}): {feature_names}")

    # 4.2 预处理 Query
    df_test["Query_preprocessed"] = df_test["Query"].apply(advanced_preprocess)

    # 4.3 提取结构化特征
    # 注意：这里我们只提取需要的列
    df_features = extract_struct_features(df_test.copy(), active_cols=feature_names)

    # 4.4 筛选数值列 (即 XGBoost 的输入)
    X_raw = df_features[feature_names]
    y_true = df_features["Label"].values

    # 4.5 标准化 (Scaler Transform)
    # SHAP 需要基于模型看到的真实输入（即 Scaled 后的数据）
    X_scaled = scaler.transform(X_raw.values)

    # 创建 DataFrame 方便 SHAP 显示列名
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

    # === 5. SHAP 计算 ===
    print("[SHAP] Calculating SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled_df)

    # === 6. 保存目录 ===
    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # A. Summary Plot (全局重要性)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_scaled_df, show=False, max_display=30)
    sum_plot_path = output_dir / f"{args.feature}_shap_summary.png"
    plt.title(f"SHAP Summary", fontsize=16)
    plt.tight_layout()
    plt.savefig(sum_plot_path, dpi=300)
    plt.close()
    print(f"[Save] Summary plot saved to: {sum_plot_path}")

    # B. Bar Plot (特征重要性排名)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_scaled_df, plot_type="bar", show=False, max_display=30)

    bar_plot_path = output_dir / f"{args.feature}_shap_importance.png"
    plt.title(f"Feature Importance: {args.feature}", fontsize=16)
    plt.tight_layout()
    plt.savefig(bar_plot_path, dpi=300)
    plt.close()
    print(f"[Save] Bar plot saved to: {bar_plot_path}")

    # C. Waterfall Plot (局部单样本解释 - 找一个正样本/攻击样本)
    # 找到第一个被预测为攻击的样本索引
    print("[SHAP] Generating waterfall plot for one attack sample...")
    preds = model.predict(X_scaled)
    attack_indices = np.where(preds == 1)[0]

    if len(attack_indices) > 0:
        idx = attack_indices[0]
        # 注意：Waterfall 需要 Explainer 的 object 格式
        explainer_obj = shap.Explainer(model)
        shap_obj = explainer_obj(X_scaled_df)

        plt.figure(figsize=(8, 6))
        # [Image of SHAP waterfall plot]
        shap.plots.waterfall(shap_obj[idx], show=False)
        water_plot_path = (
            output_dir / f"{args.feature}_shap_waterfall_attack_sample.png"
        )
        plt.savefig(water_plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"[Save] Waterfall plot (Attack Sample) saved to: {water_plot_path}")
        print(f"       Sample Query: {df_test.iloc[idx]['Query'][:100]}...")

    else:
        print("⚠️ No attack samples found for waterfall plot.")

    print(f"{'='*60}")
    print("✅ SHAP Analysis Completed.")


if __name__ == "__main__":
    main()
