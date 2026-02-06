# src/features/mgsir/extractor_enhanced.py
import os
import pickle
import numpy as np
import pandas as pd
import csv
from src.features.mgsir.hdcan import *
from src.features.mgsir.mgsif import *
from pathlib import Path


def get_ablation_features(mode):
    """
    根据消融模式名称，返回对应的特征列名列表
    
    模式说明：
    - L1/L2/L3/L4: 单独某一层
    - L1_L2/L1_L2_L3: 累加层
    - No_L1/No_L2/No_L3/No_L4: 减去某一层
    - Full: 所有层
    """
    L1, L2, L3, L4 = LEVEL_1_KEYS, LEVEL_2_KEYS, LEVEL_3_KEYS, LEVEL_4_KEYS
    
    # 模式映射表
    mode_map = {
        # 单独层
        "L1": L1, "L1_only": L1,
        "L2": L2, "L2_only": L2,
        "L3": L3, "L3_only": L3,
        "L4": L4, "L4_only": L4,
        
        # 累加层
        "L1_L2": L1 + L2,
        "L1_L2_L3": L1 + L2 + L3,
        
        # 减法层 (保留别名以兼容旧代码)
        "No_L1": L2 + L3 + L4,
        "No_L2": L1 + L3 + L4,
        "No_L3": L1 + L2 + L4,
        "No_L4": L1 + L2 + L3,  # 同 L1_L2_L3
        
        # 完整
        "Full": L1 + L2 + L3 + L4,
    }
    
    # 返回对应模式，如果模式不存在则返回所有特征
    return mode_map.get(mode, L1 + L2 + L3 + L4)


# 高级特征提取与处理
def process_dataframe_features(
    df,
    dataset_type="Train",
    feature_dir: Path | None = None,
    save_features=False,
    feature_cols=None,
):
    """
    输入: 只有 Query 的 DataFrame
    输出: 包含 Query, Query_preprocessed, qlen, wcount... 等所有特征的 DataFrame
    """
    print(
        f"\n[Processing] 正在处理 {dataset_type} 集特征 (Input Shape: {df.shape} | Mode: {len(feature_cols) if feature_cols else 'ALL'})..."
    )
    print("[Feature] 执行高级预处理与特征工程...")
    if "Query" in df.columns:
        # 1. 生成预处理后的 Query
        df["Query_preprocessed"] = df["Query"].apply(advanced_preprocess)

        # 2. 提取结构化特征 (extract_struct_features 会根据 Query/Query_preprocessed 生成 qlen, wcount 等)
        # 这一步会将 DataFrame 的列数扩充
        # df = extract_struct_features(df)
        df = extract_struct_features(df, active_cols=feature_cols)

        print(
            f"[Processing] {dataset_type} 特征工程处理完成 (Output Shape: {df.shape})"
        )

        # # 保存逻辑需调整文件名，避免覆盖（可选，或者在外部控制目录）
        # if save_features and feature_dir is not None:
        #      # 为了避免消融实验覆盖原始的全量特征文件，我们不建议在消融模式下覆盖 feature_extracted_Train.csv
        #      # 除非我们在外部改变了 feature_dir。
        #      # 这里假设外部传入的 feature_dir 已经是独立的文件夹 (e.g. checkpoints/ablation_L1/features)
        #     final_csv_path = os.path.join(feature_dir, f"feature_extracted_{dataset_type}.csv")
        #     df.to_csv(final_csv_path, index=False, encoding="utf-8") # 简化写法
        #     print(f"[FeatureEng] 特征已保存: {final_csv_path}")

        # --- 标准处理：如果是 None，就不保存 ---
        if save_features and feature_dir is None:
            raise ValueError(
                "save_features=True 但未提供 feature_dir，请传入有效目录 Path"
            )

        # 3. 💾 控制是否保存
        if save_features:
            # # --- 只有需要保存时，才计算路径 ---
            # if output_dir is None:
            #     if os.path.isfile(feature_dir):
            #         # 输入是文件 (train.csv) -> 回退到父目录
            #         base_dir = os.path.dirname(feature_dir)
            #     else:
            #         # 输入是目录 -> 直接使用
            #         base_dir = feature_dir

            #     # 自动在同级建立 features 文件夹
            #     output_dir = os.path.join(base_dir, "features")

            # # 创建目录
            # if not os.path.exists(output_dir):
            #     os.makedirs(output_dir)
            #     print(f"[Init] 创建特征输出目录: {output_dir}")

            # 保存文件
            final_csv_path = os.path.join(
                feature_dir, f"feature_extracted_{dataset_type}.csv"
            )
            # df.to_csv(final_csv_path, index=False)
            df.to_csv(
                final_csv_path,
                index=False,
                encoding="utf-8",
                quoting=csv.QUOTE_MINIMAL,
                quotechar='"',
                escapechar="\\",
                doublequote=True,
            )
            print(f"[FeatureEng] ✅ 成功！特征文件已保存 => {final_csv_path}")

        else:
            print(
                "[FeatureEng] ⏩ save_features=False，跳过文件保存，仅返回 DataFrame。"
            )
    else:
        raise ValueError("数据中缺少 'Query' 列")
    return df


# main 使用
def prepare_datasets_from_files_enhanced(
    train_df_raw, test_df_raw, feature_name, base_dir, ablation_mode="Full"
):

    if ablation_mode == "Full":
        target_cols = get_ablation_features(ablation_mode)
        print(f"\n[Info] Using FULL feature mode.")
        print(f"[Info] Final feature list ({len(target_cols)}): {target_cols}")
    else:
        target_cols = get_ablation_features(ablation_mode)
        print(f"\n[Ablation] Mode: {ablation_mode}")
        print(f"[Ablation] Activated features ({len(target_cols)}): {target_cols}")

    # === Step 1: 分别对 训练集 和 测试集 做特征工程 ===
    # 这里原来的 train_df_raw(1列) 会变成 x_train_full(十几列)
    # <class 'pandas.core.frame.DataFrame'>
    # Index(['Query', 'Label', 'Query_preprocessed', 'qlen', 'wcount', 'sq', 'dq', 'puncts', 'comments', 'spaces', 'logic', 'arith', 'hexnum', 'alpha', 'sqlkw', 'sqlfunc'], dtype='object')
    # 2. 传递 target_cols 给处理函数
    train_full = process_dataframe_features(
        train_df_raw.copy(),
        "Train",
        base_dir / "features",
        save_features=True,
        feature_cols=target_cols,
    )
    test_full = process_dataframe_features(
        test_df_raw.copy(),
        "Val",
        base_dir / "features",
        save_features=True,
        feature_cols=target_cols,
    )

    # <class 'pandas.core.frame.DataFrame'>
    X_train = train_full.drop(["Label"], axis=1)
    # <class 'numpy.ndarray'>
    y_train = train_full["Label"].values
    X_test = test_full.drop(["Label"], axis=1)
    y_test = test_full["Label"].values

    X_train_num_scaled, X_test_num_scaled, scaler_num = (
        standardize_and_combine_features(X_train, X_test, target_cols)
    )

    train_feat = X_train_num_scaled
    test_feat = X_test_num_scaled

    scaler_num_path = os.path.join(
        base_dir / "scaler", f"scaler_for_numeric_{feature_name}.pkl"
    )

    with open(scaler_num_path, "wb") as f:
        pickle.dump(scaler_num, f)

    return {
        "x_train": X_train,  # 原始训练集特征 (DataFrame)
        "x_test": X_test,  # 原始验证集特征 (DataFrame)
        "y_train": y_train,  # 训练集标签
        "y_test": y_test,  # 验证集标签
        "num_features": (train_feat, test_feat),  # 标准化后的数值特征矩阵
        # "num_features": (X_train_num_scaled, X_test_num_scaled),
    }
