# src/utils/data_utils.py
import os
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import unquote
import warnings

import chardet
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ================= 全局配置 =================
# 忽略 Pandas/NumPy 不必要的警告
warnings.filterwarnings("ignore")


# ================= 内部工具 =================
def _resolve_path(path_or_dir, default_filename):
    """解析目录或 CSV 文件路径"""
    path_or_dir = str(path_or_dir)
    if path_or_dir.lower().endswith(".csv"):
        return path_or_dir
    else:
        return os.path.join(path_or_dir, default_filename)


def _load_raw_file(data_path: str) -> pd.DataFrame:
    """底层读取 CSV/JSON 文件并检查必要列"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"[Error] 文件不存在: {data_path}")

    file_lower = data_path.lower()
    if file_lower.endswith(".json"):
        print(f"[INFO] Loading JSON: {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            df = pd.read_json(f)
    else:
        try:
            # 优先尝试 latin1
            df = pd.read_csv(data_path, encoding="latin1", on_bad_lines="skip")
        except UnicodeDecodeError:
            with open(data_path, "rb") as f:
                # 只读前 100KB 检测编码，提升效率
                enc = chardet.detect(f.read(100000))["encoding"]
            print(f"[INFO] Detected encoding: {enc}")
            df = pd.read_csv(data_path, encoding=enc, on_bad_lines="skip")

    # --- 基础列名检查 ---
    if "Query" not in df.columns or "Label" not in df.columns:
        raise ValueError(
            f"[Error] 数据缺少必要列 (Query, Label), 当前列: {list(df.columns)}"
        )

    return df


def _clean_dataframe(df):
    """
    清洗逻辑中心：标准版 (移除URL解码，避免与特征提取重复)
    """
    # 1. 备份防止修改原数据
    df = df.copy()

    # 2. 标签标准化 (Label Normalization)
    if df["Label"].dtype not in [np.int64, np.int32, np.float64]:
        # 逻辑：只要是 'attack' (不区分大小写) 就算 1，其他算 0
        # 移除字符串开头和结尾的所有空白字符
        df["Label"] = df["Label"].apply(
            lambda x: 1 if str(x).lower().strip() == "attack" else 0
        )

    # 确保 Label 是整数类型
    df["Label"] = df["Label"].astype(int)

    # 3. Query 基础清洗
    # 步骤：填空 -> 转字符串 -> 去首尾空格
    # URL解码移至特征提取层，避免重复解码
    df["Query"] = (
        df["Query"]
        .fillna("")  # 填补 NaN
        .astype(str)  # 强转字符串
        .str.strip()  # 去掉首尾空格
    )

    return df


# ================= 数据切分 =================
def process_and_split_data(
    raw_data_path: str,
    output_dir: Optional[str] = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> None:
    """处理原始数据并切分 train/val/test"""
    """
    【数据切分工具】读取原始数据 -> 全局严格去重 -> 切分 -> 保存
    比例说明：默认 70% 训练, 10% 验证, 20% 测试
    """
    # 核心修改逻辑：自动生成输出路径
    if output_dir is None:
        # 1. 获取 raw_data_path 所在的文件夹 (例如 "data/raw/total.csv" -> "data/raw")
        raw_data_dir = os.path.dirname(raw_data_path)

        # 2. 拼接 "processed" (例如 -> "data/raw/processed")
        output_dir = os.path.join(raw_data_dir, "processed")

    print(f"[Config] 输出目录自动设置为: {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"[Processing] 数据输出目录: {output_dir}")
    print("=" * 50)
    print(f"[Processing] 开始处理数据...")

    # 1. 读取 & 清洗
    df = _load_raw_file(raw_data_path)
    df = _clean_dataframe(df)

    # 2. 全局严格去重 (要求：证明特征提取的泛化性)
    original_len = len(df)
    print(f"原始样本总数: {len(df)}")
    print(f"[Raw] 严格去重的总数据集样本标签分布:\n{df['Label'].value_counts()}")
    df.drop_duplicates(subset=["Query", "Label"], inplace=True)
    # 冲突检查 (可选)：同一个 Query 既是 0 又是 1 的脏数据，删掉
    df.drop_duplicates(subset=["Query"], keep=False, inplace=True)
    print(f"[Dropped] 去重删除样本数: {original_len - len(df)}")
    print(f"严格去重后有效样本总数: {len(df)}")
    print(f"[Unique] 严格去重的总数据集样本标签分布:\n{df['Label'].value_counts()}")

    # 3. 切分测试集 (Test)
    df_train_temp, df_test = train_test_split(
        df, test_size=test_size, stratify=df["Label"], random_state=random_state
    )

    # 4. 切分验证集 (Val)
    # 换算比例：从剩下的 80% 里切出总量的 10%
    real_val_ratio = val_size / (1 - test_size)
    df_train, df_val = train_test_split(
        df_train_temp,
        test_size=real_val_ratio,
        stratify=df_train_temp["Label"],
        random_state=random_state,
    )

    # 5. 保存
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    df_train.to_csv(train_path, index=False, encoding="utf-8")
    df_val.to_csv(val_path, index=False, encoding="utf-8")
    df_test.to_csv(test_path, index=False, encoding="utf-8")

    print(
        f"[Processing] 数据集生成完成: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}"
    )


# ================= 数据加载 =================
def load_csv(path_or_dir: str, default_filename: str) -> pd.DataFrame:
    target_path = _resolve_path(path_or_dir, default_filename)
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"文件不存在: {target_path}")
    df = pd.read_csv(target_path, encoding="utf-8")
    # print(
    #     f"[Load] {default_filename} 样本数: {len(df)}, 标签分布:\n{df['Label'].value_counts()}"
    # )
    return df


def load_train_csv(path_or_dir: str) -> pd.DataFrame:
    return load_csv(path_or_dir, "train.csv")


def load_val_csv(path_or_dir: str) -> pd.DataFrame:
    return load_csv(path_or_dir, "val.csv")


def load_test_csv(path_or_dir: str) -> pd.DataFrame:
    return load_csv(path_or_dir, "test.csv")


# ================= 数据集解析（支持 CSV / datasetX） =================
def resolve_dataset(
    dataset_name: str = None,
    train_csv_path: str = None,
    val_csv_path: str = None,
) -> Tuple[Path, Path]:
    """
    返回训练集 / 验证集 CSV 路径。
    逻辑：
    1. 用户提供 train_csv_path + val_csv_path，直接使用
    2. 否则使用 dataset_name 对应默认目录
    """
    # 用户提供 CSV 文件
    if train_csv_path and val_csv_path:
        train_path, val_path = Path(train_csv_path), Path(val_csv_path)
        if not train_path.exists() or not val_path.exists():
            raise FileNotFoundError(f"CSV 文件不存在: {train_path}, {val_path}")
        print(f"[INFO] 使用用户提供 CSV 文件: {train_path} / {val_path}")
        return train_path, val_path
    elif train_csv_path or val_csv_path:
        # 只提供一个 CSV 就报错
        raise ValueError("[ERROR] 必须同时提供训练集 CSV 和验证集 CSV")

    # 使用 dataset_name 默认目录
    if dataset_name is None:
        raise ValueError("[ERROR] 未提供 dataset_name，也未提供 CSV 文件")

    from src.config.paths import get_dataset_paths  # 避免循环导入

    dataset_paths = get_dataset_paths(dataset_name)
    train_path, val_path = dataset_paths.train_csv, dataset_paths.val_csv

    # 如果默认 CSV 不存在，尝试自动生成 processed 数据
    if not train_path.exists() or not val_path.exists():
        print(f"[INFO] 数据集 {dataset_name} 不完整，自动生成 processed 数据...")
        from src.utils.data_utils import process_and_split_data

        raw_csv = dataset_paths.raw_dir / "All_SQL_Dataset.csv"
        process_and_split_data(str(raw_csv), str(dataset_paths.processed_dir))

    return train_path, val_path


def load_data_from_csv(
    train_csv: Path, val_csv: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """从 CSV 加载训练集和验证集"""
    return load_train_csv(train_csv), load_val_csv(val_csv)
