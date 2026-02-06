# scripts/process_dataset.py
"""
数据集预处理与切分脚本

功能：
1. 读取原始 CSV（包含 Query, Label 列）
2. 执行清洗、严格去重
3. 切分 Train / Val / Test 并保存到 processed 目录

默认行为：
- 如果未提供 --file，则使用 data/<dataset>/raw/All_SQL_Dataset.csv
- 输出到 data/<dataset>/processed（或通过 --output 自定义）
"""

import argparse
import sys
from pathlib import Path

# 兼容从任何工作目录运行
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.data_utils import process_and_split_data
from src.config.paths import get_dataset_paths


def main():
    parser = argparse.ArgumentParser(description="SQLi 数据集清洗与切分工具")
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset1",
        help="数据集名称（默认 data/dataset1/raw/All_SQL_Dataset.csv）",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="可选：自定义原始 CSV 路径（包含 Query, Label 列）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="可选：自定义输出目录（默认写入 data/<dataset>/processed）",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="测试集占比（默认 0.2，对应 70/10/20 划分）",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="验证集占比（默认 0.1）",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="随机种子（默认 42，用于可复现切分）",
    )
    args = parser.parse_args()

    # 解析原始文件和默认输出目录
    if args.file:
        raw_path = Path(args.file)
        default_output = (
            raw_path.parent / "processed"
        )  # 自定义文件默认输出到同级 processed
    else:
        ds_paths = get_dataset_paths(args.dataset)
        raw_path = ds_paths.raw_dir / "All_SQL_Dataset.csv"
        default_output = ds_paths.processed_dir

    output_dir = Path(args.output) if args.output else default_output

    if not raw_path.exists():
        raise FileNotFoundError(
            f"原始数据文件不存在: {raw_path}. 请指定 --file 或将文件放在默认路径。"
        )

    print(f"[Config] Dataset   : {args.dataset}")
    print(f"[Config] Raw file  : {raw_path}")
    print(f"[Config] Output dir: {output_dir}")
    print(
        f"[Config] Split     : train/val/test = {1 - args.test_size - args.val_size:.2f}/{args.val_size:.2f}/{args.test_size:.2f}"
    )

    process_and_split_data(
        raw_data_path=str(raw_path),
        output_dir=str(output_dir),
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
