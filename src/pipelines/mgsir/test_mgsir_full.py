# src/pipelines/mgsir/test_mgsir_all_en.py
import argparse
import sys
import warnings
from pathlib import Path

# === 环境配置 ===
warnings.filterwarnings("ignore")
current_file = Path(__file__).resolve()
project_root = current_file.parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# === 导入组件 ===
# 1. 导入通用测试流程
from src.pipelines.mgsir.test_core import run_mgsir_testing_pipeline

# 2. 导入增强版特征提取器
from src.features.mgsir.mgsif import extract_struct_features_single

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="dataset1", help="指定测试数据集"
    )
    parser.add_argument(
        "--feature", type=str, default="mgsir_xgb", help="指定特征方法"
    )
    args = parser.parse_args()

    try:
        run_mgsir_testing_pipeline(
            dataset_name=args.dataset,
            feature_name=args.feature,
            extractor_func=extract_struct_features_single,  # <--- 增强版提取函数
        )
    except KeyboardInterrupt:
        print("\n[INFO] 用户手动终止程序。")
    except Exception as e:
        print(f"\n[Error] 程序发生未捕获异常: {e}")
        raise e
