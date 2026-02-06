# src/pipelines/mgsir/test_ablation_core.py
import argparse
import sys
from pathlib import Path

# === 环境配置 ===
current_file = Path(__file__).resolve()
project_root = current_file.parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.pipelines.mgsir.test_core import run_mgsir_testing_pipeline
from src.features.mgsir.mgsif import extract_struct_features_single
from src.features.mgsir.extractor import get_ablation_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dataset1")
    # [修改] 设置 default="Full"
    parser.add_argument("--mode", type=str, default="Full")
    args = parser.parse_args()

    feature_name = f"mgsir_xgb_{args.mode}"

    target_cols = get_ablation_features(args.mode)

    extractor_wrapper = lambda q: extract_struct_features_single(
        q, active_cols=target_cols
    )
    extractor_wrapper.__module__ = "ablation"
    extractor_wrapper.__name__ = f"extract_{args.mode}"

    print(f"Starting Ablation Testing: {args.mode} (Features: {len(target_cols)})")

    run_mgsir_testing_pipeline(
        dataset_name=args.dataset,
        feature_name=feature_name,
        extractor_func=extractor_wrapper,
        sub_dir="ablation",
    )


if __name__ == "__main__":
    main()
