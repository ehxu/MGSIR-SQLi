# src/pipelines/mgsir/train_ablation_core.py
import argparse
import sys
from pathlib import Path

# === 环境配置 ===
current_file = Path(__file__).resolve()
project_root = current_file.parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.pipelines.mgsir.train_core import run_mgsir_training_pipeline
from src.features.mgsir.extractor import prepare_datasets_from_files_enhanced


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dataset1")
    # [修改] 设置 default="Full"
    parser.add_argument(
        "--mode", type=str, default="Full", help="L1, L1_L2, Full, No_L3, L2_only, etc."
    )
    args = parser.parse_args()

    feature_name = f"mgsir_xgb_{args.mode}"

    def ablation_extractor_wrapper(train_df, val_df, f_name, b_dir):
        return prepare_datasets_from_files_enhanced(
            train_df, val_df, f_name, b_dir, ablation_mode=args.mode
        )

    ablation_extractor_wrapper.__name__ = f"extractor_{args.mode}"

    print(f"Starting Ablation Training: {args.mode}")
    run_mgsir_training_pipeline(
        dataset_name=args.dataset,
        feature_name=feature_name,
        extractor_func=ablation_extractor_wrapper,
        sub_dir="ablation",
    )


if __name__ == "__main__":
    main()
