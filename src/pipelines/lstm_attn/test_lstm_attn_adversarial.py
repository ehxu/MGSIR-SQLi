# src/pipelines/deep/test_lstm_attn_adversarial.py
import argparse
import sys
from pathlib import Path

# === 环境配置 ===
current_file = Path(__file__).resolve()
project_root = current_file.parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# 导入 LSTM+Attention 的测试流水线核心函数
from src.pipelines.lstm_attn.test_lstm_attn import run_lstm_attn_testing

# === 测试集顺序定义 ===
ORDERED_TEST_SETS = [
    ("Set A", "randomcase", "Syntactic: Random Case"),
    ("Set B", "space2comment", "Syntactic: Inline Comment"),
    ("Set C", "charencode", "Encoding: URL Encoding"),
    ("Set D", "whitespace", "Encoding: Whitespace Manipulation"),
    ("Set E", "versioned", "Semantic: Versioned Comments"),
    ("Set F", "symbolic", "Semantic: Symbolic Replacement"),
    ("Set G", "equaltolike", "Semantic: Operator Substitution"),
    ("Set H", "mix", "Comprehensive: Mixed Attacks"),
]


def main():
    parser = argparse.ArgumentParser(
        description="LSTM+Attention Adversarial Testing Script"
    )
    parser.add_argument("--dataset", type=str, default="dataset1", help="Dataset name")
    parser.add_argument(
        "--feature", type=str, default="lstm_attn", help="Model feature name"
    )
    parser.add_argument(
        "--file", type=str, default=None, help="Optional: Run only specific file"
    )
    args = parser.parse_args()

    adv_dir = project_root / "data" / args.dataset / "adversarial"
    if not adv_dir.exists():
        print(f"[Error] Adversarial dir not found: {adv_dir}")
        sys.exit(1)

    # 确定运行文件列表
    files_to_run = []
    if args.file:
        target_file = adv_dir / args.file
        if not target_file.exists():
            if not str(target_file).endswith(".csv"):
                target_file = adv_dir / f"{args.file}.csv"
            if not target_file.exists():
                target_file = adv_dir / f"test_adv_{args.file}.csv"

        if target_file.exists():
            files_to_run.append(
                (
                    "Custom",
                    target_file.stem.replace("test_adv_", ""),
                    "Single File",
                    target_file,
                )
            )
        else:
            print(f"[Error] File not found: {args.file}")
            sys.exit(1)
    else:
        print(f"Loading test sets in paper order...")
        for set_id, mode_name, desc in ORDERED_TEST_SETS:
            fname = f"test_adv_{mode_name}.csv"
            fpath = adv_dir / fname
            if fpath.exists():
                files_to_run.append((set_id, mode_name, desc, fpath))
            else:
                print(f"[Warn] Missing: {fname}")

    print(f"{'='*60}")
    print(f"🚀 Starting LSTM+Attention Adversarial Benchmark")
    print(f"🎯 Feature: {args.feature}")
    print(f"{'='*60}\n")

    failed = False
    for set_id, mode_name, desc, file_path in files_to_run:
        print(f">>> Running {set_id}: {desc}")
        try:
            # 调用 run_lstm_attn_testing
            run_lstm_attn_testing(
                dataset_name=args.dataset,
                feature_name=args.feature,
                test_csv_path=str(file_path),
            )
            print(f"✅ Finished {set_id}\n")
        except Exception as e:
            print(f"❌ Failed {set_id}: {e}\n")
            failed = True

    print(f"{'='*60}")
    print("🏁 All tests completed.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
