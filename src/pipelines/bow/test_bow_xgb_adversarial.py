# src/pipelines/bow/test_bow_xgb_adversarial.py
import argparse
import sys
from pathlib import Path

# === 环境配置 ===
current_file = Path(__file__).resolve()
project_root = current_file.parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# 导入 BoW 的测试流水线
from src.pipelines.bow.test_bow_xgb import run_bow_testing

# === 测试集顺序定义 (保持与 mgsir 一致，方便对比) ===
# 格式: (Set ID, Mode Name, Description)
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
    parser = argparse.ArgumentParser(description="BoW Batch Adversarial Testing Script")
    parser.add_argument("--dataset", type=str, default="dataset1", help="Dataset name")
    parser.add_argument(
        "--feature",
        type=str,
        default="bow_xgb",  # 默认使用 BoW 模型
        help="Model feature name to load",
    )
    # [新增] 支持单独指定测试文件，方便调试
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Optional: Run only a specific csv file (e.g. 'test_adv_mix.csv')",
    )
    args = parser.parse_args()

    # 1. 确定对抗样本目录
    adv_dir = project_root / "data" / args.dataset / "adversarial"

    if not adv_dir.exists():
        print(f"[Error] Adversarial directory not found: {adv_dir}")
        print("Please run scripts/generate_adversarial_test_sqlmap.py first.")
        sys.exit(1)

    # 2. 确定要运行的文件列表
    files_to_run = []

    if args.file:
        # === 模式 1: 单文件测试 ===
        target_file = adv_dir / args.file
        if not target_file.exists():
            # 尝试加上 .csv 后缀
            if not str(target_file).endswith(".csv"):
                target_file = adv_dir / f"{args.file}.csv"

            if not target_file.exists():
                # 尝试补全 test_adv_ 前缀
                target_file = adv_dir / f"test_adv_{args.file}.csv"

        if not target_file.exists():
            print(f"[Error] 指定的文件不存在: {args.file}")
            sys.exit(1)

        # 这里的 Set ID 设为 Custom
        files_to_run.append(
            (
                "Custom",
                target_file.stem.replace("test_adv_", ""),
                "Single File Test",
                target_file,
            )
        )

    else:
        # === 模式 2: 按顺序批量测试 (Paper Mode) ===
        print(f"Loading test sets in paper order (Set A -> Set H)...")
        for set_id, mode_name, desc in ORDERED_TEST_SETS:
            # 构建文件名: test_adv_randomcase.csv
            fname = f"test_adv_{mode_name}.csv"
            fpath = adv_dir / fname

            if fpath.exists():
                files_to_run.append((set_id, mode_name, desc, fpath))
            else:
                print(f"[Warn] Missing test set: {fname} (Skipping {set_id})")

    if not files_to_run:
        print("[Error] No valid test files found to run.")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"🚀 Starting BoW Adversarial Robustness Benchmark")
    print(f"🎯 Model:   {args.feature}")
    print(f"📂 Dataset: {args.dataset}")
    print(f"⚔️  Queue:   {len(files_to_run)} test sets")
    print(f"{'='*60}\n")

    # 3. 循环测试
    failed = False
    for set_id, mode_name, desc, file_path in files_to_run:
        print(f">>> Running {set_id}: {desc}")
        print(f"    File: {file_path.name}")

        try:
            # 调用 BoW 专用的测试函数
            run_bow_testing(
                dataset_name=args.dataset,
                feature_name=args.feature,
                test_csv_path=str(file_path),
            )
            print(f"✅ Finished {set_id}\n")
        except Exception as e:
            print(f"❌ Failed {set_id}: {e}\n")
            failed = True

    print(f"{'='*60}")
    print("🏁 All adversarial tests completed.")
    print("Check results/metrics/all_results.csv for summary.")
    print(f"{'='*60}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
