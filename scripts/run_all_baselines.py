# scripts/run_all_baselines.py
import subprocess
import sys
import time
import os
from pathlib import Path
import argparse

# === 全局配置 ===
DEFAULT_DATASET = "dataset1"
PYTHON_EXEC = sys.executable  # 自动获取当前环境的 Python 解释器

# === Pipeline 定义 (已与 check_files_existence.py 严格对齐) ===
# 格式: (Category, FeatureName, TrainScript, TestScript)
PIPELINES = [
    # ---------------------------
    # 1. Ours (Core Methods)
    # ---------------------------
    (
        "Ours",
        "mgsir_xgb",
        "src/pipelines/mgsir/train_mgsir_full.py",
        "src/pipelines/mgsir/test_mgsir_full.py",
    ),
    # ---------------------------
    # 2. Shallow NLP (Baselines)
    # ---------------------------
    (
        "Shallow",
        "bow_xgb",
        "src/pipelines/bow/train_bow_xgb.py",
        "src/pipelines/bow/test_bow_xgb.py",
    ),
    (
        "Shallow",
        "tfidf_xgb",
        "src/pipelines/tfidf/train_tfidf_xgb.py",
        "src/pipelines/tfidf/test_tfidf_xgb.py",
    ),
    # ---------------------------
    # 3. Vector Embeddings
    # ---------------------------
    (
        "Vector",
        "w2v_xgb",
        "src/pipelines/w2v/train_w2v_xgb.py",
        "src/pipelines/w2v/test_w2v_xgb.py",
    ),
    (
        "Vector",
        "fasttext_xgb",
        "src/pipelines/fasttext/train_fasttext_xgb.py",
        "src/pipelines/fasttext/test_fasttext_xgb.py",
    ),
    # ---------------------------
    # 4. Deep Features (Hybrid)
    # ---------------------------
    (
        "DeepFeat",
        "bert_xgb",
        "src/pipelines/bert/train_bert_xgb.py",
        "src/pipelines/bert/test_bert_xgb.py",
    ),
    # ---------------------------
    # 5. End-to-End Deep Learning
    # ---------------------------
    (
        "DeepE2E",
        "textcnn",
        "src/pipelines/textcnn/train_textcnn.py",
        "src/pipelines/textcnn/test_textcnn.py",
    ),
    (
        "DeepE2E",
        "char_cnn",
        "src/pipelines/char_cnn/train_char_cnn.py",
        "src/pipelines/char_cnn/test_char_cnn.py",
    ),
    (
        "DeepE2E",
        "cnn_bilstm",
        "src/pipelines/cnn_bilstm/train_cnn_bilstm.py",
        "src/pipelines/cnn_bilstm/test_cnn_bilstm.py",
    ),
    (
        "DeepE2E",
        "lstm_attn",
        "src/pipelines/lstm_attn/train_lstm_attn.py",
        "src/pipelines/lstm_attn/test_lstm_attn.py",
    ),
]


def _build_benchmark_env(threads: int | None, device: str | None):
    env = os.environ.copy()
    if threads is not None:
        env["MLFE_THREADS"] = str(threads)
        env["OMP_NUM_THREADS"] = str(threads)
        env["OPENBLAS_NUM_THREADS"] = str(threads)
        env["MKL_NUM_THREADS"] = str(threads)
        env["VECLIB_MAXIMUM_THREADS"] = str(threads)
        env["NUMEXPR_NUM_THREADS"] = str(threads)
    if device is not None:
        env["MLFE_DEVICE"] = str(device)
    return env


def run_command(script_path, stage_name, feature_name, dataset: str, env):
    """执行单个脚本并记录时间"""
    if not os.path.exists(script_path):
        print(f"[Runner] ❌ Error: Script not found: {script_path}")
        return False

    cmd = [PYTHON_EXEC, script_path, "--dataset", dataset]

    # mgsir 系列需要指定 feature 参数，其他模型通常硬编码在脚本里
    if "mgsir" in feature_name:
        cmd.extend(["--feature", feature_name])

    print(f"\n[Runner] >>> {stage_name}: {feature_name} ...")
    start_t = time.time()

    try:
        subprocess.run(cmd, check=True, env=env)
        duration = time.time() - start_t
        print(f"[Runner] ✅ Success ({duration:.1f}s)")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Runner] ❌ Failed (Exit Code: {e.returncode})")
        return False
    except Exception as e:
        print(f"[Runner] ❌ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Optional: force threads for fair benchmarking (e.g., 1)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional: cpu/gpu/mps/auto for fair benchmarking",
    )
    parser.add_argument(
        "--apply-to",
        type=str,
        choices=["test", "all"],
        default="test",
        help="Apply threads/device to `test` only (default) or to `all` stages (train+test).",
    )
    args = parser.parse_args()

    # Engineering-realistic default: only enforce constraints during testing/benchmarking.
    test_env = _build_benchmark_env(args.threads, args.device)
    train_env = test_env if args.apply_to == "all" else os.environ.copy()

    # 1. 切换到项目根目录
    # [Fix] 使用 current_path 避免 "current_file not defined"
    current_path = Path(__file__).resolve()
    project_root = current_path.parents[1]
    os.chdir(project_root)

    print(f"{'='*60}")
    print(f"🚀 MLFE-SQLi Baseline Runner (Train & Standard Test)")
    print(f"📂 Dataset: {args.dataset}")
    if args.threads is not None:
        print(f"🧵 Threads: {args.threads}")
    if args.device is not None:
        print(f"🧠 Device:  {args.device}")
    print(f"🎯 Apply:   {args.apply_to}")
    print(f"🏠 Root:    {project_root}")
    print(f"📋 Total:   {len(PIPELINES)} models")
    print(f"{'='*60}\n")

    summary = []

    for category, name, train_script, test_script in PIPELINES:
        print(f"\n{'#'*60}")
        print(f"Processing: {name} ({category})")
        print(f"{'#'*60}")

        # Step 1: Training
        train_success = run_command(
            train_script, "Training", name, dataset=args.dataset, env=train_env
        )

        # Step 2: Testing (只有训练成功才跑)
        test_success = False
        if train_success:
            test_success = run_command(
                test_script, "Testing", name, dataset=args.dataset, env=test_env
            )
        else:
            print("[Runner] ⚠️ Skipping Test because Training failed.")

        summary.append(
            {
                "Method": name,
                "Train": "✅" if train_success else "❌",
                "Test": "✅" if test_success else "❌",
            }
        )

    # === 最终报告 ===
    print(f"\n{'='*60}")
    print("🏁 Execution Summary")
    print(f"{'='*60}")
    print(f"{'Method':<25} | {'Train':<5} | {'Test':<5}")
    print("-" * 40)
    for item in summary:
        print(f"{item['Method']:<25} | {item['Train']:<5} | {item['Test']:<5}")
    print(f"{'='*60}")
    print("👉 Check results in: results/metrics/all_results.csv")


if __name__ == "__main__":
    main()
