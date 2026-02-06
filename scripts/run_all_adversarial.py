# scripts/run_all_adversarial.py
import subprocess
import sys
import time
import os
from pathlib import Path
import argparse

# === 全局配置 ===
DEFAULT_DATASET = "dataset1"
PYTHON_EXEC = sys.executable

# === Adversarial Pipeline 定义 ===
# 格式: (Category, FeatureName, AdversarialScript)
# 顺序严格对齐 run_all_baselines.py
PIPELINES = [
    # ---------------------------
    # 1. Ours
    # ---------------------------
    (
        "Ours",
        "mgsir_xgb",
        "src/pipelines/mgsir/test_adversarial_mgsir.py",
    ),
    # (
    #     "Ours",
    #     "mgsir_xgb",
    #     "src/pipelines/mgsir/test_adversarial_mgsir.py",
    # ),
    # ---------------------------
    # 2. Shallow NLP
    # ---------------------------
    (
        "Shallow",
        "bow_xgb",
        "src/pipelines/bow/test_bow_xgb_adversarial.py",
    ),
    (
        "Shallow",
        "tfidf_xgb",
        "src/pipelines/tfidf/test_tfidf_xgb_adversarial.py",
    ),
    # ---------------------------
    # 3. Vector Embeddings
    # ---------------------------
    (
        "Vector",
        "w2v_xgb",
        "src/pipelines/w2v/test_w2v_xgb_adversarial.py",
    ),
    (
        "Vector",
        "fasttext_xgb",
        "src/pipelines/fasttext/test_fasttext_xgb_adversarial.py",
    ),
    # ---------------------------
    # 4. Deep Features
    # ---------------------------
    (
        "DeepFeat",
        "bert_xgb",
        "src/pipelines/bert/test_bert_xgb_adversarial.py",
    ),
    # ---------------------------
    # 5. End-to-End Deep Learning
    # ---------------------------
    (
        "DeepE2E",
        "textcnn",
        "src/pipelines/textcnn/test_textcnn_adversarial.py",
    ),
    (
        "DeepE2E",
        "char_cnn",
        "src/pipelines/char_cnn/test_char_cnn_adversarial.py",
    ),
    (
        "DeepE2E",
        "cnn_bilstm",
        "src/pipelines/cnn_bilstm/test_cnn_bilstm_adversarial.py",
    ),
    (
        "DeepE2E",
        "lstm_attn",
        "src/pipelines/lstm_attn/test_lstm_attn_adversarial.py",
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


def run_adv_command(script_path, feature_name, dataset: str, env):
    """执行单个对抗性测试脚本"""
    if not os.path.exists(script_path):
        print(f"[Runner] ❌ Error: Script not found: {script_path}")
        return False

    # 显式传递 --feature 参数，确保加载正确的模型（特别是针对 mgsir 和 shared scripts）
    cmd = [PYTHON_EXEC, script_path, "--dataset", dataset, "--feature", feature_name]

    print(f"\n[Runner] >>> Adversarial Benchmark: {feature_name} ...")
    start_t = time.time()

    try:
        # 实时输出子进程日志
        subprocess.run(cmd, check=True, env=env)
        duration = time.time() - start_t
        print(f"[Runner] ✅ Finished {feature_name} ({duration:.1f}s)")
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
        help="Compatibility flag (adversarial runner only performs testing).",
    )
    args = parser.parse_args()

    bench_env = _build_benchmark_env(args.threads, args.device)

    # 1. 切换到项目根目录
    current_path = Path(__file__).resolve()
    project_root = current_path.parents[1]
    os.chdir(project_root)

    print(f"{'='*60}")
    print(f"⚔️  MLFE-SQLi Adversarial Robustness Runner ⚔️")
    print(f"📂 Dataset: {args.dataset}")
    if args.threads is not None:
        print(f"🧵 Threads: {args.threads}")
    if args.device is not None:
        print(f"🧠 Device:  {args.device}")
    print(f"🏠 Root:    {project_root}")
    print(f"📋 Targets: {len(PIPELINES)} models")
    print(f"{'='*60}\n")

    summary = []

    for category, name, adv_script in PIPELINES:
        print(f"\n{'#'*60}")
        print(f"Testing Model: {name} ({category})")
        print(f"{'#'*60}")

        success = run_adv_command(adv_script, name, dataset=args.dataset, env=bench_env)

        summary.append({"Method": name, "Status": "✅ Pass" if success else "❌ Fail"})

    # === 最终报告 ===
    print(f"\n{'='*60}")
    print("🏁 Adversarial Execution Summary")
    print(f"{'='*60}")
    print(f"{'Method':<25} | {'Status':<10}")
    print("-" * 40)
    for item in summary:
        print(f"{item['Method']:<25} | {item['Status']:<10}")
    print(f"{'='*60}")
    print("👉 Check results in: results/metrics/all_results.csv")


if __name__ == "__main__":
    main()
