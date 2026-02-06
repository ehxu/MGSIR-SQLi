import subprocess
import time
import sys
import os
import argparse

PYTHON_EXEC = sys.executable

# === 定义你要跑的消融实验列表 ===
ABLATION_MODES = [
    # 1. Additive (累加)
    "L1",
    "L1_L2",
    "L1_L2_L3",
    "Full",
    # 2. Subtractive (减法 - 验证缺失某层的影响)
    "No_L4",  # (可选: 效果等于 L1_L2_L3，不想重复跑可以注释掉)
    "No_L3",
    "No_L2",
    "No_L1",
    # 3. Individual (独立 - 验证单层能力)
    "L1_only",  # (可选: 效果等于 L1)
    "L2_only",
    "L3_only",
    "L4_only",
]

DEFAULT_DATASET = "dataset1"


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


def run_cmd(script, mode, dataset: str, env):
    cmd = [PYTHON_EXEC, script, "--dataset", dataset, "--mode", mode]
    print(f"\n[Batch] Running: {' '.join(cmd)}")
    subprocess.check_call(cmd, env=env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--apply-to",
        type=str,
        choices=["test", "all"],
        default="test",
        help="Apply threads/device to `test` only (default) or to `all` stages (train+test).",
    )
    args = parser.parse_args()

    test_env = _build_benchmark_env(args.threads, args.device)
    train_env = test_env if args.apply_to == "all" else os.environ.copy()

    total_start = time.time()
    print(f"Starting Batch Ablation for {len(ABLATION_MODES)} modes...")

    for mode in ABLATION_MODES:
        print(f"\n{'#'*50}")
        print(f" >>> Processing Ablation Mode: {mode}")
        print(f"{'#'*50}")

        try:
            # 1. Train
            run_cmd(
                "src/pipelines/mgsir/train_ablation_core.py",
                mode,
                dataset=args.dataset,
                env=train_env,
            )

            # 2. Test
            run_cmd(
                "src/pipelines/mgsir/test_ablation_core.py",
                mode,
                dataset=args.dataset,
                env=test_env,
            )

        except subprocess.CalledProcessError:
            print(f"[Error] Failed at mode {mode}. Stopping batch.")
            break

    print(f"\n[Batch] All ablations finished in {time.time() - total_start:.2f}s")


if __name__ == "__main__":
    main()
