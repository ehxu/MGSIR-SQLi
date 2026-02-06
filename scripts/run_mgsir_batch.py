# scripts/run_mgsir_batch.py
import subprocess
import sys
import time
import shutil
from pathlib import Path
import os
import argparse

# ================= 配置区域 =================
# 指定你要使用的数据集名称
DEFAULT_DATASET_NAME = "dataset1"
DATASET_NAME = DEFAULT_DATASET_NAME

# 指定 Python 解释器路径 (确保使用当前环境)
PYTHON_EXEC = sys.executable

# Checkpoints 相对路径 (用于清理)
CHECKPOINTS_DIR = Path("results/checkpoints")


# ================= 任务定义辅助函数 =================
def make_task(script_path: str, feature: str, description: str):
    """
    快速生成任务配置字典
    :param script_path: 脚本相对于项目根目录的路径
    :param feature: 特征/Pipeline 名称
    :param description: 打印在控制台的描述信息
    """
    return {
        "script": script_path,
        "feature": feature,
        "desc": description,
        "args": ["--dataset", DATASET_NAME, "--feature", feature],
    }


# ================= 定义任务列表 =================
# 注意：这里调用的都是我们刚刚重构过的入口脚本
TASKS = [
    # --- Training Phase ---
    make_task(
        "src/pipelines/mgsir/train_mgsir_full.py",
        "mgsir_xgb",
        "🔥 Training [Enhanced] (Core: train_core)",
    ),
    # --- Testing Phase ---
    make_task(
        "src/pipelines/mgsir/test_mgsir_full.py",
        "mgsir_xgb",
        "🧪 Testing [Enhancede] (Core: test_core)",
    ),
]

# ================= 功能函数 =================


def clean_checkpoints():
    """执行前清空 checkpoints 文件夹，防止旧模型干扰"""
    target_dir = Path.cwd() / CHECKPOINTS_DIR

    print(f"\n{'='*80}")
    print(f"🧹 [Clean] 正在清理 Checkpoints 目录: {target_dir}")

    try:
        if target_dir.exists():
            shutil.rmtree(target_dir)
            print(f"✅ [Deleted] 旧目录已删除")

        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ [Created] 新目录已创建")

    except Exception as e:
        print(f"❌ [Error] 清空目录失败: {e}")
        sys.exit(1)

    print(f"{'='*80}\n")


def run_command(task, index, total):
    """运行单个任务"""
    script_path = Path(task["script"]).resolve()
    description = task["desc"]
    args = task["args"]

    # 进度条头部
    print(f"👉 [Task {index}/{total}] {description}")
    print(f"   Script: {task['script']}")

    if not script_path.exists():
        print(f"❌ [Error] 找不到脚本文件: {script_path}")
        return False

    start_time = time.time()
    cmd = [PYTHON_EXEC, str(script_path)] + args

    try:
        # check=True: 如果脚本返回非0状态码，抛出 CalledProcessError
        # 这里不捕获 stdout，让子脚本的日志直接打印到终端，方便看进度
        subprocess.run(cmd, check=True, env=task.get("env") or None)

        duration = time.time() - start_time
        print(f"✅ [Done] 耗时: {duration:.2f}s")
        print(f"{'-'*80}")  # 分隔线
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n❌ [Failed] 任务执行失败 (Exit Code: {e.returncode})")
        return False
    except KeyboardInterrupt:
        print("\n🛑 [Aborted] 用户手动停止。")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ [Error] 未知错误: {e}")
        return False


# ================= 主流程 =================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_NAME)
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

    test_env = os.environ.copy()
    if args.threads is not None:
        test_env["MLFE_THREADS"] = str(args.threads)
        test_env["OMP_NUM_THREADS"] = str(args.threads)
        test_env["OPENBLAS_NUM_THREADS"] = str(args.threads)
        test_env["MKL_NUM_THREADS"] = str(args.threads)
        test_env["VECLIB_MAXIMUM_THREADS"] = str(args.threads)
        test_env["NUMEXPR_NUM_THREADS"] = str(args.threads)
    if args.device is not None:
        test_env["MLFE_DEVICE"] = str(args.device)
    train_env = test_env if args.apply_to == "all" else os.environ.copy()

    # patch dataset into tasks
    for t in TASKS:
        # args are like: ["--dataset", DATASET_NAME, "--feature", ...]
        for i in range(len(t["args"]) - 1):
            if t["args"][i] == "--dataset":
                t["args"][i + 1] = args.dataset
        # Train/Test scripts are mixed in TASKS; decide by script name.
        if "/train_" in t["script"]:
            t["env"] = train_env
        else:
            t["env"] = test_env

    total_start = time.time()

    print(f"\n🚀 开始批量执行 mgsir Pipeline")
    print(f"📂 项目根目录: {Path.cwd()}")
    print(f"📊 数据集名称: {args.dataset}")
    if args.threads is not None:
        print(f"🧵 Threads: {args.threads}")
    if args.device is not None:
        print(f"🧠 Device:  {args.device}")
    print(f"🎯 Apply:   {args.apply_to}")

    # 1. 清理旧模型
    # clean_checkpoints()

    success_count = 0
    total_tasks = len(TASKS)

    # 2. 循环执行任务
    for i, task in enumerate(TASKS, 1):
        success = run_command(task, i, total_tasks)

        if not success:
            print(f"\n{'!'*80}")
            print("⚠️  批处理流程因错误而终止。")
            print("   请检查上方错误日志，修复后重试。")
            print(f"{'!'*80}")
            sys.exit(1)

        success_count += 1

    total_end = time.time()
    total_duration = total_end - total_start

    print(f"\n{'#'*80}")
    print(f"🎉 所有任务执行完毕! ({success_count}/{total_tasks})")
    print(f"⏱️  总耗时: {total_duration:.2f}s")
    print(f"📂 结果请查看 results/logs/ 和 results/checkpoints/")
    print(f"{'#'*80}")


if __name__ == "__main__":
    main()
