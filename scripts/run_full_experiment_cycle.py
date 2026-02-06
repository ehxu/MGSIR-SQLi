# scripts/run_full_experiment_cycle.py
import subprocess
import sys
import time
import os
from pathlib import Path
import argparse

# === 颜色定义 ===
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

# === 解释器 ===
PYTHON_EXEC = sys.executable

# === 定义实验顺序 ===
# 格式: (描述, 脚本文件名)
STAGES = [
    ("Stage 1: Baseline Training & Standard Testing", "scripts/run_all_baselines.py"),
    (
        "Stage 2: Ablation Study (mgsir Enhanced)",
        "scripts/run_mgsir_ablation.py",
    ),
    ("Stage 3: Adversarial Robustness Benchmark", "scripts/run_all_adversarial.py"),
]


def run_stage(description, script_rel_path):
    print(f"\n{CYAN}{'='*80}{RESET}")
    print(f"{CYAN}🚀 STARTING: {description}{RESET}")
    print(f"{CYAN}📜 Script:   {script_rel_path}{RESET}")
    print(f"{CYAN}{'='*80}{RESET}\n")

    # 确保路径存在
    if not os.path.exists(script_rel_path):
        print(f"{RED}[Error] Script not found: {script_rel_path}{RESET}")
        return False

    start_t = time.time()
    try:
        # 实时流式输出，check=True 遇到错误会抛出异常
        subprocess.run([PYTHON_EXEC, script_rel_path], check=True)
        duration = time.time() - start_t
        print(f"\n{GREEN}✅ FINISHED: {description} (Time: {duration:.2f}s){RESET}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n{RED}❌ FAILED: {description} (Exit Code: {e.returncode}){RESET}")
        return False
    except Exception as e:
        print(f"\n{RED}❌ ERROR: {e}{RESET}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dataset1")
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

    env = os.environ.copy()
    if args.threads is not None:
        env["MLFE_THREADS"] = str(args.threads)
        env["OMP_NUM_THREADS"] = str(args.threads)
        env["OPENBLAS_NUM_THREADS"] = str(args.threads)
        env["MKL_NUM_THREADS"] = str(args.threads)
        env["VECLIB_MAXIMUM_THREADS"] = str(args.threads)
        env["NUMEXPR_NUM_THREADS"] = str(args.threads)
    if args.device is not None:
        env["MLFE_DEVICE"] = str(args.device)

    # 1. 切换到项目根目录 (MLFE-SQLi)
    current_path = Path(__file__).resolve()
    project_root = current_path.parents[1]
    os.chdir(project_root)

    print(f"{GREEN}🎉 Welcome to MGSIR SQLi Detection System {RESET}")
    print(
        f"{YELLOW}⚠️  Note: Make sure you are in the project root or scripts folder.{RESET}"
    )
    print(f"{YELLOW}📂 Working Directory set to: {project_root}{RESET}")

    total_start = time.time()

    # 2. 依次执行
    for desc, script in STAGES:
        # pass dataset/benchmark settings down to sub-runners
        cmd = [PYTHON_EXEC, script, "--dataset", args.dataset]
        if args.threads is not None:
            cmd += ["--threads", str(args.threads)]
        if args.device is not None:
            cmd += ["--device", str(args.device)]
        cmd += ["--apply-to", str(args.apply_to)]

        print(f"\n{CYAN}{'='*80}{RESET}")
        print(f"{CYAN}🚀 STARTING: {desc}{RESET}")
        print(f"{CYAN}📜 Script:   {script}{RESET}")
        print(f"{CYAN}{'='*80}{RESET}\n")

        start_t = time.time()
        try:
            subprocess.run(cmd, check=True, env=env)
            duration = time.time() - start_t
            print(f"\n{GREEN}✅ FINISHED: {desc} (Time: {duration:.2f}s){RESET}")
            success = True
        except subprocess.CalledProcessError as e:
            print(f"\n{RED}❌ FAILED: {desc} (Exit Code: {e.returncode}){RESET}")
            success = False
        except Exception as e:
            print(f"\n{RED}❌ ERROR: {e}{RESET}")
            success = False

        if not success:
            print(f"\n{RED}⛔ Pipeline halted due to failure in: {desc}{RESET}")
            sys.exit(1)

    total_duration = time.time() - total_start

    print(f"\n{GREEN}{'='*80}{RESET}")
    print(f"{GREEN}🎉🎉🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY! 🎉🎉🎉{RESET}")
    print(f"{GREEN}⏱️  Total Time: {total_duration/60:.2f} minutes{RESET}")
    print(f"{GREEN}📄 Check results in: results/metrics/all_results.csv{RESET}")
    print(f"{GREEN}{'='*80}{RESET}")


if __name__ == "__main__":
    main()
