# scripts/plots/plot_efficiency_tradeoff.py
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as ticker

# === 环境配置 ===
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config.paths import GLOBAL_RESULTS_CSV

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as ticker
from adjustText import adjust_text

# ... (环境配置代码保持不变) ...
from src.config.paths import GLOBAL_RESULTS_CSV

# === 绘图风格设置 (更偏向学术出版) ===
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
# 如果系统支持，可以开启 LaTeX 渲染 (可选，需要本地有 LaTeX 环境)
# plt.rcParams['text.usetex'] = True
# plt.rcParams['font.family'] = 'serif' 

def plot_tradeoff_publication(csv_path, output_path):
    # 1. 加载与清洗数据 (逻辑保持不变)
    if not csv_path.exists(): return
    df = pd.read_csv(csv_path)
    df = df[(df["latency_avg_ms"] > 0) & (df["f1"] > 0) & (~df["notes"].str.contains("Custom", na=False))].copy()
    
    def shorten_name(name):
        name = name.replace("_enhanced_Full", "_enhanced")
        name = name.replace("_xgb", "").replace("_enhanced", "+")
        name = name.replace("mgsir", "MGSIR").replace("textcnn", "TextCNN")
        name = name.replace("char_cnn", "CharCNN").replace("bert", "BERT")
        name = name.replace("lstm_attn", "LSTM+Attn")
        name = name.replace("cnn_bilstm", "CNN+BiLSTM")
        if "L1" in name or "No_" in name or "only" in name: return None
        return name

    df["short_name"] = df["method"].apply(shorten_name)
    df = df.dropna(subset=["short_name"]).drop_duplicates(subset=["short_name"], keep="first")

    # === 2. 关键修改：定义样式映射 ===
    
    # 区分“你的方法”和“其他方法”
    # 假设你的方法包含 "mgsir"
    df['type'] = df['short_name'].apply(lambda x: 'Ours' if 'mgsir' in x else 'Baselines')
    
    # 定义颜色：主角用红色，配角用冷色/灰色
    # 注意：这里我们创建一个字典来具体指定每个点的颜色，或者利用 hue
    # 为了更精细控制，我们手动构建调色板
    
    # 获取所有名字并排序，确保 mgsir 在最后绘制（防止被遮挡）
    df = df.sort_values(by='type', ascending=True) 
    
    unique_names = df['short_name'].unique()
    palette = {}
    markers = {}
    
    for name in unique_names:
        if "mgsir" in name:
            palette[name] = "#D62728"  # Tab:Red (鲜艳红)
            markers[name] = "*"        # 星号
        elif "BERT" in name or "CNN" in name or "LSTM" in name:
            palette[name] = "#1F77B4"  # Tab:Blue (深蓝 - 深度学习)
            markers[name] = "s"        # 方块
        else:
            palette[name] = "#7F7F7F"  # Tab:Gray (灰色 - 传统机器学习)
            markers[name] = "o"        # 圆点

    # 3. 绘图
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300) # 调整为适合论文双栏或半页的尺寸

    # 气泡大小
    bubble_sizes = np.sqrt(df["model_size_mb"]) * 60 + 30

    sns.scatterplot(
        data=df,
        x="latency_avg_ms",
        y="f1",
        size="model_size_mb",
        sizes=(bubble_sizes.min(), bubble_sizes.max()),
        hue="short_name",
        style="short_name", # 同时映射形状
        markers=markers,    # 使用自定义形状
        palette=palette,    # 使用自定义颜色
        alpha=0.9,
        edgecolor="k",      # 黑色描边，增加清晰度
        linewidth=1,
        legend=False,       # 手动添加图例
        ax=ax,
        zorder=5
    )

    # 4. 坐标轴与网格
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:g}'))
    
    # 轴标签：加粗，清晰
    ax.set_xlabel("Inference Latency (ms) [Log Scale]", fontweight="bold")
    ax.set_ylabel("F1-Score", fontweight="bold")
    
    # 标题：论文里通常不需要标题（因为有 Caption），或者写得非常简洁
    # 这里我们保留一个简洁的
    # ax.set_title("Accuracy vs. Efficiency Trade-off", fontweight="bold", pad=15)

    # 5. 理想区域 (更微妙的画法)
    # 画一个指向左上角的箭头
    ax.annotate('Ideal Zone\n(High Acc & Low Latency)', 
            xy=(0.02, 0.98), xycoords='axes fraction',
            xytext=(0.15, 0.90), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
            horizontalalignment='left', verticalalignment='top',
            fontsize=11, color='#333333', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    # 6. 标签 (Smart Labeling)
    texts = []
    for i, row in df.iterrows():
        # mgsir 加粗放大
        is_ours = "mgsir" in row["short_name"]
        weight = "bold" if is_ours else "normal"
        size = 12 if is_ours else 10
        color = "#b30000" if is_ours else "#333333" # 红色强调文字或深灰
        
        texts.append(
            ax.text(
                row["latency_avg_ms"],
                row["f1"],
                row["short_name"],
                fontsize=size,
                fontweight=weight,
                color=color
            )
        )

    adjust_text(texts, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5), ax=ax)

    # 7. 手动构建图例 (模型大小 + 方法类型)
    # 大小图例
    sizes_legend = [0.1, 10, 100, 400]
    legend_elements = []
    for s_val in sizes_legend:
        s_size = np.sqrt(s_val) * 60 + 30
        legend_elements.append(plt.scatter([], [], s=s_size, c='gray', alpha=0.5, edgecolor='k', label=f'{s_val} MB'))
    
    # 添加形状图例说明 (可选，或者直接在 Caption 里写)
    # 这里我们只放大小图例，因为名字已经在图上标了，形状的区别是辅助性的
    
    leg = ax.legend(handles=legend_elements, title="Model Size", loc="lower right", 
                    frameon=True, framealpha=0.9, edgecolor="gray", fontsize=10)
    leg.get_title().set_fontweight('bold')

    # 布局优化
    plt.tight_layout()
    plt.grid(True, which="major", ls="--", alpha=0.5)
    
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=600, bbox_inches="tight") # 600 dpi for print
    print(f"[Success] Publication-ready plot saved to: {output_path}")

# ... (Main函数保持不变)
def main():
    csv_path = GLOBAL_RESULTS_CSV
    output_path = project_root / "results" / "figures" / "efficiency_tradeoff_optimized.png"

    # 确保在使用完整数据运行前 CSV 存在
    if not csv_path.exists():
        print(f"CSV file not found at {csv_path}. Create dummy data for testing? (Y/N)")
        # (此处省略了创建虚拟数据的代码，直接假设 CSV 存在)
        return

    plot_tradeoff_publication(csv_path, output_path)


if __name__ == "__main__":
    main()