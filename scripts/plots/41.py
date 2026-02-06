import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re
import os

# ================= 配置部分 =================
LATEX_FILE_PATH = 'results/latex_tables/table1_overall_results.tex'
OUTPUT_IMG_PATH = 'fpr_fnr_tradeoff.png'

# 设置学术风格字体
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12

def parse_latex_table(filepath):
    """
    读取 LaTeX 表格并提取 Method, FNR, FPR
    假设表格列顺序: Method & Acc & Prec & Recall & F1 & FNR & FPR
    """
    methods = []
    fnrs = []
    fprs = []
    
    # 如果文件不存在，先创建一个示例文件供测试 (方便您直接运行)
    if not os.path.exists(filepath):
        print(f"[Info] 文件 {filepath} 不存在，正在创建示例文件...")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(r"""
\begin{table*}[!ht]
\centering
\caption{Overall Performance Comparison}
\label{tab:overall_results}
\begin{tabular}{lcccccc}
\toprule
Method & Accuracy (\%) & Precision (\%) & Recall/TPR (\%) & F1-Score (\%) & FNR (\%) & FPR (\%) \\
\midrule
BoW & 99.42 & 99.78 & 98.63 & 99.20 & 1.37 & 0.13 \\
TF-IDF & 99.37 & 99.64 & 98.63 & 99.13 & 1.37 & 0.20 \\
Word2Vec & 99.29 & 99.29 & 98.76 & 99.03 & 1.24 & 0.41 \\
FastText & 99.45 & 99.78 & 98.72 & 99.25 & 1.28 & 0.13 \\
TextCNN & 99.51 & 99.78 & 98.90 & 99.33 & 1.10 & 0.13 \\
Char-CNN & 99.84 & 99.78 & 99.78 & 99.78 & 0.22 & 0.13 \\
BiLSTM-Attn & 98.99 & 98.93 & 98.32 & 98.63 & 1.68 & 0.61 \\
CNN-BiLSTM & 99.38 & 99.34 & 98.98 & 99.16 & 1.02 & 0.38 \\
BERT & 99.48 & 99.87 & 98.72 & 99.29 & 1.28 & 0.08 \\
MGSIR & 99.72 & 99.96 & 99.29 & 99.62 & 0.71 & 0.03 \\
\bottomrule
\end{tabular}
\end{table*}
            """)

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"[Processing] Reading from {filepath}...")
    
    for line in lines:
        line = line.strip()
        # 跳过注释、命令、表头等非数据行
        if line.startswith('%') or line.startswith('\\') or not '&' in line:
            continue
        if 'Accuracy' in line or 'Method' in line: # 跳过标题行
            continue
            
        # 分割并清洗数据
        parts = [p.strip() for p in line.split('&')]
        
        # 确保列数足够 (您的表格有 7 列)
        if len(parts) >= 7:
            try:
                # 提取 Method (Col 0)
                method_name = parts[0].replace('\\', '').strip()
                
                # 提取 FNR (Col 5) 和 FPR (Col 6) -> 去掉可能的 \\ 和空格
                fnr_val = float(re.sub(r'[^\d\.]', '', parts[5]))
                fpr_val = float(re.sub(r'[^\d\.]', '', parts[6]))
                
                methods.append(method_name)
                fnrs.append(fnr_val)
                fprs.append(fpr_val)
                print(f"  Parsed: {method_name} -> FNR={fnr_val}, FPR={fpr_val}")
            except ValueError:
                continue
                
    return methods, fnrs, fprs

def plot_tradeoff():
    # 1. 获取数据
    models, fnr_data, fpr_data = parse_latex_table(LATEX_FILE_PATH)
    
    if not models:
        print("[Error] 未找到数据，请检查 LaTeX 文件格式")
        return

    # 2. 绘图设置
    x = np.arange(len(models))  # 标签位置
    width = 0.35  # 柱状图宽度

    # 加宽图片以容纳所有标签
    fig, ax = plt.subplots(figsize=(14, 6))

    # 3. 绘制柱状图
    # FPR (红色 - 误报)
    rects1 = ax.bar(x - width/2, fpr_data, width, label='FPR (False Positive Rate)', color='#D62728', alpha=0.85)
    # FNR (蓝色 - 漏报)
    rects2 = ax.bar(x + width/2, fnr_data, width, label='FNR (False Negative Rate)', color='#1F77B4', alpha=0.85)

    # 4. 数值标签函数
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            # 对于很小的值 (如 0.03)，标签稍微拿高一点防止重叠
            offset = 5 if height < 0.2 else 3
            ax.annotate(f'{height:.2f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, offset),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    # 5. 坐标轴与图例
    ax.set_ylabel('Error Rate (%) - Lower is Better', fontweight='bold', fontsize=12)
    ax.set_title('FPR & FNR Trade-off Analysis', fontsize=14, fontweight='bold', pad=20)
    
    ax.set_xticks(x)
    # 关键修改：旋转 30 度，防止标签重叠
    ax.set_xticklabels(models, rotation=30, ha='right', fontweight='medium', fontsize=11)
    
    # 调整 Y 轴范围 (留出顶部空间)
    ax.set_ylim(0, max(max(fnr_data), max(fpr_data)) * 1.2)
    
    ax.legend(loc='upper right', frameon=True, fontsize=11)

    # 6. 高亮注释 (MGSIR FPR)
    # 找到 MGSIR 的索引
    if 'MGSIR' in models:
        idx = models.index('MGSIR')
        target_x = idx - width/2
        target_y = fpr_data[idx]
        
        ax.annotate('Lowest FPR (0.03%)\nOperational Optimal', 
                    xy=(target_x, target_y), 
                    xytext=(target_x - 0.4, 1.45), 
                    # 调整箭头样式，使其从上方优雅地指下来
                    arrowprops=dict(facecolor='#333333', shrink=0.05, width=1.5, headwidth=8, 
                                    connectionstyle="arc3,rad=0.2"), # rad负值让弧线反向，更自然
                    fontsize=10, color='#D62728', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#D62728", alpha=0.9))

    # 7. 布局优化并保存
    sns.despine()
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG_PATH, dpi=300)
    print(f"[Success] 图表已保存至: {OUTPUT_IMG_PATH}")

if __name__ == "__main__":
    plot_tradeoff()