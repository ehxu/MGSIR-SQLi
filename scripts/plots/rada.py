import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path

# ==========================================
# 1. 配置路径与模型
# ==========================================

# 注意：请确保此路径指向包含 Robustness (Recall) 数据的表格
# 虽然您提示中写了 table3_efficiency，但雷达图需要的是 robustness 数据
# 建议确认文件名是否为 table2_robustness.tex 或类似名称
TEX_FILE_PATH = Path("results/latex_tables/table2_robustness.tex") 

# 精选的 5 个代表性模型及其样式配置
SELECTED_MODELS = {
    'MGSIR':   {'color': '#d62728', 'width': 3.0, 'style': '-',  'marker': 'o', 'label': 'MGSIR (Ours)'},
    'Char-CNN': {'color': '#9467bd', 'width': 2.0, 'style': '--', 'marker': 's', 'label': 'Char-CNN'},
    'BERT':     {'color': '#1f77b4', 'width': 2.0, 'style': '-.', 'marker': '^', 'label': 'BERT'},
    'TextCNN':  {'color': '#ff7f0e', 'width': 1.5, 'style': ':',  'marker': 'x', 'label': 'TextCNN'},
    'Word2Vec': {'color': '#2ca02c', 'width': 1.5, 'style': ':',  'marker': 'd', 'label': 'Word2Vec'}
}

# ==========================================
# 2. LaTeX 解析函数
# ==========================================
def parse_latex_table(file_path):
    if not file_path.exists():
        print(f"[Error] File not found: {file_path}")
        # 为了演示，这里如果没有文件，返回一个空数据或抛出异常
        return [], {}

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    header_indices = {}
    labels = []
    data_dict = {} # {ModelName: [val1, val2...]}

    parsing_data = False
    
    for line in lines:
        line = line.strip()
        
        # 1. 识别表头 (Attack, BoW, TF-IDF...)
        if "Attack" in line and "&" in line and not header_indices:
            # 去除 LaTeX 命令如 \textbf{}
            clean_line = re.sub(r'\\textbf\{([^}]*)\}', r'\1', line)
            parts = [p.strip().replace(r'\\', '') for p in clean_line.split('&')]
            
            # 记录每个模型在表格中的列索引
            for idx, col_name in enumerate(parts):
                col_name = col_name.strip()
                if col_name != "Attack":
                    header_indices[idx] = col_name
                    data_dict[col_name] = []
            continue

        # 2. 识别数据行开始
        if r"\midrule" in line:
            parsing_data = True
            continue

        # 3. 识别结束
        if r"\bottomrule" in line or "Average" in line:
            parsing_data = False
            continue

        # 4. 解析数据行
        if parsing_data and "&" in line:
            parts = [p.strip().replace(r'\\', '') for p in line.split('&')]
            
            # 第一列是攻击名称 (Label)
            attack_name = parts[0]
            labels.append(attack_name)
            
            # 后续列是数值
            for idx, val_str in enumerate(parts):
                if idx in header_indices:
                    model_name = header_indices[idx]
                    try:
                        val = float(val_str)
                    except ValueError:
                        val = 0.0
                    data_dict[model_name].append(val)

    return labels, data_dict

# ==========================================
# 3. 绘图主逻辑
# ==========================================
def plot_radar(labels, full_data, output_path):
    # 准备闭环数据
    def make_closed(data):
        return np.concatenate((data, [data[0]]))

    num_vars = len(labels)
    if num_vars == 0:
        print("[Error] No data parsed. Please check the .tex file content.")
        return

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] 

    # 创建画布
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

    # 绘图顺序：先画表现差的，最后画 MGSIR，防止遮挡
    plot_order = ['TextCNN', 'Word2Vec', 'BERT', 'Char-CNN', 'MGSIR']

    for model_key in plot_order:
        # 如果 LaTeX 表格里的名字和 key 不完全一致，需要做个映射或模糊匹配
        # 这里假设 LaTeX 表格里的名字就是 'TextCNN', 'MGSIR' 等
        
        # 尝试查找数据
        target_data = None
        for data_key in full_data.keys():
            if model_key in data_key: # 简单匹配，例如 'MGSIR' 匹配 'MGSIR'
                target_data = full_data[data_key]
                break
        
        if target_data:
            values = make_closed(target_data)
            style = SELECTED_MODELS[model_key]
            
            ax.plot(angles, values, 
                    color=style['color'], 
                    linewidth=style['width'], 
                    linestyle=style['style'], 
                    label=style['label'], 
                    marker=style['marker'], 
                    markersize=5)
            
            # 只有 MGSIR 填充颜色
            if model_key == 'MGSIR':
                ax.fill(angles, values, color=style['color'], alpha=0.15)
        else:
            print(f"[Warning] Model '{model_key}' not found in LaTeX table columns.")

    # 装饰
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=11, weight='bold')

    ax.set_rlabel_position(0)
    plt.yticks([20, 40, 60, 80, 100], ["20%", "40%", "60%", "80%", "100%"], color="grey", size=9)
    plt.ylim(0, 105)

    # 图例
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10, frameon=True, shadow=True)
    plt.title('Robustness Defense Perimeter (Recall %)', size=16, weight='bold', y=1.08)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[Success] Radar chart saved to {output_path}")

# ==========================================
# 4. 执行
# ==========================================
if __name__ == "__main__":
    # 解析
    print(f"Reading from: {TEX_FILE_PATH}")
    labels, full_data = parse_latex_table(TEX_FILE_PATH)
    
    # 绘图
    output_img = Path("results/figures/radar_chart_selected.png")
    output_img.parent.mkdir(parents=True, exist_ok=True)
    
    if labels:
        plot_radar(labels, full_data, output_img)