import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# 1. 数据准备 (命名已更新为与 Methodology 一致)
# ==========================================
# 实验 A: 移除实验 (Removal Impact)
# 对应 LaTeX: w/o H4 (Integrity Check), etc.
labels_rem = [
    'Full MGSIR', 
    'w/o H4\n(Integrity)',      # H4: Integrity Check
    'w/o H3\n(Contextual)',     # H3: Contextual Disruption
    'w/o H2\n(Syntactic)',      # H2: Syntactic Patterns
    'w/o H1\n(Distributional)'  # H1: Distributional Statistics
]
f1_rem = [99.62, 99.58, 99.42, 99.40, 99.36]
delta_rem = [0, -0.04, -0.20, -0.22, -0.26]

# 实验 B: 单层实验 (Individual Capability)
# 对应 LaTeX: H1 Only, etc.
labels_ind = [
    'H1 Only\n(Distributional)', 
    'H3 Only\n(Contextual)', 
    'H2 Only\n(Syntactic)', 
    'H4 Only\n(Integrity)'
]
f1_ind = [99.05, 98.80, 98.69, 74.61]

# ==========================================
# 2. 绘图 (双子图配置)
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5), dpi=300) # 稍微增加高度以容纳换行标签

# -------------------------------------------------------
# 子图 1：Removal Impact (Zoomed in)
# -------------------------------------------------------
colors_rem = ['#d62728'] + ['#1f77b4'] * 4 # 红色强调 Full，其余蓝色
bars1 = ax1.bar(labels_rem, f1_rem, color=colors_rem, alpha=0.85, edgecolor='k', width=0.6)

# 设置 Y 轴范围，放大差异 (99.3 ~ 99.7)
ax1.set_ylim(99.3, 99.68) 
ax1.set_ylabel('F1-Score (%)', fontsize=12, fontweight='bold')
ax1.set_title('Impact of Removing Layers (Ablation)', fontsize=14, fontweight='bold', pad=15)
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# 标注 F1 值和下降幅度
for i, rect in enumerate(bars1):
    height = rect.get_height()
    # 标注数值
    ax1.text(rect.get_x() + rect.get_width()/2., height + 0.005,
             f'{height:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 标注下降量 (Delta)，除了第一个
    if i > 0: 
        # 为了美观，将 delta 写在柱子内部顶部，字体白色
        ax1.text(rect.get_x() + rect.get_width()/2., height - 0.03,
                 f'({delta_rem[i]:.2f})', ha='center', va='top', fontsize=10, color='white', fontweight='bold')

# -------------------------------------------------------
# 子图 2：Individual Capability
# -------------------------------------------------------
colors_ind = ['#2ca02c', '#ff7f0e', '#9467bd', '#7f7f7f'] # 绿, 橙, 紫, 灰
bars2 = ax2.bar(labels_ind, f1_ind, color=colors_ind, alpha=0.85, edgecolor='k', width=0.55)

ax2.set_ylim(70, 105) # 范围大一点，因为 H4 只有 74.61
ax2.set_ylabel('F1-Score (%)', fontsize=12, fontweight='bold')
ax2.set_title('Individual Layer Performance', fontsize=14, fontweight='bold', pad=15)
ax2.grid(axis='y', linestyle='--', alpha=0.5)

# 标注数值
for rect in bars2:
    height = rect.get_height()
    ax2.text(rect.get_x() + rect.get_width()/2., height + 1.0,
             f'{height:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# -------------------------------------------------------
# 3. 通用装饰与保存
# -------------------------------------------------------
# 调整 X 轴标签字体，防止太小
ax1.tick_params(axis='x', labelsize=10)
ax2.tick_params(axis='x', labelsize=10)

plt.tight_layout()

# 保存图片
save_path = 'results/figures/ablation_study.png'
os.makedirs(os.path.dirname(save_path), exist_ok=True)

plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {save_path}")
# plt.show()
