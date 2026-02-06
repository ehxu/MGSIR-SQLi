# src/visualization/metric_plots.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score


# 绘制单个模型的 Acc/Prec/Rec/F1 柱状图
def plot_metrics_single(metrics_dict):

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "SimSun",
                "Noto Serif CJK SC",
                "WenQuanYi Micro Hei",
                "DejaVu Serif",
                "Liberation Serif",
            ],
            "mathtext.fontset": "stix",
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.dpi": 150,
            "figure.figsize": (8, 6),
            "savefig.bbox": "tight",
        }
    )

    # 创建图形和坐标轴
    fig, ax = plt.subplots()

    # 指标标签
    metrics_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    x = np.arange(len(metrics_labels))  # 指标位置
    values = [
        metrics_dict["acc"],
        metrics_dict["prec"],
        metrics_dict["rec"],
        metrics_dict["f1"],
    ]

    # 创建颜色映射 (从蓝色到绿色)
    colors = plt.cm.Blues(np.linspace(0.6, 1, len(metrics_labels)))

    # 绘制条形图
    bars = ax.bar(x, values, width=0.7, color=colors, edgecolor="black", linewidth=0.7)

    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.annotate(
            f"{height:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 垂直偏移
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # 设置标题和标签
    ax.set_title("Classification Metrics", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_labels)
    ax.set_ylim(0, 1.15)  # 留出空间给标签

    # 添加脚注
    plt.figtext(
        0.5, 0.01, "Model Performance Metrics", ha="center", fontsize=10, style="italic"
    )

    # 展示
    plt.tight_layout()
    # plt.show()


def save_confusion_matrix(true_y, pred_y, save_path, title_suffix=""):
    cm = confusion_matrix(true_y, pred_y)
    # [Robustness] Handle division by zero if class counts are 0
    with np.errstate(divide="ignore", invalid="ignore"):
        recall_m = (cm.T / cm.sum(axis=1)).T
        precision_m = cm / cm.sum(axis=0)

    # Replace NaNs with 0 (case where a class is missing in GT or Pred)
    recall_m = np.nan_to_num(recall_m)
    precision_m = np.nan_to_num(precision_m)

    labels = [0, 1]
    plt.figure(figsize=(18, 5))
    cmap = sns.light_palette("blue", as_cmap=True)

    # Confusion Matrix
    plt.subplot(1, 3, 1)
    sns.heatmap(
        cm, annot=True, cmap=cmap, fmt="d", xticklabels=labels, yticklabels=labels
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix {title_suffix}")

    # Precision Matrix
    plt.subplot(1, 3, 2)
    sns.heatmap(
        precision_m,
        annot=True,
        cmap=cmap,
        fmt=".3f",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Precision Matrix {title_suffix}")

    # Recall Matrix
    plt.subplot(1, 3, 3)
    sns.heatmap(
        recall_m,
        annot=True,
        cmap=cmap,
        fmt=".3f",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Recall Matrix {title_suffix}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def compute_plot_auc(y_true, y_score, model_name="Model", plot_path=None):
    """
    计算 AUC 并绘制 ROC 曲线。
    [Update] Automatically plots both Linear and Log-Scale ROC curves for WAF analysis.
    """
    auc_value = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)

    # === 1. Standard Linear Scale Plot ===
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc_value:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({model_name})")
    plt.legend(loc="lower right")
    plt.grid(True)

    if plot_path:
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] ROC 曲线 (Linear) 已保存到: {plot_path}")
    else:
        plt.show()
    plt.close()

    # === 2. [New] Log Scale Plot (For Low FPR Analysis) ===
    if plot_path:
        log_plot_path = plot_path.replace(".png", "_log.png")

        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc_value:.4f})", linewidth=2)
        plt.plot([0, 1], [0, 1], "k--", label="Random")

        # Set X-axis to Log Scale
        plt.xscale("log")
        plt.xlabel("False Positive Rate (Log Scale)")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve (Log Scale) - {model_name}")

        # Focus on the WAF relevant region (e.g., 0.0001 to 1.0)
        plt.xlim([1e-4, 1.0])
        plt.ylim([0.0, 1.05])

        plt.legend(loc="lower right")
        plt.grid(True, which="both", ls="-", alpha=0.5)

        plt.savefig(log_plot_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] ROC 曲线 (Log Scale) 已保存到: {log_plot_path}")
        plt.close()

    return auc_value, fpr, tpr
