import pandas as pd
import os


# ==========================================
# 1. Table 1: Overall Performance
# ==========================================
def generate_latex_table1_overall(input_file, output_tex_file):
    df = pd.read_csv(input_file)

    # 方法顺序和名称映射
    method_order = [
        "bow_xgb",
        "tfidf_xgb",
        "w2v_xgb",
        "fasttext_xgb",
        "textcnn",
        "char_cnn",
        "lstm_attn",
        "cnn_bilstm",
        "bert_xgb",
        "mgsir_xgb",
    ]
    name_mapping = {
        "bow_xgb": "BoW",
        "tfidf_xgb": "TF-IDF",
        "w2v_xgb": "Word2Vec",
        "fasttext_xgb": "FastText",
        "textcnn": "TextCNN",
        "char_cnn": "Char-CNN",
        "lstm_attn": "BiLSTM-Attn",
        "cnn_bilstm": "CNN-BiLSTM",
        "bert_xgb": "BERT",
        "mgsir_xgb": "Hi-MGR",
    }

    df = df[df["method"].isin(method_order)].copy()
    df["method"] = pd.Categorical(df["method"], categories=method_order, ordered=True)
    df = df.sort_values("method")

    df["fnr_val"] = df["fnr"] if "fnr" in df.columns else (1 - df["rec"])

    final_df = pd.DataFrame(
        {
            "Method": df["method"].map(name_mapping),
            "Accuracy (%)": df["acc"] * 100,
            "Precision (%)": df["prec"] * 100,
            "Recall/TPR (%)": df["rec"] * 100,
            "F1-Score (%)": df["f1"] * 100,
            "FNR (%)": df["fnr_val"] * 100,
            "FPR (%)": df["fpr"] * 100,
        }
    )

    # 生成 LaTeX tabular
    tabular = final_df.to_latex(
        index=False, escape=False, column_format="lcccccc", float_format="%.2f"
    )

    # 替换表头百分号为 LaTeX 转义
    lines = tabular.splitlines()
    header_line_idx = 2
    lines[header_line_idx] = lines[header_line_idx].replace("%", r"\%")

    # 替换 \hline 为 booktabs
    hline_indices = [i for i, l in enumerate(lines) if l.strip() == r"\hline"]
    if len(hline_indices) >= 3:
        lines[hline_indices[0]] = r"\toprule"
        lines[hline_indices[1]] = r"\midrule"
        lines[hline_indices[-1]] = r"\bottomrule"

    tabular = "\n".join(lines)

    latex_table = rf"""
\begin{{table*}}[!ht]
\centering
\caption{{Overall Performance Comparison}}
\label{{tab:overall_results}}
{tabular}
\end{{table*}}
"""
    os.makedirs(os.path.dirname(output_tex_file), exist_ok=True)

    with open(output_tex_file, "w") as f:
        f.write(latex_table)

    print(latex_table)


# ==========================================
# 2. Table 2: Robustness
# ==========================================
def generate_latex_table2_robustness(input_file, output_tex_file):
    df = pd.read_csv(input_file)

    # 方法和攻击顺序
    attack_order = [
        "adv_randomcase",
        "adv_space2comment",
        "adv_charencode",
        "adv_whitespace",
        "adv_versioned",
        "adv_symbolic",
        "adv_equaltolike",
        "adv_mix",
    ]
    method_order = [
        "bow_xgb",
        "tfidf_xgb",
        "w2v_xgb",
        "fasttext_xgb",
        "textcnn",
        "char_cnn",
        "lstm_attn",
        "cnn_bilstm",
        "bert_xgb",
        "mgsir_xgb",
    ]

    method_name_mapping = {
        "bow_xgb": "BoW",
        "tfidf_xgb": "TF-IDF",
        "w2v_xgb": "Word2Vec",
        "fasttext_xgb": "FastText",
        "textcnn": "TextCNN",
        "char_cnn": "Char-CNN",
        "lstm_attn": "BiLSTM-Attn",
        "cnn_bilstm": "CNN-BiLSTM",
        "bert_xgb": "BERT",
        "mgsir_xgb": "Hi-MGR",
    }

    attack_name_mapping = {
        "adv_randomcase": "RandomCase",
        "adv_space2comment": "Space2Comment",
        "adv_charencode": "CharEncode",
        "adv_whitespace": "Whitespace",
        "adv_versioned": "Versioned",
        "adv_symbolic": "Symbolic",
        "adv_equaltolike": "EqualToLike",
        "adv_mix": "Mix",
    }

    # 数据处理
    full_order = [f"{m}_{a}" for m in method_order for a in attack_order]
    df_subset = df[df["method"].isin(full_order)].copy()
    df_subset["Method"] = df_subset["method"].apply(lambda x: x.split("_adv_")[0])
    df_subset["Attack"] = df_subset["method"].apply(
        lambda x: "adv_" + x.split("_adv_")[1]
    )

    table = df_subset.pivot(index="Attack", columns="Method", values="rec")
    table = table.reindex(index=attack_order, columns=method_order) * 100
    table.index = table.index.map(attack_name_mapping)
    table.columns = table.columns.map(method_name_mapping)

    # 增加平均行
    avg_row = table.mean(axis=0).to_frame().T  # 转置为 DataFrame
    avg_row.index = ["Average"]  # 设置索引名
    table = pd.concat([table, avg_row])

    # 生成 LaTeX

    latex_df = table.reset_index()
    latex_df.rename(columns={latex_df.columns[0]: "Attack"}, inplace=True)
    column_format = "l" + "c" * (len(latex_df.columns) - 1)
    tabular = latex_df.to_latex(
        index=False, escape=False, column_format=column_format, float_format="%.2f"
    )

    # 替换 \hline 为 booktabs，并在 Average 行前插入 \midrule
    lines = tabular.splitlines()
    top_rule_done = False
    mid_rule_done = False
    new_lines = []

    for i, line in enumerate(lines):
        if line.strip() == r"\hline":
            if not top_rule_done:
                new_lines.append(r"\toprule")
                top_rule_done = True
            elif not mid_rule_done:
                new_lines.append(r"\midrule")
                mid_rule_done = True
            else:
                new_lines.append(r"\bottomrule")
        else:
            # 检查如果当前行包含 Average，且上一行不是 \midrule，则先插入 \midrule
            if "Average" in line:
                # 插入 \midrule 分割线
                if len(new_lines) == 0 or new_lines[-1].strip() != r"\midrule":
                    new_lines.append(r"\midrule")
            new_lines.append(line)

    tabular = "\n".join(new_lines)

    latex_table = rf"""
\begin{{table*}}[!ht]
\centering
\caption{{Robustness Comparison under Different Adversarial Attacks (Recall \%) with Average}}
\label{{tab:robustness_results}}
\resizebox{{\textwidth}}{{!}}{{
{tabular}
}}
\end{{table*}}
"""

    os.makedirs(os.path.dirname(output_tex_file), exist_ok=True)

    with open(output_tex_file, "w") as f:
        f.write(latex_table)

    print(latex_table)


# ==========================================
# 3. Table 3: Efficiency
# ==========================================
def generate_latex_table3_efficiency(input_file, output_tex_file):
    # ===== 1. Method order (paper order) =====
    method_order = [
        "bow_xgb",
        "tfidf_xgb",
        "w2v_xgb",
        "fasttext_xgb",
        "textcnn",
        "char_cnn",
        "cnn_bilstm",
        "lstm_attn",
        "bert_xgb",
        "mgsir_xgb",
    ]

    # ===== 2. Method name mapping (academic) =====
    name_mapping = {
        "bow_xgb": "BoW",
        "tfidf_xgb": "TF-IDF",
        "w2v_xgb": "Word2Vec",
        "fasttext_xgb": "FastText",
        "textcnn": "TextCNN",
        "char_cnn": "Char-CNN",
        "cnn_bilstm": "CNN-BiLSTM",
        "lstm_attn": "BiLSTM-Attn",
        "bert_xgb": "BERT",
        "mgsir_xgb": "Hi-MGR",
    }

    # ===== 3. Load CSV =====
    df = pd.read_csv(input_file)

    # ===== 4. Filter clean methods only =====
    df = df[df["method"].isin(method_order)].copy()

    # ===== 5. Enforce method order =====
    df["method"] = pd.Categorical(
        df["method"],
        categories=method_order,
        ordered=True,
    )
    df = df.sort_values("method")

    # ===== 6. Select & rename columns for Table 3 =====
    table3 = df[
        [
            "method",
            "f1",
            "latency_avg_ms",
            "latency_p99_ms",
            "qps",
            "model_size_mb",
        ]
    ].rename(
        columns={
            "method": "Method",
            "f1": "F1-Score (%)",
            "latency_avg_ms": "Avg. Latency (ms)",
            "latency_p99_ms": "P99 Latency (ms)",
            "qps": "QPS",
            "model_size_mb": "Model Size (MB)",
        }
    )

    # ===== 7. Apply method name mapping =====
    table3["Method"] = table3["Method"].map(name_mapping)

    # ===== 8. Round values (paper-friendly) =====
    table3["F1-Score (%)"] = table3["F1-Score (%)"] * 100
    table3["Avg. Latency (ms)"] = table3["Avg. Latency (ms)"].round(2)
    table3["P99 Latency (ms)"] = table3["P99 Latency (ms)"].round(2)
    table3["QPS"] = table3["QPS"].round(0).astype(int)
    table3["Model Size (MB)"] = table3["Model Size (MB)"].round(2)

    # 生成 LaTeX tabular
    tabular = table3.to_latex(
        index=False, escape=False, column_format="@{}lccccc@{}", float_format="%.2f"
    )
    tabular = tabular.replace("F1-Score (%)", r"F1-Score (\%)")
    tabular = tabular.replace(
    "Avg. Latency (ms)", r"\makecell{Avg.\\Latency (ms)}"
).replace(
    "P99 Latency (ms)", r"\makecell{P99\\Latency (ms)}"
).replace(
    "Model Size (MB)", r"\makecell{Model Size\\(MB)}"
)

    # 替换 \hline 为 booktabs
    lines = tabular.splitlines()

    hline_indices = [i for i, l in enumerate(lines) if l.strip() == r"\hline"]
    if len(hline_indices) >= 3:
        lines[hline_indices[0]] = r"\toprule"
        lines[hline_indices[1]] = r"\midrule"
        lines[hline_indices[-1]] = r"\bottomrule"

    tabular = "\n".join(lines)

    latex_table = rf"""
\begin{{table}}[!ht]
\centering
\setlength{{\tabcolsep}}{{1pt}}
\caption{{Efficiency Comparison}}
\label{{tab:efficiency_results}}
{tabular}
\end{{table}}
"""

    os.makedirs(os.path.dirname(output_tex_file), exist_ok=True)

    with open(output_tex_file, "w") as f:
        f.write(latex_table)

    print(latex_table)


# ==========================================
# 4. Table 4: Efficiency
# ==========================================
def generate_latex_table4_reliability(input_file, output_tex_file):
    import os
    import pandas as pd

    # 方法顺序和名称映射（与 Table 1 保持一致）
    method_order = [
        "bow_xgb",
        "tfidf_xgb",
        "w2v_xgb",
        "fasttext_xgb",
        "textcnn",
        "char_cnn",
        "lstm_attn",
        "cnn_bilstm",
        "bert_xgb",
        "mgsir_xgb",
    ]
    name_mapping = {
        "bow_xgb": "BoW",
        "tfidf_xgb": "TF-IDF",
        "w2v_xgb": "Word2Vec",
        "fasttext_xgb": "FastText",
        "textcnn": "TextCNN",
        "char_cnn": "Char-CNN",
        "lstm_attn": "BiLSTM-Attn",
        "cnn_bilstm": "CNN-BiLSTM",
        "bert_xgb": "BERT",
        "mgsir_xgb": "Hi-MGR",
    }

    # 读取并排序
    df = pd.read_csv(input_file)
    df = df[df["method"].isin(method_order)].copy()
    df["method"] = pd.Categorical(df["method"], categories=method_order, ordered=True)
    df = df.sort_values("method")

    # 构建最终表格
    final_df = pd.DataFrame(
        {
            "Method": df["method"].map(name_mapping),
            # "Accuracy (%)": df["acc"] * 100,
            # "Precision (%)": df["prec"] * 100,
            # "Recall (%)": df["rec"] * 100,
            # "F1-Score (%)": df["f1"] * 100,
            "TPR @ 1% FPR (%)": df["rec_at_fpr_1"] * 100,
            "TPR @ 0.1% FPR (%)": df["rec_at_fpr_01"] * 100
        }
    )

    # 生成 LaTeX tabular
    tabular = final_df.to_latex(
        index=False,
        escape=False,
        column_format="lccccccc",
        float_format="%.2f",
    )

    # 转义表头中的百分号
    lines = tabular.splitlines()
    header_line_idx = 2
    lines[header_line_idx] = lines[header_line_idx].replace("%", r"\%")

    # 替换 \hline 为 booktabs
    hline_indices = [i for i, l in enumerate(lines) if l.strip() == r"\hline"]
    if len(hline_indices) >= 3:
        lines[hline_indices[0]] = r"\toprule"
        lines[hline_indices[1]] = r"\midrule"
        lines[hline_indices[-1]] = r"\bottomrule"

    tabular = "\n".join(lines)

    latex_table = rf"""
\begin{{table}}[!ht]
\centering
\setlength{{\tabcolsep}}{{1pt}}
\caption{{Reliability Performance under Low False Positive Rates}}
\label{{tab:reliability_results}}
{tabular}
\end{{table}}
"""

    os.makedirs(os.path.dirname(output_tex_file), exist_ok=True)
    with open(output_tex_file, "w") as f:
        f.write(latex_table)

    print(latex_table)


# ==========================================
# 5. Table 5: Ablation Study
# ==========================================
def generate_latex_table5_ablation(input_file, output_tex_file):
    df = pd.read_csv(input_file)

    # 1. 定义两组配置
    # A组: 移除测试 (Ablation)
    targets_wo = [
        "mgsir_xgb",  # Full
        "mgsir_xgb_No_L4",
        "mgsir_xgb_No_L3",
        "mgsir_xgb_No_L2",
        "mgsir_xgb_No_L1",
    ]
    # B组: 单独测试 (Individual)
    targets_only = [
        "mgsir_xgb_L1_only",
        "mgsir_xgb_L2_only",
        "mgsir_xgb_L3_only",
        "mgsir_xgb_L4_only",
    ]

    labels = {
        "mgsir_xgb": "Full Hi-MGR",
        "mgsir_xgb_No_L4": "w/o SSS (L4)",
        "mgsir_xgb_No_L3": "w/o Syntax (L3)",
        "mgsir_xgb_No_L2": "w/o Token (L2)",
        "mgsir_xgb_No_L1": "w/o Statistical (L1)",
        "mgsir_xgb_L1_only": "Statistical Only (L1)",
        "mgsir_xgb_L2_only": "Token Only (L2)",
        "mgsir_xgb_L3_only": "Syntax Only (L3)",
        "mgsir_xgb_L4_only": "S Only (L4)",
    }

    # 筛选数据
    all_targets = targets_wo + targets_only
    df = df[df["method"].isin(all_targets)].copy()

    # 强制排序
    df["method"] = pd.Categorical(df["method"], categories=all_targets, ordered=True)
    df = df.sort_values("method")

    # 提取指标
    res = pd.DataFrame(
        {
            "Configuration": df["method"].map(labels),
            "Precision (%)": df["prec"] * 100,
            "Recall (%)": df["rec"] * 100,
            "F1-Score (%)": df["f1"] * 100,
        }
    )

    # 计算 Delta F1 (相对于 Full)
    full_f1 = res.iloc[0]["F1-Score (%)"]
    res[r"$\Delta$ F1"] = res["F1-Score (%)"] - full_f1

    # 格式化 Delta
    res[r"$\Delta$ F1"] = res[r"$\Delta$ F1"].apply(
        lambda x: f"{x:.2f}" if abs(x) > 1e-6 else "-"
    )

    # 生成 LaTeX
    tabular = res.to_latex(
        index=False, escape=False, column_format="lcccc", float_format="%.2f"
    )

    # 手动插入分割线，把 w/o 和 only 分开
    lines = tabular.splitlines()
    new_lines = []
    header_line_idx = 2
    lines[header_line_idx] = lines[header_line_idx].replace("%", r"\%")

    # 我们知道 targets_wo 有 4 行 (包含表头是 line[0-3] 数据是 line[4-7])
    # booktabs 处理逻辑 + 中间插入分割线

    header_processed = False

    for i, line in enumerate(lines):
        if line.strip() == r"\hline":
            if not header_processed:  # 顶线
                new_lines.append(r"\toprule")
            elif "bottomrule" not in new_lines:  # 中线 (表头下)
                new_lines.append(r"\midrule")
                header_processed = True
            else:  # 底线
                new_lines.append(r"\bottomrule")
        elif "Only (L1)" in line:  # 检测到 Only 组开始，插入一条分割线
            new_lines.append(r"\midrule")
            new_lines.append(line)
        else:
            new_lines.append(line)

    tabular = "\n".join(new_lines)

    latex_table = rf"""
\begin{{table}}[!ht]
\setlength{{\tabcolsep}}{{1pt}}
\centering
\caption{{Ablation Study: Impact of Feature Layers (Removal vs. Individual)}}
\label{{tab:ablation}}
{tabular}
\end{{table}}
"""

    os.makedirs(os.path.dirname(output_tex_file), exist_ok=True)
    with open(output_tex_file, "w") as f:
        f.write(latex_table)
    print(f"Generated Comprehensive Ablation Table: {output_tex_file}")


if __name__ == "__main__":
    input_csv = "results/metrics/all_results.csv"

    # Table 1: Overall
    generate_latex_table1_overall(
        input_csv, "results/latex_tables/table1_overall_results.tex"
    )

    # Table 2: Robustness
    generate_latex_table2_robustness(
        input_csv, "results/latex_tables/table2_robustness.tex"
    )

    # Table 3: Efficiency
    generate_latex_table3_efficiency(
        input_csv, "results/latex_tables/table3_efficiency.tex"
    )

    # Table 5: Ablation
    generate_latex_table5_ablation(
        input_csv, "results/latex_tables/table5_ablation.tex"
    )
    generate_latex_table4_reliability(
        input_csv, "results/latex_tables/table4_reliability.tex"
    )
