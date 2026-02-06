# src/features/mgsir/hfes_enhanced.py

"""
Hierarchical Feature Extraction System (HFES) - Enhanced Version
----------------------------------------------------------------
该模块实现了基于多粒度分层框架 (Multi-granularity Hierarchical Framework) 的 SQL 注入特征提取。
特征体系共包含 23 维特征，分为四个抽象层级与对应粒度：
L1: Statistical Character Features (层级) / Global Distribution Granularity (粒度)
L2: Symbolic Injection Features (层级) / Discrete Symbolic Granularity (粒度)
L3: Token-level Structural Features (层级) / Sequential Lexical Granularity (粒度)
L4: Syntactic Anomaly Features (层级) / Structural Integrity Granularity (粒度)
"""

import warnings
import re
import pandas as pd
from sklearn.preprocessing import StandardScaler
from .hdcan import smart_recursive_decode_fast, fast_tokenize
from src.utils.security_utils import safe_division

# 忽略 pandas 的一些未来版本警告
warnings.filterwarnings("ignore")

# ==========================================
# 1. 静态资源定义 (Static Resources)
# ==========================================
# 优化策略：使用 set 数据结构将查找复杂度从 O(n) 降低至 O(1)

# SQL 关键字集合：涵盖 DML, DDL, 控制流及常用注入关键字
SQL_KEYWORDS = {
    'alter','analyze','begin','call','commit','create','delete','drop','execute',
    'explain','fetch','grant','handler','insert','into','join','lock','optimize',
    'prepare','repair','replace','rollback','savepoint','select','set','start',
    'truncate','union','update','values','where','from','group','by','order','limit',
    'all','distinct','exists','having','in','index','inner','left','offset','on',
    'outer','right','top','view','with','recursive','case','when','then','else',
    'end','if','between','like','regexp','rlike','and','or','xor','not','is',
    'null','true','false','binary'
}

# SQL 函数集合：涵盖常用指纹函数、时间盲注函数及报错注入函数
SQL_FUNCTIONS = {
    'ascii','char','concat','hex','length','load_file','sleep','benchmark','delay',
    'waitfor','extractvalue','updatexml','database','user','version','mid','substr',
    'substring','count','sum','avg','ord','char_length','cast','convert'
}
SQL_KEYWORDS_SET = set(SQL_KEYWORDS)
OP_CHARS = set("=<>!+-*/%&|^")  # SQL 运算符集合

# ==========================================
# 2. 正则表达式编译 (Regex Compilation)
# ==========================================
# 预编译正则以提升高并发下的匹配性能
RE_HEX = re.compile(r"0[xX][0-9a-fA-F]+\b")  # 十六进制数 (0x...)
RE_ALPHA = re.compile(r"[a-zA-Z]")  # 纯字母分布
RE_DIGIT = re.compile(r"[0-9]")  # 纯数字分布
RE_PUNC = re.compile(r"[!#$%&,.:;<=>?@\[\\\]^_`{|}~]")  # 广义标点符号
RE_ARITH = re.compile(r"(?<!/)\*(?!\*)|(?<!\*)/(?!\*)|[+\-<>]=?")  # 算术运算符
RE_SYMLOGIC = re.compile(r"(<>|!=|\|\||&&|\^)")  # 符号逻辑符 (&&, ||)
RE_LOGIC = re.compile(r"\bnot\b|\band\b|\bor\b|\bxor\b")  # 单词逻辑符


# ==========================================
# 3. 辅助函数：Token 类型统计 (Level 3 Core)
# ==========================================
def _token_type_stats(text: str):
    """
    [L3 核心算法] 基于简易词法分析的 Token 统计

    目的:
    不依赖昂贵的 AST 解析，通过一次线性扫描 (O(N)) 提取 Token 流的结构特征。

    Args:
        text (str): 预处理后的小写字符串

    Returns:
        tuple: (tok_switch, op_ratio, lit_ratio)
            - tok_switch: Token 类型切换次数 (衡量 Payload 的混淆/复杂度)
            - op_ratio: 操作符密度 (计算密集型特征)
            - lit_ratio: 字面量密度 (数据密集型特征)
    """
    if not text:
        return 0, 0.0, 0.0

    types = []
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]

        # 类型 1: 字符串字面量 (String Literal)
        if ch == "'" or ch == '"':
            quote = ch
            i += 1
            while i < n:
                if text[i] == quote:
                    i += 1
                    break
                i += 1
            types.append("STR")
            continue

        # 类型 2: 数值字面量 (Numeric Literal)
        if ch.isdigit():
            i += 1
            while i < n and (text[i].isdigit() or text[i] == "."):
                i += 1
            types.append("NUM")
            continue

        # 类型 3: 标识符/关键字 (Identifier/Keyword)
        # 简化处理：将所有字母开头的连续串视为单词，具体区分在外部逻辑完成
        if ch.isalpha() or ch == "_":
            i += 1
            while i < n and (text[i].isalnum() or text[i] == "_"):
                i += 1
            types.append("IDENT")
            continue

        # 类型 4: 括号 (Parentheses) - 重要的结构分隔符
        if ch == "(" or ch == ")":
            types.append("PAREN")
            i += 1
            continue

        # 类型 5: 操作符 (Operator)
        if ch in OP_CHARS:
            i += 1
            while i < n and text[i] in OP_CHARS:
                i += 1
            types.append("OP")
            continue

        # 跳过空白符和其他字符
        i += 1

    if not types:
        return 0, 0.0, 0.0

    # 计算结构统计量
    # tok_switch: 如果相邻两个 Token 类型不同，计数+1。
    # 例如: "1=1" (NUM, OP, NUM) -> Switch = 2.
    # 正常语句通常结构平滑，注入攻击往往结构破碎。
    tok_switch = sum(1 for a, b in zip(types, types[1:]) if a != b)

    total_tokens = len(types)
    # op_ratio: 逻辑/算术操作符占比
    op_ratio = types.count("OP") / total_tokens
    # lit_ratio: 数据(字符串+数字)占比
    lit_ratio = (types.count("STR") + types.count("NUM")) / total_tokens

    return tok_switch, op_ratio, lit_ratio


# ==========================================
# 4. 特征分层映射配置 (Feature Taxonomy)
# ==========================================

# [L1] Statistical Character Features (统计字符特征) - 6 Dim
# 粒度: Global Distribution Granularity (全局分布粒度)
# 描述 Payload 的宏观分布与信息密度
LEVEL_1_KEYS = ["qlen", "wcount", "hexnum", "alpha", "digit", "coef_k"]

# [L2] Symbolic Injection Features (符号注入特征) - 9 Dim
# 粒度: Discrete Symbolic Granularity (离散符号粒度)
# 描述离散控制符号的分布（闭合、注释、分隔）
LEVEL_2_KEYS = [
    "sq",  # Single Quotes
    "dq",  # Double Quotes
    "lparen",  # Left Parentheses
    "rparen",  # Right Parentheses
    "puncts",  # Total Punctuation
    "comments",  # SQL Comments (--, #, /*)
    "arith",  # Arithmetic Operators
    "spaces",  # Whitespace count
    "sym_logic",  # Symbolic Logic (&&, ||)
]

# [L3] Token-level Structural Features (Token级结构特征) - 6 Dim
# 粒度: Sequential Lexical Granularity (序列词法粒度)
# 描述 Token 序列流的词法组成与切换关系
LEVEL_3_KEYS = [
    "logic",  # Keyword Logic (AND, OR)
    "sqlkw",  # SQL Keywords count
    "sqlfunc",  # SQL Functions count
    "tok_switch",  # Token type switching frequency
    "op_ratio",  # Operator density
    "lit_ratio",  # Literal density (Data-centric view)
]

# [L4] Syntactic Anomaly Features (语法异常特征) - 2 Dim
# 粒度: Structural Integrity Granularity (结构完整性粒度)
# 描述结构闭合一致性与完整性异常
LEVEL_4_KEYS = ["qmismatch", "paren_mismatch"]

# 完整特征集验证: 6(L1) + 9(L2) + 6(L3) + 2(L4) = 23 Features
ALL_KEYS = LEVEL_1_KEYS + LEVEL_2_KEYS + LEVEL_3_KEYS + LEVEL_4_KEYS
FEATURE_INDEX_MAP = {k: i for i, k in enumerate(ALL_KEYS)}
BASE_FEATURE_MAP = {k: None for k in ALL_KEYS}


def get_ablation_features(mode):
    """
    消融实验配置接口 (Ablation Study Interface)
    根据输入的 mode 返回对应的特征列名列表
    """
    # 单层测试
    if mode == "L1":
        return LEVEL_1_KEYS
    if mode == "L2":
        return LEVEL_2_KEYS
    if mode == "L3":
        return LEVEL_3_KEYS
    if mode == "L4":
        return LEVEL_4_KEYS

    # 累加测试 (Cumulative)
    if mode == "L1_L2":
        return LEVEL_1_KEYS + LEVEL_2_KEYS
    if mode == "L1_L2_L3":
        return LEVEL_1_KEYS + LEVEL_2_KEYS + LEVEL_3_KEYS

    # 留一法测试 (Leave-one-out)
    if mode == "No_L4":
        return LEVEL_1_KEYS + LEVEL_2_KEYS + LEVEL_3_KEYS
    if mode == "No_L3":
        return LEVEL_1_KEYS + LEVEL_2_KEYS + LEVEL_4_KEYS
    if mode == "No_L2":
        return LEVEL_1_KEYS + LEVEL_3_KEYS + LEVEL_4_KEYS
    if mode == "No_L1":
        return LEVEL_2_KEYS + LEVEL_3_KEYS + LEVEL_4_KEYS

    # 单层独立测试 (别名)
    if mode == "L1_only":
        return LEVEL_1_KEYS
    if mode == "L2_only":
        return LEVEL_2_KEYS
    if mode == "L3_only":
        return LEVEL_3_KEYS
    if mode == "L4_only":
        return LEVEL_4_KEYS

    return ALL_KEYS


# ==========================================
# 5. 特征提取主逻辑 (Feature Extraction Pipeline)
# ==========================================
def extract_struct_features_single(q, active_cols=None):
    """
    单条样本特征提取函数

    Process:
    1. Preprocessing (Decoding & Normalization)
    2. Hierarchical Feature Extraction (L1 -> L4)
    3. Vectorization
    """
    # 鲁棒性检查：非字符串输入直接返回零向量
    if not q or not isinstance(q, str):
        if active_cols is None:
            return [0] * len(ALL_KEYS)
        else:
            return [0] * len(active_cols)

    try:
        # A. 预处理 (Preprocessing)
        # 智能递归解码，将 URL/HTML 编码还原，暴露攻击载荷
        q_clean = smart_recursive_decode_fast(q)
        q_lower = q_clean.lower()

        # B. 提取 L1: Statistical Character Features (Global Distribution Granularity)
        val_qlen = len(q_lower)

        # C. 提取 L2: Symbolic Injection Features (Discrete Symbolic Granularity)
        val_sq = q_lower.count("'") # L2: 单引号计数
        val_dq = q_lower.count('"')
        val_spaces = q_lower.count(" ")
        # 聚合注释符：--, #, /* 均为 SQL 注入常用截断符
        val_comments = q_lower.count("--") + q_lower.count("#") + q_lower.count("/*")

        # 括号计数 (用于后续 L4 计算)
        val_lparen = q_lower.count("(")
        val_rparen = q_lower.count(")")

        # D. 提取 L3: Token-level Structural Features (Sequential Lexical Granularity, Part 1)
        # [Comment Normalization] 在分词前临时去除注释，防止 "Token Glue" (如 SELECT/**/FROM)
        tokens = fast_tokenize(q_clean)
        val_wcount = len(tokens)

        # 统计 SQL 关键字和函数
        val_sqlkw = 0
        val_sqlfunc = 0
        for t in tokens:
            if t in SQL_KEYWORDS:
                val_sqlkw += 1
            if t in SQL_FUNCTIONS:
                val_sqlfunc += 1

        # 修正：如果只有一个词且是关键字，可能是正常查询参数(如 name=select)，视为误报消除
        if val_wcount == 1 and val_sqlkw == 1:
            val_sqlkw = 0

        # E. 正则辅助统计 (L1/L2 补充)
        val_hexnum = len(RE_HEX.findall(q_lower))  # L1: 编码特征
        val_alpha = len(RE_ALPHA.findall(q_lower))  # L1: 字母分布
        val_digit = len(RE_DIGIT.findall(q_lower))  # L1: 数字分布
        val_puncts = len(RE_PUNC.findall(q_lower))  # L2: 标点分布
        val_arith = len(RE_ARITH.findall(q_lower))  # L2: 算术运算
        val_sym_logic = len(RE_SYMLOGIC.findall(q_lower))  # L2: 符号逻辑
        val_logic = len(RE_LOGIC.findall(q_lower))  # L3: 单词逻辑

        # F. 提取 L3: Token-level Structural Features (Sequential Lexical Granularity, Part 2)
        # 调用核心辅助函数，一次性计算 Token 结构特征
        val_tok_switch, val_op_ratio, val_lit_ratio = _token_type_stats(q_lower)

        # G. 衍生特征计算
        # L1: coef_k (关键词稀疏系数)
        # 公式: (空格 + 注释) / 单词数
        # 意义: 衡量 Payload 的"紧凑度"。攻击载荷通常极度紧凑 (coef_k -> 0) 或大量混淆 (coef_k 异常)
        val_coef_k = safe_division(val_spaces + val_comments, val_wcount, 0.0)

        # H. 提取 L4: Syntactic Anomaly Features (Structural Integrity Granularity)
        # 意义: 闭合不匹配是 SQL 注入（尤其是盲注）的强特征
        val_qmismatch = 1 if (val_sq % 2 != 0 or val_dq % 2 != 0) else 0
        val_paren_mismatch = 1 if (val_lparen != val_rparen) else 0

        # I. 特征向量构建 (Feature Vector Construction)
        # 警告：此列表顺序必须严格对应 ALL_KEYS 的定义顺序，否则会导致特征错位！
        features = [
            # --- L1: Statistical Character Features (Global Distribution Granularity) ---
            val_qlen,  # Query Length
            val_wcount,  # Word Count
            val_hexnum,  # Hex String Count
            val_alpha,  # Alpha Char Count
            val_digit,  # Digit Char Count
            val_coef_k,  # Keyword Sparsity Coefficient (SHAP Top Feature)
            # --- L2: Symbolic Injection Features (Discrete Symbolic Granularity) ---
            val_sq,  # Single Quotes
            val_dq,  # Double Quotes
            val_lparen,  # Left Parentheses count
            val_rparen,  # Right Parentheses count
            val_puncts,  # Total Punctuation
            val_comments,  # Comment Indicators
            val_arith,  # Arithmetic Operators
            val_spaces,  # Spaces
            val_sym_logic,  # Symbolic Logic Operators
            # --- L3: Token-level Structural Features (Sequential Lexical Granularity) ---
            val_logic,  # Logical Keywords (AND/OR/NOT)
            val_sqlkw,  # SQL Keyword Count
            val_sqlfunc,  # SQL Function Count
            val_tok_switch,  # Token Switching Rate
            val_op_ratio,  # Operator Ratio
            val_lit_ratio,  # Literal Ratio (Data Density)
            # --- L4: Syntactic Anomaly Features (Structural Integrity Granularity) ---
            val_qmismatch,  # Quote Mismatch Flag
            val_paren_mismatch,  # Parenthesis Mismatch Flag
        ]

        if active_cols is None:
            return features
        else:
            return [features[FEATURE_INDEX_MAP[col]] for col in active_cols]

    except Exception as e:
        # 异常处理：保证 Pipeline 不因单条坏数据崩溃
        import logging

        logging.getLogger(__name__).warning(f"Feature extraction failed: {e}")
        if active_cols is None:
            return [0] * len(ALL_KEYS)
        else:
            return [0] * len(active_cols)


# ==========================================
# 6. 批量处理接口 (Batch Processing Interface)
# ==========================================
def extract_struct_features(data, active_cols=None):
    """
    批量提取 DataFrame 中的特征
    """
    rows = [extract_struct_features_single(q, active_cols) for q in data["Query"]]

    if active_cols is None:
        cols = ALL_KEYS
    else:
        cols = [k for k in ALL_KEYS if k in active_cols]

    feat_df = pd.DataFrame(rows, columns=cols, index=data.index)
    return pd.concat([data, feat_df], axis=1)


def standardize_and_combine_features(x_train, x_test, num_cols):
    """
    特征标准化 (Z-Score Standardization)
    """
    X_train_num = x_train[num_cols].values
    X_test_num = x_test[num_cols].values

    scaler = StandardScaler()
    scaler.fit(X_train_num)

    X_train_num_scaled = scaler.transform(X_train_num)
    X_test_num_scaled = scaler.transform(X_test_num)
    return X_train_num_scaled, X_test_num_scaled, scaler
