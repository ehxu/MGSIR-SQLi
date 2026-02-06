import csv
import pandas as pd
import numpy as np
import random
import re
import urllib.parse
from pathlib import Path

# ==========================================
# 全局配置 & 辅助数据 (Global Configuration)
# ==========================================

# SQL 关键词库 (参考 SQLMap)
# 用于 randomcase (随机大小写) 和 versioned (版本注释) 混淆
# 只有匹配到这些单词时才会进行变异，避免破坏非关键字符串
# Used for Set A (Random Case) and Set E (Versioned Comments)
KB_KEYWORDS = {
    "SELECT", "INSERT", "UPDATE", "DELETE", "FROM", "WHERE", "AND", "OR",
    "UNION", "ALL", "LIMIT", "ORDER", "BY", "GROUP", "HAVING", "AS",
    "INTO", "VALUES", "SET", "DROP", "CREATE", "TABLE", "DATABASE",
    "NULL", "LIKE", "BETWEEN", "IN", "IS", "NOT", "EXISTS", "JOIN",
    "LEFT", "RIGHT", "INNER", "OUTER", "ON", "CASE", "WHEN", "THEN",
    "ELSE", "END", "CAST", "CONVERT", "CHAR", "VARYING", "TEXT",
    "DECLARE", "EXEC", "EXECUTE", "XP_CMDSHELL", "SLEEP", "BENCHMARK",
    "USER", "SYSTEM_USER", "CURRENT_USER", "VERSION", "DATABASE", "SCHEMA"
}

# [关键修改] 空白符字符集 (参考 SQLMap space2mysqlblank.py)
# Used for Set D (Whitespace Manipulation)
# 作用：用于替换普通空格，测试模型对非标准空白符的识别能力。
# 剔除了 '\n' (换行) 和 '\r' (回车)，防止破坏 CSV 文件的行结构，确保一行一个样本。
# 保留字符含义:
# \t: 水平制表符 (Tab)
# \x0b: 垂直制表符 (Vertical Tab, MySQL支持)
# \x0c: 换页符 (Form Feed, MySQL支持)
# \xa0: 不换行空格 (Non-breaking space)
MYSQL_BLANKS = ("\t", "\x0b", "\x0c", "\xa0")

# ==========================================
# 核心混淆策略 (Obfuscation Strategies)
# ==========================================


# --- [Set A] Technique Category: Syntactic Obfuscation ---
def tamper_randomcase(payload):
    """
    [Set A: Random Case]
    Category: Syntactic Obfuscation
    功能: 将关键词随机大小写混合。
    示例: SELECT -> SeLeCt
    测试点: 模型的归一化能力 (Normalization)。
    """
    if not isinstance(payload, str):
        return payload
    retVal = payload
    # 正则查找所有单词 (至少2个字符)
    for match in re.finditer(r"\b[A-Za-z_]{2,}\b", retVal):
        word = match.group()
        # 只混淆属于 SQL 关键词的词
        if word.upper() in KB_KEYWORDS:
            new_word = ""
            for char in word:
                # 50% 概率大写，50% 概率小写
                new_word += char.upper() if random.randint(0, 1) else char.lower()
            # 替换回原句 (使用 boundary \b 避免误伤)
            retVal = re.sub(r"\b" + re.escape(word) + r"\b", new_word, retVal)
    return retVal


# --- [Set B] Technique Category: Syntactic Obfuscation ---
def tamper_space2comment(payload):
    """
    [Set B: Inline Comment]
    Category: Syntactic Obfuscation
    功能: 将空格替换为 SQL 内联注释 /**/。
    示例: SELECT FROM -> SELECT/**/FROM
    测试点: 模型的 Tokenizer (分词器) 是否能处理噪声干扰。
    逻辑: 带有状态机 (quote/doublequote) 检查，防止替换了引号内的空格。
    """
    if not isinstance(payload, str):
        return payload
    retVal = ""
    quote = False
    doublequote = False
    firstspace = False

    for i in range(len(payload)):
        # 状态机：如果遇到单/双引号，切换状态，不再替换空格
        if not firstspace:
            if payload[i].isspace():
                firstspace = True
                retVal += "/**/"
                continue
        elif payload[i] == "'":
            quote = not quote
        elif payload[i] == '"':
            doublequote = not doublequote
        elif payload[i] == " " and not doublequote and not quote:
            retVal += "/**/"
            continue
        retVal += payload[i]
    return retVal


# --- [Set C] Technique Category: Encoding Obfuscation ---
def tamper_charencode(payload):
    """
    [Set C: URL Encoding]
    Category: Encoding Obfuscation
    功能: 对 Paylaod 进行全 URL 编码。
    示例: SELECT -> %53%45%4C%45%43%54
    测试点: 模型的预处理流程中是否包含解码步骤 (Decoding)。
    """
    if not isinstance(payload, str):
        return payload
    return urllib.parse.quote(payload)


# --- [Set D] Technique Category: Encoding Obfuscation ---
def tamper_whitespace(payload):
    """
    [Set D: Whitespace Manipulation]
    Category: Encoding Obfuscation
    功能: 将普通空格替换为罕见的空白控制字符 (Tab, VT, FF, NBSP)。
    示例: SELECT FROM -> SELECT\tFROM
    测试点: 测试模型是否只认普通空格 (ASCII 32)，而忽略了其他合法分隔符。
    注意: 这里已移除换行符，保证 CSV 格式安全。
    """
    if not isinstance(payload, str):
        return payload
    retVal = ""
    quote = False
    doublequote = False
    firstspace = False

    for i in range(len(payload)):
        if not firstspace:
            if payload[i].isspace():
                firstspace = True
                retVal += random.choice(MYSQL_BLANKS)  # 随机选一种特殊空白符
                continue
        elif payload[i] == "'":
            quote = not quote
        elif payload[i] == '"':
            doublequote = not doublequote
        elif payload[i] == " " and not doublequote and not quote:
            retVal += random.choice(MYSQL_BLANKS)
            continue
        retVal += payload[i]
    return retVal


# --- [Set E] Technique Category: Semantic Obfuscation ---
def tamper_versioned(payload):
    """
    [Set E: Versioned Comments]
    Category: Semantic Obfuscation
    功能: 使用 MySQL 版本注释包裹关键词。
    示例: UNION -> /*!UNION*/
    原理: 在 MySQL 中 /*! ... */ 内的代码会被执行，但其他解析器可能将其视为注释忽略。
    测试点: 模型的去注释逻辑是否过于激进，导致丢失关键 Payload。
    """
    if not isinstance(payload, str):
        return payload

    def process(match):
        word = match.group("word")
        if word.upper() in KB_KEYWORDS:
            return "/*!%s*/" % word
        else:
            return word

    # 正则替换: 查找单词并包裹
    retVal = re.sub(r"(?<=\W)(?P<word>[A-Za-z_]+)(?=[^\w(]|\Z)", process, payload)
    # 清理可能产生的格式问题
    retVal = retVal.replace(" /*!", "/*!").replace("*/ ", "*/")
    return retVal


# --- [Set F] Technique Category: Semantic Obfuscation ---
def tamper_symbolic(payload):
    """
    [Set F: Symbolic Replacement]
    Category: Semantic Obfuscation
    功能: 将语义相同的逻辑词替换为符号。
    示例: AND -> &&, OR -> ||
    测试点: 模型的语义理解能力，是否通过语义分析而非简单的关键词匹配。
    """
    if not isinstance(payload, str):
        return payload
    retVal = payload
    # (?i) 表示忽略大小写匹配
    retVal = re.sub(r"(?i)\bAND\b", "&&", retVal)
    retVal = re.sub(r"(?i)\bOR\b", "||", retVal)
    return retVal


# --- [Set G] Technique Category: Semantic Obfuscation ---
def tamper_equaltolike(payload):
    """
    [Set G: Operator Substitution]
    Category: Semantic Obfuscation
    功能: 将等号替换为 LIKE 运算符。
    示例: id = 1 -> id LIKE 1
    测试点: 是否依赖 "=" 特征进行检测。
    """
    if not isinstance(payload, str):
        return payload
    return re.sub(r"\s*=\s*", " LIKE ", payload)


# --- [Set H] Technique Category: Comprehensive Robustness ---
def tamper_mix(payload):
    """
    [Set H: Mixed Obfuscation]
    Category: Comprehensive Robustness
    功能: 随机叠加 2-3 种上述策略。
    示例: 组合多种技术模拟真实复杂攻击。如 UNION SELECT -> %2f%2a%21UNION%2a%2f%20%2f%2a%21SeLeCt%2a%2f (URL编码 + 版本注释 + 大小写)
    测试点: 模拟真实的高级攻击场景，测试模型的极限鲁棒性。
    """
    all_strategies = [
        tamper_randomcase,  # Set A
        tamper_space2comment,  # Set B
        tamper_charencode,  # Set C
        tamper_whitespace,  # Set D
        tamper_versioned,  # Set E
        tamper_symbolic,  # Set F
        tamper_equaltolike,  # Set G
    ]
    # 随机选择 2 到 3 个策略组合
    k = random.randint(2, 3)
    selected = random.sample(all_strategies, k)

    temp = payload
    for func in selected:
        temp = func(temp)
    return temp


# ==========================================
# 主程序
# ==========================================
def generate_adversarial_dataset(input_csv, output_dir):
    print(f"[Init] Loading data from {input_csv}...")
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"[Error] Reading CSV failed: {e}")
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 映射字典：将代码中的模式名称对应到论文中的 Set 名称
    modes = {
        "randomcase": tamper_randomcase,  # Set A
        "space2comment": tamper_space2comment,  # Set B
        "charencode": tamper_charencode,  # Set C
        "whitespace": tamper_whitespace,  # Set D
        "versioned": tamper_versioned,  # Set E
        "symbolic": tamper_symbolic,  # Set F
        "equaltolike": tamper_equaltolike,  # Set G
        "mix": tamper_mix,  # Set H
    }

    results_info = []

    for mode_name, tamper_func in modes.items():
        print(f"[Process] Generating adversarial set: {mode_name}...")
        df_adv = df.copy()

        # 仅针对 Label=1 (攻击样本) 进行混淆
        # Label=0 (正常样本) 保持不变，作为对照
        mask = df_adv["Label"] == 1

        # 应用对应的 tamper 函数
        df_adv.loc[mask, "Query"] = df_adv.loc[mask, "Query"].apply(tamper_func)

        # 保存文件
        out_file = output_dir / f"test_adv_{mode_name}.csv"

        # [配置修改] quoting=csv.QUOTE_MINIMAL (默认行为)
        # 含义: "最小化引号策略"。仅当字段内容包含特殊字符（如分隔符逗号、引号、换行符等）时，才自动添加引号包裹。
        # 效果: 
        #   1. 普通混淆（如 randomcase）生成的 CSV 保持清爽，字段不带多余引号。
        #   2. 特殊混淆（如 whitespace/mix）若包含分隔符或控制符，会自动加引号防止格式崩坏。
        df_adv.to_csv(out_file, index=False, quoting=csv.QUOTE_MINIMAL)

        results_info.append({"Mode": mode_name, "Count": mask.sum()})

    return results_info


if __name__ == "__main__":
    # 定位路径
    BASE_DIR = Path(__file__).resolve().parents[1]
    INPUT_TEST_CSV = BASE_DIR / "data/dataset1/processed/test.csv"
    OUTPUT_ADV_DIR = BASE_DIR / "data/dataset1/adversarial"

    if not INPUT_TEST_CSV.exists():
        print(f"[Error] Test CSV not found at: {INPUT_TEST_CSV}")
    else:
        info = generate_adversarial_dataset(INPUT_TEST_CSV, OUTPUT_ADV_DIR)
        print("\n=== Generation Complete ===")
        print(f"{'Mode':<20} | {'Obfuscated Samples'}")
        print("-" * 40)
        for i in info:
            print(f"{i['Mode']:<20} | {i['Count']}")
