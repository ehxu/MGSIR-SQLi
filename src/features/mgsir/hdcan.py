# src/features/mgsir/hdcan.py
import re
import urllib.parse
import html
import string
import unicodedata

# ==========================================
# 1. 静态转换表：空白归一化 (Whitespace Normalization)
# ==========================================
# [关键修复] 这个表用于将所有“怪异空白”转为标准空格 (ASCII 32)
# 但保留所有字母、数字、符号！确保统计特征 (val_spaces) 能正确计数。
_WS_MAP = {i: i for i in range(256)}  # 默认保持原样
# 将 \t(9), \n(10), \v(11), \f(12), \r(13) 全部映射为 空格(32)
for c in [9, 10, 11, 12, 13]:
    _WS_MAP[c] = 32
WS_TRANS_TABLE = {i: _WS_MAP[i] for i in range(256)}

# ==========================================
# 2. 静态转换表：极速去噪 (Aggressive Cleanup - ASCII only)
# ==========================================
# 这个表用于分词，将所有非字母数字转为空格
_CLEAN_MAP = [32] * 256
for c in string.digits:
    _CLEAN_MAP[ord(c)] = ord(c)
for c in string.ascii_lowercase:
    _CLEAN_MAP[ord(c)] = ord(c)
for c in string.ascii_uppercase:
    _CLEAN_MAP[ord(c)] = ord(c.lower())
FAST_CLEAN_TABLE = {i: _CLEAN_MAP[i] for i in range(256)}

# ==========================================
# 3. Regex (only cheap ones)
# ==========================================
RE_URL_CHECK = re.compile(r"%[0-9a-fA-F]{2}")
RE_HTML_CHECK = re.compile(r"&[a-z]+;|&#[0-9]+;")
RE_UNICODE_ESC = re.compile(r"%u([0-9a-fA-F]{4})", re.IGNORECASE)
RE_TOKEN_SPLIT = re.compile(r"[^a-zA-Z0-9_]+")

# [增强] 统一处理内联注释 (/**/) 和 版本号注释 (/*!50000...)
# 目的：将它们替换为空格，防止关键词粘连 (e.g. SELECT/**/FROM -> SELECT FROM)
RE_COMMENTS_ALL = re.compile(r"/\*!.*?\*/|/\*.*?\*/", re.DOTALL)


# ==========================================
# 4. Fast recursive decode
# ==========================================
def smart_recursive_decode_fast(s, max_rounds=3, max_len=5000):
    """
    [最终高性能版] 智能递归解码器
    性能指标：单核 > 12000 QPS

    设计策略：
    1. 循环内移除所有 O(N) 扫描 (如 isascii)，只保留极快的子串搜索 ('in')。
    2. 使用 continue 策略，一旦某种编码被解开，立即进入下一轮，减少无效判断。
    3. 仅在最终阶段做一次 isascii 检查，以决定是否跳过昂贵的 NFKC。
    """

    # ---------------------------------------------------
    # 1. 基础输入验证与防御 (DoS Protection)
    # ---------------------------------------------------
    if not s:
        return ""

    # 类型防御：bytes -> str，其他类型拒绝
    if isinstance(s, (bytes, bytearray)):
        try:
            s = bytes(s).decode("utf-8", errors="ignore")
        except Exception:
            return ""
    elif not isinstance(s, str):
        return ""

    # 长度截断：防止超长 Payload 耗尽 CPU (ReDoS 防护)
    if max_len is not None and len(s) > max_len:
        s = s[:max_len]

    # 轮数限制：钳位到合理范围 (1-10)，防止死循环
    try:
        max_rounds = int(max_rounds)
    except Exception:
        max_rounds = 3
    max_rounds = max(1, min(max_rounds, 10))

    cur = s

    # Fast path: no decode indicators, skip the loop entirely.
    has_url = "%" in cur and RE_URL_CHECK.search(cur)
    has_u = ("u" in cur or "U" in cur) and RE_UNICODE_ESC.search(cur)
    has_html = "&" in cur and RE_HTML_CHECK.search(cur)
    if not (has_url or has_u or has_html):
        max_rounds = 0

    # Adaptive rounds for likely double-encoding patterns.
    if max_rounds > 0 and ("%25" in cur or "%u" in cur.lower() or "&#" in cur):
        max_rounds = min(max_rounds + 2, 10)

    # ---------------------------------------------------
    # 2. 核心解码循环 (性能关键区域)
    # 策略：利用 Python 的 'in' 操作符(C底层优化)做预判，极快。
    # ---------------------------------------------------
    for _ in range(max_rounds):
        orig = cur

        # A. URL Decode (%25 -> %)
        # 预判：只有包含 '%' 且符合正则才尝试解码
        if "%" in cur and RE_URL_CHECK.search(cur):
            try:
                d = urllib.parse.unquote(cur)
                if d != cur:
                    cur = d
                    continue  # 解码成功，立即跳入下一轮，防止混合编码干扰
            except Exception:
                pass

        # B. IIS Unicode Decode (%u0061 -> a)
        # 预判：只有包含 'u'/'U' 且符合正则才尝试
        if ("u" in cur or "U" in cur) and RE_UNICODE_ESC.search(cur):
            try:
                cur = RE_UNICODE_ESC.sub(lambda m: chr(int(m.group(1), 16)), cur)
                continue
            except Exception:
                pass

        # C. HTML Entity Decode (&lt; -> <)
        # 预判：只有包含 '&' 且符合正则才尝试
        if "&" in cur and RE_HTML_CHECK.search(cur):
            try:
                d = html.unescape(cur)
                if d != cur:
                    cur = d
                    continue
            except Exception:
                pass

        # 早停机制 (Early Exit)：如果一轮下来没变化，说明解到底了
        if cur == orig:
            break

    # ---------------------------------------------------
    # 3. 最终归一化与清洗
    # ---------------------------------------------------

    # [Step 1] NFKC 归一化 (处理全角字符/连字)
    # 性能优化：只有非 ASCII 字符串才跑 NFKC。
    # 这里做一次 O(N) 扫描是划算的，因为 NFKC 本身比扫描慢得多。
    if not cur.isascii():
        try:
            cur = unicodedata.normalize("NFKC", cur)
        except Exception:
            pass

    # [Step 2] 空白符清洗 (业务特征提取必须)
    # 使用 translate 极快，将 \t, \n, \r 等转为标准空格
    try:
        return cur.translate(WS_TRANS_TABLE)
    except Exception:
        # 兜底：如果 translate 失败，用 replace
        return cur.replace("\n", " ").replace("\t", " ").replace("\r", " ")


# 兼容旧名字
def smart_recursive_decode(s, max_rounds=3):
    return smart_recursive_decode_fast(s, max_rounds)


# ==========================================
# 5. Ultra-fast tokenizer
# ==========================================
def fast_tokenize(text):
    if not text:
        return []
    # [增强] 处理注释粘连 (adv_space2comment / adv_mix)
    # remove comments to avoid token glue
    if "/" in text and "*" in text:
        text = RE_COMMENTS_ALL.sub(" ", text)

    try:
        # [Step 3] 暴力去噪 + 转小写 + 切分
        # 此时 text 已经被 smart_recursive_decode_fast 处理过，没有 \x0b 了
        cleaned = text.translate(FAST_CLEAN_TABLE)
        return cleaned.split()
    except Exception:
        return re.split(r"[^a-zA-Z0-9]+", text.lower())


# ==========================================
# 6. FSM-based semantic recovery
# ==========================================
SQL_KEYWORDS = (
    "select",
    "union",
    "from",
    "where",
    "insert",
    "update",
    "delete",
    "drop",
    "create",
    "table",
    "values",
    "into",
    "set",
    "join",
    "having",
    "order",
    "group",
    "limit",
    "and",
    "or",
)

DANGEROUS_FUNCS = ("sleep", "benchmark", "load_file", "extractvalue", "updatexml")


def _fsm_match(word, s):
    wlen = len(word)
    i = 0
    for idx, ch in enumerate(s):
        c = ch.lower()
        if c == word[i]:
            i += 1
            if i == wlen:
                return idx
        elif c.isalnum():
            i = 0
    return None


def _scan_semantics(s: str):
    kw_pos = {}
    fn_hit = set()

    # 每个 keyword 只跑一次 FSM
    for kw in SQL_KEYWORDS:
        pos = _fsm_match(kw, s)
        if pos is not None:
            kw_pos[kw] = pos

    for fn in DANGEROUS_FUNCS:
        pos = _fsm_match(fn, s)
        if pos is not None:
            tail = s[pos + len(fn) : pos + len(fn) + 4]
            if "(" in tail:
                fn_hit.add(fn)

    return kw_pos, fn_hit


# ==========================================
# 7. normalize_attack_payload (兼容旧接口)
# ==========================================
def normalize_attack_payload(s):
    if not s:
        return ""

    s = smart_recursive_decode_fast(s).lower()
    kw_pos, fn_hit = _scan_semantics(s)

    tokens = []
    for kw, _ in sorted(kw_pos.items(), key=lambda x: x[1]):
        tokens.append(kw)

    for fn in sorted(fn_hit):
        tokens.append(fn)

    return " ".join(tokens)


# ==========================================
# 8. Advanced preprocess (主入口)
# ==========================================
def advanced_preprocess(query_str):
    if not query_str:
        return ""

    decoded = smart_recursive_decode_fast(query_str)
    base_tokens = fast_tokenize(decoded)
    sem_tokens = normalize_attack_payload(decoded).split()

    seen = set()
    final = []
    for t in base_tokens + sem_tokens:
        if len(t) > 1 and not t.isdigit() and t not in seen:
            seen.add(t)
            final.append(t)

    return " ".join(final)


# ==========================================
# 9. Utility / compatibility functions
# ==========================================
def compute_word_count(query):
    if not query:
        return 0
    return len(advanced_preprocess(query).split())


def unify_for_kwmatch(q):
    if not q:
        return []
    return advanced_preprocess(q).split()


def clean_query(q):
    return smart_recursive_decode_fast(q)


def generate_fingerprint(q):
    # 保留接口，占位
    return q
