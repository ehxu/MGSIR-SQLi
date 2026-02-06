"""
字符级深度学习特征提取器重构

保持原有接口：process_char_features()
基于统一的基类实现，提高代码复用性
"""

import os
import pickle
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from pathlib import Path

from ..base_extractor import BaseFeatureExtractor


# 字符级通常需要更长的序列
MAX_CHAR_LENGTH = 1000
# 常用字符集大小 (ASCII) 通常在 100 以内
VOCAB_SIZE = 200


class CharExtractor(BaseFeatureExtractor):
    """字符级特征提取器类"""
    
    def __init__(self, pipeline_name: str, base_dir: Path,
                 max_char_length: int = MAX_CHAR_LENGTH,
                 vocab_size: int = VOCAB_SIZE):
        super().__init__(pipeline_name, base_dir)
        self.max_char_length = max_char_length
        self.vocab_size = vocab_size
        self._extractor = None
        
    def fit(self, train_df: pd.DataFrame) -> None:
        """训练字符级Tokenizer"""
        train_texts = train_df["Query"].astype(str).tolist()
        
        # === 关键差异: char_level=True ===
        # 这样 'select' 会变成 [s, e, l, e, c, t] 的 id 序列
        self._extractor = Tokenizer(
            num_words=self.vocab_size,
            char_level=True,  # <--- 开启字符模式
            filters="",  # 字符级通常保留所有符号
            lower=True,
        )
        
        print(f"[Feature] Fitting Char Tokenizer on {len(train_texts)} samples...")
        self._extractor.fit_on_texts(train_texts)
        
        self.feature_dim = min(len(self._extractor.word_index), self.vocab_size)
        print(f"[Feature] Found {len(self._extractor.word_index)} unique characters.")
        
    def transform(self, df: pd.DataFrame):
        """转换文本为字符序列特征"""
        if self._extractor is None:
            raise RuntimeError("Extractor not fitted. Call fit() first.")
        
        texts = df["Query"].astype(str).tolist()
        sequences = self._extractor.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_char_length)
        return padded


# ==================== 原有接口函数（保持兼容） ====================
def process_char_features(
    train_df: pd.DataFrame, val_df: pd.DataFrame, pipeline_name: str, base_dir: Path
):
    """
    1. Fit Char Tokenizer on Train
    2. Convert Text to Character Sequence
    3. Pad Sequences
    4. Save Tokenizer
    
    保持原有接口不变，内部使用重构后的类实现
    """
    print(f"[Feature] Processing Char-Level Features for {pipeline_name}...")

    # 使用重构后的类实现
    extractor = CharExtractor(
        pipeline_name=pipeline_name,
        base_dir=base_dir,
        max_char_length=MAX_CHAR_LENGTH,
        vocab_size=VOCAB_SIZE
    )
    
    # 使用统一的fit_transform流程
    result = extractor.fit_transform(train_df, val_df)
    
    # 为了保持原有接口的标签处理逻辑，额外处理标签
    y_train = to_categorical(train_df["Label"].values, num_classes=2)
    y_val = to_categorical(val_df["Label"].values, num_classes=2)
    
    # 为了保持原有接口的保存逻辑，额外保存一次
    feature_dir = base_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    tok_path = feature_dir / f"tokenizer_char_{pipeline_name}.pkl"
    
    with open(tok_path, "wb") as f:
        pickle.dump(extractor._extractor, f)
    
    print(f"[Save] Tokenizer saved to: {tok_path}")
    
    # 返回结果（保持原有格式）
    return {
        "x_train": result["x_train"],
        "x_val": result["x_val"],
        "y_train": y_train,
        "y_val": y_val,
        "vocab_size": result["feature_dim"],
        "max_len": MAX_CHAR_LENGTH,
        "tokenizer_path": tok_path,
    }
