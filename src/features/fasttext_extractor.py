"""
FastText特征提取器重构

保持原有接口：process_fasttext_features()
基于统一的基类实现，提高代码复用性
"""

import numpy as np
import pandas as pd
from gensim.models import FastText
from pathlib import Path

from .base_extractor import BaseFeatureExtractor, sent2vec, tokenize


# 配置参数
FT_SIZE = 200
FT_WINDOW = 5
FT_MIN_CNT = 2
FT_SG = 1  # 1=skip-gram, 0=CBOW
FT_EPOCHS = 10
FT_WORKERS = 4
RANDOM_STATE = 42


class FastTextExtractor(BaseFeatureExtractor):
    """FastText特征提取器类"""
    
    def __init__(self, pipeline_name: str, base_dir: Path,
                 vector_size: int = FT_SIZE,
                 window: int = FT_WINDOW,
                 min_count: int = FT_MIN_CNT,
                 sg: int = FT_SG,
                 epochs: int = FT_EPOCHS):
        super().__init__(pipeline_name, base_dir)
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.epochs = epochs
        self._extractor = None
        
    def fit(self, train_df: pd.DataFrame) -> None:
        """训练FastText模型"""
        train_tokens = train_df["Query"].fillna("").map(tokenize).tolist()
        
        print(f"[Feature] Training FastText on {len(train_tokens)} samples...")
        self._extractor = FastText(
            sentences=train_tokens,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            sg=self.sg,
            workers=FT_WORKERS,
            epochs=self.epochs,
            seed=RANDOM_STATE,
        )
        
        self.feature_dim = self.vector_size
        print(f"[FastText] Model trained, vector size: {self.feature_dim}")
        
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """转换文本为FastText特征"""
        if self._extractor is None:
            raise RuntimeError("Extractor not fitted. Call fit() first.")
        
        tokens_list = df["Query"].fillna("").map(tokenize).tolist()
        features = np.vstack([sent2vec(t, self._extractor) for t in tokens_list])
        return features


# ==================== 原有接口函数（保持兼容） ====================
def process_fasttext_features(
    train_df: pd.DataFrame, val_df: pd.DataFrame, pipeline_name: str, base_dir: Path
):
    """
    1. Tokenize
    2. Train FastText on Training Data
    3. Convert sentences to vectors (sent2vec)
    4. Save FastText model
    
    保持原有接口不变，内部使用重构后的类实现
    """
    print(f"[Feature] Processing FastText for {pipeline_name}...")

    # 使用重构后的类实现
    extractor = FastTextExtractor(
        pipeline_name=pipeline_name,
        base_dir=base_dir,
        vector_size=FT_SIZE,
        window=FT_WINDOW,
        min_count=FT_MIN_CNT,
        sg=FT_SG,
        epochs=FT_EPOCHS
    )
    
    # 使用统一的fit_transform流程
    result = extractor.fit_transform(train_df, val_df)
    
    # 为了保持原有接口的保存逻辑，额外保存一次
    feature_dir = base_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    ft_path = feature_dir / f"fasttext_{pipeline_name}.model"
    extractor._extractor.save(str(ft_path))
    print(f"[Save] FastText model saved to: {ft_path}")
    
    # 返回结果（保持原有格式）
    return {
        "x_train": result["x_train"],
        "x_val": result["x_val"],
        "y_train": result["y_train"],
        "y_val": result["y_val"],
        "ft_path": ft_path,
        "feature_dim": result["feature_dim"],
    }
