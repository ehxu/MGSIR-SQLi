"""
Word2Vec特征提取器重构

保持原有接口：process_w2v_features()
基于统一的基类实现，提高代码复用性
"""

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from pathlib import Path

from .base_extractor import BaseFeatureExtractor, sent2vec, tokenize


# 配置参数
W2V_SIZE = 200
W2V_WINDOW = 5
W2V_MIN_CNT = 2
W2V_SG = 1
W2V_EPOCHS = 10
W2V_WORKERS = 4
RANDOM_STATE = 42


class W2VExtractor(BaseFeatureExtractor):
    """Word2Vec特征提取器类"""

    def __init__(
        self,
        pipeline_name: str,
        base_dir: Path,
        vector_size: int = W2V_SIZE,
        window: int = W2V_WINDOW,
        min_count: int = W2V_MIN_CNT,
        sg: int = W2V_SG,
        epochs: int = W2V_EPOCHS,
    ):
        super().__init__(pipeline_name, base_dir)
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.epochs = epochs
        self._extractor = None

    def fit(self, train_df: pd.DataFrame) -> None:
        """训练Word2Vec模型"""
        train_tokens = train_df["Query"].fillna("").map(tokenize).tolist()

        print(f"[Feature] Training Word2Vec on {len(train_tokens)} samples...")
        self._extractor = Word2Vec(
            sentences=train_tokens,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            sg=self.sg,
            workers=W2V_WORKERS,
            epochs=self.epochs,
            seed=RANDOM_STATE,
        )

        self.feature_dim = self.vector_size
        print(f"[W2V] Model trained, vector size: {self.feature_dim}")

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """转换文本为Word2Vec特征"""
        if self._extractor is None:
            raise RuntimeError("Extractor not fitted. Call fit() first.")

        tokens_list = df["Query"].fillna("").map(tokenize).tolist()
        features = np.vstack([sent2vec(t, self._extractor) for t in tokens_list])
        return features


# ==================== 原有接口函数（保持兼容） ====================
def process_w2v_features(
    train_df: pd.DataFrame, val_df: pd.DataFrame, pipeline_name: str, base_dir: Path
):
    """
    1. Tokenize
    2. Train Word2Vec on Training Data
    3. Convert sentences to vectors (sent2vec)
    4. Save Word2Vec model

    保持原有接口不变，内部使用重构后的类实现
    """
    print(f"[Feature] Processing Word2Vec for {pipeline_name}...")

    # 使用重构后的类实现
    extractor = W2VExtractor(
        pipeline_name=pipeline_name,
        base_dir=base_dir,
        vector_size=W2V_SIZE,
        window=W2V_WINDOW,
        min_count=W2V_MIN_CNT,
        sg=W2V_SG,
        epochs=W2V_EPOCHS,
    )

    # 使用统一的fit_transform流程
    result = extractor.fit_transform(train_df, val_df)

    # 为了保持原有接口的保存逻辑，额外保存一次
    feature_dir = base_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    w2v_path = feature_dir / f"w2v_{pipeline_name}.model"
    extractor._extractor.save(str(w2v_path))
    print(f"[Save] Word2Vec model saved to: {w2v_path}")

    # 返回结果（保持原有格式）
    return {
        "x_train": result["x_train"],
        "x_val": result["x_val"],
        "y_train": result["y_train"],
        "y_val": result["y_val"],
        "w2v_path": w2v_path,
        "feature_dim": result["feature_dim"],
    }
