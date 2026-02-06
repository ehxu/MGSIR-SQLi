"""
BoW特征提取器重构

保持原有接口：process_bow_features()
基于统一的基类实现，提高代码复用性
"""

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path

from .base_extractor import BaseFeatureExtractor


# 配置参数
BOW_MAX_FEATURES = 5000
BOW_NGRAM_RANGE = (1, 1)


class BoWExtractor(BaseFeatureExtractor):
    """BoW特征提取器类"""
    
    def __init__(self, pipeline_name: str, base_dir: Path, 
                 max_features: int = 5000, 
                 ngram_range: tuple = (1, 1)):
        super().__init__(pipeline_name, base_dir)
        self.max_features = max_features
        self.ngram_range = ngram_range
        self._extractor = None
        
    def fit(self, train_df: pd.DataFrame) -> None:
        """训练BoW向量器"""
        train_corpus = train_df["Query"].values
        
        self._extractor = CountVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            token_pattern=r"[A-Za-z0-9_]+",
            lowercase=True,
        )
        
        print(f"[Feature] Fitting CountVectorizer on {len(train_corpus)} samples...")
        self._extractor.fit(train_corpus)
        self.feature_dim = len(self._extractor.vocabulary_)
        print(f"[Feature] Vocabulary Size: {self.feature_dim}")
        
    def transform(self, df: pd.DataFrame):
        """转换文本为BoW特征"""
        if self._extractor is None:
            raise RuntimeError("Extractor not fitted. Call fit() first.")
        
        corpus = df["Query"].values
        features = self._extractor.transform(corpus)
        return features


# ==================== 原有接口函数（保持兼容） ====================
def process_bow_features(
    train_df: pd.DataFrame, val_df: pd.DataFrame, pipeline_name: str, base_dir: Path
):
    """
    1. 对 Train/Val 进行预处理
    2. Fit CountVectorizer on Train
    3. Transform Train & Val
    4. 保存 Vectorizer
    
    保持原有接口不变，内部使用重构后的类实现
    """
    print(f"[Feature] Processing BoW for {pipeline_name}...")

    # 使用重构后的类实现
    extractor = BoWExtractor(
        pipeline_name=pipeline_name,
        base_dir=base_dir,
        max_features=5000,
        ngram_range=(1, 1)
    )
    
    # 使用统一的fit_transform流程
    result = extractor.fit_transform(train_df, val_df)
    
    # 为了保持原有接口的保存逻辑，额外保存一次
    feature_dir = base_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    vectorizer_path = feature_dir / f"vectorizer_{pipeline_name}.pkl"
    joblib.dump(extractor._extractor, vectorizer_path)
    print(f"[Save] Vectorizer saved to: {vectorizer_path}")
    
    # 返回结果（保持原有格式）
    return {
        "x_train": result["x_train"],
        "x_val": result["x_val"],
        "y_train": result["y_train"],
        "y_val": result["y_val"],
        "vectorizer_path": vectorizer_path,
        "feature_dim": result["feature_dim"],
    }
