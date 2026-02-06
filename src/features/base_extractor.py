"""
特征提取器基类和通用工具

提供统一的特征提取接口，支持Bow, TFIDF, W2V, FastText, BERT等特征提取器的复用
"""

import os
import re
import io
import joblib
import numpy as np
import pandas as pd
import torch
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Union

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec, FastText
from transformers import BertTokenizer, BertModel


# ==================== 基类 ====================
class BaseFeatureExtractor(ABC):
    """特征提取器基类"""

    def __init__(self, pipeline_name: str, base_dir: Path):
        """
        初始化特征提取器

        Args:
            pipeline_name: 管道名称，用于标识和保存文件
            base_dir: 基础目录，特征文件将保存在 base_dir/features/ 下
        """
        self.pipeline_name = pipeline_name
        self.base_dir = Path(base_dir)
        self.feature_dir = self.base_dir / "features"
        self.feature_dir.mkdir(parents=True, exist_ok=True)

        # 存储提取器状态
        self.extractor_path = None
        self.feature_dim = None

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> None:
        """训练特征提取器（如果需要训练）"""
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> Union[np.ndarray, Any]:
        """转换数据为特征矩阵"""
        pass

    def fit_transform(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        统一的特征提取流程

        Returns:
            Dict包含x_train, x_val, y_train, y_val, feature_dim等
        """
        # 1. 训练特征提取器
        self.fit(train_df)

        # 2. 转换数据
        x_train = self.transform(train_df)
        x_val = self.transform(val_df)

        # 3. 获取标签
        y_train = train_df["Label"].values
        y_val = val_df["Label"].values

        # 4. 保存提取器
        self.save()

        return {
            "x_train": x_train,
            "x_val": x_val,
            "y_train": y_train,
            "y_val": y_val,
            "feature_dim": self.feature_dim,
            "extractor_path": self.extractor_path,
        }

    def save(self) -> None:
        """保存特征提取器"""
        if hasattr(self, "_extractor") and self._extractor is not None:
            self.extractor_path = (
                self.feature_dir / f"{self.pipeline_name}_extractor.pkl"
            )
            joblib.dump(self._extractor, self.extractor_path)

    def load(self, extractor_path: Optional[Path] = None) -> None:
        """加载特征提取器"""
        if extractor_path is None:
            extractor_path = self.extractor_path
        if extractor_path and extractor_path.exists():
            self._extractor = joblib.load(extractor_path)
        else:
            raise FileNotFoundError(f"Extractor not found: {extractor_path}")


# ==================== 通用工具函数 ====================
def get_pytorch_model_size_mb(model):
    """计算PyTorch模型大小（MB）"""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_bytes = buffer.getbuffer().nbytes
    return size_bytes / (1024 * 1024)


def sent2vec(tokens, model, vector_size=None):
    """将token列表转换为句向量（平均词向量）"""
    if not tokens:
        size = vector_size if vector_size else model.vector_size
        return np.zeros(size, dtype=np.float32)

    vecs = [model.wv[t] for t in tokens if t in model.wv]

    if not vecs:
        size = vector_size if vector_size else model.vector_size
        return np.zeros(size, dtype=np.float32)

    return np.mean(vecs, axis=0).astype(np.float32)


def preprocess_text(text: str) -> str:
    """基础预处理：转小写 + 去除首尾空格"""
    if not isinstance(text, str):
        text = str(text)
    return text.lower().strip()


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str):
    """正则分词"""
    return TOKEN_RE.findall(preprocess_text(text))
