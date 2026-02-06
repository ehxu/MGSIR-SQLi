"""
BERT特征提取器重构

保持原有接口：process_bert_features()
基于统一的基类实现，提高代码复用性
"""

import io
import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
from pathlib import Path
from typing import Optional

from .base_extractor import BaseFeatureExtractor, get_pytorch_model_size_mb
from src.utils.runtime_env import get_benchmark_device, get_benchmark_threads


# 配置参数
BERT_BASE_NAME = "bert-base-uncased"
BERT_MAX_LEN = 64
BERT_BATCH_SIZE = 32


def preprocess_text(text: str) -> str:
    """基础预处理：转小写 + 去除首尾空格"""
    if not isinstance(text, str):
        text = str(text)
    return text.lower().strip()


class BERTExtractor(BaseFeatureExtractor):
    """BERT特征提取器类"""

    def __init__(
        self,
        pipeline_name: str,
        base_dir: Path,
        model_name: str = BERT_BASE_NAME,
        max_len: int = BERT_MAX_LEN,
        batch_size: int = BERT_BATCH_SIZE,
    ):
        super().__init__(pipeline_name, base_dir)
        self.model_name = model_name
        self.max_len = max_len
        self.batch_size = batch_size

        self._tokenizer = None
        self._model = None
        self._device = None

    def _setup_device(self):
        """设置计算设备"""
        # Fair benchmarking: force CPU only if explicitly requested
        if get_benchmark_device() == "cpu":
            self._device = torch.device("cpu")
            return

        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")

    def _load_model(self):
        """加载BERT模型和分词器"""
        print(f"[Feature] Loading {self.model_name} on {self._device}...")
        self._tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self._model = BertModel.from_pretrained(self.model_name)
        self._model.to(self._device)
        self._model.eval()

    def _encode_batch(self, texts, show_progress=True):
        """批量编码文本为BERT向量"""
        all_embeddings = []
        total = len(texts)

        for i in range(0, total, self.batch_size):
            batch_texts = texts[i : i + self.batch_size]

            inputs = self._tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )

            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]

            all_embeddings.append(cls_embeddings.cpu().numpy())

        return np.vstack(all_embeddings).astype(np.float32)

    def fit(self, train_df: pd.DataFrame) -> None:
        """加载BERT模型（预训练模型，无需训练）"""
        self._setup_device()
        self._load_model()

        # 获取隐藏层大小
        self.feature_dim = self._model.config.hidden_size
        print(f"[BERT] Hidden Size: {self.feature_dim}")

        # 计算模型大小
        model_size_mb = self._get_model_size()
        print(f"[BERT] Model Size: {model_size_mb:.2f} MB")

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """转换文本为BERT特征"""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call fit() first.")

        texts = df["Query"].fillna("").map(preprocess_text).tolist()
        features = self._encode_batch(texts, show_progress=len(texts) > 1)
        return features

    def _get_model_size(self) -> float:
        """计算模型大小（MB）"""
        if self._model is None:
            return 0.0
        return get_pytorch_model_size_mb(self._model)

    def save(self) -> None:
        """BERT提取器通常不需要保存（使用预训练模型）"""
        pass

    def load(self, extractor_path: Optional[Path] = None) -> None:
        """加载BERT提取器"""
        self._setup_device()
        self._load_model()


# ==================== 原有接口函数（保持兼容） ====================
def load_bert_model(device=None):
    """加载BERT模型和分词器"""
    if device is None:
        # Fair benchmarking: force CPU only if explicitly requested
        if get_benchmark_device() == "cpu":
            device = torch.device("cpu")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    print(f"[Feature] Loading BERT ({BERT_BASE_NAME}) on {device}...")
    tokenizer = BertTokenizer.from_pretrained(BERT_BASE_NAME)
    model = BertModel.from_pretrained(BERT_BASE_NAME)
    model.to(device)
    model.eval()

    # Optional: set torch thread count if requested
    threads = get_benchmark_threads()
    if threads is not None and threads > 0:
        try:
            torch.set_num_threads(threads)
            torch.set_num_interop_threads(threads)
        except Exception:
            pass

    return tokenizer, model, device


def bert_encode_batch(
    texts, tokenizer, model, device, batch_size=BERT_BATCH_SIZE, max_len=BERT_MAX_LEN
):
    """批量将句子编码为 BERT CLS 向量"""
    all_embeddings = []
    total = len(texts)
    show_progress = total > 1

    for i in range(0, total, batch_size):
        batch_texts = texts[i : i + batch_size]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]

        all_embeddings.append(cls_embeddings.cpu().numpy())

    return np.vstack(all_embeddings).astype(np.float32)


def process_bert_features(
    train_df: pd.DataFrame, val_df: pd.DataFrame, pipeline_name: str, base_dir: Path
):
    """
    1. Load BERT
    2. Encode Train/Val text to vectors
    3. Return features (No need to save BERT model itself, as we use pretrained)

    保持原有接口不变，内部使用重构后的类实现
    """
    print(f"[Feature] Processing BERT for {pipeline_name}...")

    # 使用重构后的类实现
    extractor = BERTExtractor(
        pipeline_name=pipeline_name,
        base_dir=base_dir,
        model_name=BERT_BASE_NAME,
        max_len=BERT_MAX_LEN,
        batch_size=BERT_BATCH_SIZE,
    )

    # 使用统一的fit_transform流程
    result = extractor.fit_transform(train_df, val_df)

    # 获取额外信息
    hidden_size = result["feature_dim"]
    bert_size_mb = extractor._get_model_size()

    print(f"[INFO] Detected BERT Hidden Size: {hidden_size}")
    print(f"[INFO] BERT Backbone Size: {bert_size_mb:.2f} MB")

    # 返回结果（保持原有格式）
    return {
        "x_train": result["x_train"],
        "x_val": result["x_val"],
        "y_train": result["y_train"],
        "y_val": result["y_val"],
        "feature_dim": hidden_size,
        "bert_size_mb": bert_size_mb,
    }
