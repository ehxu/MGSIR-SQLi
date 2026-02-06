"""
特征提取器模块

提供统一的特征提取接口，支持Bow, TFIDF, W2V, FastText, BERT等特征提取器
"""

# 基类和工具函数
from .base_extractor import BaseFeatureExtractor, sent2vec, tokenize, preprocess_text, get_pytorch_model_size_mb

# 传统特征提取器
from .bow_extractor import BoWExtractor, process_bow_features
from .tfidf_extractor import TFIDFExtractor, process_tfidf_features, tokenize as tfidf_tokenize
from .w2v_extractor import W2VExtractor, process_w2v_features
from .fasttext_extractor import FastTextExtractor, process_fasttext_features
from .bert_extractor import BERTExtractor, process_bert_features, load_bert_model, bert_encode_batch, preprocess_text as bert_preprocess

# 配置参数
from .bow_extractor import BOW_MAX_FEATURES, BOW_NGRAM_RANGE
from .tfidf_extractor import TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TOKEN_RE
from .w2v_extractor import W2V_SIZE, W2V_WINDOW, W2V_MIN_CNT, W2V_SG, W2V_EPOCHS
from .fasttext_extractor import FT_SIZE, FT_WINDOW, FT_MIN_CNT, FT_SG, FT_EPOCHS
from .bert_extractor import BERT_BASE_NAME, BERT_MAX_LEN, BERT_BATCH_SIZE

__all__ = [
    # 基类和工具
    'BaseFeatureExtractor',
    'sent2vec',
    'tokenize',
    'preprocess_text',
    'get_pytorch_model_size_mb',
    
    # BoW
    'BoWExtractor',
    'process_bow_features',
    'BOW_MAX_FEATURES',
    'BOW_NGRAM_RANGE',
    
    # TFIDF
    'TFIDFExtractor',
    'process_tfidf_features',
    'tfidf_tokenize',
    'TFIDF_MAX_FEATURES',
    'TFIDF_NGRAM_RANGE',
    'TOKEN_RE',
    
    # Word2Vec
    'W2VExtractor',
    'process_w2v_features',
    'W2V_SIZE',
    'W2V_WINDOW',
    'W2V_MIN_CNT',
    'W2V_SG',
    'W2V_EPOCHS',
    
    # FastText
    'FastTextExtractor',
    'process_fasttext_features',
    'FT_SIZE',
    'FT_WINDOW',
    'FT_MIN_CNT',
    'FT_SG',
    'FT_EPOCHS',
    
    # BERT
    'BERTExtractor',
    'process_bert_features',
    'load_bert_model',
    'bert_encode_batch',
    'bert_preprocess',
    'BERT_BASE_NAME',
    'BERT_MAX_LEN',
    'BERT_BATCH_SIZE',
]
