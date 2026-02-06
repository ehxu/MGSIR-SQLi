# src/utils/math_ops.py
import numpy as np


def sigmoid(x):
    """Sigmoid 激活函数"""
    return 1.0 / (1.0 + np.exp(-x))

def logit(p, eps=1e-12):
    """Logit 变换 (Inverse Sigmoid)"""
    p = np.clip(p, eps, 1-eps)
    return np.log(p / (1.0-p))
