"""
XGBoost训练器重构

保持原有接口：train_and_save_xgb_model()
基于统一的基类实现，提高代码复用性
"""

import os
import joblib
import time
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

from pathlib import Path
from src.config.model_config import get_xgb_config
from src.config.paths import get_pipeline_paths


# ==================== 工具函数 ====================
def find_best_threshold(proba_vals, true_labels, steps=101):
    """在 0~1 搜索最佳阈值以最大化 F1"""
    best_thr, best_f1 = 0.5, 0.0

    for thr in np.linspace(0, 1, steps):
        preds = (proba_vals >= thr).astype(int)
        f1 = f1_score(true_labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    return best_thr, best_f1


# ==================== 训练器基类 ====================
class BaseTrainer:
    """模型训练器基类"""
    
    def __init__(self, pipeline_name: str, base_dir: Path, logger=None):
        self.pipeline_name = pipeline_name
        self.base_dir = Path(base_dir)
        self.logger = logger
        
        # 路径设置
        self.model_dir = self.base_dir / "model"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_file = self.model_dir / f"{pipeline_name}_model.pkl"
        self.thr_file = self.model_dir / f"{pipeline_name}_threshold.pkl"
        
    def log(self, message: str):
        """日志记录"""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
    
    def _create_model(self, **kwargs):
        """创建模型实例"""
        return XGBClassifier(**kwargs)
    
    def _search_params(self, train_x, train_y, param_space, search_cfg):
        """参数搜索"""
        model = self._create_model(**search_cfg.get('base_params', {}))
        searcher = RandomizedSearchCV(model, param_space, **search_cfg)
        searcher.fit(train_x, train_y)
        return searcher.best_params_
    
    def train(self, 
              train_x, train_y, 
              val_x, val_y,
              base_params: dict,
              search_config: dict = None,
              param_space: dict = None,
              use_best_threshold: bool = False,
              threshold_steps: int = 101) -> dict:
        """统一的模型训练流程"""
        start_total_time = time.time()
        self.log(f"\n{'='*20} XGBoost Trainer ({self.pipeline_name}) 启动 {'='*20}")
        self.log(f"[CONFIG] 基础参数: {base_params}")
        
        # 参数搜索或使用固定参数
        best_params = {}
        search_duration = 0.0
        
        if search_config and param_space:
            self.log(f"[INFO] 开始参数搜索，搜索配置: {search_config}")
            start_search_time = time.time()
            
            best_params = self._search_params(train_x, train_y, param_space, search_config)
            
            search_duration = time.time() - start_search_time
            self.log(f"[INFO] 搜索完成，耗时: {search_duration:.2f}s")
            self.log(f"[INFO] 最优参数: {best_params}")
        else:
            self.log(f"🚀 [Fast Mode] 使用预设固定参数，跳过搜索")
        
        # 合并参数
        final_params = base_params.copy()
        final_params.update(best_params)
        self.log(f"[INFO] 最终训练参数: {final_params}")
        
        # 训练最终模型
        self.log(f"[INFO] 开始训练最终模型...")
        start_train_time = time.time()
        
        model = self._create_model(**final_params)
        model.fit(
            train_x, train_y, 
            eval_set=[(train_x, train_y), (val_x, val_y)], 
            verbose=False
        )
        
        train_duration = time.time() - start_train_time
        self.log(f"[INFO] 最终模型训练耗时: {train_duration:.2f}s")
        
        # 评估模型
        metrics = self._evaluate_model(model, train_x, train_y, val_x, val_y)
        
        # 阈值搜索
        best_threshold = None
        if use_best_threshold:
            self.log("[INFO] 搜索最佳阈值...")
            val_proba = model.predict_proba(val_x)[:, 1]
            best_threshold, best_f1 = find_best_threshold(val_proba, val_y, threshold_steps)
            metrics["best_threshold"] = best_threshold
            metrics["f1_val_best_threshold"] = best_f1
            self.log(f"[INFO] 最佳阈值: {best_threshold:.4f} -> F1: {best_f1:.4f}")
        
        # 保存模型
        joblib.dump(model, self.model_file)
        self.log(f"[SAVE] 模型已保存：{self.model_file}")
        
        # 计算模型大小
        try:
            model_size_bytes = os.path.getsize(self.model_file)
            model_size_mb = model_size_bytes / (1024 * 1024)
            self.log(f"[INFO] 模型大小: {model_size_mb:.2f} MB")
        except Exception as e:
            self.log(f"[WARN] 无法计算模型大小: {e}")
            model_size_mb = -1.0
        
        # 保存阈值
        if best_threshold is not None:
            joblib.dump(best_threshold, self.thr_file)
            self.log(f"[SAVE] 阈值已保存：{self.thr_file}")
        
        total_duration = time.time() - start_total_time
        self.log(f"========== 训练完成 (总耗时: {total_duration:.2f}s) ==========\n")
        self.log(f"[FULL METRICS] {metrics}")
        
        return {
            "model_name": "xgb",
            "best_params": best_params,
            "metrics": metrics,
            "model_path": str(self.model_file),
            "thr_path": str(self.thr_file),
            "threshold": best_threshold,
            "val_f1": metrics.get("val_default_f1", 0.0),
            "val_acc": metrics.get("val_default_acc", 0.0),
            "train_time_sec": train_duration + search_duration,
            "model_size_mb": model_size_mb,
        }
    
    def _evaluate_model(self, model, train_x, train_y, val_x, val_y) -> dict:
        """评估模型性能"""
        # 预测
        pred_train = model.predict(train_x)
        pred_val = model.predict(val_x)
        
        # 计算指标
        def calc_metrics(y_true, y_pred, prefix):
            return {
                f"{prefix}_f1": f1_score(y_true, y_pred),
                f"{prefix}_acc": accuracy_score(y_true, y_pred),
                f"{prefix}_prec": precision_score(y_true, y_pred),
                f"{prefix}_rec": recall_score(y_true, y_pred),
            }
        
        metrics_train = calc_metrics(train_y, pred_train, "train_default")
        metrics_val = calc_metrics(val_y, pred_val, "val_default")
        
        # 合并指标
        metrics = {**metrics_train, **metrics_val}
        
        # 打印关键指标
        self.log(f"{'='*20} [RESULT] Train Metrics (Default) {'='*20}")
        self.log(f"F1 Score : {metrics['train_default_f1']:.4f} | Accuracy : {metrics['train_default_acc']:.4f}")
        
        self.log(f"{'='*20} [RESULT] Validation Metrics (Default) {'='*20}")
        self.log(f"F1 Score : {metrics['val_default_f1']:.4f} | Accuracy : {metrics['val_default_acc']:.4f}")
        self.log(f"Precision: {metrics['val_default_prec']:.4f} | Recall   : {metrics['val_default_rec']:.4f}")
        
        return metrics


# ==================== 原有接口函数（保持兼容） ====================
def train_and_save_xgb_model(
    train_x,
    train_y,
    val_x,
    val_y,
    model_cfg,
    pipeline_name,
    logger=None,
    sub_dir: str = None,
):
    """
    XGBoost训练主函数
    
    保持原有接口不变，内部使用重构后的类实现
    """
    # 获取 pipeline paths
    paths = get_pipeline_paths(pipeline_name, sub_dir=sub_dir)
    
    # 创建训练器实例
    trainer = BaseTrainer(pipeline_name, paths.base_dir, logger)
    
    # 获取配置
    base_params = model_cfg["base_params"]
    specific_cfg = get_xgb_config(pipeline_name)
    
    # 准备参数
    search_config = None
    param_space = None
    
    if specific_cfg.get("search") is False:
        # 极速模式：使用固定参数
        best_params = specific_cfg["params"]
        # 合并参数（保持原有逻辑）
        final_params = base_params.copy()
        final_params.update(best_params)
        
        # 直接训练，不搜索
        return trainer.train(
            train_x, train_y, val_x, val_y,
            base_params=final_params,
            search_config=None,
            param_space=None,
            use_best_threshold=model_cfg.get("use_best_threshold", False),
            threshold_steps=model_cfg.get("threshold_steps", 101)
        )
    else:
        # 搜索模式
        search_config = model_cfg["search"]
        param_space = model_cfg["param_space"]
        
        return trainer.train(
            train_x, train_y, val_x, val_y,
            base_params=base_params,
            search_config=search_config,
            param_space=param_space,
            use_best_threshold=model_cfg.get("use_best_threshold", False),
            threshold_steps=model_cfg.get("threshold_steps", 101)
        )
