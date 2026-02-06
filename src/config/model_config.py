# src/config/model_config.py

# ==============================================================================
# 1. 默认固定参数 (极速模式)
# ==============================================================================
# 这一组参数用于所有不想搜索、直接跑训练的模型 (如 TF-IDF, BERT 等)
PARAMS_XGB_DEFAULT = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_estimators": 100,  # 默认树数量
    "max_depth": 6,  # 默认深度
    "learning_rate": 0.3,  # 默认学习率
    "gamma": 0,
    "reg_lambda": 1,
    "min_child_weight": 1,
    "n_jobs": -1,
    "random_state": 42,
    "tree_method": "hist",  # 【关键】开启直方图加速，训练快
}

# ==============================================================================
# 2. 全局搜索配置 (搜索模式)
# ==============================================================================
# 这一组配置仅在 get_xgb_config 返回 search=True 时生效
XGB_CONFIG = {
    "base_params": {
        "disable_default_eval_metric": True,
        "n_jobs": -1,
        "random_state": 42,
    },
    "param_space": {
        # 如果需要搜索，可以在这里定义范围
        "max_depth": [5, 6, 7, 8, 9, 10, 12],
        "learning_rate": [0.05, 0.1, 0.15],
        "n_estimators": [300, 500, 600],
        "gamma": [0, 0.1, 0.2],
        "reg_lambda": [1, 3, 5, 10],
        "min_child_weight": [1, 3, 5],
        # 【修复】必须是列表，否则 sklearn 会报错
        "tree_method": ["hist"],
    },
    "search": {
        "scoring": "f1",
        "cv": 3,
        "n_iter": 50,  # 搜索次数
        "n_jobs": -1,
        "random_state": 42,
        "verbose": 1,
        "return_train_score": True,
    },
    "use_best_threshold": True,
    "threshold_steps": 201,
}

MODELS_CONFIG = {
    "xgb": XGB_CONFIG,
}

# ==============================================================================
# 3. 核心开关函数
# ==============================================================================


def get_xgb_config(feature_name):
    """
    根据 feature_name 决定使用【固定参数】还是【搜索模式】。
    
    当前策略: 全部使用默认参数 (极速模式)
    原因: 
    - 超参搜索耗时过长，影响实验效率
    - 默认参数在SQL注入检测任务上表现良好
    - 避免TF-IDF等模型训练过慢
    
    未来扩展: 可针对特定模型开启搜索模式
    """
    fname = str(feature_name).lower()

    # TODO: 未来可针对特定模型开启搜索
    # if "mgsir" in fname:
    #     return { "search": True, "params": {} }
    
    return {
        "search": False,  # 关闭搜索
        "params": PARAMS_XGB_DEFAULT,  # 使用默认参数
    }
