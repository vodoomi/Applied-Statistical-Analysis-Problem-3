class CFG:
    seed = 88
    early_stopping_rounds = 100
    lgbm_params = {
        "objective": "binary",
        "metric": "auc",
        "n_estimators": 10000,
        "learning_rate": 0.1,
        "max_depth": 5,
        "num_leaves": 31,
        "colsample_bytree": 0.7,
        "importance_type": "gain",
        "boosting_type" :"gbdt",
        "random_state": seed,
        "verbose" : -1, 
        "n_jobs" : -1,
    }
    cat_params = {
        "iterations" : 10000,
        "learning_rate": 0.1,
        "l2_leaf_reg": 3.0, 
        "depth": 5,
        "eval_metric" : "AUC",
        "one_hot_max_size" : 3,
        "random_state": seed,
    }

cfg = CFG()