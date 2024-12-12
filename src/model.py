import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool

from cfg import cfg

class Model:
    """
    モデルクラス

    Args:
        params (dict): モデルのハイパーパラメータ
        model_type (str): モデルの種類 ("lgb" or "catboost")
    """
    def __init__(self, params: dict, model_type: str, categorical_cols: list = None):
        self.params = params
        self.model_type = model_type
        self.categorical_cols = categorical_cols
        assert self.model_type in ["lgb", "catboost"], "model_type is invalid"

        self.model = None
        self.threshold = None

    def get_cv(self, X: pd.DataFrame, y: np.ndarray, n_splits: int = 5, random_state: int = cfg.seed):
        """
        クロスバリデーションの分割方法を取得

        Args:
            X (pd.DataFrame): 学習データ
            y (np.ndarray): 目的変数
            n_splits (int): 分割数
            random_state (int): 乱数シード

        Returns:
            list: trainとvalidのインデックスのリスト
        """
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return list(cv.split(X, y))
    
    def get_model(self):
        """
        モデルを取得

        Returns:
            model: モデル
        """
        if self.model_type == "lgb":
            return LightGBM(self.params)
        elif self.model_type == "catboost":
            if not self.categorical_cols:
                raise ValueError("categorical_cols is required for CatBoost")
            return Catboost(self.params, self.categorical_cols)
    
    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """
        モデルの学習

        Args:
            X (pd.DataFrame): 学習データ
            y (np.ndarray): 目的変数
        """
        cv = self.get_cv(X, y)
        self.model = self.get_model()
        self.model.fit(X, y, cv)

        self.calc_f1(y)
        
    def calc_f1(self, y: np.ndarray):
        self.threshold = self.model.get_threshold(y)
        f1 = f1_score(y, self.model.oof > self.threshold)
        print(f"F1: {f1:.4f}")

    def predict(self, X: pd.DataFrame):
        """
        テストデータに対する予測

        Args:
            X (pd.DataFrame): テストデータ

        Returns:
            np.ndarray: 予測結果
        """
        assert self.model, "Model is not fitted yet"
        y_pred = self.model.predict(X)
        y_pred_hard = (y_pred > self.threshold).astype(int)
        return y_pred, y_pred_hard
    

class ModelBase:
    """
    モデルの基底クラス

    Args:
        params (dict): モデルのハイパーパラメータ
    """
    def __init__(self, params: dict):
        self.params = params

        self.models = []
        self.score = None
        self.oof = None
    
    def fit(self, X: pd.DataFrame, y: np.ndarray, cv: list):
        """
        モデルの学習

        Args:
            X (pd.DataFrame): 学習データ
            y (np.ndarray): 目的変数
            cv (list): trainとvalidのインデックスのリスト
        """
        raise NotImplementedError
    
    def predict(self, X: pd.DataFrame):
        """
        テストデータに対する予測

        Args:
            X (pd.DataFrame): テストデータ

        Returns:
            np.ndarray: 予測結果
        """
        y_pred = np.zeros(X.shape[0], dtype=np.float32)
        for model in self.models:
            y_pred += model.predict_proba(X)[:, 1]
        y_pred /= len(self.models)
        return y_pred
    
    def get_threshold(self, y: np.ndarray):
        """
        F1が最良の閾値を決定する

        Args:
            y (np.ndarray): 目的変数

        Returns:
            float: 閾値
        """
        assert self.oof is not None, "Model is not fitted yet"
        best_score = 0
        best_threshold = 0
        for threshold in np.arange(0, 1, 0.01):
            score = f1_score(y, self.oof > threshold)
            if score > best_score:
                best_score = score
                best_threshold = threshold
        return best_threshold


class LightGBM(ModelBase):
    """
    LightGBMをクロスバリデーションで学習する & 予測する

    Args:
        params (dict): モデルのハイパーパラメータ
    """
    def __init__(self, params: dict):
        super().__init__(params)
    
    def fit(self, X: pd.DataFrame, y: np.ndarray, cv: list):
        """
        モデルの学習

        Args:
            X (pd.DataFrame): 学習データ
            y (np.ndarray): 目的変数
            cv (list): trainとvalidのインデックスのリスト
        """
        self.oof = np.zeros_like(y, dtype=np.float32)

        for fold_id, (train_index, valid_index) in enumerate(cv):
            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]

            model = lgb.LGBMClassifier(**self.params)
            model.fit(
                X_train, 
                y_train, 
                eval_set=[(X_valid, y_valid)], 
                callbacks=[
                    lgb.early_stopping(cfg.early_stopping_rounds, verbose=False), 
                    lgb.log_evaluation(250)
                ],
            )
            self.models.append(model)

            self.oof[valid_index] = model.predict_proba(X_valid)[:, 1]
            score = roc_auc_score(y_valid, self.oof[valid_index])
            print(f"Fold {fold_id}: {score:.4f}")
        
        self.score = roc_auc_score(y, self.oof)
        print(f"AUC: {self.score:.4f}")


class Catboost(ModelBase):
    """
    CatBoostをクロスバリデーションで学習する & 予測する

    Args:
        params (dict): モデルのハイパーパラメータ
    """
    def __init__(self, params: dict, categorical_cols: list):
        super().__init__(params)
        self.categorical_cols = categorical_cols
    
    def fit(self, X: pd.DataFrame, y: np.ndarray, cv: list):
        """
        モデルの学習

        Args:
            X (pd.DataFrame): 学習データ
            y (np.ndarray): 目的変数
            cv (list): trainとvalidのインデックスのリスト
        """
        self.oof = np.zeros_like(y, dtype=np.float32)

        for fold_id, (train_index, valid_index) in enumerate(cv):
            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]

            train_pool = Pool(
                X_train, 
                y_train, 
                cat_features=self.categorical_cols, 
                feature_names=list(X_train)
            )
            
            valid_pool = Pool(
                X_valid, 
                y_valid, 
                cat_features=self.categorical_cols, 
                feature_names=list(X_train)
            )

            model = CatBoostClassifier(**self.params)
            model.fit(
                train_pool,
                eval_set=valid_pool,
                early_stopping_rounds=cfg.early_stopping_rounds,
                use_best_model=True,
                verbose=250,
            )
            self.models.append(model)

            self.oof[valid_index] = model.predict_proba(X_valid)[:, 1]
            score = roc_auc_score(y_valid, self.oof[valid_index])
            print(f"Fold {fold_id}: {score:.4f}")
        
        self.score = roc_auc_score(y, self.oof)
        print(f"AUC: {self.score:.4f}")