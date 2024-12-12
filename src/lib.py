import joblib

import polars as pl
from sklearn.preprocessing import StandardScaler, LabelEncoder


class CustomStandardScaler():
    """
    標準化を行うクラス

    Args:
        cols(list) : 標準化するカラム名のリスト
    """
    def __init__(self, cols: list):
        self.scaler = StandardScaler()
        self.cols = cols

    def fit(self, df: pl.DataFrame):
        self.scaler.fit(df[self.cols].to_numpy())
        # 保存が必要ならコメントアウトを外す
        # joblib.dump(self.scaler, "scaler.pkl")

    def transform(self, df: pl.DataFrame, scaler=None):
        if scaler is not None:
            self.scaler = scaler
        return pl.DataFrame(self.scaler.transform(df[self.cols].to_numpy()), self.cols)

    def fit_transform(self, df: pl.DataFrame):
        self.fit(df)
        return self.transform(df)

    def inverse_transform(self, df: pl.DataFrame):
        return pl.DataFrame(self.scaler.inverse_transform(df[self.cols].to_numpy()), self.cols)
    

class CustomEncoder():
    """
    カテゴリ変数をエンコードするクラス

    Args:
        cat_cols(list) : エンコードするカラム名のリスト
        encoding(str) : エンコードする方法 ("label" or "onehot")
    """
    def __init__(self, cat_cols: list, encoding="onehot"):
        self.cat_cols = cat_cols
        self.encoding = encoding

    def fit(self, df: pl.DataFrame):
        if self.encoding == "label":
            self.encoders = {col: LabelEncoder().fit(df[col].to_numpy()) for col in self.cat_cols}
            # 保存が必要ならコメントアウトを外す
            # joblib.dump(self.encoders, "encoders.pkl")

        elif self.encoding == "onehot":
            self.use_cols = df[self.cat_cols].to_dummies(drop_first=True).columns
        

    def transform(self, df: pl.DataFrame):
        if self.encoding == "label":
            df = (
                df
                .with_columns([
                    pl.Series(col, self.encoders[col].transform(df[col].to_numpy()))
                    for col in self.cat_cols
                ])
                .select(self.cat_cols)
            )
        elif self.encoding == "onehot":
            df = (
                df
                .select(self.cat_cols)
                .to_dummies()
                .select(self.use_cols)
            )
        return df

    def fit_transform(self, df: pl.DataFrame):
        self.fit(df)
        return self.transform(df)