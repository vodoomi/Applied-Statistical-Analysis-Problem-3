import polars as pl
from lib import CustomStandardScaler, CustomEncoder

class Preprocessor():
    """ 
    DataFrameを前処理するクラス

    Args:
        scale(bool) : 標準化するかどうか
        encoding(str) : エンコードする方法 ("label" or "onehot")

    Attributes:
        encoder : CustomEncoderのインスタンス
        scaler : CustomStandardScalerのインスタンス
    """
    def __init__(self, scale=True, encoding="label"):
        self.scale = scale
        self.encoding = encoding

        self.encoder = None
        self.scaler = None

        assert encoding in ["label", "onehot"], "encoding must be 'label' or 'onehot'"

    def preprocess_data(self, df: pl.DataFrame, mode="train") -> pl.DataFrame:
        """
        DataFrameを前処理する

        Args:
            df(pl.DataFrame) : 前処理するデータ
            mode(str) : 前処理のモード ("train" or "test")

        Returns:
            pl.DataFrame : 前処理したデータ (目的変数列は除かれる)
        """
        assert mode in ["train", "test"], "mode must be 'train' or 'test'"
        
        # train_testにしか含まれないカテゴリがdefault列にあるので、それに対処
        # Label Encodingの場合は、問題ないので、OneHot Encodingの場合のみ対処
        if self.encoding == "onehot":
            df = (
                df
                .with_columns(
                    pl.col("default").str.replace("yes", "unknown").alias("default")
                )
            )

        # カテゴリ変数をエンコード
        if mode == "train":
            cat_cols = [col for col in df.columns if df[col].dtype == pl.String]
            cat_cols.remove("y")
            self.encoder = CustomEncoder(cat_cols, self.encoding)
            cat_df = self.encoder.fit_transform(df)
        elif mode == "test":
            cat_df = self.encoder.transform(df)

        # 数値変数の標準化
        if self.scale:
            if mode == "train":
                num_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Int64]]
                self.scaler = CustomStandardScaler(num_cols)
                num_df = self.scaler.fit_transform(df)
            elif mode == "test":
                num_df = self.scaler.transform(df)
        else:
            num_df = df.select([col for col in df.columns if df[col].dtype in [pl.Float64, pl.Int64]])

        # エンコードしたカテゴリ変数と標準化した数値変数を結合
        df = pl.concat([num_df, cat_df], how="horizontal")

        return df