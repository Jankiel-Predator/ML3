import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.decomposition import PCA



class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns)



class BinaryTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for col in self.columns:
            X[col] = X[col].apply(lambda x: 1 if x == "True" else 0)
        return X



def create_pipeline(drop_columns, binary_columns):
    pipeline = Pipeline([
        ('drop_columns', DropColumns(columns=drop_columns)),
        ('binary_transform', BinaryTransformer(columns=binary_columns)),
        ('encoder', ColumnTransformer([
            ('dst_ip', OrdinalEncoder(), ["DST_IP"]),
            ('src_ip', OrdinalEncoder(), ["SRC_IP"])
        ], remainder='passthrough')),
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
        ('scaler', MinMaxScaler()),
        ('pca', PCA(n_components=0.99))
    ])
    return pipeline
