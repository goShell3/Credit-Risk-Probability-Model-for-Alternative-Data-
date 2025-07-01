import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


# 🔹 1. Customer-level Aggregates
class CustomerAggregateFeature(BaseEstimator, TransformerMixin):

    def __init__(self, value_col='Value', amount_col='Amount', customer_id_col='CustomerId'):
        self.value_col = value_col
        self.amount_col = amount_col
        self.customer_id_col = customer_id_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        
        :param X:
        :return:
        """
        X = X.copy()
        agg_df = X.groupby(self.customer_id_col).agg(
            total_amount=(self.amount_col, 'sum'),
            average_amount=(self.amount_col, 'mean'),
            transaction_count=(self.amount_col, 'count'),
            amount_std=(self.amount_col, 'std')
        ).fillna(0).reset_index()
        return agg_df


# 🔹 2. Timestamp Extraction
class DateTimeFeatureExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, datetime_col='TransactionStartTime'):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        :param X:
        :return:
        """
        X = X.copy()
        if self.datetime_col in X.columns:
            X[self.datetime_col] = pd.to_datetime(X[self.datetime_col], errors='coerce')
            X['TransactionHour'] = X[self.datetime_col].dt.hour
            X['TransactionDay'] = X[self.datetime_col].dt.day
            X['TransactionMonth'] = X[self.datetime_col].dt.month
            X['TransactionYear'] = X[self.datetime_col].dt.year
            X.drop(columns=[self.datetime_col], inplace=True)
        return X


# 🔹 3. Build Full Pipeline
def build_full_pipeline(
    numeric_features=['total_amount', 'average_amount', 'transaction_count', 'amount_std',
                      'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear'],
    categorical_features=['ChannelId', 'PricingStrategy']
):
    """
    pipeline for numeric features  and categorical features

    :type numeric_features: list
    :type categorical_features: list
    """
    # Pipeline for numeric features
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Pipeline for categorical features
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine numeric and categorical pipelines
    feature_processor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ], remainder='drop')

    # Full pipeline with aggregation + datetime + processing
    full_pipeline = Pipeline([
        ('customer_agg', CustomerAggregateFeature()),
        ('datetime', DateTimeFeatureExtractor()),
        ('processor', feature_processor)
    ])

    return full_pipeline
