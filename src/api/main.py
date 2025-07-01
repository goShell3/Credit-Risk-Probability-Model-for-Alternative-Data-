import pandas as pd
from src.feature_engineering import build_full_pipeline

# Load your raw transaction data
df = pd.read_csv("data/raw_transactions.csv")

# Build the pipeline
pipeline = build_full_pipeline()

# Fit and transform
X_processed = pipeline.fit_transform(df)

print(X_processed.shape)
