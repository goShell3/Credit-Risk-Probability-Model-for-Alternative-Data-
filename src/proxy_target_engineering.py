import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


class ProxyTargetEngineer:
    def __init__(self, snapshot_date='2023-12-31'):
        # Convert to timezone-naive datetime
        self.snapshot_date = pd.to_datetime(snapshot_date).tz_localize(None)

    def compute_rfm(self, df: pd.DataFrame) -> pd.DataFrame:
        # Convert to datetime - ensure consistent timezone handling
        df['TransactionStartTime'] = pd.to_datetime(
            df['TransactionStartTime'], 
            errors='coerce',
            utc=False  # Match snapshot_date's timezone
        )
        
        # Filter out null dates if any
        df = df.dropna(subset=['TransactionStartTime'])
        
        rfm = df.groupby('CustomerId').agg({
            'TransactionStartTime': lambda x: (self.snapshot_date - x.max()).days,
            'TransactionId': 'count',
            'Amount': 'sum'
        }).rename(columns={
            'TransactionStartTime': 'Recency',
            'TransactionId': 'Frequency',
            'Amount': 'Monetary'
        }).reset_index()
        
        return rfm

    def assign_clusters(self, rfm: pd.DataFrame) -> pd.DataFrame:
        rfm_scaled = self.scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
        rfm['Cluster'] = self.kmeans.fit_predict(rfm_scaled)
        return rfm

    def determine_high_risk_cluster(self, rfm: pd.DataFrame) -> int:
        cluster_profile = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
        # High recency, low frequency, low monetary is risky
        self.high_risk_cluster = cluster_profile.sort_values(['Frequency', 'Monetary', 'Recency'], ascending=[True, True, False]).index[0]
        return self.high_risk_cluster

    def generate_target(self, df: pd.DataFrame) -> pd.DataFrame:
        rfm = self.compute_rfm(df)
        rfm = self.assign_clusters(rfm)
        high_risk = self.determine_high_risk_cluster(rfm)
        rfm['is_high_risk'] = (rfm['Cluster'] == high_risk).astype(int)
        return rfm[['CustomerId', 'is_high_risk']]
