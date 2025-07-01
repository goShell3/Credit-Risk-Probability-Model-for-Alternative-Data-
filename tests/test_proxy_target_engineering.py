import pandas as pd
import pytest
from src.proxy_target_engineering import ProxyTargetEngineer

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'CustomerId': [1, 1, 2, 2, 3, 4, 4, 4],
        'TransactionId': ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8'],
        'Amount': [100, 150, 30, 20, 500, 10, 5, 5],
        'TransactionStartTime': [
            '2023-12-01', '2023-12-15',
            '2023-10-01', '2023-11-01',
            '2023-12-31',
            '2023-01-01', '2023-01-10', '2023-02-01'
        ]
    })


def test_compute_rfm(sample_data):
    proxy = ProxyTargetEngineer(snapshot_date='2023-12-31')
    rfm = proxy.compute_rfm(sample_data)

    assert 'Recency' in rfm.columns
    assert 'Frequency' in rfm.columns
    assert 'Monetary' in rfm.columns
    assert rfm.shape[0] == 4  # 4 unique customers

def test_assign_clusters(sample_data):
    proxy = ProxyTargetEngineer(snapshot_date='2023-12-31')
    rfm = proxy.compute_rfm(sample_data)
    rfm = proxy.assign_clusters(rfm)

    assert 'Cluster' in rfm.columns
    assert rfm['Cluster'].nunique() == proxy.n_clusters

def test_determine_high_risk_cluster(sample_data):
    proxy = ProxyTargetEngineer(snapshot_date='2023-12-31')
    rfm = proxy.compute_rfm(sample_data)
    rfm = proxy.assign_clusters(rfm)
    high_risk_cluster = proxy.determine_high_risk_cluster(rfm)

    assert high_risk_cluster in [0, 1, 2]
    assert isinstance(high_risk_cluster, int)

def test_generate_target(sample_data):
    proxy = ProxyTargetEngineer(snapshot_date='2023-12-31')
    target = proxy.generate_target(sample_data)

    assert 'CustomerId' in target.columns
    assert 'is_high_risk' in target.columns
    assert target['is_high_risk'].isin([0, 1]).all()
    assert target.shape[0] == 4  # 4 customers
