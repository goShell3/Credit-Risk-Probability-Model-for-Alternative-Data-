import pandas as pd
import numpy as np
import pytest
from src.feature_engineering import CustomerAggregateFeature, DateTimeFeatureExtractor

# Sample DataFrame
@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'CustomerId': [1, 1, 2, 2, 2, 3],
        'Amount': [100, 200, 300, 150, 50, 500],
        'Value': [100, 200, 300, 150, 50, 500]
    })

def test_output_shape(sample_data):
    transformer = CustomerAggregateFeature()
    result = transformer.fit_transform(sample_data)
    
    # Expect 3 customers, 5 columns (CustomerId + 4 features)
    assert result.shape == (3, 5)

def test_total_amount(sample_data):
    transformer = CustomerAggregateFeature()
    result = transformer.fit_transform(sample_data)
    total_by_customer = result.set_index('CustomerId')['total_amount'].to_dict()

    assert total_by_customer[1] == 300  # 100 + 200
    assert total_by_customer[2] == 500  # 300 + 150 + 50
    assert total_by_customer[3] == 500  # 500

def test_average_amount(sample_data):
    transformer = CustomerAggregateFeature()
    result = transformer.fit_transform(sample_data)
    avg_by_customer = result.set_index('CustomerId')['average_amount'].to_dict()

    assert avg_by_customer[1] == 150.0  # (100 + 200)/2
    assert avg_by_customer[2] == pytest.approx(166.67, 0.1)  # (300 + 150 + 50)/3
    assert avg_by_customer[3] == 500.0  # 1 transaction

def test_transaction_count(sample_data):
    transformer = CustomerAggregateFeature()
    result = transformer.fit_transform(sample_data)
    count_by_customer = result.set_index('CustomerId')['transaction_count'].to_dict()

    assert count_by_customer[1] == 2
    assert count_by_customer[2] == 3
    assert count_by_customer[3] == 1

def test_amount_std(sample_data):
    transformer = CustomerAggregateFeature()
    result = transformer.fit_transform(sample_data)
    std_by_customer = result.set_index('CustomerId')['amount_std'].to_dict()

    assert std_by_customer[1] == pytest.approx(70.71, 0.1)  # std of [100, 200]
    assert std_by_customer[2] == pytest.approx(127.47, 0.1) # std of [300, 150, 50]
    assert std_by_customer[3] == 0.0  # std of 1 item = NaN -> filled with 0
