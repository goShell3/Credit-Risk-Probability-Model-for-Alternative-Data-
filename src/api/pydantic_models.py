from pydantic import BaseModel

class PredictionInput(BaseModel):
    total_amount: float
    avg_amount: float
    txn_count: int
    TransactionHour: int
    Channel_web: int
    # add all other expected features...

class PredictionResponse(BaseModel):
    risk_probability: float
