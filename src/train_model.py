import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import time

DATABASE_USER = "akash_property_ml"
DATABASE_PASS = "Oligopolistic2"
DATABASE_HOST = "localhost"
DATABASE_NAME = "uk_property_db"

# connecting to MySQL
engine = create_engine(f"mysql+pymysql://{DATABASE_USER}:{DATABASE_PASS}@{DATABASE_HOST}/{DATABASE_NAME}")

query = """
SELECT
    price,
    transfer_year,
    transfer_quarter,
    transfer_month,
    postcode_district,
    property_type,
    old_new,
    duration
FROM vw_property_features
WHERE town_city IN ('LONDON', 'MANCHESTER', 'BIRMINGHAM', 'BRISTOL', 'READING');
"""

df = pd.read_sql(query, engine)


df['target'] = np.log1p(df['price'])

freq_map = df['postcode_district'].value_counts(normalize=True)
df['postcode_freq'] = df['postcode_district'].map(freq_map)

feature_cols = [
    "transfer_year",
    "transfer_quarter",
    "transfer_month",
    "postcode_freq",
    "property_type",
    "old_new",
    "duration" 
]
X = pd.get_dummies(
    df[feature_cols],
    columns=["property_type", "old_new", "duration"],
    drop_first=True,
    dtype=int
)
y = df["target"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"Training set: {X_train.shape[0]:,} samples")
print(f"Testing set:  {X_test.shape[0]:,} samples")

model = xgb.XGBRegressor(
    n_estimators=300,      
    learning_rate=0.08,   
    max_depth=6,         
    subsample=0.8,        
    colsample_bytree=0.8,  
    random_state=42,
    n_jobs=-1              
)


start_time = time.time()
model.fit(X_train, y_train)
elapsed = time.time() - start_time
print(f"Training complete in {elapsed:.2f} seconds")
