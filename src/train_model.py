"""
MODEL TRAINING & FEATURE PIPELINE: UK PROPERTY VALUATION ENGINE.
- Extraction of historical transaction data from MySQL.
- Engineers leak-free spatial and categorical signals, fits a
regularised Gradient Boosted Tree (XGBoost), and serialises model weights
and preprocessing lookups for real-time downstream inference.
"""


import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import time
import json 
import os
from dotenv import load_dotenv

# 1. ENVIRONMENT & DATABASE CONFIGURATION
# Decouple database credentials using environment variables
# This prevents secrets leakage in version control.

load_dotenv()

DATABASE_USER = os.getenv("DB_USER")
DATABASE_PASS = os.getenv("DB_PASS")
DATABASE_HOST = os.getenv("DB_HOST", "localhost")
DATABASE_NAME = os.getenv("DB_NAME", "uk_property_db")

engine = create_engine(f"mysql+pymysql://{DATABASE_USER}:{DATABASE_PASS}@{DATABASE_HOST}/{DATABASE_NAME}")

# Extract pre-cleaned and indexed records across major regional real estate hubs.
# Outlier caps (£50k to £10M) and outward postcode extraction were handled at the view layer.

query = """
SELECT
    price,
    town_city,
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

print("--> Loading records from MySQL...")
df = pd.read_sql(query, engine)

# 2. TARGET TRANSFORMATION (Handling positive skew)
# Property transaction values exhibit a heavy right tail. Applying a log1p
# transformation normalizes the residual distribution, stabilizes gradient steps,
# and prevents multi-million pound properties from dominating squared-error losses.
df['target'] = np.log1p(df['price'])

# 3. TRAIN/TEST PARTITIONING (Strict Data Leakage Prevention)
# I partition before calculating any aggregate statistics.
# Target-encoding features must learn their parameters exclusively from the
# training partition to preserve out-of-sample validation integrity.
train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)

print(f"Training set: {train_df.shape[0]:,} samples")
print(f"Testing set:  {test_df.shape[0]:,} samples")


# 4. FEATURE ENGINEERING (Strictly within X_train)
# A. Spatial Liquidity / Transaction Volume Proxy
freq_map = train_df['postcode_district'].value_counts(normalize=True)
train_df['postcode_freq'] = train_df['postcode_district'].map(freq_map)
test_df['postcode_freq'] = test_df['postcode_district'].map(freq_map).fillna(0)

# B. Smoothed Bayesian Target Encoding (Neighborhood Valuation Baseline)
# High-cardinality postcodes risk overfitting if low-sample districts take raw averages.
# We apply m-estimate smoothing: (n * district_mean + m * global_mean) / (n + m)
# When transaction count (n) is low, the estimate shrinks safely toward the global prior.
global_mean_log = train_df['target'].mean()
smoothing_weight = 10

district_stats = train_df.groupby('postcode_district')['target'].agg(['count', 'mean'])
smoothed_map = (
    (district_stats['count'] * district_stats['mean'] + smoothing_weight * global_mean_log) 
    / (district_stats['count'] + smoothing_weight)
)

train_df['district_price_level'] = train_df['postcode_district'].map(smoothed_map)
test_df['district_price_level'] = test_df['postcode_district'].map(smoothed_map).fillna(global_mean_log)

# C. Macro Regional Price Level
city_price_map = train_df.groupby('town_city')['target'].mean()
train_df['city_price_level'] = train_df['town_city'].map(city_price_map)
test_df['city_price_level'] = test_df['town_city'].map(city_price_map).fillna(global_mean_log)

# 5. SCHEMA ALIGNMENT & CATEGORICAL ONE-HOT ENCODING

feature_cols = [
    "transfer_year",
    "transfer_quarter",
    "transfer_month",
    "postcode_freq",
    "district_price_level",
    "city_price_level",
    "property_type",
    "old_new",
    "duration"
]

# Guarantee deterministic column structure between train and evaluation partitions
combined = pd.concat([train_df[feature_cols], test_df[feature_cols]], axis=0)
combined_encoded = pd.get_dummies(
    combined,
    columns=["property_type", "old_new", "duration"],
    drop_first=True,
    dtype=int
)

X_train = combined_encoded.iloc[:len(train_df)]
X_test = combined_encoded.iloc[len(train_df):]
y_train = train_df['target']
y_test = test_df['target']

# 6. MODEL TRAINING (Gradient Boosted Decision Trees)

# Hyperparameters configured to mitigate variance:
# - colsample_bytree & subsample: stochastic regularization to prevent dominant splits
# - max_depth=7: captures complex spatial-structural interactions without excessive memorization
model = xgb.XGBRegressor(
    n_estimators=350,
    learning_rate=0.07,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

start_time = time.time()
model.fit(X_train, y_train)
elapsed = time.time() - start_time
print(f"Training complete in {elapsed:.2f} seconds")

# 7. MODEL EVALUATION & CURRENCY-SCALE DIAGNOSTICS
y_pred_log = model.predict(X_test)

# Invert log-predictions back into real GBP values for business metric evaluation
y_pred_real = np.expm1(y_pred_log)
y_test_real = np.expm1(y_test)

mae = mean_absolute_error(y_test_real, y_pred_real)
rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
r2 = r2_score(y_test_real, y_pred_real)
mape = np.mean(np.abs((y_test_real - y_pred_real) / y_test_real)) * 100

print("\n" + "=" * 42)
print("=" * 42)
print(f"Mean Absolute Error (MAE):       £{mae:,.2f}")
print(f"Root Mean Squared Error (RMSE):  £{rmse:,.2f}")
print(f"Mean Absolute % Error (MAPE):    {mape:.2f}%")
print(f"R-squared Score (R²):            {r2:.4f}")
print("=" * 42)

# 8. FEATURE IMPORTANCE DIAGNOSTICS (Gain)
# Gain evaluates the relative contribution of each feature to minimising tree loss

print("\n" + "=" * 42)
print("=" * 42)
importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance (Gain)": model.feature_importances_
}).sort_values(by="Importance (Gain)", ascending=False)
importance_df["Importance (Gain)"] = importance_df["Importance (Gain)"].apply(lambda x: f"{x * 100:.2f}%")
print(importance_df.to_string(index=False))

# 9. ARTIFICIAL SERIALISATION FOR PRODUCTION INFERENCE
# We serialize both model weights and the exact precomputed lookups.
# This eliminates training dependencies during production API or CLI serving.

os.makedirs("models", exist_ok=True)

# 1. Save Native XGBoost Model
model_path = "models/xgb_valuation_model.json"
model.save_model(model_path)
print(f" Saved XGBoost model artifact to: {model_path}")

# 2. Package Preprocessing Lookups & Schema
metadata = {
    "global_mean_log": float(global_mean_log),
    "feature_columns": list(X_train.columns),
    "city_price_map": city_price_map.to_dict(),
    "smoothed_district_map": smoothed_map.to_dict(),
    "freq_map": freq_map.to_dict()
}

metadata_path = "models/metadata.json"
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)
print(f" Saved pipeline metadata & lookups to: {metadata_path}")