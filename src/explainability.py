import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

METADATA_PATH = "models/metadata.json"
MODEL_PATH = "models/xgb_valuation_model.json"
OUTPUT_DIR = "reports"

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(METADATA_PATH, "r") as f:
    meta = json.load(f)

model = xgb.XGBRegressor()
model.load_model(MODEL_PATH)

# Ingest a representative sample for explanation
DATABASE_USER = os.getenv("DB_USER")
DATABASE_PASS = os.getenv("DB_PASS")
DATABASE_HOST = os.getenv("DB_HOST", "localhost")
DATABASE_PORT = os.getenv("DB_PORT", "3306")
DATABASE_NAME = os.getenv("DB_NAME", "uk_property_db")

engine = create_engine(
    f"mysql+pymysql://{DATABASE_USER}:{DATABASE_PASS}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"
)

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
WHERE town_city IN ('LONDON', 'MANCHESTER', 'BIRMINGHAM', 'BRISTOL', 'READING')
LIMIT 2000;
"""

sample_df = pd.read_sql(query, engine)

# Apply Metadata Mappings
global_mean = meta["global_mean_log"]
sample_df["postcode_freq"] = sample_df["postcode_district"].map(meta["freq_map"]).fillna(0.0)
sample_df["district_price_level"] = sample_df["postcode_district"].map(meta["smoothed_district_map"]).fillna(global_mean)
sample_df["city_price_level"] = sample_df["town_city"].map(meta["city_price_map"]).fillna(global_mean)

# One-hot encode using training schema
feature_cols = [
    "transfer_year", "transfer_quarter", "transfer_month",
    "postcode_freq", "district_price_level", "city_price_level",
    "property_type", "old_new", "duration"
]

encoded = pd.get_dummies(
    sample_df[feature_cols],
    columns=["property_type", "old_new", "duration"],
    drop_first=True,
    dtype=int
)

# Align columns precisely with training feature catalog
X_diag = encoded.reindex(columns=meta["feature_columns"], fill_value=0)

# Compute TreeSHAP Values

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_diag)

# Export Global Beeswarm Summary Plot

plt.figure(figsize=(10, 6))
shap.plots.beeswarm(shap_values, max_display=10, show=False)
plt.title("UK Property Valuation - Global Feature Impact (SHAP)", fontsize=13, pad=15)
plt.tight_layout()
beeswarm_path = os.path.join(OUTPUT_DIR, "shap_summary_beeswarm.png")
plt.savefig(beeswarm_path, dpi=300)
plt.close()
print(f" Saved global explanation to: {beeswarm_path}")

# Export Single-Instance Waterfall Plot

plt.figure(figsize=(9, 6))
shap.plots.waterfall(shap_values[0], max_display=10, show=False)
plt.title("Individual Property Valuation - SHAP Attribution", fontsize=13, pad=15)
plt.tight_layout()
waterfall_path = os.path.join(OUTPUT_DIR, "shap_single_property_waterfall.png")
plt.savefig(waterfall_path, dpi=300)
plt.close()
print(f" Saved local explanation to: {waterfall_path}")

