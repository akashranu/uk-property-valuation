"""
CLI Real-Time Valuation Service: UK Property Automated Valuation Model (AVM)
Provides low-latency single-instance inference by loading serialized model weights
and precomputed target-encoding lookups, completely decoupling inference from 
upstream database connections.
"""

import argparse
import json
import numpy as np
import pandas as pd
import xgboost as xgb

METADATA_PATH = "models/metadata.json"
MODEL_PATH = "models/xgb_valuation_model.json"

# Loads the serialised model binary and preprocessing lookup metadata.
# Returns: Tuple containing the instantiated XGBRegressor and metadata dictionary.
def load_artifacts():
    with open(METADATA_PATH, "r") as f:
        meta = json.load(f)
    
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    return model, meta

def estimate_valuation(town_city, postcode_district, property_type, duration, old_new, year, month):
    
    # Executes end-to-end valuation inference for a single property specification.
    # Applies deterministic schema projection, imputes unseen geographic entities 
    # using learned global priors, and inverts log-scale predictions into real GBP values.
    # Returns: Tuple of (point_estimate, lower_bound, upper_bound) in GBP.

    model, meta = load_artifacts()


    # Standardise inputs to match training casing
    town_city = town_city.strip().upper()
    postcode_district = postcode_district.strip().upper()
    property_type = property_type.strip().upper()
    duration = duration.strip().upper()
    old_new = old_new.strip().upper()

    quarter = (month - 1) // 3 + 1

    # Zero-shot categorical fallback: if a district or city was never seen during
    # training, default gracefully to the population global mean log-price.
    global_mean = meta["global_mean_log"]
    city_price_level = meta["city_price_map"].get(town_city, global_mean)
    district_price_level = meta["smoothed_district_map"].get(postcode_district, global_mean)
    postcode_freq = meta["freq_map"].get(postcode_district, 0.0)

    record = {
        "transfer_year": year,
        "transfer_quarter": quarter,
        "transfer_month": month,
        "postcode_freq": postcode_freq,
        "district_price_level": district_price_level,
        "city_price_level": city_price_level,
        "property_type_F": 1 if property_type == "F" else 0,
        "property_type_O": 1 if property_type == "O" else 0,
        "property_type_S": 1 if property_type == "S" else 0,
        "property_type_T": 1 if property_type == "T" else 0,
        "old_new_Y": 1 if old_new == "Y" else 0,
        "duration_L": 1 if duration == "L" else 0
    }

    # Strict feature alignment: guarantee identical column ordering to training matrix
    df_sample = pd.DataFrame([record])[meta["feature_columns"]]

    # Invert logarithmic target: exp(pred) - 1
    pred_log = model.predict(df_sample)[0]
    estimated_price = np.expm1(pred_log)

   # Empirical uncertainty interval derived from out-of-sample validation MAPE (~33.48%)
    lower_bound = estimated_price * (1 - 0.3348)
    upper_bound = estimated_price * (1 + 0.3348)

    return estimated_price, lower_bound, upper_bound

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UK Property Automated Valuation Model (AVM)")
    parser.add_argument("--city", type=str, default="LONDON", help="City name (e.g. LONDON, MANCHESTER)")
    parser.add_argument("--district", type=str, required=True, help="Postcode outward district (e.g. SW11, M1, RG1)")
    parser.add_argument("--type", type=str, default="F", choices=["D", "F", "S", "T", "O"], help="Property type: D=Detached, F=Flat, S=Semi, T=Terraced, O=Other")
    parser.add_argument("--duration", type=str, default="L", choices=["F", "L"], help="Tenure: F=Freehold, L=Leasehold")
    parser.add_argument("--new_build", type=str, default="N", choices=["Y", "N"], help="New Build: Y/N")
    parser.add_argument("--year", type=int, default=2024, help="Transaction year")
    parser.add_argument("--month", type=int, default=1, help="Transaction month (1-12)")

    args = parser.parse_args()

    price, low, high = estimate_valuation(
        town_city=args.city,
        postcode_district=args.district,
        property_type=args.type,
        duration=args.duration,
        old_new=args.new_build,
        year=args.year,
        month=args.month
    )

    print("\n" + "=" * 50)
    print("=" * 50)
    print(f"Location:      {args.district}, {args.city}")
    print(f"Specification: Type {args.type} | Tenure {args.duration} | New Build: {args.new_build}")
    print("-" * 50)
    print(f"Valuation:     £{price:,.0f}")
    print(f"Expected Range (£): £{low:,.0f} - £{high:,.0f}")
    print("=" * 50 + "\n")