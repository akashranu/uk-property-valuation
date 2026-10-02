import json
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from fastapi.responses import RedirectResponse



METADATA_PATH = "models/metadata.json"
MODEL_PATH = "models/xgb_valuation_model.json"

# Load artifacts into memory once at application startup
with open(METADATA_PATH, "r") as f:
    meta = json.load(f)

model = xgb.XGBRegressor()
model.load_model(MODEL_PATH)


app = FastAPI(
    title="UK Property Valuation API",
    description="Automated Valuation Model (AVM) for UK Residential Property using Gradient Boosted Trees",
    version="1.0.0"
)
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

# Explicit schema validation using Pydantic
class PropertyInput(BaseModel):
    city: str = Field(..., example="LONDON", description="City name")
    district: str = Field(..., example="SW11", description="Postcode outward district")
    property_type: Literal["D", "F", "S", "T", "O"] = Field(
        ..., example="F", description="D=Detached, F=Flat, S=Semi-Detached, T=Terraced, O=Other"
    )
    duration: Literal["F", "L"] = Field(..., example="L", description="F=Freehold, L=Leasehold")
    new_build: Literal["Y", "N"] = Field(..., example="N", description="Y=New Build, N=Established")
    year: int = Field(default=2024, ge=1995, le=2026, description="Transaction year")
    month: int = Field(default=1, ge=1, le=12, description="Transaction month (1-12)")

class ValuationResponse(BaseModel):
    valuation_gbp: float
    confidence_range_lower: float
    confidence_range_upper: float
    mape_uncertainty_pct: float
    meta_inputs: dict

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_version": "xgb_v1"}

@app.post("/predict", response_model=ValuationResponse)
def predict_valuation(prop: PropertyInput):
    try:
        city = prop.city.strip().upper()
        district = prop.district.strip().upper()
        p_type = prop.property_type.strip().upper()
        tenure = prop.duration.strip().upper()
        new_build = prop.new_build.strip().upper()

        quarter = (prop.month - 1) // 3 + 1
        global_mean = meta["global_mean_log"]

        city_price_level = meta["city_price_map"].get(city, global_mean)
        district_price_level = meta["smoothed_district_map"].get(district, global_mean)
        postcode_freq = meta["freq_map"].get(district, 0.0)

        record = {
            "transfer_year": prop.year,
            "transfer_quarter": quarter,
            "transfer_month": prop.month,
            "postcode_freq": postcode_freq,
            "district_price_level": district_price_level,
            "city_price_level": city_price_level,
            "property_type_F": 1 if p_type == "F" else 0,
            "property_type_O": 1 if p_type == "O" else 0,
            "property_type_S": 1 if p_type == "S" else 0,
            "property_type_T": 1 if p_type == "T" else 0,
            "old_new_Y": 1 if new_build == "Y" else 0,
            "duration_L": 1 if tenure == "L" else 0
        }

        df_input = pd.DataFrame([record])[meta["feature_columns"]]
        pred_log = model.predict(df_input)[0]
        valuation = float(np.expm1(pred_log))

        # Model validation MAPE is ~33.5%
        mape_margin = 0.3348
        lower_bound = round(valuation * (1 - mape_margin), 2)
        upper_bound = round(valuation * (1 + mape_margin), 2)

        return ValuationResponse(
            valuation_gbp=round(valuation, 2),
            confidence_range_lower=lower_bound,
            confidence_range_upper=upper_bound,
            mape_uncertainty_pct=round(mape_margin * 100, 2),
            meta_inputs=prop.model_dump()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))