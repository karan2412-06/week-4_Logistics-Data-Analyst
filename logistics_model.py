"""
Week 4: Predictive Modeling and Optimization in Logistics Systems
Reproducible demonstration using a clearly labelled synthetic dataset.
Run: python src/logistics_model.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "logistics_shipments.csv"
OUT_PATH = ROOT / "outputs" / "metrics.json"

def main():
    data = pd.read_csv(DATA_PATH)
    target = "delivery_time_hours"
    X = data.drop(columns=[target])
    y = data[target]
    numeric = ["distance_km", "shipment_weight_kg", "dispatch_hour", "hub_backlog_orders"]
    categorical = ["traffic_level", "weather", "vehicle_type", "service_level"]
    prep = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=250, min_samples_leaf=2, random_state=42, n_jobs=-1
        )
    }
    metrics = {
        "dataset_rows": int(len(data)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "results": {}
    }
    fitted = {}
    for name, estimator in models.items():
        pipeline = Pipeline([("preprocess", prep), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        prediction = pipeline.predict(X_test)
        metrics["results"][name] = {
            "MAE": float(mean_absolute_error(y_test, prediction)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, prediction))),
            "R2": float(r2_score(y_test, prediction))
        }
        fitted[name] = pipeline
    cv = cross_val_score(
        fitted["Random Forest"], X, y, cv=5,
        scoring="neg_mean_absolute_error", n_jobs=1
    )
    metrics["results"]["Random Forest"]["CV_MAE_mean"] = float(-cv.mean())
    metrics["results"]["Random Forest"]["CV_MAE_std"] = float(cv.std())
    metrics["selected_model"] = min(
        models, key=lambda name: metrics["results"][name]["MAE"]
    )
    chosen = fitted[metrics["selected_model"]]
    all_predictions = chosen.predict(X)
    metrics["high_risk_threshold_hours"] = float(np.quantile(all_predictions, 0.75))
    metrics["high_risk_count"] = int(
        (all_predictions >= metrics["high_risk_threshold_hours"]).sum()
    )
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
