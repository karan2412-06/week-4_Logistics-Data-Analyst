# Week 4 — Predictive Modeling and Optimization in Logistics Systems

## Project overview
This project demonstrates how a logistics team could estimate delivery time and use those estimates to flag shipments that may need attention. It compares Linear Regression with Random Forest Regression and reports MAE, RMSE, R², and five-fold cross-validation MAE.

**Important:** `data/logistics_shipments.csv` is a simulated teaching dataset, not real company or customer data. The results are illustrative and must not be presented as measured performance from a live logistics operation.

## Files
- `data/logistics_shipments.csv` — 900 simulated shipment records
- `src/logistics_model.py` — complete preprocessing, training, evaluation, and risk-flagging workflow
- `outputs/metrics.json` — results from the supplied dataset
- `report/Week_4_Logistics_Predictive_Modeling_Report.docx` — full written report

## Run locally
1. Install Python 3.10 or newer.
2. In this folder, run `python -m pip install -r requirements.txt`.
3. Run `python src/logistics_model.py`.
4. Review the printed metrics and `outputs/metrics.json`.

## Method
An 80/20 train-test split with a fixed random seed is used for reproducibility. Numeric fields are imputed and scaled; categorical fields are imputed and one-hot encoded. All preprocessing is inside a scikit-learn pipeline to avoid fitting preprocessing on the test set. The Random Forest is also evaluated with five-fold cross-validation.

## Practical optimization recommendations
1. Flag the slowest predicted quarter of shipments for an operational review.
2. Check whether high predicted times cluster around high traffic, poor weather, long distance, or hub backlog.
3. Test dispatch staggering, balancing workloads across hubs, and assigning suitable vehicles on a small pilot.
4. Compare on-time delivery, cost per shipment, and customer complaints before and after a change.

This notebook-free script is intended as a transparent baseline. Real deployment would require historical, time-stamped operational data, a time-based validation split, leakage checks, and monitoring for changing traffic and demand patterns.
