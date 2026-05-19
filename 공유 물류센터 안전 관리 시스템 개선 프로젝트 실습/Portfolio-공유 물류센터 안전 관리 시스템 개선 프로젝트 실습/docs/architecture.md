# Architecture Overview

## Goal
Build a practical anomaly detection and state labeling pipeline for shared logistics centers / shared factories using `data_R2.csv` as the starting point.

## Core idea
1. Normalize raw CSV schema into domain-friendly headers.
2. Create initial state labels using rule-based heuristics.
3. Train baseline models on engineered features.
4. Add modality-specific branches later (CCTV, environment, event logs).
5. Fuse branch scores into a unified safety/operation monitoring layer.

## MVP Components
- Data ingestion and header mapping
- EDA and feature exploration
- Rule-based labeling
- Baseline anomaly detection
- Error analysis and visualization

## Future expansion
- Sensor branch: time-series anomaly scoring
- CCTV branch: human presence and proximity detection
- Environment branch: temperature / humidity / gas monitoring
- Fusion: late fusion or meta-classifier

