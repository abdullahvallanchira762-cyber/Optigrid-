# Optigrid — ML Occupancy & Energy Optimization

## Quick Start

### 1. Install dependencies
```bash
pip install flask scikit-learn pandas numpy
```

### 2. Run the server
```bash
python app.py
```

### 3. Open the dashboard
Visit: http://localhost:5000

### 4. Train the model
- Upload `occupancy_data.csv` (the sample file included) OR your own CSV
- The ML model trains in seconds and predictions appear live

---

## Your CSV format

Required columns:

| Column | Type | Description |
|---|---|---|
| `hour` | int 0–23 | Hour of day |
| `day_of_week` | int 0–6 | 0=Monday, 6=Sunday |
| `is_weekend` | int 0/1 | Weekend flag |
| `zone` | string | Zone name (e.g. Floor2) |
| `zone_capacity` | int | Max people in zone |
| `occupancy_pct` | float 0–1 | Actual occupancy fraction |

Optional (used for energy analysis):
- `hvac_kwh`, `lighting_kwh`, `equipment_kwh`, `total_kwh`

---

## How it works

1. **Training**: Uploads your CSV → trains a `GradientBoostingRegressor` per zone
2. **Prediction**: For each zone × hour combination, the model predicts occupancy %
3. **Optimization**: Based on predicted occupancy, HVAC mode is selected:
   - < 8%  → **Standby** (15% load)
   - < 40% → **Eco** (60% load)
   - < 75% → **Normal** (85% load)
   - ≥ 75% → **Full** (100% load)
4. **Energy calc**: Optimized kWh = base load × mode load factor
5. **Savings**: Compared against unoptimized baseline (always full load)

## API Endpoints

- `POST /api/train` — upload CSV, returns model metrics
- `POST /api/predict` — predict single zone/hour
- `POST /api/dashboard` — predict all zones × 24 hours
- `GET  /api/status` — check if model is trained

## Replacing with real sensor data

In `app.py`, the `occAt()` simulation is only used for the sample CSV generator.
Once you upload real historical data, all predictions come from your trained model.

To connect live IoT sensors: poll `/api/predict` with real-time inputs every few minutes.
