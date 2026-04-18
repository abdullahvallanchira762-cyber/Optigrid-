"""
EnergyIQ — ML Occupancy Prediction & Energy Optimization Backend
Run: python app.py
Then open: http://localhost:5000
"""

import os, json, io, csv, math
from flask import Flask, request, jsonify, send_from_directory

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings; warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder='static')

# ── Global model store ────────────────────────────────────────────────────────
models = {}          # zone -> sklearn model
le_zone = LabelEncoder()
feature_cols = ['hour', 'day_of_week', 'is_weekend', 'zone_encoded', 'zone_capacity']
training_stats = {}  # stores MAE, R2 per zone

HVAC_MODES = [
    (0.08, 'Standby',  0.15),
    (0.40, 'Eco',      0.60),
    (0.75, 'Normal',   0.85),
    (1.01, 'Full',     1.00),
]

def hvac_mode(occ_pct):
    for threshold, label, load_factor in HVAC_MODES:
        if occ_pct < threshold:
            return label, load_factor
    return 'Full', 1.00

def energy_optimized(occ_pct, zone_capacity, hour):
    _, load = hvac_mode(occ_pct)
    hvac      = round(occ_pct * 4.0 * load + 0.5, 3)
    lighting  = round((occ_pct * 1.5 + 0.2) * load if 7 <= hour <= 19 else 0.1, 3)
    equipment = round(occ_pct * 1.2 + 0.3, 3)
    baseline  = round(4.0 + 0.5 + 1.5 + 0.2 + 1.2 + 0.3, 3)   # 100% occ, no optimization
    actual    = hvac + lighting + equipment
    saving_pct = round((1 - actual / max(baseline, 0.01)) * 100, 1)
    return {
        'hvac_kwh':      hvac,
        'lighting_kwh':  lighting,
        'equipment_kwh': equipment,
        'total_kwh':     round(actual, 3),
        'saving_pct':    max(0, saving_pct),
        'co2_avoided_kg': round((baseline - actual) * 0.233, 3),
    }


# ── Training ──────────────────────────────────────────────────────────────────
def train_from_df(df):
    global models, le_zone, training_stats

    required = {'hour','day_of_week','is_weekend','zone','zone_capacity','occupancy_pct'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df = df.copy()
    df['zone_encoded'] = le_zone.fit_transform(df['zone'].astype(str))

    models = {}
    training_stats = {}

    # One model per zone for best accuracy
    for zone_name in df['zone'].unique():
        zdf = df[df['zone'] == zone_name].copy()
        X = zdf[feature_cols]
        y = zdf['occupancy_pct']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        m = GradientBoostingRegressor(n_estimators=120, learning_rate=0.08,
                                      max_depth=4, random_state=42)
        m.fit(X_train, y_train)
        preds = m.predict(X_test)

        mae = round(mean_absolute_error(y_test, preds), 4)
        r2  = round(r2_score(y_test, preds), 4)

        models[zone_name] = m
        training_stats[zone_name] = {'mae': mae, 'r2': r2, 'samples': len(zdf)}

    return training_stats


def predict_hour(zone, hour, day_of_week, is_weekend, zone_capacity):
    if zone not in models:
        raise ValueError(f"No model for zone '{zone}'. Train first.")
    zone_enc = le_zone.transform([zone])[0]
    X = pd.DataFrame([{
        'hour': hour, 'day_of_week': day_of_week,
        'is_weekend': int(is_weekend), 'zone_encoded': zone_enc,
        'zone_capacity': zone_capacity,
    }])
    occ = float(np.clip(models[zone].predict(X)[0], 0, 1))
    return round(occ, 4)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/train', methods=['POST'])
def api_train():
    """Upload CSV → train models → return metrics"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    f = request.files['file']
    try:
        df = pd.read_csv(f)
        stats = train_from_df(df)
        zones = list(stats.keys())
        avg_r2 = round(sum(v['r2'] for v in stats.values()) / len(stats), 4)
        avg_mae = round(sum(v['mae'] for v in stats.values()) / len(stats), 4)
        return jsonify({
            'status': 'trained',
            'zones': zones,
            'zone_stats': stats,
            'avg_r2': avg_r2,
            'avg_mae': avg_mae,
            'total_samples': sum(v['samples'] for v in stats.values()),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Predict occupancy + energy for a full day or single slot"""
    if not models:
        return jsonify({'error': 'No model trained. Upload a CSV first.'}), 400

    body = request.get_json(force=True)
    zone         = body.get('zone')
    hour         = int(body.get('hour', 9))
    day_of_week  = int(body.get('day_of_week', 1))
    is_weekend   = bool(body.get('is_weekend', False))
    zone_capacity= int(body.get('zone_capacity', 80))

    try:
        occ = predict_hour(zone, hour, day_of_week, is_weekend, zone_capacity)
        energy = energy_optimized(occ, zone_capacity, hour)
        mode, _ = hvac_mode(occ)
        return jsonify({
            'zone': zone, 'hour': hour,
            'occupancy_pct': occ,
            'occupancy_count': int(occ * zone_capacity),
            'hvac_mode': mode,
            **energy,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/dashboard', methods=['POST'])
def api_dashboard():
    """Return full dashboard data: all zones × all hours for a given day"""
    if not models:
        return jsonify({'error': 'No model trained. Upload a CSV first.'}), 400

    body = request.get_json(force=True)
    day_of_week  = int(body.get('day_of_week', 1))
    is_weekend   = bool(body.get('is_weekend', False))
    zones_input  = body.get('zones', [])   # [{name, capacity}]

    results = {}
    for zinfo in zones_input:
        zname = zinfo['name']
        zcap  = int(zinfo.get('capacity', 80))
        hourly = []
        for h in range(24):
            if zname not in models:
                occ = 0.0
            else:
                occ = predict_hour(zname, h, day_of_week, is_weekend, zcap)
            energy = energy_optimized(occ, zcap, h)
            mode, _ = hvac_mode(occ)
            hourly.append({
                'hour': h,
                'occupancy_pct': occ,
                'occupancy_count': int(occ * zcap),
                'hvac_mode': mode,
                **energy,
            })
        results[zname] = hourly

    return jsonify({'day_of_week': day_of_week, 'is_weekend': is_weekend, 'zones': results})


@app.route('/api/status')
def api_status():
    return jsonify({
        'trained': len(models) > 0,
        'zones': list(models.keys()),
        'stats': training_stats,
    })


if __name__ == '__main__':
    print("\n╔══════════════════════════════════════╗")
    print("║   Optigrid ML Backend  — port 5000   ║")
    print("╚══════════════════════════════════════╝")
    print("  1. Open http://localhost:5000")
    print("  2. Upload occupancy_data.csv to train")
    print("  3. Dashboard updates with real predictions\n")
    app.run(debug=True, port=5000)
