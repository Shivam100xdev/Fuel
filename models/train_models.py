# ================================================================
#  models/train_models.py
#  Run ONCE to train Random Forest + XGBoost + (LSTM optional)
#  Usage: python models/train_models.py
# ================================================================
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MODELS_DIR = os.path.dirname(__file__)
DATA_PATH  = os.path.join(MODELS_DIR, '..', 'data', 'india_fuel_prices.csv')
FEATURES   = ['Brent_Crude','USD_INR','Inflation','Excise_Duty',
               'OPEC_Cut','VIX','State_VAT','Month','t']

def evaluate(y_true, y_pred, name):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None))) * 100
    print(f"  {name:<20} MAE=Rs{mae:.2f}  RMSE=Rs{rmse:.2f}  R²={r2:.4f}  MAPE={mape:.2f}%")
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}

def train_for_fuel(df, fuel_type):
    print(f"\n{'='*55}")
    print(f"  Training models for: {fuel_type.upper()}")
    print(f"{'='*55}")

    sub    = df.dropna(subset=[fuel_type] + FEATURES).copy()
    split  = int(len(sub) * 0.80)
    tr, te = sub.iloc[:split], sub.iloc[split:]
    X_tr, y_tr = tr[FEATURES].values, tr[fuel_type].values
    X_te, y_te = te[FEATURES].values, te[fuel_type].values

    scaler   = StandardScaler()
    X_tr_s   = scaler.fit_transform(X_tr)
    X_te_s   = scaler.transform(X_te)

    # ── Random Forest ────────────────────────────────────────
    print("\n[1] Random Forest (300 trees) …")
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=14, min_samples_leaf=2,
        max_features='sqrt', n_jobs=-1, random_state=42)
    rf.fit(X_tr_s, y_tr)
    p_rf = rf.predict(X_te_s)
    m_rf = evaluate(y_te, p_rf, "Random Forest")

    # ── Gradient Boosting (XGBoost-style) ───────────────────
    print("\n[2] Gradient Boosting (XGBoost-style) …")
    xgb = GradientBoostingRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.04,
        subsample=0.8, min_samples_leaf=2,
        loss='huber', random_state=42)
    xgb.fit(X_tr_s, y_tr)
    p_xgb = xgb.predict(X_te_s)
    m_xgb = evaluate(y_te, p_xgb, "Grad. Boosting")

    # ── Ensemble ─────────────────────────────────────────────
    p_ens = 0.5 * p_rf + 0.5 * p_xgb
    print()
    m_ens = evaluate(y_te, p_ens, "Ensemble")

    # ── Save ─────────────────────────────────────────────────
    fk = fuel_type.lower()
    joblib.dump(rf,     os.path.join(MODELS_DIR, f'rf_{fk}.pkl'))
    joblib.dump(xgb,    os.path.join(MODELS_DIR, f'xgb_{fk}.pkl'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, f'scaler_{fk}.pkl'))
    print(f"\n  Saved: rf_{fk}.pkl  xgb_{fk}.pkl  scaler_{fk}.pkl")

    # ── Plot ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor('#0f0e0a')
    for ax in axes: ax.set_facecolor('#1a1710')

    fig.suptitle(f'{fuel_type} Price Prediction — Model Comparison', color='#d4af37',
                 fontsize=14, fontweight='bold')

    # Actual vs Predicted
    ax = axes[0]
    ax.plot(te['Date'].values, y_te,   color='#10b981', lw=2, label='Actual')
    ax.plot(te['Date'].values, p_rf,   color='#60a5fa', lw=1.5, ls='--', label='RF')
    ax.plot(te['Date'].values, p_xgb,  color='#f59e0b', lw=1.5, ls='-.', label='GBM')
    ax.plot(te['Date'].values, p_ens,  color='#c084fc', lw=2, label='Ensemble')
    ax.set_title('Actual vs Predicted', color='white'); ax.legend(facecolor='#1a1710', labelcolor='white')
    ax.tick_params(colors='#9c9070'); ax.grid(alpha=0.2)

    # Feature importance
    fi = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
    axes[1].barh(fi.index, fi.values, color='#10b981', alpha=0.85)
    axes[1].set_title('RF Feature Importance', color='white')
    axes[1].tick_params(colors='#9c9070'); axes[1].grid(alpha=0.2, axis='x')

    # Residuals
    residuals = y_te - p_ens
    axes[2].hist(residuals, bins=30, color='#60a5fa', alpha=0.8, edgecolor='#0f0e0a')
    axes[2].axvline(0, color='#f87171', lw=2, ls='--')
    axes[2].set_title('Ensemble Residuals', color='white')
    axes[2].tick_params(colors='#9c9070'); axes[2].grid(alpha=0.2)

    plt.tight_layout()
    out = os.path.join(MODELS_DIR, f'{fk}_results.png')
    plt.savefig(out, dpi=120, bbox_inches='tight', facecolor='#0f0e0a')
    plt.close()
    print(f"  Plot saved: {out}")

    return m_rf, m_xgb, m_ens


def train_lstm(df, fuel_type='Petrol'):
    """Optional LSTM — runs if TensorFlow is installed."""
    print(f"\n[3] LSTM Neural Network for {fuel_type} …")
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        print(f"  TensorFlow {tf.__version__} available ✅")
    except ImportError:
        print("  TensorFlow not installed. Skipping LSTM.")
        print("  Install with: pip install tensorflow")
        return None

    COLS    = ['Brent_Crude','USD_INR','Inflation','Excise_Duty',
               'VIX','State_VAT','Month','t', fuel_type]
    sub     = df[df['City']=='Delhi'].dropna(subset=COLS).sort_values('Date')
    data    = sub[COLS].values
    SEQ_LEN = 12  # 12 months lookback

    from sklearn.preprocessing import MinMaxScaler
    sc      = MinMaxScaler()
    scaled  = sc.fit_transform(data)

    X, y = [], []
    for i in range(SEQ_LEN, len(scaled)):
        X.append(scaled[i-SEQ_LEN:i])
        y.append(scaled[i, -1])
    X, y = np.array(X), np.array(y)

    split    = int(len(X) * 0.80)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, len(COLS))),
        Dropout(0.2), BatchNormalization(),
        LSTM(32, return_sequences=False),
        Dropout(0.15),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='huber', metrics=['mae'])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4),
    ]
    history = model.fit(X_tr, y_tr, epochs=50, batch_size=8,
                        validation_split=0.15, callbacks=callbacks, verbose=1)

    y_pred_s = model.predict(X_te)
    dummy    = np.zeros((len(y_pred_s), len(COLS)))
    dummy[:, -1] = y_pred_s.flatten()
    y_pred   = sc.inverse_transform(dummy)[:, -1]

    dummy2 = np.zeros((len(y_te), len(COLS)))
    dummy2[:, -1] = y_te
    y_true = sc.inverse_transform(dummy2)[:, -1]

    m = evaluate(y_true, y_pred, "LSTM")
    model.save(os.path.join(os.path.dirname(__file__), f'lstm_{fuel_type.lower()}.keras'))
    joblib.dump(sc, os.path.join(os.path.dirname(__file__), f'lstm_scaler_{fuel_type.lower()}.pkl'))
    print(f"  Saved: lstm_{fuel_type.lower()}.keras")
    return m


if __name__ == '__main__':
    print("\n" + "="*55)
    print("  FUEL PRICE PREDICTION — Training Pipeline")
    print("="*55)

    df = pd.read_csv(DATA_PATH)
    df['Date']  = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Year']  = df['Date'].dt.year
    df['t']     = (df['Year'] - 2022) * 12 + df['Month']

    results = {}
    for fuel in ['Petrol', 'Diesel']:
        rf_m, xgb_m, ens_m = train_for_fuel(df, fuel)
        results[fuel] = {'RF': rf_m, 'GBM': xgb_m, 'Ensemble': ens_m}

    # Optional LSTM
    lstm_m = train_lstm(df, 'Petrol')
    if lstm_m:
        results['Petrol']['LSTM'] = lstm_m

    # Summary table
    print("\n" + "="*55)
    print("  FINAL MODEL COMPARISON")
    print("="*55)
    print(f"{'Fuel':<10} {'Model':<20} {'R²':>8} {'MAE Rs':>8} {'MAPE%':>8}")
    print("-"*58)
    for fuel, models in results.items():
        for mname, m in models.items():
            print(f"{fuel:<10} {mname:<20} {m['R2']:>8.4f} {m['MAE']:>8.2f} {m['MAPE']:>7.2f}%")
    print("="*55)
    print("\nDone! Now run: streamlit run app/streamlit_app.py\n")
