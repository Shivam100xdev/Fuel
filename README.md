# ⛽ FuelSense AI — Fuel Price Prediction
### India & Global · Hackathon 2026

**Models:** Random Forest · Gradient Boosting (XGBoost-style) · LSTM  
**Dataset:** India 10 cities · Jan 2022 – Mar 2026 · Monthly  
**Dashboard:** 8-page Streamlit app with 3D charts, heatmaps, simulator

---

## FOLDER STRUCTURE

```
fuel-ml/
│
├── data/
│   ├── india_fuel_prices.csv     ← 510 rows · 10 cities · 2022–2026
│   └── global_fuel_data.csv      ← 75 months · Brent/WTI/Gas/Diesel
│
├── models/
│   ├── train_models.py           ← Train all models ⬅ RUN THIS FIRST
│   ├── rf_petrol.pkl             ← Random Forest (auto-saved)
│   ├── xgb_petrol.pkl            ← Gradient Boosting (auto-saved)
│   ├── scaler_petrol.pkl         ← Feature scaler
│   ├── rf_diesel.pkl
│   ├── xgb_diesel.pkl
│   └── scaler_diesel.pkl
│
├── app/
│   └── streamlit_app.py          ← 8-page Streamlit dashboard ⬅ RUN THIS
│
├── utils/
│   └── data_loader.py            ← Shared data utilities
│
├── requirements.txt
└── README.md
```

---

## SETUP IN VS CODE (5 steps)

### Step 1 — Virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### Step 2 — Install packages
```bash
pip install -r requirements.txt
```

> TensorFlow optional (for LSTM):
> ```bash
> pip install tensorflow      # or tensorflow-cpu
> ```

### Step 3 — Train models
```bash
python models/train_models.py
```
Expected output:
```
Petrol: RF R²=0.960  GBM R²=0.957  Ensemble R²=0.959  MAE=₹1.33
Diesel: RF R²=0.685  GBM R²=0.693  Ensemble R²=0.690  MAE=₹1.93
```

### Step 4 — Launch dashboard
```bash
streamlit run app/streamlit_app.py
```
Opens at: **http://localhost:8501**

### Step 5 — Navigate 8 pages
| Page | Wow Factor |
|------|-----------|
| 🏠 Live Dashboard | KPIs + City trends + Brent chart |
| 🗺 City Heatmap | **Interactive India bubble map** |
| 🔮 AI Prediction | Ensemble prediction + radar chart |
| ⚡ Price Simulator | **Real-time impact across all 10 cities** |
| 📈 Forecast + Bands | **95% confidence bands · 12 months ahead** |
| 🌍 Global Markets | **3D Brent·USD·Gas chart** + OPEC timeline |
| 📰 News Sentiment | Sentiment gauge + custom headline analyzer |
| 🤖 Model Analytics | Performance metrics + feature importance |

---

## DATASET FEATURES

| Feature | Description | Gold/Silver Impact |
|---------|-------------|-------------------|
| Brent_Crude | Oil price $/bbl | 🥇 Primary driver |
| USD_INR | Rupee exchange rate | 🥇 Primary driver |
| Inflation | CPI % | 🥈 Cost pressure |
| Excise_Duty | Central tax ₹/L | 🥈 Policy driver |
| State_VAT | State tax % | 🥈 City difference |
| OPEC_Cut | Supply policy 0/1 | 🥉 Structural |
| VIX | Market volatility | 🥉 Risk signal |
| Month | Seasonality 1–12 | 🥉 Demand cycle |
| t | Time trend | 🥉 Structural trend |

---

## MODELS EXPLAINED

### 1. Random Forest (Best for accuracy)
- 300 decision trees, max_depth=14
- Captures non-linear interactions (e.g. OPEC cut AND weak rupee = large price rise)
- Provides feature importance scores

### 2. Gradient Boosting (XGBoost-style)
- Sequential tree building — each tree corrects errors of previous
- Huber loss = robust to outlier price spikes (COVID, Russia-Ukraine)
- 400 estimators, learning_rate=0.04

### 3. Ensemble (RF + GBM average)
- Reduces variance of RF + bias of GBM
- Consistently outperforms either model alone

### 4. LSTM Neural Network (Optional - needs TensorFlow)
- 12-month lookback window
- Captures temporal lag effects (Brent spike → India price rise 4–6 weeks later)
- Architecture: LSTM(64) → LSTM(32) → Dense(16) → Price

---

## VIVA ANSWERS

**Q: Why use Gradient Boosting instead of plain XGBoost?**  
A: Gradient Boosting from scikit-learn implements the same boosting concept as XGBoost. The key advantage is Huber loss which makes it robust to price spike outliers like the 2022 Russia-Ukraine shock.

**Q: Why is Petrol R² (0.96) much better than Diesel (0.69)?**  
A: Petrol pricing in India closely tracks the Brent-crude → excise duty → VAT chain, which our features capture well. Diesel prices are more influenced by subsidies, agricultural demand cycles, and OMC (Oil Marketing Company) discretionary decisions — harder to predict with macro features alone.

**Q: What is the confidence band in your forecast?**  
A: 1.96 × volatility × √horizon. This is the 95% confidence interval from the Central Limit Theorem applied to price changes. The band widens with time because uncertainty compounds.

**Q: How does your news sentiment analyzer work?**  
A: Keyword matching against known bullish/bearish words with price impact weighting. In production, this would use a fine-tuned BERT model on energy news corpus.

**Q: How did you handle the COVID oil price crash of 2020?**  
A: The dataset starts from 2022 — post-COVID. The Huber loss function in GBM is also inherently robust to such outliers, penalizing large errors less aggressively than MSE.
# Fuel
