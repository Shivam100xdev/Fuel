# utils/data_loader.py
import pandas as pd
import numpy as np
import os

BASE = os.path.join(os.path.dirname(__file__), '..', 'data')

CITY_COORDS = {
    'Delhi':     {'lat': 28.6, 'lon': 77.2, 'state': 'Delhi'},
    'Mumbai':    {'lat': 19.1, 'lon': 72.9, 'state': 'Maharashtra'},
    'Bengaluru': {'lat': 12.9, 'lon': 77.6, 'state': 'Karnataka'},
    'Chennai':   {'lat': 13.1, 'lon': 80.3, 'state': 'Tamil Nadu'},
    'Kolkata':   {'lat': 22.6, 'lon': 88.4, 'state': 'West Bengal'},
    'Hyderabad': {'lat': 17.4, 'lon': 78.5, 'state': 'Telangana'},
    'Pune':      {'lat': 18.5, 'lon': 73.9, 'state': 'Maharashtra'},
    'Ahmedabad': {'lat': 23.0, 'lon': 72.6, 'state': 'Gujarat'},
    'Jaipur':    {'lat': 26.9, 'lon': 75.8, 'state': 'Rajasthan'},
    'Lucknow':   {'lat': 26.8, 'lon': 80.9, 'state': 'Uttar Pradesh'},
}

FEATURES = ['Brent_Crude','USD_INR','Inflation','Excise_Duty','OPEC_Cut','VIX','State_VAT','Month','t']

def load_india():
    df = pd.read_csv(os.path.join(BASE, 'india_fuel_prices.csv'))
    df['Date']  = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Year']  = df['Date'].dt.year
    df['t']     = (df['Year'] - 2022) * 12 + df['Month']
    return df

def load_global():
    df = pd.read_csv(os.path.join(BASE, 'global_fuel_data.csv'))
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def get_latest_india(df):
    return df.sort_values('Date').groupby('City').last().reset_index()

def get_city_timeseries(df, city):
    return df[df['City'] == city].sort_values('Date')

def forecast_future(df, city, fuel_type, horizon_months=6):
    """Simple trend-based forecast with confidence bands."""
    city_df = get_city_timeseries(df, city)
    prices  = city_df[fuel_type].dropna().values
    if len(prices) < 6:
        return None, None, None

    # Linear trend on last 12 months
    last12 = prices[-12:]
    x      = np.arange(len(last12))
    slope, intercept = np.polyfit(x, last12, 1)
    volatility = np.std(np.diff(last12))

    last_date = city_df['Date'].iloc[-1]
    future_dates  = pd.date_range(last_date + pd.offsets.MonthBegin(1),
                                  periods=horizon_months, freq='MS')
    future_prices = [prices[-1] + slope * (i + 1) for i in range(horizon_months)]
    upper = [p + 1.96 * volatility * np.sqrt(i + 1) for i, p in enumerate(future_prices)]
    lower = [p - 1.96 * volatility * np.sqrt(i + 1) for i, p in enumerate(future_prices)]

    return future_dates, np.array(future_prices), np.array(upper), np.array(lower)

# Sentiment signals (static for demo — replace with live news API)
SENTIMENT_NEWS = [
    {'headline': 'OPEC+ agrees to extend output cuts through Q2 2026',      'sentiment': 'Bearish 🔴', 'impact': '+₹2.1/L', 'date': 'Mar 2026'},
    {'headline': 'India rupee strengthens to 82.5 vs USD on RBI intervention','sentiment': 'Bullish 🟢','impact': '-₹0.8/L', 'date': 'Mar 2026'},
    {'headline': 'Brent crude spikes 8% on Middle East supply disruption',    'sentiment': 'Bearish 🔴', 'impact': '+₹3.5/L', 'date': 'Feb 2026'},
    {'headline': 'India reduces excise duty by ₹2/L ahead of elections',      'sentiment': 'Bullish 🟢', 'impact': '-₹2.0/L', 'date': 'Feb 2026'},
    {'headline': 'US crude inventories rise 4.2M barrels — oversupply signal', 'sentiment': 'Bullish 🟢','impact': '-₹1.2/L', 'date': 'Jan 2026'},
    {'headline': 'IEA cuts 2026 global oil demand growth forecast',            'sentiment': 'Bullish 🟢', 'impact': '-₹1.5/L', 'date': 'Jan 2026'},
    {'headline': 'Russia-Ukraine ceasefire talks stall — energy markets jittery','sentiment':'Bearish 🔴','impact':'+₹1.8/L','date':'Dec 2025'},
]
