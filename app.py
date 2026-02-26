import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import io
from datetime import datetime, timedelta
import openmeteo_requests
import requests_cache
from retry_requests import retry
from statsmodels.tsa.arima.model import ARIMA

# ==========================================
# 1. UI & CONFIGURATION
# ==========================================
st.set_page_config(page_title="Changi Aviation Weather", page_icon="✈️", layout="wide")

st.title("✈️ Changi Airport Daily Weather Forecast")
st.markdown("""
This tool uses an **ARIMAX** statistical model to forecast daily rainfall severity at Changi Airport (Station S24). 
It dynamically scrapes real-time local ground truth data and regional meteorological signals to generate predictions for ground handling and tarmac safety roster planning.
""")

# ==========================================
# 2. LOAD ASSETS (Weights Only)
# ==========================================
@st.cache_resource
def load_models():
    # We load the lightweight 5KB weights file (No scalers needed for ARIMAX!)
    with open('arimax_weights.pkl', 'rb') as f:
        weights = pickle.load(f)
    return weights

try:
    weights = load_models()
    models_loaded = True
except Exception as e:
    st.error(f"⚠️ Could not load assets. Please ensure 'arimax_weights.pkl' is in the repo. Error: {e}")
    models_loaded = False

# ==========================================
# 3. DATA SCRAPING FUNCTIONS
# ==========================================
def cleanup(df):
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        'Highest 30 Min Rainfall (mm)': 'Highest 30-min Rainfall (mm)', 'Highest 30 min Rainfall (mm)': 'Highest 30-min Rainfall (mm)',
        'Highest 60 Min Rainfall (mm)': 'Highest 60-min Rainfall (mm)', 'Highest 60 min Rainfall (mm)': 'Highest 60-min Rainfall (mm)',
        'Highest 120 Min Rainfall (mm)': 'Highest 120-min Rainfall (mm)', 'Highest 120 min Rainfall (mm)': 'Highest 120-min Rainfall (mm)',
        'Mean Temperature (Â°C)': 'Mean Temperature (°C)', 'Maximum Temperature (Â°C)': 'Maximum Temperature (°C)', 'Minimum Temperature (Â°C)': 'Minimum Temperature (°C)'
    })
    if {'Year','Month','Day'}.issubset(df.columns): 
        df['Date'] = pd.to_datetime(df[['Year','Month','Day']])
    target = ['Date','Daily Rainfall Total (mm)','Highest 30-min Rainfall (mm)','Highest 60-min Rainfall (mm)','Highest 120-min Rainfall (mm)',
              'Mean Temperature (°C)','Maximum Temperature (°C)','Minimum Temperature (°C)','Mean Wind Speed (km/h)','Max Wind Speed (km/h)']
    return df[[c for c in target if c in df.columns]]

@st.cache_data(ttl=3600) # Cache for 1 hour
def fetch_recent_data():
    # A. Scrape Local Data (S24) for current and previous month
    now = datetime.now()
    prev = now.replace(day=1) - timedelta(days=1)
    dates_to_scrape = [(prev.year, prev.month), (now.year, now.month)]
    
    dfs = []
    url = "https://www.weather.gov.sg/files/dailydata/DAILYDATA_"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    for y, m in dates_to_scrape:
        try:
            r = requests.get(f"{url}S24_{y}{m:02d}.csv", headers=headers, timeout=10)
            if r.status_code == 200: 
                dfs.append(cleanup(pd.read_csv(io.BytesIO(r.content), encoding='latin1', na_values=['—', '-'])))
        except Exception:
            pass
            
    stn_df = pd.concat(dfs, ignore_index=True).sort_values('Date').reset_index(drop=True)
    stn_df = stn_df.set_index('Date')
    stn_df.columns = [f"{c}_S24" for c in stn_df.columns]
    
    # B. Scrape Regional Data (Open-Meteo) for the last 40 days
    start_date = (now - timedelta(days=40)).strftime("%Y-%m-%d")
    end_date = now.strftime("%Y-%m-%d")
    locs = {"Singapore": (1.352,103.82), "Johor": (1.49,103.74), "Batam": (1.04,104.03)}
    
    session = retry(requests_cache.CachedSession('.cache', expire_after=3600), retries=5, backoff_factor=0.2)
    om = openmeteo_requests.Client(session=session)
    res = om.weather_api("https://archive-api.open-meteo.com/v1/archive", params={
        "latitude": [v[0] for v in locs.values()], "longitude": [v[1] for v in locs.values()],
        "start_date": start_date, "end_date": end_date, "hourly": ["precipitation", "weather_code", "wind_speed_10m"]
    })

    reg_dfs = []
    for i, (name, _) in enumerate(locs.items()):
        h = res[i].Hourly()
        dates = pd.date_range(pd.to_datetime(h.Time(), unit="s", utc=True), pd.to_datetime(h.TimeEnd(), unit="s", utc=True), freq=pd.Timedelta(seconds=h.Interval()), inclusive="left")
        d = pd.DataFrame({"Date": dates.strftime('%Y-%m-%d'), "Rain": h.Variables(0).ValuesAsNumpy(),
                          "Code": h.Variables(1).ValuesAsNumpy(), "Wind": h.Variables(2).ValuesAsNumpy()})
        d = d.groupby("Date").agg({"Rain": "sum", "Code": "max", "Wind": "mean"})
        d.columns = [f"{c}_{name}" for c in d.columns]
        reg_dfs.append(d)
        
    reg_df = pd.concat(reg_dfs, axis=1)
    reg_df.index = pd.to_datetime(reg_df.index)
    stn_df.index = pd.to_datetime(stn_df.index)
    
    # C. Merge Data
    common = stn_df.index.intersection(reg_df.index)
    df = pd.concat([stn_df.loc[common], reg_df.loc[common]], axis=1).dropna(thresh=10, axis=1)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(df.mean(numeric_only=True))
    
    return df

# ==========================================
# 4. PREDICTION LOGIC (Dynamic Reconstruction)
# ==========================================
if models_loaded:
    st.subheader("Run Forecast")
    if st.button("🔮 Forecast Tomorrow's Weather"):
        with st.spinner("Fetching latest sensor data from NEA and Regional satellites..."):
            df = fetch_recent_data()
            
        if df.empty:
            st.error("Failed to retrieve sufficient data. Please try again later.")
        else:
            try:
                # 1. Prepare the raw scraped data (NO SCALING NEEDED!)
                target_col = 'Daily Rainfall Total (mm)_S24'
                exog_cols = [c for c in df.columns if c != target_col]
                
                recent_y = df[target_col].values
                recent_x = df[exog_cols]
                
                # 2. Reconstruct the Model dynamically
                # Build an "empty" model with the recent data structure
                live_model = ARIMA(recent_y, exog=recent_x, order=(30,1,0))
                
                # Inject the 42-year trained weights into this live model
                live_model_fitted = live_model.smooth(weights)
                
                # 3. Forecast Tomorrow
                latest_exog = recent_x.iloc[-1:] # Use today's exog to predict tomorrow
                pred_mm = live_model_fitted.forecast(steps=1, exog=latest_exog).iloc[0]
                pred_mm = max(pred_mm, 0) # No negative rain
                
                # 4. Classify
                if pred_mm < 0.2:
                    severity = "Class 0: No Rain / Clear Skies"
                    color = "success"
                    action = "Normal Operations."
                elif pred_mm <= 10:
                    severity = "Class 1: Moderate Rain"
                    color = "info"
                    action = "Standard Wet Weather Ground Handling."
                else:
                    severity = "Class 2: Heavy Storm"
                    color = "error"
                    action = "🚨 INITIATE TARMAC SAFETY PROTOCOLS. Potential flight delays."
                
                # 5. Display Dashboard
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Rainfall (Tomorrow)", f"{pred_mm:.2f} mm")
                with col2:
                    st.metric("Latest Local Temp", f"{df['Mean Temperature (°C)_S24'].iloc[-1]:.1f} °C")
                
                if color == "success":
                    st.success(f"**Severity:** {severity}\n\n**Recommendation:** {action}")
                elif color == "info":
                    st.info(f"**Severity:** {severity}\n\n**Recommendation:** {action}")
                else:
                    st.error(f"**Severity:** {severity}\n\n**Recommendation:** {action}")
                
                with st.expander("View Recent Meteorological Data"):
                    st.dataframe(df.tail(5))
                    
            except Exception as e:
                st.error(f"Prediction failed. Error: {e}")
