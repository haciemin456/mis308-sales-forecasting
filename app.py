import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import datetime

st.set_page_config(
    page_title="Sales Forecasting — MIS 308",
    page_icon="📈",
    layout="centered"
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📈 Sales Forecasting App")
st.markdown("**MIS 308 Data Science Project** | Ostim Technical University")
st.markdown("**Student:** Hacı Emin Ayhan — 220106012")
st.divider()

# ── Load & Train Model ────────────────────────────────────────────────────────
@st.cache_resource
def train_model():
    df = pd.read_csv('Sample_-_Superstore.csv', encoding='latin1')
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Ship Date']  = pd.to_datetime(df['Ship Date'])

    df['Year']        = df['Order Date'].dt.year
    df['Month']       = df['Order Date'].dt.month
    df['DayOfWeek']   = df['Order Date'].dt.dayofweek
    df['Quarter']     = df['Order Date'].dt.quarter
    df['ShipDays']    = (df['Ship Date'] - df['Order Date']).dt.days

    le_seg  = LabelEncoder().fit(df['Segment'])
    le_reg  = LabelEncoder().fit(df['Region'])
    le_cat  = LabelEncoder().fit(df['Category'])
    le_sub  = LabelEncoder().fit(df['Sub-Category'])
    le_ship = LabelEncoder().fit(df['Ship Mode'])

    df['Segment_enc']     = le_seg.transform(df['Segment'])
    df['Region_enc']      = le_reg.transform(df['Region'])
    df['Category_enc']    = le_cat.transform(df['Category'])
    df['SubCategory_enc'] = le_sub.transform(df['Sub-Category'])
    df['ShipMode_enc']    = le_ship.transform(df['Ship Mode'])

    FEATURES = ['Quantity','Discount','Segment_enc','Region_enc',
                'Category_enc','SubCategory_enc','ShipMode_enc',
                'Year','Month','DayOfWeek','Quarter','ShipDays']

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(df[FEATURES], df['Sales'])

    encoders = {
        'Segment': le_seg, 'Region': le_reg,
        'Category': le_cat, 'Sub-Category': le_sub, 'Ship Mode': le_ship
    }
    categories = {
        'Segment': df['Segment'].unique().tolist(),
        'Region': df['Region'].unique().tolist(),
        'Category': df['Category'].unique().tolist(),
        'Sub-Category': df['Sub-Category'].unique().tolist(),
        'Ship Mode': df['Ship Mode'].unique().tolist(),
    }
    return model, encoders, categories

with st.spinner("Loading model..."):
    model, encoders, categories = train_model()

st.success("✅ Model loaded and ready!")
st.divider()

# ── Input Form ────────────────────────────────────────────────────────────────
st.subheader("🛒 Enter Order Details")

col1, col2 = st.columns(2)

with col1:
    category    = st.selectbox("Category",    sorted(categories['Category']))
    sub_cat     = st.selectbox("Sub-Category", sorted(categories['Sub-Category']))
    segment     = st.selectbox("Segment",     sorted(categories['Segment']))

with col2:
    region      = st.selectbox("Region",      sorted(categories['Region']))
    ship_mode   = st.selectbox("Ship Mode",   sorted(categories['Ship Mode']))
    order_date  = st.date_input("Order Date", datetime.date(2017, 1, 1))

col3, col4 = st.columns(2)
with col3:
    quantity = st.slider("Quantity", 1, 14, 3)
with col4:
    discount = st.slider("Discount", 0.0, 0.8, 0.0, step=0.05,
                         format="%.2f")

ship_days = st.slider("Ship Days (order → delivery)", 1, 7, 3)

st.divider()

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🔮 Predict Sales", use_container_width=True, type="primary"):
    order_dt = pd.Timestamp(order_date)

    try:
        seg_enc  = encoders['Segment'].transform([segment])[0]
        reg_enc  = encoders['Region'].transform([region])[0]
        cat_enc  = encoders['Category'].transform([category])[0]
        sub_enc  = encoders['Sub-Category'].transform([sub_cat])[0]
        ship_enc = encoders['Ship Mode'].transform([ship_mode])[0]
    except Exception:
        st.error("Unknown category value. Please select from the dropdown.")
        st.stop()

    X = np.array([[
        quantity, discount,
        seg_enc, reg_enc, cat_enc, sub_enc, ship_enc,
        order_dt.year, order_dt.month, order_dt.dayofweek,
        order_dt.quarter, ship_days
    ]])

    prediction = model.predict(X)[0]

    st.subheader("📊 Prediction Result")
    st.metric(label="Predicted Sales Amount", value=f"${prediction:,.2f}")

    # Interpretation
    if prediction < 100:
        level, color = "Low", "🔵"
    elif prediction < 500:
        level, color = "Medium", "🟡"
    elif prediction < 1500:
        level, color = "High", "🟠"
    else:
        level, color = "Very High", "🔴"

    st.info(f"{color} Sales Level: **{level}**")

    # Summary
    st.markdown("#### Order Summary")
    summary = {
        "Category": category, "Sub-Category": sub_cat,
        "Segment": segment, "Region": region,
        "Ship Mode": ship_mode, "Quantity": quantity,
        "Discount": f"{discount*100:.0f}%",
        "Order Date": str(order_date), "Ship Days": ship_days,
        "**Predicted Sales**": f"**${prediction:,.2f}**"
    }
    for k, v in summary.items():
        st.markdown(f"- {k}: {v}")

st.divider()
st.caption("MIS 308 Data Science Project | Ostim Technical University | 2025-2026 Spring") 
