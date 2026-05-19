import streamlit as st
import cv2
import time
import torch
import numpy as np

from tv_engine import (
    get_weather_tile,
    forecast_sequence
)

from model import ConvLSTMWeather

# =========================
# MODEL
# =========================

model = ConvLSTMWeather()
model.eval()

st.set_page_config(layout="wide")

st.title("📺 WEATHER TV CHANNEL AI")

# =========================
# LIVE TILE MAP
# =========================

tile = get_weather_tile("clouds_new")

st.image(tile, caption="Live Satellite Map (OpenWeather)", use_container_width=True)

# =========================
# PREDICTION SIMULATION
# =========================

weather_type = np.random.randint(0, 3)

st.subheader("AI Forecast Engine")

frames = forecast_sequence(tile, weather_type)

placeholder = st.empty()

progress = st.progress(0)

# =========================
# TV BROADCAST LOOP
# =========================

for i, frame in enumerate(frames):

    placeholder.image(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        use_container_width=True
    )

    progress.progress(int((i+1)/len(frames)*100))

    time.sleep(0.6)

# =========================
# PROBABILITY GRAPH
# =========================

st.subheader("Forecast Confidence")

probs = torch.softmax(torch.randn(3), dim=0).numpy()

st.bar_chart({
    "Clear": [probs[0]],
    "Cloudy": [probs[1]],
    "Rain": [probs[2]]
})