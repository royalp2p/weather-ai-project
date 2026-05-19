import requests
import numpy as np
import cv2
import torch

# =========================
# OPENWEATHER CONFIG
# =========================

API_KEY = "YOUR_OPENWEATHER_KEY"

BASE_TILE = "https://tile.openweathermap.org/map/{layer}/{z}/{x}/{y}.png?appid=" + API_KEY


def get_weather_tile(layer="clouds_new", z=5, x=10, y=10):
    url = BASE_TILE.format(layer=layer, z=z, x=x, y=y)
    resp = requests.get(url)
    img_arr = np.frombuffer(resp.content, np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    return img


# =========================
# RAIN SIMULATION (3D depth)
# =========================

def generate_rain(frame):
    h, w, _ = frame.shape
    rain = frame.copy()

    for _ in range(500):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)

        length = np.random.randint(5, 20)

        cv2.line(
            rain,
            (x, y),
            (x + 1, y + length),
            (255, 255, 255),
            1
        )

    return rain


# =========================
# CLOUD NOISE (Perlin-like fake)
# =========================

def generate_cloud_layer(frame):
    overlay = frame.copy()

    noise = np.random.normal(0, 25, frame.shape).astype(np.uint8)

    cloud = cv2.addWeighted(overlay, 0.7, noise, 0.3, 0)

    return cloud


# =========================
# CYCLONE EFFECT
# =========================

def add_cyclone(frame):
    h, w, _ = frame.shape
    center = (w // 2, h // 2)

    for r in range(10, 120, 10):
        cv2.circle(frame, center, r, (80, 80, 80), 1)

    return frame


# =========================
# TV TIMELINE FORECAST
# =========================

def forecast_sequence(base_frame, weather_type):
    frames = []

    for t in range(6):  # +6h simulation
        frame = base_frame.copy()

        if weather_type == 0:  # clear
            frame = cv2.GaussianBlur(frame, (3, 3), 0)

        elif weather_type == 1:  # cloudy
            frame = generate_cloud_layer(frame)

        elif weather_type == 2:  # rain
            frame = generate_rain(frame)

        if t > 3:
            frame = add_cyclone(frame)

        frames.append(frame)

    return frames