import streamlit as st
import torch
import cv2
import numpy as np

from model import ImprovedCNN

# MODEL

model = ImprovedCNN()

model.load_state_dict(
    torch.load("improved.pth")
)

model.eval()

# CLASSES

classes = [
    "Ясно",
    "Облачно",
    "Дождь"
]

# UI

st.title("AI Прогноз погоды по спутниковым снимкам")

st.write(
    "Загрузите 3 последовательных спутниковых изображения"
)

# FILES

files = st.file_uploader(
    "Загрузить изображения",
    type=["jpg", "png"],
    accept_multiple_files=True
)

# PREDICTION

if files and len(files) == 3:

    images = []

    cols = st.columns(3)

    for i, file in enumerate(files):

        file_bytes = np.asarray(
            bytearray(file.read()),
            dtype=np.uint8
        )

        img = cv2.imdecode(file_bytes, 1)

        cols[i].image(
            img,
            caption=f"Снимок {i+1}"
        )

        img = cv2.resize(img, (128, 128))

        img = img / 255.0

        img = torch.tensor(
            img,
            dtype=torch.float32
        ).permute(2, 0, 1)

        images.append(img)

    # CONCAT 3 IMAGES

    final_img = torch.cat(images, dim=0)

    final_img = final_img.unsqueeze(0)

    # MODEL PREDICTION

    with torch.no_grad():

        output = model(final_img)

        pred = torch.argmax(output, dim=1).item()

    # RESULT

    st.success(
        f"Прогноз погоды: {classes[pred]}"
    )

# ERROR

elif files and len(files) != 3:

    st.error(
        "Нужно загрузить ровно 3 изображения"
    )