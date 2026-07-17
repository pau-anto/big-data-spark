import io
import struct
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Tulipes vs Lys", layout="wide")
st.title("🌷 Classification Tulipes / Lys")

PREDICTIONS_PATH = "../output/predictions/"
IMG_SIZE = (64, 64)  # doit correspondre à TARGET_SIZE du notebook


@st.cache_data
def load_predictions(path):
    return pd.read_parquet(path)


def bytes_to_image(pixel_bytes, size=IMG_SIZE):
    w, h = size
    n = w * h * 3
    values = struct.unpack(f"{n}B", pixel_bytes)
    img = Image.frombytes("RGB", (w, h), bytes(values))
    return img


try:
    df = load_predictions(PREDICTIONS_PATH)
except Exception as e:
    st.error(f"Impossible de lire les prédictions : {e}")
    st.stop()

st.sidebar.header("Filtres")
variants = df["variant"].unique().tolist()
selected_variant = st.sidebar.selectbox("Variante de prétraitement", variants)

filtered = df[df["variant"] == selected_variant]

st.subheader(f"Résultats — {selected_variant}")
accuracy = (filtered["true_label"] == filtered["predicted_label"]).mean()
st.metric("Accuracy sur cet échantillon", f"{accuracy:.1%}")

st.caption(f"{len(filtered)} images au total — aperçu des 12 premières")
cols = st.columns(4)
for i, row in filtered.head(12).reset_index(drop=True).iterrows():
    with cols[i % 4]:
        img = bytes_to_image(row["pixels"])
        st.image(img, caption=row["image_id"], width="stretch")
        correct = row["true_label"] == row["predicted_label"]
        icon = "✅" if correct else "❌"
        st.write(f"{icon} Vrai : **{row['true_label']}**")
        st.write(f"Prédit : **{row['predicted_label']}** ({row['confidence']:.1%})")

st.divider()
st.header("Tester avec ta propre image")

MODEL_DIR = "../output/model/"


def resize_to_pixels(pil_img, size=(64, 64)):
    img = pil_img.convert("RGB").resize(size, Image.LANCZOS)
    raw = img.tobytes()
    return list(raw)  # valeurs 0-255, RGB entrelacé


def rgb_to_gray(pixels):
    gray = []
    for i in range(0, len(pixels), 3):
        r, g, b = pixels[i], pixels[i + 1], pixels[i + 2]
        gray.append(0.299 * r + 0.587 * g + 0.114 * b)
    return gray


def build_features(pixels, variant):
    if variant == "color_bytes":
        return np.array(pixels, dtype=np.float32).reshape(1, -1)
    if variant == "color_normalized":
        return np.array([p / 255.0 for p in pixels], dtype=np.float32).reshape(1, -1)
    if variant == "grayscale":
        return np.array(rgb_to_gray(pixels), dtype=np.float32).reshape(1, -1)
    if variant == "grayscale_normalized":
        gray = rgb_to_gray(pixels)
        return np.array([g / 255.0 for g in gray], dtype=np.float32).reshape(1, -1)


variant_choice = st.selectbox(
    "Modèle à utiliser",
    ["color_bytes", "color_normalized", "grayscale", "grayscale_normalized"],
    key="upload_variant",
)

uploaded_file = st.file_uploader("Dépose une image (tulipe ou lys)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    pil_img = Image.open(uploaded_file)
    st.image(pil_img, caption="Image déposée", width="stretch")

    pixels = resize_to_pixels(pil_img)
    X = build_features(pixels, variant_choice)

    model = joblib.load(f"{MODEL_DIR}{variant_choice}.joblib")

    encoder_path = f"{MODEL_DIR}{variant_choice}_encoder.joblib"
    try:
        encoder = joblib.load(encoder_path)
        pred_encoded = model.predict(X)
        pred_label = encoder.inverse_transform(pred_encoded)[0]
        proba = model.predict_proba(X)[0]
        confidence = proba.max()
    except FileNotFoundError:
        pred_label = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        confidence = proba.max()

    st.subheader(f"Prédiction : **{pred_label}**")
    st.metric("Confiance", f"{confidence:.1%}")