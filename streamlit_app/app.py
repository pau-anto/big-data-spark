import io
import struct

import streamlit as st
from PIL import Image
import pandas as pd

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

cols = st.columns(4)
for i, row in filtered.reset_index(drop=True).iterrows():
    with cols[i % 4]:
        img = bytes_to_image(row["pixels"])
        st.image(img, caption=row["image_id"], width="stretch")        
        correct = row["true_label"] == row["predicted_label"]
        icon = "✅" if correct else "❌"
        st.write(f"{icon} Vrai : **{row['true_label']}**")
        st.write(f"Prédit : **{row['predicted_label']}** ({row['confidence']:.1%})")