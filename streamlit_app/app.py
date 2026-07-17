import io
import struct
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Tulipes vs Lys", layout="wide")
st.title("🌷 Classification Tulipes / Lys")

PREDICTIONS_PATH = "../output/predictions/"
IMG_SIZE = (64, 64)


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

df["correct"] = df["true_label"] == df["predicted_label"]

st.sidebar.header("Filtres")
variants = df["variant"].unique().tolist()
selected_variant = st.sidebar.selectbox("Variante de prétraitement", variants)

filtered = df[df["variant"] == selected_variant]

st.subheader(f"Résultats — {selected_variant}")

# ── KPIs ─────────────────────────────────────────────────────────────────
n_tulipes = (filtered["true_label"] == "tulipes").sum()
n_lys = (filtered["true_label"] == "lys").sum()
accuracy = filtered["correct"].mean()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total images", len(filtered))
c2.metric("🌷 Tulipes (vérité)", n_tulipes)
c3.metric("🌸 Lys (vérité)", n_lys)
c4.metric("Accuracy", f"{accuracy:.1%}")
c5.metric("Confiance moyenne", f"{filtered['confidence'].mean():.1%}")

st.divider()

# ── Comparaison des 4 variantes ─────────────────────────────────────────
st.subheader("Comparaison des 4 variantes")
summary = (
    df.groupby("variant")
    .agg(n_test=("correct", "count"), accuracy=("correct", "mean"), avg_confidence=("confidence", "mean"))
    .reset_index()
)
fig_compare = px.bar(
    summary, x="variant", y="accuracy",
    text=[f"{a:.1%}" for a in summary["accuracy"]],
    color="variant",
    labels={"accuracy": "Accuracy", "variant": "Variante"},
)
fig_compare.update_traces(textposition="outside")
fig_compare.update_layout(yaxis_range=[0, 1], showlegend=False)
st.plotly_chart(fig_compare, use_container_width=True)

st.divider()

# ── Ligne 1 : pie chart + barres de confiance ───────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Répartition Tulipes / Lys (vérité terrain)")
    fig_pie = px.pie(
        values=[n_tulipes, n_lys],
        names=["🌷 Tulipes", "🌸 Lys"],
        color_discrete_sequence=["#FF6B9D", "#A78BFA"],
        hole=0.4
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.subheader("Confiance par image (échantillon)")
    df_sorted = filtered.head(20).sort_values("confidence", ascending=True)
    colors = ["#2ECC71" if c else "#E74C3C" for c in df_sorted["correct"]]
    fig_bar = go.Figure(go.Bar(
        x=df_sorted["confidence"],
        y=df_sorted["image_id"],
        orientation="h",
        marker_color=colors,
        text=[f"{c:.0%}" for c in df_sorted["confidence"]],
        textposition="outside"
    ))
    fig_bar.add_vline(x=0.5, line_dash="dash", line_color="gray", annotation_text="Seuil 50%")
    fig_bar.update_layout(xaxis_range=[0, 1.15], xaxis_title="Confiance", yaxis_title="Image",
                           height=500)
    st.plotly_chart(fig_bar, use_container_width=True)
st.caption("🟢 Prédiction correcte · 🔴 Prédiction incorrecte")


# ── Histogramme de la confiance ───────
st.divider()
st.subheader("Distribution de la confiance du modèle")
fig_conf = px.histogram(
    filtered, x="confidence", color="correct",
    nbins=20,
    color_discrete_map={True: "#2ECC71", False: "#E74C3C"},
    labels={"confidence": "Confiance de la prédiction", "correct": "Prédiction correcte"},
    barmode="overlay",
    opacity=0.75,
)
fig_conf.add_vline(x=0.5, line_dash="dash", line_color="gray", annotation_text="Seuil 50%")
fig_conf.update_layout(
    xaxis_tickformat=".0%",
    yaxis_title="Nombre d'images",
    legend_title="Correct ?",
)
st.plotly_chart(fig_conf, use_container_width=True)

# ── Galerie d'images ────────────────────────────────────────────────────
st.divider()
st.subheader("Galerie")
st.caption(f"{len(filtered)} images au total — aperçu des 12 premières")
cols = st.columns(4)
for i, row in filtered.head(12).reset_index(drop=True).iterrows():
    with cols[i % 4]:
        img = bytes_to_image(row["pixels"])
        st.image(img, caption=row["image_id"], width="stretch")
        icon = "✅" if row["correct"] else "❌"
        st.write(f"{icon} Vrai : **{row['true_label']}**")
        st.write(f"Prédit : **{row['predicted_label']}** ({row['confidence']:.1%})")

# ── Tableau détaillé ─────────────────────────────────────────────────────
st.divider()
st.subheader("Résultats détaillés")
st.dataframe(
    filtered[["image_id", "true_label", "predicted_label", "confidence", "prob_lys", "prob_tulipes", "correct"]]
    .style
    .format({"confidence": "{:.1%}", "prob_lys": "{:.1%}", "prob_tulipes": "{:.1%}"})
    .apply(lambda row: ["background-color: #E5F9E5" if row["correct"] else "background-color: #FFE5E5" for _ in row], axis=1),
    use_container_width=True
)

st.divider()
st.header("Tester avec ta propre image")

MODEL_DIR = "../output/model/"


def resize_to_pixels(pil_img, size=(64, 64)):
    img = pil_img.convert("RGB").resize(size, Image.LANCZOS)
    raw = img.tobytes()
    return list(raw)


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