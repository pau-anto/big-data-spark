"""
app.py — Dashboard Streamlit : Classification Tulipes / Lys

"""

import io
import os
import struct
import subprocess
import uuid

import streamlit as st
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Tulipes vs Lys", layout="wide")
st.title("🌷 Classification Tulipes / Lys")

PREDICTIONS_PATH = "../output/predictions/"
SUMMARY_PATH     = "../output/summary/"
MODEL_DIR        = "../output/model/"
UPLOAD_TMP_DIR   = "../output/upload_tmp/"
PREDICT_SCRIPT   = "../predict_upload.py"
IMG_SIZE = (64, 64)

os.makedirs(UPLOAD_TMP_DIR, exist_ok=True)


@st.cache_data
def load_predictions(path):
    return pd.read_parquet(path)


@st.cache_data
def load_summary(path):
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

# ── Comparaison des 4 variantes (agrégation calculée par Spark) ──────────
st.subheader("Comparaison des 4 variantes")
try:
    summary = load_summary(SUMMARY_PATH)
    fig_compare = px.bar(
        summary, x="variant", y="accuracy",
        text=[f"{a:.1%}" for a in summary["accuracy"]],
        color="variant",
        labels={"accuracy": "Accuracy", "variant": "Variante"},
    )
    fig_compare.update_traces(textposition="outside")
    fig_compare.update_layout(yaxis_range=[0, 1], showlegend=False)
    st.plotly_chart(fig_compare, use_container_width=True)
except Exception as e:
    st.warning(f"Résumé Spark introuvable ({e}) .")

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

# ── Histogramme de la confiance ─────────────────────────────────────────
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

# ── Tester avec ta propre image — tout le calcul passe par Spark ────────
st.divider()
st.header("Tester avec ta propre image")

variant_choice = st.selectbox(
    "Modèle à utiliser",
    ["color_bytes", "color_normalized", "grayscale", "grayscale_normalized"],
    key="upload_variant",
)

uploaded_file = st.file_uploader("Dépose une image (tulipe ou lys)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image_bytes = uploaded_file.read()
    st.image(image_bytes, caption="Image déposée", width="stretch")

    tmp_id = uuid.uuid4().hex
    image_path = os.path.join(UPLOAD_TMP_DIR, f"{tmp_id}.jpg")
    output_path = os.path.join(UPLOAD_TMP_DIR, f"{tmp_id}_result.parquet")

    with open(image_path, "wb") as f:
        f.write(image_bytes)

    with st.spinner("Traitement Spark en cours..."):
        result = subprocess.run(
            ["python", PREDICT_SCRIPT, image_path, variant_choice, output_path, MODEL_DIR],
            capture_output=True, text=True,
        )

    if result.returncode != 0:
        st.error("Erreur pendant le traitement Spark :")
        st.code(result.stderr)
    else:
        pred_df = pd.read_parquet(output_path)
        row = pred_df.iloc[0]
        st.subheader(f"Prédiction : **{row['predicted_label']}**")
        st.metric("Confiance", f"{row['confidence']:.1%}")