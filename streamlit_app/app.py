import streamlit as st

st.set_page_config(page_title="Amlezia - Image Pipeline", layout="wide")
st.title("Pipeline de traitement d'images")

uploaded = st.file_uploader("Charger une image", type=["png", "jpg"])
if uploaded:
    st.image(uploaded, caption="Image chargée", use_column_width=True)
