"""
predict_upload.py — Prédiction Spark sur une image unique déposée via Streamlit.
Reprend EXACTEMENT les mêmes fonctions de prétraitement que le notebook,
pour garantir que le calcul reste dans Spark (règle "DataFrame uniquement").

Usage : python predict_upload.py <image_path> <variant> <output_parquet> <model_dir>
"""

import sys
import os
import io
import struct

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, IntegerType, FloatType, ArrayType, BinaryType
)
import joblib
import numpy as np
import pandas as pd

TARGET_W, TARGET_H = 64, 64


# ── Mêmes fonctions que le notebook (parsing) ──────────────────────────────
def decode_image_bytes(raw_bytes: bytes):
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        img = img.resize((TARGET_W, TARGET_H), Image.LANCZOS)
        raw = img.tobytes()
        n = len(raw)
        pixels = list(struct.unpack(f"{n}B", raw))
        return (TARGET_W, TARGET_H, 3, [float(p) for p in pixels])
    except Exception:
        return None


_decode_schema = StructType([
    StructField("width", IntegerType(), False),
    StructField("height", IntegerType(), False),
    StructField("channels", IntegerType(), False),
    StructField("pixels", ArrayType(FloatType()), False),
])


# ── Mêmes fonctions que le notebook (prétraitement) ─────────────────────────
def normalize_rgb(pixels):
    if pixels is None:
        return None
    return [p / 255.0 for p in pixels]


def rgb_to_grayscale(pixels):
    if pixels is None:
        return None
    gray = []
    for i in range(0, len(pixels), 3):
        r, g, b = pixels[i], pixels[i + 1], pixels[i + 2]
        gray.append(0.299 * r + 0.587 * g + 0.114 * b)
    return gray


def rgb_to_normalized_gray(pixels_norm):
    if pixels_norm is None:
        return None
    gray = []
    for i in range(0, len(pixels_norm), 3):
        r, g, b = pixels_norm[i], pixels_norm[i + 1], pixels_norm[i + 2]
        gray.append(0.299 * r + 0.587 * g + 0.114 * b)
    return gray


def pixels_to_bytes(pixels):
    if pixels is None:
        return None
    int_pixels = [int(p) for p in pixels]
    return struct.pack(f"{len(int_pixels)}B", *int_pixels)


def main():
    image_path, variant, output_path, model_dir = sys.argv[1:5]

    spark = (
        SparkSession.builder
        .appName("LivePrediction")
        .master("local[2]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # ── Lecture de l'image comme le ferait binaryFile, mais pour un seul fichier ──
    with open(image_path, "rb") as f:
        raw_bytes = f.read()

    decode_udf = F.udf(decode_image_bytes, _decode_schema)
    df = spark.createDataFrame([(raw_bytes,)], ["raw_bytes"])
    df = (
        df.withColumn("decoded", decode_udf(F.col("raw_bytes")))
        .select(F.col("decoded.pixels").alias("pixels"))
    )

    # ── Applique EXACTEMENT le même prétraitement que le notebook, selon la variante ──
    if variant == "color_bytes":
        bytes_udf = F.udf(pixels_to_bytes, BinaryType())
        df = df.withColumn("feature", bytes_udf(F.col("pixels")))

    elif variant == "color_normalized":
        norm_udf = F.udf(normalize_rgb, ArrayType(FloatType()))
        df = df.withColumn("feature", norm_udf(F.col("pixels")))

    elif variant == "grayscale":
        gray_udf = F.udf(rgb_to_grayscale, ArrayType(FloatType()))
        df = df.withColumn("feature", gray_udf(F.col("pixels")))

    elif variant == "grayscale_normalized":
        norm_udf = F.udf(normalize_rgb, ArrayType(FloatType()))
        norm_gray_udf = F.udf(rgb_to_normalized_gray, ArrayType(FloatType()))
        df = df.withColumn("pixels_norm", norm_udf(F.col("pixels")))
        df = df.withColumn("feature", norm_gray_udf(F.col("pixels_norm")))

    # ── collect() : autorisé en section ML, comme dans le notebook ─────────────
    row = df.select("feature").collect()[0]

    if variant == "color_bytes":
        raw = row["feature"]
        X = np.array(list(struct.unpack(f"{len(raw)}B", raw)), dtype=np.float32).reshape(1, -1)
    else:
        X = np.array(row["feature"], dtype=np.float32).reshape(1, -1)

    # ── Chargement du modèle et prédiction ──────────────────────────────────────
    model = joblib.load(os.path.join(model_dir, f"{variant}.joblib"))
    encoder_path = os.path.join(model_dir, f"{variant}_encoder.joblib")

    proba = model.predict_proba(X)[0]
    if os.path.exists(encoder_path):
        encoder = joblib.load(encoder_path)
        pred_label = str(encoder.inverse_transform(model.predict(X))[0])
        classes = list(encoder.classes_)
    else:
        pred_label = str(model.predict(X)[0])
        classes = list(model.classes_)

    proba_dict = dict(zip(classes, [float(p) for p in proba]))

    result = pd.DataFrame([{
        "variant": variant,
        "predicted_label": pred_label,
        "confidence": float(max(proba_dict.values())),
        "prob_lys": proba_dict.get("lys", 0.0),
        "prob_tulipes": proba_dict.get("tulipes", 0.0),
    }])
    result.to_parquet(output_path, index=False)

    spark.stop()


if __name__ == "__main__":
    main()
