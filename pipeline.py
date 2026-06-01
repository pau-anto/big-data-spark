#!/usr/bin/env python
# coding: utf-8

# # Classification Tulipes / Lys — Pipeline Spark
# **ESGI · Spark core · Juillet 2026**
# 
# Pipeline mono-notebook : parsing → prétraitement → entraînement → inférence → visualisation.
# 
# Un seul fichier Parquet est écrit, juste avant la visualisation.
# 
# > **Règles** : DataFrame uniquement · pas de `collect()` / `toPandas()` / `toList()` · Python pur = affichage seulement
# 

# ## 0 · Session Spark & imports

# In[1]:


import sys
import os
import io
import struct

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    IntegerType, FloatType, ArrayType, StringType
)

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

spark = (
    SparkSession.builder
    .appName("TulipsLilies_Pipeline")
    .config("spark.sql.files.ignoreCorruptFiles", "true")
    # Augmente la mémoire pour éviter les crash sur les images
    .config("spark.driver.memory", "2g")
    .config("spark.executor.memory", "2g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
print(f"Spark version : {spark.version}")


# ## 1 · Chemins & constantes

# In[3]:


TRAIN_PATH   = "data/Train/"         # contient tulip/ et lily/
TEST_PATH    = "data/Test/"          # contient tulip/ et lily/
OUTPUT_PREDS = "output/predictions/" # seul Parquet écrit
MODEL_PATH   = "output/model/"
TARGET_SIZE  = (64, 64)


# ## 2 · Parsing
# Lecture distribuée des images avec `binaryFile`.
# La fonction `decode_image_bytes` est enregistrée comme UDF Spark : elle tourne sur chaque worker,
# redimensionne chaque image à 64×64 et retourne les pixels RGB aplatis.
# Rien ne revient sur le driver.
# 

# In[4]:


TARGET_W, TARGET_H = TARGET_SIZE

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
    StructField("width",    IntegerType(), False),
    StructField("height",   IntegerType(), False),
    StructField("channels", IntegerType(), False),
    StructField("pixels",   ArrayType(FloatType()), False),
])
decode_udf = F.udf(decode_image_bytes, _decode_schema)

def parse_images(path):
    raw = (
        spark.read.format("binaryFile")
        .option("recursiveFileLookup", "true")
        .option("pathGlobFilter", "*.{jpg,jpeg,png,JPG,PNG}")
        .load(path)
    )
    return (
        raw
        .select(
            F.regexp_extract(F.col("path"), r"([^/]+)$", 1).alias("image_id"),
            F.regexp_extract(F.col("path"), r"/([^/]+)/[^/]+$", 1).alias("label"),
            F.col("content").alias("raw_bytes"),
        )
        .withColumn("decoded", decode_udf(F.col("raw_bytes")))
        .filter(F.col("decoded").isNotNull())
        .select(
            "image_id", "label",
            F.col("decoded.pixels").alias("pixels"),
        )
    )

train_parsed_df = parse_images(TRAIN_PATH)
test_parsed_df  = parse_images(TEST_PATH)

print(f"Images train : {train_parsed_df.count()}")
print(f"Images test  : {test_parsed_df.count()}")


# ## Arrêt Spark

# In[ ]:




