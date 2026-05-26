from PIL import Image
import numpy as np
from pathlib import Path
from pyspark.sql import SparkSession

import os
import sys
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# --- Config ---
# Chemin absolu basé sur l'emplacement du script
BASE_DIR = Path(__file__).resolve().parent.parent  # remonte à big-data-spark/
INPUT_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output_parsed"
TARGET_SIZE = (32, 32)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Convertir en string absolu pour éviter les problèmes de chemin dans Spark
OUTPUT_DIR_STR = str(OUTPUT_DIR)

def load_and_resize(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize(TARGET_SIZE)
    return np.array(img)

def process_image(image_info: tuple):
    image_path, split, label, output_dir = image_info
    try:
        name_without_ext = Path(image_path).stem

        output_subdir = Path(output_dir) / split / label
        output_subdir.mkdir(parents=True, exist_ok=True)

        img = Image.open(image_path).convert("RGB")
        img = img.resize(TARGET_SIZE)
        array = np.array(img)

        output_path = output_subdir / f"{name_without_ext}.npz"
        np.savez(str(output_path), image_array=array, label=np.array(label), split=np.array(split))

        print(f"[{split}/{label}] {Path(image_path).name} -> shape {array.shape}")

    except Exception as e:
        print(f"Erreur sur {image_path} : {e}")
# --- Collecte des chemins ---
image_infos = []

for split in ["Train", "Test"]:
    for label in ["tulipes", "lys"]:
        folder = INPUT_DIR / split / label
        for filename in folder.iterdir():
            if filename.suffix.lower() in (".jpg", ".jpeg", ".png"):
                # On passe output_dir en absolu dans le tuple
                image_infos.append((str(filename), split, label, OUTPUT_DIR_STR))

print(f"{len(image_infos)} images trouvées")

# --- Spark ---
# Create Spark session with minimal config
spark = SparkSession.builder \
    .appName("µS1-ImageParsing") \
    .master("local[1]") \
    .config("spark.python.worker.faulthandler.enabled", "true") \
    .getOrCreate()

#Verify Spark session was created
print("✓ Spark session created successfully")
print(f"  Spark version: {spark.version}")
print(f"  Spark app name: {spark.conf.get('spark.app.name')}")

rdd = spark.sparkContext.parallelize(image_infos)
rdd.foreach(process_image)

spark.stop()