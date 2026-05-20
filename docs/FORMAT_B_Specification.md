# FORMAT B Specification

**Version:** 1.0  
**Date:** 20 mai 2026  
**Auteur:** Pauline

---

## Description

**Format B = Sortie de µS2 (Preprocessing + Inference)**

Images du Format A sont préprocessées (grayscale + normalisation) et passées au modèle ML pour prédictions.

Format B contient l'image préprocessée + résultats ML + timings.

---

## Sérialisation

- **Format:** Binary (Raw bytes) + ML metadata
- **Image encoder:** `array.tobytes()` (NumPy)
- **Compression:** Snappy (Parquet)
- **ML metadata:** JSON strings pour class_probabilities

---

## Schéma Parquet

**19 fields: Format A transformé + preprocessing + inference + timings**

| Field Name | Type | Required | Description |
|---|---|---|---|
| **image_id** | StringType | YES | Identifiant unique (ex: `tulipe_001`) |
| **pixels_binary** | BinaryType | YES | Image préprocessée en raw bytes (512×512×1 float32 = 1,048,576 bytes) |
| **resized_width** | IntegerType | YES | Largeur (512) |
| **resized_height** | IntegerType | YES | Hauteur (512) |
| **num_channels** | IntegerType | YES | Nombre de canaux (1 = grayscale) |
| **processing_timestamp** | LongType | YES | Timestamp µS1 (millisecondes) |
| **is_grayscale** | BooleanType | YES | true (confirmation grayscale) |
| **normalized_min** | FloatType | YES | Min value après normalisation (0.0) |
| **normalized_max** | FloatType | YES | Max value après normalisation (1.0) |
| **clahe_applied** | BooleanType | YES | CLAHE enhancement appliquée? (true/false) |
| **model_name** | StringType | YES | Nom du modèle (ex: `cnn_v1`) |
| **model_version** | StringType | YES | Version du modèle (ex: `1.0.0`) |
| **predicted_class** | StringType | YES | Classe prédite (`tulipe` ou `lys`) |
| **predicted_class_id** | IntegerType | YES | ID classe (0 ou 1) |
| **confidence** | FloatType | YES | Confiance (0.0-1.0) |
| **class_probabilities** | StringType | YES | JSON: `{"tulipe": 0.97, "lys": 0.03}` |
| **us2_preprocessing_duration_ms** | LongType | YES | Temps preprocessing (ms) |
| **us2_inference_duration_ms** | LongType | YES | Temps inference (ms) |
| **us2_total_duration_ms** | LongType | YES | Temps total µS2 (ms) |

---

## Stockage

- **Path:** `/data/processed/µs2_inference/`
- **Format:** Parquet (.parquet)
- **Compression:** snappy

---

## Exemple

See `FORMAT_B_Example.json` for a complete example of a tulip image in Format B.

---

## Pipeline µS2

```
Format A (RGB uint8)
  ↓ Deserialize (bytes → array)
  ↓ Convert RGB → Grayscale
  ↓ Normalize [0-255] → [0.0-1.0]
  ↓ Optional CLAHE enhancement
  ↓ Grayscale float32
  ↓ Load ML model
  ↓ Run inference
  ↓ Get predictions (class, confidence, probabilities)
  ↓ Serialize + package Format B
  ↓
/data/processed/µs2_inference/
```
