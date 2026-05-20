# FORMAT A Specification

**Version:** 1.0  
**Date:** 20 mai 2026  
**Auteur:** Pauline 

---

## Description

**Format A = Sortie de µS1 (Parsing)**

Images originales sont chargées, redimensionnées à 512×512, et converties en binaire (raw bytes).

Format A contient les pixels sérialisés + métadonnées minimales pour µS2.

---

## Sérialisation

- **Format:** Binary (Raw bytes)
- **Encoder:** `array.tobytes()` (NumPy)
- **Compression:** Snappy (Parquet)
- **Raison:** Simple, performant, standard

---

## Schéma Parquet

| Field Name | Type | Required | Description |
|---|---|---|---|
| **image_id** | StringType | YES | Identifiant unique (ex: `tulipe_001`) |
| **pixels_binary** | BinaryType | YES | Image en raw bytes (512×512×3 uint8 = 786,432 bytes) |
| **resized_width** | IntegerType | YES | Largeur après resize (512) |
| **resized_height** | IntegerType | YES | Hauteur après resize (512) |
| **num_channels** | IntegerType | YES | Nombre de canaux (3 = RGB) |
| **processing_timestamp** | LongType | YES | Timestamp traitement (millisecondes) |

---

## Stockage

- **Path:** `/data/processed/µs1_parsed/`
- **Format:** Parquet (.parquet)
- **Compression:** snappy

---

## Exemple

See `FORMAT_A_Example.json` for a complete example of a tulip image in Format A.

--- 

## Pipeline µS1

```
JPG file
  ↓ Load (cv2.imread)
  ↓ Validate (dimensions, channels)
  ↓ Resize (512×512)
  ↓ Serialize (array.tobytes())
  ↓ Write Parquet (snappy)
  ↓
/data/processed/µs1_parsed/
```
