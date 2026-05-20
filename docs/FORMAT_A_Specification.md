# FORMAT A Spécification
**Version:** 1.0  
**Date:** 20 mai 2026  
**Auteur:** Pauline 
---
## Description
**Format A = Sortie de µS1 (Analyse syntaxique)**
Les images originales sont chargées, redimensionnées à 512×512, et converties en binaire (raw bytes).
Le Format A contient les pixels sérialisés + métadonnées minimales pour µS2.
---
## Sérialisation
- **Format:** Binaire (Raw bytes)
- **Encodeur:** `array.tobytes()` (NumPy)
- **Compression:** Snappy (Parquet)
---
## Schéma Parquet
| Nom du champ | Type | Requis | Description |
|---|---|---|---|
| **image_id** | StringType | OUI | Identifiant unique (ex: `tulipe_001`) |
| **pixels_binary** | BinaryType | OUI | Image en raw bytes (512×512×3 uint8 = 786 432 bytes) |
| **resized_width** | IntegerType | OUI | Largeur après redimensionnement (512) |
| **resized_height** | IntegerType | OUI | Hauteur après redimensionnement (512) |
| **num_channels** | IntegerType | OUI | Nombre de canaux (3 = RGB) |
| **processing_timestamp** | LongType | OUI | Timestamp du traitement (millisecondes) |
---
## Stockage
- **Chemin:** `/data/processed/µs1_parsed/`
- **Format:** Parquet (.parquet)
- **Compression:** snappy
---
## Exemple
Voir `FORMAT_A_Example.json` pour un exemple complet d'une image de tulipe au Format A.
--- 
## Pipeline µS1
```
Fichier JPG
  ↓ Chargement (cv2.imread)
  ↓ Validation (dimensions, canaux)
  ↓ Redimensionnement (512×512)
  ↓ Sérialisation (array.tobytes())
  ↓ Écriture Parquet (snappy)
  ↓
/data/processed/µs1_parsed/
```
