# FORMAT A Specification
**Version:** 1.1
**Date:** May 26, 2026  
**Author:** Pauline 

---

## Description
**Format A = Output of µS1 (Parsing)**
Original images are loaded, resized to 512×512, and converted to binary (raw bytes).
Format A contains serialized pixels + minimal metadata for µS2.

---

## Serialization
- **Format:** Binary (Raw bytes)
- **Encoder:** `array.tobytes()` (NumPy)
- **Compression:** Snappy (Parquet)
  
---

## Parquet Schema
| Field Name | Type | Required | Description |
|---|---|---|---|
| **image_id** | StringType | YES | Unique identifier (ex: `tulip_001`) |
| **pixels_binary** | BinaryType | YES | Image in raw bytes (512×512×3 uint8 = 786,432 bytes) |
| **resized_width** | IntegerType | YES | Width after resizing (512) |
| **resized_height** | IntegerType | YES | Height after resizing (512) |
| **num_channels** | IntegerType | YES | Number of channels (3 = RGB) |
| **processing_timestamp** | LongType | YES | Processing timestamp (milliseconds) |

---

## Storage
- **Path:** `/data/processed/µs1_parsed/`
- **Format:** Parquet (.parquet)
- **Compression:** snappy
  
---

## Example
See `FORMAT_A_Example.json` for a complete example of a tulip image in Format A.

--- 

## µS1 Pipeline
```
JPG File
  ↓ Loading (cv2.imread)
  ↓ Validation (dimensions, channels)
  ↓ Resizing (512×512)
  ↓ Serialization (array.tobytes())
  ↓ Parquet Writing (snappy)
  ↓
/data/processed/µs1_parsed/
```
