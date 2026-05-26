# FORMAT B Specification
**Version:** 1.1
**Date:** May 26, 2026  
**Author:** Pauline
---
## Description
**Format B = Output of µS2 (Preprocessing + Inference)**
Images from Format A are preprocessed (grayscale + normalization) and passed to the ML model for predictions.
Format B contains the preprocessed image + ML results + timings.
---
## Serialization
- **Format:** Binary (Raw bytes) + ML metadata
- **Image encoder:** `array.tobytes()` (NumPy)
- **Compression:** Snappy (Parquet)
- **ML metadata:** JSON strings for class_probabilities
---
## Parquet Schema
**18 fields: Format A transformed + preprocessing + inference + timings**
| Field Name | Type | Required | Description |
|---|---|---|---|
| **image_id** | StringType | YES | Unique identifier (ex: `tulip_001`) |
| **pixels_binary** | BinaryType | YES | Preprocessed image in raw bytes (512×512×1 float32 = 1,048,576 bytes) |
| **resized_width** | IntegerType | YES | Width (512) |
| **resized_height** | IntegerType | YES | Height (512) |
| **num_channels** | IntegerType | YES | Number of channels (1 = grayscale) |
| **processing_timestamp** | LongType | YES | µS1 timestamp (milliseconds) |
| **is_grayscale** | BooleanType | YES | true (grayscale applied) |
| **normalized_min** | FloatType | YES | Min value after normalization (0.0) |
| **normalized_max** | FloatType | YES | Max value after normalization (1.0) |
| **model_name** | StringType | YES | Model name (ex: `cnn_v1`) |
| **model_version** | StringType | YES | Model version (ex: `1.0.0`) |
| **predicted_class** | StringType | YES | Predicted class (`tulip` or `lily`) |
| **predicted_class_id** | IntegerType | YES | Class ID (0 or 1) |
| **confidence** | FloatType | YES | Confidence (0.0-1.0) |
| **class_probabilities** | StringType | YES | JSON: `{"tulip": 0.97, "lily": 0.03}` |
| **us2_preprocessing_duration_ms** | LongType | YES | Preprocessing time (ms) |
| **us2_inference_duration_ms** | LongType | YES | Inference time (ms) |
| **us2_total_duration_ms** | LongType | YES | Total µS2 time (ms) |
---
## Storage
- **Path:** `/data/processed/µs2_inference/`
- **Format:** Parquet (.parquet)
- **Compression:** snappy
---
## Example
See `FORMAT_B_Example.json` for a complete example of a tulip image in Format B.
---
## µS2 Pipeline
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
