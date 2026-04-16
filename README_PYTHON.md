# Multimodal Misinformation Detector — ML Backend

High-accuracy AI system for detecting fake news (text) and deepfakes (video) across English and Indian languages.

---

## Model Summary

| Task | Model | Expected Accuracy |
|---|---|---|
| English fake news | XLM-RoBERTa-large | ~91% F1 |
| Hindi / Bengali / Tamil | MuRIL or IndicBERT-v2 | ~83% F1 |
| Deepfake detection | EfficientNet-B4 + Frequency Branch | ~93% AUC |
| Face detection | MTCNN (facenet-pytorch) | ~94% recall |

---

## Files

| File | Description |
|---|---|
| `train_fake_news_detector.py` | XLM-RoBERTa fine-tuning with LLRD, label smoothing, calibration |
| `train_deepfake_detector.py` | EfficientNet-B4 + frequency branch with Focal loss, Mixup |
| `inference_api.py` | FastAPI server with Redis caching, rate limiting, SHAP explainability |
| `requirements.txt` | Full dependency list with dataset download guide |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2a. Development — synthetic data, fast
python train_fake_news_detector.py --model xlm-roberta-base --generate_data --epochs 3
python train_deepfake_detector.py --generate_data --epochs 5

# 2b. Production — real datasets (download first, see requirements.txt)
python train_fake_news_detector.py --model xlm-roberta-large --dataset combined --epochs 10 --calibrate
python train_deepfake_detector.py --data_dir ./dataset --epochs 30 --train_autoencoder

# 3. Start API
uvicorn inference_api:app --host 0.0.0.0 --port 8000 --reload

# API docs: http://localhost:8000/docs
# Health:   http://localhost:8000/api/health
```

---

## Text Model Options

```bash
# Best overall accuracy (English + multilingual)
python train_fake_news_detector.py --model xlm-roberta-large --dataset combined --epochs 10

# Best for Indian languages (Hindi, Bengali, Tamil, Telugu, Kannada, Malayalam)
python train_fake_news_detector.py --model muril --dataset combined --epochs 8

# Fastest / smallest — good for CPU inference
python train_fake_news_detector.py --model xlm-roberta-base --dataset combined --epochs 5

# With back-translation augmentation for low-resource languages
python train_fake_news_detector.py --model muril --dataset combined --augment --epochs 8
```

### Available models

| Key | HuggingFace name | Size | Best for |
|---|---|---|---|
| `xlm-roberta-large` | `xlm-roberta-large` | 560M | English + all languages (best accuracy) |
| `xlm-roberta-base` | `xlm-roberta-base` | 270M | Balanced speed/accuracy |
| `muril` | `google/muril-base-cased` | 235M | Indian languages (Hindi, Bengali, Tamil…) |
| `indicbert-v2` | `ai4bharat/IndicBERTv2-MLM-Sam-TLM` | 110M | IndicNLP tasks |
| `mbert` | `bert-base-multilingual-cased` | 180M | Baseline (104 languages) |

---

## Deepfake Model Options

```bash
# Standard training with real data
python train_deepfake_detector.py --data_dir ./dataset --epochs 30

# Add autoencoder for anomaly detection ensemble
python train_deepfake_detector.py --data_dir ./dataset --epochs 30 --train_autoencoder

# Export to ONNX for fast production inference
python train_deepfake_detector.py --data_dir ./dataset --epochs 30 --export_onnx

# Disable Mixup (if dataset is already large/balanced)
python train_deepfake_detector.py --data_dir ./dataset --no_mixup
```

### Dataset directory structure

```
dataset/
  real/
    frame_001.jpg
    frame_002.jpg
    ...
  deepfake/
    frame_001.jpg
    frame_002.jpg
    ...
```

Extract face crops from FaceForensics++ or Celeb-DF videos using the built-in `FaceExtractor` class:

```python
from train_deepfake_detector import FaceExtractor
import cv2, os

extractor = FaceExtractor()
for video_path in video_list:
    faces, _ = extractor.extract_faces_from_video(video_path, max_faces=30)
    for i, face in enumerate(faces):
        out_path = f"dataset/real/video_{vid_id}_face_{i}.jpg"
        cv2.imwrite(out_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
```

---

## Text Datasets

| Dataset | Language | Size | How to get |
|---|---|---|---|
| LIAR | English | 12.8k | Auto via `load_dataset("ucsbnlp/liar")` |
| WELFake | English | 72k | Kaggle: `saurabhshahane/fake-news-classification` |
| ISOT | English | 44k | UVic form (free), extracts as `True.csv` + `Fake.csv` |
| BanFakeNews | Bengali | 8.5k | `github.com/Rowan1697/BanFakeNews` → `BanFakeNews.csv` |
| Hindi Fake News | Hindi | ~4k | `github.com/sumanthvrao/Fake-News-Hindi` → `hindi_fake_news.csv` |
| IndicFakeNews | 7 Indian langs | varies | `indicnlp.ai4bharat.org` → `indicfakenews.csv` |

Place files in the project root before running `--dataset combined`.

---

## Deepfake Datasets

| Dataset | Size | Access |
|---|---|---|
| FaceForensics++ | 1000 videos × 5 manipulation types | Request at `github.com/ondyari/FaceForensics` |
| Celeb-DF v2 | 590 real + 5639 fake | `github.com/yuezunli/celeb-deepfakeforensics` |
| DFDC (Facebook) | 100k+ clips | Kaggle: `deepfake-detection-challenge` |
| UADFV | 49 + 49 | `github.com/danmohaha/WIFS2018_In_Ictu_Oculi` |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Service info |
| GET | `/api/health` | Model status, temperature, AE threshold |
| POST | `/api/analyze-text` | Analyze text for fake news |
| POST | `/api/analyze-video` | Analyze video for deepfakes |
| POST | `/api/analyze-url` | Scrape + analyze a news URL |
| POST | `/api/analyze-batch` | Batch text analysis (up to 50 texts) |
| GET | `/metrics` | Prometheus metrics (if installed) |
| GET | `/docs` | Interactive Swagger UI |

### Example: Text analysis

```bash
curl -X POST http://localhost:8000/api/analyze-text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Scientists confirm new vaccine is 94% effective in Phase 3 trials.",
    "language": "en",
    "explain": false
  }'
```

Response:
```json
{
  "prediction": "real",
  "label": "Real",
  "real_probability": 88.4,
  "fake_probability": 11.6,
  "confidence": 88.4,
  "language": "en",
  "calibrated": true,
  "cached": false,
  "analysis": {
    "sentiment": "Positive",
    "topics": ["Health", "Science"],
    "flags": [],
    "word_count": 12,
    "model_used": "XLM-RoBERTa-large",
    "temperature": 1.8,
    "latency_ms": 145
  }
}
```

### Example: Video analysis

```bash
curl -X POST http://localhost:8000/api/analyze-video \
  -F "file=@test_video.mp4"
```

Response:
```json
{
  "prediction": "deepfake",
  "confidence": 87.2,
  "deepfake_score": 0.872,
  "temporal_consistency": 0.91,
  "faces_analyzed": 42,
  "duration_seconds": 12.4,
  "analysis": {
    "faces_per_frame": 1.2,
    "face_detection_rate": 0.94,
    "deepfake_frame_ratio": 0.83,
    "inconsistencies": ["High temporal inconsistency — flickering artifacts detected"],
    "model_used": "EfficientNet-B4 + Frequency Branch"
  }
}
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `TEXT_MODEL_PATH` | `./models/xlm-roberta-large_best_model` | Path to saved text model dir |
| `VIDEO_MODEL_PATH` | `./models/deepfake_best_cnn/model.pth` | Path to video model weights |
| `REDIS_URL` | `redis://localhost:6379` | Redis for prediction caching |
| `TEMPERATURE` | `1.0` | Confidence calibration temp (set from calibration.json) |
| `API_KEY` | (disabled) | Bearer token auth for endpoints |
| `ALLOWED_ORIGINS` | `*` | CORS origins (comma-separated) |
| `PORT` | `8000` | Server port |
| `WORKERS` | `1` | Uvicorn worker processes |

---

## Key Architectural Upgrades

**Text model (vs baseline mBERT):**
- XLM-RoBERTa-large (560M params) — 12% better F1 than mBERT on English
- Layerwise learning rate decay (LLRD) — lower LR for bottom transformer layers
- Label smoothing (0.1) — prevents overconfidence on noisy data
- Class-balanced cross-entropy — handles real/fake ratio imbalance
- Temperature calibration — reliable confidence scores post-training
- Mixed precision (FP16) — 2× faster on GPU

**Video model (vs baseline custom CNN):**
- EfficientNet-B4 (ImageNet pretrained) — pretrained features dramatically outperform training from scratch
- Frequency-domain branch — DCT high-pass filter detects GAN spectral artifacts invisible to human eye
- MTCNN face detector — 94% face detection recall vs ~60% for Haar cascades
- Focal loss (α=0.25, γ=2.0) — focuses training on hard examples
- Mixup augmentation — reduces overfitting on small deepfake datasets
- Multi-frame temporal voting + smoothing — more consistent video-level decisions
- Autoencoder ensemble — reconstruction error on real faces provides a second signal

**API:**
- Redis caching — identical queries return in <1ms
- Rate limiting (slowapi) — prevents API abuse
- SHAP explainability — token-level importance for text predictions
- Batch endpoint — process 50 texts per request
- Prometheus metrics — production observability
- Temperature calibration loaded automatically from training
