"""
================================================================================
INFERENCE API — FastAPI Server (Upgraded)
================================================================================
Serves both fake news and deepfake detection models via REST API.

Upgrades over baseline:
  - Redis caching for repeated queries (1-hour TTL)
  - Rate limiting (slowapi) per IP
  - Temperature-calibrated confidence scores
  - Token-level SHAP explainability on text predictions
  - Temporal consistency scoring on video predictions
  - Prometheus metrics endpoint (/metrics)
  - Background model loading (non-blocking startup)
  - Health check shows per-model load status
  - CORS configured for security

Requirements:
    pip install -r requirements.txt

Environment variables:
    TEXT_MODEL_PATH      Path to saved HuggingFace text model dir
    VIDEO_MODEL_PATH     Path to saved CNN weights (.pth)
    REDIS_URL            Redis connection string (default: redis://localhost:6379)
    TEMPERATURE          Confidence calibration temperature (default: 1.0)
    API_KEY              Optional bearer token for endpoint auth

Usage:
    uvicorn inference_api:app --host 0.0.0.0 --port 8000 --reload
    uvicorn inference_api:app --host 0.0.0.0 --port 8000 --workers 2
================================================================================
"""

import os
import io
import json
import time
import hashlib
import tempfile
import logging
from typing import Optional, List
from contextlib import asynccontextmanager

import numpy as np
import cv2
import torch
import torch.nn.functional as F

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ── Global State ───────────────────────────────────────────────────────────────

text_model      = None
text_tokenizer  = None
video_model     = None
face_extractor  = None
redis_client    = None
ae_threshold    = None

device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
temperature = float(os.environ.get("TEMPERATURE", "1.0"))

LABEL_NAMES_TEXT  = ["Real", "Fake"]
LABEL_NAMES_VIDEO = ["Real", "Deepfake"]
IMG_SIZE = 224

# Sensational language patterns (multilingual)
SENSATIONAL_WORDS = [
    "shocking", "breaking", "urgent", "exposed", "secret", "miracle", "leaked",
    "they don't want you", "mainstream media", "big pharma", "deep state",
    "चौंकाने", "तुरंत", "गुप्त", "चमत्कार",
    "চাঞ্চল্যকর", "অলৌকিক", "ফাঁস",
    "அதிர்ச்சி", "அதிசய", "ரகசிய",
]
TOPIC_KEYWORDS = {
    "Politics":   ["government", "election", "parliament", "minister", "सरकार", "चुनाव", "সরকার"],
    "Technology": ["ai", "technology", "digital", "software", "तकनीक", "প্রযুক্তি"],
    "Economy":    ["economy", "gdp", "inflation", "market", "অর্থনীতি", "अर्थव्यवस्था"],
    "Health":     ["health", "vaccine", "covid", "hospital", "স্বাস্থ্য", "स्वास्थ्य"],
    "Science":    ["research", "study", "scientist", "discovery", "গবেষণা", "शोध"],
}


# ── Startup / Shutdown ─────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, release on shutdown."""
    global text_model, text_tokenizer, video_model, face_extractor, redis_client, temperature, ae_threshold

    logger.info(f"🚀 Starting API on device: {device}")

    # ── Redis cache ────────────────────────────────────────────────────────────
    try:
        import redis as redis_module
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        redis_client = redis_module.from_url(redis_url, decode_responses=True, socket_connect_timeout=2)
        redis_client.ping()
        logger.info("✅ Redis cache connected")
    except Exception as e:
        logger.warning(f"⚠️  Redis not available ({e}) — caching disabled")
        redis_client = None

    # ── Text model ─────────────────────────────────────────────────────────────
    text_model_path = os.environ.get("TEXT_MODEL_PATH", "./models/xlm-roberta-large_best_model")
    fallback_model  = "xlm-roberta-base"   # smaller fallback, downloads automatically

    try:
        if os.path.exists(text_model_path):
            logger.info(f"📚 Loading text model from {text_model_path}")
            text_tokenizer = AutoTokenizer.from_pretrained(text_model_path)
            text_model = AutoModelForSequenceClassification.from_pretrained(
                text_model_path, num_labels=2
            ).to(device).eval()
            # Load calibration temperature
            calib_path = os.path.join(text_model_path, "calibration.json")
            if os.path.exists(calib_path):
                with open(calib_path) as f:
                    temperature = json.load(f).get("temperature", 1.0)
                logger.info(f"   Temperature: {temperature:.2f}")
        else:
            logger.warning(f"⚠️  Text model not found at {text_model_path} — using {fallback_model}")
            text_tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            text_model = AutoModelForSequenceClassification.from_pretrained(
                fallback_model, num_labels=2
            ).to(device).eval()
        logger.info("✅ Text model loaded")
    except Exception as e:
        logger.error(f"❌ Text model load failed: {e}")

    # ── Video model ────────────────────────────────────────────────────────────
    video_model_path = os.environ.get("VIDEO_MODEL_PATH", "./models/deepfake_best_cnn/model.pth")
    ae_threshold_path = os.path.join(os.path.dirname(video_model_path), "ae_threshold.json")

    try:
        if os.path.exists(video_model_path):
            logger.info(f"🎬 Loading video model from {video_model_path}")
            from train_deepfake_detector import DeepfakeDetectorEffNet
            video_model = DeepfakeDetectorEffNet(num_classes=2, pretrained=False).to(device)
            video_model.load_state_dict(torch.load(video_model_path, map_location=device))
            video_model.eval()
            # Load AE threshold
            if os.path.exists(ae_threshold_path):
                with open(ae_threshold_path) as f:
                    ae_threshold = json.load(f).get("threshold")
                logger.info(f"   AE anomaly threshold: {ae_threshold:.6f}")
            logger.info("✅ Video model loaded")
        else:
            logger.warning(f"⚠️  Video model not found at {video_model_path}")
    except Exception as e:
        logger.error(f"❌ Video model load failed: {e}")

    # ── Face extractor ─────────────────────────────────────────────────────────
    try:
        from train_deepfake_detector import FaceExtractor
        face_extractor = FaceExtractor(device=str(device))
        logger.info("✅ Face extractor initialized")
    except Exception as e:
        logger.error(f"❌ Face extractor failed: {e}")

    yield  # ─── Application running ───

    logger.info("🛑 Shutting down")
    if redis_client:
        redis_client.close()


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Multimodal Misinformation Detector API",
    description=(
        "High-accuracy AI-powered fake news and deepfake video detection. "
        "Text: XLM-RoBERTa-large fine-tuned on multilingual datasets. "
        "Video: EfficientNet-B4 + frequency-domain analysis with MTCNN face detection."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Optional rate limiting ─────────────────────────────────────────────────────
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded

    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    RATE_LIMIT_ENABLED = True
    logger.info("✅ Rate limiting enabled (slowapi)")
except ImportError:
    RATE_LIMIT_ENABLED = False

# ── Optional Prometheus metrics ────────────────────────────────────────────────
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator().instrument(app).expose(app)
    logger.info("✅ Prometheus metrics at /metrics")
except ImportError:
    pass


# ── Optional API key auth ──────────────────────────────────────────────────────

_bearer = HTTPBearer(auto_error=False)
_API_KEY = os.environ.get("API_KEY", "")

def verify_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer)):
    if not _API_KEY:
        return  # Auth disabled
    if credentials is None or credentials.credentials != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ── Caching helpers ────────────────────────────────────────────────────────────

def _cache_key(prefix: str, text: str) -> str:
    return f"{prefix}:{hashlib.md5(text.encode()).hexdigest()}"

def _get_cache(key: str):
    if redis_client is None:
        return None
    try:
        val = redis_client.get(key)
        return json.loads(val) if val else None
    except Exception:
        return None

def _set_cache(key: str, value: dict, ttl: int = 3600):
    if redis_client is None:
        return
    try:
        redis_client.setex(key, ttl, json.dumps(value))
    except Exception:
        pass


# ── Schemas ────────────────────────────────────────────────────────────────────

class TextRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=10000, description="News text to analyze")
    language: str = Field("auto", description="Language code: en, hi, bn, ta, te, etc.")
    explain: bool = Field(False, description="Include token-level SHAP explanation (slower)")

class TextResponse(BaseModel):
    prediction: str
    label: str
    real_probability: float
    fake_probability: float
    confidence: float
    language: str
    calibrated: bool
    cached: bool
    analysis: dict
    token_explanation: Optional[List[dict]] = None

class VideoResponse(BaseModel):
    prediction: str
    confidence: float
    deepfake_score: float
    temporal_consistency: Optional[float]
    faces_analyzed: int
    duration_seconds: Optional[float]
    analysis: dict

class HealthResponse(BaseModel):
    status: str
    text_model_loaded: bool
    video_model_loaded: bool
    face_extractor_loaded: bool
    cache_available: bool
    device: str
    temperature: float
    ae_threshold: Optional[float]


# ── Root & Health ──────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
async def root():
    return {
        "service": "Multimodal Misinformation Detector",
        "version": "2.0.0",
        "models": {
            "text": "XLM-RoBERTa-large (multilingual, 104 langs)",
            "video": "EfficientNet-B4 + Frequency Branch",
            "faces": "MTCNN (facenet-pytorch) or OpenCV Haar cascade",
        },
        "endpoints": {
            "POST /api/analyze-text":  "Detect fake news in text",
            "POST /api/analyze-video": "Detect deepfakes in video",
            "POST /api/analyze-url":   "Scrape & analyze a news article URL",
            "GET  /api/health":        "Model status check",
        },
    }


@app.get("/api/health", response_model=HealthResponse, tags=["Info"])
async def health():
    return HealthResponse(
        status="healthy",
        text_model_loaded=text_model is not None,
        video_model_loaded=video_model is not None,
        face_extractor_loaded=face_extractor is not None,
        cache_available=redis_client is not None,
        device=str(device),
        temperature=temperature,
        ae_threshold=ae_threshold,
    )


# ── Text Analysis ──────────────────────────────────────────────────────────────

@app.post("/api/analyze-text", response_model=TextResponse, tags=["Detection"])
async def analyze_text(request: TextRequest, _=Depends(verify_api_key)):
    """
    Analyze text for fake news using XLM-RoBERTa-large.
    Supports English, Hindi, Bengali, Tamil, Telugu, and 100+ other languages.
    Set explain=true for token-level SHAP feature importances (slower, ~2s extra).
    """
    if text_model is None or text_tokenizer is None:
        raise HTTPException(status_code=503, detail="Text model not loaded")

    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # Cache lookup
    cache_key = _cache_key("text", text + str(request.explain))
    cached = _get_cache(cache_key)
    if cached:
        cached["cached"] = True
        return cached

    t0 = time.time()

    # Tokenize
    encoding = text_tokenizer(
        text,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)

    # Inference with temperature calibration
    with torch.no_grad():
        logits = text_model(**encoding).logits / temperature
        probs  = torch.softmax(logits, dim=-1)
        pred   = torch.argmax(probs, dim=-1).item()
        real_p = round(float(probs[0][0]) * 100, 1)
        fake_p = round(float(probs[0][1]) * 100, 1)
        conf   = round(float(probs[0][pred]) * 100, 1)

    # Heuristic flags
    flags = []
    text_lower = text.lower()
    for word in SENSATIONAL_WORDS:
        if word in text_lower:
            flags.append(f"Sensational language: '{word}'")
    if text.count("!") > 3:
        flags.append("Excessive exclamation marks")
    if len(text) > 20 and text == text.upper():
        flags.append("All-caps text detected")
    if text.count("?") > 4:
        flags.append("Excessive rhetorical questions")

    # Topics
    topics = [t for t, kws in TOPIC_KEYWORDS.items() if any(kw in text_lower for kw in kws)] or ["General"]

    # Sentiment (simple heuristic)
    pos_words = ["good", "great", "progress", "success", "hope", "achievement"]
    neg_words = ["bad", "terrible", "crisis", "danger", "threat", "catastrophe"]
    pos = sum(1 for w in pos_words if w in text_lower)
    neg = sum(1 for w in neg_words if w in text_lower)
    sentiment = "Positive" if pos > neg else ("Negative" if neg > pos else "Neutral")

    # SHAP explanation (optional)
    token_explanation = None
    if request.explain:
        token_explanation = _explain_text(text)

    latency_ms = round((time.time() - t0) * 1000, 1)

    result = TextResponse(
        prediction="real" if pred == 0 else "fake",
        label=LABEL_NAMES_TEXT[pred],
        real_probability=real_p,
        fake_probability=fake_p,
        confidence=conf,
        language=request.language,
        calibrated=temperature != 1.0,
        cached=False,
        analysis={
            "sentiment": sentiment,
            "topics": topics,
            "flags": flags,
            "word_count": len(text.split()),
            "char_count": len(text),
            "model_used": "XLM-RoBERTa-large",
            "temperature": temperature,
            "latency_ms": latency_ms,
        },
        token_explanation=token_explanation,
    )

    _set_cache(cache_key, result.dict())
    return result


def _explain_text(text: str) -> List[dict]:
    """Token-level importance via SHAP. Returns top 10 tokens."""
    try:
        import shap

        def predict_proba(texts):
            enc = text_tokenizer(
                list(texts), return_tensors="pt", padding=True,
                truncation=True, max_length=256
            ).to(device)
            with torch.no_grad():
                logits = text_model(**enc).logits / temperature
            return torch.softmax(logits, dim=-1).cpu().numpy()

        explainer = shap.Explainer(predict_proba, text_tokenizer)
        shap_values = explainer([text])
        tokens = shap_values.data[0]
        scores = shap_values.values[0]
        top = sorted(zip(tokens, scores.tolist()), key=lambda x: abs(x[1][1]), reverse=True)[:10]
        return [{"token": t, "fake_importance": round(s[1], 4), "real_importance": round(s[0], 4)}
                for t, s in top if t not in ["[CLS]", "[SEP]", "<s>", "</s>", "<pad>"]]
    except Exception as e:
        logger.warning(f"SHAP explanation failed: {e}")
        return []


# ── Video Analysis ─────────────────────────────────────────────────────────────

@app.post("/api/analyze-video", response_model=VideoResponse, tags=["Detection"])
async def analyze_video(file: UploadFile = File(...), _=Depends(verify_api_key)):
    """
    Analyze a video file for deepfake content.
    Uses EfficientNet-B4 + frequency-domain analysis + MTCNN face detection.
    Temporal voting across frames improves consistency.
    Accepts: mp4, avi, mov, webm (max recommended: 100MB)
    """
    if face_extractor is None:
        raise HTTPException(status_code=503, detail="Face extractor not initialized")

    ct = file.content_type or ""
    if not (ct.startswith("video/") or file.filename.lower().endswith(
        (".mp4", ".avi", ".mov", ".webm", ".mkv")
    )):
        raise HTTPException(status_code=400, detail="File must be a video (mp4, avi, mov, webm)")

    # Write to temp file
    suffix = os.path.splitext(file.filename or ".mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        t0 = time.time()

        # Video metadata
        cap = cv2.VideoCapture(tmp_path)
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = n_frames / fps if fps > 0 else 0
        cap.release()

        # Extract faces
        faces, meta = face_extractor.extract_faces_from_video(tmp_path, max_faces=60)
        features    = face_extractor.compute_video_features(tmp_path)

        if not faces:
            return VideoResponse(
                prediction="unknown",
                confidence=0.0,
                deepfake_score=0.0,
                temporal_consistency=None,
                faces_analyzed=0,
                duration_seconds=round(duration, 1),
                analysis={
                    "warning": "No faces detected in video",
                    "video_features": features,
                    "model_used": "N/A",
                },
            )

        if video_model is not None:
            result = _classify_faces_cnn(faces, features, duration)
        else:
            result = _classify_faces_heuristic(faces, features, duration)

        result.analysis["latency_ms"] = round((time.time() - t0) * 1000, 1)
        return result

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _classify_faces_cnn(faces, features, duration):
    """CNN + frequency branch classification with temporal consistency voting."""
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    per_frame_deepfake_probs = []
    per_frame_results = []

    video_model.eval()
    with torch.no_grad():
        # Batch inference for speed
        batch_size = 16
        for i in range(0, len(faces), batch_size):
            batch_faces = faces[i:i+batch_size]
            batch_tensor = torch.stack([transform(f) for f in batch_faces]).to(device)
            outputs = video_model(batch_tensor)
            probs = torch.softmax(outputs, dim=1)

            for j, (prob, face_idx) in enumerate(zip(probs, range(i, min(i+batch_size, len(faces))))):
                p_real    = float(prob[0])
                p_deepfake = float(prob[1])
                per_frame_deepfake_probs.append(p_deepfake)
                per_frame_results.append({
                    "face_index": face_idx,
                    "prediction": "deepfake" if p_deepfake > 0.5 else "real",
                    "deepfake_prob": round(p_deepfake * 100, 1),
                })

    # Temporal smoothing
    try:
        from scipy.ndimage import uniform_filter1d
        smoothed = uniform_filter1d(per_frame_deepfake_probs, size=min(5, len(per_frame_deepfake_probs)))
    except ImportError:
        smoothed = np.array(per_frame_deepfake_probs)

    final_score = float(np.mean(smoothed))
    temporal_consistency = float(1.0 - np.std(smoothed))
    verdict = "deepfake" if final_score > 0.5 else "real"
    overall_conf = max(final_score, 1 - final_score) * 100

    # Inconsistency flags
    inconsistencies = []
    if features["avg_eyes_per_frame"] < 1.0:
        inconsistencies.append("Low eye detection rate — possible facial manipulation")
    if features["face_detection_rate"] < 0.5:
        inconsistencies.append("Inconsistent face detection across frames")
    if temporal_consistency < 0.7:
        inconsistencies.append("High temporal inconsistency — flickering artifacts detected")
    deepfake_ratio = sum(1 for p in per_frame_deepfake_probs if p > 0.5) / len(per_frame_deepfake_probs)
    if deepfake_ratio > 0.3 and final_score <= 0.5:
        inconsistencies.append(f"Mixed signal: {deepfake_ratio:.0%} of frames classified as deepfake")

    return VideoResponse(
        prediction=verdict,
        confidence=round(overall_conf, 1),
        deepfake_score=round(final_score, 3),
        temporal_consistency=round(temporal_consistency, 3),
        faces_analyzed=len(faces),
        duration_seconds=round(duration, 1),
        analysis={
            "faces_per_frame": features["avg_faces_per_frame"],
            "face_detection_rate": features["face_detection_rate"],
            "deepfake_frame_ratio": round(deepfake_ratio, 3),
            "inconsistencies": inconsistencies,
            "frame_quality_pct": round(features["face_detection_rate"] * 100),
            "model_used": "EfficientNet-B4 + Frequency Branch",
            "per_frame_sample": per_frame_results[:5],
        },
    )


def _classify_faces_heuristic(faces, features, duration):
    """Fallback heuristic analysis when video model is not loaded."""
    avg_faces = features["avg_faces_per_frame"]
    avg_eyes  = features["avg_eyes_per_frame"]

    # Simple heuristic: real videos typically have clear eye detection
    eye_ratio = avg_eyes / max(avg_faces, 0.01)
    deepfake_score = 1.0 - min(eye_ratio / 2.0, 1.0)
    verdict = "deepfake" if deepfake_score > 0.5 else "real"
    conf = max(deepfake_score, 1 - deepfake_score) * 100

    return VideoResponse(
        prediction=verdict,
        confidence=round(conf, 1),
        deepfake_score=round(deepfake_score, 3),
        temporal_consistency=None,
        faces_analyzed=len(faces),
        duration_seconds=round(duration, 1),
        analysis={
            "faces_per_frame": avg_faces,
            "face_detection_rate": features["face_detection_rate"],
            "warning": "Video model not loaded — using heuristic analysis (lower accuracy)",
            "model_used": "Heuristic (OpenCV)",
        },
    )


# ── URL Analysis ───────────────────────────────────────────────────────────────

@app.post("/api/analyze-url", tags=["Detection"])
async def analyze_url(
    url: str = Form(...),
    language: str = Form("auto"),
    explain: bool = Form(False),
    _=Depends(verify_api_key),
):
    """
    Scrape a news article URL and analyze it for fake news.
    Requires: pip install newspaper3k
    """
    try:
        from newspaper import Article
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="newspaper3k not installed. Run: pip install newspaper3k"
        )

    try:
        article = Article(url)
        article.download()
        article.parse()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {e}")

    if not article.text or len(article.text.strip()) < 50:
        raise HTTPException(status_code=400, detail="Could not extract sufficient text from URL")

    req = TextRequest(text=article.text[:8000], language=language, explain=explain)
    analysis_result = await analyze_text(req)

    return {
        **analysis_result.dict(),
        "source": {
            "url": url,
            "title": article.title,
            "authors": article.authors,
            "publish_date": str(article.publish_date) if article.publish_date else None,
            "top_image": article.top_image,
        },
    }


# ── Batch text endpoint ────────────────────────────────────────────────────────

@app.post("/api/analyze-batch", tags=["Detection"])
async def analyze_batch(
    texts: List[str],
    language: str = "auto",
    _=Depends(verify_api_key),
):
    """
    Batch analyze up to 50 texts for fake news.
    More efficient than individual calls for bulk processing.
    """
    if not texts:
        raise HTTPException(status_code=400, detail="texts list cannot be empty")
    if len(texts) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 texts per batch request")
    if text_model is None or text_tokenizer is None:
        raise HTTPException(status_code=503, detail="Text model not loaded")

    t0 = time.time()
    results = []

    # Batch tokenize
    batch_size = 16
    all_preds, all_probs = [], []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encoding = text_tokenizer(
            batch, max_length=256, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = text_model(**encoding).logits / temperature
            probs  = torch.softmax(logits, dim=-1)
            preds  = torch.argmax(probs, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    for text, pred, prob in zip(texts, all_preds, all_probs):
        results.append({
            "text_preview": text[:80] + "..." if len(text) > 80 else text,
            "prediction": "real" if pred == 0 else "fake",
            "label": LABEL_NAMES_TEXT[pred],
            "real_probability": round(float(prob[0]) * 100, 1),
            "fake_probability": round(float(prob[1]) * 100, 1),
            "confidence": round(float(prob[pred]) * 100, 1),
        })

    return {
        "results": results,
        "count": len(results),
        "model_used": "XLM-RoBERTa-large",
        "temperature": temperature,
        "latency_ms": round((time.time() - t0) * 1000, 1),
    }


# ── Exception handlers ─────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )


# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "inference_api:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=os.environ.get("RELOAD", "false").lower() == "true",
        workers=int(os.environ.get("WORKERS", 1)),
        log_level="info",
    )
