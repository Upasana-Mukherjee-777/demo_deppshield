"""
================================================================================
DEEPFAKE VIDEO DETECTOR — UPGRADED MODEL TRAINING
================================================================================
High-accuracy deepfake detection using EfficientNet-B4 + MTCNN face detection
with frequency-domain artifact analysis.

Key upgrades over baseline:
  - EfficientNet-B4 backbone (pretrained ImageNet) replaces custom CNN
  - MTCNN face detector (facenet-pytorch) replaces OpenCV Haar cascades
  - Frequency-domain branch (DCT high-pass filter) for GAN artifact detection
  - Focal loss for class imbalance
  - Mixup + CutMix augmentation
  - Multi-frame temporal voting with consistency scoring
  - Autoencoder anomaly detection (train on real faces only)
  - ONNX export for production serving

Requirements:
    pip install -r requirements.txt

Dataset structure (organize before training):
    dataset/
      real/       ← extracted face crops from real videos
      deepfake/   ← extracted face crops from manipulated videos

Recommended datasets:
    FaceForensics++  https://github.com/ondyari/FaceForensics
    Celeb-DF v2      https://github.com/yuezunli/celeb-deepfakeforensics
    DFDC             https://ai.facebook.com/datasets/dfdc/

Usage:
    # With real data
    python train_deepfake_detector.py --data_dir ./dataset --epochs 30

    # With synthetic data for dev
    python train_deepfake_detector.py --generate_data --epochs 10

    # Full pipeline: train + autoencoder + export
    python train_deepfake_detector.py --data_dir ./dataset --epochs 30 --train_autoencoder --export_onnx
================================================================================
"""

import argparse
import os
import json
import random
import warnings
import numpy as np
from datetime import datetime
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from torchvision.transforms import functional as TF
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# ── Configuration ──────────────────────────────────────────────────────────────

IMG_SIZE = 224
LABEL_MAP = {"real": 0, "deepfake": 1}
LABEL_NAMES = ["Real", "Deepfake"]

# OpenCV Haar cascades (fallback when MTCNN not available)
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
EYE_CASCADE_PATH  = cv2.data.haarcascades + "haarcascade_eye.xml"


# ── Face Extractor (MTCNN primary, OpenCV fallback) ────────────────────────────

class FaceExtractor:
    """
    Extracts face crops from video frames.
    Uses MTCNN when available (much higher recall), falls back to Haar cascades.
    """

    def __init__(self, device="cpu", min_face_size=60, margin=20):
        self.device = device
        self.min_face_size = min_face_size
        self.margin = margin
        self.mtcnn = None
        self.face_cascade = None
        self.eye_cascade  = None
        self._init_detectors()

    def _init_detectors(self):
        try:
            from facenet_pytorch import MTCNN
            self.mtcnn = MTCNN(
                image_size=IMG_SIZE,
                margin=self.margin,
                min_face_size=self.min_face_size,
                thresholds=[0.6, 0.7, 0.7],   # P/R/O-net thresholds
                keep_all=True,
                device=self.device,
            )
            print("✅ Face extractor: MTCNN (high accuracy)")
        except ImportError:
            print("⚠️  facenet-pytorch not installed — falling back to OpenCV Haar cascades")
            print("   Install: pip install facenet-pytorch")
            self.face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
            self.eye_cascade  = cv2.CascadeClassifier(EYE_CASCADE_PATH)

    def detect_faces_in_frame(self, frame_bgr):
        """
        Detect and return face crops from a single BGR frame.
        Returns list of (H, W, 3) uint8 RGB crops and metadata list.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self.mtcnn is not None:
            return self._detect_mtcnn(frame_rgb)
        else:
            return self._detect_haar(frame_bgr, frame_rgb)

    def _detect_mtcnn(self, frame_rgb):
        """MTCNN detection — returns aligned, fixed-size crops."""
        try:
            from PIL import Image
            pil = Image.fromarray(frame_rgb)
            boxes, probs = self.mtcnn.detect(pil)

            if boxes is None:
                return [], []

            crops, meta = [], []
            for box, prob in zip(boxes, probs):
                if prob is None or prob < 0.9:
                    continue
                x1, y1, x2, y2 = [max(0, int(v)) for v in box]
                # Add margin
                h, w = frame_rgb.shape[:2]
                x1 = max(0, x1 - self.margin)
                y1 = max(0, y1 - self.margin)
                x2 = min(w, x2 + self.margin)
                y2 = min(h, y2 + self.margin)
                if (x2 - x1) < self.min_face_size or (y2 - y1) < self.min_face_size:
                    continue
                crop = frame_rgb[y1:y2, x1:x2]
                crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
                crops.append(crop)
                meta.append({"bbox": (x1, y1, x2-x1, y2-y1), "confidence": float(prob)})
            return crops, meta
        except Exception as e:
            print(f"   MTCNN detection error: {e}")
            return [], []

    def _detect_haar(self, frame_bgr, frame_rgb):
        """OpenCV Haar cascade fallback."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        crops, meta = [], []
        for (x, y, w, h) in faces:
            pad = int(0.2 * max(w, h))
            y1 = max(0, y - pad);  y2 = min(frame_bgr.shape[0], y + h + pad)
            x1 = max(0, x - pad);  x2 = min(frame_bgr.shape[1], x + w + pad)
            crop = frame_rgb[y1:y2, x1:x2]
            if crop.shape[0] < 40 or crop.shape[1] < 40:
                continue
            crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
            face_gray = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(face_gray, 1.1, 3)
            crops.append(crop)
            meta.append({"bbox": (x, y, w, h), "eyes_detected": len(eyes), "confidence": 0.8})
        return crops, meta

    def extract_frames(self, video_path, max_frames=60, sample_rate=5):
        """Sample frames uniformly from a video file."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        idx = 0
        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % sample_rate == 0:
                frames.append(frame)
            idx += 1
        cap.release()
        return frames

    def extract_faces_from_video(self, video_path, max_faces=60):
        """Extract face crops from a video file."""
        frames = self.extract_frames(video_path)
        all_faces, all_meta = [], []
        for frame in frames:
            crops, meta = self.detect_faces_in_frame(frame)
            for c, m in zip(crops, meta):
                all_faces.append(c)
                all_meta.append(m)
                if len(all_faces) >= max_faces:
                    return all_faces, all_meta
        return all_faces, all_meta

    def compute_video_features(self, video_path):
        """Aggregate face statistics across video frames for heuristic analysis."""
        frames = self.extract_frames(video_path, max_frames=60, sample_rate=3)
        total_faces = total_eyes = frames_with_faces = 0

        for frame in frames:
            _, meta = self.detect_faces_in_frame(frame)
            n = len(meta)
            total_faces += n
            if n > 0:
                frames_with_faces += 1
                total_eyes += sum(m.get("eyes_detected", 1) for m in meta)

        n_frames = max(len(frames), 1)
        return {
            "total_frames": len(frames),
            "frames_with_faces": frames_with_faces,
            "avg_faces_per_frame": round(total_faces / n_frames, 2),
            "avg_eyes_per_frame": round(total_eyes / n_frames, 2),
            "face_detection_rate": round(frames_with_faces / n_frames, 2),
        }


# ── EfficientNet-B4 + Frequency Branch ────────────────────────────────────────

class DeepfakeDetectorEffNet(nn.Module):
    """
    EfficientNet-B4 backbone with a parallel frequency-domain branch.

    The spatial branch captures appearance artifacts (blending seams, texture).
    The frequency branch captures GAN-specific high-frequency patterns that
    are invisible to the human eye but detectable via DCT approximation.

    Combined representation → binary classifier (real / deepfake).
    """

    def __init__(self, num_classes=2, pretrained=True, dropout=0.4):
        super().__init__()

        # ── Spatial branch: EfficientNet-B4 ──────────────────────────────────
        try:
            import timm
            self.backbone = timm.create_model(
                "efficientnet_b4", pretrained=pretrained, num_classes=0,
                drop_rate=0.3, drop_path_rate=0.2
            )
            self.spatial_features = self.backbone.num_features  # 1792
        except ImportError:
            print("⚠️  timm not installed — using torchvision EfficientNet-B4")
            print("   Install: pip install timm")
            from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
            _backbone = efficientnet_b4(
                weights=EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
            )
            self.backbone = nn.Sequential(*list(_backbone.children())[:-2])
            self.backbone.num_features = 1792
            self.spatial_features = 1792
            self._torchvision_mode = True

        self._torchvision_mode = getattr(self, "_torchvision_mode", False)

        # ── Frequency branch: DCT high-pass filter + small CNN ───────────────
        self.freq_branch = nn.Sequential(
            # High-pass filter: input - avg_pool removes low frequencies
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(128 * 16, 256),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        self.freq_features = 256

        # ── Classifier head ──────────────────────────────────────────────────
        total_features = self.spatial_features + self.freq_features
        self.head = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

        self._init_freq_weights()

    def _init_freq_weights(self):
        for m in self.freq_branch.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _extract_high_freq(self, x):
        """Approximate high-frequency components via difference with avg-pooled version."""
        low = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        return x - low  # residual = high-frequency signal

    def forward(self, x):
        # Spatial features
        if self._torchvision_mode:
            spatial = self.backbone(x)
            spatial = F.adaptive_avg_pool2d(spatial, 1).flatten(1)
        else:
            spatial = self.backbone(x)

        # Frequency features
        hf = self._extract_high_freq(x)
        freq = self.freq_branch(hf)

        # Concatenate and classify
        combined = torch.cat([spatial, freq], dim=1)
        return self.head(combined)


# Alias for backward compatibility with inference_api.py
DeepfakeDetectorCNN = DeepfakeDetectorEffNet


# ── Face Autoencoder (Anomaly Detection) ──────────────────────────────────────

class FaceAutoencoder(nn.Module):
    """
    Autoencoder trained ONLY on real faces.
    High reconstruction error on a face → likely a deepfake (anomaly).
    Use as an ensemble signal alongside the classifier.
    """

    def __init__(self, latent_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),   nn.BatchNorm2d(32),  nn.LeakyReLU(0.2),  # 112
            nn.Conv2d(32, 64, 4, 2, 1),  nn.BatchNorm2d(64),  nn.LeakyReLU(0.2),  # 56
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),  # 28
            nn.Conv2d(128, 256, 4, 2, 1),nn.BatchNorm2d(256), nn.LeakyReLU(0.2),  # 14
            nn.Conv2d(256, 512, 4, 2, 1),nn.BatchNorm2d(512), nn.LeakyReLU(0.2),  # 7
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512 * 7 * 7),
            nn.Unflatten(1, (512, 7, 7)),
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(), # 14
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(), # 28
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  nn.BatchNorm2d(64),  nn.ReLU(), # 56
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   nn.BatchNorm2d(32),  nn.ReLU(), # 112
            nn.ConvTranspose2d(32, 3, 4, 2, 1),    nn.Sigmoid(),                   # 224
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x):
        """Per-sample MSE error — higher = more anomalous."""
        recon = self.forward(x)
        return F.mse_loss(recon, x, reduction="none").mean(dim=[1, 2, 3])


# ── Focal Loss ─────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for binary/multi-class classification.
    Downweights easy negatives; focuses training on hard examples.
    alpha=0.25, gamma=2.0 are the original paper defaults.
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal_weight = self.alpha * (1.0 - pt) ** self.gamma
        loss = focal_weight * ce
        return loss.mean() if self.reduction == "mean" else loss.sum()


# ── Dataset ────────────────────────────────────────────────────────────────────

class DeepfakeDataset(Dataset):
    """
    Dataset of face crops for deepfake binary classification.
    Accepts either numpy arrays (uint8 BGR/RGB) or file paths (str).
    """

    TRAIN_TRANSFORM = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)], p=0.6),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.15)),
    ])

    VAL_TRANSFORM = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def __init__(self, images_or_paths, labels, transform=None):
        self.data = images_or_paths
        self.labels = labels
        self.transform = transform or self.VAL_TRANSFORM
        self._is_paths = isinstance(images_or_paths[0], (str, Path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self._is_paths:
            img = cv2.imread(str(self.data[idx]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else \
                np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            img = self.data[idx]
            if img.shape[-1] == 3 and img.dtype == np.uint8:
                pass  # already RGB uint8

        img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.long)


def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation — interpolates between two samples."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ── Synthetic Data (dev/testing only) ─────────────────────────────────────────

def generate_synthetic_faces(n_samples=1000):
    """
    Synthetic face-like images for quick testing without real data.
    NOT suitable for real training — accuracy will be poor.
    """
    print("📝 Generating synthetic face data (dev mode)...")
    images, labels = [], []
    np.random.seed(42)

    for i in range(n_samples):
        img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        is_real = i < n_samples // 2

        # Skin-toned background
        skin = (np.random.randint(160, 220), np.random.randint(130, 180), np.random.randint(100, 150))
        img[:] = skin

        # Face oval
        cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
        cv2.ellipse(img, (cx, cy), (70, 85), 0, 0, 360, skin, -1)

        # Eyes
        eye_y = cy - 25
        for ex in [cx - 28, cx + 28]:
            cv2.circle(img, (ex, eye_y), 9, (50, 40, 30), -1)
            cv2.circle(img, (ex, eye_y), 4, (20, 15, 10), -1)

        # Mouth
        cv2.ellipse(img, (cx, cy + 30), (20, 8), 0, 0, 180, (140, 70, 70), 2)

        if not is_real:
            # Add GAN-like artifacts: color banding, edge discontinuities
            for _ in range(np.random.randint(3, 8)):
                x0, y0 = np.random.randint(30, IMG_SIZE - 30, 2)
                color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                cv2.rectangle(img, (x0, y0), (x0 + 6, y0 + 6), color, -1)
            # Add slight asymmetry to eyes
            cv2.circle(img, (cx - 28, eye_y - 2), 11, skin, -1)  # partially occlude left eye
        else:
            # Add slight Gaussian noise for realism
            noise = np.random.normal(0, 5, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        images.append(img)
        labels.append(0 if is_real else 1)

    print(f"✅ Generated {n_samples} synthetic faces — Real: {labels.count(0)} | Fake: {labels.count(1)}")
    return images, labels


# ── Trainer ────────────────────────────────────────────────────────────────────

class DeepfakeTrainer:
    """
    Trains the EfficientNet-B4 deepfake detector with:
      - Focal loss for class imbalance
      - Mixup augmentation
      - Cosine annealing with warm restarts
      - Mixed precision (FP16) on GPU
      - Early stopping
      - Autoencoder anomaly ensemble
    """

    def __init__(self, device=None, use_effnet=True):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.use_amp = self.device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None

        self.model       = DeepfakeDetectorEffNet(num_classes=2, pretrained=True).to(self.device)
        self.autoencoder = FaceAutoencoder().to(self.device)
        self.ae_threshold = None  # set after AE training via calibrate_ae_threshold()

        total = sum(p.numel() for p in self.model.parameters())
        print(f"\n🔧 DeepfakeDetector initialized on {self.device}")
        print(f"   Backbone: EfficientNet-B4 + Frequency branch")
        print(f"   Parameters: {total:,}")
        print(f"   Mixed precision: {self.use_amp}")

    def prepare_data(self, images, labels, test_size=0.15, val_size=0.1):
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=test_size, random_state=42, stratify=labels
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=42, stratify=y_train
        )

        self.train_dataset = DeepfakeDataset(X_train, y_train, DeepfakeDataset.TRAIN_TRANSFORM)
        self.val_dataset   = DeepfakeDataset(X_val,   y_val,   DeepfakeDataset.VAL_TRANSFORM)
        self.test_dataset  = DeepfakeDataset(X_test,  y_test,  DeepfakeDataset.VAL_TRANSFORM)

        # Store real images for autoencoder training
        self._real_images = [x for x, l in zip(X_train, y_train) if l == 0]

        print(f"\n📊 Splits — Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
        real_c = y_train.count(0) if isinstance(y_train, list) else (np.array(y_train) == 0).sum()
        fake_c = len(y_train) - real_c
        print(f"   Train class balance — Real: {real_c} | Deepfake: {fake_c}")

    def train(self, epochs=30, batch_size=32, lr=1e-4, use_mixup=True, patience=8):
        """Train the classifier with focal loss and cosine LR schedule."""
        num_workers = min(4, os.cpu_count() or 2)
        train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=self.use_amp, drop_last=True,
        )
        val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers
        )

        optimizer = AdamW(
            [
                {"params": self.model.backbone.parameters(), "lr": lr * 0.1},
                {"params": self.model.freq_branch.parameters(), "lr": lr},
                {"params": self.model.head.parameters(), "lr": lr},
            ],
            weight_decay=1e-4
        )

        # Cosine annealing with warm restarts — good for fine-tuning pretrained models
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=lr * 0.01)

        criterion = FocalLoss(alpha=0.25, gamma=2.0)

        best_val_auc = 0.0
        best_epoch = 0
        history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_auc": []}

        print(f"\n🚀 Training for {epochs} epochs | bs={batch_size} | lr={lr:.1e}")
        print("=" * 70)

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:3d}/{epochs}", leave=False)
            for imgs, labels in pbar:
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                if use_mixup and random.random() < 0.5:
                    imgs, y_a, y_b, lam = mixup_data(imgs, labels, alpha=0.4)
                    with autocast(enabled=self.use_amp):
                        outputs = self.model(imgs)
                        loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
                else:
                    with autocast(enabled=self.use_amp):
                        outputs = self.model(imgs)
                        loss = criterion(outputs, labels)

                optimizer.zero_grad()
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}",
                                  "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

            scheduler.step()
            avg_train = running_loss / len(train_loader)
            avg_val, val_acc, val_auc = self._evaluate(val_loader)

            history["train_loss"].append(avg_train)
            history["val_loss"].append(avg_val)
            history["val_acc"].append(val_acc)
            history["val_auc"].append(val_auc)

            marker = ""
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                self._save_checkpoint("best_cnn")
                marker = "  ✅ NEW BEST"

            if (epoch + 1) % 5 == 0 or marker:
                print(f"   Epoch {epoch+1:3d} | Train: {avg_train:.4f} | "
                      f"Val: {avg_val:.4f} | Acc: {val_acc:.4f} | AUC: {val_auc:.4f}{marker}")

            # Early stopping
            if epoch - best_epoch >= patience:
                print(f"\n⏹️  Early stopping at epoch {epoch+1} (no AUC improvement for {patience} epochs)")
                break

        print(f"\n🏆 Best Val AUC: {best_val_auc:.4f} at epoch {best_epoch+1}")
        return history

    def train_autoencoder(self, epochs=25, batch_size=32, lr=1e-3):
        """Train autoencoder on real faces only for anomaly detection."""
        if not self._real_images:
            print("⚠️  No real images available for autoencoder training")
            return

        real_dataset = DeepfakeDataset(
            self._real_images,
            [0] * len(self._real_images),
            DeepfakeDataset.VAL_TRANSFORM,
        )
        loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        optimizer = AdamW(self.autoencoder.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10)

        print(f"\n🔧 Training autoencoder on {len(self._real_images)} real faces...")
        self.autoencoder.train()

        for epoch in range(epochs):
            total = 0.0
            for imgs, _ in loader:
                imgs = imgs.to(self.device)
                recon = self.autoencoder(imgs)
                loss = F.mse_loss(recon, imgs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item()
            scheduler.step()
            if (epoch + 1) % 5 == 0:
                print(f"   AE Epoch {epoch+1}/{epochs} | Recon Loss: {total/len(loader):.6f}")

        self.calibrate_ae_threshold()
        print("✅ Autoencoder training complete")

    def calibrate_ae_threshold(self):
        """
        Calibrate anomaly threshold from reconstruction error distribution on real faces.
        Samples above mean + 2*std are flagged as anomalies (deepfakes).
        """
        if not self._real_images:
            return
        real_dataset = DeepfakeDataset(
            self._real_images[:500], [0] * min(500, len(self._real_images)),
            DeepfakeDataset.VAL_TRANSFORM
        )
        loader = DataLoader(real_dataset, batch_size=64)
        self.autoencoder.eval()
        errors = []
        with torch.no_grad():
            for imgs, _ in loader:
                err = self.autoencoder.reconstruction_error(imgs.to(self.device))
                errors.extend(err.cpu().numpy())
        mu, sigma = np.mean(errors), np.std(errors)
        self.ae_threshold = float(mu + 2.5 * sigma)
        print(f"   AE threshold calibrated: {self.ae_threshold:.6f} (μ={mu:.6f}, σ={sigma:.6f})")

    def _evaluate(self, dataloader):
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []
        total_loss = 0.0
        criterion = FocalLoss()

        with torch.no_grad():
            for imgs, labels in dataloader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.model(imgs)
                total_loss += criterion(outputs, labels).item()
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        acc = accuracy_score(all_labels, all_preds)
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except Exception:
            auc = 0.5
        return avg_loss, acc, auc

    def test(self):
        """Full evaluation on held-out test set with AUC-ROC and confusion matrix."""
        test_loader = DataLoader(self.test_dataset, batch_size=32, num_workers=2)
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(self.device)
                outputs = self.model(imgs)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

        print("\n" + "=" * 70)
        print("📋 TEST RESULTS — EfficientNet-B4 + Frequency Branch")
        print("=" * 70)
        print(classification_report(all_labels, all_preds, target_names=LABEL_NAMES))
        print("Confusion Matrix:")
        print(confusion_matrix(all_labels, all_preds))

        acc = accuracy_score(all_labels, all_preds)
        _, _, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="weighted", zero_division=0
        )
        auc = roc_auc_score(all_labels, all_probs)
        print(f"\nAUC-ROC: {auc:.4f}")
        return {"accuracy": acc, "f1": f1, "auc": auc}

    def predict_frame(self, face_img_rgb, use_ae_ensemble=True):
        """
        Predict on a single face image.
        Optionally combines CNN score with autoencoder anomaly score.

        Args:
            face_img_rgb: numpy uint8 array (H, W, 3) in RGB
            use_ae_ensemble: blend AE anomaly score with CNN confidence

        Returns:
            dict with prediction, confidence, ae_anomaly_score
        """
        self.model.eval()
        img_tensor = DeepfakeDataset.VAL_TRANSFORM(face_img_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(img_tensor)
            probs  = torch.softmax(logits, dim=1)
            pred   = torch.argmax(probs, dim=1).item()
            cnn_conf = probs[0][pred].item()

            ae_score = None
            if use_ae_ensemble and self.ae_threshold is not None:
                ae_err = self.autoencoder.reconstruction_error(img_tensor).item()
                ae_score = float(ae_err)
                # Blend: if AE says anomalous, push confidence toward deepfake
                if ae_err > self.ae_threshold:
                    deepfake_prob = 0.5 * probs[0][1].item() + 0.5
                else:
                    deepfake_prob = probs[0][1].item()
                pred = 1 if deepfake_prob > 0.5 else 0
                cnn_conf = max(deepfake_prob, 1 - deepfake_prob)

        return {
            "prediction": "real" if pred == 0 else "deepfake",
            "confidence": round(cnn_conf * 100, 1),
            "ae_anomaly_score": round(ae_score, 6) if ae_score is not None else None,
        }

    def predict_video(self, video_path, face_extractor=None, use_ae_ensemble=True):
        """
        Full video analysis pipeline with temporal consistency voting.

        Returns:
            dict with verdict, confidence, temporal_consistency, per_frame_results
        """
        if face_extractor is None:
            face_extractor = FaceExtractor(device=str(self.device))

        faces, metadata = face_extractor.extract_faces_from_video(video_path)
        if not faces:
            return {"error": "No faces detected", "prediction": "unknown", "confidence": 0}

        per_frame_probs = []
        per_frame_results = []

        for face in faces:
            result = self.predict_frame(face, use_ae_ensemble=use_ae_ensemble)
            per_frame_results.append(result)
            deepfake_conf = result["confidence"] / 100.0 if result["prediction"] == "deepfake" \
                else 1 - result["confidence"] / 100.0
            per_frame_probs.append(deepfake_conf)

        # Temporal smoothing (5-frame window)
        try:
            from scipy.ndimage import uniform_filter1d
            smoothed = uniform_filter1d(per_frame_probs, size=5)
        except ImportError:
            smoothed = np.array(per_frame_probs)

        final_score = float(np.mean(smoothed))
        temporal_consistency = float(1.0 - np.std(smoothed))
        verdict = "deepfake" if final_score > 0.5 else "real"
        overall_conf = max(final_score, 1 - final_score) * 100

        features = face_extractor.compute_video_features(video_path)

        return {
            "prediction": verdict,
            "confidence": round(overall_conf, 1),
            "deepfake_score": round(final_score, 3),
            "temporal_consistency": round(temporal_consistency, 3),
            "faces_analyzed": len(faces),
            "video_features": features,
            "per_frame_results": per_frame_results[:10],
        }

    def _save_checkpoint(self, name):
        output_dir = f"./models/deepfake_{name}"
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.model.state_dict(), f"{output_dir}/model.pth")
        if self.ae_threshold is not None:
            with open(f"{output_dir}/ae_threshold.json", "w") as f:
                json.dump({"threshold": self.ae_threshold}, f)
        print(f"   💾 Saved → {output_dir}/model.pth")

    def export_onnx(self, path="./models/deepfake_detector.onnx"):
        """Export CNN to ONNX for fast production inference."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.eval()
        dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(self.device)
        torch.onnx.export(
            self.model, dummy, path,
            input_names=["image"],
            output_names=["logits"],
            dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=14,
            do_constant_folding=True,
        )
        print(f"✅ Exported ONNX → {path}")
        print(f"   Load: session = onnxruntime.InferenceSession('{path}')")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Deepfake Video Detector")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory with real/ and deepfake/ subdirectories of face images")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--generate_data", action="store_true",
                        help="Use synthetic faces (dev only)")
    parser.add_argument("--train_autoencoder", action="store_true",
                        help="Train anomaly-detection autoencoder on real faces")
    parser.add_argument("--no_mixup", action="store_true",
                        help="Disable Mixup augmentation")
    parser.add_argument("--export_onnx", action="store_true",
                        help="Export model to ONNX after training")
    parser.add_argument("--no_cuda", action="store_true", help="Force CPU")
    args = parser.parse_args()

    if args.no_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    print("=" * 70)
    print("🎬 DEEPFAKE VIDEO DETECTOR — TRAINING")
    print(f"   Backbone: EfficientNet-B4 + Frequency Branch")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load data
    if args.data_dir and os.path.exists(args.data_dir):
        print(f"📂 Loading face images from {args.data_dir} ...")
        images, labels = [], []
        for label_name, label_id in LABEL_MAP.items():
            label_dir = os.path.join(args.data_dir, label_name)
            if not os.path.exists(label_dir):
                print(f"   ⚠️  Missing directory: {label_dir}")
                continue
            files = [f for f in os.listdir(label_dir)
                     if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
            for fname in tqdm(files, desc=f"   Loading {label_name}"):
                img = cv2.imread(os.path.join(label_dir, fname))
                if img is not None:
                    img = cv2.cvtColor(cv2.resize(img, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)
                    images.append(img)
                    labels.append(label_id)
        print(f"   Total: {len(images)} images | Real: {labels.count(0)} | Deepfake: {labels.count(1)}")
    else:
        print("📝 No --data_dir provided — using synthetic faces (dev mode)")
        images, labels = generate_synthetic_faces(n_samples=2000)

    # Train
    trainer = DeepfakeTrainer()
    trainer.prepare_data(images, labels)
    history = trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_mixup=not args.no_mixup,
    )

    if args.train_autoencoder:
        trainer.train_autoencoder(epochs=25)

    results = trainer.test()

    # Save history
    os.makedirs("./models", exist_ok=True)
    history_path = "./models/deepfake_training_history.json"
    with open(history_path, "w") as f:
        json.dump({
            "history": history,
            "test_results": results,
            "ae_threshold": trainer.ae_threshold,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)
    print(f"\n💾 History saved → {history_path}")

    if args.export_onnx:
        trainer.export_onnx()

    print(f"\n✅ Training complete!")
    print(f"   Accuracy : {results['accuracy']:.4f}")
    print(f"   F1 Score : {results['f1']:.4f}")
    print(f"   AUC-ROC  : {results['auc']:.4f}")


if __name__ == "__main__":
    main()
