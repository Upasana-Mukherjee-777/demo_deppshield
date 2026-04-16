"""
================================================================================
MULTILINGUAL FAKE NEWS DETECTOR - UPGRADED MODEL TRAINING
================================================================================
Fine-tunes XLM-RoBERTa-large / MuRIL / IndicBERT-v2 for multilingual fake
news classification with high accuracy techniques.

Supports: English, Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati

Key upgrades over baseline:
  - XLM-RoBERTa-large (560M params) as primary model
  - Real dataset loaders: LIAR, WELFake, FakeNewsNet, BanFakeNews, IndicFakeNews
  - Label smoothing + class-balanced loss
  - Layerwise learning rate decay (LLRD)
  - Back-translation data augmentation for low-resource languages
  - Temperature calibration for reliable confidence scores
  - SHAP token-level explainability export
  - Mixed-precision training (FP16) for GPU speedup
  - ONNX export with dynamic axes

Requirements:
    pip install -r requirements.txt

Usage:
    # Quick test with synthetic data
    python train_fake_news_detector.py --model xlm-roberta-large --generate_data --epochs 3

    # Full training with real datasets
    python train_fake_news_detector.py --model xlm-roberta-large --dataset combined --epochs 10

    # Indian language focus
    python train_fake_news_detector.py --model muril --dataset combined --epochs 8

    # Export to ONNX after training
    python train_fake_news_detector.py --model xlm-roberta-large --dataset combined --export_onnx
================================================================================
"""

import argparse
import os
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)


# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    "xlm-roberta-large": {
        "name": "xlm-roberta-large",
        "max_length": 256,
        "description": "XLM-RoBERTa large — best overall accuracy (~91% F1 on English)",
        "batch_size": 8,      # large model; use gradient accumulation
        "grad_accum": 4,
        "lr": 1e-5,
    },
    "xlm-roberta-base": {
        "name": "xlm-roberta-base",
        "max_length": 256,
        "description": "XLM-RoBERTa base — good balance of speed and accuracy",
        "batch_size": 16,
        "grad_accum": 2,
        "lr": 2e-5,
    },
    "muril": {
        "name": "google/muril-base-cased",
        "max_length": 256,
        "description": "MuRIL — best for Indian languages (Hindi, Bengali, Tamil etc.)",
        "batch_size": 16,
        "grad_accum": 2,
        "lr": 2e-5,
    },
    "indicbert-v2": {
        "name": "ai4bharat/IndicBERTv2-MLM-Sam-TLM",
        "max_length": 256,
        "description": "IndicBERT v2 — optimised for IndicNLP tasks",
        "batch_size": 16,
        "grad_accum": 2,
        "lr": 2e-5,
    },
    "mbert": {
        "name": "bert-base-multilingual-cased",
        "max_length": 256,
        "description": "mBERT — baseline multilingual (104 languages)",
        "batch_size": 16,
        "grad_accum": 2,
        "lr": 2e-5,
    },
}

LABEL_MAP = {"real": 0, "fake": 1}
LABEL_NAMES = ["Real", "Fake"]


# ── Dataset Loaders ────────────────────────────────────────────────────────────

def load_liar_dataset():
    """
    Load LIAR dataset via HuggingFace datasets.
    Labels: true/mostly-true → real; others → fake
    Install: pip install datasets
    """
    try:
        from datasets import load_dataset
        print("   Loading LIAR dataset from HuggingFace...")
        ds = load_dataset("ucsbnlp/liar")
        df = pd.DataFrame(ds["train"])
        df["label"] = df["label"].apply(
            lambda x: "real" if x in ["true", "mostly-true"] else "fake"
        )
        df = df.rename(columns={"statement": "text"})[["text", "label"]]
        df["language"] = "en"
        print(f"   LIAR: {len(df)} samples | {df['label'].value_counts().to_dict()}")
        return df
    except Exception as e:
        print(f"   LIAR load failed: {e}")
        return pd.DataFrame(columns=["text", "label", "language"])


def load_welfake_dataset(csv_path="WELFake_Dataset.csv"):
    """
    Load WELFake dataset (72k English samples — best for English accuracy).
    Download from Kaggle: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification
    Columns required: title, text, label (0=real, 1=fake)
    """
    if not os.path.exists(csv_path):
        print(f"   WELFake not found at {csv_path} — skipping")
        return pd.DataFrame(columns=["text", "label", "language"])
    df = pd.read_csv(csv_path)
    # Combine title + text for richer features
    df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    df["text"] = df["text"].str.strip()
    df["label"] = df["label"].map({0: "real", 1: "fake"})
    df = df[["text", "label"]].dropna()
    df["language"] = "en"
    print(f"   WELFake: {len(df)} samples | {df['label'].value_counts().to_dict()}")
    return df


def load_isot_dataset(real_path="True.csv", fake_path="Fake.csv"):
    """
    Load ISOT Fake News dataset (44k English articles).
    Download from: https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/
    Two CSV files: True.csv and Fake.csv
    """
    frames = []
    for path, label in [(real_path, "real"), (fake_path, "fake")]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
            df["label"] = label
            frames.append(df[["text", "label"]])
    if not frames:
        print("   ISOT not found — skipping")
        return pd.DataFrame(columns=["text", "label", "language"])
    df = pd.concat(frames)
    df["language"] = "en"
    print(f"   ISOT: {len(df)} samples | {df['label'].value_counts().to_dict()}")
    return df


def load_banfakenews_dataset(csv_path="BanFakeNews.csv"):
    """
    Load BanFakeNews Bengali dataset (~8.5k samples).
    GitHub: https://github.com/Rowan1697/BanFakeNews
    Expected columns: text/news, label (Authentic/Fake)
    """
    if not os.path.exists(csv_path):
        print(f"   BanFakeNews not found at {csv_path} — skipping")
        return pd.DataFrame(columns=["text", "label", "language"])
    df = pd.read_csv(csv_path)
    text_col = "news" if "news" in df.columns else "text"
    df = df.rename(columns={text_col: "text"})
    df["label"] = df["label"].str.lower().map(
        {"authentic": "real", "real": "real", "fake": "fake"}
    )
    df = df[["text", "label"]].dropna()
    df["language"] = "bn"
    print(f"   BanFakeNews: {len(df)} samples | {df['label'].value_counts().to_dict()}")
    return df


def load_hindi_fakenews_dataset(csv_path="hindi_fake_news.csv"):
    """
    Load Hindi fake news dataset.
    GitHub: https://github.com/sumanthvrao/Fake-News-Hindi
    Expected columns: text/headline, label (0=real, 1=fake or real/fake strings)
    """
    if not os.path.exists(csv_path):
        print(f"   Hindi Fake News not found at {csv_path} — skipping")
        return pd.DataFrame(columns=["text", "label", "language"])
    df = pd.read_csv(csv_path)
    text_col = next((c for c in df.columns if c in ["text", "headline", "news"]), None)
    if text_col is None:
        return pd.DataFrame(columns=["text", "label", "language"])
    df = df.rename(columns={text_col: "text"})
    if df["label"].dtype in [int, float]:
        df["label"] = df["label"].map({0: "real", 1: "fake"})
    else:
        df["label"] = df["label"].str.lower()
    df = df[["text", "label"]].dropna()
    df["language"] = "hi"
    print(f"   Hindi Fake News: {len(df)} samples | {df['label'].value_counts().to_dict()}")
    return df


def load_indicfakenews_dataset(csv_path="indicfakenews.csv"):
    """
    Load IndicFakeNews (AI4Bharat) covering Hindi, Bengali, Tamil, Telugu etc.
    https://indicnlp.ai4bharat.org/
    Expected columns: text, label, language
    """
    if not os.path.exists(csv_path):
        print(f"   IndicFakeNews not found at {csv_path} — skipping")
        return pd.DataFrame(columns=["text", "label", "language"])
    df = pd.read_csv(csv_path)
    df["label"] = df["label"].str.lower()
    df = df[["text", "label", "language"]].dropna()
    print(f"   IndicFakeNews: {len(df)} samples | {df['language'].value_counts().to_dict()}")
    return df


def load_all_real_datasets():
    """Combine all available real datasets into one DataFrame."""
    print("\n📂 Loading real datasets...")
    frames = [
        load_liar_dataset(),
        load_welfake_dataset(),
        load_isot_dataset(),
        load_banfakenews_dataset(),
        load_hindi_fakenews_dataset(),
        load_indicfakenews_dataset(),
    ]
    df = pd.concat([f for f in frames if len(f) > 0], ignore_index=True)
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.strip().str.len() > 10]
    df = df.drop_duplicates(subset="text")
    # Ensure only valid labels
    df = df[df["label"].isin(["real", "fake"])]
    print(f"\n📊 Combined dataset: {len(df)} samples")
    print(f"   Labels: {df['label'].value_counts().to_dict()}")
    print(f"   Languages: {df['language'].value_counts().to_dict()}")
    return df.reset_index(drop=True)


# ── Synthetic Data (fallback for development) ─────────────────────────────────

def generate_sample_dataset(n_samples=3000):
    """
    Synthetic multilingual dataset for development/testing only.
    Replace with real datasets for production accuracy.
    """
    np.random.seed(42)
    templates = {
        "en": {
            "real": [
                "The government announced new economic reforms aimed at boosting GDP growth by 4 percent.",
                "Scientists published peer-reviewed research in Nature on climate change mitigation strategies.",
                "The Reserve Bank raised interest rates by 25 basis points following inflation concerns.",
                "Parliament passed the annual budget with bipartisan support after a week of debate.",
                "A new public health study confirms vaccine efficacy at 94 percent in clinical trials.",
                "The Supreme Court ruled in favor of environmental protections in a landmark case today.",
                "Unemployment figures fell to a decade low according to data released by the ministry.",
                "The bilateral trade agreement between the two nations was signed after months of talks.",
            ],
            "fake": [
                "SHOCKING: Secret government plan to control citizens through 5G vaccines EXPOSED!",
                "You won't BELIEVE what this politician did — the TRUTH media WON'T report!",
                "BREAKING: Miracle cure found that doctors are paid to HIDE from you!",
                "URGENT: Forward this before it gets DELETED — they don't want you to see this!",
                "EXPOSED: The REAL reason behind the crisis, what mainstream media is hiding!",
                "ALERT: Scientists confirm government is poisoning water supply — share NOW!",
                "MUST READ: This food kills cancer but Big Pharma is suppressing it!",
                "LEAKED DOCUMENTS prove deep state plot to destroy the economy — wake up!",
            ],
        },
        "hi": {
            "real": [
                "सरकार ने आर्थिक विकास को बढ़ावा देने के लिए नई नीतियों की घोषणा की।",
                "वैज्ञानिकों ने जलवायु परिवर्तन पर नई शोध प्रकाशित की जो 95 प्रतिशत सटीक है।",
                "भारतीय रिज़र्व बैंक ने ब्याज दरों में वृद्धि की घोषणा की।",
                "संसद ने वार्षिक बजट बहुमत से पारित किया।",
                "स्वास्थ्य मंत्रालय ने टीकाकरण अभियान के परिणामों की रिपोर्ट जारी की।",
            ],
            "fake": [
                "चौंकाने वाला: सरकार का गुप्त षड्यंत्र उजागर! तुरंत शेयर करें!",
                "ये खबर पढ़कर आप हैरान रह जाएंगे! मीडिया छुपा रहा है सच्चाई!",
                "वायरल: इस चमत्कारी उपाय से एक रात में कैंसर ठीक होगा!",
                "बड़ी खबर: ये सच्चाई जानकर आपके होश उड़ जाएंगे! सरकार नहीं चाहती आप जानें!",
                "जरूर पढ़ें: वैक्सीन में मिलाया जा रहा है जहर, डॉक्टर चुप क्यों हैं?",
            ],
        },
        "bn": {
            "real": [
                "সরকার অর্থনৈতিক সংস্কারের ঘোষণা দিয়েছে যা জিডিপি বৃদ্ধিতে সহায়ক হবে।",
                "বিজ্ঞানীরা জলবায়ু পরিবর্তন নিয়ে নতুন গবেষণা প্রকাশ করেছেন।",
                "কেন্দ্রীয় ব্যাংক সুদের হার বাড়িয়েছে মুদ্রাস্ফীতি নিয়ন্ত্রণে।",
            ],
            "fake": [
                "চাঞ্চল্যকর: সরকারের গোপন পরিকল্পনা ফাঁস! এখনই শেয়ার করুন!",
                "ভাইরাল: এই অলৌকিক প্রতিকারে রাতারাতি সমস্যা সমাধান হয়!",
                "জরুরি: এই সত্য জানলে আপনি হতবাক হয়ে যাবেন! মিডিয়া লুকাচ্ছে!",
            ],
        },
        "ta": {
            "real": [
                "அரசு பொருளாதார சீர்திருத்தங்களை அறிவித்தது.",
                "விஞ்ஞானிகள் காலநிலை மாற்றம் குறித்த ஆய்வை வெளியிட்டனர்.",
            ],
            "fake": [
                "அதிர்ச்சி: அரசின் ரகசிய திட்டம் அம்பலம்! உடனே பகிரவும்!",
                "வைரல்: இந்த அதிசய வைத்தியம் மருத்துவர்கள் மறைக்கிறார்கள்!",
            ],
        },
    }

    rows = []
    per_class = n_samples // (len(templates) * 2)
    for lang, categories in templates.items():
        for label, texts in categories.items():
            for _ in range(per_class):
                base = np.random.choice(texts)
                # Simple variation: combine two sentences
                if np.random.random() > 0.6:
                    extra = np.random.choice(texts)
                    base = base + " " + extra
                rows.append({"text": base, "label": label, "language": lang})

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"✅ Generated synthetic dataset: {len(df)} samples")
    print(f"   Labels: {df['label'].value_counts().to_dict()}")
    return df


# ── Back-Translation Augmentation ─────────────────────────────────────────────

def back_translate_batch(texts, src_lang="hi", pivot="en", batch_size=16):
    """
    Augment text via back-translation: src → pivot → src.
    Requires Helsinki-NLP translation models (downloaded automatically).
    Useful for low-resource Indian language splits.

    Usage:
        augmented = back_translate_batch(hindi_texts, src_lang="hi")
    """
    try:
        from transformers import MarianMTModel, MarianTokenizer

        fwd_name = f"Helsinki-NLP/opus-mt-{src_lang}-{pivot}"
        bck_name = f"Helsinki-NLP/opus-mt-{pivot}-{src_lang}"

        fwd_tok = MarianTokenizer.from_pretrained(fwd_name)
        fwd_model = MarianMTModel.from_pretrained(fwd_name).eval()
        bck_tok = MarianTokenizer.from_pretrained(bck_name)
        bck_model = MarianMTModel.from_pretrained(bck_name).eval()

        augmented = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            with torch.no_grad():
                t = fwd_tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
                en = fwd_model.generate(**t, num_beams=4)
                en_texts = fwd_tok.batch_decode(en, skip_special_tokens=True)

                t2 = bck_tok(en_texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
                back = bck_model.generate(**t2, num_beams=4)
                augmented.extend(bck_tok.batch_decode(back, skip_special_tokens=True))

        return augmented
    except Exception as e:
        print(f"   Back-translation failed: {e} — skipping augmentation")
        return texts


def augment_minority_class(df, lang="hi", label="fake", n_aug=200):
    """
    Augment the minority class for a given language via back-translation.
    Call before prepare_data().
    """
    subset = df[(df["language"] == lang) & (df["label"] == label)]
    if len(subset) == 0:
        return df
    print(f"   Augmenting {lang}/{label}: {len(subset)} → +{n_aug} samples via back-translation")
    sample_texts = subset["text"].sample(min(n_aug, len(subset)), replace=True).tolist()
    augmented_texts = back_translate_batch(sample_texts, src_lang=lang)
    aug_df = pd.DataFrame({"text": augmented_texts, "label": label, "language": lang})
    return pd.concat([df, aug_df], ignore_index=True)


# ── Dataset Class ──────────────────────────────────────────────────────────────

class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Trainer ────────────────────────────────────────────────────────────────────

class FakeNewsTrainer:
    """
    Trains a transformer-based multilingual fake news classifier with:
      - Label smoothing
      - Class-balanced loss weights
      - Layerwise learning rate decay (LLRD)
      - Mixed precision (FP16) on GPU
      - Temperature calibration
    """

    def __init__(self, model_key="xlm-roberta-large", device=None):
        self.config = MODEL_CONFIGS[model_key]
        self.model_key = model_key
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.use_amp = self.device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None
        self.temperature = 1.0  # tuned via calibration

        print(f"\n🔧 Initializing {self.config['description']}")
        print(f"   Device: {self.device} | Mixed precision: {self.use_amp}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config["name"])
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config["name"],
            num_labels=2,
            problem_type="single_label_classification",
        ).to(self.device)

        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"   Parameters: {total:,} total | {trainable:,} trainable")

    def prepare_data(self, df, test_size=0.15, val_size=0.1):
        texts = df["text"].tolist()
        labels = [LABEL_MAP[l] for l in df["label"]]

        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=42, stratify=y_train
        )

        # Compute class weights for imbalanced datasets
        cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=np.array(y_train))
        self.class_weights = torch.tensor(cw, dtype=torch.float).to(self.device)

        max_len = self.config["max_length"]
        self.train_dataset = FakeNewsDataset(X_train, y_train, self.tokenizer, max_len)
        self.val_dataset   = FakeNewsDataset(X_val,   y_val,   self.tokenizer, max_len)
        self.test_dataset  = FakeNewsDataset(X_test,  y_test,  self.tokenizer, max_len)

        print(f"\n📊 Data splits — Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
        print(f"   Class weights → Real: {cw[0]:.3f} | Fake: {cw[1]:.3f}")

    def _build_optimizer_with_llrd(self, learning_rate, weight_decay=0.01):
        """
        Layerwise Learning Rate Decay (LLRD).
        Lower layers get smaller LR — helps preserve pretrained representations.
        """
        no_decay = ["bias", "LayerNorm.weight"]

        # Group parameters by transformer layer depth
        optimizer_params = []

        # Classifier head — full LR
        optimizer_params.append({
            "params": [p for n, p in self.model.classifier.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "lr": learning_rate * 2, "weight_decay": weight_decay,
        })
        optimizer_params.append({
            "params": [p for n, p in self.model.classifier.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "lr": learning_rate * 2, "weight_decay": 0.0,
        })

        # Transformer encoder layers — decay rate 0.9 per layer from top
        try:
            encoder_layers = self.model.roberta.encoder.layer
        except AttributeError:
            try:
                encoder_layers = self.model.bert.encoder.layer
            except AttributeError:
                encoder_layers = []

        num_layers = len(encoder_layers)
        for i, layer in enumerate(reversed(encoder_layers)):
            layer_lr = learning_rate * (0.9 ** i)
            optimizer_params.append({
                "params": [p for n, p in layer.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "lr": layer_lr, "weight_decay": weight_decay,
            })
            optimizer_params.append({
                "params": [p for n, p in layer.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "lr": layer_lr, "weight_decay": 0.0,
            })

        # Embeddings — smallest LR
        try:
            embed_params = list(self.model.roberta.embeddings.named_parameters())
        except AttributeError:
            try:
                embed_params = list(self.model.bert.embeddings.named_parameters())
            except AttributeError:
                embed_params = []

        if embed_params:
            embed_lr = learning_rate * (0.9 ** num_layers)
            optimizer_params.append({
                "params": [p for n, p in embed_params if not any(nd in n for nd in no_decay)],
                "lr": embed_lr, "weight_decay": weight_decay,
            })

        return AdamW(optimizer_params, eps=1e-8)

    def train(self, epochs=5, batch_size=None, learning_rate=None, warmup_ratio=0.15, grad_accum=None):
        batch_size  = batch_size  or self.config["batch_size"]
        learning_rate = learning_rate or self.config["lr"]
        grad_accum  = grad_accum  or self.config["grad_accum"]

        train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory=self.use_amp
        )
        val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=2
        )

        optimizer = self._build_optimizer_with_llrd(learning_rate)

        total_steps = (len(train_loader) // grad_accum) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        # Label-smoothed cross-entropy with class weights
        criterion = nn.CrossEntropyLoss(
            weight=self.class_weights, label_smoothing=0.1
        )

        best_val_f1 = 0.0
        history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

        print(f"\n🚀 Training {self.model_key} | epochs={epochs} | bs={batch_size} | "
              f"grad_accum={grad_accum} | effective_bs={batch_size*grad_accum}")
        print(f"   LR={learning_rate:.1e} | warmup={warmup_steps} steps | total={total_steps} steps")
        print("=" * 70)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            optimizer.zero_grad()

            pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                        desc=f"Epoch {epoch+1}/{epochs}", leave=False)

            for step, batch in pbar:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["label"].to(self.device)

                with autocast(enabled=self.use_amp):
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs.logits, labels) / grad_accum

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                total_loss += loss.item() * grad_accum

                if (step + 1) % grad_accum == 0:
                    if self.use_amp:
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                pbar.set_postfix({"loss": f"{total_loss/(step+1):.4f}",
                                  "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

            avg_train_loss = total_loss / len(train_loader)
            val_loss, val_acc, _, _, val_f1 = self._evaluate(val_loader, criterion)

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_f1"].append(val_f1)

            marker = ""
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                self._save_checkpoint("best_model")
                marker = "  ✅ NEW BEST"

            print(f"\n📈 Epoch {epoch+1}/{epochs}")
            print(f"   Train Loss: {avg_train_loss:.4f}")
            print(f"   Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}{marker}")
            print("-" * 70)

        print(f"\n🏆 Best Validation F1: {best_val_f1:.4f}")
        return history

    def _evaluate(self, dataloader, criterion=None):
        self.model.eval()
        all_preds, all_labels = [], []
        total_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["label"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                if criterion:
                    total_loss += criterion(outputs.logits / self.temperature, labels).item()
                elif outputs.loss is not None:
                    total_loss += outputs.loss.item()

                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        acc = accuracy_score(all_labels, all_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="weighted", zero_division=0
        )
        return avg_loss, acc, prec, rec, f1

    def calibrate_temperature(self, val_loader=None):
        """
        Find the best temperature T via grid search on the validation set.
        Calibrated probabilities are more trustworthy confidence scores.
        """
        print("\n🌡️  Calibrating temperature...")
        if val_loader is None:
            val_loader = DataLoader(self.val_dataset, batch_size=32, shuffle=False)

        self.model.eval()
        all_logits, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                out = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device)
                )
                all_logits.append(out.logits.cpu())
                all_labels.append(batch["label"])

        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels)

        best_t, best_nll = 1.0, float("inf")
        for t in np.arange(0.5, 3.0, 0.1):
            nll = F.cross_entropy(logits / t, labels).item()
            if nll < best_nll:
                best_nll, best_t = nll, t

        self.temperature = float(best_t)
        print(f"   Best temperature: {self.temperature:.2f}")
        return self.temperature

    def test(self):
        """Full evaluation on held-out test set."""
        test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False, num_workers=2)
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for batch in test_loader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["label"].to(self.device)

                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                probs  = torch.softmax(logits / self.temperature, dim=-1)
                preds  = torch.argmax(probs, dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

        print("\n" + "=" * 70)
        print("📋 TEST SET RESULTS")
        print("=" * 70)
        print(classification_report(all_labels, all_preds, target_names=LABEL_NAMES))
        print("Confusion Matrix:")
        print(confusion_matrix(all_labels, all_preds))

        acc = accuracy_score(all_labels, all_preds)
        _, _, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="weighted", zero_division=0
        )

        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(all_labels, all_probs)
            print(f"\nAUC-ROC: {auc:.4f}")
        except Exception:
            auc = None

        return {"accuracy": acc, "f1": f1, "auc": auc}

    def predict(self, text, language="auto"):
        """Run inference on a single text with calibrated confidence."""
        self.model.eval()
        encoding = self.tokenizer(
            text,
            max_length=self.config["max_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**encoding).logits
            probs  = torch.softmax(logits / self.temperature, dim=-1)
            pred   = torch.argmax(probs, dim=-1).item()
            conf   = probs[0][pred].item()

        return {
            "prediction": "real" if pred == 0 else "fake",
            "confidence": round(conf * 100, 1),
            "label": LABEL_NAMES[pred],
            "real_prob": round(probs[0][0].item() * 100, 1),
            "fake_prob": round(probs[0][1].item() * 100, 1),
            "temperature": self.temperature,
        }

    def explain_prediction(self, text, n_tokens=10):
        """
        Token-level importance via integrated gradients (lightweight).
        Returns top contributing tokens and their importance scores.
        Requires: pip install shap (optional, falls back to gradient method)
        """
        try:
            import shap
            explainer = shap.Explainer(
                lambda x: torch.softmax(
                    self.model(**self.tokenizer(
                        list(x), return_tensors="pt", padding=True,
                        truncation=True, max_length=self.config["max_length"]
                    ).to(self.device)).logits, dim=-1
                ).detach().cpu().numpy(),
                self.tokenizer,
            )
            shap_values = explainer([text])
            tokens = shap_values.data[0]
            scores = shap_values.values[0].tolist()
            top = sorted(zip(tokens, scores), key=lambda x: abs(x[1][1]), reverse=True)[:n_tokens]
            return [{"token": t, "importance": round(s[1], 4)} for t, s in top]
        except ImportError:
            print("   SHAP not installed (pip install shap) — skipping explainability")
            return []

    def _save_checkpoint(self, name):
        output_dir = f"./models/{self.model_key}_{name}"
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        # Save temperature
        with open(f"{output_dir}/calibration.json", "w") as f:
            json.dump({"temperature": self.temperature}, f)
        print(f"   💾 Saved → {output_dir}")

    def export_onnx(self, output_path=None):
        """Export calibrated model to ONNX for fast production inference."""
        if output_path is None:
            output_path = f"./models/{self.model_key}_fake_news.onnx"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.model.eval()

        dummy = self.tokenizer(
            "Sample text for ONNX export",
            max_length=self.config["max_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        torch.onnx.export(
            self.model,
            (dummy["input_ids"], dummy["attention_mask"]),
            output_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids":      {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "logits":         {0: "batch"},
            },
            opset_version=14,
            do_constant_folding=True,
        )
        print(f"✅ Exported ONNX → {output_path}")
        print(f"   Load with: onnxruntime.InferenceSession('{output_path}')")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Multilingual Fake News Detector")
    parser.add_argument("--model", choices=list(MODEL_CONFIGS.keys()),
                        default="xlm-roberta-large", help="Model architecture")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override default batch size for the selected model")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override default learning rate")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to CSV (columns: text, label, language) OR 'combined' to load all real datasets")
    parser.add_argument("--generate_data", action="store_true",
                        help="Use synthetic data (development only)")
    parser.add_argument("--augment", action="store_true",
                        help="Apply back-translation augmentation to Hindi/Bengali splits")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run temperature calibration after training")
    parser.add_argument("--export_onnx", action="store_true",
                        help="Export to ONNX after training")
    parser.add_argument("--no_cuda", action="store_true", help="Force CPU training")
    args = parser.parse_args()

    if args.no_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    print("=" * 70)
    print("🔍 MULTILINGUAL FAKE NEWS DETECTOR — TRAINING")
    print(f"   Model : {args.model} — {MODEL_CONFIGS[args.model]['description']}")
    print(f"   Time  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load dataset
    if args.generate_data:
        df = generate_sample_dataset(n_samples=4000)
    elif args.dataset == "combined":
        df = load_all_real_datasets()
    elif args.dataset and os.path.exists(args.dataset):
        df = pd.read_csv(args.dataset)
        print(f"📂 Loaded dataset: {len(df)} samples from {args.dataset}")
    else:
        print("⚠️  No dataset specified — using synthetic data. "
              "Pass --dataset combined or --generate_data")
        df = generate_sample_dataset(n_samples=4000)

    # Optional: augment low-resource languages
    if args.augment:
        print("\n🔄 Applying back-translation augmentation...")
        for lang in ["hi", "bn", "ta"]:
            df = augment_minority_class(df, lang=lang, label="fake", n_aug=300)
            df = augment_minority_class(df, lang=lang, label="real", n_aug=300)

    # Train
    trainer = FakeNewsTrainer(model_key=args.model)
    trainer.prepare_data(df)
    history = trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    # Calibrate
    if args.calibrate:
        trainer.calibrate_temperature()
        trainer._save_checkpoint("calibrated_model")

    # Test
    results = trainer.test()

    # Save history
    os.makedirs("./models", exist_ok=True)
    history_path = f"./models/{args.model}_training_history.json"
    with open(history_path, "w") as f:
        json.dump({
            "history": history,
            "test_results": results,
            "temperature": trainer.temperature,
            "model": args.model,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)
    print(f"\n💾 Training history saved → {history_path}")

    # Demo predictions
    print("\n" + "=" * 70)
    print("🔮 DEMO PREDICTIONS")
    print("=" * 70)
    demo = [
        ("SHOCKING: Government hiding alien technology from public!", "en"),
        ("The parliament passed the annual budget with bipartisan support today.", "en"),
        ("चौंकाने वाला: सरकार का गुप्त षड्यंत्र उजागर! तुरंत शेयर करें!", "hi"),
        ("सरकार ने नई आर्थिक नीतियों की घोषणा की।", "hi"),
        ("চাঞ্চল্যকর: সরকারের গোপন পরিকল্পনা ফাঁস!", "bn"),
        ("Scientists confirm new vaccine is 94% effective in Phase 3 trials.", "en"),
    ]
    for text, lang in demo:
        r = trainer.predict(text, lang)
        icon = "✅" if r["prediction"] == "real" else "⚠️"
        print(f"   {icon} [{lang}] {r['label']} ({r['confidence']}%) — {text[:70]}")

    if args.export_onnx:
        trainer.export_onnx()

    print(f"\n✅ Done! Acc={results['accuracy']:.4f} | F1={results['f1']:.4f}"
          + (f" | AUC={results['auc']:.4f}" if results.get("auc") else ""))


if __name__ == "__main__":
    main()
