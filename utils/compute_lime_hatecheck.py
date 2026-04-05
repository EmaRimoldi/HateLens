#!/usr/bin/env python3
"""
Legacy standalone LIME script for HateCheck.

Prefer: ``uv run hatelens lime --hatecheck`` (writes under ``outputs/lime/``).
"""

import os
import random
import pickle
import logging
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils.datasets import create_hatecheck_dataset

# ------------------------------------------------------------------------------
# Configuration & Reproducibility
# ------------------------------------------------------------------------------
SEED = 33
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
warnings.filterwarnings("ignore")

RESULTS_DIR = os.environ.get("HATELENS_LIME_DIR", "outputs/lime")
MODEL_CHECKPOINT_PRE = os.environ.get(
    "HATELENS_BASE_MODEL", "PY007/TinyLlama-1.1B-step-50K-105b"
)
MODEL_CHECKPOINT_POST = os.environ.get(
    "HATELENS_CKPT_HATECHECK", "outputs/runs/tinyllama/hatecheck/best_checkpoint"
)
CLASS_NAMES = ["nothate", "hate"]

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def get_device() -> torch.device:
    """Return the best available device: CUDA, MPS (Mac), or CPU."""
    if torch.cuda.is_available():
        logger.info("CUDA is available. Using GPU.")
        return torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        logger.info("MPS is available. Using Apple GPU.")
        return torch.device("mps")
    else:
        logger.info("CUDA and MPS not available. Using CPU.")
        return torch.device("cpu")

def prepare_model_and_tokenizer(checkpoint: str, device: torch.device):
    """Load sequence classification model and tokenizer from checkpoint."""
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=os.path.isdir(checkpoint))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=len(CLASS_NAMES),
        local_files_only=os.path.isdir(checkpoint)
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device).eval()
    logger.info("Loaded model from %s onto %s", checkpoint, device)
    return model, tokenizer

def predict_proba(texts, model, tokenizer, device):
    """Return class probabilities for a list of texts."""
    enc = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    return F.softmax(logits, dim=-1).cpu().numpy()

def compute_lime_weights_signed(
    texts,
    tokenizer,
    explainer: LimeTextExplainer,
    model,
    device,
    n_samples: int = 500,
    num_features: int = 10,
    num_samples_lime: int = 500,
):
    """
    Aggregate signed LIME weights over a random subset of texts.
    Returns two lists of (word, normalized_weight): positive and negative.
    """
    logger.info("Computing LIME weights on %d samples", min(n_samples, len(texts)))
    weights = defaultdict(float)
    indices = random.sample(range(len(texts)), min(n_samples, len(texts)))

    for idx in tqdm(indices, desc="LIME weights"):
        text = texts.iloc[idx]
        probs = predict_proba([text], model, tokenizer, device)
        pred_label = int(np.argmax(probs, axis=1)[0])
        explanation = explainer.explain_instance(
            text,
            lambda xs: predict_proba(xs, model, tokenizer, device),
            num_features=num_features,
            labels=[pred_label],
            num_samples=num_samples_lime,
        )
        for word, weight in explanation.as_list(label=pred_label):
            weights[word] += weight

    # normalize by vocabulary size
    vocab_size = len(tokenizer)
    for word in weights:
        weights[word] /= vocab_size

    pos = [(w, wt) for w, wt in weights.items() if wt > 0]
    neg = [(w, wt) for w, wt in weights.items() if wt < 0]

    # sort by absolute weight
    pos.sort(key=lambda x: abs(x[1]), reverse=True)
    neg.sort(key=lambda x: abs(x[1]), reverse=True)

    return pos, neg

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    device = get_device()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load test set
    logger.info("Loading HateCheck test data")
    hatecheck = create_hatecheck_dataset()
    df_test = hatecheck["test"].to_pandas()
    X_test = df_test["test_case"]

    explainer = LimeTextExplainer(class_names=CLASS_NAMES)

    # --- Pre-Fine-Tuning LIME ---
    model_pre, tokenizer_pre = prepare_model_and_tokenizer(MODEL_CHECKPOINT_PRE, device)
    positive_pre, negative_pre = compute_lime_weights_signed(
        X_test, tokenizer_pre, explainer, model_pre, device,
        n_samples=len(X_test), num_features=10, num_samples_lime=500
    )
    with open(os.path.join(RESULTS_DIR, "positive_words_pre_FT_hatecheck.pkl"), "wb") as f:
        pickle.dump(positive_pre, f)
    with open(os.path.join(RESULTS_DIR, "negative_words_pre_FT_hatecheck.pkl"), "wb") as f:
        pickle.dump(negative_pre, f)
    logger.info("Saved pre-FT LIME weights")

    # --- Post-Fine-Tuning LIME ---
    model_post, tokenizer_post = prepare_model_and_tokenizer(MODEL_CHECKPOINT_POST, device)
    positive_post, negative_post = compute_lime_weights_signed(
        X_test, tokenizer_post, explainer, model_post, device,
        n_samples=len(X_test), num_features=10, num_samples_lime=500
    )
    with open(os.path.join(RESULTS_DIR, "positive_words_post_FT_hatecheck.pkl"), "wb") as f:
        pickle.dump(positive_post, f)
    with open(os.path.join(RESULTS_DIR, "negative_words_post_FT_hatecheck.pkl"), "wb") as f:
        pickle.dump(negative_post, f)
    logger.info("Saved post-FT LIME weights")

if __name__ == "__main__":
    main()
