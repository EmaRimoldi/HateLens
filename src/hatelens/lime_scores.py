"""LIME-based word attribution (optional ``lime`` dependency)."""

from __future__ import annotations

import logging
import os
import pickle
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer
from tqdm import tqdm

from hatelens.datasets import create_dynahate_dataset, create_hatecheck_dataset, data_dir
from hatelens.modeling import CLASS_NAMES, default_checkpoints, load_sequence_classifier
from hatelens.paths import repo_root

logger = logging.getLogger(__name__)


def predict_proba_texts(texts: list[str], model, tokenizer, device) -> np.ndarray:
    enc = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(
        device
    )
    with torch.no_grad():
        logits = model(**enc).logits
    return F.softmax(logits, dim=-1).cpu().numpy()


def compute_lime_weights_signed(
    texts,
    tokenizer,
    explainer: LimeTextExplainer,
    model,
    device,
    *,
    n_samples: int = 500,
    num_features: int = 10,
    num_samples_lime: int = 500,
):
    n = min(n_samples, len(texts))
    weights: dict[str, float] = defaultdict(float)
    indices = random.sample(range(len(texts)), n)

    for idx in tqdm(indices, desc="LIME"):
        text = texts.iloc[idx]
        probs = predict_proba_texts([text], model, tokenizer, device)
        pred_label = int(np.argmax(probs, axis=1)[0])
        explanation = explainer.explain_instance(
            text,
            lambda xs: predict_proba_texts(list(xs), model, tokenizer, device),
            num_features=num_features,
            labels=[pred_label],
            num_samples=num_samples_lime,
        )
        for word, weight in explanation.as_list(label=pred_label):
            weights[word] += weight

    vocab_size = max(len(tokenizer), 1)
    for w in weights:
        weights[w] /= vocab_size

    pos = [(w, wt) for w, wt in weights.items() if wt > 0]
    neg = [(w, wt) for w, wt in weights.items() if wt < 0]
    pos.sort(key=lambda x: abs(x[1]), reverse=True)
    neg.sort(key=lambda x: abs(x[1]), reverse=True)
    return pos, neg


def run_lime_for_dataset(
    dataset_name: str,
    *,
    n_samples: int = 500,
    num_features: int = 10,
) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    ck = default_checkpoints()
    base_id = ck["base"]["id"]
    post_path = ck["post"][dataset_name]

    if dataset_name == "hatecheck":
        seed = 33
        ds = create_hatecheck_dataset(data_dir())
        text_col = "test_case"
        n_use = len(ds["test"])
    else:
        seed = 42
        ds = create_dynahate_dataset(data_dir())
        text_col = "text"
        n_use = min(n_samples, len(ds["test"]))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df_test = ds["test"].to_pandas()
    X_test = df_test[text_col]
    explainer = LimeTextExplainer(class_names=list(CLASS_NAMES))
    results_root = repo_root() / "results"
    results_root.mkdir(parents=True, exist_ok=True)

    logger.info("LIME pre-FT %s", dataset_name)
    model_pre, tok_pre = load_sequence_classifier(base_id, device=device)
    pos_pre, neg_pre = compute_lime_weights_signed(
        X_test,
        tok_pre,
        explainer,
        model_pre,
        device,
        n_samples=n_use,
        num_features=num_features,
    )
    for name, data in (
        (f"positive_pre_FT_{dataset_name}.pkl", pos_pre),
        (f"negative_pre_FT_{dataset_name}.pkl", neg_pre),
    ):
        with open(results_root / name, "wb") as f:
            pickle.dump(data, f)

    logger.info("LIME post-FT %s", dataset_name)
    model_post, tok_post = load_sequence_classifier(post_path, device=device)
    pos_post, neg_post = compute_lime_weights_signed(
        X_test,
        tok_post,
        explainer,
        model_post,
        device,
        n_samples=n_use,
        num_features=num_features,
    )
    for name, data in (
        (f"positive_post_FT_{dataset_name}.pkl", pos_post),
        (f"negative_post_FT_{dataset_name}.pkl", neg_post),
    ):
        with open(results_root / name, "wb") as f:
            pickle.dump(data, f)
