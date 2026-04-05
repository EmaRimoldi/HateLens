# Related work and landscape

This note is **not** a systematic survey; it anchors HateLens among adjacent public work and tools.

## Similar GitHub / ecosystem directions

- **HateTinyLLM** ([arXiv:2405.01577](https://arxiv.org/abs/2405.01577)) studies hate detection with **TinyLlama-1.1B**, Phi-2, and OPT-1.3B using LoRA/adapters — very close to this repository’s model class and motivation.
- **Mod_HATE** ([GitHub: Social-AI-Studio/Mod_HATE](https://github.com/Social-AI-Studio/Mod_HATE)) uses **modular LoRA** for multimodal (meme) hate; different modality, shared PEFT theme.
- **hare-hate-speech** ([GitHub: joonkeekim/hare-hate-speech](https://github.com/joonkeekim/hare-hate-speech)) — LLM-centric explanations (EMNLP 2023 Findings); complements post-hoc token attributions like LIME.
- **Efficient hate detection with LoRA-tuned BERTweet** ([arXiv:2511.06051](https://arxiv.org/abs/2511.06051)) — encoder + three-layer LoRA, efficiency focus on a **much smaller** base than 1B decoder LMs.

## Explainability

- **LIME** remains a common **model-agnostic** baseline for token/word-level narratives; gradient/saliency toolkits (e.g. Captum) are an alternative axis for BERT-style classifiers. Blog walkthrough: [Explaining BERT-based hate speech detection with LIME and Saliency](https://omseeth.github.io/blog/2025/Explaining-BERT/).

## Where HateLens can differentiate (honestly)

Many repos stop at accuracy curves. This fork adds:

- **Correct PEFT inference** for shipped adapters.
- **Batched evaluation** and CSV metric summaries.
- **HateCheck functional diagnostics** (`hatelens diagnose-hatecheck`) using preserved `functionality` metadata — aligned with the original HateCheck motivation (functional tests), without claiming new SOTA.

## Gaps / not claimed

- No new dataset, no human study of explanation quality, and no multimodal scope.
- LIME faithfulness and robustness are **not** solved here; they remain open research questions.
