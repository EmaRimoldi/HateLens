# Datasets (quick reference)

Full onboarding lives in the root `README.md`. This page summarizes acquisition and roles:

- **DynaHate** — CSV in `data/DynaHate/`; binary labels after mapping.
- **HateEval** — English TSVs in `data/HateEval/` (`train_en.tsv`, `dev_en.tsv`); public test is unlabeled; the loader stratifies dev for val/test (see `datasets.py` docstring).
- **HateXplain** — loaded via Hugging Face; rationale spans are best-effort aligned for structured training.
- **HateCheck** — `data/hatecheck/hatecheck_split.csv` for functional evaluation.

If downloads fail (SSL / firewall), place files manually and re-run.
