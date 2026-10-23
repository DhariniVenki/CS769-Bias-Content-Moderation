# biascompare — POC

A proof of concept for a comparative model-bias testing framework. Given two (or more)
Hugging Face text-classification models, a dataset of social-media posts, and an optional
instruction prompt, it measures each model's **gender bias beyond its general sensitivity
to wording** and compares every model against a reference model — following Yang, Levy,
Goldberg & Wallace (2026), *"Compared to What? Baselines and Metrics for Counterfactual
Prompting"* (arXiv:2605.01048).

The POC only has to show one thing: **given 2 models, we can compare their biases.**

See [`DESIGN.md`](./DESIGN.md) for the full design rationale — every decision made while
building this, and why — including the exact statistics (§6) and output JSON schema (§7).

## Quickstart: Google Colab

1. Open the notebook in Colab:
   [colab.research.google.com/github/DhariniVenki/CS769-Bias-Content-Moderation/blob/main/poc.ipynb](https://colab.research.google.com/github/DhariniVenki/CS769-Bias-Content-Moderation/blob/main/poc.ipynb)
2. Run the cells top to bottom. The first cell clones the repo into `/content/` and
   installs the package; re-run it any time to pull newly pushed changes, and restart the
   runtime after any change to a dataclass or a top-level import (see the notebook's setup
   note for details).
3. No Hugging Face token is required for the two default models (both public). An optional
   token can be set as a Colab Secret named `HF_TOKEN` for gated/private models.

Outside Colab, the same setup cell assumes it is running from the repository root and
skips the clone/install step — the notebook works locally too.

## Local development

```bash
pip install -e .[dev]
pytest -q
```

Tests are fully offline (fake scorers, no model downloads or network access). Viz tests
are skipped automatically if `matplotlib` is not installed.

## Layout

See [`DESIGN.md` §8](./DESIGN.md#8-code-layout) for the full code layout and the reasoning
behind it. In short: perturbation and scoring logic live in `src/biascompare/`, the ten
hand-written posts are in `data/posts.jsonl`, `poc.ipynb` only calls the package, and
results are written to `results/`.
