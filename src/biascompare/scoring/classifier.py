"""Hugging Face text-classification scorer. See DESIGN.md §4.

`torch` and `transformers` are imported lazily, inside `HFClassifierScorer`
methods only, so importing this module (and the rest of the package) never
requires either to be installed.
"""

from __future__ import annotations

import gc
from collections.abc import Sequence

from biascompare.inputs import ModelSpec
from biascompare.scoring.base import ProbPair, binary_from_logits, resolve_label_index

_DEFAULT_MAX_LENGTH = 512


def pick_device(requested: str = "auto") -> str:
    """Resolve "auto" to cuda > mps > cpu; any other value is returned unchanged."""
    if requested != "auto":
        return requested

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class HFClassifierScorer:
    """Scorer backed by `AutoModelForSequenceClassification` (DESIGN §4).

    Loads eagerly in `__init__`; call `close()` once done so its memory (and
    CUDA/MPS cache) is freed before the next model loads — the POC loads one
    model at a time.
    """

    scorer_type = "hf-classifier"
    uses_instruction = False  # accepted for protocol parity; classifiers ignore it (DESIGN §3)

    def __init__(
        self,
        spec: ModelSpec,
        instruction: str | None = None,
        device: str = "auto",
        batch_size: int = 16,
    ) -> None:
        self.name = spec.name
        self._instruction = instruction
        self._batch_size = batch_size
        self._device = pick_device(device)

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "HFClassifierScorer requires the `torch` and `transformers` packages "
                "to be installed. Install the project's dependencies (see "
                "pyproject.toml) before scoring."
            ) from exc

        token = spec.resolved_token()
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(spec.name, token=token)
            self._model = AutoModelForSequenceClassification.from_pretrained(spec.name, token=token)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load model {spec.name!r}. If this is a gated or private "
                "model, pass a valid `token` on its ModelSpec or set the HF_TOKEN "
                f"environment variable. Original error: {exc}"
            ) from exc

        self._model.to(self._device)
        self._model.eval()

        self._multi_label = self._model.config.problem_type == "multi_label_classification"
        self._toxic_index = resolve_label_index(self._model.config.id2label, spec.toxic_label)
        self._max_length = min(self._tokenizer.model_max_length, _DEFAULT_MAX_LENGTH)
        self._torch = torch

    def score(self, texts: Sequence[str]) -> list[ProbPair]:
        """Batch-score `texts`, truncating/padding per DESIGN §4."""
        torch = self._torch
        results: list[ProbPair] = []

        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            ).to(self._device)

            with torch.inference_mode():
                outputs = self._model(**encoded)

            logits = outputs.logits.detach().cpu().numpy()
            for row in logits:
                results.append(binary_from_logits(row, self._toxic_index, self._multi_label))

        return results

    def close(self) -> None:
        """Free the model and any accelerator cache it held."""
        torch = self._torch
        device = self._device
        if hasattr(self, "_model"):
            del self._model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()


def hf_classifier_factory(
    spec: ModelSpec, instruction: str | None, device: str
) -> HFClassifierScorer:
    """Default `ScorerFactory`: builds an `HFClassifierScorer` for `spec`."""
    return HFClassifierScorer(spec=spec, instruction=instruction, device=device)
