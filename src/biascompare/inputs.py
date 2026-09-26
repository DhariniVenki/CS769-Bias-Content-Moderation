"""Input data types and loading/validation. See DESIGN.md §3.

`ModelSpec` describes a Hugging Face text-classification model to score with,
and `Post`/`load_posts` describe and load the dataset of social-media posts
to perturb and score. Both dataclasses are frozen: nothing here is mutated
after construction.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

_VALID_GENDERS = frozenset({"male", "female"})
_REQUIRED_POST_FIELDS = ("id", "platform", "text", "gender")


@dataclass(frozen=True)
class ModelSpec:
    """A Hugging Face model to score with, plus which label means "toxic".

    `token` is an optional per-model access token for gated/private models.
    It is excluded from `repr()` and from equality/hash comparison, and
    `to_public_dict()` never includes it, so it can never leak into logs or
    the results JSON (DESIGN §3).
    """

    name: str
    toxic_label: str = "toxic"
    token: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"ModelSpec.name must be a non-empty string, got {self.name!r}")
        if not isinstance(self.toxic_label, str) or not self.toxic_label.strip():
            raise ValueError(
                f"ModelSpec.toxic_label must be a non-empty string, got {self.toxic_label!r}"
            )

    @property
    def label(self) -> str:
        """Short display name: the last path component of the HF repo id."""
        return self.name.rstrip("/").split("/")[-1]

    def resolved_token(self) -> str | None:
        """The explicit token if set, else `HF_TOKEN` from the environment, else None.

        `None` tells `from_pretrained` to fall back to a cached
        `huggingface-cli login` session, if any.
        """
        if self.token is not None:
            return self.token
        return os.environ.get("HF_TOKEN") or None

    def to_public_dict(self) -> dict:
        """Serialisable view for results/logs. Never includes the token."""
        return {"name": self.name, "toxic_label": self.toxic_label}


@dataclass(frozen=True)
class Post:
    """One social-media post plus the metadata needed to perturb and score it."""

    id: str
    platform: str
    text: str
    gender: str
    intended_toxicity: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty_str("id", self.id)
        _require_nonempty_str("platform", self.platform)
        _require_nonempty_str("text", self.text)
        _require_nonempty_str("gender", self.gender)
        if self.gender not in _VALID_GENDERS:
            raise ValueError(
                f"Post.gender must be one of {sorted(_VALID_GENDERS)}, "
                f"got {self.gender!r} (id={self.id!r})"
            )


def _require_nonempty_str(field_name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Post.{field_name} must be a non-empty string, got {value!r}")


def load_posts(path: str | Path) -> tuple[Post, ...]:
    """Load and validate posts from a JSONL file (DESIGN §3).

    One JSON object per non-blank line. Validates: each line is valid JSON
    (error names the line number), the required fields (id, platform, text,
    gender) are present and non-empty, `gender` is "male" or "female", ids
    are unique across the file, and the file yields at least one post.

    Raises ValueError with an actionable message for any violation.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise ValueError(f"Posts file not found: {file_path}")

    posts: list[Post] = []
    seen_ids: set[str] = set()

    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            record = _parse_json_line(line, line_number, file_path)
            _require_dict_record(record, line_number, file_path)
            _require_present_fields(record, line_number, file_path)

            try:
                post = Post(
                    id=record["id"],
                    platform=record["platform"],
                    text=record["text"],
                    gender=record["gender"],
                    intended_toxicity=record.get("intended_toxicity"),
                )
            except ValueError as exc:
                raise ValueError(f"Line {line_number} of {file_path}: {exc}") from exc

            if post.id in seen_ids:
                raise ValueError(f"Duplicate post id {post.id!r} on line {line_number} of {file_path}")
            seen_ids.add(post.id)
            posts.append(post)

    if not posts:
        raise ValueError(f"No posts found in {file_path}: file is empty or contains only blank lines")

    return tuple(posts)


def _parse_json_line(line: str, line_number: int, file_path: Path) -> object:
    try:
        return json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON on line {line_number} of {file_path}: {exc}") from exc


def _require_dict_record(record: object, line_number: int, file_path: Path) -> None:
    if not isinstance(record, dict):
        raise ValueError(
            f"Line {line_number} of {file_path} must be a JSON object, got {type(record).__name__}"
        )


def _require_present_fields(record: dict, line_number: int, file_path: Path) -> None:
    missing = [f for f in _REQUIRED_POST_FIELDS if f not in record]
    if missing:
        raise ValueError(
            f"Line {line_number} of {file_path} is missing required field(s): {missing}"
        )
