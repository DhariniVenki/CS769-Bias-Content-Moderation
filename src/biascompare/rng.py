"""Deterministic RNG derivation for bootstrap resampling. See DESIGN.md §6.

DESIGN §6 requires every bootstrap to use "an rng derived from seed" without
mandating a specific scheme. `derive_rng` combines the run's `seed` with a
purpose tag and an index into a `numpy.random.SeedSequence`-compatible key,
so every bootstrap in a run gets its own independent, reproducible stream
instead of accidentally reusing one generator's state across models or
comparisons.
"""

from __future__ import annotations

import numpy as np

# Purpose tags, kept as plain ints (not strings) so the seed key stays a
# flat list of non-negative ints, which is what `np.random.default_rng`
# expects from a sequence seed.
DIRECTION_CI = 1
DELTA_DIFF_CI = 2
DIRECTION_DIFF_CI = 3


def derive_rng(seed: int, tag: int, index: int) -> np.random.Generator:
    """A reproducible `Generator` for one specific bootstrap in the run.

    `tag` distinguishes the kind of bootstrap (see the module constants);
    `index` distinguishes which model or comparison it belongs to. The same
    `(seed, tag, index)` always yields the same stream.
    """
    return np.random.default_rng([seed, tag, index])
