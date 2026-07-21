"""
Gini coefficient of Fourier component norms.

This is the progress measure from Nanda et al. (2023), not a metric we
invented. It is included here because an earlier draft of this project
mis-cited Nanda et al. as reporting a rise-then-fall in weight norm (they
report the opposite: weight norm decreases through most of training) and
used that mis-citation to manufacture a "negative result." This module is
what the paper actually reports Nanda's real progress measure with.
"""
import numpy as np


def gini(x):
    """
    Gini coefficient of a 1D array of non-negative magnitudes.
    0 = perfectly uniform, 1 = fully concentrated in one component.
    """
    x = np.sort(np.abs(np.asarray(x, dtype=np.float64)))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float(((2 * idx - n - 1) * x).sum() / (n * x.sum()))


def fourier_spectrum(embeds):
    """
    embeds: (n_snapshots, p, d_model) array of number-token embeddings
    logged over training (see model.GrokkTransformer.number_embeddings).

    Returns (n_snapshots, p//2 + 1): magnitude of the real FFT along the
    p axis, averaged over the d_model axis.
    """
    return np.abs(np.fft.rfft(embeds, axis=1)).mean(axis=2)


def gini_series(embeds):
    """Gini coefficient of the Fourier spectrum at each logged step."""
    spec = fourier_spectrum(embeds)
    return np.array([gini(spec[i]) for i in range(spec.shape[0])])


def half_rise_step(steps, gini_values):
    """
    First step at which the Gini series reaches half its total rise
    (final value minus initial value). Used to measure how many steps
    the Gini rise leads the grokking transition by.
    """
    g0, g1 = gini_values[0], gini_values[-1]
    if g1 <= g0:
        return None
    threshold = g0 + 0.5 * (g1 - g0)
    for s, g in zip(steps, gini_values):
        if g >= threshold:
            return int(s)
    return None


if __name__ == "__main__":
    # Regression test against the bug this module was written to catch:
    # without model.GrokkTransformer.number_embeddings()'s .copy() call,
    # every logged snapshot on CPU is a view of the SAME final weights,
    # and gini_series() would return an identical value at every step.
    rng = np.random.RandomState(0)
    n_steps, p, d = 6, 97, 32
    fake_embeds = np.stack(
        [rng.randn(p, d) * (1 + 0.5 * i) for i in range(n_steps)]
    )
    series = gini_series(fake_embeds)
    assert len(set(np.round(series, 6))) > 1, (
        "gini_series returned identical values across steps -- this is "
        "the exact symptom of the embedding-snapshot aliasing bug"
    )
    print("gini values across synthetic steps:", np.round(series, 4))
    print("PASS: gini_series produces distinct values per step")
