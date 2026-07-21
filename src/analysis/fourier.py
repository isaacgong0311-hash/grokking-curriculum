"""
Compares the dominant Fourier frequencies in the number-token embeddings
across curriculum conditions, following the "grokking = rotation on a
circle via a sparse set of key frequencies" account in Nanda et al. (2023).

Used to produce the "different orderings converge on the same solution"
result (paper Section 4.5): if two conditions grok, do their embeddings
land on the same frequencies, or different ones?
"""
import numpy as np

from gini import fourier_spectrum  # same FFT logic, reused rather than duplicated


def top_frequencies(embeds_final_step, k=5):
    """
    embeds_final_step: (p, d_model) embedding snapshot, typically the last
    one logged for a run.
    Returns the indices of the k frequencies with the largest mean |DFT|
    magnitude, sorted by magnitude descending.
    """
    spec = fourier_spectrum(embeds_final_step[None, ...])[0]  # (p//2+1,)
    return np.argsort(spec)[::-1][:k]


def overlap(freqs_a, freqs_b):
    """Number and identity of frequencies shared between two top-k sets."""
    a, b = set(freqs_a.tolist()), set(freqs_b.tolist())
    shared = sorted(a & b)
    return len(shared), shared


if __name__ == "__main__":
    # Reproduces the exact comparison reported in the paper:
    # random vs easy_hard share 4 of their top 5 frequencies: [15, 20, 27, 31].
    # These index values come directly from the notebook run's printed
    # output (Study 2, seed 0, addition mod 97) and are not recomputed
    # here from raw embeddings, since those .npz files live on the
    # author's Google Drive and are not duplicated in this repo (see
    # README, "What is and isn't in this repo").
    random_top5 = np.array([15, 20, 23, 27, 31])
    easy_hard_top5 = np.array([15, 20, 27, 30, 31])

    n_shared, shared = overlap(random_top5, easy_hard_top5)
    print("random top 5:     ", sorted(random_top5.tolist()))
    print("easy_hard top 5:  ", sorted(easy_hard_top5.tolist()))
    print("shared: %s (%d of 5)" % (shared, n_shared))

    assert shared == [15, 20, 27, 31]
    assert n_shared == 4
    print("\nPASS: matches paper Section 4.5 exactly")
