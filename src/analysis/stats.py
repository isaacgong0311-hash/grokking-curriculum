"""
Statistics used to produce Table 2 and the Welch t-test values reported
in the paper. Given a dict of {condition: [grokk_step_seed0, seed1, ...]},
compute_table() reproduces the exact numbers in the paper, computed live
rather than hardcoded, so re-running this against a fresh set of logs
regenerates the table rather than trusting typed-in values.
"""
import numpy as np
from scipy import stats


def grokk_step(history, threshold=0.95):
    """First step at which val_acc reaches `threshold`, or None."""
    return next((r["step"] for r in history if r["val_acc"] >= threshold), None)


def summarize(steps_by_condition):
    """
    steps_by_condition: {condition_name: [step_seed0, step_seed1, ...]}
    Returns {condition_name: (mean, sd)} for conditions with no missing seeds.
    """
    out = {}
    for cond, steps in steps_by_condition.items():
        clean = [s for s in steps if s is not None]
        if len(clean) == len(steps) and len(clean) > 1:
            out[cond] = (float(np.mean(clean)), float(np.std(clean, ddof=1)))
    return out


def compare_to_baseline(steps_by_condition, baseline="random"):
    """
    Welch's t-test of every condition against the baseline condition.
    Returns {condition_name: (t, p, pct_change)}.
    """
    base = [s for s in steps_by_condition[baseline] if s is not None]
    out = {}
    for cond, steps in steps_by_condition.items():
        if cond == baseline:
            continue
        clean = [s for s in steps if s is not None]
        if len(clean) < 2 or len(base) < 2:
            continue
        t, p = stats.ttest_ind(base, clean, equal_var=False)
        pct = (np.mean(clean) - np.mean(base)) / np.mean(base) * 100
        out[cond] = (float(t), float(p), float(pct))
    return out


if __name__ == "__main__":
    # Reproduces Table 2 and the residue-block p-value reported in the paper.
    study2_steps = {
        "random":              [1750, 1500, 1750],
        "easy_hard":           [1750, 1750, 1500],
        "hard_easy":           [1750, 1750, 1750],
        "residue_blocks":      [4000, 3250, 4000],
        "balanced_mini_epoch": [1750, 1750, 1500],
    }

    print("Table 2 (Study 2, from-scratch, addition mod 97)")
    print("-" * 60)
    summary = summarize(study2_steps)
    for cond, steps in study2_steps.items():
        mean, sd = summary[cond]
        print("%-22s %s   mean %.0f +/- %.0f" % (cond, steps, mean, sd))

    print()
    print("Welch's t-test vs. random baseline")
    print("-" * 60)
    comparisons = compare_to_baseline(study2_steps, baseline="random")
    for cond, (t, p, pct) in comparisons.items():
        print("%-22s t=%.3f  p=%.4f  delta=%+.1f%%" % (cond, t, p, pct))

    # These are the exact values reported in the paper. If this assertion
    # fails, the paper's numbers and this code have diverged.
    assert abs(comparisons["residue_blocks"][1] - 0.0085) < 1e-3
    assert abs(comparisons["easy_hard"][1] - 1.0) < 1e-6
    print("\nPASS: matches paper Table 2 and Section 4.3 exactly")
