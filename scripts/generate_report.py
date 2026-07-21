"""
Regenerates Table 2 and the LaTeX table used in the paper directly from
results/summary.json. Run this instead of trusting numbers typed into the
paper by hand.

Usage:
    python scripts/generate_report.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "analysis"))
from stats import summarize, compare_to_baseline  # noqa: E402

HERE = os.path.dirname(__file__)
SUMMARY_PATH = os.path.join(HERE, "..", "results", "summary.json")

NICE = {
    "random": "Random (baseline)",
    "easy_hard": "Easy to Hard",
    "hard_easy": "Hard to Easy",
    "residue_blocks": "Residue Blocks",
    "balanced_mini_epoch": "Balanced Mini-Epoch",
}
ORDER = ["random", "easy_hard", "hard_easy", "residue_blocks", "balanced_mini_epoch"]


def main():
    with open(SUMMARY_PATH) as f:
        data = json.load(f)

    s2 = data["study2_from_scratch"]["grokking_steps_full_grid"]
    steps = {k: v["steps"] for k, v in s2.items()}
    summary = summarize(steps)
    comparisons = compare_to_baseline(steps, baseline="random")

    print("=" * 60)
    print("Study 2 (from-scratch): grokking step per condition")
    print("=" * 60)
    for cond in ORDER:
        mean, sd = summary[cond]
        print("%-22s %s   mean %d +/- %d" % (NICE[cond], steps[cond], round(mean), round(sd)))

    print()
    print("Welch's t-test vs. random baseline")
    print("-" * 60)
    for cond in ORDER:
        if cond == "random":
            continue
        t, p, pct = comparisons[cond]
        flag = "  <-- significant at 0.05" if p < 0.05 else ""
        print("%-22s t=%7.3f  p=%.4f  delta=%+6.1f%%%s" % (NICE[cond], t, p, pct, flag))

    print()
    print("=" * 60)
    print("LaTeX table (paste into paper)")
    print("=" * 60)
    print(r"\begin{table}[H]")
    print(r"\centering")
    print(
        r"\caption{From-scratch transformer, addition mod 97, 3 seeds. Step at which "
        r"validation accuracy first reaches 95\%.}"
    )
    print(r"\small")
    print(r"\begin{tabular}{@{}lccccc@{}}")
    print(r"\toprule")
    print(
        r"\textbf{Condition} & \textbf{Seed 0} & \textbf{Seed 1} & \textbf{Seed 2}"
        r" & \textbf{Mean $\pm$ SD} & \textbf{vs.\ random} \\"
    )
    print(r"\midrule")
    for cond in ORDER:
        seeds = steps[cond]
        mean, sd = summary[cond]
        if cond == "random":
            vs = "---"
        else:
            vs = "$p = %.4f$" % comparisons[cond][1]
        print(
            r"%s & %s & %s & %s & $%d \pm %d$ & %s \\"
            % (NICE[cond], seeds[0], seeds[1], seeds[2], round(mean), round(sd), vs)
        )
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    print()
    print("=" * 60)
    print("Study 1 (GPT-2 fine-tuning, exploratory)")
    print("=" * 60)
    s1 = data["study1_gpt2_finetuning"]
    for cond, v in s1["multi_seed_conditions"].items():
        print("%-12s %s   mean %d +/- %d" % (cond, v["grokking_steps"], v["mean"], v["sd"]))
    wt = s1["welch_t_test_random_vs_easy_hard"]
    print("Welch t-test random vs easy_hard: t=%.2f p=%.3f" % (wt["t"], wt["p"]))
    print()
    print("NOTE: this effect did NOT replicate in Study 2 (see comparison above,")
    print("p=1.0). See paper Section 5 for discussion.")


if __name__ == "__main__":
    main()
