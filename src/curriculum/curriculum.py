"""
Curriculum ordering logic shared by Study 1 (GPT-2 fine-tuning) and
Study 2 (from-scratch transformer).

Both studies import this module unchanged. That is a factual claim, not a
convenience: it is what lets the paper say the two studies use "identical
curriculum code" and rules out a curriculum-implementation difference as
the explanation for why Study 1's difficulty effect did not replicate in
Study 2 (see paper Section 5, "Why did the difficulty effect not
replicate?").
"""
import random


def make_pairs(modulus, operation):
    """All (a, b, c) triples for a op b = c (mod p), a,b in [0, p)."""
    p = modulus
    out = []
    for a in range(p):
        for b in range(p):
            c = (a + b) % p if operation == "add" else (a * b) % p
            out.append((a, b, c))
    return out


def split_data(pairs, train_fraction, seed):
    """Deterministic random split into (train_pairs, val_pairs)."""
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * train_fraction)
    return shuffled[:n_train], shuffled[n_train:]


def is_wrapping(triple, modulus, operation):
    """
    True if the raw (un-reduced) operation crosses the modulus boundary.

    Addition:       (a + b) >= p  -> wrapping ("hard": needs real mod reduction)
    Multiplication: (a * b) >= p  -> wrapping

    Non-wrapping pairs are ones where the raw arithmetic result already
    equals the answer, e.g. 3 + 4 = 7 when p = 97. This is the difficulty
    proxy used by the easy_hard / hard_easy conditions.
    """
    a, b, _ = triple
    raw = (a + b) if operation == "add" else (a * b)
    return raw >= modulus


def apply_curriculum(pairs, curriculum, modulus, operation, seed=0):
    """
    Reorder training pairs according to `curriculum`. Returns a new list;
    does not mutate the input.

    Conditions
    ----------
    random               No-op here. The training loop is responsible for
                          setting shuffle=True for this condition so the
                          DataLoader reshuffles every epoch. This function
                          returning the pairs unchanged is what the paper's
                          Methods section refers to as "for the random
                          condition the loader reshuffles every epoch."
    easy_hard             Non-wrapping pairs first, then wrapping pairs.
                          Stable sort: order within each group is preserved.
    hard_easy              Wrapping pairs first, then non-wrapping pairs.
    residue_blocks        All pairs with result c=0, then c=1, ..., c=p-1.
    balanced_mini_epoch    Round-robin: one pair per residue class per
                          round, so every contiguous window of (up to) p
                          examples touches every residue class once.
    """
    if curriculum == "random":
        return list(pairs)

    if curriculum in ("easy_hard", "hard_easy"):
        difficulty = lambda t: 1 if is_wrapping(t, modulus, operation) else 0
        ordered = sorted(pairs, key=difficulty)
        return ordered[::-1] if curriculum == "hard_easy" else ordered

    if curriculum == "residue_blocks":
        blocks = {}
        for t in pairs:
            blocks.setdefault(t[2], []).append(t)
        out = []
        for c in sorted(blocks):
            out.extend(blocks[c])
        return out

    if curriculum == "balanced_mini_epoch":
        blocks = {}
        for t in pairs:
            blocks.setdefault(t[2], []).append(t)
        rng = random.Random(seed)
        for c in blocks:
            rng.shuffle(blocks[c])
        out = []
        iterators = {c: iter(v) for c, v in sorted(blocks.items())}
        active = sorted(blocks.keys())
        while active:
            next_round = []
            for c in active:
                try:
                    out.append(next(iterators[c]))
                    next_round.append(c)
                except StopIteration:
                    pass
            active = next_round
        return out

    raise ValueError(
        "Unknown curriculum '%s'. Choose from: random, easy_hard, "
        "hard_easy, residue_blocks, balanced_mini_epoch" % curriculum
    )


CURRICULA = ["random", "easy_hard", "hard_easy", "residue_blocks", "balanced_mini_epoch"]


if __name__ == "__main__":
    # Sanity check: every ordering must be a permutation of the input.
    # This is the same check run before every experiment grid in this repo.
    pairs = make_pairs(modulus=13, operation="add")
    train_pairs, _ = split_data(pairs, train_fraction=0.5, seed=0)
    for cur in CURRICULA:
        ordered = apply_curriculum(train_pairs, cur, modulus=13, operation="add", seed=0)
        assert sorted(ordered) == sorted(train_pairs), cur + " is not a permutation"
        print("%-22s ok  first 3: %s" % (cur, ordered[:3]))
    print("\nall orderings valid")
