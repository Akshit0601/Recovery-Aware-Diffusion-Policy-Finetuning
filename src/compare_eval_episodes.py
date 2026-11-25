"""Paired comparison of two per-episode evaluation CSVs (written by DPPOTrainer.evaluate).

Both evals must use the same rl.eval_seed and episodes-per-regime, so rows with the
same `episode_key` are the same scenario (initial object pose + disturbance draw).

Usage:
    python -m src.compare_eval_episodes A.csv B.csv
e.g.
    python -m src.compare_eval_episodes \
        results/dppo_checkpoints/eval_episodes/evalonly_pretrained_n100.csv \
        results/dppo_checkpoints/eval_episodes/evalonly_dppo_best_iter_100_n100.csv
"""
import argparse
import csv
from math import comb

REGIMES = ["nominal", "train_approaching", "train_pre_grasp", "ho_approaching", "ho_pre_grasp"]


def load(path):
    with open(path) as f:
        return {r["episode_key"]: r for r in csv.DictReader(f)}


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", help="baseline CSV (A)")
    ap.add_argument("b", help="comparison CSV (B)")
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    keys = A.keys() & B.keys()
    if not keys:
        raise SystemExit("No shared episode_key: were both evals run with the same eval_seed and n?")
    if len(keys) != len(A) or len(keys) != len(B):
        print(f"warning: {len(A)} vs {len(B)} rows, {len(keys)} shared; comparing shared rows only")

    print(f"A = {args.a}\nB = {args.b}\n")
    print(f"{'regime':<20}{'n':>5}{'A %':>8}{'B %':>8}{'Δ':>7}{'A✗B✓':>7}{'A✓B✗':>7}{'p (McNemar)':>13}")
    rows_all = []
    for regime in REGIMES + ["all_disturbed"]:
        if regime == "all_disturbed":
            ks = [k for k in keys if A[k]["regime"] != "nominal"]
        else:
            ks = [k for k in keys if A[k]["regime"] == regime]
        if not ks:
            continue
        a = [int(A[k]["success"]) for k in ks]
        b = [int(B[k]["success"]) for k in ks]
        fixed = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)   # A failed, B succeeded
        broke = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)   # A succeeded, B failed
        pa, pb = 100 * sum(a) / len(ks), 100 * sum(b) / len(ks)
        print(f"{regime:<20}{len(ks):>5}{pa:>8.1f}{pb:>8.1f}{pb - pa:>+7.1f}{fixed:>7}{broke:>7}"
              f"{mcnemar_exact(fixed, broke):>13.3f}")
        if regime != "all_disturbed":
            rows_all += [(k, x, y) for k, x, y in zip(ks, a, b) if x != y]

    print("\nDiscordant disturbed episodes (Δx, Δy in cm; Δyaw in deg):")
    print(f"{'regime':<20}{'outcome':<10}{'Δx':>7}{'Δy':>7}{'Δyaw':>8}{'applied A/B':>13}")
    for k, _, y in sorted(rows_all, key=lambda t: (A[t[0]]["regime"], t[1])):
        r = A[k]
        if r["regime"] == "nominal":
            print(f"{'nominal':<20}{'B fixed' if y else 'B broke':<10}")
            continue
        print(f"{r['regime']:<20}{'B fixed' if y else 'B broke':<10}"
              f"{100 * float(r['delta_x']):>7.1f}{100 * float(r['delta_y']):>7.1f}{float(r['delta_yaw_deg']):>8.1f}"
              f"{r['disturbance_applied'] + '/' + B[k]['disturbance_applied']:>13}")


if __name__ == "__main__":
    main()
