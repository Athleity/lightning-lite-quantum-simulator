"""Neural vs matching decoders on the rotated surface code memory (Tier 5 QEC).

Compares plain MWPM, correlated MWPM and the hybrid MLP decoder (src/NeuralDecoder.py)
on identical test shots:

  docs/images/qec_neural_decoder.png   per-round logical error vs p (one panel per
                                       distance) and Lambda(5 -> 7) per decoder

Protocol. For every (d, p) one NeuralDecoder is trained on fresh shots of that exact
circuit (SurfaceCodeMemory(d, p, rounds = d), SD6 noise), then SurfaceCodeMemory.compare
decodes the same held-out shots with all three decoders. Test seeds differ from the
training seeds. Lambda(5 -> 7) is eps_5 / eps_7 per decoder at P_LAMBDA.

Caveats stated up front: the network is a small MLP trained on at most 1e6 shots, not
Google's recurrent transformer, and the syndrome space at d = 7 is far larger than the
training set, so it is expected to be at or below MWPM there. Results are reported as
measured, and no check requires the neural decoder to win. Neural 'seconds' in compare
include its internal MWPM pass (the hybrid input), so network-only inference time is
timed separately.

Run:
  python3 benchmarks/qec_neural.py              full run (long: hours on CPU)
  QEC_QUICK=1 python3 benchmarks/qec_neural.py  trimmed run (minutes)
"""

import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from SurfaceCode import SurfaceCodeMemory, lambda_pairwise  # noqa: E402
from NeuralDecoder import NeuralDecoder  # noqa: E402

QUICK = os.environ.get("QEC_QUICK", "0") == "1"

IMAGES = ROOT / "docs" / "images"
IMAGES.mkdir(parents=True, exist_ok=True)

P_LAMBDA = 0.002
SWEEP_D = (3,) if QUICK else (3, 5)
P_GRID = (0.002, 0.004) if QUICK else (0.002, 0.003, 0.004, 0.006)
LAMBDA_D = (5, 7)
TRAIN_SHOTS = 100_000 if QUICK else 1_000_000
EPOCHS = 5 if QUICK else 10
BATCH = 512
TEST_SHOTS = 100_000 if QUICK else 500_000
TIMING_SHOTS = 20_000

NAMES = ("mwpm", "correlated", "neural")
LABELS = {"mwpm": "MWPM", "correlated": "correlated MWPM", "neural": "neural (hybrid MLP)"}
COLOURS = {"mwpm": "#999999", "correlated": "#3b6ea5", "neural": "#d1495b"}
MARKS = {"mwpm": "s", "correlated": "o", "neural": "^"}

_checks = []
_cache = {}


def check(name, ok, detail=""):
    _checks.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def hidden_for(d):
    return 128 if d == 3 else 256


def net_only_seconds(nd, shots=TIMING_SHOTS, seed=424242):
    """Seconds per shot of the network forward pass alone (features precomputed)."""
    det, _ = nd.circuit.compile_detector_sampler(seed=seed).sample(
        shots, separate_observables=True)
    t0 = time.perf_counter()
    feats = nd._features(det)
    t_feat = time.perf_counter() - t0
    x = torch.from_numpy(feats.astype(np.float32))
    nd.net.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, shots, 65536):
            nd.net(x[i:i + 65536])
    t_net = time.perf_counter() - t0
    return t_net / shots, t_feat / shots


def run_point(d, p):
    """Train a neural decoder for (d, p) and compare all three on shared test shots."""
    key = (d, p)
    if key in _cache:
        return _cache[key]
    print(f"\n  d={d} p={p:g}: training ({TRAIN_SHOTS} shots, {EPOCHS} epochs, "
          f"hidden {hidden_for(d)})", flush=True)
    nd = NeuralDecoder(d, d, p, hidden=hidden_for(d), seed=0)
    hist = nd.train(n_samples=TRAIN_SHOTS, epochs=EPOCHS, batch=BATCH, verbose=False)
    print(f"    train {nd.train_seconds:.0f} s, val error {hist['val_error'][0]:.3e} -> "
          f"{hist['val_error'][-1]:.3e}", flush=True)
    mem = nd.memory
    cmp_ = mem.compare(("mwpm", "correlated", nd), max_shots=TEST_SHOTS,
                       max_errors=10**9, seed=10_000 + 100 * d + int(p * 1e4))
    net_s, feat_s = net_only_seconds(nd)
    out = {"cmp": cmp_, "train_s": nd.train_seconds, "net_us": net_s * 1e6,
           "feat_us": feat_s * 1e6, "hist": hist}
    _cache[key] = out
    return out


def report(d, p, res):
    cmp_ = res["cmp"]
    print(f"\n  d={d} p={p:g}, {cmp_.shots} shared test shots, rounds = {cmp_.rounds}")
    print(f"  {'decoder':<22}{'errors':>8}{'eps/round':>14}{'+/-':>10}{'decode us/shot':>16}")
    for n in NAMES:
        r = cmp_.rates[n]
        print(f"  {LABELS[n]:<22}{r.errors:>8d}{r.eps:>14.3e}{r.eps_se:>10.1e}"
              f"{r.seconds / r.shots * 1e6:>16.1f}")
    for n in ("correlated", "neural"):
        red, se, z = cmp_.reduction(n)
        verdict = "better" if z > 2 else "worse" if z < -2 else "no significant difference"
        print(f"  {n} vs MWPM: failures change {red:+.1%} +/- {se:.1%}, z = {z:+.2f} ({verdict})")
    print(f"  neural: train {res['train_s']:.0f} s | network forward {res['net_us']:.2f} us/shot"
          f" | MWPM feature pass {res['feat_us']:.1f} us/shot")


def main():
    print("=" * 64)
    print("Neural vs matching decoders" + ("  [QUICK]" if QUICK else ""))
    print("=" * 64)

    # ---- 1. Sweep over p ---------------------------------------------------------------
    print("\n[1/3] Logical error vs physical error rate")
    sweep = {}
    for d in SWEEP_D:
        for p in P_GRID:
            res = run_point(d, p)
            sweep[(d, p)] = res
            report(d, p, res)

    # ---- 2. Lambda(5 -> 7) ---------------------------------------------------------------
    print(f"\n[2/3] Lambda({LAMBDA_D[0]} -> {LAMBDA_D[1]}) at p = {P_LAMBDA:g}")
    lam = {}
    for d in LAMBDA_D:
        report(d, P_LAMBDA, run_point(d, P_LAMBDA))
    print(f"\n  {'decoder':<22}{'Lambda':>10}{'+/-':>8}")
    for n in NAMES:
        try:
            est = lambda_pairwise(run_point(LAMBDA_D[0], P_LAMBDA)["cmp"].rates[n],
                                  run_point(LAMBDA_D[1], P_LAMBDA)["cmp"].rates[n])
            lam[n] = est
            print(f"  {LABELS[n]:<22}{est.value:>10.2f}{est.se:>8.2f}")
        except ValueError as exc:
            print(f"  {LABELS[n]:<22}  not estimable ({exc})")
    if "mwpm" in lam and "neural" in lam:
        diff = lam["neural"].value - lam["mwpm"].value
        se = math.hypot(lam["neural"].se, lam["mwpm"].se)
        print(f"  neural minus MWPM: {diff:+.2f} +/- {se:.2f} (unpaired, shared test shots "
              f"make the true uncertainty smaller)")

    # ---- 3. Plot ---------------------------------------------------------------------------
    print("\n[3/3] Figure")
    ncols = len(SWEEP_D) + 1
    fig, axes = plt.subplots(1, ncols, figsize=(4.8 * ncols, 4.2))
    for ax, d in zip(axes, SWEEP_D):
        for n in NAMES:
            ps, eps, se = [], [], []
            for p in P_GRID:
                r = sweep[(d, p)]["cmp"].rates[n]
                if r.errors > 0:
                    ps.append(p), eps.append(r.eps), se.append(r.eps_se)
            ax.errorbar(ps, eps, yerr=se, color=COLOURS[n], marker=MARKS[n], capsize=3,
                        lw=1.5, label=LABELS[n])
        ax.set_xscale("log"), ax.set_yscale("log")
        ax.set_xlabel("physical error rate p")
        ax.set_ylabel("logical error per round")
        ax.set_title(f"d = {d}, same test shots")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.3, which="both")
    ax = axes[-1]
    x = np.arange(len(NAMES))
    vals = [lam[n].value if n in lam else 0.0 for n in NAMES]
    errs = [lam[n].se if n in lam else 0.0 for n in NAMES]
    ax.bar(x, vals, yerr=errs, capsize=4, color=[COLOURS[n] for n in NAMES])
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.set_xticks(x, [LABELS[n].replace(" (", "\n(") for n in NAMES], fontsize=8)
    ax.set_ylabel(rf"$\Lambda$ (d={LAMBDA_D[0]}$\to${LAMBDA_D[1]}, p={P_LAMBDA:g})")
    ax.set_title("Error suppression")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = IMAGES / "qec_neural_decoder.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")

    # ---- Checks --------------------------------------------------------------------------------
    print("\nChecks")
    shots = {c["cmp"].shots for c in _cache.values()}
    check("all three decoders scored on identical shots (per point)",
          all(len({r.shots for r in c["cmp"].rates.values()}) == 1 for c in _cache.values()))
    for d in SWEEP_D:
        for n in NAMES:
            e = [sweep[(d, p)]["cmp"].rates[n].eps for p in P_GRID]
            check(f"{LABELS[n]}: eps rises with p (d={d})", all(a < b for a, b in zip(e, e[1:])),
                  " < ".join(f"{v:.1e}" for v in e))
    z_all = {key: c["cmp"].reduction("neural")[2] for key, c in sweep.items()}
    worst = min(z_all.items(), key=lambda kv: kv[1])
    check("neural no worse than MWPM within 2 sigma at every sweep point (paired z >= -2)",
          worst[1] >= -2.0, f"worst z = {worst[1]:+.2f} at d={worst[0][0]}, p={worst[0][1]:g}")
    for n in NAMES:
        if n in lam:
            check(f"{LABELS[n]}: Lambda(5->7) > 1", lam[n].value > 1.0,
                  f"{lam[n].value:.2f} +/- {lam[n].se:.2f}")
    check("figure written", path.exists())

    print("\nInformational (not checks)")
    for key, c in sorted(_cache.items()):
        red, se, z = c["cmp"].reduction("neural")
        print(f"  d={key[0]} p={key[1]:g}: neural vs MWPM failures {red:+.1%}, z = {z:+.2f}")

    passed = sum(_checks)
    print(f"\n{passed}/{len(_checks)} checks pass")
    return 0 if all(_checks) else 1


if __name__ == "__main__":
    sys.exit(main())