"""Lattice surgery CNOT error vs distance."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from LatticeSurgery import LatticeSurgery

P = 0.002
SHOTS = 20000
DS = [3, 5]
errors, ses = [], []

for d in DS:
    r = LatticeSurgery(d, P).sample_cnot(n_shots=SHOTS, seed=42)
    errors.append(r.error_rate)
    ses.append(r.error_rate_se)
    print(f"  d={d}: CNOT error {r.error_rate:.3e} +/- {r.error_rate_se:.1e}")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.errorbar(DS, errors, yerr=ses, marker="o", markersize=9, capsize=4,
            color="#3b6ea5", lw=1.8, label="logical CNOT")
ax.set_yscale("log")
ax.set_xticks(DS)
ax.set_xlabel("code distance d")
ax.set_ylabel("logical CNOT error rate")
ax.set_title(f"Fault-tolerant CNOT via lattice surgery (p = {P})")
ax.grid(alpha=0.3, which="both")
ax.legend(frameon=False)

if len(DS) >= 2 and errors[0] > 0 and errors[1] > 0:
    ratio = errors[0] / errors[1]
    ax.text(0.05, 0.05, f"ratio d=3 -> 5: {ratio:.2f}x",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "..", "docs", "images",
                   "qec_lattice_surgery.png")
fig.savefig(out, dpi=150)
print(f"saved {out}")
