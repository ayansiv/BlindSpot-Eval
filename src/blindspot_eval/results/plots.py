"""Frontier plots: one panel per arm, one line per carrier, over the 0.1-10% FPR region.

Colors follow the carrier, never its rank. Each carrier has a fixed slot in the
reference categorical palette, so C3 is the same color in every figure.
Bootstrap bands are drawn as light fills under 2px lines. Where the controls
can't resolve low FPRs (FPR below 1/n_neg), that region is shaded.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from blindspot_eval.results.frontier import ArmResult  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CARRIER_SLOT = {"C0": 0, "C1": 1, "C2": 2, "C3": 3, "C4": 4, "C5": 5}


def carrier_color(carrier: str, others: list[str]) -> str:
    if carrier in CARRIER_SLOT:
        return SLOTS[CARRIER_SLOT[carrier]]
    extra = sorted(c for c in others if c not in CARRIER_SLOT)
    return SLOTS[6 + extra.index(carrier)] if extra.index(carrier) < 2 else MUTED


def plot_arm(result: ArmResult, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    x = result.fpr_grid
    names = [c.carrier for c in result.carriers]

    floors = [c.fpr_floor for c in result.carriers if c.fpr_floor]
    worst_floor = max(floors) if floors else None
    if worst_floor and worst_floor > x[0]:
        ax.axvspan(x[0], min(worst_floor, x[-1]), color=GRID, alpha=0.6, lw=0, zorder=0)
        ax.text(x[0] * 1.08, 0.02, "unresolved: FPR < 1/n controls", color=MUTED, fontsize=7.5, va="bottom")

    for c in result.carriers:
        if not c.tpr_grid:
            continue
        color = carrier_color(c.carrier, names)
        if c.tpr_lo and c.tpr_hi:
            ax.fill_between(x, c.tpr_lo, c.tpr_hi, color=color, alpha=0.14, lw=0, zorder=1)
        label = f"{c.carrier}  area {c.pauc:.2f}" if c.pauc is not None else c.carrier
        if c.gap_vs_baseline is not None:
            label += f"  gap {c.gap_vs_baseline:+.2f}"
        ax.plot(x, c.tpr_grid, color=color, lw=2, solid_capstyle="round", label=label, zorder=2)

    ax.set_xscale("log")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("False-positive rate on matched benign controls", color=INK_2, fontsize=9)
    ax.set_ylabel("Catch rate before release", color=INK_2, fontsize=9)
    ax.set_xticks([1e-3, 1e-2, 1e-1], ["0.1%", "1%", "10%"])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25%", "50%", "75%", "100%"])
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(True, color=GRID, lw=1, ls="-")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)

    cost = f"{result.mean_tokens_read:,.0f} monitor tokens/trace"
    if result.tokens_per_policy_token:
        cost += f" ({result.tokens_per_policy_token:.1f} per policy token)"
    ax.set_title(f"{result.arm}", loc="left", color=INK, fontsize=11, pad=18)
    ax.text(0, 1.02, cost, transform=ax.transAxes, color=INK_2, fontsize=8)
    leg = ax.legend(loc="lower right", fontsize=8, frameon=False, labelcolor=INK)
    for line in leg.get_lines():
        line.set_linewidth(2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def results_table(results: list[ArmResult]) -> str:
    """Markdown table view of every arm x carrier, the text companion to the plots."""
    rows = ["| Arm | Carrier | n pos | n neg | unverified | area 0.1–10% | 95% CI | gap vs C0 | gap CI | FPR floor |",
            "|---|---|---|---|---|---|---|---|---|---|"]

    def f(v, fmt="{:.3f}"):
        return "—" if v is None else fmt.format(v)

    def ci(v):
        return "—" if v is None else f"[{v[0]:.3f}, {v[1]:.3f}]"

    for r in results:
        for c in r.carriers:
            rows.append(
                f"| {r.arm} | {c.carrier} | {c.n_pos} | {c.n_neg} | {c.n_pos_unverified} | {f(c.pauc)} | "
                f"{ci(c.pauc_ci)} | {f(c.gap_vs_baseline, '{:+.3f}')} | {ci(c.gap_ci)} | {f(c.fpr_floor, '{:.4f}')} |"
            )
    return "\n".join(rows) + "\n"
