"""The shared look of the charts in figures/. Each notebook draws its own chart with these."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path as Outline

FIGURES = Path(__file__).resolve().parent / "figures"

SURFACE = "#fcfcfb"                                          # background
INK, INK_SOFT, INK_MUTED = "#0b0b0b", "#52514e", "#898781"   # titles, labels, axis numbers
GRID, AXIS = "#e1e0d9", "#c3c2b7"                            # thin lines
BLUE, LIGHT_BLUE = "#2a78d6", "#86b6ef"

WIDTH, ROW, TOP, BOTTOM, RIGHT = 8, 0.36, 1.05, 0.6, 0.7     # layout, in inches


def new_chart(title, subtitle, labels, x_label, left=2.4, bold=()):
    """An empty chart with one row per label, top to bottom.

    `left` is the room for the labels, in inches. `bold` lists the rows that are group headings.
    """
    plt.rcParams["font.family"] = ["Segoe UI", "Arial", "DejaVu Sans"]
    height = TOP + ROW * len(labels) + BOTTOM
    fig = plt.figure(figsize=(WIDTH, height), facecolor=SURFACE)
    ax = fig.add_axes([left / WIDTH, BOTTOM / height, (WIDTH - left - RIGHT) / WIDTH, ROW * len(labels) / height],
                      facecolor=SURFACE)
    fig.text(0.2 / WIDTH, 1 - 0.18 / height, title, fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.2 / WIDTH, 1 - 0.48 / height, subtitle, fontsize=10, color=INK_SOFT, va="top")

    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_yticks(range(len(labels)), labels, fontsize=10, color=INK_SOFT)
    for row in bold:
        ax.get_yticklabels()[row].set(color=INK, fontweight="bold")
    ax.set_xlabel(x_label, fontsize=9, color=INK_MUTED, labelpad=6)
    ax.tick_params(axis="x", length=0, labelsize=9, colors=INK_MUTED)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[:].set_visible(False)
    return fig, ax


def bars(ax, values, labels, x_max, color=BLUE, thickness=14, corner=3):
    """Bars from zero, one per row: square at the baseline, rounded at the far end, with a label at the tip.

    `thickness` and `corner` are in points (1/72 inch).
    """
    ax.set_xlim(0, x_max)
    x_per_point = x_max / (ax.bbox.width / ax.figure.dpi * 72)        # data units per point, across
    y_per_point = len(values) / (ax.bbox.height / ax.figure.dpi * 72)  # and down
    half, rx, ry = thickness / 2 * y_per_point, corner * x_per_point, corner * y_per_point
    for row, (value, label) in enumerate(zip(values, labels)):
        outline = Outline(
            [(0, row - half), (value - rx, row - half), (value, row - half), (value, row - half + ry),
             (value, row + half - ry), (value, row + half), (value - rx, row + half), (0, row + half), (0, row - half)],
            [Outline.MOVETO, Outline.LINETO, Outline.CURVE3, Outline.CURVE3,
             Outline.LINETO, Outline.CURVE3, Outline.CURVE3, Outline.LINETO, Outline.CLOSEPOLY])
        ax.add_patch(PathPatch(outline, facecolor=color, edgecolor="none"))
        ax.text(value + 6 * x_per_point, row, label, va="center", fontsize=10, fontweight="bold", color=INK)
    ax.axvline(0, color=AXIS, linewidth=1)


def dots(ax, values, color=BLUE, **options):
    """A dot per row, with a thin ring in the background colour so touching marks stay readable."""
    return ax.scatter(values, range(len(values)), s=60, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3, **options)


def reference_line(ax, x, label):
    """A vertical line across the chart, with its name above the top row."""
    ax.axvline(x, color=INK_MUTED, linewidth=1)
    ax.text(x, -0.62, label, fontsize=9, color=INK_SOFT, ha="center", va="bottom")


def save(fig, name):
    """Save the chart into figures/."""
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name, dpi=200, facecolor=SURFACE)
