"""Shared matplotlib style and color palette for publication figures."""

import matplotlib as mpl

# ── Categorical palette (10 entries; indexed by other figure scripts) ──────────
COLORS: list[str] = [
    "#BF3E3E",  # [0] muted crimson
    "#C86820",  # [1] muted sienna
    "#C49A28",  # [2] muted amber
    "#D4A85A",  # [3] tan
    "#7A9E30",  # [4] olive green
    "#2D7349",  # [5] green
    "#23697A",  # [6] teal 
    "#2B5B7D",  # [7] deep blue 
    "#3A4E88",  # [8] slate blue
    "#1C3E6E",  # [9] dark navy
]

# ── Semantic colors for Figure 1 (and reuse in other figures) ─────────────────
C = {
    "plasma_lpa": "#224D6D",  
    "data": "#767575",  
    "er_eff": "#296B45",
    "assem_eff": "#2C7A82",
    "er_load": "#8C4A18",
}


def apply_style() -> None:
    """Apply clean, journal-quality matplotlib style (no ggplot)."""
    mpl.rcParams.update(
        {
            # Font — Arial preferred; fall back to Helvetica or DejaVu Sans
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "#CCCCCC",
            "legend.handlelength": 1.5,
            "legend.handletextpad": 0.5,
            # Lines / markers
            "lines.linewidth": 1.5,
            "lines.markersize": 4.5,
            "patch.linewidth": 0.6,
            # Axes
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#444444",
            "axes.labelcolor": "#222222",
            "axes.grid": False,
            "grid.color": "#E4E4E4",
            "grid.linestyle": "-",
            "grid.linewidth": 0.4,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            # Ticks
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.color": "#444444",
            "ytick.color": "#444444",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            # Save
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.facecolor": "white",
            "savefig.transparent": True,
        }
    )


def panel_label(
    ax, label: str, x: float = -0.13, y: float = 1.05, size: int = 9
) -> None:
    """Add a bold uppercase panel label (e.g. 'A') to the top-left of axes."""
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=size,
        fontweight="bold",
        va="top",
        ha="left",
        color="grey",
    )
