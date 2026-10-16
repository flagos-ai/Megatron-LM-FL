"""Paper vs demo figure styling for MegaLens analyzers.

Switch ``PAPER_MODE`` to ``False`` to restore the wide-screen demo defaults.

When ``PAPER_MODE`` is ``True``:
  - figsizes are sized for IEEE/ACM 2-column layout
  - font sizes are tuned for ~9pt body text
  - dpi=300 PDF rendering
  - line/marker sizes are reduced

Per-analyzer code should import the symbols below in preference to hardcoding.
"""

PAPER_MODE: bool = True

if PAPER_MODE:
    # Analyzer reports use multi-panel figures with titles, legends and dense
    # rank labels.  The previous IEEE-column sizes (7 x 2.4/3.6 inches) were
    # too small for those layouts and caused subplots, legends, and tick labels
    # to be clipped in exported PDFs.
    FIG_W_SINGLE: float = 5.4
    FIG_W_DOUBLE: float = 13.5
    FIG_H_SHORT: float = 3.2
    FIG_H: float = 4.8
    FIG_H_TALL: float = 8.6
    FONT_SIZE_TITLE: int = 11
    FONT_SIZE_SUPTITLE: int = 13
    FONT_SIZE_LABEL: int = 10
    FONT_SIZE_TICK: int = 9
    FONT_SIZE_LEGEND: int = 9
    FONT_SIZE_ANNOT: int = 8
    LINE_WIDTH: float = 1.2
    MARKER_SIZE: float = 3.5
    BAR_EDGE_WIDTH: float = 0.5
    DPI: int = 300
    HATCH_LINEWIDTH: float = 0.5
else:
    FIG_W_SINGLE = 12.0
    FIG_W_DOUBLE = 22.0
    FIG_H_SHORT = 5.0
    FIG_H = 7.0
    FIG_H_TALL = 13.0
    FONT_SIZE_TITLE = 14
    FONT_SIZE_SUPTITLE = 22
    FONT_SIZE_LABEL = 12
    FONT_SIZE_TICK = 11
    FONT_SIZE_LEGEND = 10
    FONT_SIZE_ANNOT = 10
    LINE_WIDTH = 2.0
    MARKER_SIZE = 5.0
    BAR_EDGE_WIDTH = 0.8
    DPI = 200
    HATCH_LINEWIDTH = 1.0


def apply_global_rcparams() -> None:
    """Push values into matplotlib rcParams.

    Call once at the top of each analyzer's plotting section so individual
    plot calls only need to pass figsize. The other text/legend/tick settings
    are pulled from rcParams.
    """
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.size": FONT_SIZE_LABEL,
            "axes.titlesize": FONT_SIZE_TITLE,
            "axes.labelsize": FONT_SIZE_LABEL,
            "xtick.labelsize": FONT_SIZE_TICK,
            "ytick.labelsize": FONT_SIZE_TICK,
            "legend.fontsize": FONT_SIZE_LEGEND,
            "figure.titlesize": FONT_SIZE_SUPTITLE,
            "lines.linewidth": LINE_WIDTH,
            "lines.markersize": MARKER_SIZE,
            "patch.linewidth": BAR_EDGE_WIDTH,
            "hatch.linewidth": HATCH_LINEWIDTH,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.12,
            "figure.autolayout": False,
            "pdf.fonttype": 42,  # embed TrueType (editable in Illustrator)
            "ps.fonttype": 42,
        }
    )
