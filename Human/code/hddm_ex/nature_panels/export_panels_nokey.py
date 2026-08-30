"""Standalone panels a-f without any colour key, at high raster resolution.

Written for the case where the panels are placed in a slide or a poster whose
key is supplied once elsewhere, so repeating it inside every panel is wasted
space.  Only the key is dropped: axes, annotations, aspect ratios and font
sizes are the ones the submission panels use.

Which panels carry a key at all:
  a  none      -- task identity is in each cell's title
  b  figure-level legend under the two violins  -> dropped via show_key=False
  c  none      -- the three slope annotations are drawn in the task colours
  d  in-panel colour key stacked in the upper region -> show_key=False
  e  none      -- as c
  f  as d

Outputs go to figures/nature/no_key/ so they never overwrite the submission
versions, and are written as both PDF and PNG.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --- path repair after reorganisation into _organized/code/ ---------------
# BASE_DIR: original project directory, still the source of data and the
# destination of figures. CODE_DIR: where this script and its sibling
# modules now live.
BASE_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(CODE_DIR))

from manuscript_common import figure_style as fs  # noqa: E402
import panel_a_ddm_traces as pa  # noqa: E402
import panel_b_boundary_ndt as pb  # noqa: E402
import panel_ce_feature_modulation as pce  # noqa: E402
import panel_df_posterior_coefs as pdf_  # noqa: E402

# Matches the composite: at 180 mm the 6 pt tick labels are ~42 px tall at
# 500 dpi, where the digits' anti-aliased stems read as soft.
RASTER_DPI = 900

# --- figure output redirected to _organized/figures/ -----------------------
FIG_ROOT = BASE_DIR.parents[1] / "figures" / "hddm_ex" / "hddm_results_4chains_2000samples"
OUT_DIR = FIG_ROOT / "nature/no_key"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig, name: str, mm: float) -> None:
    """Scale to an exact printed width, then write both formats."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / name
    # save_at_width iterates the bbox to land on the requested width; it writes
    # the PDF and a PNG at the rc dpi, which RASTER_DPI has already set.
    w, h = fs.save_at_width(fig, stem, mm=mm)
    px_w = int(round(w / 25.4 * RASTER_DPI))
    ov = fs.check_text_overlaps(fig)
    print(f"{name:36s} {w:6.2f} x {h:6.2f} mm | ~{px_w} px wide | "
          f"overlaps {len(ov)}")
    plt.close(fig)


def main() -> None:
    fs.apply(mpl, dpi=RASTER_DPI)

    # Each panel is built through its own module's standalone branch (fig=None)
    # so it gets the figure size that module was designed around.  Constructing
    # the figure here instead changed the axes width and pushed one slope label
    # into the neighbouring cell's y-axis label.

    # a -- no key of its own; task identity is in each cell's title, set on one
    # line at standalone cell width.
    fig, axes = pa.build(wrap_title=False)
    fs.letter(axes[0], "a", dx=-0.06, dy=0.02)
    fig.subplots_adjust(wspace=0.10)
    _save(fig, "panel_a_ddm_traces_nokey", 180)

    # b -- show_key controls the legend under the two violins.
    fig, axes = pb.build(show_key=False)
    fs.letter(axes[0], "b")
    _save(fig, "panel_b_boundary_ndt_nokey", 114)

    # c, e -- already key-free; the slope annotations carry the colours.
    for feature in ("Visual_Z", "Physical_Z"):
        cfg = pce.FEATURES[feature]
        fig, axes = pce.build(feature)
        fs.letter(axes[0], cfg["letter"])
        _save(fig, f"panel_{cfg['letter']}_{feature.lower()}_modulation_nokey", 180)

    # d, f -- the stacked in-panel key is dropped; head-room shrinks with it.
    for feature in pdf_.FEATURES:
        cfg = pdf_.FEATURES[feature]
        fig, ax, _ = pdf_.build(feature, show_key=False)
        fs.letter(ax, cfg["letter"])
        _save(fig, f"panel_{cfg['letter']}_{feature}_posterior_coefs_nokey", 88)


if __name__ == "__main__":
    main()
