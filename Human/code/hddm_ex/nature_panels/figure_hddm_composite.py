"""Composite main figure: panels a-f on one page.

Layout rationale
----------------
Three rows:

    row 1   a   b        trajectories (three tasks) | a and v posteriors
    row 2   c   d        Visual_Z   behavioural modulation | drift coefficient
    row 3   e   f        Physical_Z behavioural modulation | drift coefficient

Reading left-to-right then top-to-bottom gives a, b, c, d, e, f, so the letters
are identical to the standalone panel files and manuscript cross-references
hold for both versions.  Six separate rows would need ~273 mm of height against
a 247 mm page limit, which is why every row carries two panel groups.

The page is set to a 4:3 canvas.  Row heights are still derived from the
standalone panels' axes aspect ratios, so the extra height a 4:3 page provides
over the content is left as white space at the foot rather than used to stretch
the axes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --- path repair after reorganisation into _organized/code/ ---------------
# BASE_DIR: original project directory, still the source of data and the
# destination of figures. CODE_DIR: where this script and its sibling
# modules now live.
BASE_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parents[1]
# --- figure output redirected to _organized/figures/ -----------------------
FIG_ROOT = BASE_DIR.parents[1] / "figures" / "hddm_ex" / "hddm_results_4chains_2000samples"
OUT_DIR = FIG_ROOT / "nature"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(CODE_DIR / "manuscript_common"))
sys.path.insert(0, str(CODE_DIR / "nature_panels"))
import figure_style as fs  # noqa: E402

import panel_a_ddm_traces as pa  # noqa: E402
import panel_b_boundary_ndt as pb  # noqa: E402
import panel_ce_feature_modulation as pce  # noqa: E402
import panel_df_posterior_coefs as pdf_  # noqa: E402

# Row heights in inches, derived from the standalone panels' axes aspect
# ratios rather than chosen to fill the page.
#
# Each panel's aspect ratio is a design decision made in its own script: the
# trajectory and violin axes are near-square (1.14 and 1.06), the behavioural
# scatters slightly wide (1.16), the coefficient densities wider still (1.74).
# Placing three cells side by side in half a page width makes each cell narrow,
# so a row height chosen to consume leftover page height stretches every axes
# vertically -- at a height chosen to fill the page the measured aspects were
# 0.36 to 0.58, i.e. every panel distorted by a factor of 2-3.5 against its
# standalone shape.
#
# So the height follows from the width: row_h = axes_width / target_aspect +
# non-axes overhead (titles, tick labels, the figure-level key).  Unused page
# height is left as white space; filling the page is not a goal that outranks
# keeping each panel the shape it was designed to be.
ROW_H = [1.2605, 1.6520, 1.6672]
# Canvas width, fixed at the double-column limit.  This leaves a's three cells
# ~1.0 in wide against a single-line title of 1.64 in, so the composite wraps
# those titles over three lines (see panel_a_ddm_traces.draw_condition);
# widening the page instead would put the figure over the 180 mm limit and make
# every font size shrink below the 5 pt floor when scaled back at typesetting.
FIG_W = 7.09  # 180 mm, double-column width
# Page height follows the content: the rows are sized to hold each panel's
# aspect ratio, and padding the canvas out to a fixed ratio would only add a
# white band under the last row.  Set to None for content height.
PAGE_ASPECT = None

# Row 1 split between a's three cells and b's two.  b's violins are near-square
# and carry a fixed-width y label plus tick column, so the seam does not align
# with the c/d seam below.
TOP_SPLIT = 0.4919
A_WSPACE = 0.06
B_WSPACE = 0.22

# The drift-coefficient cell is wider than a single behavioural cell: it holds
# three long task names stacked in the upper region as a colour key.
WIDTH_RATIOS = [1.0, 1.0, 1.0, 1.50]

# Gap between the three behavioural cells, as a fraction of a cell's width.
# Each gap needs only enough room for the next cell's y tick labels and its
# rotated y label; measured at 0.04 the nearest ink of adjacent cells clears by
# 2.5 mm with no text overlaps, against 7.1 mm at 0.10.  Narrowing the gap
# widens the axes, so the row heights above were re-solved at this value to
# hold the panels' aspect ratios.
BEH_WSPACE = 0.04

# Height reserved at the foot of the page for the three-task colour key.
KEY_BAND_H = 0.26

# Raster resolution for the PNG.  See main().
RASTER_DPI = 900


def build():
    content_h = sum(ROW_H)
    page_h = content_h if PAGE_ASPECT is None else FIG_W / PAGE_ASPECT
    # Any surplus page height is parked in a trailing spacer row rather than
    # handed to the axes, which would stretch every panel away from its
    # standalone shape.
    spare = max(0.0, page_h - content_h)
    fig = plt.figure(figsize=(FIG_W, max(page_h, content_h + KEY_BAND_H)),
                     layout="constrained")
    fig.get_layout_engine().set(w_pad=0.045, h_pad=0.035,
                                wspace=0.055, hspace=0.055)
    # A band at the foot for the figure-level key.  Reserved as a spacer row so
    # the key never lands on the last row's axis labels.
    heights = list(ROW_H) + [max(spare, KEY_BAND_H)]
    gs = fig.add_gridspec(len(heights), 1, height_ratios=heights)

    # --- row 1: trajectories (a) beside the parameter posteriors (b) -------
    # a's cells carry no y ticks, so their gaps only need to clear the ~0.15 in
    # right overhang of the trajectory labels; b's do carry ticks and a y
    # label, so its pair needs the wider gap.
    top = gs[0].subgridspec(1, 2, width_ratios=[TOP_SPLIT, 1.0 - TOP_SPLIT],
                            wspace=0.16)
    sub_a = top[0, 0].subgridspec(1, 3, wspace=A_WSPACE)
    axes_a = [fig.add_subplot(sub_a[0, k]) for k in range(3)]
    pa.build(fig=fig, axes=axes_a)
    fs.letter(axes_a[0], "a", dx=-0.06, dy=0.02)

    sub_b = top[0, 1].subgridspec(1, 2, wspace=B_WSPACE)
    axes_b = [fig.add_subplot(sub_b[0, k]) for k in range(2)]
    pb.build(fig=fig, axes=axes_b)
    fs.letter(axes_b[0], "b", dx=-0.42, dy=0.10)

    # --- rows 2-3: one feature per row ------------------------------------
    row_specs = [
        (1, "Visual_Z", "visual_z", "c", "d"),
        (2, "Physical_Z", "physical_z", "e", "f"),
    ]
    handles = {"traces": axes_a, "params": axes_b}
    for row, feature, feat_lower, lb, lc in row_specs:
        sub = gs[row].subgridspec(1, 4, width_ratios=WIDTH_RATIOS,
                                  wspace=BEH_WSPACE)
        axes_beh = [fig.add_subplot(sub[0, k]) for k in range(3)]
        ax_coef = fig.add_subplot(sub[0, 3])
        pce.build(feature, fig=fig, axes=axes_beh, force_in_axes=True)
        pdf_.build(feat_lower, fig=fig, ax=ax_coef, show_key=False)
        # The letter sits above the axes, level with the topmost y tick label,
        # and the tick column reaches to within 24 px of the page edge -- so
        # neither moving it left nor lifting it resolves the overlap.  Pruning
        # the uppermost tick frees that slot without changing the data range or
        # the remaining tick spacing.
        for ax_b in axes_beh:
            ax_b.yaxis.set_major_locator(
                mticker.MaxNLocator(nbins="auto", prune="upper"))
        fs.letter(axes_beh[0], lb, dx=-0.26, dy=0.10)
        # The coefficient cell sits directly right of a behavioural cell whose
        # outermost x tick label is centred on its tick and so overhangs into
        # the gap, so this letter is placed just inside its own axes instead of
        # outside to the left.
        fs.letter(ax_coef, lc, dx=0.02, dy=0.03)
        handles[feature] = (axes_beh, ax_coef)

    # One key for the whole figure: the three tasks appear in every panel, so
    # keying them per panel would restate the same three names four times and
    # cost each row its height.  b's two violins fill row 1 to within 1 mm of
    # the right edge, so unlike the four-row layout there is no cell beside it
    # to hold the key; it goes under the last row instead.
    #
    # Anchored to the figure's lower edge with a reserved band of page height,
    # not to the last row's axes: with the page height set by content there is
    # no surplus below the last row, and an axes-anchored key lands on that
    # row's x-axis labels.  The band is added to the figure height rather than
    # taken from the rows, so the panels keep their aspect ratios.
    key = [mpatches.Patch(facecolor=fs.TASK_FILL[t], edgecolor=fs.TASK_COL[t],
                          linewidth=0.9, label=t) for t in fs.TASK_ORDER]
    fig.legend(handles=key, loc="lower center", ncol=3, frameon=False,
               fontsize=fs.SECONDARY, handlelength=1.0, handleheight=1.0,
               handletextpad=0.5, columnspacing=1.6, borderaxespad=0.3)

    return fig, handles


def main():
    # 900 dpi rather than the 500 used for the standalone panels.  The
    # composite is the figure a reader zooms into, and at 180 mm wide 500 dpi
    # puts the 6 pt tick labels at ~42 px tall, where the anti-aliased stems of
    # the digits are only 1-2 px and read as soft.  900 dpi is also what the
    # journal asks for line-art raster.
    fs.apply(mpl, dpi=RASTER_DPI)
    fig, _ = build()
    stem = OUT_DIR / "figure_hddm_composite"
    stem.parent.mkdir(parents=True, exist_ok=True)
    # Saved at the canvas size itself, not through save_within: the row heights
    # already hold each panel's aspect ratio, and bbox_inches="tight" would
    # crop the band reserved for the key.
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=RASTER_DPI)
    w, h = fig.get_figwidth() * 25.4, fig.get_figheight() * 25.4
    ov = fs.check_text_overlaps(fig)
    ink = fs.check_ink_under_text(fig)
    px = (int(round(fig.get_figwidth() * RASTER_DPI)),
          int(round(fig.get_figheight() * RASTER_DPI)))
    print(f"composite: {w:.2f} x {h:.2f} mm | png {px[0]} x {px[1]} px "
          f"@ {RASTER_DPI} dpi")
    print(f"text overlaps: {len(ov)}")
    print(f"foreign ink under slope labels: {len(ink)} {ink[:4]}")
    for pair in ov[:8]:
        print("   ", pair)
    plt.close(fig)


if __name__ == "__main__":
    main()
