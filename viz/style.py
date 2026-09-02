"""
style.py
========
Print-safe monochrome figure style.

Why monochrome
--------------
Journal figures are frequently printed in greyscale, and a reader with a
colour-vision deficiency sees the same thing on screen.  Encoding a series in
hue alone therefore risks losing the distinction entirely.  Every series here
is separated redundantly, by three channels at once:

    grey level  +  line style  +  marker

so any one of them is enough to tell two curves apart.  Six levels are
provided; a figure needing more than six series almost always wants to be
several figures.

Usage
-----
    from viz import apply_style, series_kw, style_ax, save_fig

    apply_style()                      # once, before creating any figure
    fig, ax = plt.subplots()
    ax.plot(t, a, label='MEKF', **series_kw(0))
    ax.plot(t, b, label='UKF',  **series_kw(1))
    style_ax(ax, xlabel='Time [s]', ylabel='Position error [m]')
    save_fig(fig, 'outputs/fig.png')
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt


# ── Palette ───────────────────────────────────────────────────────────────────
# Greys only.  The levels are spaced so that adjacent series differ by at least
# ~0.2 in luminance, which stays legible after a 300 dpi greyscale print.
GREY = {
    'ink'    : '#000000',
    'dark'   : '#262626',
    'mid'    : '#595959',
    'light'  : '#8C8C8C',
    'faint'  : '#BFBFBF',
    'grid'   : '#D9D9D9',
    'panel'  : '#FAFAFA',
    'page'   : '#FFFFFF',
}

# Six redundantly-encoded series.  Dash patterns are given explicitly so they
# render identically across matplotlib versions.
SERIES = (
    dict(color=GREY['ink'],   linestyle='-',                     marker='o'),
    dict(color=GREY['mid'],   linestyle=(0, (5, 2)),             marker='s'),
    dict(color=GREY['dark'],  linestyle=(0, (1, 1.4)),           marker='^'),
    dict(color=GREY['light'], linestyle=(0, (6, 1.5, 1, 1.5)),   marker='D'),
    dict(color=GREY['dark'],  linestyle=(0, (3, 1, 1, 1, 1, 1)), marker='v'),
    dict(color=GREY['light'], linestyle=(0, (2, 2)),             marker='x'),
)

# Greyscale colour map for images, depth maps and heat maps.
IMAGE_CMAP = 'gray'

# Hatches for bar charts and filled regions, which cannot rely on grey alone
# when they sit side by side.
HATCHES = ('', '///', '...', 'xxx', '\\\\\\', '+++')


def apply_style(base_font: float = 10.0) -> None:
    """
    Install the monochrome style globally.

    Call once before creating figures.  Safe to call more than once.
    """
    matplotlib.rcParams.update({
        # Page
        'figure.facecolor'   : GREY['page'],
        'savefig.facecolor'  : GREY['page'],
        'axes.facecolor'     : GREY['panel'],
        'savefig.bbox'       : 'tight',
        'savefig.dpi'        : 300,
        'figure.dpi'         : 110,

        # Ink
        'text.color'         : GREY['ink'],
        'axes.labelcolor'    : GREY['ink'],
        'axes.edgecolor'     : GREY['mid'],
        'xtick.color'        : GREY['dark'],
        'ytick.color'        : GREY['dark'],
        'axes.linewidth'     : 0.8,

        # Type — a serif face matches most journal body text
        'font.family'        : 'serif',
        'font.serif'         : ['DejaVu Serif', 'Times New Roman', 'serif'],
        'font.size'          : base_font,
        'axes.titlesize'     : base_font + 1,
        'axes.labelsize'     : base_font,
        'legend.fontsize'    : base_font - 1,
        'xtick.labelsize'    : base_font - 1,
        'ytick.labelsize'    : base_font - 1,
        'mathtext.fontset'   : 'dejavuserif',

        # Grid
        'axes.grid'          : True,
        'grid.color'         : GREY['grid'],
        'grid.linewidth'     : 0.6,
        'grid.alpha'         : 1.0,
        'axes.axisbelow'     : True,

        # Lines
        'lines.linewidth'    : 1.4,
        'lines.markersize'   : 4.5,
        'lines.markeredgewidth': 0.9,

        # Legend
        'legend.frameon'     : True,
        'legend.framealpha'  : 1.0,
        'legend.edgecolor'   : GREY['light'],
        'legend.facecolor'   : GREY['page'],
        'legend.borderpad'   : 0.5,

        # Images default to greyscale
        'image.cmap'         : IMAGE_CMAP,

        # No coloured property cycle — series_kw() is the intended route,
        # but anything that ignores it still comes out in grey.
        'axes.prop_cycle'    : matplotlib.cycler(
            color=[s['color'] for s in SERIES]),
    })


def series_kw(i: int, *, marker: bool = False, **overrides) -> dict:
    """
    Plot keyword arguments for series `i`, cycling if `i` exceeds the palette.

    Parameters
    ----------
    i        : series index
    marker   : include the marker (default False — markers are for sparse
               series and scatter-like plots, and clutter dense time histories)
    overrides: any matplotlib keyword to override

    Returns
    -------
    dict suitable for ``ax.plot(..., **series_kw(i))``
    """
    spec = dict(SERIES[i % len(SERIES)])
    if not marker:
        spec.pop('marker', None)
    else:
        spec.setdefault('markerfacecolor', GREY['page'])
        spec.setdefault('markeredgecolor', spec['color'])
    spec.update(overrides)
    return spec


def hatch(i: int) -> str:
    """Hatch pattern for filled region / bar `i`."""
    return HATCHES[i % len(HATCHES)]


def style_ax(ax, *, xlabel: str = None, ylabel: str = None,
             title: str = None, legend: bool = False,
             legend_loc: str = 'best') -> None:
    """Apply the shared axis treatment: labels, spines, grid, optional legend."""
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, pad=6)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.grid(True)
    if legend:
        ax.legend(loc=legend_loc)


def add_panel_label(ax, text: str, *, dx: float = -0.10, dy: float = 1.04):
    """Put a bold panel label — (a), (b), … — outside the top-left corner."""
    ax.text(dx, dy, text, transform=ax.transAxes,
            fontsize=matplotlib.rcParams['axes.titlesize'],
            fontweight='bold', va='bottom', ha='left', color=GREY['ink'])


def save_fig(fig, path, *, dpi: int = 300, close: bool = True) -> str:
    """Save at print resolution on a white page, and report the path."""
    from pathlib import Path
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight',
                facecolor=GREY['page'], edgecolor='none')
    if close:
        plt.close(fig)
    return path
