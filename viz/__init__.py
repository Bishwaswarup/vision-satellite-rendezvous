"""
viz — shared figure styling for the rendezvous simulator.

Everything here produces PRINT-SAFE MONOCHROME output: series are separated by
grey level, line style and marker rather than by hue, so a figure survives a
greyscale journal printer and remains readable to colour-blind readers.
"""

from .style import (
    SERIES, GREY, apply_style, series_kw, style_ax, save_fig,
    IMAGE_CMAP, add_panel_label,
)

__all__ = [
    'SERIES', 'GREY', 'apply_style', 'series_kw', 'style_ax', 'save_fig',
    'IMAGE_CMAP', 'add_panel_label',
]
