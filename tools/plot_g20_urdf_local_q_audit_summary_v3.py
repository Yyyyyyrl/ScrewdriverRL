#!/usr/bin/env python3
"""Render the audit summary with an inline mimic color key."""

from __future__ import annotations

from matplotlib.axes import Axes

from tools import plot_g20_urdf_local_q_audit_summary as base


def main() -> int:
    original_legend = Axes.legend
    original_xlabel = Axes.set_xlabel

    def legend(self, *args, **kwargs):
        if kwargs.get("loc") == "lower right" and kwargs.get("fontsize") == 9:
            return None
        return original_legend(self, *args, **kwargs)

    def set_xlabel(self, xlabel, *args, **kwargs):
        if xlabel == "Passive-joint RMSE (degrees)":
            xlabel = (
                "Passive-joint RMSE (degrees)"
                "  ·  orange = OG  ·  green = measured"
            )
        return original_xlabel(self, xlabel, *args, **kwargs)

    Axes.legend = legend
    Axes.set_xlabel = set_xlabel
    try:
        return base.main()
    finally:
        Axes.legend = original_legend
        Axes.set_xlabel = original_xlabel


if __name__ == "__main__":
    raise SystemExit(main())
