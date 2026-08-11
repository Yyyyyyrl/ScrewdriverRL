#!/usr/bin/env python3
"""Render the audit summary with the mimic legend above its bars."""

from __future__ import annotations

from matplotlib.axes import Axes

from tools import plot_g20_urdf_local_q_audit_summary as base


def main() -> int:
    original = Axes.legend

    def legend(self, *args, **kwargs):
        if kwargs.get("loc") == "lower right" and kwargs.get("fontsize") == 9:
            kwargs["loc"] = "upper right"
            kwargs["bbox_to_anchor"] = (1.0, 1.08)
            kwargs["ncol"] = 2
            kwargs["fontsize"] = 8.5
        return original(self, *args, **kwargs)

    Axes.legend = legend
    try:
        return base.main()
    finally:
        Axes.legend = original


if __name__ == "__main__":
    raise SystemExit(main())
