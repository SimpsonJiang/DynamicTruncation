"""Shared toy plot colors: blue density shading, no hatching or red/green pairs."""


def apply_style():
    import matplotlib as mpl
    from cycler import cycler

    mpl.rcParams["axes.prop_cycle"] = cycler(color=["#0072B2", "#D55E00", "#7B3294", "#555555", "#E69F00"])
