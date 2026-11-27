"""``python3 -m analysis COMMAND``: the paper's figures from the runs in ``results/``.

  derive                 shared data for several figures (NumPy only; seconds)
  activity               the spikes of Fig. 2a-c: a forward replay in NEST (seconds)
  figure NAME [NAME..]   one or more figures by name (``list`` shows them), or ``all``
  architecture           the schematic of the circuit (Supp. Figs S1-S2; SVG, then PDF if Chrome is available)
  all                    derive, activity, every figure and the schematic, in that order
  list                   every figure: its name, output file and what it needs first

Run inside the model image, e.g. ``docker/run.bash nest -- 'python3 -B -m analysis all'``. Outputs go to
``analysis/figures/`` and ``analysis/source_data/`` (see analysis/README.md).
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Tuple

#: diagnostic supplementary figures: name -> (module, output file, needs)
DIAGNOSTICS: Dict[str, Tuple[str, str, str]] = {
    "supp_forward": ("fig2_forward", "supp_forward", "-"),
    "supp_observer": ("fig3_estimation", "supp_observer", "-"),
    "supp_observer_pole": ("extended_observer", "supp_observer_pole", "-"),
    "supp_screening": ("extended_screening", "supp_screening", "-"),
}


def figure_table() -> List[Tuple[str, str, str]]:
    from .figures import FIGURE_MAKERS

    rows = [(name, output, needs) for name, (_, output, needs) in FIGURE_MAKERS.items()]
    rows += [(name, output, needs) for name, (_, output, needs) in DIAGNOSTICS.items()]
    return rows


def run_figures(names: List[str]) -> None:
    import importlib

    from . import figures

    known = set(figures.FIGURE_MAKERS) | set(DIAGNOSTICS)
    if names == ["all"]:
        names = list(figures.FIGURE_MAKERS) + list(DIAGNOSTICS)
    unknown = [n for n in names if n not in known]
    if unknown:
        raise SystemExit("unknown figure(s): %s; python3 -m analysis list shows them" % ", ".join(unknown))
    for name in names:
        print("figure", name, flush=True)
        if name in figures.FIGURE_MAKERS:
            figures.FIGURE_MAKERS[name][0]()
        else:
            importlib.import_module(".diagnostics." + DIAGNOSTICS[name][0], __package__).draw()
    figures.finish()
    print("figures written to analysis/figures/, panel data to analysis/source_data/plotexports/")


def architecture() -> None:
    from .architecture import draw_figure1_b_final, render_figure1_b_final

    draw_figure1_b_final.make()
    try:
        render_figure1_b_final.main()
    except SystemExit as stop:
        print(stop)


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m analysis", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("derive", "activity", "figure", "architecture", "all", "list"))
    parser.add_argument("names", nargs="*", help="figure names for 'figure' (or 'all')")
    args = parser.parse_args(argv)
    if args.command == "list":
        print("%-24s %-28s %s" % ("name", "output (analysis/figures/)", "run first"))
        for name, output, needs in figure_table():
            print("%-24s %-28s %s" % (name, output + ".pdf", needs))
        return 0
    if args.command == "derive":
        from . import derive
        derive.main()
    elif args.command == "activity":
        from . import activity
        activity.main()
    elif args.command == "figure":
        if not args.names:
            parser.error("name one or more figures, or 'all'")
        run_figures(args.names)
    elif args.command == "architecture":
        architecture()
    elif args.command == "all":
        from . import activity, derive
        derive.main()
        activity.main()
        run_figures(["all"])
        architecture()
    return 0


if __name__ == "__main__":
    sys.exit(main())
