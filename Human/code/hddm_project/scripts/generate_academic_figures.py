#!/usr/bin/env python
"""Generate publication-style figure variants without overwriting originals."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

from academic_figure_style import (
    apply_academic_style,
    install_academic_savefig_patch,
    update_module_style,
)


DEFAULT_RUNS = [
    ("plot_rt_condition_sample_style", ["--input", "all_dat_merged.csv"], True),
    ("plot_group_posteriors", [], True),
    ("plot_hddm_rtcritical_group_posteriors_combined", [], True),
    ("plot_visual_z_posterior_distributions", [], True),
    ("plot_physical_z_posterior_distributions", [], True),
    ("plot_visual_z_double_dissociation", ["--input", "all_dat_merged.csv", "--hddm-style", "forest"], True),
    ("plot_visual_z_double_dissociation", ["--input", "all_dat_merged.csv", "--hddm-style", "halfeye"], True),
    ("plot_visual_z_double_dissociation", ["--input", "all_dat_merged.csv", "--hddm-style", "academic"], True),
    ("plot_physical_z_reversal", ["--input", "all_dat_merged.csv"], True),
    ("plot_rt_hddm_taskwise_sets", ["--input", "all_dat_merged.csv"], True),
    ("plot_hddm_taskwise_sets", ["--input", "all_dat_merged.csv"], True),
    ("plot_hddm_taskwise_result_figures", ["--input", "all_dat_merged.csv"], True),
]


WRAPPED_LABEL_MODULES = {
    "plot_group_posteriors",
}


def ensure_link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src)


def prepare_output_root(source_output: Path, academic_output: Path) -> None:
    academic_output.mkdir(parents=True, exist_ok=True)
    (academic_output / "figures").mkdir(parents=True, exist_ok=True)
    (academic_output / "tables").mkdir(parents=True, exist_ok=True)
    ensure_link(source_output / "models", academic_output / "models")
    ensure_link(source_output / "prepared_hddm_data.csv", academic_output / "prepared_hddm_data.csv")


def run_module(module_name: str, extra_args: list[str], output_dir: Path, chains: int, accepts_chains: bool) -> None:
    module = importlib.import_module(module_name)
    update_module_style(module, wrapped_labels=module_name in WRAPPED_LABEL_MODULES)

    # Keep imported dependencies harmonized too, especially the combined HDDM figure.
    if module_name == "plot_hddm_rtcritical_group_posteriors_combined":
        group_posteriors = importlib.import_module("plot_group_posteriors")
        update_module_style(group_posteriors, wrapped_labels=True)

    argv = [
        module_name,
        *extra_args,
        "--output-dir",
        str(output_dir),
    ]
    if accepts_chains:
        argv.extend(["--chains", str(chains)])
    old_argv = sys.argv[:]
    try:
        sys.argv = argv
        print(f"\n=== {module_name} {' '.join(extra_args)} ===")
        module.main()
    finally:
        sys.argv = old_argv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-output", default="outputs")
    parser.add_argument("--academic-output", default="outputs_academic")
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional module names to run, e.g. plot_group_posteriors.",
    )
    args = parser.parse_args()

    root = Path.cwd()
    source_output = root / args.source_output
    academic_output = root / args.academic_output

    os.environ["HDDM_ACADEMIC_FIGURES"] = "1"
    os.environ.setdefault("SCIENTIFIC_VISUALIZATION_SKILL", "skills/scientific-visualization")
    # Path repair after reorganisation: the sibling plot modules moved to
    # _organized/code/hddm_project/scripts/ while root stays the cwd.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(root / "scripts"))

    apply_academic_style()
    install_academic_savefig_patch()
    prepare_output_root(source_output, academic_output)

    runs = DEFAULT_RUNS
    if args.only:
        selected = set(args.only)
        runs = [run for run in DEFAULT_RUNS if run[0] in selected]
        missing = selected - {run[0] for run in runs}
        if missing:
            raise SystemExit(f"Unknown --only module(s): {', '.join(sorted(missing))}")

    for module_name, extra_args, accepts_chains in runs:
        run_module(module_name, extra_args, academic_output, args.chains, accepts_chains)

    print(f"\nAcademic figures written under: {academic_output / 'figures'}")


if __name__ == "__main__":
    main()
