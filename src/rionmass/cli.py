"""Command-line calibration (rionmass run) and optional Qt interface (rionmass gui)."""

from __future__ import annotations

import argparse
import sys

from .configuration import CalibrationConfig, ODSImportConfig, parse_isomer_energy_spec
from .reference_masses import parse_reference_mass_spec
from .version import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rionmass", description="Mass calibration toolkit for stored-ion mass measurements."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the full calibration workflow against an ODS table.")
    run.add_argument("--ods", required=True, help="Path to the template-fit ODS table.")
    run.add_argument("--sheet-index", type=int, required=True, help="Zero-based sheet index within the ODS file.")
    run.add_argument(
        "--elbien-source",
        choices=["file", "nist"],
        default="file",
        help="Electron-binding-energy source: an ElBiEn_*.dat table (default) or scraped/cached NIST data.",
    )
    run.add_argument(
        "--elbien",
        default=None,
        help="Path to the electron-binding-energy table (ElBiEn_*.dat). Required if --elbien-source=file.",
    )
    run.add_argument(
        "--nist-cache",
        default=None,
        help="Optional override for the NIST binding-energy cache path (only used with --elbien-source=nist).",
    )
    run.add_argument(
        "--electronic-inputs",
        default=None,
        help="JSON electronic corrections/covariance or individual ionization energies; overrides the table.",
    )
    run.add_argument(
        "--electronic-assume-independent",
        action="store_true",
        help="Explicit independence approximation for primitive table binding errors (not the default).",
    )
    run.add_argument("--electronic-method", choices=["fixed_weights", "refit"], default="fixed_weights")
    run.add_argument("--electronic-step-mev", type=float, default=0.001)
    run.add_argument("--electronic-workers", type=int, default=1)
    run.add_argument("--exclude", type=int, nargs="*", default=[], help="Row indices to exclude from the ODS sheet.")
    run.add_argument("--unmeasured", nargs="*", default=[], help="Species ids to flag as unknown, e.g. 205Au78+.")
    run.add_argument("--observable", choices=["frequency", "time"], default="frequency")
    run.add_argument(
        "--harmonic", type=float, default=None, help="Divide ODS frequency columns by this harmonic before conversion."
    )
    run.add_argument(
        "--revname", default=None, help="Optional path to also write the legacy revolution-time text file."
    )
    run.add_argument("--p", type=int, default=3, help="Number of polynomial terms (degree = p - 1).")
    run.add_argument("--iteration-max", type=int, default=100)
    run.add_argument("--tol", type=float, default=1e-14)
    run.add_argument(
        "--reference-mass",
        action="append",
        default=[],
        metavar="NUCLIDE=ME_KEV,SIGMA_KEV",
        help="Repeatable custom ground-state atomic mass excess and 1-sigma uncertainty in keV; optional # source.",
    )
    run.add_argument(
        "--isomer",
        action="append",
        default=[],
        metavar="NAME=ENERGY_MEV[,UNCERTAINTY_MEV]",
        help="Repeatable isomer excitation energy.",
    )
    run.add_argument(
        "--loo-anchor",
        action="append",
        default=[],
        metavar="SPECIES_ID",
        help="Repeatable reference anchor: keep in final calibration; exclude its holdout from scatter.",
    )
    run.add_argument(
        "--loo-anchor-policy",
        choices=["retain_in_training", "remove_from_validation"],
        default="retain_in_training",
        help="Whether anchors remain in other LOO training sets.",
    )
    run.add_argument("--compare-orders", type=int, nargs="*", default=[2, 3, 4, 5])
    run.add_argument("--outdir", default=None, help="If given, export result tables as CSV files here.")
    run.add_argument("--quiet", action="store_true", help="Suppress the printed calibration report.")

    sub.add_parser("gui", help="Launch the PySide6 GUI.")

    return parser


def _run(args: argparse.Namespace) -> int:
    from .api import export_result_tables, run_full_calibration_workflow

    ods_config = ODSImportConfig(
        filename=args.ods,
        sheet_index=args.sheet_index,
        exclusion_list=tuple(args.exclude),
        unmeasured_list=tuple(args.unmeasured),
        observable=args.observable,
        harmonic=args.harmonic,
        revname=args.revname,
    )
    try:
        isomer_energies = parse_isomer_energy_spec(args.isomer)
        calibration_config = CalibrationConfig(
            elbien_source=args.elbien_source,
            elbien_path=args.elbien,
            nist_cache_path=args.nist_cache,
            isomer_excitation_energy=isomer_energies,
            reference_mass_overrides=parse_reference_mass_spec(args.reference_mass),
            p=args.p,
            iterationMax=args.iteration_max,
            tol=args.tol,
            compare_orders=tuple(args.compare_orders),
            loo_anchor_species=tuple(args.loo_anchor),
            loo_anchor_policy=args.loo_anchor_policy,
            electronic_inputs_path=args.electronic_inputs,
            electronic_assume_independent=args.electronic_assume_independent,
            electronic_method=args.electronic_method,
            electronic_step_MeV=args.electronic_step_mev,
            electronic_workers=args.electronic_workers,
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    result = run_full_calibration_workflow(ods_config, calibration_config)

    if not args.quiet:
        from .plotting import print_calibration_report

        print_calibration_report(result.fit)
        print(f"\nLeave-one-out extra systematic: {result.extra_sys_keV:.4f} keV")
        if calibration_config.loo_anchor_species:
            print(
                f"LOO anchors: {', '.join(calibration_config.loo_anchor_species)}; "
                f"policy: {calibration_config.loo_anchor_policy}; "
                f"{result.systematic_result['N']} residuals used for scatter"
            )
        if not result.electronic_uncertainty_complete:
            print("Electronic uncertainty/covariance is incomplete; conditional errors are reported separately.")
        if len(result.unknown_df):
            print("\nUnknown masses:")
            print(result.unknown_df.to_string(index=False))
        else:
            print("\nNo unknown ions in this dataset.")

    if args.outdir:
        written = export_result_tables(result, args.outdir)
        print(f"\nWrote {len(written)} result file(s) to {args.outdir}")

    return 0


def _gui(_args: argparse.Namespace) -> int:
    try:
        from .gui.app import main as gui_main
    except ImportError as exc:
        print(
            f"The GUI requires the 'gui' extra: pip install \"rionmass[gui]\"\n(import failed: {exc})",
            file=sys.stderr,
        )
        return 1
    return gui_main() or 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _run(args)
    if args.command == "gui":
        return _gui(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
