#!/usr/bin/env python3
"""Legacy argparse run-management shim backed by typed core services."""

from __future__ import annotations

import argparse
import json
import logging

from tabulate import tabulate

from gemma_tuner.core.run_queries import (
    RunQuery,
    build_overview,
    cleanup_runs,
    get_run_details,
)
from gemma_tuner.core.run_queries import (
    list_runs as query_runs,
)


def list_runs(args, output_dir="output"):
    rows = [item.as_row() for item in query_runs(output_dir, _build_query(args))]
    logging.getLogger(__name__).info(
        "\n"
        + tabulate(
            rows,
            headers=[
                "Run ID",
                "Type",
                "Status",
                "Profile",
                "Model",
                "Dataset",
                "Finetuning Run ID",
                "Start Time",
                "Directory",
                "WER",
            ],
            tablefmt="grid",
        )
    )


def overview(args, output_dir="output"):
    result = build_overview(output_dir, _build_query(args))
    logger = logging.getLogger(__name__)
    logger.info(f"Total runs: {result.total_runs}")
    logger.info(f"Finetuning runs: {result.finetuning_runs}")
    logger.info(f"Evaluation runs: {result.evaluation_runs}")
    if result.average_wer is not None:
        logger.info(f"Average WER (completed evaluation runs): {result.average_wer:.4f}")
    if result.best_runs:
        logger.info("\nBest performing runs (Model, Dataset, WER, Run ID):")
        for best in result.best_runs:
            logger.info(f"- {best.model}, {best.dataset}: {best.wer:.4f} ({best.run_id})")


def details(args, output_dir="output"):
    run = get_run_details(output_dir, str(args.run_id))
    if run is None:
        logging.getLogger(__name__).error(f"Run with ID '{args.run_id}' not found.")
        return
    logging.getLogger(__name__).info(json.dumps(run.metadata, indent=4))


def cleanup(args, output_dir="output"):
    result = cleanup_runs(output_dir)
    logger = logging.getLogger(__name__)
    if not result.deleted_runs and not result.failed_runs:
        logger.info("No failed or cancelled runs found.")
        return
    for deleted in result.deleted_runs:
        logger.info(f"Deleted {deleted.status} run: {deleted.run_dir}")
    if result.deleted_runs:
        logger.info(f"Total space freed: {result.total_bytes_freed} bytes")
    for run_dir, error in result.failed_runs.items():
        logger.error(f"Error deleting directory '{run_dir}': {error}")


def run_wizard(args):
    try:
        from gemma_tuner.wizard import wizard_main

        wizard_main()
    except ImportError as e:
        logging.getLogger(__name__).error(f"❌ Failed to import wizard: {e}")
        logging.getLogger(__name__).error("Make sure wizard.py is present and dependencies are installed:")
        logging.getLogger(__name__).error("  pip install rich questionary")
    except Exception as e:
        logging.getLogger(__name__).error(f"❌ Wizard error: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Legacy run-management CLI.")
    subparsers = parser.add_subparsers(dest="command")

    list_parser = subparsers.add_parser("list", help="List runs")
    _add_run_filters(list_parser)
    list_parser.set_defaults(func=list_runs)

    overview_parser = subparsers.add_parser("overview", help="Show an overview of runs")
    _add_run_filters(overview_parser)
    overview_parser.set_defaults(func=overview)

    details_parser = subparsers.add_parser("details", help="Show details for a specific run")
    details_parser.add_argument("run_id", help="ID of the run")
    details_parser.set_defaults(func=details)

    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up failed and cancelled runs")
    cleanup_parser.set_defaults(func=cleanup)

    wizard_parser = subparsers.add_parser("finetune-wizard", help="Launch the interactive wizard")
    wizard_parser.set_defaults(func=run_wizard)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


def _add_run_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--type", choices=["finetuning", "evaluation", "blacklist"], help="Filter by run type")
    parser.add_argument("--profile", help="Filter by profile name (exact match)")
    parser.add_argument("--model", help="Filter by model name (substring match)")
    parser.add_argument("--dataset", help="Filter by dataset name (substring match)")
    parser.add_argument("--finetuning_run_id", help="Filter by finetuning run ID")
    parser.add_argument("--from", dest="from_date", type=str, help="Filter by start date (YYYY-MM-DD)")
    parser.add_argument("--to", dest="to_date", type=str, help="Filter by end date (YYYY-MM-DD)")
    parser.add_argument("--min-wer", type=float, help="Minimum WER")
    parser.add_argument("--max-wer", type=float, help="Maximum WER")
    parser.add_argument("--include-failed", action="store_true", help="Include failed runs")


def _build_query(args) -> RunQuery:
    return RunQuery.from_filters(
        type=getattr(args, "type", None),
        profile=getattr(args, "profile", None),
        model=getattr(args, "model", None),
        dataset=getattr(args, "dataset", None),
        finetuning_run_id=getattr(args, "finetuning_run_id", None),
        from_date=getattr(args, "from_date", None),
        to_date=getattr(args, "to_date", None),
        min_wer=getattr(args, "min_wer", None),
        max_wer=getattr(args, "max_wer", None),
        include_failed=bool(getattr(args, "include_failed", False)),
    )


if __name__ == "__main__":
    main()
