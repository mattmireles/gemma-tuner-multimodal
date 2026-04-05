from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Optional

from gemma_tuner.core.runs import RunConstants

METADATA_FILENAME = RunConstants.METADATA_FILE
COMPLETION_MARKER = RunConstants.COMPLETION_MARKER
EVAL_PREFIX = RunConstants.EVAL_SUBDIR
FAILED_STATUSES = {RunConstants.STATUS_FAILED, RunConstants.STATUS_CANCELLED}
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
DATE_FORMAT = "%Y-%m-%d"


@dataclass(frozen=True)
class RunQuery:
    type: Optional[str] = None
    profile: Optional[str] = None
    model: Optional[str] = None
    dataset: Optional[str] = None
    finetuning_run_id: Optional[str] = None
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None
    min_wer: Optional[float] = None
    max_wer: Optional[float] = None
    include_failed: bool = False

    @classmethod
    def from_filters(
        cls,
        *,
        type: Optional[str] = None,
        profile: Optional[str] = None,
        model: Optional[str] = None,
        dataset: Optional[str] = None,
        finetuning_run_id: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        min_wer: Optional[float] = None,
        max_wer: Optional[float] = None,
        include_failed: bool = False,
    ) -> "RunQuery":
        return cls(
            type=type,
            profile=profile,
            model=model,
            dataset=dataset,
            finetuning_run_id=finetuning_run_id,
            from_date=_parse_date(from_date),
            to_date=_parse_date(to_date),
            min_wer=min_wer,
            max_wer=max_wer,
            include_failed=include_failed,
        )


@dataclass(frozen=True)
class RunListItem:
    run_id: str
    run_type: str
    status: str
    profile: str
    model: str
    dataset: str
    finetuning_run_id: str
    start_time: str
    run_dir: str
    wer: Optional[float]
    metadata: dict[str, Any] = field(repr=False)

    def as_row(self) -> list[Any]:
        return [
            self.run_id,
            self.run_type,
            self.status,
            self.profile,
            self.model,
            self.dataset,
            self.finetuning_run_id,
            self.start_time,
            self.run_dir,
            self.wer if self.wer is not None else "",
        ]


@dataclass(frozen=True)
class BestRun:
    model: str
    dataset: str
    wer: float
    run_id: str


@dataclass(frozen=True)
class RunOverview:
    total_runs: int
    finetuning_runs: int
    evaluation_runs: int
    average_wer: Optional[float]
    best_runs: list[BestRun]


@dataclass(frozen=True)
class RunDetails:
    run_id: str
    run_dir: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CleanupRun:
    run_dir: str
    status: str
    bytes_freed: int


@dataclass(frozen=True)
class CleanupResult:
    deleted_runs: list[CleanupRun]
    failed_runs: dict[str, str]

    @property
    def total_bytes_freed(self) -> int:
        return sum(item.bytes_freed for item in self.deleted_runs)


def list_runs(output_dir: str, query: RunQuery) -> list[RunListItem]:
    return [_to_run_list_item(metadata) for metadata in _filter_runs(output_dir, query)]


def build_overview(output_dir: str, query: RunQuery) -> RunOverview:
    filtered_runs = _filter_runs(output_dir, query)

    num_runs = len(filtered_runs)
    num_finetuning_runs = sum(1 for metadata in filtered_runs if metadata.get("run_type") == "finetuning")
    num_evaluation_runs = sum(1 for metadata in filtered_runs if metadata.get("run_type") == "evaluation")

    evaluation_runs = [
        metadata
        for metadata in filtered_runs
        if metadata.get("run_type") == "evaluation"
        and metadata.get("status") == "completed"
        and (metadata.get("metrics") or {}).get("wer") is not None
    ]

    average_wer: Optional[float] = None
    best_runs: list[BestRun] = []
    if evaluation_runs:
        average_wer = sum(float((metadata.get("metrics") or {})["wer"]) for metadata in evaluation_runs) / len(
            evaluation_runs
        )
        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        for metadata in evaluation_runs:
            key = (metadata.get("model", ""), metadata.get("dataset", ""))
            if key not in grouped or float((metadata.get("metrics") or {})["wer"]) < float(
                (grouped[key].get("metrics") or {})["wer"]
            ):
                grouped[key] = metadata
        best_runs = [
            BestRun(
                model=model,
                dataset=dataset,
                wer=float((metadata.get("metrics") or {})["wer"]),
                run_id=str(metadata.get("run_id", "")),
            )
            for (model, dataset), metadata in grouped.items()
        ]

    return RunOverview(
        total_runs=num_runs,
        finetuning_runs=num_finetuning_runs,
        evaluation_runs=num_evaluation_runs,
        average_wer=average_wer,
        best_runs=best_runs,
    )


def get_run_details(output_dir: str, run_id: str) -> Optional[RunDetails]:
    for metadata in _iter_run_metadata(output_dir):
        if str(metadata.get("run_id")) == str(run_id):
            return RunDetails(run_id=str(run_id), run_dir=str(metadata.get("run_dir", "")), metadata=metadata)
    return None


def cleanup_runs(output_dir: str) -> CleanupResult:
    deleted_runs: list[CleanupRun] = []
    failed_runs: dict[str, str] = {}

    if not os.path.isdir(output_dir):
        return CleanupResult(deleted_runs=deleted_runs, failed_runs=failed_runs)

    for run_dir in os.listdir(output_dir):
        full_run_dir = os.path.join(output_dir, run_dir)
        if not os.path.isdir(full_run_dir):
            continue
        status = get_run_status(full_run_dir)
        if status not in FAILED_STATUSES:
            continue
        bytes_freed = _directory_size(full_run_dir)
        try:
            shutil.rmtree(full_run_dir)
            deleted_runs.append(CleanupRun(run_dir=full_run_dir, status=status, bytes_freed=bytes_freed))
        except Exception as exc:  # pragma: no cover - filesystem/permission dependent
            failed_runs[full_run_dir] = str(exc)

    return CleanupResult(deleted_runs=deleted_runs, failed_runs=failed_runs)


def get_run_status(run_dir: str) -> str:
    metadata = _load_metadata(run_dir)
    if metadata is None:
        return "unknown"

    status = metadata.get("status")
    # Check explicit status field first — a failed/cancelled status should never
    # be overridden by a stale completion marker file.
    if status == "failed":
        return "failed"
    if status == "cancelled":
        return "cancelled"
    if os.path.exists(os.path.join(run_dir, COMPLETION_MARKER)):
        return "completed"
    if metadata.get("end_time") is None:
        return "incomplete"
    return "unknown"


def _filter_runs(output_dir: str, query: RunQuery) -> list[dict[str, Any]]:
    return [metadata for metadata in _iter_run_metadata(output_dir) if _matches_query(metadata, query)]


def _iter_run_metadata(output_dir: str) -> Iterable[dict[str, Any]]:
    if not os.path.isdir(output_dir):
        return []

    discovered: list[dict[str, Any]] = []

    for run_dir in os.listdir(output_dir):
        full_run_dir = os.path.join(output_dir, run_dir)
        if not os.path.isdir(full_run_dir):
            continue

        if "+" not in run_dir:
            metadata = _load_metadata(full_run_dir)
            if metadata is not None:
                discovered.append(metadata)

            for subdir in os.listdir(full_run_dir):
                if not subdir.startswith(EVAL_PREFIX):
                    continue
                eval_dir = os.path.join(full_run_dir, subdir)
                if not os.path.isdir(eval_dir):
                    continue
                nested_meta = _load_metadata(eval_dir)
                if nested_meta is not None:
                    discovered.append(nested_meta)
            continue

        eval_dir = os.path.join(full_run_dir, "eval")
        if not os.path.isdir(eval_dir):
            continue
        metadata = _load_metadata(eval_dir)
        if metadata is None:
            continue
        model, dataset = run_dir.split("+", 1)
        normalized = dict(metadata)
        normalized["model"] = model
        normalized["dataset"] = dataset
        discovered.append(normalized)

    return discovered


def _load_metadata(run_dir: str) -> Optional[dict[str, Any]]:
    metadata_path = os.path.join(run_dir, METADATA_FILENAME)
    try:
        with open(metadata_path, "r") as handle:
            metadata = json.load(handle)
    except FileNotFoundError:
        return None
    metadata["run_dir"] = run_dir
    return metadata


def _matches_query(metadata: dict[str, Any], query: RunQuery) -> bool:
    if query.model and query.model not in str(metadata.get("model", "")):
        return False
    if query.dataset and query.dataset not in str(metadata.get("dataset", "")):
        return False
    if not query.include_failed and metadata.get("status", "unknown") == "failed":
        return False
    if query.type and metadata.get("run_type") != query.type:
        return False
    if query.profile and metadata.get("profile") != query.profile:
        return False
    if query.finetuning_run_id and str(metadata.get("finetuning_run_id")) != str(query.finetuning_run_id):
        return False

    if query.from_date or query.to_date:
        start_time = metadata.get("start_time")
        if not start_time:
            return False
        try:
            entry_timestamp = datetime.strptime(start_time, DATETIME_FORMAT)
        except ValueError:
            return False
        if query.from_date and entry_timestamp < query.from_date:
            return False
        if query.to_date and entry_timestamp > query.to_date:
            return False

    wer_value = (metadata.get("metrics") or {}).get("wer")
    if query.min_wer is not None:
        if wer_value is None or float(wer_value) < query.min_wer:
            return False
    if query.max_wer is not None:
        if wer_value is None or float(wer_value) > query.max_wer:
            return False

    return True


def _to_run_list_item(metadata: dict[str, Any]) -> RunListItem:
    metrics = metadata.get("metrics") or {}
    return RunListItem(
        run_id=str(metadata.get("run_id", "")),
        run_type=str(metadata.get("run_type", "")),
        status=str(metadata.get("status", "unknown")),
        profile=str(metadata.get("profile", "")),
        model=str(metadata.get("model", "")),
        dataset=str(metadata.get("dataset", "")),
        finetuning_run_id=str(metadata.get("finetuning_run_id", "")),
        start_time=str(metadata.get("start_time", "")),
        run_dir=str(metadata.get("run_dir", "")),
        wer=float(metrics["wer"]) if metrics.get("wer") is not None else None,
        metadata=metadata,
    )


def _parse_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.strptime(value, DATE_FORMAT)


def _directory_size(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
    return total
