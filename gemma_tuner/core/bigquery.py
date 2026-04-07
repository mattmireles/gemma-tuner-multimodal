#!/usr/bin/env python3
"""
BigQuery integration utilities for the Gemma macOS Tuner wizard.

This module encapsulates all logic related to authenticating with Google Cloud,
listing datasets and tables, introspecting schemas, constructing dynamic SQL
queries, executing them efficiently, and exporting results into the CSV format
expected by the existing dataset loader (no loader changes required).

LLM-first documentation philosophy:
- All public functions include precise contracts and cross-references to callers
- Magic numbers are avoided; defaults are named and documented
- Edge cases and error semantics are explicit

CSV output contract (consumed by `utils/dataset_utils.py` via `config.ini`):
- Required columns: `id`, `audio_path`, and a transcript column (dynamically named)
- Optional column: `language`
- File path: data/datasets/{dataset_name}/{dataset_name}_prepared.csv
- Caller is responsible for adding `[dataset:{dataset_name}]` to `config.ini`
  and setting `text_column` to match the transcript column name.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Tuple

# Lazy import hint: We import the SDK inside functions to reduce import cost


@dataclass
class TableField:
    """Schema field descriptor returned by `get_table_schema()`.

    Attributes:
        name: Field name in the table schema
        field_type: BigQuery type (STRING, INTEGER, etc.)
        mode: NULLABLE/REPEATED/REQUIRED (as string for simplicity)
    """

    name: str
    field_type: str
    mode: str


class Defaults:
    """Central place for configurable defaults used across this module."""

    # Timeouts (seconds)
    GCLOUD_AUTH_TIMEOUT_S: int = 10
    LIST_TIMEOUT_S: int = 60
    QUERY_TIMEOUT_S: int = 15 * 60  # 15 minutes for large queries (guarded by LIMIT)

    # Output
    DATE_FORMAT: str = "%Y%m%d"


def _verify_dataframe_dependencies() -> None:
    """Ensure required dataframe dependencies are present before BQ export.

    This is a fast, explicit check so the wizard can show a helpful message
    instead of surfacing a lower-level import error from the BigQuery SDK.

    Required:
    - pandas: used to materialize query results to CSV
    - db-dtypes: required by google-cloud-bigquery -> DataFrame conversions

    Raises:
        RuntimeError: with an actionable message describing how to install deps
    """
    try:
        import pandas  # type: ignore  # noqa: F401
    except Exception:
        raise RuntimeError(
            "Missing dependency: pandas. Install with 'pip install pandas' or 'pip install -r requirements/requirements.txt'."
        )
    try:
        import db_dtypes  # type: ignore  # noqa: F401
    except Exception:
        raise RuntimeError(
            "Missing dependency: db-dtypes. Install with 'pip install db-dtypes' or 'pip install -r requirements/requirements.txt'."
        )


def _has_gcloud_cli() -> bool:
    try:
        r = subprocess.run(
            ["gcloud", "--version"], capture_output=True, text=True, timeout=Defaults.GCLOUD_AUTH_TIMEOUT_S
        )
        return r.returncode == 0
    except Exception:
        return False


def check_gcp_auth() -> bool:
    """Return True if Application Default Credentials are configured.

    Called by:
        - `wizard.select_bigquery_table_and_export()` before any BQ operations

    Strategy:
        - Prefer a quick `gcloud auth application-default print-access-token` probe
        - Fallback: respect GOOGLE_APPLICATION_CREDENTIALS presence (best-effort)
    """
    try:
        if _has_gcloud_cli():
            r = subprocess.run(
                ["gcloud", "auth", "application-default", "print-access-token"],
                capture_output=True,
                text=True,
                timeout=Defaults.GCLOUD_AUTH_TIMEOUT_S,
            )
            # Read into a local, check it, immediately delete.
            # Never log r.stdout — it contains a live GCP bearer token.
            token = r.stdout.strip()
            is_valid = r.returncode == 0 and bool(token)
            del token  # Minimize window the token lives in memory
            if is_valid:
                return True
    except Exception:
        pass
    # Fallback: consider env pointing to a credentials file as a weak positive
    return bool(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))


def list_datasets(project_id: str) -> List[str]:
    """List dataset IDs in a project using the BigQuery SDK; fallback to `bq ls -d`.

    Raises ValueError if project_id contains characters unsafe for CLI use.
    Returns empty list on GCP connectivity or auth errors (wizard falls back to manual input).
    """
    # Validate before any network or subprocess call so an unsafe identifier
    # raises clearly to the caller rather than being caught by the broad except below.
    _assert_safe_identifier(project_id, "project_id")
    try:
        import google.auth  # type: ignore
        from google.cloud import bigquery  # type: ignore

        credentials, _ = google.auth.default(quota_project_id=project_id)
        client = bigquery.Client(project=project_id, credentials=credentials)
        return [d.dataset_id for d in client.list_datasets(project=project_id, max_results=1000)]
    except Exception:
        # CLI fallback
        try:
            r = subprocess.run(
                ["bq", "ls", "-d", f"--project_id={project_id}"],
                capture_output=True,
                text=True,
                timeout=Defaults.LIST_TIMEOUT_S,
            )
            if r.returncode != 0:
                return []
            lines = [ln.strip() for ln in r.stdout.splitlines()]
            ds = [ln.split()[-1] for ln in lines if ln and not ln.lower().startswith(("dataset", "---"))]
            return [d for d in ds if d and d != "dataset_id"]
        except Exception:
            return []


def list_tables(project_id: str, dataset_id: str) -> List[str]:
    """List table IDs in a dataset; empty list on GCP errors.

    Raises ValueError if project_id or dataset_id contain characters unsafe for CLI use.
    Uses BigQuery SDK first; falls back to `bq ls`.
    """
    # Validate up-front so unsafe identifiers raise to the caller
    # rather than being silently swallowed by the broad except below.
    _assert_safe_identifier(project_id, "project_id")
    _assert_safe_identifier(dataset_id, "dataset_id")
    try:
        import google.auth  # type: ignore
        from google.cloud import bigquery  # type: ignore

        credentials, _ = google.auth.default(quota_project_id=project_id)
        client = bigquery.Client(project=project_id, credentials=credentials)
        return [t.table_id for t in client.list_tables(dataset_id, max_results=10000)]
    except Exception:
        try:
            r = subprocess.run(
                ["bq", "ls", f"{project_id}:{dataset_id}"],
                capture_output=True,
                text=True,
                timeout=Defaults.LIST_TIMEOUT_S,
            )
            if r.returncode != 0:
                return []
            lines = [ln.strip() for ln in r.stdout.splitlines()]
            tbls = [ln.split()[-1] for ln in lines if ln and not ln.lower().startswith(("table", "---"))]
            return [t for t in tbls if t and t != "table_id"]
        except Exception:
            return []


def get_table_schema(project_id: str, dataset_id: str, table_id: str) -> List[TableField]:
    """Fetch table schema as a list of `TableField` records.

    Returns empty list if the table is not accessible.
    """
    _assert_safe_identifier(project_id, "project_id")
    _assert_safe_identifier(dataset_id, "dataset_id")
    _assert_safe_identifier(table_id, "table_id")
    try:
        import google.auth  # type: ignore
        from google.cloud import bigquery  # type: ignore

        credentials, _ = google.auth.default(quota_project_id=project_id)
        client = bigquery.Client(project=project_id, credentials=credentials)
        table = client.get_table(f"{project_id}.{dataset_id}.{table_id}")
        fields: List[TableField] = [TableField(f.name, f.field_type, f.mode) for f in table.schema]
        return fields
    except Exception:
        return []


def verify_table(
    project_id: str,
    dataset_id: str,
    table_id: str,
    *,
    location: Optional[str] = None,
) -> Tuple[bool, str]:
    """Run a minimal query against a table/view to validate accessibility.

    Returns (ok, message). On failure, message contains the BigQuery error string.

    Called by:
        - wizard.select_bigquery_table_and_export() after user selects a table
    """
    # Validate identifiers to prevent SQL injection via backtick breakout
    for name, label in ((project_id, "project_id"), (dataset_id, "dataset_id"), (table_id, "table_id")):
        # Use _assert_safe_identifier (not _assert_safe_column_name) — GCP project IDs
        # routinely contain hyphens (e.g. my-project-123) which column names cannot.
        _assert_safe_identifier(name, label)
    try:
        import google.auth  # type: ignore
        from google.cloud import bigquery  # type: ignore

        credentials, _ = google.auth.default(quota_project_id=project_id)
        client = bigquery.Client(project=project_id, credentials=credentials)
        sql = f"SELECT 1 FROM `{project_id}.{dataset_id}.{table_id}` LIMIT 1"
        job = client.query(sql, location=location)
        # Force execution
        _ = list(job.result(max_results=1))
        return True, "ok"
    except Exception as e:
        return False, str(e)


def _assert_safe_column_name(name: str) -> None:
    """Raise ValueError if name contains characters that could break out of backtick quoting.

    BigQuery column names are backtick-quoted in generated SQL. A backtick character
    inside the name would terminate the quote and allow SQL injection. Only allow
    alphanumerics and underscores, which covers all valid BigQuery column identifiers.
    """
    if not re.fullmatch(r"[A-Za-z0-9_]+", name):
        raise ValueError(f"Unsafe column name rejected: {name!r}")


def _assert_safe_identifier(name: str, label: str = "identifier") -> None:
    """Raise ValueError if name contains characters unsafe for use in bq CLI arguments.

    GCP project IDs and dataset IDs may contain letters, digits, and hyphens only.
    This is called before passing project_id / dataset_id to subprocess.run() so
    that malformed inputs are rejected with a clear error instead of being silently
    passed to the bq CLI where they could cause unexpected argument parsing.

    Valid GCP identifier characters: letters, digits, hyphens, underscores.
    Rejects anything containing shell metacharacters or whitespace.

    Called by:
    - list_datasets() before bq CLI fallback
    - list_tables() before bq CLI fallback
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError(f"Unsafe {label} rejected: {name!r}")


def get_distinct_languages(
    project_id: str,
    dataset_id: str,
    table_id: str,
    language_column: str = "language",
    location: Optional[str] = None,
) -> List[str]:
    """Return distinct language values from the specified table (best-effort).

    If the column does not exist, returns an empty list.

    Note: the `where_sql` parameter was removed — it accepted raw SQL strings with
    no sanitization. Use the `language_column` and `location` params to constrain the
    query; all filtering beyond that should happen after the values are returned.
    """
    _assert_safe_identifier(project_id, "project_id")
    _assert_safe_identifier(dataset_id, "dataset_id")
    _assert_safe_identifier(table_id, "table_id")
    # language_column is already validated by _assert_safe_column_name below
    try:
        import google.auth  # type: ignore
        from google.cloud import bigquery  # type: ignore

        # Quick schema check
        fields = get_table_schema(project_id, dataset_id, table_id)
        if not any(f.name.lower() == language_column.lower() for f in fields):
            return []

        # Validate column name before interpolating into SQL.
        _assert_safe_column_name(language_column)

        credentials, _ = google.auth.default(quota_project_id=project_id)
        client = bigquery.Client(project=project_id, credentials=credentials)

        q = (
            f"SELECT DISTINCT CAST(`{language_column}` AS STRING) AS lang "
            f"FROM `{project_id}.{dataset_id}.{table_id}` "
            f"WHERE `{language_column}` IS NOT NULL ORDER BY lang"
        )
        job_config = bigquery.QueryJobConfig()
        job = client.query(q, job_config=job_config, location=location)
        return [str(row["lang"]) for row in job.result()]
    except Exception:
        return []


def _sanitize_dataset_name_component(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", value)
    return cleaned.strip("_").lower()


def generate_dataset_name(base_dataset: str, base_table: str) -> str:
    """Generate a reproducible, readable dataset name for local CSV materialization.

    Format: bq_{dataset}_{table}_{YYYYMMDD}

    Both base_dataset and base_table are included so that two different tables
    from the same dataset exported on the same day get distinct directory names
    rather than silently overwriting each other.
    """
    today = datetime.utcnow().strftime(Defaults.DATE_FORMAT)
    return f"bq_{_sanitize_dataset_name_component(base_dataset)}_{_sanitize_dataset_name_component(base_table)}_{today}"


def build_query_and_export(
    *,
    project_id: str,
    tables: Sequence[Tuple[str, str]],  # list of (dataset_id, table_id)
    audio_col: str,
    transcript_col: str,
    transcript_target: str,  # Now accepts any column name, not just text_perfect/text_verbatim
    language_col: Optional[str] = None,
    languages: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    sample: Literal["first", "random"] = "first",
    out_dir: Path = Path("."),
    location: Optional[str] = None,
) -> Path:
    """Construct a dynamic SELECT with filters and export results to `_prepared.csv`.

    Called by:
        - `wizard.select_bigquery_table_and_export()` after user mapping

    Query behavior:
        - Projects only the required columns for the fine-tuner: id, audio_path, transcript target, language
        - If the source does not have an `id` column, synthesizes one via ROW_NUMBER()
        - Applies language filter when `language_col` and `languages` are provided
        - Supports multi-table UNION ALL when multiple tables are selected
        - Optional random sampling (`ORDER BY RAND()`) and LIMIT

    Returns:
        Path to the created dataset directory (containing `_prepared.csv`, `.bq_query.sql`, `metadata.json`).

    Raises:
        RuntimeError: When the query runs but returns 0 rows, or cannot write output
        ValueError: When required inputs are invalid
    """
    if not project_id:
        raise ValueError("project_id is required")
    if not tables:
        raise ValueError("at least one (dataset, table) must be provided")
    if not transcript_target:
        raise ValueError("transcript_target column name is required")

    _assert_safe_identifier(project_id, "project_id")
    for _tbl_dataset_id, _tbl_table_id in tables:
        _assert_safe_identifier(_tbl_dataset_id, "dataset_id")
        _assert_safe_identifier(_tbl_table_id, "table_id")

    # Verify userland dataframe dependencies before invoking BigQuery APIs
    _verify_dataframe_dependencies()

    # Validate all column names BEFORE defining _select_for, which captures them
    # via closure. Doing this first ensures no unsanitized name reaches the SQL
    # template even if the function is refactored to call _select_for earlier.
    # Backtick quoting prevents reserved-word conflicts but a backtick *inside* a
    # name would break the quoting and allow injection — reject anything non-identifier.
    for col_name in filter(None, [audio_col, transcript_col, transcript_target, language_col]):
        _assert_safe_column_name(col_name)

    # Import SDK lazily
    import google.auth  # type: ignore
    from google.cloud import bigquery  # type: ignore

    # Helper: Build per-table SELECT
    def _select_for(dataset_id: str, table_id: str, id_source_col: Optional[str]) -> str:
        """Return a SELECT that projects id/audio_path/transcript[/language].

        id_source_col:
            - If provided, CAST that column to STRING as `id`
            - Otherwise, synthesize a stable `id` via ROW_NUMBER()

        Note: language_col is validated by _assert_safe_column_name before this
        function is defined. has_language (set after schema discovery) determines
        whether language_col is actually present in the table; _select_for uses
        NULL AS language when the column is absent.
        """
        # Backtick-quote all column identifiers to handle reserved words and names with spaces.
        lang_sel = f", `{language_col}` AS language" if language_col else ", NULL AS language"
        source = f"`{project_id}.{dataset_id}.{table_id}`"
        id_expr = (
            f"CAST(`{id_source_col}` AS STRING) AS id" if id_source_col else "CAST(ROW_NUMBER() OVER() AS STRING) AS id"
        )
        return (
            f"SELECT {id_expr}, CAST(`{audio_col}` AS STRING) AS audio_path, "
            f"CAST(`{transcript_col}` AS STRING) AS `{transcript_target}`{lang_sel} FROM {source}"
        )

    # Discover schema of the first table to check `id` and column existence
    fields = get_table_schema(project_id, tables[0][0], tables[0][1])
    # Map lower -> original for stable case-preserving lookups
    field_lower_to_original = {f.name.lower(): f.name for f in fields}
    field_names_lower = set(field_lower_to_original.keys())
    # Map field names to their types for type checking
    field_type_map = {f.name.lower(): f.field_type for f in fields}

    if audio_col.lower() not in field_names_lower:
        raise ValueError(f"Audio column '{audio_col}' not found in {tables[0][0]}.{tables[0][1]}")
    if transcript_col.lower() not in field_names_lower:
        raise ValueError(f"Transcript column '{transcript_col}' not found in {tables[0][0]}.{tables[0][1]}")

    # Prefer an existing identifier column if present, but only if it's a numeric type.
    # Priority order: id, note_id, sample_id, uid, guid, then any *_id column.
    # Only consider INT64, NUMERIC, BIGNUMERIC, or FLOAT64 columns as valid ID sources.
    id_source_col: Optional[str] = None
    numeric_types = {"INT64", "INTEGER", "NUMERIC", "BIGNUMERIC", "FLOAT64"}
    priority_id_names = [
        "id",
        "note_id",
        "sample_id",
        "uid",
        "guid",
    ]
    for candidate in priority_id_names:
        if candidate in field_names_lower:
            field_type = field_type_map.get(candidate, "").upper()
            # Only use this column if it's a numeric type
            if field_type in numeric_types:
                id_source_col = field_lower_to_original[candidate]
                break
    if id_source_col is None:
        # Fallback: pick the first column that ends with _id and is numeric
        for lower_name in sorted(field_names_lower):
            if lower_name.endswith("_id"):
                field_type = field_type_map.get(lower_name, "").upper()
                if field_type in numeric_types:
                    id_source_col = field_lower_to_original[lower_name]
                    break

    has_language = (language_col or "").lower() in field_names_lower if language_col else False

    # Build the core SELECT (UNION ALL if many tables)
    select_parts: List[str] = []
    for ds, tb in tables:
        select_parts.append(_select_for(ds, tb, id_source_col))
    core_select = "\nUNION ALL\n".join(select_parts)

    # WHERE clause
    where_clauses: List[str] = []
    if has_language and languages:
        where_clauses.append("language IN UNNEST(@languages)")
    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    # ORDER/LIMIT
    order_sql = "ORDER BY RAND()" if sample == "random" else ""
    limit_sql = "LIMIT @limit" if isinstance(limit, int) and limit > 0 else ""

    # Wrap the UNION in a subquery so WHERE/ORDER/LIMIT apply to the combined set
    sql = (
        f"WITH unioned AS (\n{core_select}\n)\n"
        f"SELECT id, audio_path, `{transcript_target}`, language FROM unioned\n"
        f"{where_sql}\n{order_sql}\n{limit_sql}"
    ).strip()

    # Prepare destination dir and filenames
    first_ds, first_tb = tables[0]
    dataset_name = (
        generate_dataset_name(first_ds, first_tb)
        if len(tables) == 1
        else (
            f"bq_{_sanitize_dataset_name_component(first_ds)}_multi_{datetime.utcnow().strftime(Defaults.DATE_FORMAT)}"
        )
    )
    dataset_dir = out_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    prepared_csv = dataset_dir / f"{dataset_name}_prepared.csv"
    sql_path = dataset_dir / ".bq_query.sql"
    meta_path = dataset_dir / "metadata.json"

    # Execute query
    credentials, _ = google.auth.default(quota_project_id=project_id)
    client = bigquery.Client(project=project_id, credentials=credentials)

    params: List[bigquery.ScalarQueryParameter] = []
    if has_language and languages:
        params.append(bigquery.ArrayQueryParameter("languages", "STRING", list(languages)))
    if isinstance(limit, int) and limit > 0:
        params.append(bigquery.ScalarQueryParameter("limit", "INT64", int(limit)))

    job_config = bigquery.QueryJobConfig(query_parameters=params)

    # Persist SQL before execution for debuggability even if the query fails
    try:
        sql_path.write_text(sql, encoding="utf-8")
    except Exception:
        pass

    job = client.query(sql, job_config=job_config, location=location)
    try:
        df = job.result().to_dataframe(create_bqstorage_client=False)
    except Exception as e:
        # Persist minimal metadata and re-raise with actionable context
        try:
            meta_path.write_text(
                json.dumps(
                    {
                        "project_id": project_id,
                        "tables": [f"{ds}.{tb}" for ds, tb in tables],
                        "audio_col": audio_col,
                        "transcript_col": transcript_col,
                        "transcript_target": transcript_target,
                        "language_col": language_col,
                        "languages": list(languages) if languages else None,
                        "limit": limit,
                        "sample": sample,
                        "location": location,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass

        # Improve the most common failure: STRING vs INT64 filter mismatch
        message = str(e)
        hint = ""
        if "Cannot read field of type STRING as INT64" in message:
            # Extract field name if present to tailor the hint
            m = re.search(r"Field:\s*(\w+)", message)
            field_name = m.group(1) if m else "the field"
            hint = (
                f"\nHint: {field_name} is a STRING in BigQuery. If you filtered on it in 'Advanced WHERE', "
                f"wrap numeric-looking values in quotes, e.g. {field_name}='12345', or CAST({field_name} AS INT64)=12345."
            )
        raise RuntimeError(f"BigQuery query failed: {message}{hint}")

    if df.empty:
        # Persist artifacts for debuggability even on empty results
        try:
            sql_path.write_text(sql, encoding="utf-8")
            meta_path.write_text(
                json.dumps(
                    {
                        "project_id": project_id,
                        "tables": [f"{ds}.{tb}" for ds, tb in tables],
                        "audio_col": audio_col,
                        "transcript_col": transcript_col,
                        "transcript_target": transcript_target,
                        "language_col": language_col,
                        "languages": list(languages) if languages else None,
                        "limit": limit,
                        "sample": sample,
                        "location": location,
                        "row_count": 0,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass
        raise RuntimeError("BigQuery query returned 0 rows with the selected filters.")

    # Enforce column order and types expected by the loader
    # - Ensure id/audio_path exist and are strings
    # - Ensure transcript target column exists; language optional
    expected_cols = ["id", "audio_path", transcript_target]
    if "language" in df.columns:
        expected_cols.append("language")

    # Coerce basic types
    for col in ["id", "audio_path", transcript_target]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Write the full prepared CSV before splitting.
    # dataset_utils.load_dataset_split() checks for {dataset_name}_prepared.csv as a
    # fallback path when no split-specific file exists. Without this write, that fallback
    # always raises FileNotFoundError for BigQuery-exported datasets.
    df[expected_cols].to_csv(prepared_csv, index=False)

    # Split into train/validation sets (80/20 split)
    train_df = df.sample(frac=0.8, random_state=42)
    validation_df = df.drop(train_df.index)

    # Write separate CSVs for train and validation splits
    train_csv_path = dataset_dir / "train.csv"
    validation_csv_path = dataset_dir / "validation.csv"

    train_df[expected_cols].to_csv(train_csv_path, index=False)
    validation_df[expected_cols].to_csv(validation_csv_path, index=False)

    # Persist SQL and metadata for reproducibility
    sql_path.write_text(sql, encoding="utf-8")
    meta = {
        "project_id": project_id,
        "tables": [f"{ds}.{tb}" for ds, tb in tables],
        "audio_col": audio_col,
        "transcript_col": transcript_col,
        "transcript_target": transcript_target,
        "language_col": language_col,
        "languages": list(languages) if languages else None,
        "limit": limit,
        "sample": sample,
        "location": location,
        "row_count": int(len(df)),
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return dataset_dir
