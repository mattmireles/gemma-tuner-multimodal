from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

import datasets


@dataclass(frozen=True)
class DatasetLoadContext:
    dataset_name: str
    dataset_config: dict
    split: str
    max_samples: int | None
    patches_dir: str
    streaming_enabled: bool
    dataset_section: object
    source: str
    source_type: str
    dataset_dir: str
    split_path: str
    prepared_fallback_path: str
    cache_dir: str
    modality: str
    text_sub_mode: str
    prompt_column: str | None
    image_sub_mode: str = "caption"
    image_path_column: str = "image_path"


class DatasetSourceAdapter(Protocol):
    name: str

    def supports(self, context: DatasetLoadContext) -> bool: ...

    def patch_source(self, context: DatasetLoadContext) -> str: ...

    def load_base_dataset(self, context: DatasetLoadContext): ...


class _BaseCsvAdapter:
    name = "csv"

    def patch_source(self, context: DatasetLoadContext) -> str:
        return context.source

    def load_base_dataset(self, context: DatasetLoadContext):
        data_file_to_use = context.split_path if os.path.exists(context.split_path) else context.prepared_fallback_path
        if not os.path.exists(data_file_to_use):
            raise FileNotFoundError(
                f"Neither split file '{context.split_path}' nor fallback prepared file "
                f"'{context.prepared_fallback_path}' was found for dataset '{context.dataset_name}'. "
                f"Run 'python main.py prepare {context.dataset_name}' to generate splits, "
                f"or create '{context.prepared_fallback_path}'."
            )

        if context.max_samples is not None and not context.streaming_enabled:
            return datasets.load_dataset(
                "csv",
                data_files={context.split: data_file_to_use},
                cache_dir=context.cache_dir,
                split=f"{context.split}[:{context.max_samples}]",
                streaming=False,
            )

        return datasets.load_dataset(
            "csv",
            data_files={context.split: data_file_to_use},
            cache_dir=context.cache_dir,
            split=context.split,
            streaming=context.streaming_enabled,
        )


class LocalCsvDatasetSourceAdapter(_BaseCsvAdapter):
    name = "local-csv"

    def supports(self, context: DatasetLoadContext) -> bool:
        return not context.streaming_enabled and context.source_type not in {"granary", "bigquery", "bigquery-prepared"}


class StreamingDatasetSourceAdapter(_BaseCsvAdapter):
    name = "streaming-csv"

    def supports(self, context: DatasetLoadContext) -> bool:
        return context.streaming_enabled or context.source_type in {"streaming", "gcs"}


class GranaryDatasetSourceAdapter(_BaseCsvAdapter):
    name = "granary"

    def supports(self, context: DatasetLoadContext) -> bool:
        return context.source_type == "granary"

    def patch_source(self, context: DatasetLoadContext) -> str:
        return context.dataset_name


class BigQueryPreparedDatasetSourceAdapter(_BaseCsvAdapter):
    name = "bigquery-prepared"

    def supports(self, context: DatasetLoadContext) -> bool:
        return context.source_type in {"bigquery", "bigquery-prepared"}

    def patch_source(self, context: DatasetLoadContext) -> str:
        return context.source or context.dataset_name


DATASET_SOURCE_ADAPTERS: tuple[DatasetSourceAdapter, ...] = (
    GranaryDatasetSourceAdapter(),
    BigQueryPreparedDatasetSourceAdapter(),
    StreamingDatasetSourceAdapter(),
    LocalCsvDatasetSourceAdapter(),
)


def resolve_dataset_source_adapter(context: DatasetLoadContext) -> DatasetSourceAdapter:
    for adapter in DATASET_SOURCE_ADAPTERS:
        if adapter.supports(context):
            return adapter
    return LocalCsvDatasetSourceAdapter()
