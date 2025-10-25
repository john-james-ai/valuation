# Recommended Polars Parquet read/write kwargs and mapping notes.

# Polars-friendly read kwargs for parquet (pass to pl.scan_parquet or pl.read_parquet)
POLARS_PARQUET_READ_KWARGS = {
    # columns: list[str] | None -> subset of columns to read
    "columns": None,
    # n_rows: int | None -> limit rows when eager-reading (pl.read_parquet supports n_rows)
    "n_rows": None,
    # parallel: "auto" | "columns" | "row_groups" | "none"
    "parallel": "auto",
    # rechunk contiguous memory after read (recommended True)
    "rechunk": True,
    # low_memory: reduce memory use at cost of performance
    "low_memory": False,
    # use_statistics: let parquet reader use file statistics for predicate pushdown / planning
    "use_statistics": True,
    # hive_partitioning: enable hive-style partitioning (useful for partitioned datasets)
    "hive_partitioning": True,
    # row_count_name, row_count_offset: optional for adding a row counter column when scanning parquet
    # "row_count_name": None,
    # "row_count_offset": None,
    # Note: Polars does not accept 'engine', 'dtype_backend' or 'use_threads' kwargs.
    # Filters: apply predicates via LazyFrame.filter(...) before collect(), e.g.:
    #   lf = pl.scan_parquet(...).filter(pl.col("col") == value)
}

# Polars-friendly write kwargs for parquet (pass to DataFrame.write_parquet)
POLARS_PARQUET_WRITE_KWARGS = {
    # compression: "zstd" | "snappy" | "gzip" | "lz4" | "uncompressed"
    "compression": "snappy",
    # compression_level: int | None
    "compression_level": None,
    # statistics: bool -> write statistics to enable predicate pushdown on reads
    "statistics": True,
    # row_group_size: int -> number of rows per row group (tune for your IO pattern)
    "row_group_size": 256000,
    # data_page_size: int | None -> parquet data page size in bytes
    "data_page_size": None,
    # Note: Polars' write_parquet does not currently offer a direct 'partition_cols' argument.
    # To write partitioned datasets you can:
    #  - use pyarrow.dataset or pyarrow.parquet.write_to_dataset
    #  - or manually group and write partition subfolders
}

# Usage notes / mapping from your original kwargs:
# - 'engine': 'pyarrow'  -> Polars uses pyarrow under the hood for parquet. No 'engine' kwarg required.
# - 'columns': None      -> pass through to "columns"
# - 'filters': None      -> apply predicates in Polars with LazyFrame.filter(...) before collect()
# - 'use_threads': True  -> Polars manages parallelism via 'parallel' option; no use_threads kwarg
# - 'dtype_backend': 'pyarrow' -> Polars manages dtypes; convert dtype mappings to Polars dtypes
#
# Example:
#   read_kwargs = POLARS_PARQUET_READ_KWARGS.copy()
#   read_kwargs["columns"] = ["col1","col2"]
#   lf = pl.scan_parquet(path, **read_kwargs)
#   lf = lf.filter(pl.col("year") >= 2000)   # apply filters in lazy pipeline
#   df = lf.collect()
#
# For writing partitioned parquet with PyArrow:
#   import pyarrow as pa, pyarrow.parquet as pq, pyarrow.dataset as ds
#   table = pa.Table.from_pandas(df.to_pandas())  # or from pyarrow arrays
#   pq.write_to_dataset(table, root_path=out_dir, partition_cols=["country","year"], compression="snappy")
