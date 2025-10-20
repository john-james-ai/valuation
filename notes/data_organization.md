# Data Organization

This structure is often called a "stage-first" or "layer-first" layout, and it's the foundation of most data lakes.

```
.
└── data/
    ├── dev/
    │   ├── raw/
    │   │   ├── sales
    │   │   ├── customer
    │   │   └── store
    │   ├── clean/
    │   │   ├── sales
    │   │   ├── customer
    │   │   └── store
    │   └── ingest/
    │       └── ...
    ├── prod/
    │   └── ...
    └── test/
        └── ...
```

-----

## Rationale

1. **Clear Access Control (IAM):** This is the most important reason. It allows you to set permissions by *stage*, which is much safer and easier to manage.

   * **Ingestion tools** (like Fivetran, Airbyte, or custom scripts) get **write-only** access to the `raw/` directory.
   * **Transformation tools** (like dbt, Spark, or data scientists) get **read-only** access to `raw/` and **write** access to `clean/`.
   * **BI tools** (like Tableau or Power BI) get **read-only** access to the `clean/` directory.
   * With Option 1 (`data/dev/sales/raw`), it's much harder to grant a tool access to *all* raw data without giving it access to clean data as well.
1. **Matches the Data Pipeline:** The structure logically follows the flow of data (ETL/ELT). All "dirty" data is in one place, and all "clean," analytics-ready data is in another.
1. **Scalability:** When you add a new data source (e.g., `marketing`, `finance`), you simply add a new folder inside `raw/` and `clean/`. The top-level hierarchy remains simple and organized.

