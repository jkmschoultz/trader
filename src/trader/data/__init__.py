"""Market data layer: the Parquet bar lake, ingestion, and session calendars.

Importing this package pulls in pandas and pyarrow, which live in the ``data``
optional dependency group rather than the base install:

    pip install -e ".[data]"

The Saxo-facing half (``trader.saxo.charts``, ``trader.saxo.exchanges``) has no
such dependency, so the API client stays importable without them.
"""
