import os
import random
from typing import List, Optional, Union, Callable, Dict, Any, Sequence

from nanorlhf.nanosets.io.ipc import write_table, read_table

from nanorlhf.nanosets.io.json_io import from_json,    from_jsonl,    to_json,    to_jsonl
from nanorlhf.nanosets.table.record_batch import RecordBatch
from nanorlhf.nanosets.table.table import Table


def _concat_tables(tables: List[Table]) -> Table:
    """Concatenate multiple Tables by appending their RecordBatches.

    All tables must share the same schema.
    """
    if not tables:
        raise ValueError("No tables to concatenate.")
    schema = tables[0].schema
    for t in tables:
        if t.schema != schema:
            raise ValueError("All tables must share the same schema to concatenate.")
    batches = []
    for t in tables:
        batches.extend(t.batches)
    return Table(batches)


def _slice_table(table: Table, indices: Sequence[int]) -> Table:
    """Create a new table by selecting a subset of rows by indices (simple, educational implementation).

    This materializes rows to Python then rebuilds a table. Fine for education, not optimized.
    """
    rows = table.to_pylist()
    picked = [rows[i] for i in indices]
    return Table([RecordBatch.from_pylist(picked)])  # schema inferred again


def _select_columns(table: Table, keep: List[str]) -> Table:
    """Return table with only the selected columns (rebuilds a single-batch table for simplicity)."""
    rows = table.to_pylist()
    kept = [{k: r.get(k, None) if r is not None else None for k in keep} for r in rows]
    return Table([RecordBatch.from_pylist(kept)])


def _remove_columns(table: Table, drop: List[str]) -> Table:
    rows = table.to_pylist()
    pruned = []
    for r in rows:
        if r is None:
            pruned.append(None)
        else:
            pruned.append({k: v for k, v in r.items() if k not in drop})
    return Table([RecordBatch.from_pylist(pruned)])


def _ext(path: str) -> str:
    """Return lowercase file extension without the leading dot."""
    base = os.path.basename(path)
    if "." not in base:
        return ""
    return base.rsplit(".", 1)[1].lower()


class Dataset:
    """A lightweight, HF-like facade around a single `Table`."""

    def __init__(self, table: Table):
        self._table = table

    @property
    def table(self) -> Table:
        return self._table

    def __len__(self) -> int:
        return self._table.length

    def __repr__(self):
        """String representation showing number of rows and schema"""
        return f"Dataset(num_rows={len(self)}, schema={self._table.schema})"

    def save_to_disk(self, path: str):
        """Save as NANO-IPC file."""
        with open(path, "wb") as fp:
            write_table(fp, self._table)

    @staticmethod
    def load_from_disk(path: str) -> "Dataset":
        """Load a NANO-IPC file into a Dataset."""
        table = read_table(path)
        return Dataset(table)

    def to_json(self, path: str, lines: bool = True):
        """Write as JSONL (`lines=True`, default) or regular JSON (`lines=False`)"""
        with open(path, "w", encoding="utf-8") as fp:
            if lines:
                to_jsonl(fp, self._table)
            else:
                to_json(fp, self._table)

    def to_dict(self) -> List[Optional[dict]]:
        """Materialize the whole table as a list of row dicts."""
        return self._table.to_pylist()

    def select_columns(self, column_names: List[str]) -> "Dataset":
        return Dataset(_select_columns(self._table, column_names))

    def remove_columns(self, column_names: List[str]) -> "Dataset":
        return Dataset(_remove_columns(self._table, column_names))

    def select(self, indices: Sequence[int]) -> "Dataset":
        return Dataset(_slice_table(self._table, indices))

    def shuffle(self, seed: Optional[int] = None) -> "Dataset":
        rng = random.Random(seed)
        idx = list(range(len(self)))
        rng.shuffle(idx)
        return self.select(idx)

    def map(
        self,
        function: Callable[[Union[Dict[str, Any], List[Dict[str, Any]]]], Union[Dict[str, Any], List[Dict[str, Any]]]],
        batched: bool = False,
        batch_size: int = 1000,
    ) -> "Dataset":
        """Apply a function on rows (row-wise or in mini-batches) and rebuild the dataset.

        Educational implementation: materializes rows → applies → from_pylist.
        """
        rows = self._table.to_pylist()
        out_rows: List[Optional[Dict[str, Any]]] = []

        if not batched:
            for r in rows:
                out = function(r)
                out_rows.append(out)  # function returns one row (dict or None)
        else:
            for start in range(0, len(rows), batch_size):
                chunk = rows[start: start + batch_size]
                mapped = function(chunk)
                if not isinstance(mapped, list):
                    raise TypeError("When batched=True, `function` must return a list of rows.")
                out_rows.extend(mapped)

        return Dataset(Table([RecordBatch.from_pylist(out_rows)]))

    def filter(self, predicate: Callable[[Dict[str, Any]], bool]) -> "Dataset":
        rows = self._table.to_pylist()
        kept = [r for r in rows if (r is not None and predicate(r))]
        return Dataset(Table([RecordBatch.from_pylist(kept)]))


def load_dataset(data_files: Union[str, List[str]]) -> Dataset:
    """
    HF-like loader that infers format from file extension.

    Supported extensions:
      - .json              → from_json()
      - .jsonl / .ndjson   → from_jsonl()
      - .nano              → read_table()

    Accepts:
      - str                : single file → Dataset
      - list[str]          : multiple files (same schema) → concatenated Dataset
    """

    def _load_one(file: str) -> Table:
        e = _ext(file)
        if e == "json":
            return from_json(file)
        if e in ("jsonl", "ndjson"):
            return from_jsonl(file)
        if e == "nano":
            return read_table(file)
        raise ValueError(f"Unsupported extension for {file!r}. Expected .json, .jsonl/.ndjson, or .nano")

    def _load_many(files: Union[str, List[str]]) -> Dataset:
        flist = [files] if isinstance(files, str) else list(files)
        tables = [_load_one(f) for f in flist]
        table = _concat_tables(tables) if len(tables) > 1 else tables[0]
        return Dataset(table)

    if isinstance(data_files, (str, list)):
        return _load_many(data_files)

    raise TypeError("data_files must be str, list[str], or dict[str, str|list[str]].")


load_from_disk = load_dataset
