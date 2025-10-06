import os
import random
from typing import List, Optional, Union, Callable, Dict, Any, Sequence

from nanorlhf.nanosets.io.ipc import read_table, write_table
from nanorlhf.nanosets.io.json_io import from_json, from_jsonl, to_json, to_jsonl
from nanorlhf.nanosets.table.record_batch import RecordBatch
from nanorlhf.nanosets.table.table import Table


def _concat_tables(tables: List[Table]) -> Table:
    """
    Concatenate multiple Tables by appending their RecordBatches.
    All tables must share the same schema.

    Args:
        tables (List[Table]): List of tables to concatenate.

    Returns:
        Table: A new table containing all rows from the input tables.
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
    """
    Create a new table by selecting a subset of rows by indices (simple, educational implementation).
    This materializes rows to Python then rebuilds a table. Fine for education, not optimized.

    Args:
        table (Table): The input table to slice.
        indices (Sequence[int]): List of row indices to select.

    Returns:
        Table: A new table containing only the selected rows.
    """
    rows = table.to_pylist()
    picked = [rows[i] for i in indices]
    return Table([RecordBatch.from_pylist(picked)])  # schema inferred again


def _select_columns(table: Table, keep: List[str]) -> Table:
    """
    Return table with only the selected columns (rebuilds a single-batch table for simplicity).

    Args:
        table (Table): The input table.
        keep (List[str]): List of column names to retain.

    Returns:
        Table: A new table containing only the specified columns.
    """
    rows = table.to_pylist()
    kept = [{k: r.get(k, None) if r is not None else None for k in keep} for r in rows]
    return Table([RecordBatch.from_pylist(kept)])


def _remove_columns(table: Table, drop: List[str]) -> Table:
    """
    Return table with specified columns removed (rebuilds a single-batch table for simplicity).

    Args:
        table (Table): The input table.
        drop (List[str]): List of column names to remove.

    Returns:
        Table: A new table without the specified columns.
    """
    rows = table.to_pylist()
    pruned = []
    for r in rows:
        if r is None:
            pruned.append(None)
        else:
            pruned.append({k: v for k, v in r.items() if k not in drop})
    return Table([RecordBatch.from_pylist(pruned)])


def _ext(path: str) -> str:
    """
    Return lowercase file extension without the leading dot.

    Args:
        path (str): The file path.

    Returns:
        str: The file extension in lowercase, or empty string if none.
    """
    base = os.path.basename(path)
    if "." not in base:
        return ""
    return base.rsplit(".", 1)[1].lower()


class Dataset:
    """
    A lightweight, HF-like facade around a single `Table`.

    Args:
        table (Table): The underlying Table instance.
    """

    def __init__(self, table: Table):
        self._table = table

    @property
    def table(self) -> Table:
        """
        Access the underlying Table instance.

        Returns:
            Table: The underlying table.
        """
        return self._table

    def __len__(self) -> int:
        """
        Return the number of rows in the dataset.

        Returns:
            int: Number of rows.
        """
        return self._table.length

    def __repr__(self):
        """
        String representation showing number of rows and schema

        Returns:
            str: Representation string.
        """
        return f"Dataset(num_rows={len(self)}, schema={self._table.schema})"

    def save_to_disk(self, path: str):
        """
        Save as NANO-IPC file.

        Args:
            path (str): Output file path.
        """
        with open(path, "wb") as fp:
            write_table(fp, self._table)

    def to_json(self, path: str, lines: bool = True):
        """
        Write as JSON or JSONL file.

        Args:
            path (str): Output file path.
            lines (bool): Whether to write as JSONL (one JSON object per line) or a single JSON array.
        """
        with open(path, "w", encoding="utf-8") as fp:
            if lines:
                to_jsonl(fp, self._table)
            else:
                to_json(fp, self._table)

    def to_dict(self) -> List[Optional[dict]]:
        """
        Materialize the whole table as a list of row dicts.

        Returns:
            List[Optional[dict]]: List of rows, where each row is a dict or None.
        """
        return self._table.to_pylist()

    def select_columns(self, column_names: List[str]) -> "Dataset":
        """
        Return a new Dataset with only the specified columns.

        Args:
            column_names (List[str]): List of column names to retain.

        Returns:
            Dataset: A new dataset with only the selected columns.
        """
        return Dataset(_select_columns(self._table, column_names))

    def remove_columns(self, column_names: List[str]) -> "Dataset":
        """
        Return a new Dataset with the specified columns removed.

        Args:
            column_names (List[str]): List of column names to remove.

        Returns:
            Dataset: A new dataset without the specified columns.
        """
        return Dataset(_remove_columns(self._table, column_names))

    def select(self, indices: Sequence[int]) -> "Dataset":
        """
        Return a new Dataset with only the specified row indices.

        Args:
            indices (Sequence[int]): List of row indices to select.

        Returns:
            Dataset: A new dataset containing only the selected rows.
        """
        return Dataset(_slice_table(self._table, indices))

    def shuffle(self, seed: Optional[int] = None) -> "Dataset":
        """
        Return a new Dataset with rows shuffled randomly.

        Args:
            seed (Optional[int]): Random seed for reproducibility.

        Returns:
            Dataset: A new dataset with rows in random order.
        """
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
        """
        Apply a function on rows (row-wise or in mini-batches) and rebuild the dataset.

        Args:
            function (Callable): The mapping function. If `batched=False`, it takes a single row dict and returns a row dict or None.
                                 If `batched=True`, it takes a list of row dicts and returns a list of row dicts (same length).
            batched (bool): Whether to apply the function in batches.
            batch_size (int): Size of each batch if `batched=True`.

        Returns:
            Dataset: A new dataset with the mapped rows.
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

    Args:
        data_files (Union[str, List[str]]): Single file path or list of file paths to load.

    Returns:
        Dataset: Loaded dataset.

    Discussion:
        Q. What extensions are supported?
          - .json              → from_json()
          - .jsonl / .ndjson   → from_jsonl()
          - .nano              → read_table()

        Q. Can I load multiple files?
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
