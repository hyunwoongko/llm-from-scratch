import json
from typing import Any, Dict, List, Optional, TextIO, Union

from nanorlhf.nanosets.table.record_batch import RecordBatch
from nanorlhf.nanosets.table.table import Table


def materialize(obj: Union[Table, RecordBatch]) -> List[Optional[Dict[str, Any]]]:
    """
    Internal helper: obtain rows (materialized).

    Args:
        obj: Table or RecordBatch.

    Returns:
        rows via `to_pylist()`.

    Notes:
        - Rows are Python dicts (or None) with nested lists/structs already materialized.
        - Whole-object materialization (not streaming).
    """
    if isinstance(obj, (Table, RecordBatch)):
        return obj.to_pylist()
    raise TypeError(f"Unsupported object: {type(obj).__name__}")


def to_json(
    fp: TextIO,
    obj: Union[Table, RecordBatch],
    indent: Optional[int] = 2,
) -> None:
    """
    Write a Table or RecordBatch to a JSON file (rows-only).

    Output:
        [ { ... }, { ... }, ... ]

    Args:
        fp: Writable text file.
        obj: Table or RecordBatch.
        indent: Pretty-print indent (None for compact).
    """
    rows = materialize(obj)
    json.dump(rows, fp, ensure_ascii=False, indent=indent)


def to_jsonl(fp: TextIO, obj: Union[Table, RecordBatch]) -> None:
    """
    Write a Table or RecordBatch to a JSON Lines (JSONL) file.
    One JSON object per line; no schema.

    Args:
        fp: Writable text file.
        obj: Table or RecordBatch.
    """
    rows = materialize(obj)
    for row in rows:
        fp.write(json.dumps(row, ensure_ascii=False))
        fp.write("\n")


def from_json(path: str) -> Table:
    """
    Load a Table from a JSON file (rows-only list at root).

    Expected input:
        [ {"col": ...}, null, ... ]

    Args:
        path: UTF-8 JSON file path.

    Returns:
        Table: Single-batch table.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise TypeError("JSON root must be a list of rows (rows-only).")

    batch = RecordBatch.from_pylist(data)
    return Table([batch])


def from_jsonl(path: str) -> Table:
    """
    Load a Table from a JSON Lines (JSONL) file.
    One JSON object per line, or 'null' for a null row.

    Args:
        path: UTF-8 JSONL file path.

    Returns:
        Table: Single-batch table.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    batch = RecordBatch.from_pylist(rows)
    return Table([batch])
