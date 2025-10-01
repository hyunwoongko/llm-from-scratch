import json
from typing import Any, Dict, List, Optional, TextIO, Union

from nanorlhf.nanosets.dtype.array import Array
from nanorlhf.nanosets.dtype.dtype import DataType, FMT, STRING, LIST, STRUCT
from nanorlhf.nanosets.dtype.list_array import ListArray
from nanorlhf.nanosets.dtype.primitive_array import PrimitiveArray
from nanorlhf.nanosets.dtype.string_array import StringArray
from nanorlhf.nanosets.dtype.struct_array import StructArray
from nanorlhf.nanosets.table.field import Field
from nanorlhf.nanosets.table.record_batch import RecordBatch
from nanorlhf.nanosets.table.schema import Schema
from nanorlhf.nanosets.table.table import Table


def _rows_and_schema_from_obj(obj: Union[Table, RecordBatch]) -> (List[Dict[str, Any]], Schema):
    """
    Internal helper: obtain (rows, schema) without extra copies.

    Args:
        obj: Table or RecordBatch.

    Returns:
        (rows, schema): rows via `to_pylist()`, and the object's schema.

    Notes:
        - Rows come out as Python dicts with nested lists/structs materialized.
        - This is whole-object materialization (not streaming).
    """
    if isinstance(obj, Table):
        return obj.to_pylist(), obj.schema
    if isinstance(obj, RecordBatch):
        return obj.to_pylist(), obj.schema
    raise TypeError(f"Unsupported object: {type(obj).__name__}")


def _schema_to_json(schema: Schema) -> Dict[str, Any]:
    """
    Encode Schema → JSON-friendly dict (dtype stored as {'kind': <name>}).

    Args:
        schema: Schema instance.

    Returns:
        Dict[str, Any]: JSON-ready representation of the schema.
    """
    return {
        "fields": [
            {"name": f.name, "dtype": {"kind": f.dtype.name}, "nullable": f.nullable}
            for f in schema.fields
        ]
    }


def _schema_from_json(obj: Dict[str, Any]) -> Schema:
    """
    Decode JSON-friendly dict → Schema.

    Args:
        obj: Dict with shape {"fields": [{"name":..., "dtype":{"kind":...}, "nullable":...}, ...]}

    Returns:
        Schema: Parsed Schema instance.

    Raises:
        ValueError: if dtype metadata is malformed.
    """
    items = obj.get("fields", [])
    parsed: List[Field] = []
    for fd in items:
        name = fd["name"]
        kind = fd.get("dtype", {}).get("kind")
        if not isinstance(kind, str):
            raise ValueError("Schema dtype meta must contain {'kind': <str>}.")
        parsed.append(Field(name=name, dtype=DataType(kind), nullable=fd.get("nullable", True)))
    return Schema(tuple(parsed))


def _build_array_from_values(dtype: DataType, values: List[Any]) -> Array:
    """
    Materialize a column Array from Python values using a top-level dtype.

    Behavior:
        - Primitive dtypes (dtype in FMT): pass dtype to PrimitiveArray.from_pylist for exact coercion.
        - STRING: StringArray.from_pylist(values)
        - LIST:   ListArray.from_pylist(values)   (element shape inferred internally)
        - STRUCT: StructArray.from_pylist(values) (child fields inferred from dict keys)

    Args:
        dtype: Top-level DataType (e.g., int64, string, list, struct).
        values: Column-aligned Python values (None allowed).

    Returns:
        Array: Concrete column array instance.
    """
    if dtype in FMT:
        return PrimitiveArray.from_pylist(values, dtype=dtype)
    if dtype is STRING or dtype.name == "string":
        return StringArray.from_pylist(values)
    if dtype is LIST or dtype.name == "list":
        return ListArray.from_pylist(values)
    if dtype is STRUCT or dtype.name == "struct":
        return StructArray.from_pylist(values)
    raise TypeError(f"Unsupported dtype in JSON schema: {dtype!r}")


def to_json(fp: TextIO, obj: Union[Table, RecordBatch], *, include_schema: bool = True,
            indent: Optional[int] = 2) -> None:
    """
    Write a Table or RecordBatch to a JSON file (optionally including schema).

    This serializes data as a JSON object:
        {
          "schema": { ... },   # optional, if include_schema=True
          "rows": [ { ... }, { ... }, ... ]
        }

    Args:
        fp: Writable *text* file-like object (opened with mode 'w' and proper encoding).
        obj: Table or RecordBatch.
        include_schema: If True, include a 'schema' object with field names, dtypes, and nullability.
        indent: Indentation level for pretty-printing (None for compact).

    Notes:
        - Rows are produced by `to_pylist()`, so nested Lists/Structs appear as Python lists/dicts.
        - This is a whole-file write (not streaming). For streaming, prefer `to_jsonl()`.
    """
    rows, schema = _rows_and_schema_from_obj(obj)
    payload: Dict[str, Any] = {"rows": rows}
    if include_schema:
        payload["schema"] = _schema_to_json(schema)
    json.dump(payload, fp, ensure_ascii=False, indent=indent)


def to_jsonl(fp: TextIO, obj: Union[Table, RecordBatch]) -> None:
    """
    Write a Table or RecordBatch to a JSON Lines (JSONL) file.

    Each row dictionary is written on its own line as a standalone JSON object.
    This is suited for streaming pipelines (append-friendly, line-by-line processing).

    Args:
        fp: Writable *text* file-like object (opened with mode 'w' and proper encoding).
        obj: Table or RecordBatch.

    Notes:
        - Schema is not written in JSONL; if needed, emit it separately.
        - Each line ends with a single '\\n'.
    """
    rows, _ = _rows_and_schema_from_obj(obj)
    for row in rows:
        fp.write(json.dumps(row, ensure_ascii=False))
        fp.write("\n")


def from_json(path: str) -> Table:
    """
    Load a Table from a JSON file.

    Supported input formats:
        1) Full object with schema:
            {
              "schema": {"fields": [{"name": "...", "dtype": {"kind": "int64"}, "nullable": true}, ...]},
              "rows":   [ {"col": ...}, null, ... ]
            }

        2) Rows-only (schema omitted):
            [ {"col": ...}, null, ... ]

    Behavior:
        - If 'schema' exists, we build each column according to that top-level dtype
          and the field order in the schema, by slicing values from rows:
              col_values = [ row.get(name) if row is not None else None ]
          Then:
              * Primitive → PrimitiveArray.from_pylist(col_values, dtype=schema_dtype)
              * STRING   → StringArray.from_pylist(col_values)
              * LIST     → ListArray.from_pylist(col_values)     (element shape inferred)
              * STRUCT   → StructArray.from_pylist(col_values)   (child fields inferred)
        - If no 'schema', we infer entirely via `RecordBatch.from_pylist(rows)`.

    Args:
        path: Path to the JSON file (UTF-8).

    Returns:
        Table: A single-batch table constructed from the rows.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Case 2) rows-only (array at top)
    if isinstance(data, list):
        rows: List[Optional[Dict[str, Any]]] = data  # type: ignore[assignment]
        batch = RecordBatch.from_pylist(rows)
        return Table([batch])

    # Case 1) object with optional schema + rows
    if not isinstance(data, dict):
        raise TypeError("JSON root must be either a list of rows or an object with {'rows': ...}.")

    rows = data.get("rows")
    if not isinstance(rows, list):
        raise TypeError("'rows' must be a list of row dicts or nulls.")

    schema_obj = data.get("schema")
    if schema_obj is None:
        # No schema → full inference
        batch = RecordBatch.from_pylist(rows)
        return Table([batch])

    # Schema present → honor field order and top-level dtypes
    schema = _schema_from_json(schema_obj)
    arrays: List[Array] = []

    # Column-aligned extraction, preserving field order
    for field in schema.fields:
        name = field.name
        col_values: List[Any] = []
        for r in rows:
            if r is None:
                col_values.append(None)
            else:
                if not isinstance(r, dict):
                    raise TypeError(f"Each row must be dict or None, got {type(r).__name__}")
                col_values.append(r.get(name, None))

        arrays.append(_build_array_from_values(field.dtype, col_values))
    batch = RecordBatch(Schema(tuple(schema.fields)), arrays)
    return Table([batch])


def from_jsonl(path: str) -> Table:
    """
    Load a Table from a JSON Lines (JSONL) file.

    Each line must be a standalone JSON object representing one row,
    or the literal 'null' for a null row.

    Behavior:
        - Schema is not expected inside JSONL; dtypes and nullability are inferred from the rows.
        - Use `to_jsonl()` for writing.

    Args:
        path: Path to the JSONL file (UTF-8).

    Returns:
        Table: A single-batch table constructed from the rows.
    """
    rows: List[Optional[Dict[str, Any]]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    batch = RecordBatch.from_pylist(rows)
    return Table([batch])
