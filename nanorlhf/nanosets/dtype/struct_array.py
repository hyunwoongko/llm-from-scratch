from typing import Optional, Dict, Any, List

from nanorlhf.nanosets.base.bitmap import Bitmap
from nanorlhf.nanosets.dtype.array import Array, ArrayBuilder
from nanorlhf.nanosets.dtype.dtype import STRUCT
from nanorlhf.nanosets.dtype.list_array import ListArrayBuilder, infer_child_builder
from nanorlhf.nanosets.dtype.primitive_array import infer_primitive_dtype, PrimitiveArrayBuilder
from nanorlhf.nanosets.dtype.string_array import StringArrayBuilder


def get_struct_array_builder_from_rows(rows: List[Optional[Dict[str, Any]]]) -> "StructArrayBuilder":
    """
    Infer and construct a `StructArrayBuilder` for a nested struct field from dict-like rows.

    Parameters:
        rows (List[Optional[Dict[str, Any]]]): Column values for a nested struct field,
            where each entry is a `dict` (row) or `None`.

    Returns:
        StructArrayBuilder: A builder configured with inferred field names and child builders.
        Use `append(row)` for each input row and `finish()` to obtain a `StructArray`.

    Examples:
        >>> rows = [{"id": 1, "name": "a"}, None, {"id": 2}]
        >>> sb = get_struct_array_builder_from_rows(rows)
        >>> for r in rows:
        ...     sb.append(r)
        >>> arr = sb.finish()
        >>> arr.to_pylist()
        [{'id': 1, 'name': 'a'}, None, {'id': 2, 'name': None}]

    Discussion:
        Q. How does this function work?
            This helper inspects a column of nested struct rows (`List[Optional[Dict[str, Any]]]`)
            to derive a fixed schema (ordered field names) and a corresponding set of child builders.

            Algorithm:
                1) Scan all non-null rows to collect field names in first-appearance order.
                2) Build per-field, column-aligned value lists (length = number of rows).
                   Missing keys are represented as `None`.
                3) For each field, call `inference_builder_for_column(values)` to obtain a child builder.
                4) Return `StructBuilder(field_names, child_builders, strict_keys=True)`.

            If all rows are `None` (no keys observed), a schema cannot be inferred.
            Implementations may choose to return an "empty" `StructBuilder` (no fields)
            or raise a `ValueError` depending on the desired behavior.
            (In the educational version, returning an empty `StructBuilder` is acceptable.)
    """
    inner_names: List[str] = []
    seen = set()
    for row in rows:
        if row is None:
            continue
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                inner_names.append(key)

    if not inner_names:
        # all None → default to empty struct with zero children (but still valid rows)
        # Education-friendly: make a struct with no fields; each append(None) still aligns lengths.
        return StructArrayBuilder([], [], strict_keys=False)

    # Build inner columns to infer builders per inner field.
    num_rows = len(rows)
    inner_columns: Dict[str, List[Optional[Any]]] = {name: [None] * num_rows for name in inner_names}
    for index, row in enumerate(rows):
        if row is None:
            continue
        for name in inner_names:
            inner_columns[name][index] = row.get(name, None)

    inner_child_builders = []
    for name in inner_names:
        inner_builder = inference_builder_for_column(inner_columns[name])
        inner_child_builders.append(inner_builder)

    return StructArrayBuilder(inner_names, inner_child_builders, strict_keys=False)


def inference_builder_for_column(values: List[Optional[Any]]):
    """
    Return a builder for a single struct field (column) by inspecting its values.

    Parameters:
        values (List[Optional[Any]]): Column-aligned values for a single field.
            Use `None` for missing entries at any row.

    Returns:
        Builder: An initialized builder for this field’s data type.

    Examples:
        >>> inference_builder_for_column([1, None, 2])
        PrimitiveBuilder(INT64)
        >>> inference_builder_for_column([["a"], [], None])
        ListBuilder(StringBuilder())
        >>> inference_builder_for_column([None, None])
        StringBuilder()

    Discussion:
        Q. How does this function work?
            This helper performs one-pass type inference over a field’s column (length `N`),
            where each entry corresponds to one parent row. The first non-`None` sample selects
            a candidate category; all other non-`None` entries must belong to the same category,
            otherwise a `TypeError` is raised.

            Inference rules:
                - `dict`:
                    → nested `StructBuilder` inferred via `get_struct_array_builder_from_rows(values)`
                - `list`/`tuple`:
                    → `ListBuilder` with child inferred by `infer_child_builder(values)`
                - `str`:
                    → `StringBuilder`
                - `bool`/`int`/`float`:
                    → `PrimitiveBuilder(infer_primitive_dtype(values))`
                - all `None`:
                    → `StringBuilder`

            No arrays are materialized here; the function returns a builder instance ready
            to receive values via `append(...)` and later finalize with `finish()`.
    """

    # Pick first non-None sample
    sample: Any = None
    for v in values:
        if v is not None:
            sample = v
            break

    # All None → choose a sensible default (StringBuilder).
    if sample is None:
        return StringArrayBuilder()

    # dict → nested StructBuilder (recursive)
    if isinstance(sample, dict):
        for v in values:
            if v is None:
                continue
            if not isinstance(v, dict):
                raise TypeError("Mixed types in struct field: expected dict or None.")
        return get_struct_array_builder_from_rows(values)

    # list/tuple → ListBuilder with inferred child builder
    if isinstance(sample, (list, tuple)):
        for v in values:
            if v is None:
                continue
            if not isinstance(v, (list, tuple)):
                raise TypeError("Mixed types in list field: expected list/tuple or None.")
        child_builder = infer_child_builder(values)  # uses your existing logic
        return ListArrayBuilder(child_builder)

    # str → StringBuilder
    if isinstance(sample, str):
        for v in values:
            if v is None:
                continue
            if not isinstance(v, str):
                raise TypeError("Mixed types in string field: expected str or None.")
        return StringArrayBuilder()

    # primitives → PrimitiveBuilder(dtype)
    if isinstance(sample, (bool, int, float)):
        for v in values:
            if v is None:
                continue
            if not isinstance(v, (bool, int, float)):
                raise TypeError("Mixed types in primitive field: expected bool/int/float or None.")
        dtype = infer_primitive_dtype(values)  # handles None entries
        return PrimitiveArrayBuilder(dtype)

    raise TypeError(f"Unsupported field type in struct: {type(sample).__name__}")


class StructArray(Array):
    """
    A column of fixed-shape records (structs) composed of child arrays.

    Args:
        names (List[str]): list of field names
        children (List[Array]): list of child arrays, it holds the actual data
        validity (Optional[Bitmap]): validity bitmap, if None all elements are valid

    Discussion:
        Q. How is StructArray represented dictionary-like format?
            StructArray holds a list of field names and a list of child arrays.
            Each child array is an Array instance which has same type defined in field names.

        Q. How is StructArray different from ListArray?
            StructArray has *no offsets*. Each row `i` maps directly to index `i` in every child array.
            ListArray uses `int32[n+1]` offsets to mark variable-length boundaries,
            but StructArray is a fixed-shape record: every row has the same set of fields.
    """

    def __init__(self, names: list[str], children: list[Array], validity=None):
        assert len(names) == len(children), "names` and `children` must have the same length"
        length = len(children[0]) if children else 0
        for child in children:
            assert len(child) == length, "All child arrays must have the same length"

        super().__init__(STRUCT, length, validity)
        self.names = names
        self.children = children

    def to_pylist(self) -> list:
        """
        Convert to a list of Python dicts, respecting struct-level nulls.

        Returns:
            List[Optional[Dict[str, Any]]]: Each row is either:
                - None (if the struct is null)
                - dict mapping field name -> field value (possibly None per child)
        """
        child_pylists = [ch.to_pylist() for ch in self.children]
        out = []
        for i in range(self.length):
            if self.is_null(i):
                out.append(None)
                continue
            row = {}
            for name, child_py in zip(self.names, child_pylists):
                row[name] = child_py[i]
            out.append(row)
        return out

    @classmethod
    def from_pylist(cls, data: List[Optional[Dict[str, Any]]], *, strict_keys: bool = False) -> "StructArray":
        """
        Build a `StructArray` from dict-like rows (or `None`) with schema + builder inference.

        Discussion:
            Q. How does this method work?
                Inference rules per field (column):
                  - `dict`            → `StructBuilder` (recursive)
                  - `list`/`tuple`    → `ListBuilder(infer_child_builder(...))`
                  - `str`             → `StringBuilder`
                  - `bool`/`int`/`float`  → `PrimitiveBuilder(infer_primitive_dtype(...))`
                  - all `None`        → `StringBuilder` (education-friendly default)

                Missing keys in a row are treated as `None`. Parent `None` marks the entire row null.
                Field order preserves first appearance order across non-null rows.
        """
        num_rows = len(data)

        # 1) Collect ordered field names from all non-null rows (first-appearance order).
        field_names: List[str] = []
        seen = set()
        for row in data:
            if row is None:
                continue
            if not isinstance(row, dict):
                raise TypeError(f"Each row must be dict or None, got {type(row).__name__}")
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    field_names.append(key)

        if not field_names:
            raise ValueError("Cannot infer struct fields: no keys found in non-null rows.")

        # 2) Build column-wise lists aligned to num_rows for builder inference.
        columns = {name: [None] * num_rows for name in field_names}
        parent_validity: List[int] = []

        for index, row in enumerate(data):
            if row is None:
                parent_validity.append(0)
                # children will get None at this index (already prefilled)
                continue
            if not isinstance(row, dict):
                raise TypeError(f"Row {index} must be dict or None, got {type(row).__name__}")
            parent_validity.append(1)
            for name in field_names:
                columns[name][index] = row.get(name, None)

        # 3) For each field (column), infer a child builder.
        child_builders: List = []
        for name in field_names:
            column_values = columns[name]
            builder = inference_builder_for_column(column_values)
            child_builders.append(builder)

        # 4) Stream rows through a StructBuilder, then finish.
        struct_array_builder = StructArrayBuilder(field_names, child_builders, strict_keys=strict_keys)
        for row in data:
            struct_array_builder.append(row)
        return struct_array_builder.finish()


class StructArrayBuilder(ArrayBuilder[Optional[Dict[str, Any]], StructArray]):
    """
    Incrementally builds a StructArray from dict-like rows.

    Args:
        names: Field names (order defines column order).
        child_builders: One builder per field, aligned with `names`.
        strict_keys: If True, raising on unexpected keys in input rows. If False, ignore them.

    Example:
        >>> from nanorlhf.nanosets.dtype.primitive_array import PrimitiveArrayBuilder, INT64
        >>> from nanorlhf.nanosets.dtype.string_array import StringArrayBuilder
        >>> from nanorlhf.nanosets.dtype.list_array import ListArrayBuilder
        >>> names = ["id", "name", "tags"]

        >>> sb = StructArrayBuilder(
        ...     names=["id", "name", "tags"],
        ...     child_builders=[PrimitiveArrayBuilder(INT64), StringArrayBuilder(), ListArrayBuilder(StringArrayBuilder())],
        ... )

        >>> sb.append({"id": 1, "name": "alice", "tags": ["ml", "rl"]})
        >>> sb.append(None)  # whole row null
        >>> sb.append({"id": 2, "name": "bob"})  # tags missing → None

        >>> arr = sb.finish()
        >>> assert arr.to_pylist() == [
        ...     {"id": 1, "name": "alice", "tags": ["ml", "rl"]},
        ...     None,
        ...     {"id": 2, "name": "bob", "tags": None},
        ... ]
        """

    def __init__(
        self,
        names: List[str],
        child_builders: List[ArrayBuilder],
        strict_keys: bool = True,
    ):
        if len(names) != len(child_builders):
            raise ValueError("`names` and `child_builders` must have the same length.")

        self.names = names
        self.child_builders = child_builders
        self.strict_keys = strict_keys
        self.validity: List[int] = []

    def append(self, row: Optional[Dict[str, Any]]) -> "StructArrayBuilder":
        """
        Append one struct row (`dict`) or `None` (null row).

        Parameters:
            row (Optional[Dict[str, Any]]): Input row or `None`.

        Discussion:
            Q. How does this method behave for different inputs?
                - If `row` is `None`: parent validity = `0`; every child gets `None`.
                - If `row` is `dict`: parent validity = `1`; distribute values to each child by name.
                  Missing key → `None`; Unexpected key → error (if `strict_keys=True`).
        """
        if row is None:
            self.validity.append(0)
            for child in self.child_builders:
                child.append(None)
            return self

        if not isinstance(row, dict):
            raise TypeError(f"Struct row must be dict or None, got {type(row).__name__}")

        if self.strict_keys:
            keys = set(row.keys())
            names_set = set(self.names)
            extras = sorted(keys - names_set)
            missing = [name for name in self.names if name not in keys]
            if extras or missing:
                raise KeyError(
                    f"Struct row keys mismatch. extras={extras}, missing={missing}"
                )

        self.validity.append(1)
        for name, child in zip(self.names, self.child_builders):
            value = row.get(name, None)
            child.append(value)
        return self

    def finish(self) -> StructArray:
        """
        Finalize and return a `StructArray`.

        Returns:
            StructArray: The finished array with aligned children and validity bitmap.
        """
        # Finalize each child builder
        children_arrays: List[Array] = []
        for builder in self.child_builders:
            children_arrays.append(builder.finish())

        # Validity bitmap (if any nulls)
        validity = Bitmap.from_pylist(self.validity)

        # All children must have same length as parent
        expected = len(self.validity)

        bad = [(name, len(arr)) for name, arr in zip(self.names, children_arrays) if len(arr) != expected]
        if bad:
            detail = ", ".join([f"{name}={ln}" for name, ln in bad])
            raise AssertionError(
                f"[StructBuilder] finished child lengths must match rows. "
                f"rows={expected}; lens({detail})"
            )

        # Create and return the StringArray
        return StructArray(self.names, children_arrays, validity)
