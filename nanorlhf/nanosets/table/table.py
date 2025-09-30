from typing import List, Optional, Dict, Any

from nanorlhf.nanosets.table.record_batch import RecordBatch
from nanorlhf.nanosets.table.schema import Schema


class Table:
    """
    A `Table` is a logical collection of one or more `RecordBatch` objects
    that all share the same `Schema`.

    Args:
        batches (list[RecordBatch]): The list of `RecordBatch` instances forming this table.

    Notes:
        - There must be at least one batch.
        - All batches must share the exact same `Schema`.
        - `self.length` is the sum of row counts across all batches.

    Examples:
        >>> from nanorlhf.nanosets.table.schema import Schema
        >>> from nanorlhf.nanosets.table.field import Field
        >>> from nanorlhf.nanosets.data_type.data_type import INT32, FLOAT32, STRING
        >>> from nanorlhf.nanosets.data_type.primitive_array import PrimitiveArray
        >>> from nanorlhf.nanosets.data_type.string_array import StringArray

        >>>
        >>> # Define schema
        >>> schema = Schema((
        ...     Field("x", INT32),
        ...     Field("y", FLOAT32),
        ...     Field("z", STRING),
        ... ))
        >>>
        >>> # Create arrays
        >>> arr_x = PrimitiveArray.from_pylist([1, 2, 3], dtype=INT32)
        >>> arr_y = PrimitiveArray.from_pylist([0.1, 0.2, 0.3], dtype=FLOAT32)
        >>> arr_z = StringArray.from_pylist(["hi", "there", "!"])
        >>>
        >>> # Create Table
        >>> table = Table.from_arrays(schema, [arr_x, arr_y, arr_z])
        >>> print(table.to_pylist())
        [{'x': 1, 'y': 0.1, 'z': 'hi'}, {'x': 2, 'y': 0.2, 'z': 'there'}, {'x': 3, 'y': 0.3, 'z': '!'}]

    Discussion:
        Q. When to use `Table` vs `RecordBatch`?
            Use `RecordBatch` for a single chunk (e.g., a streaming unit or a compute partition).
            Use `Table` when you need a unified logical dataset across multiple chunks,
            while preserving batch-level execution benefits.
    """

    def __init__(self, batches: List[RecordBatch]):
        assert batches, "Table must have at least one RecordBatch"
        self.schema = batches[0].schema
        for b in batches:
            assert b.schema == self.schema, "All RecordBatches must have the same schema"
        self.batches = batches
        self.length = sum(b.length for b in batches)

    @classmethod
    def from_pylist(cls, rows: List[Optional[Dict[str, Any]]]) -> "Table":
        """
        Construct a `Table` from a list of Python dictionaries, one per row.

        Args:
            rows (list[dict]): List of row dictionaries mapping field name -> value.

        Returns:
            Table: A `Table` instance containing the data from the input rows.
        """
        batch = RecordBatch.from_pylist(rows)
        return cls([batch])

    @classmethod
    def from_batches(cls, batches: List[RecordBatch]) -> "Table":
        """
        Construct a `Table` from a list of `RecordBatch` instances.

        Args:
            batches (list[RecordBatch]): List of `RecordBatch` instances to form the table.

        Returns:
            Table: A `Table` instance containing the data from the input batches.
        """
        return cls(batches)

    def to_pylist(self) -> List[dict]:
        """
        Convert the entire `Table` into a list of Python dictionaries, one per row.

        Returns:
            list[dict]: A list of row dictionaries mapping field name -> value.

        Discussion:
            Implementation details:
                1) For each batch, each column is converted once via `to_pylist()`.
                2) Rows are assembled by indexing into those per-column Python lists.
                3) This keeps per-element overhead minimal within a batch,
                   but still constructs Python objects for each row.

                Example:
                    Suppose a table has:
                        x = [1, 2, 3]
                        y = [0.1, 0.2, 0.3]
                        z = ["hi", "arrow", "world"]

                    Step 1: Each column is converted to a Python list:
                        cols = [
                            [1, 2, 3],
                            [0.1, 0.2, 0.3],
                            ["hi", "arrow", "world"]
                        ]

                    Step 2: Each row is assembled as a dictionary:
                        Row 0 → {'x': 1, 'y': 0.1, 'z': 'hi'}
                        Row 1 → {'x': 2, 'y': 0.2, 'z': 'arrow'}
                        Row 2 → {'x': 3, 'y': 0.3, 'z': 'world'}

                    Step 3: The result is a list of dictionaries:
                        [
                            {'x': 1, 'y': 0.1, 'z': 'hi'},
                            {'x': 2, 'y': 0.2, 'z': 'arrow'},
                            {'x': 3, 'y': 0.3, 'z': 'world'}
                        ]

            Warning:
                For large datasets, this operation can be memory and allocation heavy.
                Prefer columnar operations or batch-wise iteration where possible.

        Examples:
            >>> rows = table.to_pylist()
            >>> rows[0]
            {'x': 1, 'y': 0.1, 'z': 'hi'}
        """
        rows = []
        for b in self.batches:
            cols = [c.to_pylist() for c in b.columns]
            for r in range(b.length):
                row = {}
                for f, col in zip(b.schema.fields, cols):
                    row[f.name] = col[r]
                rows.append(row)
        return rows
