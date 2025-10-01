import os
from typing import Dict, Any, List, Optional

from nanorlhf.nanosets import load_dataset


def show(title: str, ds, n: int = 2, cols: Optional[List[str]] = None):
    print(f"=== {title} ===")
    view = ds.select_columns(cols).to_dict()[:n] if cols else ds.to_dict()[:n]
    for i, row in enumerate(view):
        print(f"{i}: {row}")
    print()


def main():
    # 0) Input file (assume gsm8k JSONL is available)
    src_jsonl = "data/gsm8k.jsonl"
    os.makedirs("data", exist_ok=True)

    # 1) Load JSONL → Dataset
    ds = load_dataset(src_jsonl)
    show("HEAD (original)", ds, n=2)

    # 2) Save as NANO-IPC & reload
    nano_path = "data/gsm8k.nano"
    ds.save_to_disk(nano_path)
    del ds
    ds = load_dataset(nano_path)  # auto-detected by .nano extension
    show("HEAD (reloaded)", ds, n=2)

    # 3) Column projection (select specific columns)
    ds_q = ds.select_columns(["question"])
    show("Select only 'question'", ds_q, n=2)

    # 4) Drop columns
    ds_no_q = ds.remove_columns(["question"])
    show("Removed 'question'", ds_no_q, n=2)

    # 5) Row subset by indices
    first_5 = ds.select(range(5))
    show("First 5 rows", first_5, n=5)

    # 6) Shuffle (fixed seed for reproducibility)
    ds_shuf = ds.shuffle(seed=42)
    show("Shuffled (seed=42)", ds_shuf, n=3)

    # 7) filter: keep rows where the question is short
    def short_q(row: Dict[str, Any]) -> bool:
        if row is None:
            return False
        q = (row.get("question") or "").strip()
        return len(q) < 120

    ds_short = ds.filter(short_q)
    show("Filtered: question length < 120", ds_short, n=3)

    # 8) map (row-wise): add derived columns (question_len, answer_len)
    def add_lengths(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        q = (row.get("question") or "").strip()
        a = (row.get("answer") or "").strip()
        new = dict(row)
        new["question_len"] = len(q)
        new["answer_len"] = len(a)
        return new

    ds_len = ds.map(add_lengths, batched=False)
    show("Map (row-wise): add_lengths", ds_len, n=3, cols=["question", "question_len", "answer_len"])

    # 9) map (batched): simple preprocessing (lowercase, trim)
    def normalize_batch(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for r in rows:
            if r is None:
                out.append(None)
                continue
            q = (r.get("question") or "").strip().lower()
            a = (r.get("answer") or "").strip()
            out.append({"question": q, "answer": a})
        return out

    ds_norm = ds.map(normalize_batch, batched=True, batch_size=100)
    show("Map (batched): normalize question", ds_norm, n=3)

    # 10) Chaining example: shuffle → select subset → select columns → add derived columns → save JSON
    out = (
        ds.shuffle(seed=1)
        .select(range(10))
        .select_columns(["question", "answer"])
        .map(add_lengths, batched=False)
    )
    show("Chained ops (shuffle→select→select_columns→map)", out, n=3)

    # 11) Save JSONL (lines=True)
    ds_len.to_json("data/gsm8k_len.jsonl", lines=True)

    # 11) Save JSON (lines=False)
    ds_len.to_json("data/gsm8k_len.json", lines=False)

    # 12) Load/concatenate multiple files (example)
    files = ["data/gsm8k.jsonl"] * 7
    ds_many = load_dataset(files)
    print(f"=== len original: {len(ds)}, len concatenated: {len(ds_many)} ===")


if __name__ == "__main__":
    main()
