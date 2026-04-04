"""
Convert a CSV file of website records to the JSON document format expected
by the retrieval pipeline.

Expected CSV columns (configurable via --text-col / --id-col):
    url, title, content   (minimum)

Output JSON schema (list of documents):
[
  {
    "id":       "unique-document-id",
    "text":     "the text that will be embedded / indexed",
    "metadata": { ...all other CSV columns... }
  },
  ...
]

Usage
-----
    python data/csv_to_json.py \
        --input  websites.csv \
        --output documents.json \
        --text-col content \
        --id-col url
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def csv_to_documents(
    csv_path: str | Path,
    text_col: str = "content",
    id_col: str | None = None,
    extra_text_cols: list[str] | None = None,
    encoding: str = "utf-8",
) -> list[dict]:
    """
    Load a CSV and return a list of document dicts.

    Parameters
    ----------
    csv_path:
        Path to the input CSV file.
    text_col:
        Column whose value becomes the ``text`` field (the chunk that gets
        embedded and searched).
    id_col:
        Column to use as document ``id``. If None, a zero-based integer index
        is used.
    extra_text_cols:
        Additional columns to concatenate into ``text`` (e.g. title).
        They are prepended: "<title>\n\n<content>".
    encoding:
        CSV file encoding.

    Returns
    -------
    list of dicts with keys ``id``, ``text``, ``metadata``.
    """
    df = pd.read_csv(csv_path, encoding=encoding)
    df.columns = df.columns.str.strip()

    if text_col not in df.columns:
        raise ValueError(
            f"text column '{text_col}' not found. Available: {list(df.columns)}"
        )

    # Build the text field
    text_parts = []
    if extra_text_cols:
        for col in extra_text_cols:
            if col not in df.columns:
                raise ValueError(f"extra text column '{col}' not found.")
            text_parts.append(df[col].fillna("").astype(str))
    text_parts.append(df[text_col].fillna("").astype(str))

    texts = text_parts[0]
    for part in text_parts[1:]:
        texts = texts + "\n\n" + part

    # Build IDs
    if id_col:
        if id_col not in df.columns:
            raise ValueError(f"id column '{id_col}' not found.")
        ids = df[id_col].astype(str).tolist()
    else:
        ids = [str(i) for i in range(len(df))]

    # All columns except text_col become metadata
    meta_cols = [c for c in df.columns if c not in ([text_col] + (extra_text_cols or []))]

    documents = []
    for i, (doc_id, text) in enumerate(zip(ids, texts.tolist())):
        metadata = {col: df.iloc[i][col] for col in meta_cols}
        # Ensure JSON-serialisable types
        metadata = {
            k: (v if not hasattr(v, "item") else v.item())
            for k, v in metadata.items()
        }
        documents.append({"id": doc_id, "text": text, "metadata": metadata})

    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert CSV to pipeline JSON format.")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    parser.add_argument(
        "--text-col", default="content", help="Column to use as document text (default: content)"
    )
    parser.add_argument(
        "--id-col", default=None, help="Column to use as document ID (default: row index)"
    )
    parser.add_argument(
        "--prepend-cols",
        nargs="*",
        default=None,
        help="Extra columns to prepend to the text field (e.g. --prepend-cols title)",
    )
    parser.add_argument("--encoding", default="utf-8", help="CSV encoding (default: utf-8)")
    args = parser.parse_args()

    documents = csv_to_documents(
        csv_path=args.input,
        text_col=args.text_col,
        id_col=args.id_col,
        extra_text_cols=args.prepend_cols,
        encoding=args.encoding,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(documents)} documents to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
