"""
Build a FAISS index from a documents.json corpus using a HuggingFace embedding model.

Default model: microsoft/harrier-oss-v1-0.6b
  - 1,024-dim L2-normalised embeddings
  - Sentence-transformers compatible
  - Queries use prompt_name="web_search_query"; documents use no prompt

Output (written to --output-dir, default: faiss_index/)
--------------------------------------------------------
index.faiss   — IndexFlatIP FAISS index (inner-product = cosine sim after L2 norm)
doc_ids.json  — Ordered list of document IDs; position i ↔ index row i

Usage
-----
    # Full corpus with default model
    python set_up_faiss_index.py --docs documents.json

    # Custom model, small batch (low VRAM), GPU
    python set_up_faiss_index.py \\
        --docs documents.json \\
        --output-dir faiss_index \\
        --model microsoft/harrier-oss-v1-0.6b \\
        --batch-size 8 \\
        --device cuda \\
        --dtype float16

    # Preview without writing files
    python set_up_faiss_index.py --docs documents.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import faiss
except ImportError as e:
    raise ImportError("Install faiss-cpu: pip install faiss-cpu") from e

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError("Install sentence-transformers: pip install sentence-transformers") from e

from data.document_store import DocumentStore


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# HFEmbeddings
# ---------------------------------------------------------------------------

class HFEmbeddings:
    """
    Thin wrapper around a SentenceTransformer model.

    Separates document encoding (no task prefix) from query encoding
    (uses prompt_name="web_search_query"), as required by asymmetric
    embedding models such as microsoft/harrier-oss-v1-0.6b.

    Parameters
    ----------
    model_name:
        HuggingFace model ID or local path.
    device:
        Torch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
        ``"auto"`` lets sentence-transformers pick the best available device.
    dtype:
        Model weight dtype (``"auto"``, ``"float32"``, ``"float16"``).
        ``"auto"`` uses the model's native dtype.
    """

    def __init__(
        self,
        model_name: str = "microsoft/harrier-oss-v1-0.6b",
        device: str = "auto",
        dtype: str = "auto",
    ) -> None:
        model_kwargs: dict = {}
        if dtype != "auto":
            model_kwargs["torch_dtype"] = dtype

        device_arg = None if device == "auto" else device

        _log(f"Loading model '{model_name}' (device={device}, dtype={dtype}) …")
        self._model = SentenceTransformer(
            model_name,
            model_kwargs=model_kwargs if model_kwargs else {"dtype": "auto"},
            device=device_arg,
        )
        _log("  Model loaded.")

    @property
    def dim(self) -> int:
        """Embedding dimensionality."""
        return self._model.get_sentence_embedding_dimension()

    def encode_documents(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of document passages.

        No task prefix is applied (asymmetric models encode documents
        without a prompt).

        Returns
        -------
        np.ndarray of shape (n, dim), dtype float32, L2-normalised.
        """
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return embeddings.astype(np.float32)

    def encode_queries(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode a list of queries using the ``web_search_query`` task prompt.

        Returns
        -------
        np.ndarray of shape (n, dim), dtype float32, L2-normalised.
        """
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            prompt_name="web_search_query",
        )
        return embeddings.astype(np.float32)


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def build_faiss_index(
    docs_path: str | Path,
    output_dir: str | Path,
    model_name: str = "microsoft/harrier-oss-v1-0.6b",
    batch_size: int = 32,
    device: str = "auto",
    dtype: str = "auto",
    dry_run: bool = False,
) -> tuple[faiss.Index, list[str]]:
    """
    Embed all documents in *docs_path* and build an IndexFlatIP FAISS index.

    Parameters
    ----------
    docs_path:
        Path to ``documents.json`` (output of ``csv_to_json.py`` or
        ``load_hf_dataset.py``).
    output_dir:
        Directory where ``index.faiss`` and ``doc_ids.json`` are written.
    model_name:
        HuggingFace model ID or local path passed to :class:`HFEmbeddings`.
    batch_size:
        Embedding batch size (lower = less VRAM).
    device:
        Torch device (``"auto"``, ``"cpu"``, ``"cuda"``).
    dtype:
        Model weight dtype (``"auto"``, ``"float32"``, ``"float16"``).
    dry_run:
        If True, embed documents but skip writing any files.

    Returns
    -------
    index : faiss.Index
        The built (and optionally serialised) FAISS index.
    doc_ids : list[str]
        Ordered document IDs; position i corresponds to index row i.
    """
    # ---- Load corpus ---------------------------------------------------------
    store = DocumentStore.from_json(docs_path)
    _log(f"Corpus loaded: {len(store)} documents from '{docs_path}'.")

    # ---- Embed ---------------------------------------------------------------
    embedder = HFEmbeddings(model_name=model_name, device=device, dtype=dtype)
    _log(f"Embedding {len(store)} documents (batch_size={batch_size}) …")
    embeddings = embedder.encode_documents(store.texts, batch_size=batch_size, show_progress=True)
    _log(f"  Embeddings shape: {embeddings.shape}")

    # ---- Build FAISS index ---------------------------------------------------
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    _log(f"  FAISS IndexFlatIP built: {index.ntotal} vectors, dim={dim}.")

    doc_ids: list[str] = store.ids  # position i → doc ID

    if dry_run:
        _log("Dry-run mode — no files written.")
        return index, doc_ids

    # ---- Write output --------------------------------------------------------
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    index_path = out / "index.faiss"
    faiss.write_index(index, str(index_path))
    _log(f"Wrote FAISS index → {index_path}")

    ids_path = out / "doc_ids.json"
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(doc_ids, f, ensure_ascii=False, indent=2)
    _log(f"Wrote doc IDs     → {ids_path}")

    return index, doc_ids


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed a documents.json corpus and build a FAISS index.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--docs",
        default="documents.json",
        metavar="PATH",
        help="Input corpus JSON (produced by csv_to_json.py or load_hf_dataset.py).",
    )
    parser.add_argument(
        "--output-dir",
        default="faiss_index",
        metavar="DIR",
        help="Output directory for index.faiss and doc_ids.json.",
    )
    parser.add_argument(
        "--model",
        default="microsoft/harrier-oss-v1-0.6b",
        metavar="MODEL",
        help="HuggingFace model ID or local path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="Embedding batch size (lower = less VRAM).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Torch device.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model weight dtype.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Embed documents but do not write any files.",
    )
    args = parser.parse_args()

    build_faiss_index(
        docs_path=args.docs,
        output_dir=args.output_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
        dtype=args.dtype,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
