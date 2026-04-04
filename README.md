# rag-9000

A modular retrieval evaluation framework. Build a corpus from a CSV of websites, run multiple retrieval strategies, and compare them on your own ground-truth queries to find the optimal pipeline.

## Retrieval capabilities

| Capability | Role | When to use |
|---|---|---|
| **Bi-encoder** (FAISS) | First-stage dense retrieval | Semantic similarity, fast at scale |
| **BM25** | First-stage sparse retrieval | Keyword precision, no GPU needed |
| **Reciprocal Rank Fusion** | Merges bi-encoder + BM25 results | Hybrid: gets the best of both |
| **Cross-encoder** | Second-stage reranker | Higher accuracy on a small candidate pool |

These can be combined freely. Six presets are provided out of the box:

| Pipeline | Stage 1 | Fusion | Stage 2 |
|---|---|---|---|
| `bi_encoder_only` | FAISS | — | — |
| `bm25_only` | BM25 | — | — |
| `bi_encoder_reranked` | FAISS | — | Cross-encoder |
| `bm25_reranked` | BM25 | — | Cross-encoder |
| `hybrid_rrf` | FAISS + BM25 | RRF | — |
| `hybrid_rrf_reranked` | FAISS + BM25 | RRF | Cross-encoder |

## Installation

```bash
pip install -r requirements.txt
```

## Quickstart

### Option A — Use a HuggingFace dataset (fastest)

Download and convert a public RAG dataset in one command:

```bash
pip install datasets
python data/load_hf_dataset.py --sample-size 200
```

Default dataset: [`neural-bridge/rag-dataset-1200`](https://huggingface.co/datasets/neural-bridge/rag-dataset-1200) (context / question / answer, Apache 2.0).  
This writes `documents.json` and `ground_truth.json` directly — skip to step 2.

**Common options:**

| Flag | Default | Description |
|---|---|---|
| `--dataset` | `neural-bridge/rag-dataset-1200` | Any HuggingFace dataset ID |
| `--split` | `train` | `train`, `test`, `validation`, `all` |
| `--sample-size N` | _(full split)_ | Randomly sample N rows |
| `--batch-size N` | `64` | Rows per processing batch |
| `--deduplicate` | off | Collapse identical passages |
| `--dry-run` | off | Preview without writing files |

### Option B — Convert your own CSV

```bash
python data/csv_to_json.py \
    --input   websites.csv \
    --output  documents.json \
    --text-col content \
    --id-col   url \
    --prepend-cols title
```

`--prepend-cols` concatenates extra columns (e.g. title) before the main text field. The output is a JSON array where each entry has `id`, `text`, and `metadata`.

### 2. Build your FAISS index

Use the provided script to embed your corpus and write the index:

```bash
python set_up_faiss_index.py --docs documents.json --output-dir faiss_index
```

This writes two files to `faiss_index/`:
- `index.faiss` — `IndexFlatIP` FAISS index (cosine similarity on L2-normalised embeddings)
- `doc_ids.json` — ordered list of document IDs; position `i` ↔ index row `i`

Default model: [`microsoft/harrier-oss-v1-0.6b`](https://huggingface.co/microsoft/harrier-oss-v1-0.6b) (1,024-dim, MIT licence, 94 languages).

**Common options:**

| Flag | Default | Description |
|---|---|---|
| `--model` | `microsoft/harrier-oss-v1-0.6b` | HF model ID or local path |
| `--batch-size N` | `32` | Embedding batch size (lower = less VRAM) |
| `--device` | `auto` | `cpu`, `cuda`, `mps`, or `auto` |
| `--dtype` | `auto` | `float32`, `float16`, or `auto` |
| `--dry-run` | off | Embed but do not write files |

### 3. Create ground-truth queries

If you used `load_hf_dataset.py`, `ground_truth.json` is already generated. For a custom corpus, create it manually:

```json
[
  {"query": "What is transfer learning?", "relevant_ids": ["https://example.com/transfer-learning"]},
  {"query": "GDPR compliance checklist",  "relevant_ids": ["https://example.com/gdpr", "https://example.com/compliance"]}
]
```

### 4. Run the evaluation

```bash
python run_evaluation.py \
    --docs  documents.json \
    --index faiss_index/index.faiss \
    --gt    ground_truth.json \
    --k     10
```

This evaluates all six presets and prints a comparison table sorted by NDCG@10:

```
| Pipeline                | Recall@10 | Precision@10 |    MRR | NDCG@10 | MAP@10 | HitRate@10 | Latency(ms) |
|-------------------------|-----------|--------------|--------|---------|--------|------------|-------------|
| hybrid_rrf_reranked     |    0.8714 |       0.1520 | 0.7341 |  0.7891 | 0.6823 |     0.9102 |       312.4 |
| hybrid_rrf              |    0.8421 |       0.1440 | 0.6912 |  0.7503 | 0.6401 |     0.8834 |        28.1 |
| bi_encoder_reranked     |    0.8102 |       0.1380 | 0.6714 |  0.7210 | 0.6103 |     0.8612 |       298.7 |
| ...                     |       ... |          ... |    ... |     ... |    ... |        ... |         ... |

Best pipeline (NDCG@10): hybrid_rrf_reranked
```

### 5. Save results

```bash
python run_evaluation.py ... --out-csv results/eval.csv --out-json results/eval.json
```

`--out-csv` writes one row per query per pipeline (useful for per-query analysis).  
`--out-json` writes the aggregated metrics per pipeline.

## Evaluating a subset of pipelines

```bash
python run_evaluation.py \
    --docs documents.json --gt ground_truth.json \
    --pipelines bm25_only hybrid_rrf
```

Pipelines that require a FAISS index (bi-encoder variants) are automatically skipped if `--index` is not provided.

## Custom pipeline

```python
from config import PipelineConfig
from data import DocumentStore
from retrieval.pipeline import build_pipeline
from evaluation.evaluator import Evaluator

store = DocumentStore.from_json("documents.json")

my_config = PipelineConfig(
    name="my_hybrid",
    retrievers=["bi_encoder", "bm25"],
    fusion="rrf",
    reranker="cross_encoder",
    top_k_retrieve=100,   # candidate pool per retriever
    top_k_final=5,        # results returned
    rrf_k=60,             # RRF smoothing constant
)

pipeline = build_pipeline(
    config=my_config,
    store=store,
    faiss_index_path="faiss_index/index.faiss",
    bi_encoder_model="sentence-transformers/all-mpnet-base-v2",
    cross_encoder_model="cross-encoder/ms-marco-electra-base",
)

evaluator = Evaluator(pipelines={"my_hybrid": pipeline}, k=5)
evaluator.load_ground_truth("ground_truth.json")
result = evaluator.compare()
```

## Metrics

All metrics are computed at rank K (default: 10).

| Metric | Description |
|---|---|
| **Recall@K** | Fraction of relevant documents found in the top K |
| **Precision@K** | Fraction of the top K results that are relevant |
| **MRR** | Reciprocal rank of the first relevant result |
| **NDCG@K** | Normalised Discounted Cumulative Gain (position-weighted) |
| **MAP@K** | Mean Average Precision |
| **HitRate@K** | 1 if any relevant document appears in top K, else 0 |

## Testing

```bash
python -m pytest tests/ -v
```

Tests are fully offline — no model downloads or network calls required.

| Test class | Coverage |
|---|---|
| `TestFaissIndex` | Index loading, nearest-neighbour search, doc_ids alignment, DocumentStore round-trip |
| `TestHFEmbeddingsInterface` | Import, dim property, prompt usage for queries vs documents, float32 output |

## Project structure

```
rag-9000/
├── config.py               # PipelineConfig dataclass + preset configs
├── run_evaluation.py       # CLI entrypoint
├── set_up_faiss_index.py   # HFEmbeddings class + FAISS index builder
├── data/
│   ├── load_hf_dataset.py  # HuggingFace dataset → documents.json + ground_truth.json
│   ├── csv_to_json.py      # CSV → documents.json converter
│   └── document_store.py   # In-memory corpus with ID↔index lookups
├── retrieval/
│   ├── base.py             # RetrievalResult, BaseRetriever, BaseReranker
│   ├── bi_encoder.py       # FAISS dense retrieval
│   ├── bm25_retriever.py   # BM25 sparse retrieval
│   ├── rrf.py              # Reciprocal Rank Fusion
│   ├── cross_encoder.py    # Cross-encoder reranker
│   └── pipeline.py         # Orchestrator + build_pipeline() factory
├── evaluation/
│   ├── metrics.py          # Per-query metric computation
│   └── evaluator.py        # Multi-pipeline runner + reporting
└── tests/
    └── test_faiss_index.py # Unittests for HFEmbeddings + FAISS index pipeline
```
