# rag-9000

A modular retrieval evaluation framework. Build a corpus from a CSV or a HuggingFace dataset, run up to six retrieval strategies in one command, compare them on your own ground-truth queries, and visualise the results.

---

## Cookbook — end-to-end in five steps

Follow these steps in order. Every command is copy-pasteable.

### Step 0 — Install dependencies

```bash
pip install -r requirements.txt
```

All packages are pinned by minimum version. GPU is optional; everything runs on CPU.

---

### Step 1 — Prepare your corpus

Choose **Option A** (quickest) or **Option B** (your own data).

#### Option A — Download a HuggingFace dataset

```bash
python data/load_hf_dataset.py --sample-size 200
```

Default dataset: [`neural-bridge/rag-dataset-1200`](https://huggingface.co/datasets/neural-bridge/rag-dataset-1200) (Apache 2.0).

This writes two files automatically — **skip to Step 2**:
- `documents.json` — the document corpus
- `ground_truth.json` — query → relevant document IDs

Common flags:

| Flag | Default | Description |
|---|---|---|
| `--dataset REPO_ID` | `neural-bridge/rag-dataset-1200` | Any HuggingFace dataset ID |
| `--split SPLIT` | `train` | `train`, `test`, `validation`, or `all` |
| `--sample-size N` | _(full split)_ | Randomly sample N rows |
| `--deduplicate` | off | Collapse identical passages into one document |
| `--docs-out PATH` | `documents.json` | Output path for the corpus |
| `--gt-out PATH` | `ground_truth.json` | Output path for ground truth |
| `--dry-run` | off | Preview without writing any files |

#### Option B — Convert your own CSV

```bash
python data/csv_to_json.py \
    --input        websites.csv \
    --output       documents.json \
    --text-col     content \
    --id-col       url \
    --prepend-cols title
```

`--prepend-cols` concatenates extra columns (e.g. `title`) before the main text field with a blank line separator.

Common flags:

| Flag | Default | Description |
|---|---|---|
| `--input PATH` | _(required)_ | Input CSV file |
| `--output PATH` | `documents.json` | Output JSON file |
| `--text-col COL` | _(required)_ | Column containing the main document text |
| `--id-col COL` | _(auto: row index)_ | Column to use as document ID |
| `--prepend-cols COL …` | _(none)_ | Extra columns to prepend to the text |

Then **create `ground_truth.json` manually**:

```json
[
  {"query": "What is transfer learning?", "relevant_ids": ["https://example.com/transfer-learning"]},
  {"query": "GDPR compliance checklist",  "relevant_ids": ["https://example.com/gdpr", "https://example.com/compliance"]}
]
```

---

### Step 2 — Build the FAISS index

Required for any pipeline that uses the bi-encoder (dense retrieval). Skip if you plan to run BM25-only.

```bash
python set_up_faiss_index.py \
    --docs       documents.json \
    --output-dir faiss_index
```

This writes to `faiss_index/`:
- `index.faiss` — `IndexFlatIP` index (cosine similarity on L2-normalised embeddings)
- `doc_ids.json` — ordered document IDs; position `i` in the index = `doc_ids[i]`

Default model: [`microsoft/harrier-oss-v1-0.6b`](https://huggingface.co/microsoft/harrier-oss-v1-0.6b) (1 024-dim, MIT, 94 languages).

Common flags:

| Flag | Default | Description |
|---|---|---|
| `--model MODEL` | `microsoft/harrier-oss-v1-0.6b` | HF model ID or local path |
| `--batch-size N` | `32` | Embedding batch size (reduce if you run out of memory) |
| `--device DEVICE` | `auto` | `cpu`, `cuda`, `mps`, or `auto` |
| `--dtype DTYPE` | `auto` | `float32`, `float16`, or `auto` |
| `--dry-run` | off | Embed but do not write files |

---

### Step 3 — Run the evaluation

```bash
python run_evaluation.py \
    --docs  documents.json \
    --index faiss_index/index.faiss \
    --gt    ground_truth.json \
    --k     10
```

This evaluates all six preset pipelines and prints a comparison table sorted by NDCG@K:

```
| Pipeline                | Recall@10 | Precision@10 |    MRR | NDCG@10 | MAP@10 | HitRate@10 | Latency(ms) |
|-------------------------|-----------|--------------|--------|---------|--------|------------|-------------|
| hybrid_rrf_reranked     |    0.8714 |       0.1520 | 0.7341 |  0.7891 | 0.6823 |     0.9102 |       312.4 |
| hybrid_rrf              |    0.8421 |       0.1440 | 0.6912 |  0.7503 | 0.6401 |     0.8834 |        28.1 |
| bi_encoder_reranked     |    0.8102 |       0.1380 | 0.6714 |  0.7210 | 0.6103 |     0.8612 |       298.7 |
| ...                     |       ... |          ... |    ... |     ... |    ... |        ... |         ... |

Best pipeline (NDCG@10): hybrid_rrf_reranked
```

To save results for later analysis:

```bash
python run_evaluation.py \
    --docs  documents.json \
    --index faiss_index/index.faiss \
    --gt    ground_truth.json \
    --k     10 \
    --out-csv  results/eval.csv \
    --out-json results/eval.json
```

- `--out-csv` — one row per query × pipeline (per-query breakdown)
- `--out-json` — aggregated metrics per pipeline

To run only a subset of pipelines:

```bash
python run_evaluation.py \
    --docs      documents.json \
    --gt        ground_truth.json \
    --pipelines bm25_only hybrid_rrf
```

Pipelines requiring a FAISS index are automatically skipped when `--index` is not provided.

---

### Step 4 — Visualise the results

```bash
python visualise_evaluation.py \
    --json    results/eval.json \
    --csv     results/eval.csv \
    --out-dir results/plots
```

Four charts are written to `results/plots/`:

| File | What it shows | Requires |
|---|---|---|
| `metrics_bar.png` | Grouped bar: aggregate metrics per pipeline | `--json` |
| `radar.png` | Radar polygon: multi-metric comparison | `--json` |
| `latency_bar.png` | Mean query latency per pipeline | `--csv` |
| `metrics_boxplot.png` | Per-query metric distributions (box plots) | `--csv` |

To display charts interactively instead of saving:

```bash
python visualise_evaluation.py --json results/eval.json --show
```

---

### Step 5 — Run the tests

```bash
python -m pytest tests/ -v
```

All 119 tests are fully offline — no model downloads or network calls.

---

## Retrieval capabilities

| Capability | Role | When to use |
|---|---|---|
| **Bi-encoder** (FAISS) | First-stage dense retrieval | Semantic similarity, fast at scale |
| **BM25** | First-stage sparse retrieval | Keyword precision, no GPU needed |
| **Reciprocal Rank Fusion** | Merges bi-encoder + BM25 results | Hybrid: best of both worlds |
| **Cross-encoder** | Second-stage reranker | Higher accuracy on a small candidate pool |

Six preset pipelines are available out of the box:

| Pipeline | Stage 1 | Fusion | Stage 2 |
|---|---|---|---|
| `bi_encoder_only` | FAISS | — | — |
| `bm25_only` | BM25 | — | — |
| `bi_encoder_reranked` | FAISS | — | Cross-encoder |
| `bm25_reranked` | BM25 | — | Cross-encoder |
| `hybrid_rrf` | FAISS + BM25 | RRF | — |
| `hybrid_rrf_reranked` | FAISS + BM25 | RRF | Cross-encoder |

---

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

---

## Custom pipeline (Python API)

```python
from config import PipelineConfig
from data.document_store import DocumentStore
from retrieval.pipeline import build_pipeline
from evaluation.evaluator import Evaluator

store = DocumentStore.from_json("documents.json")

my_config = PipelineConfig(
    name="my_hybrid",
    retrievers=["bi_encoder", "bm25"],
    fusion="rrf",
    reranker="cross_encoder",
    top_k_retrieve=100,   # candidate pool size per retriever
    top_k_final=5,        # final results returned to the caller
    rrf_k=60,             # RRF smoothing constant (higher = smoother fusion)
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
result = evaluator.run()
print(result.summary_table())
```

---

## Project structure

```
rag-9000/
├── config.py                   # PipelineConfig dataclass + six preset configs
├── run_evaluation.py           # CLI entrypoint: evaluate all/subset of pipelines
├── set_up_faiss_index.py       # HFEmbeddings + FAISS index builder CLI
├── visualise_evaluation.py     # Chart generator (bar, radar, latency, boxplot)
├── requirements.txt
├── data/
│   ├── document_store.py       # In-memory corpus; O(1) id↔index lookups
│   ├── csv_to_json.py          # CSV → documents.json converter
│   └── load_hf_dataset.py      # HuggingFace dataset → documents.json + ground_truth.json
├── retrieval/
│   ├── base.py                 # RetrievalResult, BaseRetriever, BaseReranker ABCs
│   ├── bi_encoder.py           # Dense retrieval via FAISS + SentenceTransformer
│   ├── bm25_retriever.py       # Sparse retrieval via BM25Okapi
│   ├── rrf.py                  # Reciprocal Rank Fusion
│   ├── cross_encoder.py        # Cross-encoder reranker
│   └── pipeline.py             # RetrieverPipeline orchestrator + build_pipeline() factory
├── evaluation/
│   ├── metrics.py              # Per-query metric functions (recall, NDCG, MRR, …)
│   └── evaluator.py            # Multi-pipeline runner, comparison table, CSV/JSON export
└── tests/
    ├── test_document_store.py
    ├── test_csv_to_json.py
    ├── test_load_hf_dataset.py
    ├── test_metrics.py
    ├── test_evaluator.py
    ├── test_bm25_retriever.py
    ├── test_rrf.py
    ├── test_bi_encoder.py
    ├── test_cross_encoder.py
    ├── test_pipeline.py
    └── test_faiss_index.py
```

---

## Adding a new retriever or reranker

1. Subclass `BaseRetriever` or `BaseReranker` in [retrieval/base.py](retrieval/base.py).
2. Add the new name as a `Literal` type in [config.py](config.py) and handle it in `PipelineConfig.__post_init__`.
3. Instantiate it in `build_pipeline()` in [retrieval/pipeline.py](retrieval/pipeline.py).
4. Wire it into `RetrieverPipeline.run()`.
