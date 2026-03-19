# Advanced PDF Analyzer

A high-precision PDF analysis engine built for document question answering, extractive summarization, and concept explanation. Built on top of HuggingFace Transformers, Sentence Transformers, and spaCy with a hybrid retrieval pipeline and proper confidence scoring.

---

## Features

- **Sliding window chunking** — overlapping 400-token windows with 100-token stride; no content is silently truncated
- **Hybrid retrieval** — BM25 (lexical) and CrossEncoder (semantic) scores fused at `0.4 / 0.6` weight ratio
- **Correct span selection** — QA answers found via a full `(seq_len × seq_len)` upper-triangular score matrix, not a broken index slice
- **No-answer detection** — SQuAD 2.0 unanswerable-question pattern: CLS token score vs. best span score
- **Cross-encoder answer validation** — candidate answers cross-checked against all retrieved contexts before selection
- **Batched concept explanation** — flan-t5 explanations generated in a single batched call with per-concept retry fallback
- **Consistent device management** — single `DeviceManager` drives all models and pipelines; no hardcoded `device=0`
- **Normalised sentence scoring** — TF-IDF sentence importance normalised by term count, not raw sum

---

## Architecture

```
PDF File
   │
   ▼
extract_text_with_metadata()
   │  PyPDF2 page extraction + text cleaning
   │
   ▼
SlidingWindowChunker
   │  400-token windows, 100-token stride
   │  Tokenizer-aware splitting
   │
   ▼
PDFAugmentedRetriever
   │  BM25 (lexical) + CrossEncoder (semantic)
   │  Combined score: 0.4 × BM25 + 0.6 × semantic
   │
   ├──────────────────────────────────────┐
   ▼                                      ▼
answer_question()              analyze_document()
   │                                      │
   │  DeBERTa-v3 QA model                 │  TF-IDF + KMeans clustering
   │  Score matrix span selection         │  Normalised sentence scoring
   │  No-answer detection                 │  CrossEncoder calibration
   │  AnswerValidator cross-check         │
   │                                      ▼
   ▼                             List[DocumentResult]
DetailedExplainer
   │  spaCy concept extraction
   │  flan-t5-large batched explanation
   ▼
Final answer + explanations
```

---

## Models Used

| Role | Model | Notes |
|---|---|---|
| Question Answering | `deepset/deberta-v3-large-squad2` | Highest validation score (0.87) |
| Summarization | `facebook/bart-large-cnn` | Abstractive, generation pipeline |
| Retrieval (semantic) | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranking + answer validation |
| Concept Explanation | `google/flan-t5-large` | Instruction-tuned, batched inference |
| NLP (concept extraction) | `en_core_web_sm` | spaCy noun chunks + NER |

---

## Installation

```bash
pip install torch transformers sentence-transformers scikit-learn rank-bm25 PyPDF2 spacy pydantic tqdm
python -m spacy download en_core_web_sm
```

GPU inference is used automatically if CUDA is available. CPU fallback requires no configuration changes.

---

## Usage

### Document summarization

```python
from pdf_analyzer import AdvancedPDFAnalyzer

analyzer = AdvancedPDFAnalyzer()
result = analyzer.analyze_document("research_paper.pdf")

print(f"Average confidence: {result['avg_confidence']:.3f}")

for item in result['summary']:
    print(f"[Page {item.source_page}] ({item.confidence:.2f}) {item.content}")
```

### Question answering

```python
documents = analyzer.extract_text_with_metadata("research_paper.pdf")

answer = analyzer.answer_question(
    question="What methodology was used in the study?",
    documents=documents
)

print(answer['answer'])
print(f"Confidence: {answer['confidence']:.3f}")
print(f"Source page: {answer['page_number']}")

for concept, explanation in answer['explanations']['explanations'].items():
    print(f"\n{concept}:\n{explanation}")
```

---

## Confidence Scoring

Confidence in `answer_question` is computed in two stages:

1. **Span confidence** — `P(start_idx) × P(end_idx)` derived from the score matrix over the full sequence, not an arbitrary multiplier.
2. **Validation score** — the answer is scored against all retrieved contexts using the CrossEncoder. Final score is `0.7 × span_confidence + 0.3 × validation_score`.

Answers below a confidence threshold of `0.35` are flagged with `[Low Confidence]`.

Confidence in `analyze_document` is calibrated using normalised TF-IDF importance combined with intra-cluster CrossEncoder agreement:

```
calibrated = 0.7 × (sentence_score / max_score) + 0.3 × mean(cross_encoder_cluster_scores)
```

---

## Key Design Decisions

**Why sliding windows instead of page-level truncation?**
The QA model has a 512-token hard limit. Feeding full pages silently discards everything beyond that limit. Overlapping windows with stride ensure content near chunk boundaries is always captured by an adjacent window.

**Why a score matrix for span selection?**
The original `end_prob[0, max_start_idx:]` slice reindexes end positions relative to the start, producing incorrect absolute positions. The correct approach is to compute all `(start, end)` pair scores simultaneously via an outer product, then mask the lower triangle to enforce `end >= start`.

**Why cross-validate answers?**
A high span confidence from a single context can be a false positive. Validating the answer against all retrieved contexts using the CrossEncoder penalises answers that are locally fluent but globally unsupported.

**Why batch the explainer?**
The original loop called the pipeline once per concept, incurring tokenization, padding, and forward-pass overhead for each. Batching sends all prompts in a single call, with GPU memory allowing up to 8 prompts per batch.

---

## Project Structure

```
.
├── neura_pdf.py       # Main analysis engine
└── README.md             # This file
```

---

## Part of Ogent

This analyzer is the document processing and retrieval backend for **Ogent** — a RAG-based document chat system built with FastAPI and TypeScript React, developed under [Neura Lumina](https://neuralumina.com).

---

*Built for production-grade document intelligence. Not a tutorial wrapper.*
