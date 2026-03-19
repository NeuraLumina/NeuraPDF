# neura_pdf

> High-precision PDF analysis engine with question answering, extractive summarization, and concept explanation — built for production RAG pipelines.

A product of **[Neura Lumina](https://neuralumina.com)** / Ogent.

---

## Features

- Hybrid BM25 + CrossEncoder retrieval with fused scoring
- Sliding window chunking — no content silently truncated at 512 tokens
- DeBERTa-v3 question answering with score-matrix span selection
- No-answer detection using the SQuAD 2.0 CLS-token pattern
- Cross-encoder answer validation across all retrieved contexts
- Batched flan-t5 concept explanation with per-concept retry fallback
- Consistent GPU/CPU device management throughout
- Three terminal interfaces: `summarize`, `ask`, and `chat`

---

## Installation

```bash
pip install torch transformers sentence-transformers scikit-learn rank-bm25 PyPDF2 spacy pydantic tqdm
python -m spacy download en_core_web_sm
```

GPU inference is used automatically if CUDA is available. No configuration changes are needed for CPU fallback.

---

## Usage

```
python neura_pdf.py <command> <pdf> [options]
```

### Commands at a glance

| Command | Alias | Description |
|---|---|---|
| `summarize` | `sum`, `s` | Summarize a PDF with confidence scores |
| `ask` | `a` | Ask one or more questions about a PDF |
| `chat` | `c` | Interactive chat session with a PDF |

---

## summarize

Runs the full document analysis pipeline and prints each summary point with its source page and confidence score.

```bash
python neura_pdf.py summarize paper.pdf
```

**Options**

| Flag | Description |
|---|---|
| `--evidence`, `-e` | Show supporting evidence passages under each summary point |
| `--output FILE`, `-o` | Save results to a JSON file |

**Examples**

```bash
python neura_pdf.py summarize paper.pdf

python neura_pdf.py summarize paper.pdf --evidence

python neura_pdf.py summarize paper.pdf --evidence --output summary.json
```

**Sample output**

```
SUMMARY  —  paper.pdf
────────────────────────────────────────────────────────────
  Chunks processed : 42
  Avg confidence   : 0.731

  [1] Page 2  confidence 0.841
  The proposed method achieves state-of-the-art results on three
  benchmark datasets by combining sparse and dense retrieval.

  [2] Page 5  confidence 0.612
  Evaluation was conducted on 1,200 annotated question-answer pairs
  drawn from four domains.
```

---

## ask

Answers one or more questions about a PDF. Supports single questions, batch questions, and an interactive loop.

```bash
python neura_pdf.py ask paper.pdf -q "What methodology was used?"
```

**Options**

| Flag | Description |
|---|---|
| `--question QUESTION`, `-q` | Question to ask (repeatable for multiple) |
| `--interactive`, `-i` | Drop into an interactive question loop |
| `--explain`, `-e` | Print concept explanations under each answer |
| `--output FILE`, `-o` | Append results to a JSON file |

**Examples**

```bash
python neura_pdf.py ask paper.pdf -q "What is the main finding?"

python neura_pdf.py ask paper.pdf \
  -q "What methodology was used?" \
  -q "What were the results?" \
  -q "Who funded the study?" \
  --output answers.json

python neura_pdf.py ask paper.pdf --interactive --explain
```

**Sample output**

```
QUESTION
────────────────────────────────────────────────────────────
  What methodology was used?

  Answer  (page 3  confidence 0.786)
  The authors used a hybrid retrieval approach combining BM25
  with a cross-encoder reranker fine-tuned on MS MARCO.

  Concept explanations:
    hybrid retrieval
      Hybrid retrieval refers to combining multiple retrieval
      strategies — typically sparse (BM25) and dense (neural)
      methods — to improve recall and precision simultaneously...
```

---

## chat

A persistent interactive session. Type questions freely and receive answers attributed to Ogent. Type `history` at any point to replay the full conversation.

```bash
python neura_pdf.py chat paper.pdf
```

**Options**

| Flag | Description |
|---|---|
| `--save-history FILE` | Save the full conversation to a JSON file on exit |

**Examples**

```bash
python neura_pdf.py chat paper.pdf

python neura_pdf.py chat paper.pdf --save-history session.json
```

**Session commands**

| Input | Action |
|---|---|
| Any text | Ask a question |
| `history` | Print all previous turns |
| `exit` / `quit` / `q` | End the session |

**Sample session**

```
DOCUMENT CHAT  —  paper.pdf
────────────────────────────────────────────────────────────
  48 chunks loaded. Type 'history' to review, 'exit' to quit.

  You: What problem does this paper solve?

  Ogent:  page 1  confidence 0.803
  The paper addresses the lack of efficient multi-hop reasoning
  in open-domain question answering systems.

  You: How does it compare to previous work?

  Ogent:  page 7  confidence 0.694
  It outperforms prior methods by 6.2 F1 points on HotpotQA
  while using 40% fewer parameters.
```

---

## Output format

All commands accept `--output` / `--save-history` to persist results as JSON alongside terminal output.

**summarize output**

```json
{
  "file": "paper.pdf",
  "avg_confidence": 0.731,
  "summary": [
    {
      "page": 2,
      "confidence": 0.841,
      "content": "The proposed method achieves...",
      "supporting_evidence": ["...", "..."]
    }
  ]
}
```

**ask output**

```json
[
  {
    "question": "What methodology was used?",
    "answer": "The authors used a hybrid retrieval approach...",
    "confidence": 0.786,
    "page_number": 3,
    "explanations": {
      "concepts": ["hybrid retrieval"],
      "explanations": { "hybrid retrieval": "..." }
    }
  }
]
```

**chat history output**

```json
[
  { "question": "What problem does this paper solve?", "answer": "...", "confidence": 0.803 },
  { "question": "How does it compare to previous work?", "answer": "...", "confidence": 0.694 }
]
```

---

## Confidence scoring

Confidence scores are colour-coded across all commands:

| Colour | Range | Meaning |
|---|---|---|
| 🟢 Green | ≥ 0.70 | High confidence |
| 🟡 Yellow | 0.40 – 0.69 | Moderate confidence |
| 🔴 Red | < 0.40 | Low confidence — treat with caution |

Answers below `0.35` are prefixed with `[Low Confidence]` in the output.

Confidence is computed as `0.7 × span_score + 0.3 × cross_encoder_validation_score`, not an arbitrary multiplier.

---

## Models used

| Role | Model |
|---|---|
| Question answering | `deepset/deberta-v3-large-squad2` |
| Summarization | `facebook/bart-large-cnn` |
| Retrieval / reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Concept explanation | `google/flan-t5-large` |
| NLP / concept extraction | `en_core_web_sm` (spaCy) |

---

## Project structure

```
.
├── neura_pdf.py    # Full analysis engine + CLI
└── README.md       # This file
```

---

## License

Copyright (c) 2026 Neura Lumina

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).

You are free to use, share, and adapt this software for non-commercial purposes, provided that appropriate credit is given to Neura Lumina and a link to the original project is included.

Commercial use of this software in any form is strictly prohibited without prior written permission from Neura Lumina.

Full license: https://creativecommons.org/licenses/by-nc/4.0/

---

*Part of the Neura Lumina platform — [neuralumina.com](https://neuralumina.com)*
