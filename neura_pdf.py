# neura_pdf.py
import os
import re
import json
import torch
import numpy as np
import logging
import argparse
import sys
import textwrap
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    pipeline,
    LogitsProcessor,
    LogitsProcessorList,
)
from sentence_transformers import CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import PyPDF2
from sklearn.cluster import KMeans
import spacy

# -------------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("NeuraPDF")

# -------------------------------------------------------------------
# Configuration (can be overridden via environment variables)
# -------------------------------------------------------------------
@dataclass
class Config:
    qa_model_name: str = os.getenv("NEURAPDF_QA_MODEL", "deepset/deberta-v3-large-squad2")
    summarizer_name: str = os.getenv("NEURAPDF_SUMMARIZER", "facebook/bart-large-cnn")
    explainer_name: str = os.getenv("NEURAPDF_EXPLAINER", "google/flan-t5-base") #google/flan-t5-large
    cross_encoder_name: str = os.getenv("NEURAPDF_CROSS_ENCODER", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    spacy_model: str = os.getenv("NEURAPDF_SPACY_MODEL", "en_core_web_sm")
    max_model_cache: int = int(os.getenv("NEURAPDF_MAX_MODELS", "3"))
    chunk_window: int = int(os.getenv("NEURAPDF_CHUNK_WINDOW", "400"))
    chunk_stride: int = int(os.getenv("NEURAPDF_CHUNK_STRIDE", "100"))
    retrieval_top_k: int = int(os.getenv("NEURAPDF_RETRIEVAL_TOP_K", "5"))
    no_answer_threshold: float = float(os.getenv("NEURAPDF_NO_ANSWER_THRESHOLD", "0.0"))
    confidence_low_threshold: float = float(os.getenv("NEURAPDF_LOW_CONFIDENCE", "0.35"))
    temperature: float = float(os.getenv("NEURAPDF_TEMPERATURE", "0.85"))
    device: Optional[torch.device] = None

# -------------------------------------------------------------------
# Device manager
# -------------------------------------------------------------------
class DeviceManager:
    def __init__(self, config: Config):
        if config.device is not None:
            self._device = config.device
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self._device}")
        self._pipeline_device = 0 if self._device.type == "cuda" else -1

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def pipeline_device(self) -> int:
        return self._pipeline_device

    def move(self, model: torch.nn.Module) -> torch.nn.Module:
        return model.to(self._device)

    def move_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.to(self._device) for k, v in inputs.items()}

# -------------------------------------------------------------------
# Simple LRU model cache
# -------------------------------------------------------------------
class ModelCache:
    def __init__(self, max_models: int):
        self._cache: Dict[str, object] = {}
        self._access_order: List[str] = []
        self.max_models = max_models

    def get(self, key: str):
        if key in self._cache:
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None

    def put(self, key: str, value: object):
        if key in self._cache:
            self._access_order.remove(key)
        elif len(self._cache) >= self.max_models:
            lru_key = self._access_order.pop(0)
            evicted = self._cache.pop(lru_key)
            if hasattr(evicted, "cpu"):
                evicted.cpu()
            del evicted
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.debug(f"Evicted model '{lru_key}' from cache")
        self._cache[key] = value
        self._access_order.append(key)

# -------------------------------------------------------------------
# Logits processor for temperature scaling
# -------------------------------------------------------------------
class TemperatureScaler(LogitsProcessor):
    def __init__(self, temperature: float):
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        self.temperature = temperature

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        return scores / self.temperature

# -------------------------------------------------------------------
# Data models
# -------------------------------------------------------------------
class DocumentResult(BaseModel):
    content: str
    confidence: float
    source_page: int
    supporting_evidence: List[str]

# -------------------------------------------------------------------
# Sliding window chunker
# -------------------------------------------------------------------
class SlidingWindowChunker:
    def __init__(self, tokenizer, window_size: int, stride: int):
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.stride = stride

    def chunk(self, text: str, page_number: int) -> List[Dict]:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + self.window_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append({
                "content": chunk_text,
                "metadata": {
                    "page": page_number,
                    "token_start": start,
                    "token_end": end,
                    "chunk_index": len(chunks)
                }
            })
            if end == len(tokens):
                break
            start += self.stride
        return chunks

# -------------------------------------------------------------------
# Retriever (BM25 + Cross-Encoder fusion)
# -------------------------------------------------------------------
class PDFAugmentedRetriever:
    def __init__(self, documents: List[Dict], cross_encoder: CrossEncoder):
        self.documents = documents
        self.cross_encoder = cross_encoder
        texts = [doc["content"] for doc in documents]
        self.bm25 = BM25Okapi([t.split() for t in texts])
        self.tfidf = TfidfVectorizer(stop_words="english").fit(texts)

    def retrieve(self, query: str, top_k: int) -> List[Dict]:
        texts = [doc["content"] for doc in self.documents]
        bm25_scores = self.bm25.get_scores(query.split())
        pairs = [(query, t) for t in texts]
        semantic_scores = self.cross_encoder.predict(pairs)
        combined = 0.4 * bm25_scores + 0.6 * semantic_scores
        top_indices = np.argsort(combined)[-top_k:][::-1]
        return [
            {**self.documents[i], "retrieval_score": float(combined[i])}
            for i in top_indices
        ]

# -------------------------------------------------------------------
# Answer validator
# -------------------------------------------------------------------
class AnswerValidator:
    def __init__(self, cross_encoder: CrossEncoder):
        self.cross_encoder = cross_encoder

    def validate(self, answer: str, contexts: List[str]) -> float:
        if not answer.strip() or not contexts:
            return 0.0
        pairs = [(answer, ctx) for ctx in contexts]
        scores = self.cross_encoder.predict(pairs)
        norm = 1.0 / (1.0 + np.exp(-np.array(scores)))
        return float(np.max(norm))

    def is_no_answer(self, start_logits: torch.Tensor, end_logits: torch.Tensor, threshold: float) -> bool:
        cls_start = start_logits[0, 0].item()
        cls_end   = end_logits[0, 0].item()
        best_start = start_logits[0].max().item()
        best_end   = end_logits[0].max().item()
        return (cls_start + cls_end) > (best_start + best_end) + threshold

# -------------------------------------------------------------------
# Seq2Seq pipeline replacement
# text2text-generation was removed in newer transformers versions;
# this wrapper provides an identical call interface using the model directly.
# -------------------------------------------------------------------
class _Seq2SeqPipeline:
    """Drop-in replacement for pipeline('text2text-generation', ...)."""

    def __init__(self, model_name: str, device: torch.device):
        logger.info(f"Loading seq2seq explainer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device).eval()
        self.device = device

    def __call__(
        self,
        inputs,
        max_length: int = 250,
        min_length: int = 80,
        num_beams: int = 4,
        do_sample: bool = False,
        batch_size: int = 8,
        **kwargs,
    ):
        single = isinstance(inputs, str)
        texts = [inputs] if single else inputs
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                out = self.model.generate(
                    **enc,
                    max_new_tokens=max_length,
                    min_new_tokens=min(min_length, max_length),
                    num_beams=num_beams,
                    do_sample=do_sample,
                    early_stopping=True,
                )
            for ids in out:
                text = self.tokenizer.decode(ids, skip_special_tokens=True)
                results.append({"generated_text": text})
        return results[0] if single else results

class DetailedExplainer:
    def __init__(self, device_manager: DeviceManager, config: Config, model_cache: ModelCache):
        self.nlp = None
        try:
            self.nlp = spacy.load(config.spacy_model)
        except OSError:
            logger.warning(f"Spacy model '{config.spacy_model}' not found. Attempting download...")
            try:
                spacy.cli.download(config.spacy_model)
                self.nlp = spacy.load(config.spacy_model)
            except Exception as e:
                logger.error(f"Could not load or download spacy model: {e}. Concept extraction disabled.")
        self.device_manager = device_manager
        self.model_cache = model_cache
        self.explainer = self._load_explainer(config)

    def _load_explainer(self, config: Config):
        key = f"explainer:{config.explainer_name}"
        cached = self.model_cache.get(key)
        if cached is None:
            cached = _Seq2SeqPipeline(
                model_name=config.explainer_name,
                device=self.device_manager.device
            )
            self.model_cache.put(key, cached)
        return cached

    def extract_concepts(self, text: str) -> List[str]:
        if not self.nlp:
            return []
        doc = self.nlp(text)
        concepts = set()
        for chunk in doc.noun_chunks:
            if len(chunk) > 1 and not chunk.root.is_stop:
                concepts.add(chunk.text.strip())
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "NORP", "EVENT", "WORK_OF_ART"]:
                concepts.add(ent.text.strip())
        return list(concepts)

    def _build_prompt(self, concept: str, context: str, depth: int = 1) -> str:
        constraints = "in depth" if depth == 1 else "with more technical precision, citing specific mechanisms"
        return (
            f"Explain the concept '{concept}' {constraints} using the following context.\n"
            f"Your explanation must be at least 3 sentences and reference the context directly.\n"
            f"Context:\n{context}\n"
        )

    def explain_concept(self, concept: str, context: str, min_length: int = 80, max_retries: int = 2) -> str:
        for attempt in range(max_retries + 1):
            prompt = self._build_prompt(concept, context, depth=attempt + 1)
            result = self.explainer(
                prompt,
                max_length=250 + attempt * 50,
                min_length=min_length + attempt * 20,
                do_sample=False,
                num_beams=4
            )
            # Fix: handle both single dict and list-of-dicts return types
            if isinstance(result, list):
                explanation = result[0]["generated_text"].strip()
            else:
                explanation = result["generated_text"].strip()
            if len(explanation.split()) >= min_length // 4:
                return explanation
        return explanation

    def explain_text(self, text: str, context: str) -> Dict:
        concepts = self.extract_concepts(text)
        if not concepts:
            return {"concepts": [], "explanations": {}}
        prompts = [self._build_prompt(c, context) for c in concepts]
        results = self.explainer(
            prompts,
            max_length=250,
            min_length=80,
            do_sample=False,
            num_beams=4,
            batch_size=min(8, len(prompts))
        )
        explanations = {}
        for concept, result in zip(concepts, results):
            # When batch input is used, result is always a dict
            explanation = result["generated_text"].strip()
            if len(explanation.split()) < 20:
                explanation = self.explain_concept(concept, context)
            explanations[concept] = explanation
        return {"concepts": concepts, "explanations": explanations}

# -------------------------------------------------------------------
# Main analyzer class (stateless – retriever passed explicitly)
# -------------------------------------------------------------------
class NeuraPDFAnalyzer:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.device_manager = DeviceManager(self.config)
        self._model_cache = ModelCache(self.config.max_model_cache)

        logger.info(f"Loading QA model: {self.config.qa_model_name}")
        self.qa_tokenizer = AutoTokenizer.from_pretrained(self.config.qa_model_name)
        qa_model = AutoModelForQuestionAnswering.from_pretrained(self.config.qa_model_name)
        self.qa_model = self.device_manager.move(qa_model.eval().half())

        logger.info(f"Loading CrossEncoder: {self.config.cross_encoder_name}")
        self.cross_encoder = CrossEncoder(self.config.cross_encoder_name)

        self.answer_validator = AnswerValidator(self.cross_encoder)
        self.explainer = DetailedExplainer(self.device_manager, self.config, self._model_cache)

        # Summarizer not used in current implementation – skip loading
        self.summarizer = None

        self.chunker = SlidingWindowChunker(
            self.qa_tokenizer,
            self.config.chunk_window,
            self.config.chunk_stride
        )

        self.temperature_processor = LogitsProcessorList([TemperatureScaler(self.config.temperature)])

    # -------------------------------------------------------------------
    # PDF processing
    # -------------------------------------------------------------------
    def extract_text_with_metadata(self, file_path: str) -> Tuple[List[Dict], PDFAugmentedRetriever]:
        logger.info(f"Processing {file_path}")
        documents = []
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(tqdm(reader.pages, desc="Extracting pages")):
                try:
                    text = page.extract_text()
                    if not text or not text.strip():
                        continue
                    cleaned = self._clean_text(text)
                    chunks = self.chunker.chunk(cleaned, i + 1)
                    for chunk in chunks:
                        chunk["metadata"].update({
                            "source": os.path.basename(file_path),
                            "char_count": len(chunk["content"]),
                            "word_count": len(chunk["content"].split()),
                        })
                    documents.extend(chunks)
                except Exception as e:
                    logger.warning(f"Page {i+1} extraction error: {e}")
        if not documents:
            raise ValueError("No extractable content found in PDF")
        retriever = PDFAugmentedRetriever(documents, self.cross_encoder)
        return documents, retriever

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"[\x00-\x1F\x7F-\x9F]", " ", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
        return text.strip()

    # -------------------------------------------------------------------
    # Summarization (custom clustering/TF-IDF method)
    # -------------------------------------------------------------------
    def analyze_document(self, file_path: str) -> Dict:
        documents, retriever = self.extract_text_with_metadata(file_path)
        summary = self._generate_summary_with_confidence(documents, retriever)
        return {
            "document_metadata": [doc["metadata"] for doc in documents],
            "summary": summary,
            "avg_confidence": float(np.mean([s.confidence for s in summary])) if summary else 0.0
        }

    def _generate_summary_with_confidence(self, documents: List[Dict], retriever: PDFAugmentedRetriever) -> List[DocumentResult]:
        sentences = [
            (doc["metadata"]["page"], s.strip())
            for doc in documents
            for s in doc["content"].split(". ")
            if len(s.split()) > 6
        ]
        if not sentences:
            return []
        pages, texts = zip(*sentences)
        vectorizer = TfidfVectorizer(max_features=500)
        X = vectorizer.fit_transform(texts)
        n_clusters = min(5, len(texts))
        summary_results = []
        for cluster in self._cluster_text(X, n_clusters):
            cluster_items = [(pages[i], texts[i]) for i in cluster]
            cluster_texts = [t for _, t in cluster_items]
            scores = self._cross_validate_sentences(cluster_texts)
            best_idx = int(np.argmax(scores))
            best_page, best_sent = cluster_items[best_idx]
            raw_score = scores[best_idx]
            confidence = float(1.0 / (1.0 + np.exp(-raw_score)))
            summary_results.append(DocumentResult(
                content=best_sent,
                confidence=round(min(confidence, 0.95), 4),
                source_page=best_page,
                supporting_evidence=self._find_supporting_evidence(best_sent, retriever)
            ))
        return summary_results

    def _cluster_text(self, X, n_clusters: int) -> List[List[int]]:
        if X.shape[0] < n_clusters:
            return [[i] for i in range(X.shape[0])]
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        clusters = [[] for _ in range(n_clusters)]
        for idx, label in enumerate(labels):
            clusters[label].append(idx)
        return [c for c in clusters if c]

    def _cross_validate_sentences(self, sentences: List[str]) -> List[float]:
        if not sentences:
            return []
        if len(sentences) == 1:
            return [1.0]
        cluster_context = " ".join(sentences)
        pairs = [(s, cluster_context) for s in sentences]
        scores = self.cross_encoder.predict(pairs)
        return scores.tolist() if hasattr(scores, "tolist") else list(scores)

    def _find_supporting_evidence(self, sentence: str, retriever: PDFAugmentedRetriever, top_k: int = 2) -> List[str]:
        results = retriever.retrieve(sentence, top_k=top_k)
        return [r["content"] for r in results]

    # -------------------------------------------------------------------
    # Question answering
    # -------------------------------------------------------------------
    def answer_question(self, question: str, retriever: PDFAugmentedRetriever) -> Dict:
        relevant_docs = retriever.retrieve(question, top_k=self.config.retrieval_top_k)

        answers = []
        for doc in relevant_docs:
            context = doc["content"]
            retrieval_score = doc["retrieval_score"]
            inputs = self.qa_tokenizer(
                question,
                context,
                add_special_tokens=True,
                return_tensors="pt",
                max_length=512,
                truncation="only_second",
                return_overflowing_tokens=False
            )
            inputs = self.device_manager.move_inputs(inputs)

            with torch.no_grad():
                outputs = self.qa_model(**inputs)
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits

                if self.answer_validator.is_no_answer(start_logits, end_logits, self.config.no_answer_threshold):
                    continue

                start_probs = torch.nn.functional.softmax(start_logits, dim=-1)
                end_probs = torch.nn.functional.softmax(end_logits, dim=-1)

                seq_len = start_probs.size(1)
                score_matrix = torch.triu(start_probs[0].unsqueeze(1) * end_probs[0].unsqueeze(0))
                best_span = score_matrix.argmax()
                start_idx = (best_span // seq_len).item()
                end_idx = (best_span % seq_len).item()
                span_confidence = float(score_matrix[start_idx, end_idx])

                normalised_retrieval = float(1.0 / (1.0 + np.exp(-retrieval_score)))
                confidence = span_confidence * normalised_retrieval

                answer_tokens = inputs["input_ids"][0][start_idx:end_idx + 1]
                answer = self.qa_tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
                if not answer:
                    continue

                explanations_result = self.explainer.explain_text(answer, context)

                answers.append({
                    "answer": answer,
                    "confidence": confidence,
                    "span_confidence": span_confidence,
                    "context": context,
                    "page_number": doc["metadata"]["page"],
                    "explanations": explanations_result
                })

        if not answers:
            return {"answer": "No confident answer found", "confidence": 0.0, "explanations": {}}

        all_contexts = [a["context"] for a in answers]
        for a in answers:
            validation_score = self.answer_validator.validate(a["answer"], all_contexts)
            a["confidence"] = 0.7 * a["confidence"] + 0.3 * validation_score

        best = max(answers, key=lambda x: x["confidence"])

        if best["confidence"] < self.config.confidence_low_threshold:
            best["answer"] = f"[Low Confidence] {best['answer']}"

        return best

    # -------------------------------------------------------------------
    # Resource cleanup
    # -------------------------------------------------------------------
    def close(self):
        for model in [self.qa_model, self.cross_encoder.model]:
            if hasattr(model, "cpu"):
                model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Resources released")

    def __del__(self):
        self.close()

# -------------------------------------------------------------------
# CLI with coloured output
# -------------------------------------------------------------------
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def _bar(width: int = 60) -> str:
    return DIM + "─" * width + RESET

def _header(title: str) -> str:
    return f"\n{BOLD}{CYAN}{title}{RESET}\n{_bar()}"

def _confidence_colour(score: float) -> str:
    if score >= 0.70:
        return f"{GREEN}{score:.3f}{RESET}"
    elif score >= 0.40:
        return f"{YELLOW}{score:.3f}{RESET}"
    return f"{RED}{score:.3f}{RESET}"

def _wrap(text: str, width: int = 90, indent: str = "  ") -> str:
    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent)

def cmd_summarize(args):
    analyzer = NeuraPDFAnalyzer()
    result   = analyzer.analyze_document(args.pdf)
    print(_header(f"SUMMARY  —  {os.path.basename(args.pdf)}"))
    print(f"  {DIM}Chunks processed : {len(result['document_metadata'])}{RESET}")
    print(f"  {DIM}Avg confidence   : {_confidence_colour(result['avg_confidence'])}\n")
    for i, item in enumerate(result['summary'], 1):
        conf_str = _confidence_colour(item.confidence)
        print(f"  {BOLD}[{i}]{RESET} Page {item.source_page}  confidence {conf_str}")
        print(_wrap(item.content))
        if args.evidence and item.supporting_evidence:
            print(f"\n  {DIM}Supporting evidence:{RESET}")
            for ev in item.supporting_evidence:
                print(_wrap(f"• {ev[:200]}…", indent="    "))
        print()
    if args.output:
        data = {
            "file": args.pdf,
            "avg_confidence": result['avg_confidence'],
            "summary": [
                {
                    "page": s.source_page,
                    "confidence": s.confidence,
                    "content": s.content,
                    "supporting_evidence": s.supporting_evidence
                }
                for s in result['summary']
            ]
        }
        with open(args.output, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  {GREEN}Saved to {args.output}{RESET}\n")

def cmd_ask(args):
    analyzer  = NeuraPDFAnalyzer()
    documents, retriever = analyzer.extract_text_with_metadata(args.pdf)
    questions = args.question if args.question else []
    if args.interactive or not questions:
        print(_header(f"INTERACTIVE QA  —  {os.path.basename(args.pdf)}"))
        print(f"  {DIM}Type your question and press Enter. Type 'exit' to quit.{RESET}\n")
        while True:
            try:
                q = input(f"  {CYAN}Question:{RESET} ").strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n  {DIM}Session ended.{RESET}\n")
                break
            if not q:
                continue
            if q.lower() in ("exit", "quit", "q"):
                print(f"\n  {DIM}Session ended.{RESET}\n")
                break
            _print_answer(analyzer, q, retriever, args)
    else:
        for q in questions:
            print(_header("QUESTION"))
            print(_wrap(q))
            _print_answer(analyzer, q, retriever, args)

def _print_answer(analyzer: NeuraPDFAnalyzer, question: str, retriever, args):
    result = analyzer.answer_question(question, retriever)
    print(f"\n  {BOLD}Answer{RESET}  (page {result.get('page_number', '?')}  "
          f"confidence {_confidence_colour(result['confidence'])})")
    print(_wrap(result['answer']))
    if args.explain:
        explanations = result.get('explanations', {}).get('explanations', {})
        if explanations:
            print(f"\n  {DIM}Concept explanations:{RESET}")
            for concept, explanation in explanations.items():
                print(f"\n    {BOLD}{concept}{RESET}")
                print(_wrap(explanation, indent="      "))
    if args.output:
        entry = {
            "question": question,
            "answer": result['answer'],
            "confidence": result['confidence'],
            "page_number": result.get('page_number'),
            "explanations": result.get('explanations', {})
        }
        mode = "a" if os.path.exists(args.output) else "w"
        existing = []
        if mode == "a":
            with open(args.output) as f:
                try:
                    existing = json.load(f)
                except json.JSONDecodeError:
                    existing = []
        existing.append(entry)
        with open(args.output, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"\n  {GREEN}Saved to {args.output}{RESET}")
    print()

def cmd_chat(args):
    analyzer  = NeuraPDFAnalyzer()
    documents, retriever = analyzer.extract_text_with_metadata(args.pdf)
    history: List[Dict] = []
    print(_header(f"DOCUMENT CHAT  —  {os.path.basename(args.pdf)}"))
    print(f"  {DIM}{len(documents)} chunks loaded. Type 'history' to review, 'exit' to quit.{RESET}\n")
    while True:
        try:
            q = input(f"  {CYAN}You:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {DIM}Session ended.{RESET}\n")
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit", "q"):
            print(f"\n  {DIM}Session ended.{RESET}\n")
            break
        if q.lower() == "history":
            if not history:
                print(f"  {DIM}No history yet.{RESET}\n")
            for turn in history:
                print(f"\n  {BOLD}Q:{RESET} {turn['question']}")
                print(f"  {BOLD}A:{RESET} {turn['answer']}")
            print()
            continue
        result = analyzer.answer_question(q, retriever)
        answer = result['answer']
        conf   = result['confidence']
        page   = result.get('page_number', '?')
        history.append({"question": q, "answer": answer, "confidence": conf})
        print(f"\n  {BOLD}{GREEN}NeuraPDF:{RESET}  "
              f"{DIM}page {page}  confidence {_confidence_colour(conf)}{RESET}")
        print(_wrap(answer))
        print()
    if args.save_history and history:
        with open(args.save_history, "w") as f:
            json.dump(history, f, indent=2)
        print(f"  {GREEN}History saved to {args.save_history}{RESET}\n")

# -------------------------------------------------------------------
# CLI parser
# -------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neura_pdf",
        description=f"{BOLD}NeuraPDF — Advanced PDF Analyzer by Neura Lumina{RESET}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(f"""
        {DIM}Examples:
          Summarize a PDF:
            python neura_pdf.py summarize paper.pdf

          Ask a single question:
            python neura_pdf.py ask paper.pdf -q "What is the main finding?"

          Ask multiple questions and save output:
            python neura_pdf.py ask paper.pdf \\
              -q "What methodology was used?" \\
              -q "What were the results?" \\
              --output answers.json

          Interactive Q&A session:
            python neura_pdf.py ask paper.pdf --interactive

          Chat mode with history:
            python neura_pdf.py chat paper.pdf --save-history session.json
        {RESET}
        """)
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_sum = sub.add_parser("summarize", aliases=["sum", "s"], help="Summarize a PDF document")
    p_sum.add_argument("pdf")
    p_sum.add_argument("--evidence", "-e", action="store_true", help="Show supporting evidence")
    p_sum.add_argument("--output",   "-o", metavar="FILE",      help="Save results to JSON")
    p_ask = sub.add_parser("ask", aliases=["a"], help="Ask one or more questions about a PDF")
    p_ask.add_argument("pdf")
    p_ask.add_argument("--question",    "-q", action="append", metavar="QUESTION")
    p_ask.add_argument("--interactive", "-i", action="store_true")
    p_ask.add_argument("--explain",     "-e", action="store_true")
    p_ask.add_argument("--output",      "-o", metavar="FILE")
    p_chat = sub.add_parser("chat", aliases=["c"], help="Interactive chat session with a PDF")
    p_chat.add_argument("pdf")
    p_chat.add_argument("--save-history", metavar="FILE")
    return parser

def main():
    parser = build_parser()
    args   = parser.parse_args()
    if not os.path.isfile(args.pdf):
        print(f"\n  {RED}Error: file not found — {args.pdf}{RESET}\n")
        sys.exit(1)
    if not args.pdf.lower().endswith(".pdf"):
        print(f"\n  {YELLOW}Warning: file does not have a .pdf extension.{RESET}\n")
    dispatch = {
        "summarize": cmd_summarize, "sum": cmd_summarize, "s": cmd_summarize,
        "ask":       cmd_ask,       "a":   cmd_ask,
        "chat":      cmd_chat,      "c":   cmd_chat,
    }
    dispatch[args.command](args)

if __name__ == "__main__":
    main()
