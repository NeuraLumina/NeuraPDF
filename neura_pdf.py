import os
import re
import json
import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering,
    pipeline,
    LogitsProcessor,
    LogitsProcessorList,
    PreTrainedModel,
    PreTrainedTokenizer
)
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import PyPDF2
from sklearn.cluster import KMeans
import spacy
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

print('====================== VERSION 7 ======================')


@dataclass
class DeviceManager:
    _device: torch.device = field(init=False)

    def __post_init__(self):
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] Using {'GPU' if self._device.type == 'cuda' else 'CPU'} for inference.")

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def pipeline_device(self) -> int:
        return 0 if self._device.type == 'cuda' else -1

    def move(self, model: torch.nn.Module) -> torch.nn.Module:
        return model.to(self._device)

    def move_inputs(self, inputs: dict) -> dict:
        return {k: v.to(self._device) for k, v in inputs.items()}


class TemperatureScaler(LogitsProcessor):
    def __init__(self, temperature: float = 0.85):
        if temperature <= 0:
            raise ValueError("Temperature must be positive.")
        self.temperature = temperature

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        return scores / self.temperature


class DocumentResult(BaseModel):
    content: str
    confidence: float
    source_page: int
    supporting_evidence: List[str]


class OptimalModelSelector:
    def __init__(self, device_manager: DeviceManager):
        self.device_manager = device_manager
        self.qa_models = {
            "deberta-v3": ("deepset/deberta-v3-large-squad2", 0.87),
            "minilm": ("deepset/minilm-uncased-squad2", 0.84),
            "roberta": ("deepset/roberta-base-squad2", 0.82)
        }
        self.summarization_models = {
            "bart": ("facebook/bart-large-cnn", 0.85),
            "pegasus": ("google/pegasus-xsum", 0.83)
        }
        self.current_models: Dict[str, Tuple[PreTrainedModel, PreTrainedTokenizer]] = {}

    def get_best_model(self, task_type: str) -> Tuple[PreTrainedModel, PreTrainedTokenizer, float]:
        model_map = self.qa_models if "qa" in task_type else self.summarization_models
        best_model_name, (model_path, best_score) = max(model_map.items(), key=lambda x: x[1][1])

        if best_model_name not in self.current_models:
            logging.info(f"Loading {best_model_name} for {task_type}")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model_cls = AutoModelForQuestionAnswering if "qa" in task_type else AutoModelForSeq2SeqLM
            model = model_cls.from_pretrained(model_path)
            model = self.device_manager.move(model.eval().half())
            self.current_models[best_model_name] = (model, tokenizer)

        return *self.current_models[best_model_name], best_score


class SlidingWindowChunker:
    def __init__(self, tokenizer: PreTrainedTokenizer, window_size: int = 400, stride: int = 100):
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
                'content': chunk_text,
                'metadata': {
                    'page': page_number,
                    'token_start': start,
                    'token_end': end,
                    'chunk_index': len(chunks)
                }
            })
            if end == len(tokens):
                break
            start += self.stride
        return chunks


class PDFAugmentedRetriever:
    def __init__(self, documents: List[Dict]):
        self.documents = documents
        texts = [doc['content'] for doc in documents]
        self.bm25 = BM25Okapi([t.split() for t in texts])
        self.encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.tfidf = TfidfVectorizer(stop_words='english').fit(texts)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        texts = [doc['content'] for doc in self.documents]
        bm25_scores = self.bm25.get_scores(query.split())
        pairs = [(query, t) for t in texts]
        semantic_scores = np.array(self.encoder.predict(pairs))
        combined_scores = 0.4 * bm25_scores + 0.6 * semantic_scores
        top_indices = np.argsort(combined_scores)[-top_k:][::-1]
        return [
            {**self.documents[i], 'retrieval_score': float(combined_scores[i])}
            for i in top_indices
        ]


class AnswerValidator:
    def __init__(self, cross_encoder: CrossEncoder):
        self.cross_encoder = cross_encoder

    def validate(self, answer: str, contexts: List[str]) -> float:
        if not answer.strip() or not contexts:
            return 0.0
        pairs = [(answer, ctx) for ctx in contexts]
        scores = self.cross_encoder.predict(pairs)
        return float(np.max(scores))

    def is_no_answer(self, start_logits: torch.Tensor, end_logits: torch.Tensor, threshold: float = 0.0) -> bool:
        cls_start = start_logits[0, 0].item()
        cls_end = end_logits[0, 0].item()
        best_start = start_logits[0].max().item()
        best_end = end_logits[0].max().item()
        return (cls_start + cls_end) > (best_start + best_end) + threshold


class DetailedExplainer:
    def __init__(self, device_manager: DeviceManager, explanation_model: str = "google/flan-t5-large"):
        self.explainer = pipeline(
            "text2text-generation",
            model=explanation_model,
            tokenizer=explanation_model,
            device=device_manager.pipeline_device
        )
        self.nlp = spacy.load("en_core_web_sm")

    def extract_concepts(self, text: str) -> List[str]:
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
            explanation = result[0]["generated_text"].strip()
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
            explanation = result["generated_text"].strip()
            if len(explanation.split()) < 20:
                explanation = self.explain_concept(concept, context)
            explanations[concept] = explanation

        return {"concepts": concepts, "explanations": explanations}


class AdvancedPDFAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger("PDFAnalyzer")
        self._verify_dependencies()

        self.device_manager = DeviceManager()
        self.model_selector = OptimalModelSelector(self.device_manager)

        self.qa_model, self.qa_tokenizer, _ = self.model_selector.get_best_model("qa")
        self.chunker = SlidingWindowChunker(self.qa_tokenizer)

        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.answer_validator = AnswerValidator(self.cross_encoder)

        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=self.device_manager.pipeline_device,
            framework="pt"
        )

        self.generation_logits_processor = LogitsProcessorList([
            TemperatureScaler(temperature=0.85)
        ])

        self.detailed_explainer = DetailedExplainer(self.device_manager)

    def _verify_dependencies(self):
        try:
            PyPDF2.PdfReader
        except ImportError:
            raise ImportError("PyPDF2 required: pip install pypdf2")

    def extract_text_with_metadata(self, file_path: str) -> List[Dict]:
        self.logger.info(f"Processing {file_path}")
        documents = []

        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(tqdm(reader.pages)):
                try:
                    text = page.extract_text()
                    if not text or not text.strip():
                        continue
                    cleaned = self._clean_text(text)
                    page_number = i + 1
                    chunks = self.chunker.chunk(cleaned, page_number)
                    for chunk in chunks:
                        chunk['metadata'].update({
                            'source': os.path.basename(file_path),
                            'char_count': len(chunk['content']),
                            'word_count': len(chunk['content'].split()),
                        })
                    documents.extend(chunks)
                except Exception as e:
                    self.logger.warning(f"Page {i + 1} error: {str(e)}")

        if not documents:
            raise ValueError("No extractable content found in PDF")

        return documents

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)
        return text.strip()

    def analyze_document(self, file_path: str) -> Dict:
        documents = self.extract_text_with_metadata(file_path)
        retriever = PDFAugmentedRetriever(documents)
        summary = self._generate_summary_with_confidence(documents, retriever)
        return {
            'document_metadata': [doc['metadata'] for doc in documents],
            'summary': summary,
            'avg_confidence': float(np.mean([s.confidence for s in summary])) if summary else 0.0
        }

    def _generate_summary_with_confidence(
        self,
        documents: List[Dict],
        retriever: PDFAugmentedRetriever
    ) -> List[DocumentResult]:
        sentences = [
            (doc['metadata']['page'], s.strip())
            for doc in documents
            for s in doc['content'].split('. ')
            if len(s.split()) > 6
        ]
        if not sentences:
            return []

        pages, texts = zip(*sentences)

        vectorizer = TfidfVectorizer(max_features=500)
        X = vectorizer.fit_transform(texts)

        summary_results = []
        for cluster in self._cluster_text(X, n_clusters=min(5, len(texts))):
            cluster_items = [(pages[i], texts[i]) for i in cluster]
            cluster_texts = [t for _, t in cluster_items]
            scores = self._cross_validate_sentences(cluster_texts)
            best_idx = int(np.argmax(scores))
            best_page, best_sent = cluster_items[best_idx]
            best_score = scores[best_idx]

            cross_score = self.cross_encoder.predict([(best_sent, t) for t in cluster_texts if t != best_sent])
            calibrated_confidence = min(0.95, float(best_score / (X.sum(axis=1).max() + 1e-9)) * 0.7 + 0.3 * float(np.mean(cross_score)) if len(cross_score) > 0 else float(best_score))

            summary_results.append(DocumentResult(
                content=best_sent,
                confidence=calibrated_confidence,
                source_page=best_page,
                supporting_evidence=self._find_supporting_evidence(best_sent, retriever)
            ))

        return summary_results

    def answer_question(self, question: str, documents: List[Dict]) -> Dict:
        retriever = PDFAugmentedRetriever(documents)
        relevant_docs = retriever.retrieve(question, top_k=5)

        answers = []
        for doc in relevant_docs:
            context = doc['content']
            similarity_score = doc['retrieval_score']

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

                if self.answer_validator.is_no_answer(start_logits, end_logits):
                    continue

                start_probs = torch.nn.functional.softmax(start_logits, dim=-1)
                end_probs = torch.nn.functional.softmax(end_logits, dim=-1)

                seq_len = start_probs.size(1)
                score_matrix = torch.triu(
                    start_probs[0].unsqueeze(1) * end_probs[0].unsqueeze(0)
                )
                best_span = score_matrix.argmax()
                start_idx = (best_span // seq_len).item()
                end_idx = (best_span % seq_len).item()

                span_confidence = float(score_matrix[start_idx, end_idx])
                confidence = span_confidence * float(similarity_score)

                answer_tokens = inputs["input_ids"][0][start_idx:end_idx + 1]
                answer = self.qa_tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()

                if not answer:
                    continue

                explanations_result = self.detailed_explainer.explain_text(answer, context)

                answers.append({
                    "answer": answer,
                    "confidence": confidence,
                    "span_confidence": span_confidence,
                    "context": context,
                    "page_number": doc['metadata']['page'],
                    "explanations": explanations_result
                })

        if not answers:
            return {"answer": "No confident answer found", "confidence": 0.0, "explanations": {}}

        all_contexts = [a['context'] for a in answers]
        for a in answers:
            validation_score = self.answer_validator.validate(a['answer'], all_contexts)
            a['confidence'] = 0.7 * a['confidence'] + 0.3 * validation_score

        best_answer = max(answers, key=lambda x: x['confidence'])

        if best_answer['confidence'] < 0.35:
            best_answer['answer'] = f"[Low Confidence] {best_answer['answer']}"

        return best_answer

    def _cluster_text(self, X, n_clusters: int = 5) -> List[List[int]]:
        if X.shape[0] < n_clusters:
            return [[i] for i in range(X.shape[0])]
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        clusters: List[List[int]] = [[] for _ in range(n_clusters)]
        for idx, label in enumerate(labels):
            clusters[label].append(idx)
        return [c for c in clusters if c]

    def _cross_validate_sentences(self, sentences: List[str]) -> List[float]:
        if not sentences:
            return []
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        scores = np.asarray(tfidf_matrix.sum(axis=1)).flatten()
        norms = np.asarray((tfidf_matrix != 0).sum(axis=1)).flatten()
        normalised = np.where(norms > 0, scores / norms, 0.0)
        return normalised.tolist()

    def _find_supporting_evidence(self, sentence: str, retriever: PDFAugmentedRetriever, top_k: int = 2) -> List[str]:
        results = retriever.retrieve(sentence, top_k=top_k)
        return [r['content'] for r in results]


import argparse
import sys
import textwrap


BOLD  = "\033[1m"
CYAN  = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED   = "\033[91m"
DIM   = "\033[2m"
RESET = "\033[0m"

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
    analyzer = AdvancedPDFAnalyzer()
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
    analyzer  = AdvancedPDFAnalyzer()
    documents = analyzer.extract_text_with_metadata(args.pdf)

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
            _print_answer(analyzer, q, documents, args)
    else:
        for q in questions:
            print(_header(f"QUESTION"))
            print(_wrap(q))
            _print_answer(analyzer, q, documents, args)


def _print_answer(analyzer: AdvancedPDFAnalyzer, question: str, documents: List[Dict], args):
    result = analyzer.answer_question(question, documents)

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
    analyzer  = AdvancedPDFAnalyzer()
    documents = analyzer.extract_text_with_metadata(args.pdf)
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

        result = analyzer.answer_question(q, documents)
        answer = result['answer']
        conf   = result['confidence']
        page   = result.get('page_number', '?')

        history.append({"question": q, "answer": answer, "confidence": conf})

        print(f"\n  {BOLD}{GREEN}Ogent:{RESET}  "
              f"{DIM}page {page}  confidence {_confidence_colour(conf)}{RESET}")
        print(_wrap(answer))
        print()

    if args.save_history and history:
        with open(args.save_history, "w") as f:
            json.dump(history, f, indent=2)
        print(f"  {GREEN}History saved to {args.save_history}{RESET}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf_analyzer",
        description=f"{BOLD}Advanced PDF Analyzer — Neura Lumina / Ogent{RESET}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(f"""
        {DIM}Examples:
          Summarize a PDF:
            python pdf_analyzer.py summarize paper.pdf

          Ask a single question:
            python pdf_analyzer.py ask paper.pdf -q "What is the main finding?"

          Ask multiple questions and save output:
            python pdf_analyzer.py ask paper.pdf \\
              -q "What methodology was used?" \\
              -q "What were the results?" \\
              --output answers.json

          Interactive Q&A session:
            python pdf_analyzer.py ask paper.pdf --interactive

          Chat mode with history:
            python pdf_analyzer.py chat paper.pdf --save-history session.json
        {RESET}
        """)
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ── summarize ────────────────────────────────────────────────
    p_sum = sub.add_parser("summarize", aliases=["sum", "s"],
                            help="Summarize a PDF document")
    p_sum.add_argument("pdf",             help="Path to the PDF file")
    p_sum.add_argument("--evidence", "-e", action="store_true",
                        help="Show supporting evidence for each summary point")
    p_sum.add_argument("--output", "-o",  metavar="FILE",
                        help="Save results to a JSON file")

    # ── ask ──────────────────────────────────────────────────────
    p_ask = sub.add_parser("ask", aliases=["a"],
                            help="Ask one or more questions about a PDF")
    p_ask.add_argument("pdf",              help="Path to the PDF file")
    p_ask.add_argument("--question", "-q", action="append", metavar="QUESTION",
                        help="Question to ask (repeatable)")
    p_ask.add_argument("--interactive", "-i", action="store_true",
                        help="Enter interactive question loop")
    p_ask.add_argument("--explain", "-e",  action="store_true",
                        help="Show concept explanations for each answer")
    p_ask.add_argument("--output", "-o",   metavar="FILE",
                        help="Append results to a JSON file")

    # ── chat ─────────────────────────────────────────────────────
    p_chat = sub.add_parser("chat", aliases=["c"],
                             help="Interactive chat session with a PDF")
    p_chat.add_argument("pdf",                  help="Path to the PDF file")
    p_chat.add_argument("--save-history",       metavar="FILE",
                         help="Save conversation history to a JSON file on exit")

    return parser


def main():
    parser  = build_parser()
    args    = parser.parse_args()

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
