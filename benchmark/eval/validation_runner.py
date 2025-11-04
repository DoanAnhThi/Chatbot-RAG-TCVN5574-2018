"""Asynchronous evaluation runner for the validation dataset."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from src.config import settings
from src.services.indexing.indexer import build_or_load_vectorstore
from src.services.pipelines.generation.answer_generator import generate_answer
from src.services.pipelines.retrieval.formatter import format_documents
from src.services.pipelines.retrieval.retriever import get_retriever


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "benchmark" / "data" / "validation_data.json"
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "benchmark" / "outputs"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "benchmark" / "reports"


@dataclass
class Sample:
    idx: int
    payload: Dict[str, Any]

    @property
    def id(self) -> Any:
        return self.payload.get("id", self.idx)

    @property
    def question(self) -> str:
        return self.payload.get("question_vi") or self.payload.get("question")

    @property
    def reference_answer(self) -> str:
        return self.payload.get("answer_key_vi") or self.payload.get("answer")


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the benchmark validation set through the retrieval-augmented pipeline.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to the JSON validation dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUTS_DIR,
        help="Directory for per-question outputs (JSONL).",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help="Directory for aggregate run metadata.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Maximum number of concurrent questions to evaluate.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on how many records from the dataset to process.",
    )
    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=settings.retrieval_top_k,
        help="Override the number of chunks retrieved per question.",
    )
    return parser.parse_args(argv)


def load_dataset(path: Path, limit: int | None = None) -> List[Sample]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with path.open("r", encoding="utf-8") as fp:
        raw_items = json.load(fp)

    samples: List[Sample] = []
    for idx, payload in enumerate(raw_items):
        samples.append(Sample(idx=idx, payload=payload))
        if limit is not None and len(samples) >= limit:
            break

    return samples


def _truncate(text: str, limit: int = 600) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _evaluate_sample_sync(
    sample: Sample,
    retriever,
) -> Dict[str, Any]:
    question = sample.question
    if not question:
        raise ValueError(f"Sample {sample.id} is missing a question field")

    docs = retriever.get_relevant_documents(question)
    result = generate_answer(question, docs)

    retrieved_context = [
        {
            "id": doc.metadata.get("id"),
            "source": doc.metadata.get("source"),
            "title": doc.metadata.get("title"),
            "chunk_id": doc.metadata.get("chunk_id"),
            "score": doc.metadata.get("score"),
            "content": _truncate(doc.page_content or ""),
        }
        for doc in docs
    ]

    return {
        "sample_id": sample.id,
        "question": question,
        "reference_answer": sample.reference_answer,
        "predicted_answer": result["answer"],
        "confidence": result.get("confidence"),
        "needs_clarification": result.get("needs_clarification", False),
        "formatted_context": format_documents(docs),
        "retrieved_chunks": retrieved_context,
        "metadata": {
            "eval_note": sample.payload.get("eval_note"),
            "section_hint": sample.payload.get("section_hint"),
        },
    }


async def evaluate_dataset(
    samples: Iterable[Sample],
    retriever,
    concurrency: int,
) -> List[Dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(1, concurrency))
    samples_list = list(samples)
    results: List[Tuple[int, Dict[str, Any]]] = []

    async def _run(sample: Sample) -> Tuple[int, Dict[str, Any]]:
        async with semaphore:
            try:
                record = await asyncio.to_thread(_evaluate_sample_sync, sample, retriever)
                record["status"] = "ok"
                return sample.idx, record
            except Exception as exc:  # pragma: no cover - defensive logging
                return sample.idx, {
                    "sample_id": sample.id,
                    "question": sample.question,
                    "status": "error",
                    "error": str(exc),
                }

    tasks = [asyncio.create_task(_run(sample)) for sample in samples_list]

    for task in asyncio.as_completed(tasks):
        results.append(await task)

    ordered: List[Dict[str, Any]] = [None] * len(samples_list)  # type: ignore
    for idx, record in results:
        ordered[idx] = record

    return ordered


def _ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_summary(
    path: Path,
    history_path: Path,
    run_meta: Dict[str, Any],
) -> None:
    with path.open("w", encoding="utf-8") as fp:
        json.dump(run_meta, fp, ensure_ascii=False, indent=2)

    with history_path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(run_meta, ensure_ascii=False) + "\n")


async def async_main(args: argparse.Namespace) -> Dict[str, Any]:
    samples = load_dataset(args.dataset, args.limit)

    vectorstore = build_or_load_vectorstore()
    retriever = get_retriever(vectorstore, k=args.retrieval_top_k)

    results = await evaluate_dataset(samples, retriever, args.concurrency)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    _ensure_dirs(args.output_dir, args.reports_dir)

    results_path = args.output_dir / f"validation_results_{run_id}.jsonl"
    _write_jsonl(results_path, results)

    summary = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset.resolve()),
        "total_samples": len(samples),
        "completed": sum(1 for r in results if r.get("status") == "ok"),
        "errors": [r for r in results if r.get("status") == "error"],
        "average_confidence": _average_confidence(results),
        "clarification_requests": sum(1 for r in results if r.get("needs_clarification")),
        "retrieval_top_k": args.retrieval_top_k,
        "concurrency": args.concurrency,
        "vectorstore_dir": settings.vectorstore_dir,
        "processed_data_dir": settings.processed_data_dir,
        "results_file": str(results_path.resolve()),
    }

    summary_path = args.reports_dir / f"validation_summary_{run_id}.json"
    history_path = args.reports_dir / "history.jsonl"
    _write_summary(summary_path, history_path, summary)

    latest_symlink = args.output_dir / "latest.jsonl"
    try:
        if latest_symlink.exists() or latest_symlink.is_symlink():
            latest_symlink.unlink()
        latest_symlink.symlink_to(results_path.name)
    except OSError:
        pass  # symlink not supported on all platforms

    return {
        "results_path": results_path,
        "summary_path": summary_path,
        "run_id": run_id,
    }


def _average_confidence(results: Iterable[Dict[str, Any]]) -> float | None:
    confidences = [r.get("confidence") for r in results if isinstance(r.get("confidence"), (int, float))]
    if not confidences:
        return None
    return sum(confidences) / len(confidences)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    summary = asyncio.run(async_main(args))
    print("Evaluation run completed:")
    print("  run_id:", summary["run_id"])
    print("  results:", summary["results_path"])
    print("  summary:", summary["summary_path"])


if __name__ == "__main__":  # pragma: no cover
    main()

