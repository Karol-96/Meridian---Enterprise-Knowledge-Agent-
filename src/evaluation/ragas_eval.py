"""
Meridian — RAGAS Evaluation Pipeline
=====================================
Benchmarks the RAG pipeline on 4 core metrics:

1. Faithfulness     — Is the answer grounded in the retrieved context? (no hallucination)
2. Response Relevancy — Does the answer actually address the question?
3. Context Precision — Are the retrieved chunks relevant and ranked well?
4. Context Recall    — Did retrieval find all the info needed to answer?

Run with:  python3 -m src.evaluation.ragas_eval
"""

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import ResponseRelevancy
from ragas.metrics._context_precision import LLMContextPrecisionWithReference
from ragas.metrics._context_recall import LLMContextRecall
from openai import OpenAI as OpenAIClient
from ragas.llms import llm_factory


load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")


# ── EVAL DATASET ──────────────────────────────────────────────────────────────
# Each sample needs:
#   - user_input:          the question
#   - reference:           the ground-truth answer (human-written)
#   - retrieved_contexts:  list of chunk texts returned by retriever
#   - response:            the LLM-generated answer
#
# We define the questions + ground truth here, then run our actual RAG
# pipeline to fill in retrieved_contexts and response automatically.

EVAL_QUESTIONS = [
    {
        "question": "How does Kafka handle replication?",
        "ground_truth": (
            "Kafka replicates data by maintaining multiple copies of each partition "
            "across different brokers. One broker serves as the leader for a partition "
            "and handles all reads and writes, while follower brokers replicate the data. "
            "Kafka maintains an in-sync replica set (ISR) to track which followers are "
            "up to date. If the leader fails, one of the in-sync followers is elected "
            "as the new leader."
        ),
    },
    {
        "question": "What is the checkpoint mechanism in Apache Flink?",
        "ground_truth": (
            "Flink's checkpoint mechanism periodically snapshots the state of a "
            "streaming application to durable storage. It uses the Chandy-Lamport "
            "algorithm to create consistent snapshots without stopping the data flow. "
            "Checkpoint barriers are injected into the data streams and when all barriers "
            "arrive at a task, its state is snapshotted. On failure, Flink restores "
            "from the latest completed checkpoint to resume processing with exactly-once "
            "semantics."
        ),
    },
    {
        "question": "How do you schedule a DAG in Apache Airflow?",
        "ground_truth": (
            "In Airflow, DAGs are scheduled using the schedule_interval parameter "
            "which accepts cron expressions or timedelta objects. The scheduler "
            "continuously monitors DAGs and triggers new runs based on the defined "
            "schedule. Each DAG run represents a logical execution for a specific "
            "data interval. DAGs can also be triggered manually or via the API."
        ),
    },
    {
        "question": "How does Kafka guarantee message ordering?",
        "ground_truth": (
            "Kafka guarantees message ordering within a single partition. Messages "
            "sent to the same partition are appended in order and consumers read them "
            "in the same order they were written. To ensure ordering for related "
            "messages, producers use a partition key so all related messages go to "
            "the same partition. Kafka does not guarantee ordering across different "
            "partitions."
        ),
    },
    {
        "question": "What is a Flink savepoint and how does it differ from a checkpoint?",
        "ground_truth": (
            "A savepoint is a manually triggered, portable snapshot of a Flink job's "
            "complete state. Unlike checkpoints which are automatic and managed by "
            "Flink for fault tolerance, savepoints are user-initiated and designed "
            "for operational purposes like upgrading a job, changing parallelism, "
            "or migrating between clusters. Savepoints are not automatically cleaned "
            "up and must be explicitly managed by the user."
        ),
    },
]


async def build_eval_dataset() -> EvaluationDataset:
    """
    Runs each eval question through the live RAG pipeline,
    then packages everything into a RAGAS EvaluationDataset.
    """
    from src.retrieval.retriever import retrieve
    from src.retrieval.generator import generate

    samples = []

    for item in EVAL_QUESTIONS:
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"\n  Running RAG for: {question}")

        # Step 1: Retrieve chunks
        chunks = await retrieve(question)
        retrieved_texts = [c["text"] for c in chunks]

        # Step 2: Generate answer
        result = await generate(question, chunks)
        answer = result["answer"]

        print(f"  Retrieved {len(chunks)} chunks, generated {len(answer)} char answer")

        # Step 3: Package as RAGAS sample
        samples.append(
            SingleTurnSample(
                user_input=question,
                reference=ground_truth,
                retrieved_contexts=retrieved_texts,
                response=answer,
            )
        )

    return EvaluationDataset(samples=samples)


async def run_evaluation():
    """
    Builds the eval dataset from live RAG, then scores it with RAGAS.
    """
    print("\nMeridian — RAGAS Evaluation")
    print("=" * 60)

    # ── Build dataset from live RAG pipeline ──
    print("\n[Step 1] Running RAG pipeline on eval questions...")
    dataset = await build_eval_dataset()

    # ── Configure the judge LLM (RAGAS uses this to score) ──
    print("\n[Step 2] Scoring with RAGAS metrics...")
    judge_llm = llm_factory(
        "gpt-4o-mini",
        client=OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))
    )

    # ── Define metrics ──
    metrics = [
        Faithfulness(llm=judge_llm),
        ResponseRelevancy(llm=judge_llm),
        LLMContextPrecisionWithReference(llm=judge_llm),
        LLMContextRecall(llm=judge_llm),
    ]

    # ── Run evaluation ──
    results = evaluate(dataset=dataset, metrics=metrics)

    # ── Print results ──
    df = results.to_pandas()

    # Metric columns = everything except the text fields
    text_cols = {"user_input", "reference", "retrieved_contexts", "response"}
    metric_cols = [c for c in df.columns if c not in text_cols]

    print("\n" + "=" * 60)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 60)

    # Overall averages
    print("\nOverall Scores (averaged across all questions):")
    print("-" * 40)
    for col in metric_cols:
        avg = df[col].mean()
        print(f"  {col:30s} : {avg:.4f}")

    # Per-question breakdown
    print("\n\nPer-Question Breakdown:")
    print("-" * 60)

    for i, row in df.iterrows():
        print(f"\n  Q{i+1}: {row['user_input'][:60]}...")
        for col in metric_cols:
            val = row[col]
            if isinstance(val, float):
                print(f"      {col:30s} : {val:.4f}")
            else:
                print(f"      {col:30s} : {val}")

    # Save to CSV
    output_path = Path(__file__).parent / "ragas_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\n\nFull results saved to: {output_path}")

    return results


if __name__ == "__main__":
    asyncio.run(run_evaluation())
