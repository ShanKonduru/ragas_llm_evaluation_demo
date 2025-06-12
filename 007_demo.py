# Demo #7: Ragas for Context Relevancy & Context Recall
import os
from dotenv import load_dotenv
import asyncio
from ragas.metrics import ContextRelevance, ContextRecall
from ragas import evaluate
from datasets import Dataset

load_dotenv()


async def evaluate_context_metrics():
    # Example data for RAG evaluation
    # Ensure ground_truth is provided for Context Recall
    data_rag = {
        "question": ["What is the capital of Japan?", 
                     "Who painted the Mona Lisa?"],
        "answer": [
            "The capital of Japan is Tokyo.",
            "Leonardo da Vinci painted the Mona Lisa.",
        ],
        "contexts": [
            [
                "Tokyo is the capital city of Japan and its largest metropolis."
            ],  # Relevant context
            [
                "Leonardo da Vinci was an Italian polymath who created the Mona Lisa, among other works. The painting is famous for its enigmatic smile."
            ],  # Relevant context
        ],
        "ground_truths": [
            [
                "Tokyo is the largest city and capital of Japan."
            ],  # Full ground truth for recall
            ["The Mona Lisa was painted by Leonardo da Vinci between 1503 and 1519."],
        ],
    }
    dataset_rag = Dataset.from_dict(data_rag)

    context_relevancy_metric = ContextRelevance()
    context_recall_metric = ContextRecall()

    print("\n--- Evaluating Context Relevancy and Context Recall ---")
    results = await evaluate(
        dataset_rag, metrics=[context_relevancy_metric, context_recall_metric]
    )
    print(results)


if __name__ == "__main__":
    asyncio.run(evaluate_context_metrics())
