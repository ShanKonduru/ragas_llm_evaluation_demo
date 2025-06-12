# Demo #6: Ragas for Answer Relevancy
import os
from dotenv import load_dotenv
import asyncio
from ragas.metrics import AnswerRelevancy
from ragas import evaluate
from datasets import Dataset

load_dotenv()


async def evaluate_answer_relevancy():
    # Scenario 1: Relevant answer
    data_relevant = {
        "question": ["What is the primary function of a CPU?"],
        "answer": [
            "The primary function of a CPU (Central Processing Unit) is to execute instructions and perform calculations involved in computer programs."
        ],
        "contexts": [
            [
                "The CPU is the electronic circuitry within a computer that carries out the instructions of a computer program."
            ]
        ],  # Context is optional for AnswerRelevancy but good practice
    }
    dataset_relevant = Dataset.from_dict(data_relevant)

    # Scenario 2: Irrelevant answer (e.g., talks about GPUs instead)
    data_irrelevant = {
        "question": ["What is the primary function of a CPU?"],
        "answer": [
            "GPUs are designed for parallel processing, making them ideal for rendering graphics and machine learning tasks."
        ],
        "contexts": [
            [
                "The CPU is the electronic circuitry within a computer that carries out the instructions of a computer program."
            ]
        ],
    }
    dataset_irrelevant = Dataset.from_dict(data_irrelevant)

    answer_relevancy_metric = AnswerRelevancy()

    print("\n--- Evaluating Answer Relevancy (Relevant Scenario) ---")
    result_relevant = await evaluate(
        dataset_relevant, metrics=[answer_relevancy_metric]
    )
    print(result_relevant)

    print("\n--- Evaluating Answer Relevancy (Irrelevant Scenario) ---")
    result_irrelevant = await evaluate(
        dataset_irrelevant, metrics=[answer_relevancy_metric]
    )
    print(result_irrelevant)


if __name__ == "__main__":
    asyncio.run(evaluate_answer_relevancy())
