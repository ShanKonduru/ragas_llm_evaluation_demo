# Demo #5: Ragas for Faithfulness
import os
from dotenv import load_dotenv
import asyncio
from ragas.metrics import Faithfulness
from ragas.dataset_schema import SingleTurnSample
from ragas import evaluate
from datasets import Dataset  # To create a dataset for evaluation

load_dotenv()  # Load environment variables for API keys

# Ensure your OpenAI API key is set in .env or as an environment variable
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"


async def evaluate_faithfulness():
    # Example data simulating a RAG output
    # Scenario 1: Answer is faithful to the context
    data_faithful = {
        "question": ["What is the capital of France?"],
        "answer": ["Paris is the capital of France."],
        "contexts": [
            [
                "Paris is the capital and most populous city of France. It is located on the Seine River."
            ]
        ],
    }
    dataset_faithful = Dataset.from_dict(data_faithful)

    # Scenario 2: Answer is NOT faithful (hallucination)
    data_unfaithful = {
        "question": ["What is the capital of France?"],
        "answer": [
            "The capital of France is Berlin, which is also a major city in Germany."
        ],
        "contexts": [
            [
                "Paris is the capital and most populous city of France. It is located on the Seine River."
            ]
        ],
    }
    dataset_unfaithful = Dataset.from_dict(data_unfaithful)

    faithfulness_metric = Faithfulness()

    print("\n--- Evaluating Faithfulness (Faithful Scenario) ---")
    result_faithful = await evaluate(dataset_faithful, metrics=[faithfulness_metric])
    print(result_faithful)

    print("\n--- Evaluating Faithfulness (Unfaithful Scenario - Hallucination) ---")
    result_unfaithful = await evaluate(
        dataset_unfaithful, metrics=[faithfulness_metric]
    )
    print(result_unfaithful)


if __name__ == "__main__":
    asyncio.run(evaluate_faithfulness())
