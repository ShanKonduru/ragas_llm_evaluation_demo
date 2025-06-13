import os
from dotenv import load_dotenv

from datasets import Dataset
import asyncio

from Ragas_Faithfulness_Evaluator import RagasFaithfulnessEvaluator

if __name__ == "__main__":

    load_dotenv()

    open_ai_api_key = os.environ.get("OPENAI_API_KEY")
    if not open_ai_api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Please set it in your .env file or environment variables."
        )

    async def main():
        # Define the datasets outside the class instance
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
        evaluator = RagasFaithfulnessEvaluator(
            open_ai_api_key, model_name="gpt-4o")
        # Pass the datasets to the run_evaluation method
        await evaluator.run_evaluation(dataset_faithful)

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

        evaluator = RagasFaithfulnessEvaluator(
            open_ai_api_key, model_name="gpt-4o")
        # Pass the datasets to the run_evaluation method
        await evaluator.run_evaluation(dataset_unfaithful)

    asyncio.run(main())
