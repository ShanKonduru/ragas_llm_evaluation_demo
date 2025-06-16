import os
from dotenv import load_dotenv
import asyncio
from ragas.metrics import AnswerRelevancy
from ragas import evaluate
from datasets import Dataset
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
import inspect

load_dotenv()

open_ai_api_key = os.environ.get("OPENAI_API_KEY")
if not open_ai_api_key:
    print("OPENAI_API_KEY not found. Please set it in your .env file or environment variables.")
    exit(1)

evaluator_llm = LangchainLLMWrapper(ChatOpenAI(
    model="gpt-4o", openai_api_key=open_ai_api_key))


async def maybe_await(obj):
    return await obj if inspect.isawaitable(obj) else obj


async def evaluate_faithfulness():
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

    metrics = AnswerRelevancy()

    print("\n--- Evaluating Answer Relevancy (Relevant Scenario) ---")
    result_relevant = evaluate(dataset_relevant, metrics=[
                               metrics], llm=evaluator_llm)
    print(
        f"Type of object returned by ragas.evaluate (result_relevant): {type(result_relevant)}")
    print(f"Is it awaitable? {inspect.isawaitable(result_relevant)}")
    result_relevant = await maybe_await(result_relevant)
    print(result_relevant)

    print("\n--- Evaluating Answer Relevancy (Irrelevant Scenario) ---")
    result_irrelevant = evaluate(dataset_irrelevant, metrics=[
                                 metrics], llm=evaluator_llm)
    print(
        f"Type of object returned by ragas.evaluate (result_irrelevant): {type(result_irrelevant)}")
    print(f"Is it awaitable? {inspect.isawaitable(result_irrelevant)}")
    result_irrelevant = await maybe_await(result_irrelevant)
    print(result_irrelevant)


if __name__ == "__main__":
    asyncio.run(evaluate_faithfulness())
