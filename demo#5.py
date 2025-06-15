import os
from dotenv import load_dotenv
import asyncio
from ragas.metrics import Faithfulness
from ragas import evaluate
from datasets import Dataset
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
import inspect

load_dotenv()

open_ai_api_key = os.environ.get("OPENAI_API_KEY")
if not open_ai_api_key:
    print("OPENAI_API_KEY='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'")
    print("OPENAI_API_KEY not found. Please set it in your .env file or environment variables.")
    exit(1)

evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", openai_api_key=open_ai_api_key))


async def maybe_await(obj):
    return await obj if inspect.isawaitable(obj) else obj


async def evaluate_faithfulness():
    data_faithful = {
        "question": ["What is the capital of France?"],
        "answer": ["Paris is the capital of France."],
        "contexts": [
            ["Paris is the capital and most populous city of France. It is located on the Seine River."]
        ],
    }
    dataset_faithful = Dataset.from_dict(data_faithful)

    data_unfaithful = {
        "question": ["What is the capital of France?"],
        "answer": ["The capital of France is Berlin, which is also a major city in Germany."],
        "contexts": [
            ["Paris is the capital and most populous city of France. It is located on the Seine River."]
        ],
    }
    dataset_unfaithful = Dataset.from_dict(data_unfaithful)

    faithfulness_metric = Faithfulness()

    print("\n--- Evaluating Faithfulness (Faithful Scenario) ---")
    eval_return_obj_faithful = evaluate(dataset_faithful, metrics=[faithfulness_metric], llm=evaluator_llm)
    print(f"Type of object returned by ragas.evaluate (faithful): {type(eval_return_obj_faithful)}")
    print(f"Is it awaitable? {inspect.isawaitable(eval_return_obj_faithful)}")
    result_faithful = await maybe_await(eval_return_obj_faithful)
    print(result_faithful)

    print("\n--- Evaluating Faithfulness (Unfaithful Scenario - Hallucination) ---")
    eval_return_obj_unfaithful = evaluate(dataset_unfaithful, metrics=[faithfulness_metric], llm=evaluator_llm)
    print(f"Type of object returned by ragas.evaluate (unfaithful): {type(eval_return_obj_unfaithful)}")
    print(f"Is it awaitable? {inspect.isawaitable(eval_return_obj_unfaithful)}")
    result_unfaithful = await maybe_await(eval_return_obj_unfaithful)
    print(result_unfaithful)


if __name__ == "__main__":
    asyncio.run(evaluate_faithfulness())
