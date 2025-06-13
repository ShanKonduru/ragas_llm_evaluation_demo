from ragas.metrics import Faithfulness
from ragas import evaluate
from datasets import Dataset
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
import inspect


class RagasFaithfulnessEvaluator:
    """
    A class to encapsulate the Ragas faithfulness evaluation logic.
    It handles environment variable loading, LLM initialization, and running evaluations.
    """

    def __init__(self, open_ai_api_key, model_name: str = "gpt-4o"):
        """
        Initializes the RagasFaithfulnessEvaluator.

        Args:
            model_name (str): The name of the OpenAI model to use for evaluation.
        """

        self.open_ai_api_key = open_ai_api_key

        # Initialize the OpenAI LLM for Ragas evaluation
        # Ensure 'langchain-openai' is installed: pip install langchain-openai
        self.evaluator_llm = LangchainLLMWrapper(
            ChatOpenAI(model=model_name, openai_api_key=self.open_ai_api_key)
        )
        self.faithfulness_metric = Faithfulness()

    @staticmethod
    async def maybe_await(obj):
        """
        Awaits an object if it's a coroutine or other awaitable, otherwise returns the object as-is.
        This helps handle cases where an async function might sometimes return a direct result,
        avoiding TypeError when awaiting non-awaitable objects.
        """
        return await obj if inspect.isawaitable(obj) else obj

    async def run_evaluation(self, dataset: Dataset):
        """
        Runs the evaluation for the given data set
        and prints the results.

        Args:
            dataset (datasets.Dataset): The dataset for the scenario.
        """
        print(f"\n--- Evaluating the data set ({dataset}) ---")
        # Call evaluate and store the returned object
        eval_return_obj_faithful = evaluate(
            dataset, metrics=[
                self.faithfulness_metric], llm=self.evaluator_llm
        )
        print(
            f"Type of object returned by ragas.evaluate (faithful): {type(eval_return_obj_faithful)}"
        )
        print(
            f"Is it awaitable? {inspect.isawaitable(eval_return_obj_faithful)}")
        # Conditionally await the object using the helper function
        result_faithful = await self.maybe_await(eval_return_obj_faithful)
        print(result_faithful)