import os
from dotenv import load_dotenv
import asyncio
import logging

from ragas.metrics import Faithfulness
from ragas import evaluate
from datasets import Dataset

from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


class RagasFaithfulnessEvaluator:
    """
    A class to perform faithfulness evaluation using Ragas with Ollama LLMs and Embeddings.
    """

    def __init__(self,
                 logger,
                 ollama_chat_model_name: str = "phi3:mini",
                 ollama_embedding_model_name: str = "nomic-embed-text",
                 ollama_base_url: str = "http://localhost:11434",
                 ollama_timeout: int = 120):
        """
        Initializes the RagasFaithfulnessEvaluator with Ollama model names and base URL.

        Args:
            ollama_chat_model_name (str): The name of the Ollama model to use for chat (e.g., "llama3.2").
            ollama_embedding_model_name (str): The name of the Ollama model to use for embeddings (e.g., "nomic-embed-text").
            ollama_base_url (str): The base URL for the Ollama server.
        """
        self.ollama_timeout = ollama_timeout
        self.logger = logger
        self.ollama_chat_model_name = os.getenv(
            "OLLAMA_CHAT_MODEL_NAME", ollama_chat_model_name)
        self.ollama_embedding_model_name = os.getenv(
            "OLLAMA_EMBEDDING_MODEL_NAME", ollama_embedding_model_name)
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", ollama_base_url)

        self.evaluator_llm = None
        self.evaluator_embeddings = None
        self.faithfulness_metric = None

    async def _initialize_ollama_llm(self):
        """Initializes and sets the wrapped Ollama Chat LLM."""
        try:
            ollama_chat_model = ChatOllama(
                model=self.ollama_chat_model_name,
                temperature=0.0,  # Keep temperature low for deterministic evaluations
                base_url=self.ollama_base_url,
                timeout=self.ollama_timeout  # Pass the timeout here
            )
            self.evaluator_llm = LangchainLLMWrapper(ollama_chat_model)
            self.logger.info(
                f"Ollama Chat LLM ('{self.ollama_chat_model_name}') initialized.")
        except Exception as e:
            self.logger.error(
                f"Error initializing Ollama Chat LLM ('{self.ollama_chat_model_name}'): {e}")
            self.logger.error("Please ensure:")
            self.logger.error(
                f"1. Ollama server is running (run 'ollama serve' in your terminal).")
            self.logger.error(
                f"2. Model '{self.ollama_chat_model_name}' is pulled in Ollama (run 'ollama pull {self.ollama_chat_model_name}').")
            self.logger.error("3. 'langchain-community' is installed.")
            raise

    async def _initialize_ollama_embeddings(self):
        """Initializes and sets the wrapped Ollama Embeddings."""
        try:
            ollama_embeddings_model = OllamaEmbeddings(
                model=self.ollama_embedding_model_name, base_url=self.ollama_base_url)
            self.evaluator_embeddings = LangchainEmbeddingsWrapper(
                ollama_embeddings_model)
            self.logger.info(
                f"Ollama Embeddings ('{self.ollama_embedding_model_name}') initialized.")
        except Exception as e:
            self.logger.error(
                f"Error initializing Ollama Embeddings ('{self.ollama_embedding_model_name}'): {e}")
            self.logger.error("Please ensure:")
            self.logger.error(
                f"1. Ollama server is running (run 'ollama serve' in your terminal).")
            self.logger.error(
                f"2. Model '{self.ollama_embedding_model_name}' is pulled in Ollama (run 'ollama pull {self.ollama_embedding_model_name}').")
            self.logger.error("3. 'langchain-community' is installed.")
            raise

    async def setup(self):
        """Initializes all necessary components (LLM, Embeddings, and Ragas metric)."""
        try:
            await self._initialize_ollama_llm()
            await self._initialize_ollama_embeddings()

            self.faithfulness_metric = Faithfulness()
            # Assign LLM and Embeddings to the metric as required by Ragas 0.2.15
            self.faithfulness_metric.llm = self.evaluator_llm
            self.faithfulness_metric.embeddings = self.evaluator_embeddings
            self.logger.info("Ragas Faithfulness metric initialized.")
            return True
        except Exception:
            self.logger.error("Failed to set up RagasFaithfulnessEvaluator.")
            return False

    def create_faithfulness_dataset(self, question: str, answer: str, contexts: list[str], ground_truth: str) -> Dataset:
        """Creates a Hugging Face Dataset for faithfulness evaluation."""
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [ground_truth]
        }
        return Dataset.from_dict(data)

    async def evaluate_dataset(self, dataset: Dataset, scenario_name: str = "Evaluation Scenario") -> dict:
        """
        Evaluates a given dataset for faithfulness.

        Args:
            dataset (Dataset): The dataset to evaluate.
            scenario_name (str): A descriptive name for the evaluation scenario (for logging).

        Returns:
            dict: The evaluation results.
        """
        if not self.faithfulness_metric:
            self.logger.error("Evaluator not set up. Call .setup() first.")
            return {"error": "Evaluator not set up"}

        self.logger.info(
            f"\n--- Evaluating Faithfulness ({scenario_name}) with Ollama ({self.ollama_chat_model_name}) ---")
        try:
            result = await evaluate(
                dataset,
                metrics=[self.faithfulness_metric],
                llm=self.evaluator_llm,
                embeddings=self.evaluator_embeddings
            )
            self.logger.info(f"{scenario_name} Result:\n%s", result)
            return result
        except TypeError as e:
            self.logger.error(f"Caught TypeError for {scenario_name}: {e}")
            self.logger.error(
                "This indicates an issue with `evaluate` not returning an awaitable object.")
            self.logger.error(
                "Please double-check Ollama server status and model availability.")
            self.logger.error(
                "Consider upgrading Ragas if this persists, as it likely indicates a version-specific quirk.")
            return {"error": str(e)}
        except Exception as e:
            self.logger.error(
                f"An unexpected error occurred for {scenario_name}: {e}")
            return {"error": str(e)}
