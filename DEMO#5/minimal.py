# Demo #5: Ragas for Faithfulness (with Ollama Local LLM) - Optimized for Ragas 0.2.15
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv() # Load environment variables

# --- Configuration ---
OLLAMA_CHAT_MODEL_NAME = os.getenv("OLLAMA_CHAT_MODEL_NAME", "llama3.2")
OLLAMA_EMBEDDING_MODEL_NAME = os.getenv("OLLAMA_EMBEDDING_MODEL_NAME", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

async def initialize_ollama_llm(model_name: str, base_url: str):
    """Initializes and returns a wrapped Ollama Chat LLM."""
    try:
        ollama_chat_model = ChatOllama(
            model=model_name,
            temperature=0.0, # Keep temperature low for deterministic evaluations
            base_url=base_url
        )
        logger.info(f"Ollama Chat LLM ('{model_name}') initialized.")
        return LangchainLLMWrapper(ollama_chat_model)
    except Exception as e:
        logger.error(f"Error initializing Ollama Chat LLM ('{model_name}'): {e}")
        logger.error("Please ensure:")
        logger.error(f"1. Ollama server is running (run 'ollama serve' in your terminal).")
        logger.error(f"2. Model '{model_name}' is pulled in Ollama (run 'ollama pull {model_name}').")
        logger.error("3. 'langchain-community' is installed.")
        raise

async def initialize_ollama_embeddings(model_name: str, base_url: str):
    """Initializes and returns wrapped Ollama Embeddings."""
    try:
        ollama_embeddings_model = OllamaEmbeddings(model=model_name, base_url=base_url)
        logger.info(f"Ollama Embeddings ('{model_name}') initialized.")
        return LangchainEmbeddingsWrapper(ollama_embeddings_model)
    except Exception as e:
        logger.error(f"Error initializing Ollama Embeddings ('{model_name}'): {e}")
        logger.error("Please ensure:")
        logger.error(f"1. Ollama server is running (run 'ollama serve' in your terminal).")
        logger.error(f"2. Model '{model_name}' is pulled in Ollama (run 'ollama pull {model_name}').")
        logger.error("3. 'langchain-community' is installed.")
        raise

def create_faithfulness_dataset(question: str, answer: str, contexts: list[str], ground_truth: str) -> Dataset:
    """Creates a Hugging Face Dataset for faithfulness evaluation."""
    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
        "ground_truth": [ground_truth]
    }
    return Dataset.from_dict(data)

async def evaluate_faithfulness_ollama_v0_2_15():
    """Evaluates faithfulness using Ragas and Ollama with different scenarios."""
    evaluator_llm = None
    evaluator_embeddings = None

    try:
        evaluator_llm = await initialize_ollama_llm(OLLAMA_CHAT_MODEL_NAME, OLLAMA_BASE_URL)
        evaluator_embeddings = await initialize_ollama_embeddings(OLLAMA_EMBEDDING_MODEL_NAME, OLLAMA_BASE_URL)
    except Exception:
        # Initialization failed, exit gracefully
        return

    # Example data for faithful scenario
    dataset_faithful = create_faithfulness_dataset(
        question="What is the capital of France?",
        answer="Paris is the capital of France.",
        contexts=["Paris is the capital and most populous city of France. It is located on the Seine River."],
        ground_truth="Paris"
    )

    # Example data for unfaithful scenario (hallucination)
    dataset_unfaithful = create_faithfulness_dataset(
        question="What is the capital of France?",
        answer="The capital of France is Berlin, which is also a major city in Germany.",
        contexts=["Paris is the capital and most populous city of France. It is located on the Seine River."],
        ground_truth="Paris"
    )

    # Instantiate the Faithfulness metric
    faithfulness_metric = Faithfulness()
    # Explicitly assign your wrapped Ollama LLM and Embeddings to the metric
    faithfulness_metric.llm = evaluator_llm
    faithfulness_metric.embeddings = evaluator_embeddings

    # --- Evaluating Faithfulness (Faithful Scenario) ---
    logger.info(f"\n--- Evaluating Faithfulness (Faithful Scenario) with Ollama ({OLLAMA_CHAT_MODEL_NAME}) ---")
    try:
        result_faithful = await evaluate(
            dataset_faithful,
            metrics=[faithfulness_metric],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings
        )
        logger.info("Faithful Scenario Result:\n%s", result_faithful)
    except TypeError as e:
        logger.error(f"Caught TypeError for faithful scenario: {e}")
        logger.error("This indicates an issue with `evaluate` not returning an awaitable object.")
        logger.error("Please double-check Ollama server status and model availability.")
        logger.error("Consider upgrading Ragas if this persists, as it likely indicates a version-specific quirk.")
    except Exception as e:
        logger.error(f"An unexpected error occurred for faithful scenario: {e}")

    # --- Evaluating Faithfulness (Unfaithful Scenario - Hallucination) ---
    logger.info(f"\n--- Evaluating Faithfulness (Unfaithful Scenario - Hallucination) with Ollama ({OLLAMA_CHAT_MODEL_NAME}) ---")
    try:
        result_unfaithful = await evaluate(
            dataset_unfaithful,
            metrics=[faithfulness_metric],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings
        )
        logger.info("Unfaithful Scenario Result:\n%s", result_unfaithful)
    except TypeError as e:
        logger.error(f"Caught TypeError for unfaithful scenario: {e}")
        logger.error("This indicates an issue with `evaluate` not returning an awaitable object.")
        logger.error("Please double-check Ollama server status and model availability.")
        logger.error("Consider upgrading Ragas if this persists, as it likely indicates a version-specific quirk.")
    except Exception as e:
        logger.error(f"An unexpected error occurred for unfaithful scenario: {e}")

if __name__ == "__main__":
    asyncio.run(evaluate_faithfulness_ollama_v0_2_15())