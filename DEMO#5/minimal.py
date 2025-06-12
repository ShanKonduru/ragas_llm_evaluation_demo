# Demo #5: Ragas for Faithfulness (with Ollama Local LLM) - Optimized for Ragas 0.2.15
import os
from dotenv import load_dotenv
import asyncio
from ragas.metrics import Faithfulness
from ragas import evaluate # No 'init' here
from datasets import Dataset

# Import the necessary components for Ollama from LangChain
# from langchain_community.chat_models import ChatOllama # OLD
# from langchain_community.embeddings import OllamaEmbeddings # OLD
from langchain_ollama import ChatOllama, OllamaEmbeddings # NEW

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

load_dotenv() # Load environment variables (e.g., if you set OLLAMA_HOST or OLLAMA_BASE_URL)

async def evaluate_faithfulness_ollama_v0_2_15():
    # --- Configure your Ollama LLM and Embeddings ---
    OLLAMA_MODEL_NAME = "llama3.2" # Or "llama3.3" or even "nomic-embed-text" for embeddings

    try:
        # Initialize ChatOllama for the evaluation LLM
        ollama_chat_model = ChatOllama(
            model=OLLAMA_MODEL_NAME,
            temperature=0.0, # Keep temperature low for deterministic evaluations
            # If Ollama is not on default localhost:11434, uncomment and set base_url:
            # base_url="http://localhost:11434"
        )
        evaluator_llm = LangchainLLMWrapper(ollama_chat_model)

        # Initialize OllamaEmbeddings
        # It's crucial to define embeddings for Ragas's internal operations,
        # even if a specific metric like Faithfulness primarily uses the LLM.
        # Consider using a dedicated embedding model like "nomic-embed-text"
        # if your LLM isn't performing well for embeddings or causes issues.
        ollama_embeddings_model = OllamaEmbeddings(model="nomic-embed-text")
        evaluator_embeddings = LangchainEmbeddingsWrapper(ollama_embeddings_model)

        print(f"Ollama LLM ('{OLLAMA_MODEL_NAME}') and Embeddings initialized.")

    except Exception as e:
        print(f"Error initializing Ollama LLM or Embeddings: {e}")
        print("Please ensure:")
        print("1. Ollama server is running (run 'ollama serve' in your terminal).")
        print(f"2. Model '{OLLAMA_MODEL_NAME}' is pulled in Ollama (run 'ollama pull {OLLAMA_MODEL_NAME}').")
        print("3. 'langchain-community' is installed.")
        return # Exit if LLM fails to initialize

    # Example data (your hardcoded examples are perfectly fine)
    data_faithful = {
        "question": ["What is the capital of France?"],
        "answer": ["Paris is the capital of France."],
        "contexts": [["Paris is the capital and most populous city of France. It is located on the Seine River."]],
        "ground_truth": ["Paris"]
    }
    dataset_faithful = Dataset.from_dict(data_faithful)

    data_unfaithful = {
        "question": ["What is the capital of France?"],
        "answer": ["The capital of France is Berlin, which is also a major city in Germany."],
        "contexts": [["Paris is the capital and most populous city of France. It is located on the Seine River."]],
        "ground_truth": ["Paris"]
    }
    dataset_unfaithful = Dataset.from_dict(data_unfaithful)

    # Instantiate the Faithfulness metric
    faithfulness_metric = Faithfulness()
    # Explicitly assign your wrapped Ollama LLM and Embeddings to the metric
    # This is critical for Ragas 0.2.15 where global `init` isn't available.
    faithfulness_metric.llm = evaluator_llm
    faithfulness_metric.embeddings = evaluator_embeddings


    print(f"\n--- Evaluating Faithfulness (Faithful Scenario) with Ollama ({OLLAMA_MODEL_NAME}) ---")
    try:
        result_faithful = await evaluate(
            dataset_faithful,
            metrics=[faithfulness_metric],
            # Crucially, pass LLM and Embeddings directly to evaluate() for Ragas 0.2.15
            llm=evaluator_llm,
            embeddings=evaluator_embeddings
        )
        print(result_faithful)
    except TypeError as e:
        print(f"Caught TypeError for faithful scenario: {e}")
        print("This indicates an issue with `evaluate` not returning an awaitable object.")
        print("Please double-check Ollama server status and model availability.")
        print("Consider upgrading Ragas if this persists, as it likely indicates a version-specific quirk.")
    except Exception as e:
        print(f"An unexpected error occurred for faithful scenario: {e}")


    print(f"\n--- Evaluating Faithfulness (Unfaithful Scenario - Hallucination) with Ollama ({OLLAMA_MODEL_NAME}) ---")
    try:
        result_unfaithful = await evaluate(
            dataset_unfaithful,
            metrics=[faithfulness_metric],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings
        )
        print(result_unfaithful)
    except TypeError as e:
        print(f"Caught TypeError for unfaithful scenario: {e}")
        print("This indicates an issue with `evaluate` not returning an awaitable object.")
        print("Please double-check Ollama server status and model availability.")
        print("Consider upgrading Ragas if this persists, as it likely indicates a version-specific quirk.")
    except Exception as e:
        print(f"An unexpected error occurred for unfaithful scenario: {e}")


if __name__ == "__main__":
    asyncio.run(evaluate_faithfulness_ollama_v0_2_15())