# Demo #5: Ragas for Faithfulness (with Ollama Local LLM)
import os
from dotenv import load_dotenv
import asyncio
from ragas.metrics import Faithfulness
from ragas import evaluate, init # <-- IMPORT `init` HERE
from datasets import Dataset

# Import the necessary components for Ollama from LangChain
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

load_dotenv() # Load environment variables (e.g., if you set OLLAMA_HOST or OLLAMA_BASE_URL)

async def evaluate_faithfulness_ollama():
    # --- Configure your Ollama LLM and Embeddings ---
    OLLAMA_MODEL_NAME = "llama3.2" # Or "llama3.3"

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
        ollama_embeddings_model = OllamaEmbeddings(
            model=OLLAMA_MODEL_NAME, # Use the same model, or a dedicated embedding model like "nomic-embed-text"
            # If Ollama is not on default localhost:11434, uncomment and set base_url:
            # base_url="http://localhost:11434"
        )
        evaluator_embeddings = LangchainEmbeddingsWrapper(ollama_embeddings_model)

        # *** CRUCIAL ADDITION: Initialize Ragas globally with your Ollama instances ***
        # This explicitly tells Ragas to use these models for all its internal operations.
        init(llm=evaluator_llm, embeddings=evaluator_embeddings)
        print("Ragas initialized successfully with Ollama LLM and Embeddings.")

    except Exception as e:
        print(f"Error initializing Ollama LLM/Embeddings or Ragas: {e}")
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
    # While you could set .llm and .embeddings here, `init()` handles it globally.
    # It doesn't hurt to leave them, but it's often redundant after `init()`.
    # faithfulness_metric.llm = evaluator_llm
    # faithfulness_metric.embeddings = evaluator_embeddings


    print(f"\n--- Evaluating Faithfulness (Faithful Scenario) with Ollama ({OLLAMA_MODEL_NAME}) ---")
    result_faithful = await evaluate(
        dataset_faithful,
        metrics=[faithfulness_metric],
        # Although `init()` should handle it, explicitly passing them here adds robustness
        # and can sometimes resolve tricky initialization issues in certain versions.
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )
    print(result_faithful)

    print(f"\n--- Evaluating Faithfulness (Unfaithful Scenario - Hallucination) with Ollama ({OLLAMA_MODEL_NAME}) ---")
    result_unfaithful = await evaluate(
        dataset_unfaithful,
        metrics=[faithfulness_metric],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )
    print(result_unfaithful)

if __name__ == "__main__":
    asyncio.run(evaluate_faithfulness_ollama())