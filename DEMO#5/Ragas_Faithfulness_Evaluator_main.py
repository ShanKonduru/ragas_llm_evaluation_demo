from Ragas_Faithfulness_Evaluator import RagasFaithfulnessEvaluator
import logging
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    """Main function to demonstrate the RagasFaithfulnessEvaluator class."""
    evaluator = RagasFaithfulnessEvaluator(logger)

    # Setup the evaluator (initialize LLMs, embeddings, and Ragas metric)
    setup_successful = await evaluator.setup()
    if not setup_successful:
        logger.error("Exiting due to setup failure.")
        return

    # Create datasets
    dataset_faithful = evaluator.create_faithfulness_dataset(
        question="What is the capital of France?",
        answer="Paris is the capital of France.",
        contexts=["Paris is the capital and most populous city of France. It is located on the Seine River."],
        ground_truth="Paris"
    )

    dataset_unfaithful = evaluator.create_faithfulness_dataset(
        question="What is the capital of France?",
        answer="The capital of France is Berlin, which is also a major city in Germany.",
        contexts=["Paris is the capital and most populous city of France. It is located on the Seine River."],
        ground_truth="Paris"
    )

    # Evaluate scenarios
    await evaluator.evaluate_dataset(dataset_faithful, "Faithful Scenario")
    await evaluator.evaluate_dataset(dataset_unfaithful, "Unfaithful Scenario - Hallucination")

if __name__ == "__main__":
    asyncio.run(main())