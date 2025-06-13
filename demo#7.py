import os
from dotenv import load_dotenv
import asyncio
from ragas.metrics import ContextRelevance, ContextRecall
from ragas import evaluate
from datasets import Dataset

load_dotenv()

# Initialize Ragas metrics globally or within functions if preferred
# Initializing here avoids re-instantiating them if they don't hold state
context_relevancy_metric = ContextRelevance()
context_recall_metric = ContextRecall()


async def evaluate_positive_scenarios():
    """
    Evaluates scenarios where contexts are expected to have high relevance and recall.
    """
    print("\n--- Evaluating Positive Scenarios (Expected High Scores) ---")

    data_positive = {
        "question": ["What is the capital of Japan?",
                     "Who painted the Mona Lisa?"],
        "answer": [
            "The capital of Japan is Tokyo.",
            "Leonardo da Vinci painted the Mona Lisa.",
        ],
        "contexts": [
            [
                "Tokyo is the capital city of Japan and its largest metropolis."
            ],
            [
                "Leonardo da Vinci was an Italian polymath who created the Mona Lisa, among other works. The painting is famous for its enigmatic smile. It was painted between 1503 and 1519."
            ],
        ],
        "reference": [
            "Tokyo is the largest city and capital of Japan.",
            "The Mona Lisa was painted by Leonardo da Vinci between 1503 and 1519.",
        ],
        "ground_truths": [
            [
                "Tokyo is the largest city and capital of Japan."
            ],
            ["The Mona Lisa was painted by Leonardo da Vinci between 1503 and 1519."],
        ],
    }
    dataset_positive = Dataset.from_dict(data_positive)

    results = evaluate(
        dataset_positive, metrics=[
            context_relevancy_metric, context_recall_metric]
    )
    print(results)


async def evaluate_negative_scenarios():
    """
    Evaluates scenarios where contexts are expected to have low relevance and/or recall.
    """
    print("\n--- Evaluating Negative Scenarios (Expected Low Scores) ---")

    data_negative = {
        "question": [
            "What is the capital of France and what is it famous for?",
            "Who wrote 'Romeo and Juliet'?",
            "What are the benefits of exercise?",
            "Tell me about the recent moon landing.",
        ],
        "answer": [
            "The capital of France is Paris. It is famous for the Eiffel Tower and its art museums.",
            "William Shakespeare wrote 'Romeo and Juliet'.",
            "Exercise helps improve cardiovascular health and can reduce stress.",
            "Humans recently landed on the moon as part of the Artemis program."
        ],
        "contexts": [
            [
                # Low Context Relevance: Irrelevant information mixed in.
                "Paris is the capital city of France, known for the Eiffel Tower and the Louvre Museum. The weather in Paris in winter can be quite cold. French cuisine is also very diverse."
            ],
            [
                # Low Context Recall: Missing key information from reference.
                "William Shakespeare was an English playwright and poet."
            ],
            [
                # Low Context Relevance AND Low Context Recall: Irrelevant info, and missing key benefits.
                "Regular exercise can lead to muscle gain and improved mood. Many people enjoy jogging. Healthy eating is also important."
            ],
            [
                # Zero Context Recall: The context provides no support for the reference.
                "The first moon landing was in 1969. Astronauts Neil Armstrong and Buzz Aldrin were the first to walk on the moon."
            ]
        ],
        "reference": [
            "The capital of France is Paris, famous for landmarks like the Eiffel Tower and world-renowned art museums such as the Louvre.",
            "William Shakespeare, an English playwright, authored 'Romeo and Juliet', a tragic love story.",
            "Benefits of exercise include improved cardiovascular health, reduced stress, weight management, and stronger bones.",
            "A recent human moon landing occurred as part of the Artemis III mission in 2026."
        ],
        "ground_truths": [
            [
                "The capital of France is Paris, famous for landmarks like the Eiffel Tower and world-renowned art museums such as the Louvre."
            ],
            [
                "William Shakespeare, an English playwright, authored 'Romeo and Juliet', a tragic love story."
            ],
            [
                "Benefits of exercise include improved cardiovascular health, reduced stress, weight management, and stronger bones."
            ],
            [
                "A recent human moon landing occurred as part of the Artemis III mission in 2026."
            ]
        ],
    }
    dataset_negative = Dataset.from_dict(data_negative)

    results = evaluate(
        dataset_negative, metrics=[
            context_relevancy_metric, context_recall_metric]
    )
    print(results)


if __name__ == "__main__":
    # Run both evaluation functions
    asyncio.run(evaluate_positive_scenarios())
    asyncio.run(evaluate_negative_scenarios())
