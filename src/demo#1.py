
import os
from dotenv import load_dotenv
import asyncio

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import BleuScore

load_dotenv()

async def main_evaluation(description): # Define an async function
    final_score = await scorer.single_turn_ascore(sample)
    print(f"For {description} the BLEU Score: {final_score}")

if __name__ == "__main__":
    test_cases = [
        # PROS: High BLEU score for highly similar or exact matches
        {
            "response": "The Eiffel Tower is located in Paris.",
            "reference": "The Eiffel Tower is located in Paris.",
            "description": "Exact Match"
        },
        {
            "response": "The Eiffel Tower is in Paris.",
            "reference": "The Eiffel Tower is located in Paris.",
            "description": "Minor Variation"
        },
        # CONS: Low BLEU score despite semantic similarity (synonyms/paraphrasing)
        {
            "response": "The Parisian landmark is in France's capital.",
            "reference": "The Eiffel Tower is located in Paris.",
            "description": "Semantic Similarity, Lexical Dissimilarity"
        },
        {
            "response": "India is home to the Taj Mahal.",
            "reference": "The Taj Mahal is in India.",
            "description": "Semantic Similarity with Word Order Change"
        },
        # CONS: Can give decent score for grammatically incorrect but high word overlap
        {
            "response": "located Tower Eiffel The Paris in is.",
            "reference": "The Eiffel Tower is located in Paris.",
            "description": "Grammatically Incorrect but High Word Overlap"
        },
        # CONS: Fails to capture factual errors if word overlap is high
        {
            "response": "The Eiffel Tower is located in India.", # Factual error!
            "reference": "The Eiffel Tower is located in Paris.",
            "description": "Factual Error, High Word Overlap"
        },
        # CONS: Heavily penalizes missing a single important word
        {
            "response": "Eiffel Tower is located in Paris.",
            "reference": "The Eiffel Tower is located in Paris.",
            "description": "Missing Important Word"
        },
        # CONS: May reward generic or short sentences that share common words
        {
            "response": "It is good.",
            "reference": "The new movie is really good.",
            "description": "Generic/Short Sentence"
        },
    ]
    
    print("--- BLEU Evaluation Examples (using NLTK) ---")
    for i, case in enumerate(test_cases):
        response=case['response']
        reference=case['reference']
        description=case['description']
        sample = SingleTurnSample(
            response=response,
            reference=reference
        )
        scorer = BleuScore()
        asyncio.run(main_evaluation(description)) # Run the async function