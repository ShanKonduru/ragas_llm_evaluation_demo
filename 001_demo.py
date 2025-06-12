
import os
from dotenv import load_dotenv
import asyncio

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import BleuScore

load_dotenv()

async def main_evaluation(): # Define an async function
    final_score = await scorer.single_turn_ascore(sample)
    print(f"BLEU Score: {final_score}")

if __name__ == "__main__":
    sample = SingleTurnSample(
        response="The Eiffel Tower is located in India.",
        reference="The Eiffel Tower is located in Paris."
    )
    scorer = BleuScore()

    asyncio.run(main_evaluation()) # Run the async function