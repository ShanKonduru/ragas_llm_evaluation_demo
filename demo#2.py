# Importing evaluate library
import evaluate


def rouge(reference, candidate):
    """
    Calculate ROUGE score for a given reference and candidate sentence.

    Args:
        reference (list of str): List of reference sentences.
        candidate (list of str): List of candidate sentences.

    Returns:
        dict: ROUGE scores.
    """
    rouge_metric = evaluate.load("rouge")
    results = rouge_metric.compute(predictions=candidate, references=reference)
    return results


if __name__ == "__main__":
    print("--- Basic ROUGE Example ---")
    # Example sentences (non-tokenized)
    reference_basic = ["the cat is on the mat"]
    candidate_basic = ["the cat is on mat"]

    # Access ROUGE scores (no need for indexing into the result)
    rouge_results_basic = rouge(reference_basic, candidate_basic)
    print(f"ROUGE-1 F1 Score: {rouge_results_basic['rouge1']*100:.2f}%")
    print(f"ROUGE-L F1 Score: {rouge_results_basic['rougeL']*100:.2f}%\n")

    # ---
    print("--- ROUGE Evaluation: Pros and Cons Examples ---")
    # Each example will have a reference and a candidate
    # ROUGE scores (especially F1) will be printed for ROUGE-1 (unigram recall)
    # and ROUGE-L (longest common subsequence recall)

    # ---
    # Pros of ROUGE
    print("\n--- PROS of ROUGE ---")

    # Example 1: Capturing Key Information (Recall-Oriented)
    print("\n--- Example 1: Capturing Key Information (Recall-Oriented) ---")
    reference_1 = ["The quick brown fox jumps over the lazy dog in the field."]
    candidate_1 = ["A fox jumps over a dog in the field, quickly."]
    rouge_results_1 = rouge(reference_1, candidate_1)
    print(f"Reference: \"{reference_1[0]}\"")
    print(f"Candidate: \"{candidate_1[0]}\"")
    print(f"ROUGE-1 F1 Score: {rouge_results_1['rouge1']*100:.2f}%")
    print(f"ROUGE-L F1 Score: {rouge_results_1['rougeL']*100:.2f}%")
    print("Explanation: ROUGE can still give a decent score even with different phrasing, as long as key words/phrases are present (recall).")

    # Example 2: Handling Summarization (Recall is important)
    print("\n--- Example 2: Handling Summarization ---")
    reference_2 = [
        "The company announced record profits for the fourth quarter due to strong sales growth and efficient operations."]
    candidate_2 = ["Record profits were announced due to strong sales."]
    rouge_results_2 = rouge(reference_2, candidate_2)
    print(f"Reference: \"{reference_2[0]}\"")
    print(f"Candidate: \"{candidate_2[0]}\"")
    print(f"ROUGE-1 F1 Score: {rouge_results_2['rouge1']*100:.2f}%")
    print(f"ROUGE-L F1 Score: {rouge_results_2['rougeL']*100:.2f}%")
    print("Explanation: ROUGE often performs well for summarization, rewarding candidates that recall important information from the reference, even if they're shorter.")

    # ---
    # Cons of ROUGE
    print("\n--- CONS of ROUGE ---")

    # Example 3: Insensitivity to Factual Errors with High Overlap
    print("\n--- Example 3: Insensitivity to Factual Errors ---")
    reference_3 = ["The capital of France is Paris."]
    candidate_3 = ["The capital of France is London."]  # Factual error!
    rouge_results_3 = rouge(reference_3, candidate_3)
    print(f"Reference: \"{reference_3[0]}\"")
    print(f"Candidate: \"{candidate_3[0]}\"")
    print(f"ROUGE-1 F1 Score: {rouge_results_3['rouge1']*100:.2f}%")
    print(f"ROUGE-L F1 Score: {rouge_results_3['rougeL']*100:.2f}%")
    print("Explanation: ROUGE gives a high score due to word overlap, completely missing the factual inaccuracy. This is a major limitation.")

    # Example 4: Insensitivity to Fluency/Grammar
    print("\n--- Example 4: Insensitivity to Fluency/Grammar ---")
    reference_4 = ["The dog chased the ball across the park."]
    # Grammatically incorrect
    candidate_4 = ["ball the chased dog the park across the."]
    rouge_results_4 = rouge(reference_4, candidate_4)
    print(f"Reference: \"{reference_4[0]}\"")
    print(f"Candidate: \"{candidate_4[0]}\"")
    print(f"ROUGE-1 F1 Score: {rouge_results_4['rouge1']*100:.2f}%")
    print(f"ROUGE-L F1 Score: {rouge_results_4['rougeL']*100:.2f}%")
    print("Explanation: ROUGE can still yield a high score if individual words (ROUGE-1) or common subsequences (ROUGE-L) are present, even if the sentence is gibberish or ungrammatical.")

    # Example 5: Over-reliance on Keyword Overlap (Missing Nuance)
    print("\n--- Example 5: Over-reliance on Keyword Overlap ---")
    reference_5 = [
        "The government implemented a new policy to reduce carbon emissions."]
    # Semantically similar, lexically different
    candidate_5 = [
        "The new government rule aims to cut down greenhouse gases."]
    rouge_results_5 = rouge(reference_5, candidate_5)
    print(f"Reference: \"{reference_5[0]}\"")
    print(f"Candidate: \"{candidate_5[0]}\"")
    print(f"ROUGE-1 F1 Score: {rouge_results_5['rouge1']*100:.2f}%")
    print(f"ROUGE-L F1 Score: {rouge_results_5['rougeL']*100:.2f}%")
    print("Explanation: Despite the candidate being a perfectly valid paraphrase, ROUGE scores can be lower because it lacks semantic understanding and relies purely on lexical overlap.")

    # Example 6: Bias towards longer summaries/candidates (ROUGE-N recall)
    print("\n--- Example 6: Bias towards Longer Candidates ---")
    reference_6 = ["The quick brown fox jumps over the lazy dog."]
    # Adds irrelevant info
    candidate_6 = [
        "The quick brown fox jumps over the lazy dog. It was a sunny day and the dog was very lazy."]
    rouge_results_6 = rouge(reference_6, candidate_6)
    print(f"Reference: \"{reference_6[0]}\"")
    print(f"Candidate: \"{candidate_6[0]}\"")
    print(f"ROUGE-1 F1 Score: {rouge_results_6['rouge1']*100:.2f}%")
    print(f"ROUGE-L F1 Score: {rouge_results_6['rougeL']*100:.2f}%")
    print("Explanation: A longer candidate that incorporates the reference can sometimes achieve a high ROUGE score, even if it adds irrelevant or redundant information, particularly if looking at recall metrics alone.")

    print("--- ROUGE Evaluation: Lack of Diversity Example ---")
    # Example 7: Lack of Diversity (Repetitive Output) - FIXED
    print("\n--- Example 7 (FIXED): Lack of Diversity / Repetitive Output ---")
    # For a single candidate, we provide a list of *one* list of references.
    # The inner list can contain multiple valid human references for that *single* candidate.
    # This is the correct way to structure multiple references for one prediction with `evaluate`
    # However, to showcase "lack of diversity" by the *model's output*, we'll still use a single reference
    # and then show how ROUGE still scores high on repetitive candidates.

    # We will use a single reference for the score calculation to simplify,
    # but the *concept* of diversity for an LLM means it *could* have generated
    # different valid outputs if prompted multiple times. ROUGE doesn't measure that.

    single_reference_for_diversity_test = [
        "The quick brown fox jumps over the lazy dog."]

    # Good, non-repetitive
    candidate_7_a = ["The quick brown fox jumps over the lazy dog."]
    candidate_7_b = [
        "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog."]  # Repetitive
    # Even more repetitive
    candidate_7_c = [
        "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog."]

    print(f"Reference: \"{single_reference_for_diversity_test[0]}\"")

    print("\n--- Candidate A (Non-Repetitive) ---")
    # Pass reference as [[reference_string]] for a single candidate
    rouge_results_7_a = rouge(
        [single_reference_for_diversity_test], [candidate_7_a[0]])
    print(f"Candidate: \"{candidate_7_a[0]}\"")
    print(f"ROUGE-1 F1 Score: {rouge_results_7_a['rouge1']*100:.2f}%")
    print(f"ROUGE-L F1 Score: {rouge_results_7_a['rougeL']*100:.2f}%")
    print("Explanation: This candidate is good, and ROUGE reflects high overlap. However, ROUGE doesn't assess if the model *could* have generated other valid outputs for diversity.")

    print("\n--- Candidate B (Repetitive) ---")
    rouge_results_7_b = rouge(
        [single_reference_for_diversity_test], [candidate_7_b[0]])
    print(f"Candidate: \"{candidate_7_b[0]}\"")
    print(f"ROUGE-1 F1 Score: {rouge_results_7_b['rouge1']*100:.2f}%")
    print(f"ROUGE-L F1 Score: {rouge_results_7_b['rougeL']*100:.2f}%")
    print("Explanation: Despite being repetitive, ROUGE scores can remain high or even increase slightly because the 'correct' n-grams are repeated, increasing recall, without penalizing redundancy. This shows a lack of diversity penalty.")

    print("\n--- Candidate C (Highly Repetitive) ---")
    rouge_results_7_c = rouge(
        [single_reference_for_diversity_test], [candidate_7_c[0]])
    print(f"Candidate: \"{candidate_7_c[0]}\"")
    print(f"ROUGE-1 F1 Score: {rouge_results_7_c['rouge1']*100:.2f}%")
    print(f"ROUGE-L F1 Score: {rouge_results_7_c['rougeL']*100:.2f}%")
    print("Explanation: The more a correct phrase is repeated, the higher the ROUGE recall can go, even though the output quality for a human is severely degraded due to lack of diversity and extreme redundancy. ROUGE provides no inherent mechanism to penalize this.")
