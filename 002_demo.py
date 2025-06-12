# Importing evaluate library
import evaluate


def bleu(reference, candidate):
    """
    Calculate BLEU score for a given reference and candidate sentence.

    Args:
        reference (list of str): List of reference sentences.
        candidate (list of str): List of candidate sentences.

    Returns:
        float: BLEU score.
    """
    bleu_metric = evaluate.load("bleu")
    results = bleu_metric.compute(predictions=candidate, references=reference)
    return results["bleu"]


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
    # Example sentences (non-tokenized)
    reference = ["the cat is on the mat"]
    candidate = ["the cat is on mat"]

    # BLEU expects plain text inputs
    bleu_results = bleu(reference, candidate)
    print(f"BLEU Score: %{bleu_results * 100:.2f}")

    # Access ROUGE scores (no need for indexing into the result)
    rouge_results = rouge(reference, candidate)
    print(f"ROUGE-1 F1 Score: %{rouge_results['rouge1']*100:.2f}")
    print(f"ROUGE-L F1 Score: %{rouge_results['rougeL']*100:.2f}")
