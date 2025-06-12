from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer

# Example sentences
reference = ["the cat is on the mat"]
candidate = ["the cat is on mat"]

# BLEU Score Calculation
bleu = corpus_bleu(candidate, [reference])
print(f"BLEU Score: {bleu.score}")

# ROUGE Score Calculation
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
scores = scorer.score(reference[0], candidate[0])
print(f"ROUGE-1: {scores['rouge1']}")
print(f"ROUGE-L: {scores['rougeL']}")
