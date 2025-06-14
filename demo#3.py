# NLTK and rouge-score (Traditional & Granular)

import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

# Download necessary NLTK data
nltk.download("punkt_tab")

# Example sentences
reference = ["the cat is on the mat"]
candidate = ["the cat is on mat"]

# Tokenize the reference and candidate
reference_tokenized = [nltk.word_tokenize(ref) for ref in reference]
candidate_tokenized = [nltk.word_tokenize(cand) for cand in candidate]

# BLEU Score Calculation using NLTK
bleu_score = sentence_bleu(reference_tokenized, candidate_tokenized[0])
print(f"BLEU Score (NLTK): {bleu_score * 100:.2f}")

# ROUGE Score Calculation using rouge-score
scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
scores = scorer.score(reference[0], candidate[0])
print(f"ROUGE-1 F1 Score: {scores['rouge1'].fmeasure:.2f}")
print(f"ROUGE-L F1 Score: {scores['rougeL'].fmeasure:.2f}")
