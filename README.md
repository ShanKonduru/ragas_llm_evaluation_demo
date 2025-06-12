# Ragas_llm_evaluation_demo

## Description

## BLEU vs. ROUGE: When to Use Which?

Both BLEU and ROUGE scores serve different purposes. 
BLEU is more suited for tasks where precision is important, such as machine translation, where it is necessary to generate grammatically and contextually correct sentences.
ROUGE, on the other hand, is recall-oriented, making it better for summarization tasks where it is more important to capture all key points rather than the exact phrasing.

Use BLEU when evaluating machine translation tasks, where precision and fluency are critical.
Use ROUGE for summarization tasks where capturing key ideas and recall is more important than exact wording.

## Conclusion

BLEU are ROUGE are two of the most widely used performance evaluation metrics for evaluating NLP based models, especially in tasks related to machine translation and text summarization. Both scores are effective and serve different purpose focusing on different tasks, like BLEU emphasizes the precision, like how much of the generated output appears in the reference, and ROUGE focuses on recall, like how much of the reference appears in the generated output.

## Key Takeaways:

BLEU is very useful in evaluating translation models, where the precision is more important, and ofcourse it penalizes shorter candidate sentences.
ROUGE is most likely suitable for tasks related to text summarization, where the length of the generated output is usually shorter than the reference, and recall is a key factor here.
While these both metrics BLEU and ROUGE are very useful, they should be done with human evaluation for some tasks that require more nuanced understanding, such as natural language generation.
With the full understanding of BLEU and ROUGE scores for NLP evaluation, you can completely assess your NLP models on your own using these performance evaluation metrics effectively.

## Installation


1.  **Initialize git (Windows):**
    Run the `000_init.bat` file.

2.  **Create a virtual environment (Windows):**
    Run the `001_env.bat` file.

3.  **Activate the virtual environment (Windows):**
    Run the `002_activate.bat` file.

4.  **Install dependencies:**
    Run the `003_setup.bat` file. This will install all the packages listed in `requirements.txt`.

5.  **Deactivate the virtual environment (Windows):**
    Run the `005_deactivate.bat` file.

## Usage

1.  **Run the main application (Windows):**
    Run the `004_run.bat` file.

    [Provide instructions on how to use your application.]

## Batch Files (Windows)

This project includes the following batch files to help with common development tasks on Windows:

* `000_init.bat`: Initialized git and also usn and pwd config setup also done.
* `001_env.bat`: Creates a virtual environment named `venv`.
* `002_activate.bat`: Activates the `venv` virtual environment.
* `003_setup.bat`: Installs the Python packages listed in `requirements.txt` using `pip`.
* `004_run.bat`: Executes the main Python script (`main.py`).
* `005_run_test.bat`: Executes the pytest  scripts (`test_main.py`).
* `008_deactivate.bat`: Deactivates the currently active virtual environment.

## Contributing

[Explain how others can contribute to your project.]

## License

[Specify the project license, if any.]
