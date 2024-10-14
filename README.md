# Arabic LLM Fine-tuning Project

This project demonstrates the process of fine-tuning a large language model (LLM) for Arabic instruction following, using the LoRA (Low-Rank Adaptation) technique. It includes scripts for fine-tuning and evaluating the model's performance.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Setup](#setup)
3. [Fine-tuning Process](#fine-tuning-process)
4. [Evaluation](#evaluation)


## Project Overview

This project aims to improve the performance of the Qwen1.5-7B model on Arabic language tasks, specifically instruction following. We use a dataset of six million Arabic instruction-response pairs and employ LoRA for efficient fine-tuning.

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/arabic-llm-finetuning.git
   cd arabic-llm-finetuning
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Fine-tuning Process

1. Review and adjust the hyperparameters at the top of the script if needed.

2. Run the fine-tuning script:
   ```
   python finetuning_llm.py
   ```


## Evaluation

To run the evaluation:

1. Ensure you have both the base model and fine-tuned model available (either locally or on the Hugging Face Hub).

2. Run the evaluation script:
   ```
   python evaluate_model.py
   ```

3. The script will:
   - Evaluate them on a subset of the dataset
   - Calculate metrics: Accuracy, F1 Score, and ROUGE-L F1


For more detailed information about the fine-tuning process, please refer to our [Medium article][https://eman-lotfy-elrefai.medium.com/how-to-fine-tune-llm-for-arabic-instructions-using-lora-9cf137b54e22]
