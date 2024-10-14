import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from rouge import Rouge

def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def evaluate_model(model, tokenizer, dataset, num_samples=100):
    rouge = Rouge()
    accuracies = []
    f1_scores = []
    rouge_scores = []

    for sample in tqdm(dataset.select(range(num_samples))):
        prompt = sample['instruction']
        true_response = sample['output']
        generated_response = generate_response(model, tokenizer, prompt)

        # Simple accuracy (exact match)
        accuracy = int(generated_response.strip() == true_response.strip())
        accuracies.append(accuracy)

        # F1 score (word-level)
        pred_words = set(generated_response.lower().split())
        true_words = set(true_response.lower().split())
        f1 = f1_score(
            [1] * len(true_words) + [0] * len(pred_words - true_words),
            [1] * len(true_words & pred_words) + [0] * len(pred_words - true_words),
            average='binary'
        )
        f1_scores.append(f1)

        # ROUGE score
        rouge_score = rouge.get_scores(generated_response, true_response)[0]
        rouge_scores.append(rouge_score['rouge-l']['f'])

    return {
        'accuracy': np.mean(accuracies),
        'f1_score': np.mean(f1_scores),
        'rouge_l_f': np.mean(rouge_scores)
    }

def main():
    # Load dataset
    dataset = load_dataset("akbargherbal/six_millions_instruction_dataset_for_arabic_llm_ft", split="train")

    # Evaluate base model
    base_model_name = "Alibaba-NLP/gte-Qwen1.5-7B-instruct"
    base_model, base_tokenizer = load_model_and_tokenizer(base_model_name)
    print("Evaluating base model...")
    base_results = evaluate_model(base_model, base_tokenizer, dataset)

    # Evaluate fine-tuned model
    finetuned_model_name = "Eman52/Qwen_Finetuned_Arabic_Instruction_Set"
    finetuned_model, finetuned_tokenizer = load_model_and_tokenizer(finetuned_model_name)
    print("Evaluating fine-tuned model...")
    finetuned_results = evaluate_model(finetuned_model, finetuned_tokenizer, dataset)

    # Print results
    print("\nBase Model Results:")
    print(base_results)
    print("\nFine-tuned Model Results:")
    print(finetuned_results)

    # Calculate improvements
    improvements = {
        k: finetuned_results[k] - base_results[k]
        for k in base_results.keys()
    }
    print("\nImprovements:")
    print(improvements)

if __name__ == "__main__":
    main()