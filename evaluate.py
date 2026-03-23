#!/usr/bin/env python3
"""Script to evaluate a trained model using Hydra configuration."""

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import torch

from src.utils.metrics import MetricsCalculator
from src.utils.prompt import format_prompt

def format_instruction(sample):
    text = sample["text"]
    return {
        "prompt": format_prompt(text),
        "text": text,
        "label": len(sample["labels"]) > 0 and sample["labels"][0] != 0,
        "reference_explanation": sample.get("explanation", "")
    }

@hydra.main(version_base=None, config_path="../configs", config_name="evaluate")
def main(cfg: DictConfig):
    print(f"Evaluating model: {cfg.model_path} on {cfg.dataset.test.path}/{cfg.dataset.test.name})")
    
    # Load evaluation dataset
    dataset = instantiate(cfg.dataset.test)
    eval_data = dataset.map(format_instruction).select(range(cfg.dataset.test.max_samples)) # samples for speed
    
    # Load model and tokenizer
    tokenizer = instantiate(cfg.model_pipeline.tokenizer)
    model = instantiate(cfg.model_pipeline.model, device_map="auto", torch_dtype=torch.float16)
    
    y_true = []
    y_pred = []
    explanations_pred = []
    explanations_ref = []
    
    print("Running inference...")
    for item in eval_data:
        inputs = tokenizer(item["prompt"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        # simplistic parsing
        is_unfair_pred = "true" in response.lower()
        y_true.append(item["label"])
        y_pred.append(is_unfair_pred)
        
        # Collect explanations if available
        explanations_pred.append(response)
        if item["reference_explanation"]:
            explanations_ref.append(item["reference_explanation"])
    
    # Compute classification metrics
    print("\nEvaluation Results:")
    print("="*50)
    
    metrics = MetricsCalculator.compute_classification_metrics(y_pred, y_true)
    print("\nClassification Metrics:")
    print(f"  Accuracy:       {metrics['accuracy']:.4f}")
    print(f"  F1 (Macro):     {metrics['f1_macro']:.4f}")
    print(f"  Precision:      {metrics['precision_macro']:.4f}")
    print(f"  Recall:         {metrics['recall_macro']:.4f}")
    
    # Compute BERTScore for explanation quality if references available
    if explanations_ref:
        print("\nComputing BERTScore for explanation quality...")
        bertscore_metrics = MetricsCalculator.compute_bertscore(
            explanations_pred[:len(explanations_ref)],
            explanations_ref,
            model_name="microsoft/deberta-xlarge-mnli"
        )
        print("\nExplanation Quality (BERTScore):")
        print(f"  Precision:      {bertscore_metrics['bertscore_precision']:.4f}")
        print(f"  Recall:         {bertscore_metrics['bertscore_recall']:.4f}")
        print(f"  F1:             {bertscore_metrics['bertscore_f1']:.4f}")
    else:
        print("\nNote: Reference explanations not available for BERTScore evaluation.")
    
    print("\n" + "="*50)
    print("End of Evaluation.")

if __name__ == "__main__":
    main()
