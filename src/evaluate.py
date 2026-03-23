#!/usr/bin/env python3
"""Script to evaluate a trained model using Hydra configuration."""

import json
import re

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import torch

from src.utils.metrics import MetricsCalculator
from src.utils.prompt import format_prompt


def _parse_judge_response(response: str) -> dict:
    try:
        data = json.loads(response)
        score = data.get("score")
        reason = data.get("reason", "")
        return {"score": str(score), "reason": str(reason)}
    except json.JSONDecodeError:
        pass

    match = re.search(r"\b([1-5])\b", response)
    if match:
        return {"score": match.group(1), "reason": response.strip()}
    return {"score": "0", "reason": response.strip()}


def _judge_explanation(
    tokenizer,
    model,
    clause: str,
    predicted_explanation: str,
    reference_explanation: str,
    max_new_tokens: int,
) -> dict:
    judge_prompt = (
        "You are a strict legal NLP evaluator. Compare a model explanation against a reference explanation. "
        "Return JSON only with fields: score (integer 1-5), reason (short string). "
        "Scoring rubric: 5=excellent semantic match and legal correctness, 4=good with minor omissions, "
        "3=partially correct, 2=weak/incorrect, 1=wrong or irrelevant.\n\n"
        f"Clause:\n{clause}\n\n"
        f"Reference explanation:\n{reference_explanation}\n\n"
        f"Predicted explanation:\n{predicted_explanation}\n"
    )

    inputs = tokenizer(judge_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return _parse_judge_response(response)

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
    clauses = []
    explanations_pred = []
    explanations_ref = []
    judge_clauses_all = []
    judge_predictions_all = []
    judge_references_all = []
    
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
        clauses.append(item["text"])
        
        # Collect explanations if available
        explanations_pred.append(response)
        if item["reference_explanation"]:
            explanations_ref.append(item["reference_explanation"])
            judge_clauses_all.append(item["text"])
            judge_predictions_all.append(response)
            judge_references_all.append(item["reference_explanation"])
    
    # Compute classification metrics
    print("\nEvaluation Results:")
    print("="*50)
    
    metrics = MetricsCalculator.compute_classification_metrics(y_pred, y_true)
    print("\nClassification Metrics:")
    print(f"  Accuracy:       {metrics['accuracy']:.4f}")
    print(f"  F1 (Macro):     {metrics['f1_macro']:.4f}")
    print(f"  Precision:      {metrics['precision_macro']:.4f}")
    print(f"  Recall:         {metrics['recall_macro']:.4f}")

    use_llm_judge = "llm_judge" in cfg.metrics
    if use_llm_judge and judge_references_all:
        judge_limit = min(len(judge_references_all), int(cfg.llm_judge_max_samples))
        print(f"\nComputing LLM-as-a-Judge on {judge_limit} samples...")

        judge_clauses = judge_clauses_all[:judge_limit]
        judge_predictions = judge_predictions_all[:judge_limit]
        judge_references = judge_references_all[:judge_limit]

        def _judge_fn(clause: str, pred: str, ref: str) -> dict:
            return _judge_explanation(
                tokenizer=tokenizer,
                model=model,
                clause=clause,
                predicted_explanation=pred,
                reference_explanation=ref,
                max_new_tokens=int(cfg.llm_judge_max_new_tokens),
            )

        judge_metrics = MetricsCalculator.compute_llm_judge_metrics(
            clauses=judge_clauses,
            predictions=judge_predictions,
            references=judge_references,
            judge_fn=_judge_fn,
        )

        print("\nLLM-as-a-Judge Metrics:")
        print(f"  Mean Score (1-5): {judge_metrics['llm_judge_mean']:.4f}")
        print(f"  Pass@4 Rate:      {judge_metrics['llm_judge_pass_rate']:.4f}")
        print(f"  Scored Samples:   {int(judge_metrics['llm_judge_count'])}")
    elif use_llm_judge:
        print("\nNote: LLM-as-a-judge skipped because reference explanations are unavailable.")
    
    # Compute BERTScore for explanation quality if references available
    # if explanations_ref:
    #     print("\nComputing BERTScore for explanation quality...")
    #     bertscore_metrics = MetricsCalculator.compute_bertscore(
    #         explanations_pred[:len(explanations_ref)],
    #         explanations_ref,
    #         model_name="microsoft/deberta-xlarge-mnli"
    #     )
    #     print("\nExplanation Quality (BERTScore):")
    #     print(f"  Precision:      {bertscore_metrics['bertscore_precision']:.4f}")
    #     print(f"  Recall:         {bertscore_metrics['bertscore_recall']:.4f}")
    #     print(f"  F1:             {bertscore_metrics['bertscore_f1']:.4f}")
    # else:
    #     print("\nNote: Reference explanations not available for BERTScore evaluation.")
    
    print("\n" + "="*50)
    print("End of Evaluation.")

if __name__ == "__main__":
    main()
