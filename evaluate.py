#!/usr/bin/env python3
"""Script to evaluate a trained model using Hydra configuration."""

import json
import re
from typing import Dict

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import torch

from src.utils.metrics import MetricsCalculator
from src.utils.prompt import format_prompt


def _extract_json(text: str) -> Dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return {}


def _extract_prediction_fields(response: str) -> Dict[str, str]:
    data = _extract_json(response)
    categories = data.get("categories", "None") if data else "None"
    explanation = data.get("explanation", "") if data else response.strip()

    if not isinstance(categories, list):
        categories = str(categories).strip().split(',')
    categories = [c.strip().lower() for c in categories if c.strip()]

    return {
        "categories": categories,
        "explanation": str(explanation).strip(),
    }


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


def format_label_item(sample):
    text = sample["text"]
    return {
        "prompt": format_prompt(text),
        "text": text,
        "labels": sample["labels"],
    }


def format_explanation_item(sample):
    clause = sample["premise"]
    reference_explanation = sample["hypothesis"]
    return {
        "prompt": format_prompt(clause),
        "clause": clause,
        "reference_explanation": reference_explanation,
    }

@hydra.main(version_base=None, config_path="configs", config_name="evaluate")
def main(cfg: DictConfig):
    print(f"Evaluating model: {cfg.model_pipeline.model.pretrained_model_name_or_path}")

    labels_ds = instantiate(cfg.dataset.labels_eval)
    labels_max = int(cfg.dataset.labels_eval.max_samples)
    labels_eval_data = labels_ds.map(format_label_item)
    if labels_max > 0:
        labels_eval_data = labels_eval_data.select(range(min(labels_max, len(labels_eval_data))))

    expl_ds = instantiate(cfg.dataset.explanations_eval)
    expl_max = int(cfg.dataset.explanations_eval.max_samples)
    explanations_eval_data = expl_ds.map(format_explanation_item)
    if expl_max > 0:
        explanations_eval_data = explanations_eval_data.select(range(min(expl_max, len(explanations_eval_data))))

    tokenizer = instantiate(cfg.model_pipeline.tokenizer)
    model = instantiate(cfg.model_pipeline.model, device_map="auto", torch_dtype=torch.float16)

    y_true = []
    y_pred = []
    explanations_pred = []
    explanations_ref = []

    print("Running label evaluation on unfair_tos...")
    for item in labels_eval_data:
        inputs = tokenizer(item["prompt"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=96, pad_token_id=tokenizer.eos_token_id)

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        pred_fields = _extract_prediction_fields(response)
        categories = pred_fields["categories"].split(',') if pred_fields["categories"] != "None" else []
        y_true.append(item["labels"])
        y_pred.append(categories)

    print("\nEvaluation Results:")
    print("="*50)

    metrics = MetricsCalculator.compute_multiclass_metrics(y_pred, y_true)
    print("\nLabel Classification Metrics (unfair_tos):")
    print(f"  Accuracy:       {metrics['accuracy']:.4f}")
    print(f"  F1 (Macro):     {metrics['f1_macro']:.4f}")
    print(f"  Precision:      {metrics['precision_macro']:.4f}")
    print(f"  Recall:         {metrics['recall_macro']:.4f}")

    judge_clauses_all = []
    judge_predictions_all = []
    judge_references_all = []

    print("\nRunning explanation evaluation on contract-nli...")
    for item in explanations_eval_data:
        inputs = tokenizer(item["prompt"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=96, pad_token_id=tokenizer.eos_token_id)

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        pred_fields = _extract_prediction_fields(response)

        explanations_pred.append(pred_fields["explanation"])
        explanations_ref.append(item["reference_explanation"])
        judge_clauses_all.append(item["clause"])
        judge_predictions_all.append(pred_fields["explanation"])
        judge_references_all.append(item["reference_explanation"])

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
        print(f"  Pass Rate:      {judge_metrics['llm_judge_pass_rate']:.4f}")
    elif use_llm_judge:
        print("\nNote: LLM-as-a-judge skipped because reference explanations are unavailable.")

    if explanations_ref:
        print("\nComputing BERTScore for explanation quality...")
        bertscore_metrics = MetricsCalculator.compute_bertscore(
            explanations_pred,
            explanations_ref,
            model_name=cfg.bertscore_model
        )
        print("\nExplanation Quality Metrics (contract-nli):")
        print(f"  Precision:      {bertscore_metrics['bertscore_precision']:.4f}")
        print(f"  Recall:         {bertscore_metrics['bertscore_recall']:.4f}")
        print(f"  F1:             {bertscore_metrics['bertscore_f1']:.4f}")
    else:
        print("\nNote: No explanation references available for BERTScore evaluation.")

    print("\n" + "="*50)
    print("End of Evaluation.")

if __name__ == "__main__":
    main()
