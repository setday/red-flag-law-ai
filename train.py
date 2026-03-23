#!/usr/bin/env python3
"""Two-phase training pipeline for Red Flag Law AI."""

import json
import re
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from peft import prepare_model_for_kbit_training
from transformers import DataCollatorForLanguageModeling
from trl import SFTTrainer
import torch

from src.utils.prompt import format_prompt


UNFAIR_CATEGORIES = ['Limitation of liability', 'Unilateral termination', 'Unilateral change', 'Content removal', 'Contract by using', 'Choice of law', 'Jurisdiction', 'Arbitration', 'None']


def _build_peft_config(cfg: DictConfig):
    peft_cfg = instantiate(cfg.peft_config, _convert_="all")
    if getattr(peft_cfg, "target_modules", None) is not None:
        peft_cfg.target_modules = list(peft_cfg.target_modules)
    if getattr(peft_cfg, "modules_to_save", None) is not None:
        peft_cfg.modules_to_save = list(peft_cfg.modules_to_save)
    if getattr(peft_cfg, "layers_to_transform", None) is not None:
        peft_cfg.layers_to_transform = list(peft_cfg.layers_to_transform)
    return peft_cfg


def _extract_categories(response: str):
    try:
        parsed = json.loads(response)
        categories = parsed.get("categories", "None")
    except json.JSONDecodeError:
        categories = "None"

    if isinstance(categories, str):
        cleaned = categories.strip()
        if cleaned.lower() == "none" or cleaned == "":
            return ["None"]
        # Handle serialized list inside a string like "['Arbitration']"
        list_match = re.findall(r"[A-Za-z\s]+", cleaned)
        normalized = [c.strip() for c in list_match if c.strip()]
        return normalized or [cleaned]

    if isinstance(categories, list):
        normalized = [str(c).strip() for c in categories if str(c).strip()]
        return normalized or ["None"]

    return ["None"]


def format_phase1_instruction(sample):
    text = sample["text"]
    labels = sample["labels"]

    if len(labels) > 0:
        categories = [UNFAIR_CATEGORIES[label] for label in labels]
        target = json.dumps(
            {"categories": categories, "explanation": ""},
            ensure_ascii=True,
        )
        # Keypoint: There is no explanation in phase 1
        # We suppose the model is smart enough to make a coherent explanation if we have correct labels + it learns to associate certain patterns with certain categories in attention layers
        prompt = format_prompt(text) + target[-2:]
    else:
        target = json.dumps(
            {
                "categories": ["None"],
                "explanation": "This clause is a standard, balanced legal term, without any bias.",
            },
            ensure_ascii=True,
        )
        prompt = format_prompt(text) + target
    return {"text": prompt}


def build_phase2_example(sample, tokenizer, model, max_new_tokens: int):
    clause = sample["premise"]
    reference_explanation = sample["hypothesis"]
    prompt = format_prompt(clause)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    inferred_categories = _extract_categories(response)

    target = json.dumps(
        {
            "categories": inferred_categories,
            "explanation": reference_explanation,
        },
        ensure_ascii=True,
    )
    return {"text": format_prompt(clause) + target}


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    print(f"Loading phase 1 dataset: {cfg.dataset.phase1.train.path}/{cfg.dataset.phase1.train.name}")
    phase1_dataset = instantiate(cfg.dataset.phase1.train)
    phase1_train = phase1_dataset.map(
        format_phase1_instruction,
        remove_columns=phase1_dataset.column_names,
    )

    print("Initializing QLoRA configs...")
    tokenizer = instantiate(cfg.model_pipeline.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token

    model = instantiate(cfg.model_pipeline.model, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print("Phase 1 training on unfair_tos labels...")
    phase1_trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=phase1_train,
        data_collator=data_collator,
        peft_config=_build_peft_config(cfg),
        args=instantiate(cfg.training.phase1)
    )

    phase1_trainer.train()

    print(f"Loading phase 2 dataset: {cfg.dataset.phase2.train.path}/{cfg.dataset.phase2.train.name}")
    phase2_dataset = instantiate(cfg.dataset.phase2.train)
    phase2_limit = int(cfg.dataset.phase2.max_samples)
    if phase2_limit > 0:
        phase2_dataset = phase2_dataset.select(range(min(phase2_limit, len(phase2_dataset))))

    print("Running phase 2 pseudo-label inference on contract-nli...")
    phase2_train = phase2_dataset.map(
        lambda sample: build_phase2_example(
            sample,
            tokenizer=tokenizer,
            model=phase1_trainer.model,
            max_new_tokens=int(cfg.phase2_generation.max_new_tokens),
        ),
        remove_columns=phase2_dataset.column_names,
    )

    print("Phase 2 training on inferred-label + reference-explanation data...")
    phase2_trainer = SFTTrainer(
        model=phase1_trainer.model,
        processing_class=tokenizer,
        train_dataset=phase2_train,
        data_collator=data_collator,
        peft_config=_build_peft_config(cfg),
        args=instantiate(cfg.training.phase2)
    )

    phase2_trainer.train()
    phase2_trainer.model.save_pretrained(cfg.training.phase2.output_dir)
    print("Two-phase training finished!")

if __name__ == "__main__":
    main()
