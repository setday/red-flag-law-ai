#!/usr/bin/env python3
"""Script to train the Red Flag Law AI model using QLoRA."""

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from peft import prepare_model_for_kbit_training
from transformers import DataCollatorForLanguageModeling
from trl import SFTTrainer

from src.utils.prompt import format_prompt


UNFAIR_CATEGORIES = ['Limitation of liability', 'Unilateral termination', 'Unilateral change', 'Content removal', 'Contract by using', 'Choice of law', 'Jurisdiction', 'Arbitration', 'None']


def format_instruction(sample):
    text = sample["text"]
    labels = sample["labels"]

    if len(labels) > 0:
        categories = [UNFAIR_CATEGORIES[label] for label in labels]
        # Key moment: There is no explanation; we suppose the model is smart make a coherent explanation if we have correct labels + it learns to associate certain patterns with certain categories in attention.
        prompt = format_prompt(text) + '{' + f'"categories": "{categories}", "explanation": "'
    else:
        prompt = format_prompt(text) + '{"categories": "None", "explanation": "This clause is a standard, balanced legal term, without any bias."}'
    return {"text": prompt}

@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    print(f"Loading {cfg.dataset.train.name} Dataset...")
    dataset = instantiate(cfg.dataset.train)
    train_data = dataset.map(
        format_instruction,
        remove_columns=dataset.column_names,
    )
    
    print("Initializing QLoRA configs...")
    tokenizer = instantiate(cfg.model_pipeline.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = instantiate(cfg.model_pipeline.model, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    peft_cfg = instantiate(cfg.peft_config, _convert_="all")
    if getattr(peft_cfg, "target_modules", None) is not None:
        peft_cfg.target_modules = list(peft_cfg.target_modules)
    if getattr(peft_cfg, "modules_to_save", None) is not None:
        peft_cfg.modules_to_save = list(peft_cfg.modules_to_save)
    if getattr(peft_cfg, "layers_to_transform", None) is not None:
        peft_cfg.layers_to_transform = list(peft_cfg.layers_to_transform)
    
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_data,
        data_collator=data_collator,
        peft_config=peft_cfg,
        args=instantiate(cfg.training)
    )
    
    print("Starting Training...")
    trainer.train()
    trainer.model.save_pretrained(cfg.training.output_dir)
    print("Training finished!")

if __name__ == "__main__":
    main()
