#!/usr/bin/env python3
"""Script to train the Red Flag Law AI model using QLoRA."""

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from peft import prepare_model_for_kbit_training
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

@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    print(f"Loading {cfg.dataset.train.name} Dataset...")
    dataset = instantiate(cfg.dataset.train)
    train_data = dataset.map(format_instruction)
    
    print("Initializing QLoRA configs...")
    tokenizer = instantiate(cfg.model_pipeline.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = instantiate(cfg.model_pipeline.model, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        dataset_text_field="text",
        max_seq_length=cfg.model_pipeline.model.max_seq_length,
        peft_config=instantiate(cfg.peft_config),
        args=instantiate(cfg.training)
    )
    
    print("Starting Training...")
    trainer.train()
    trainer.model.save_pretrained(cfg.training.output_dir)
    print("Training finished!")

if __name__ == "__main__":
    main()
