#!/usr/bin/env python3
"""Example script demonstrating basic usage of Red Flag Law AI."""

import sys
from pathlib import Path
from typing import Any, Dict

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import gradio as gr
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.detector import RedFlagDetector

app = FastAPI(title="Red Flag AI Inference Service (vLLM)", version="1.0")

# Global detector instance to be loaded on startup
detector = None


def _create_detector(cfg: DictConfig) -> RedFlagDetector:
    sampling_params = instantiate(cfg.sampling_params)
    return RedFlagDetector(model_path=cfg.model_path, sampling_params=sampling_params)

class AnalyzeRequest(BaseModel):
    text: str

class AnalyzeResponse(BaseModel):
    is_unfair: bool
    category: str
    explanation: str

@app.on_event("startup")
async def startup_event():
    global detector
    # Fallback for direct ASGI startup without Hydra config injection.
    if detector is None:
        raise RuntimeError("Detector is not initialized. Start API from Hydra main with api=true.")

@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def analyze_clause(request: AnalyzeRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    try:
        res = detector.analyze(request.text)
        return AnalyzeResponse(**res)
    except Exception as e:
        if isinstance(e, RuntimeError) and "not loaded" in str(e):
            return AnalyzeResponse(
                is_unfair=True, 
                category="Mock Mode / Unilateral Change", 
                explanation="API mocked successfully because model is not downloaded yet."
            )
        raise HTTPException(status_code=500, detail=str(e))

def _analyze_for_gradio(text: str) -> Dict[str, Any]:
    if detector is None:
        return {
            "is_unfair": False,
            "category": "Error",
            "explanation": "Detector is not initialized.",
        }
    if not text or not text.strip():
        return {
            "is_unfair": False,
            "category": "Validation",
            "explanation": "Text cannot be empty.",
        }
    return detector.analyze(text)


def launch_gradio(host: str, port: int):
    with gr.Blocks(title="Red Flag Law AI") as demo:
        gr.Markdown("# Red Flag Law AI")
        gr.Markdown("Analyze a legal clause and detect potentially unfair terms.")

        input_box = gr.Textbox(
            label="Legal Clause",
            lines=6,
            placeholder="Paste a clause from Terms of Service, Privacy Policy, or contract...",
        )
        submit_btn = gr.Button("Analyze", variant="primary")
        clear_btn = gr.Button("Clear")

        unfair_out = gr.Checkbox(label="Is Unfair")
        category_out = gr.Textbox(label="Category")
        explanation_out = gr.Textbox(label="Explanation", lines=5)
        json_out = gr.JSON(label="Raw Response")

        def _submit(text: str):
            result = _analyze_for_gradio(text)
            return (
                bool(result.get("is_unfair", False)),
                str(result.get("category", "")),
                str(result.get("explanation", "")),
                result,
            )

        submit_btn.click(
            _submit,
            inputs=[input_box],
            outputs=[unfair_out, category_out, explanation_out, json_out],
        )
        clear_btn.click(
            lambda: ("", False, "", "", {}),
            inputs=None,
            outputs=[input_box, unfair_out, category_out, explanation_out, json_out],
        )

    demo.launch(server_name=host, server_port=port, share=False)


@hydra.main(version_base=None, config_path="../configs", config_name="demo")
def main(cfg: DictConfig):
    global detector

    """Run example analysis or start API/Gradio service."""
    detector = _create_detector(cfg)

    if cfg.api:
        uvicorn.run(app, host=cfg.host, port=cfg.port, reload=False)
        return

    if cfg.gradio:
        launch_gradio(cfg.host, cfg.port)
        return

    clauses = [
        "We reserve the right to terminate your account at any time without notice or reason.",
        "You may cancel your subscription at any time through your account settings.",
        "The company is not liable for any damages arising from use of the service.",
        "We will notify you 30 days in advance of any changes to these terms.",
        "By using our service, you agree to binding arbitration and waive your right to sue.",
        "Your personal data may be shared with third parties for marketing purposes.",
        "We may modify or discontinue the service at any time without liability.",
        "You retain all rights to content you upload to our platform.",
        "By using our service, you agree to binding arbitration and waive your right to sue."
    ]
    
    print("=" * 80)
    print("Red Flag Law AI - Example Analysis")
    print("=" * 80)
    print()
    
    for i, clause in enumerate(clauses, 1):
        print(f"\nClause {i}: \"{clause}\"")
        result = detector.analyze(clause)
        print(f"  Is Unfair: {result['is_unfair']}")
        if result['category']:
            print(f"  Category: {result.get('category')}")
        print(f"  Explanation: {result.get('explanation')}")

if __name__ == "__main__":
    main()
