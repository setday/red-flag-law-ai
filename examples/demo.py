#!/usr/bin/env python3
"""Example script demonstrating basic usage of Red Flag Law AI."""

import sys
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.detector import RedFlagDetector

app = FastAPI(title="Red Flag AI Inference Service (vLLM)", version="1.0")

# Global detector instance to be loaded on startup
detector = None

class AnalyzeRequest(BaseModel):
    text: str

class AnalyzeResponse(BaseModel):
    is_unfair: bool
    category: str
    explanation: str

@app.on_event("startup")
async def startup_event():
    global detector
    # Ideally, we inject config here, but for module-level FastAPI, we use fallback
    model_path = "checkpoints/redflag-llama3"
    print(f"Starting up vLLM with Model: {model_path} ...")
    detector = RedFlagDetector()
    try:
        # Commented so it doesn't crash if model isn't downloaded yet.
        # detector.load_model(model_path) 
        pass
    except Exception as e:
        print("Model loading failed, fallback to mock mode.")

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

@hydra.main(version_base=None, config_path="../configs", config_name="demo")
def main(cfg: DictConfig):
    """Run example analysis on sample legal clauses or start API server."""
    if cfg.api:
        uvicorn.run("examples.demo:app", host=cfg.host, port=cfg.port, reload=False)
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
    
    local_detector = RedFlagDetector()
    try:
        local_detector.load_model(cfg.model_path)
    except Exception as e:
        print(f"Skipping actual vLLM load due to error/missing model check: {e}")
    
    for i, clause in enumerate(clauses, 1):
        print(f"\nClause {i}: \"{clause}\"")
        result = local_detector.analyze(clause) if local_detector.llm else {
            "is_unfair": True, "category": "Demo mock", "explanation": "Load real model via load_model()"
        }
        print(f"  Is Unfair: {result['is_unfair']}")
        if result['category']:
            print(f"  Category: {result.get('category')}")
        print(f"  Explanation: {result.get('explanation')}")

if __name__ == "__main__":
    main()
