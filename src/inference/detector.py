"""Red Flag Detector for inference (vLLM Backend)."""

from typing import Dict, List
import json
import re

from vllm import LLM

from src.utils.prompt import format_prompt

class RedFlagDetector:
    """Detector for identifying unfair clauses in legal documents."""

    def __init__(self, model_path, sampling_params):
        self.model_path = model_path
        self.sampling_params = sampling_params
        
        print(f"Loading vLLM model from {model_path}...")
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.85,
            dtype="bfloat16"
        )

    def analyze(self, text: str) -> Dict:
        return self.batch_analyze([text])[0]

    def batch_analyze(self, texts: List[str]) -> List[Dict]:
        prompts = [format_prompt(t) for t in texts]
        # vLLM automatically handles continuous batching
        outputs = self.llm.generate(prompts, self.sampling_params)
        
        results = []
        for out in outputs:
            text = out.outputs[0].text.strip()
            try:
                results.append(json.loads(text))
            except json.JSONDecodeError:
                results.append({"is_unfair": False, "category": "Error", "explanation": "Parse error"})
        return results

    def analyze_document(self, document: str) -> Dict:
        # Basic chunking by newlines or sentence boundaries for the sliding window
        chunks = [c.strip() for c in re.split(r'\n{2,}|\.\s+(?=[A-Z])', document) if len(c.strip()) > 20]
        results = self.batch_analyze(chunks)
        
        flagged_clauses = zip(chunks, results)
        flagged_clauses = [{"text": c, "analysis": r} for c, r in flagged_clauses if r.get("is_unfair", False)]
        
        return {
            "total_clauses_analyzed": len(chunks),
            "flagged_count": len(flagged_clauses),
            "flagged_clauses": flagged_clauses,
        }
