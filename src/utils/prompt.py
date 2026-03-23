def format_prompt(text: str) -> str:
    return "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n" \
           "Analyze the legal clause and extract risk in JSON format containing: " \
           "categories (list or None), explanation (str)." \
           "<|eot_id|><|start_header_id|>user<|end_header_id|>\n" \
          f"Clause: {text}" \
           "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
