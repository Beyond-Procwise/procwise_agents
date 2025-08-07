# ProcWise/engines/prompt_engine.py

import ollama
import json
from config.settings import settings


class PromptEngine:
    def __init__(self):
        self.llm_model = settings.extraction_model

    def deconstruct_query(self, user_query: str, prompt_library: dict) -> dict | None:
        """
        Uses an LLM to parse a user's query into structured parameters based on a prompt library.
        This is the "what" part of the system.
        """
        print(f"PromptEngine: Deconstructing query -> '{user_query}'")

        prompt_library_context = json.dumps(prompt_library.get('templates', []), indent=2)

        prompt = f"""You are a query analysis expert. Your task is to deconstruct a user's natural language query into a structured JSON object based on a provided library of known query templates.

**Instructions:**
1.  Analyze the user's query.
2.  Infer values for `category`, `criteria`, `time_period`, and `filters`.
3.  If the query mentions ranking but no specific criteria are named (e.g., "rank the suppliers"), you MUST default the `criteria` to `["price", "delivery", "risk"]`.
4.  If no time period is mentioned, `time_period` must be `null`.
5.  Find the best matching template from the "PROMPT LIBRARY" below.
6.  Infer the values for the template's input parameters (like `category`, `criteria`, `time_period`, `filters`) from the user's query.
7.  Your response MUST be a single, valid JSON object containing the `template_id` and the extracted `parameters`. If a parameter cannot be inferred, use `null`.

**PROMPT LIBRARY:**
---
{prompt_library_context}
---

**USER QUERY:**
"{user_query}"

**JSON OUTPUT:**"""
        try:
            response = ollama.generate(
                model=self.llm_model,
                prompt=prompt,
                format='json',
                options={'temperature': 0.0}
            )
            parsed_query = json.loads(response.get('response'))
            print(f"PromptEngine: Query deconstructed successfully -> {parsed_query}")
            return parsed_query
        except Exception as e:
            print(f"ERROR (PromptEngine): Failed to deconstruct query. {e}")
            return None
