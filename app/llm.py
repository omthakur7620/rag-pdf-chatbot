
# app/llm.py

import os
from typing import List
from groq import Groq


class GroqLLM:
    """
    Groq LLM wrapper with balanced grounding:
    - Uses only provided context
    - Allows summarization and paraphrasing
    - Avoids false 'not found' answers
    """

    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.0
    ):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model_name = model_name
        self.temperature = temperature

    def generate_answer(self, query: str, contexts: List[str]) -> str:
        """
        Generate an answer grounded strictly in retrieved context.
        """

        if not contexts:
            return "Answer not found in the provided document."

        context_text = "\n\n".join(
            [f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)]
        )

        system_prompt = (
            "You are a document-grounded question answering assistant.\n"
            "Answer the question using ONLY the provided context.\n"
            "You may summarize or paraphrase the context if the meaning is clearly present.\n"
            "DO NOT use any external knowledge.\n"
            "If the answer truly cannot be derived from the context, reply exactly with:\n"
            "'Answer not found in the provided document.'"
        )

        user_prompt = f"""
        Context:
        {context_text}

        Question:
        {query}

        Answer:
        """

        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        return response.choices[0].message.content.strip()

      
