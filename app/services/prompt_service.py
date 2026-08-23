from app.models.response import RetrievalResult

class PromptService:
    @staticmethod
    def build_messages(question: str, chunks: list[RetrievalResult], chat_history: list[dict[str, str]] = None) -> list[dict[str, str]]:
        if chat_history is None:
            chat_history = []
            
        system_prompt = (
            "You are an assistant answering questions about the provided webpage.\n"
            "Return ONLY the final answer. Never output internal reasoning, <think> blocks, or chain-of-thought.\n"
            "Never describe the retrieval process or how the answer was generated.\n"
            "Make answers concise and natural. Prefer 1-3 concise paragraphs or a short bullet list.\n"
            "Avoid repetitive phrases such as 'The provided text...', 'The provided webpage...', 'Based on the provided context...', or 'According to the provided context...'.\n"
            "Answer using ONLY the supplied webpage context.\n"
            "If the context does not support the answer, clearly say the information was not found.\n"
            "Do not use outside knowledge. Do not invent facts.\n"
            "\n"
            "Webpage Context:\n"
            "-------------------\n"
        )

        for chunk in chunks:
            system_prompt += f"[Source Chunk: {chunk.title}]\n{chunk.content}\n\n"
            
        messages = [{"role": "system", "content": system_prompt}]
        
        for msg in chat_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
            
        messages.append({"role": "user", "content": question})
        return messages