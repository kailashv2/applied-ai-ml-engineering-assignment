from __future__ import annotations

import os

try:
    from openai import OpenAI
except Exception:  # optional dependency for hosted mode
    OpenAI = None


def estimate_tokens(text: str) -> int:
    return max(1, len(text.split()) * 4 // 3)


def build_prompt(question: str, contexts: list[dict]) -> str:
    context_block = '\n\n'.join(
        f"[chunk:{c['id']} source:{c.get('source_name','unknown')}]\n{c['text']}" for c in contexts
    )
    return f"""You are a retrieval-grounded QA assistant.
Answer only from the provided context. Cite every factual sentence with chunk IDs like [chunk:abc123].
If the context is insufficient, say exactly: I don't have enough relevant context to answer that.

Question: {question}

Context:
{context_block}

Answer:"""


def generate_answer(question: str, contexts: list[dict], provider: str, model: str) -> tuple[str, dict]:
    prompt = build_prompt(question, contexts)
    usage = {'prompt_tokens_est': estimate_tokens(prompt), 'completion_tokens_est': 0, 'model': model, 'provider': provider}

    if provider == 'openai' and os.getenv('OPENAI_API_KEY') and OpenAI is not None:
        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.1,
        )
        text = response.choices[0].message.content or ''
        if response.usage:
            usage.update({
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens,
            })
        return text.strip(), usage

    # Local smoke-test fallback: intentionally extractive and citation-preserving.
    bullets = []
    for c in contexts[:3]:
        first_sentence = c['text'].split('. ')[0][:280]
        bullets.append(f"- {first_sentence}. [chunk:{c['id']}]")
    answer = 'Mock grounded answer from retrieved context:\n' + '\n'.join(bullets)
    usage['completion_tokens_est'] = estimate_tokens(answer)
    return answer, usage
