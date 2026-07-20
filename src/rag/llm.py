"""LLM client — one interface, any OpenAI-compatible provider.

Why it exists
-------------
Ollama (local), Groq (deployed), and OpenAI all speak the same wire protocol —
the OpenAI chat-completions API. So one client class, pointed at a different
`base_url` + key via config, covers all of them. Swapping providers is an
environment-variable change, not a code change. This is the design decision
that lets development be free/offline (Ollama qwen2.5:7b) while the deployed
demo runs a free 70B model on Groq — with byte-identical code.

Two call styles:
  chat()       plain text — used for answer generation
  json_chat()  schema-constrained JSON — used for judge verdicts and anything
               that must be machine-parseable. Small models drift out of
               format when merely ASKED for JSON; constraining decoding with
               a schema (response_format) is what keeps 7B-model judges usable.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from openai import OpenAI

if TYPE_CHECKING:
    from rag.config import Settings


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = OpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,  # Ollama ignores it, but the SDK requires one
            timeout=120.0,  # local 7B models on CPU can be slow; don't give up early
        )

    def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Plain text completion.

        temperature=0 everywhere: retrieval-grounded answers should be
        REPEATABLE — same context in, same answer out — or eval numbers
        become noise and bug reports become unreproducible.
        """
        response = self._client.chat.completions.create(
            model=model or self._settings.llm_model,
            messages=messages,
            temperature=0,
            max_tokens=max_tokens or self._settings.answer_max_tokens,
        )
        return response.choices[0].message.content or ""

    def json_chat(
        self,
        messages: list[dict],
        schema: dict,
        *,
        model: str | None = None,
    ) -> dict:
        """Structured output: the model is CONSTRAINED to emit JSON matching
        `schema` via response_format.

        Provider caveat: `json_schema` is NOT universally supported behind
        "OpenAI-compatible" endpoints. Ollama accepts it on any model; Groq
        only on its structured-outputs list (e.g. openai/gpt-oss-20b) and
        returns 400 on the llama-3.x models. Judge models must be picked from
        that list — see JUDGE_MODEL in railway.toml.

        Failure handling: parse with json.loads; on any failure retry ONCE
        with the error appended to the conversation (models usually fix their
        own malformed JSON when shown the parse error), then raise. Callers
        that can degrade gracefully catch the ValueError.
        """
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": schema, "strict": True},
        }
        attempt_messages = list(messages)
        last_error: Exception | None = None
        for _ in range(2):  # first try + one repair retry
            response = self._client.chat.completions.create(
                model=model or self._settings.llm_model,
                messages=attempt_messages,
                temperature=0,
                response_format=response_format,
            )
            raw = response.choices[0].message.content or ""
            try:
                return json.loads(raw)
            except json.JSONDecodeError as e:
                last_error = e
                # Show the model its own broken output + the parse error and
                # let it try again — cheap, and fixes the common truncation case.
                attempt_messages = list(messages) + [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": f"That was not valid JSON ({e}). "
                        "Respond again with ONLY the corrected JSON object.",
                    },
                ]
        raise ValueError(f"LLM returned invalid JSON after retry: {last_error}")

    def judge_chat(self, messages: list[dict], schema: dict) -> dict:
        """json_chat pinned to the judge model.

        A separate entry point so verification/eval callers can't forget to
        use settings.judge_model — locally it's the same model that wrote the
        answer (self-preference bias, documented in verify.py); deployed, the
        judge is configured to a DIFFERENT model to break that bias.
        """
        return self.json_chat(messages, schema, model=self._settings.judge_model)
