from pathlib import Path

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Models
    embed_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    router_model: str = "gpt-4o-mini"
    vector_size: int = 1536

    # Tunables
    top_k: int = 5
    score_threshold: float = 0.30
    chunk_words: int = 250
    collection: str = "finance_kb"

    # Data root. Defaults to <repo>/data resolved from CWD, or /app/data in the container.
    data_dir: str = Field(default="")

    # Secrets (from env)
    openai_api_key: str = ""
    slack_bot_token: str = ""
    slack_app_token: str = ""

    idk_message: str = "I don't know — I couldn't find that in my knowledge base."
    system_prompt: str = (
        "You are a strategy-investment expert. Answer using ONLY the information in "
        "the context below — never outside knowledge. You may synthesize, compare, "
        "and summarize across the provided sources to answer. For a comparison, "
        "cover each item the context supports; if the context covers some items but "
        "not others, answer for those it does cover and explicitly say which it "
        'lacks. Only if the context is essentially irrelevant to the question, reply '
        'exactly "I don\'t know." Cite the source title for each claim.'
    )

    def _resolve_data_dir(self) -> Path:
        if self.data_dir:
            return Path(self.data_dir)
        # repo layout: prefer ./data, fall back to ../data (e.g. running from notebooks/)
        for cand in (Path("data"), Path("../data")):
            if (cand / "corpus").exists():
                return cand
        return Path("data")

    @computed_field
    @property
    def corpus_dir(self) -> Path:
        return self._resolve_data_dir() / "corpus"

    @computed_field
    @property
    def qdrant_dir(self) -> Path:
        return self._resolve_data_dir() / "qdrant"

    @computed_field
    @property
    def metrics_path(self) -> Path:
        return self._resolve_data_dir() / "metrics" / "metrics.jsonl"


settings = Settings()
