from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str = ""
    model: str = "claude-haiku-4-5"
    embed_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    rerank_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    index_dir: Path = Path("data/index")
    corpus_path: Path = Path("data/corpus.json")
    variants_path: Path = Path("data/variants.json")
    daily_token_budget: int = 200_000
    budget_db: Path = Path("data/budget.sqlite3")

    model_config = {"env_file": ".env"}


settings = Settings()
