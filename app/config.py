from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str = ""
    model: str = "claude-haiku-4-5"
    embed_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    # Finetune auf Käufer/Verkäufer-Rollenpaaren (siehe README), privates
    # HF-Hub-Repo -- braucht HF_TOKEN als Env-Var zum Laden, siehe Deployment.
    rerank_model: str = "VoidFloat/chrono24-faq-reranker"
    index_dir: Path = Path("data/index")
    corpus_path: Path = Path("data/corpus.json")
    variants_path: Path = Path("data/variants.json")
    daily_token_budget: int = 200_000
    budget_db: Path = Path("data/budget.sqlite3")

    model_config = {"env_file": ".env"}


settings = Settings()
