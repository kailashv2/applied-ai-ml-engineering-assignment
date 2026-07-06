from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    vector_db_uri: str = './.lancedb'
    vector_table: str = 'chunks'
    embedding_model: str = 'local-hashing-384'
    default_k: int = 5
    min_relevance_score: float = 0.35
    generator_provider: str = 'mock'
    generator_model: str = 'gpt-4o-mini'
    openai_api_key: str | None = None


settings = Settings()
