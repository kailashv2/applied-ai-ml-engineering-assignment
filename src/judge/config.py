from pydantic_settings import BaseSettings, SettingsConfigDict


class JudgeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')
    judge_provider: str = 'mock'
    judge_model: str = 'claude-3-5-sonnet-latest'
    generator_provider: str = 'mock'
    generator_model: str = 'gpt-4o-mini'
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    judge_temperature: float = 0.0


settings = JudgeSettings()
