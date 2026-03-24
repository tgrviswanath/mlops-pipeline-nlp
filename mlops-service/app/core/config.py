from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MLFLOW_TRACKING_URI: str = "file:./mlruns"
    EXPERIMENT_NAME: str = "sentiment-analysis"
    MODEL_REGISTRY_NAME: str = "sentiment-model"
    RETRAIN_SCHEDULE_HOURS: int = 24
    DRIFT_THRESHOLD: float = 0.15
    MIN_ACCURACY_THRESHOLD: float = 0.75
    
    class Config:
        env_file = ".env"

settings = Settings()
