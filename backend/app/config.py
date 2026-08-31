from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Server settings
    port: int = 8000
    host: str = "0.0.0.0"
    environment: str = "development"
    frontend_url: str = "http://localhost:3000"

    # Model configuration
    model_repo_id: str = "JayF14/varuna-HybridTransU-net"
    model_filename: str = "best_model.pth"
    use_gpu: bool = True

    # Alert Thresholds
    alert_threshold_watch: float = 0.3
    alert_threshold_warning: float = 0.5
    alert_threshold_emergency: float = 0.7

    class Config:
        env_file = ".env"

settings = Settings()
