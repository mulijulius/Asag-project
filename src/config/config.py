from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    RANDOM_SEED: int = 42
    MODEL: str = "default_model"
    MC_DROPOUT: bool = False
    TRAINING: bool = True
    CALIBRATION: bool = False
    DEFERRAL: bool = False
    DATASETS: dict = None
    PATHS: dict = None

    @classmethod
    def create_output_directories(cls):
        paths = cls.PATHS if cls.PATHS else {}
        for directory in paths.values():
            Path(directory).mkdir(parents=True, exist_ok=True)

# Singleton instance
CFG = Config()