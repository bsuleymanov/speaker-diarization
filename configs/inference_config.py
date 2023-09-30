from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING

from configs.model_config import ModelConfig
from configs.dataset_config import DatasetConfig


@dataclass
class InferenceConfig:
    batch_size: int = MISSING
    model: ModelConfig = ModelConfig()
    dataset: DatasetConfig = DatasetConfig()