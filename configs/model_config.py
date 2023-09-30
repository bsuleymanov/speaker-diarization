from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING

@dataclass
class VADConfig:
    threshold: float = MISSING

@dataclass
class SDEPConfig:
    n_networks: int = MISSING
    n_prototypes: int = MISSING
    prototype_dim: int = MISSING
    temp_student: Optional[float] = None
    temp_teacher: Optional[float] = None

@dataclass
class ModelConfig:
    name: str = MISSING
    sdep: Optional[SDEPConfig] = None
    vad: Optional[VADConfig] = None