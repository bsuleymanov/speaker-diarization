from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING

@dataclass
class VADConfig:
    threshold: float = MISSING

@dataclass
class SDEPConfig:
    data: str = MISSING
    noise: str = MISSING
    n_global_views: int = MISSING
    n_local_views: int = MISSING
    reverb: str = MISSING
    max_frames: int = MISSING
    n_mels: int = MISSING

@dataclass
class DatasetConfig:
    name: str = MISSING
    sdep: Optional[SDEPConfig] = None
    vad: Optional[VADConfig] = None