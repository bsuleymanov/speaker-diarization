from enum import Enum

from configs.dataset_config import DatasetConfig
from datasets.dataset import SDEPDataset
from utils import asdict_filter_none


class DatasetName(Enum):
    SDEP = "sdep"
    Attentive_VAD = "attentive_vad"


def create_dataset(dataset_config: DatasetConfig):
    name = DatasetName(dataset_config.name)
    if name == DatasetName.SDEP:
        model = SDEPDataset(
            **asdict_filter_none(dataset_config.sdep)
        )
    else:
        raise NotImplementedError(f"Model {name} not implemented")

    return model