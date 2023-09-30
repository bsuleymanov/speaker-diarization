from enum import Enum

from configs.model_config import ModelConfig
from models.models import SDEPModel, VADModel
from utils import asdict_filter_none


class ModelName(Enum):
    SDEP = "sdep"
    Attentive_VAD = "attentive_vad"


def create_model(model_config: ModelConfig):
    name = ModelName(model_config.name)
    if name == ModelName.SDEP:
        # model = SDEPModel(
        #     model_config.sdep.n_networks,
        #     model_config.sdep.n_prototypes,
        #     model_config.sdep.prototype_dim,
        #     model_config.sdep.temp_student,
        #     model_config.sdep.temp_teacher,
        # )
        model = SDEPModel(
            **asdict_filter_none(model_config.sdep)
        )
    elif name == ModelName.Attentive_VAD:
        model = VADModel(
            **asdict_filter_none(model_config.sdep)
        )
    else:
        raise NotImplementedError(f"Model {name} not implemented")

    return model