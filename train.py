import clearml
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn
from torch.utils.data import DataLoader

from models import SDEPModel, SDEPModule
from losses import (cross_entropy_loss, entropy_loss, sdep_loss)
from dataset import (
    SDEPDataset, SDEPInferenceDataset
)
from embedder import train_embedder


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(config):
    # config = {
    #     "n_epochs": 1,
    #     "batch_size": 8,
    #     "n_prototypes": 10,
    #     "n_networks": 5,
    #     "feature_dim": 512,
    #     "n_global_views": 2,
    #     "n_local_views": 4,
    #     "prototype_dim": 256,
    # }

    #clearml.browser_login()
    config = OmegaConf.to_container(config)
    #task = clearml.Task.init(project_name="Speaker diarization", task_name="SDEP training")
    #task.connect_configuration(config, name="baseline", description="baseline config")
    train_embedder(config)
    #task.close()
    # train_vad(config)
    # train_clustering(config)



if __name__ == "__main__":
    main()

