from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

from models.models import SDEPModule
from losses import (sdep_loss)
from datasets.dataset import (
    SDEPDataset, SDEPInferenceDataset
)
from models.model_factory import create_model
from datasets.dataset_factory import create_dataset
from configs.train_config import TrainConfig
from configs.inference_config import InferenceConfig


def train_embedder(embedder_config):
    config: TrainConfig
    config = OmegaConf.structured(TrainConfig)
    config = OmegaConf.merge(config, embedder_config)

    model_name = config["model"]["name"]
    dataset_name = config["dataset"]["name"]
    n_global_views = config["dataset"][dataset_name]["n_global_views"]
    n_local_views = config["dataset"][dataset_name]["n_local_views"]
    n_networks = config.model[model_name].n_networks
    n_prototypes = config.model[model_name].n_prototypes

    dataset = create_dataset(config.dataset)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    sdep_model = create_model(config.model)
    sdep_model.teacher.load_state_dict(sdep_model.student.state_dict())
    for param in sdep_model.teacher.parameters():
        param.requires_grad = False
    criterion = sdep_loss
    optimizer = torch.optim.Adam(sdep_model.student.parameters(), lr=1e-3)
    start_epoch = 0
    n_epochs = config["n_epochs"]
    for epoch in range(start_epoch, n_epochs):
        for data in dataloader:
            x_local = data["local"] # batch_size x n_local_views x feature_dim
            x_global = data["global"] # batch_size x n_global_views x feature_dim
            batch_size, _, n_mels, frame_size = x_local.shape
            x_local = x_local.view(-1, n_mels, frame_size)
            x_global = x_global.view(-1, n_mels, frame_size * 2)
            teacher_hist, student_hist = sdep_model.forward(x_global, x_local)
            teacher_hist  = teacher_hist.view(batch_size, n_networks, n_global_views, n_prototypes)
            student_hist = student_hist.view(batch_size, n_networks, n_local_views, n_prototypes)
            loss = criterion(teacher_hist, student_hist)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                m = 0.01  # momentum parameter
                for param_q, param_k in zip(sdep_model.student.parameters(), sdep_model.teacher.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)


def infer_embedder(embedder_config):
    config: InferenceConfig
    config = OmegaConf.structured(TrainConfig)
    config = OmegaConf.merge(config, embedder_config)
    with torch.no_grad():
        dataset = create_dataset(config.dataset)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        sdep_teacher_model = create_model(config.model).teacher
        sdep_teacher_model.load_state_dict(...).to(config["device"])
        sdep_teacher_model.eval()
        start_epoch = 0
        n_epochs = config["n_epochs"]
        embeddings = []
        for epoch in range(start_epoch, n_epochs):
            for data in dataloader:
                x_audio = data["audio"] # batch_size x 1 x feature_dim
                batch_size, _, n_mels, frame_size = x_audio.shape
                x_audio = x_audio.view(-1, n_mels, frame_size)
                embedding = sdep_teacher_model.get_embedding(x_audio)
                embeddings.append(embedding)
        embeddings = torch.cat(embeddings, dim=0).cpu().numpy()
    return embeddings


























