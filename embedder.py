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


def train_embedder(config):
    data_path = "data/3dspeaker/test/wav.scp"
    noise_path = "data/musan/wav.scp"
    # data, noise, reverb, max_frames, n_mels, n_global_views, n_local_views
    batch_size = config["batch_size"]
    n_global_views, n_local_views = 2, 4
    dataset = SDEPDataset(
        data=data_path,
        noise=noise_path,
        reverb=None,
        max_frames=400,
        n_mels=80,
        n_global_views=n_global_views,
        n_local_views=n_local_views,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    n_networks = config["n_networks"]
    n_prototypes = config["n_prototypes"]
    sdep_model = SDEPModel(config)
    sdep_model.teacher.load_state_dict(sdep_model.student.state_dict())
    # for param in sdep_model.teacher.parameters():
    #     param.requires_grad = False
    criterion = sdep_loss
    optimizer = torch.optim.Adam(sdep_model.student.parameters(), lr=1e-3)
    start_epoch = 0
    n_epochs = config["n_epochs"]
    for epoch in range(start_epoch, n_epochs):
        for data in dataloader:
            x_local = data["local"] # batch_size x n_local_views x feature_dim
            x_global = data["global"] # batch_size x n_global_views x feature_dim
            shape = x_local.shape
            batch_size, _, n_mels, frame_size = x_local.shape
            x_local = x_local.view(-1, n_mels, frame_size)
            x_global = x_global.view(-1, n_mels, frame_size * 2)
            teacher_hist, student_hist = sdep_model.forward(x_global, x_local)
            teacher_hist  = teacher_hist.view(batch_size, n_networks, n_global_views, n_prototypes)
            student_hist = student_hist.view(batch_size, n_networks, n_local_views, n_prototypes)
            loss = criterion(teacher_hist, student_hist)
            print(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                m = 0.0  # momentum parameter
                for param_q, param_k in zip(sdep_model.student.parameters(), sdep_model.teacher.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)


def infer_embedder(config):
    with torch.no_grad():
        data_path = "data/3dspeaker/test/wav.scp"
        batch_size = config["batch_size"]
        n_global_views, n_local_views = 2, 4
        dataset = SDEPInferenceDataset(
            data=data_path,
            noise=None,
            reverb=None,
            max_frames=400,
            n_mels=80,
            n_global_views=n_global_views,
            n_local_views=n_local_views,
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        sdep_teacher_model = SDEPModule(config["n_networks"])
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


























