# нужно написать модель, состоящую из backbone и classification head,
# например, ECAPA-TDNN.
# нужна dataset preparation для обучения

import torch
import torch.nn as nn
import torchaudio
from torchaudio import transforms

from models.models import ECAPA_TDNN
from audio_utils import (
    ms_to_samples, find_voice_activity_segments
)
from utils import map_loaded_weights


class VADModel(nn.Module):
    def __init__(self, threshold=0.003):
        super(VADModel, self).__init__()
        self.backbone = ECAPA_TDNN()
        self.classification_head = lambda x: (x > threshold).float()

    def forward(self, x):
        _, attention_scores = self.backbone(x, return_attention=True)
        attention_scores = attention_scores.mean(dim=1)
        out = self.classification_head(attention_scores)
        return out


def infer():
    filepath = "Independence_Day.wav"
    waveform, sample_rate = torchaudio.load(filepath)
    audio_start_frame, audio_end_frame = 1000000, 2000000
    #waveform = waveform[:, audio_start_frame: audio_end_frame]

    chunk_size = 4 * sample_rate
    chunks = [waveform[:, i:i + chunk_size] for i in range(0, waveform.size(1), chunk_size)]

    sample_rate = 16000
    window_duration = 25
    window_size = ms_to_samples(window_duration, sample_rate)
    stride_duration = 10
    stride_size = ms_to_samples(stride_duration, sample_rate)
    n_mels = 80

    filter_bank = transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=512,
        win_length=window_size,  # 25 ms
        hop_length=stride_size,  # 10 ms
        f_min=0.0,
        f_max=8000,
        pad=0,
        n_mels=n_mels
    )

    vad_model = VADModel()
    weights = torch.load("pretrained_rdino.pth", map_location='cpu')["teacher"]
    weights = map_loaded_weights(weights, vad_model, model_name="rdino_funasr")
    vad_model.load_state_dict(weights)
    vad_model.eval()

    outputs = []
    with torch.no_grad():
        for chunk in chunks:
            mel_spec = filter_bank(chunk[:, :63900])
            output = vad_model(mel_spec)
            outputs.append(output)
    outputs = torch.cat(outputs[:-1]).view(-1)
    outputs = find_voice_activity_segments(outputs, sample_rate)
    return outputs


if __name__ == '__main__':
    infer()