from scipy.io import wavfile
from scipy import signal
import numpy as np
import torch

import random


def list_to_tensor(list_of_tensors):
    return torch.stack(list_of_tensors, dim=0)


def load_wav_scp(fpath):
    with open(fpath) as f:
        rows = [i.strip() for i in f.readlines()]
        result = {i.split()[0]: i.split()[1] for i in rows}
    return result


def ms_to_samples(ms, sample_rate):
    return int(ms * sample_rate / 1000)


def calculate_db_from_audio(audio, eps=1e-4):
    return 10 * np.log10(np.mean(audio ** 2) + eps)


def apply_room_impulse_response(audio, room_impulse_response, filter_gain):
    room_impulse_response = np.multiply(room_impulse_response, pow(10, 0.1 * filter_gain))
    audio_rir = signal.convolve(audio, room_impulse_response, mode='full')[:len(audio)]

    return audio_rir


def generate_global_and_local_audio(filename, max_frames, n_global_views, n_local_views):
    # Maximum audio length
    max_audio_size = max_frames * 160
    sample_rate, audio = wavfile.read(filename)
    if len(audio.shape) == 2:
        audio = audio[:, 0]
    audio_size = audio.shape[0]

    if audio_size <= max_audio_size:
        shortage = max_audio_size - audio_size + n_global_views
        audio = np.pad(audio, (0, shortage), 'constant', constant_values=0)
        audio_size = audio.shape[0]

    glb_rand = audio_size - max_audio_size
    assert glb_rand >= n_global_views - 1
    glb_start_point = random.sample(range(0, glb_rand), n_global_views)
    glb_start_point.sort()
    np.random.shuffle(glb_start_point)
    glb_audio = []
    for asf in glb_start_point:
        glb_audio.append(audio[int(asf): int(asf) + max_audio_size])

    local_rand = audio_size - max_audio_size // 2
    local_start_point = random.sample(range(0, local_rand), n_local_views)
    local_start_point.sort()
    np.random.shuffle(local_start_point)
    local_audio = []
    for asf in local_start_point:
        local_audio.append(audio[int(asf): int(asf) + max_audio_size // 2])

    global_audios = np.stack(glb_audio, axis=0).astype(np.float)
    local_audios = np.stack(local_audio,axis=0).astype(np.float)

    return global_audios, local_audios


def pad_split(filename, max_frames, eval_mode=False, num_eval=10):
    # maybe we should load the whole MUSAN dataset into memory
    # instead of reading filename from disk every time
    # TODO: check if this is a good idea1
    max_audio = max_frames * 160
    sample_rate, audio = wavfile.read(filename)
    if len(audio.shape) == 2:
        audio = audio[:, 0]
    audio_size = audio.shape[0]

    if audio_size <= max_audio:
        shortage = max_audio - audio_size
        audio = np.pad(audio, (0, shortage), 'constant', constant_values=0)
        audio_size = audio.shape[0]

    if eval_mode:
        start_point = np.linspace(0, audio_size - max_audio, num=num_eval)
    else:
        start_point = np.array([np.int64(random.random() * (audio_size - max_audio))])

    feat = []
    if eval_mode and max_frames == 0:
        feat.append(audio)
    else:
        for asf in start_point:
            feat.append(audio[int(asf): int(asf) + max_audio])
    feats = np.stack(feat, axis=0).astype(np.float)

    return feats
