import math
import random
import torch
import numpy as np
from scipy.io import wavfile
from torch.utils.data import Dataset
from torchaudio import transforms
from scipy import signal

from audio_utils import (
    ms_to_samples, pad_split, calculate_db_from_audio,
    apply_room_impulse_response,
    generate_global_and_local_audio,
    load_wav_scp
)

from typing import Dict, List, Optional, Tuple


class Augmentation:
    def __init__(self, noise, reverb, max_frames, n_mels, n_global_views, n_local_views):
        self.noise = noise
        self.reverb = reverb
        self.max_frames = max_frames
        self.n_mels = n_mels
        self.n_global_views = n_global_views
        self.n_local_views = n_local_views
        self.SIGPRO_MIN_RANDGAIN = -7
        self.SIGPRO_MAX_RANDGAIN = 3
        self.room_impulse_response_data = np.load("data/rirs/rir.npy")
        self.noise_types_snr = {
            #'noise': [0, 15],
            'speech': [13, 20],
            'music': [5, 15]
        }
        self.noise_types: List[str] = list(self.noise_types_snr.keys())
        self.noise_types_paths: Dict[str, List[str]] = {}
        sound_paths: Dict[str, str] = load_wav_scp(noise)
        for _, path in sound_paths.items():
            noise_type = path.split('/')[-4]
            self.noise_types_paths.setdefault(noise_type, []).append(path)
        print(self.noise_types_paths.keys())
        self.noise_random = (0, 1, 2)

    def augment_wav(self, audio, augment_profile, is_global):
        if augment_profile['add_rir'] is not None:
            audio = apply_room_impulse_response(audio, augment_profile['add_rir'], augment_profile['rir_gain'])

        if augment_profile['add_noise'] is not None:
            max_frames = self.max_frames if is_global else self.max_frames // 2
            noise_audio = pad_split(augment_profile['add_noise'], max_frames, eval_mode=False)
            noise_db = calculate_db_from_audio(noise_audio[0])
            clean_db = calculate_db_from_audio(audio)
            noise = np.sqrt(10 ** ((clean_db - noise_db - augment_profile['noise_snr']) / 10)) * noise_audio
            audio = audio + noise
        else:
            audio = np.expand_dims(audio, 0)

        return audio

    def generate_augment_profile(self):
        room_impulse_response_gain = np.random.uniform(self.SIGPRO_MIN_RANDGAIN, self.SIGPRO_MAX_RANDGAIN, 1)
        room_impulse_response_file = random.choice(self.room_impulse_response_data)
        noise_type = random.choice(self.noise_types)
        noise_file = random.choice(self.noise_types_paths[noise_type])
        noise_snr = [random.uniform(self.noise_types_snr[noise_type][0], self.noise_types_snr[noise_type][1])]

        augment_profile = {
            'add_rir': None,
            'rir_gain': None,
            'add_noise': None,
            'noise_snr': None
        }
        add_rir, add_noise = False, False
        noise_random_num = random.choices(self.noise_random, weights=(1, 3, 2), k=1)[0]
        if noise_random_num == 1:
            if random.random() > 0.75:
                add_rir = True
            else:
                add_noise = True
        elif noise_random_num == 2:
            add_rir, add_noise = True, True
        if add_rir:
            augment_profile['add_rir'] = room_impulse_response_file
            augment_profile['rir_gain'] = room_impulse_response_gain
        if add_noise:
            augment_profile['add_noise'] = noise_file
            augment_profile['noise_snr'] = noise_snr

        return augment_profile

    def __call__(self, audio, is_global):
        if is_global:
            # don't augment global audio
            return audio
        augment_profile = self.generate_augment_profile()
        audio = self.augment_wav(audio, augment_profile, is_global)
        return audio


class SDEPDataset(Dataset):
    def __init__(self, data, noise, reverb, max_frames, n_mels, n_global_views, n_local_views):
        self.data = data
        self.noise = noise
        self.reverb = reverb
        self.max_frames = max_frames
        self.n_mels = n_mels
        self.n_global_views = n_global_views
        self.n_local_views = n_local_views
        self.SIGPRO_MIN_RANDGAIN = -7
        self.SIGPRO_MAX_RANDGAIN = 3

        self.sample_rate = 16000
        self.window_duration = 25
        self.window_size = ms_to_samples(self.window_duration, self.sample_rate)
        self.stride_duration = 10
        self.stride_size = ms_to_samples(self.stride_duration, self.sample_rate)

        self.filter_bank = transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=512,
            win_length=self.window_size, # 25 ms
            hop_length=self.stride_size, # 10 ms
            f_min=0.0,
            f_max=8000,
            pad=0,
            n_mels=self.n_mels
        )
        self.augmentation = Augmentation(noise, reverb, max_frames, n_mels, n_global_views, n_local_views)
        self.data = list(load_wav_scp(data).values())[:16]

    def __getitem__(self, index):
        global_audios, local_audios = generate_global_and_local_audio(
            self.data[index], self.max_frames, self.n_global_views, self.n_local_views
        )
        global_audios_augmented = []
        local_audios_augmented = []
        for global_audio in global_audios:
            global_audios_augmented.append(self.augmentation(global_audio, True))
        for local_audio in local_audios:
            local_audios_augmented.append(self.augmentation(local_audio, False))

        global_audios_augmented = np.stack(global_audios_augmented, axis=0)
        local_audios_augmented = np.concatenate(local_audios_augmented, axis=0)

        with torch.no_grad():
            # To obtain even-numbered frames, we delete 100 points for 4s segments
            global_features = self.filter_bank(torch.FloatTensor(global_audios_augmented[:, :63900])) # channels x n_mels x time
            local_features = self.filter_bank(torch.FloatTensor(local_audios_augmented[:, :31900])) # channels x n_mels x time

        output = {
            'global': global_features,
            'local': local_features
        }

        return output

    def __len__(self):
        return len(self.data)


class SDEPInferenceDataset(Dataset):
    def __init__(self, data, noise, reverb, max_frames, n_mels, n_global_views, n_local_views):
        self.data = data
        self.n_mels = n_mels
        self.n_global_views = n_global_views
        self.n_local_views = n_local_views
        self.SIGPRO_MIN_RANDGAIN = -7
        self.SIGPRO_MAX_RANDGAIN = 3

        self.sample_rate = 16000
        self.window_duration = 25
        self.window_size = ms_to_samples(self.window_duration, self.sample_rate)
        self.stride_duration = 10
        self.stride_size = ms_to_samples(self.stride_duration, self.sample_rate)

        self.filter_bank = transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=512,
            win_length=self.window_size, # 25 ms
            hop_length=self.stride_size, # 10 ms
            f_min=0.0,
            f_max=8000,
            pad=0,
            n_mels=self.n_mels
        )
        self.data = list(load_wav_scp(data).values())[:16]

    def __getitem__(self, index):
        sample_rate, audio = wavfile.read(self.data[index])
        if len(audio.shape) == 2:
            audio = audio[:, 0]

        with torch.no_grad():
            # To obtain even-numbered frames, we delete 100 points for 4s segments
            audio_features = self.filter_bank(
                torch.FloatTensor(audio[:, :63900]))  # channels x n_mels x time

        output = {
            'audio': audio_features,
        }

        return output

    def __len__(self):
        return len(self.data)
