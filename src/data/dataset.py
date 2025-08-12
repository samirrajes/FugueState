# src/data/dataset.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import librosa

# Map-style dataset for storing guitar chord data

class ChordSpec(Dataset):
    def __init__(self,
                 root,
                 sr=22050,
                 n_fft=1024,
                 hop_length=256,
                 n_mels=128,
                 fmin=50.0,
                 fmax=8000.0,
                 duration=3.0):
        # gather all audio files
        self.files = []
        for dp, _, fnames in os.walk(root):
            for f in fnames:
                if f.lower().endswith((".wav", ".mp3", ".flac")):
                    self.files.append(os.path.join(dp, f))
        if not self.files:
            raise RuntimeError(f"No audio found under {root}")

        self.sr         = sr
        self.n_fft      = n_fft
        self.hop_length = hop_length
        self.n_mels     = n_mels
        self.fmin       = fmin
        self.fmax       = fmax
        self.duration   = duration

        # how many frames we expect
        total_samples     = int(sr * duration)
        self.target_frames = 1 + (total_samples - n_fft) // hop_length

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        # load exactly duration
        y, _ = librosa.load(path, sr=self.sr, mono=True, duration=self.duration)
        # mel-power spectrogram
        S = librosa.feature.melspectrogram(
            y=y, sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            power=2.0
        )
        log_S = librosa.power_to_db(S, ref=np.max)
        spec  = (log_S + 80.0) / 80.0 * 2.0 - 1.0

        # pad / truncate time axis
        T = spec.shape[1]
        if T < self.target_frames:
            pad_amt = self.target_frames - T
            spec = np.pad(spec, ((0,0),(0,pad_amt)), mode='constant', constant_values=-1.0)
        else:
            spec = spec[:, :self.target_frames]

        return torch.from_numpy(spec).unsqueeze(0).float()
