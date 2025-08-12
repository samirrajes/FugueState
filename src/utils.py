import torch
import torch.nn.functional as F
import librosa
import numpy as np

# References:
# https://librosa.org/doc/0.11.0/generated/librosa.feature.melspectrogram.html
# https://github.com/NVIDIA/BigVGAN/blob/main/loss.py

# Note: ChatGPT was used for help developing the mel-spectrogram loss function

# mel-spectrogram loss
def mel_loss(real_spec, fake_spec, eps=1e-8):
    # real_spec, fake_spec in [-1,1]
    log_r = (real_spec + 1) / 2 * 80 - 80
    log_f = (fake_spec + 1) / 2 * 80 - 80
    P_r = 10.0 ** (log_r / 10.0)
    P_f = 10.0 ** (log_f / 10.0)
    sc = torch.norm(P_r - P_f, p='fro') / (torch.norm(P_r, p='fro') + eps)
    lm = F.l1_loss(log_f, log_r)
    return sc + lm

# spectrogram inversion definition
# converts a normalized mel-spectrogram into a time-domain waveform using a tuned Griffin–Lim algorithm
def spec_to_audio(
    spec: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
    win_length: int = None,
    power: float = 2.0,
    n_iter: int = 400,
    momentum: float = 0.9,
    window: str = 'hann'
) -> np.ndarray:
    # map normalized spec [-1,1] to log dB scale [ -80, 0 ]
    log_S = (spec + 1.0) / 2.0 * 80.0 - 80.0
    # convert to linear-power spectrogram
    S = librosa.db_to_power(log_S, ref=1.0)
    # invert mel to stft mag
    stft_mag = librosa.feature.inverse.mel_to_stft(
        S,
        sr=sr,
        n_fft=n_fft,
        power=power
    )
    # run Griffin–Lim w/ momentum & custom window
    y = librosa.griffinlim(
        stft_mag,
        n_iter=n_iter,
        hop_length=hop_length,
        win_length=win_length or n_fft,
        window=window,
        momentum=momentum,
        init='random'
    )
    # enforce length based on hop_length and n_fft
    target_len = hop_length * (spec.shape[1] - 1) + n_fft
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode='constant')
    else:
        y = y[:target_len]
    return y
