"""
UrbanSound8K: environmental sound classification as log-mel spectrograms.

Reference:
  J. Salamon, C. Jacoby, J. P. Bello, "A Dataset and Taxonomy for Urban
  Sound Research", ACM Multimedia, 2014.  https://zenodo.org/records/1203745

8732 labeled clips (<= 4 s) of 10 classes (air_conditioner, car_horn,
children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer,
siren, street_music), pre-split into 10 folds.  Default protocol here:
folds 1-9 train, fold 10 test (override with test_fold).

Feature: mono, resampled to 22.05 kHz, padded/cropped to n_frames*hop
samples, MelSpectrogram(n_fft=1024, hop=512, n_mels) -> log(mel + 1e-6),
standardized with train-fold mean/std.  Sample shape (1, n_mels, n_frames);
the default (64, 176) gives 4 pools of VGG-9 -> 4x11.

Features are computed once and cached under
  <data_root>/UrbanSound8K/cache_logmel{n_mels}x{n_frames}/fold{k}.npz
from
  <data_root>/UrbanSound8K/metadata/UrbanSound8K.csv
  <data_root>/UrbanSound8K/audio/fold{k}/*.wav
"""

import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

US8K_CLASSES = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
                'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren',
                'street_music']

SAMPLE_RATE = 22050
N_FFT = 1024
HOP = 512


def _dataset_root(data_root):
    for cand in (os.path.join(data_root, 'UrbanSound8K'),
                 os.path.join(data_root, 'UrbanSound8K', 'UrbanSound8K')):
        if os.path.exists(os.path.join(cand, 'metadata', 'UrbanSound8K.csv')):
            return cand
    raise FileNotFoundError(
        f"UrbanSound8K not found under {data_root}: expected "
        f"UrbanSound8K/metadata/UrbanSound8K.csv and UrbanSound8K/audio/fold*/")


def _read_metadata(root):
    rows = []
    with open(os.path.join(root, 'metadata', 'UrbanSound8K.csv'), newline='') as f:
        for r in csv.DictReader(f):
            rows.append((r['slice_file_name'], int(r['fold']), int(r['classID'])))
    return rows


def _load_wave(path):
    """Load a wav as mono float32 at SAMPLE_RATE."""
    import soundfile as sf
    import torchaudio.functional as AF
    wav, sr = sf.read(path, dtype='float32', always_2d=True)  # (n, ch)
    wav = torch.from_numpy(wav.mean(axis=1))
    if sr != SAMPLE_RATE:
        wav = AF.resample(wav, sr, SAMPLE_RATE)
    return wav


def _compute_fold(root, rows, fold, n_mels, n_frames, mel):
    n_samples = (n_frames - 1) * HOP  # center=True -> exactly n_frames frames
    xs, ys = [], []
    for fname, fd, cid in rows:
        if fd != fold:
            continue
        wav = _load_wave(os.path.join(root, 'audio', f'fold{fold}', fname))
        if wav.numel() < n_samples:
            wav = torch.nn.functional.pad(wav, (0, n_samples - wav.numel()))
        else:
            wav = wav[:n_samples]
        spec = torch.log(mel(wav) + 1e-6)  # (n_mels, n_frames)
        xs.append(spec.numpy().astype(np.float32))
        ys.append(cid)
    return np.stack(xs)[:, None], np.array(ys, dtype=np.int64)


def build_cache(data_root, n_mels=64, n_frames=176, folds=range(1, 11), verbose=True):
    """Compute and cache log-mel features for the requested folds."""
    import torchaudio
    root = _dataset_root(data_root)
    cache_dir = os.path.join(root, f'cache_logmel{n_mels}x{n_frames}')
    os.makedirs(cache_dir, exist_ok=True)
    rows = _read_metadata(root)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP, n_mels=n_mels)
    for fold in folds:
        out = os.path.join(cache_dir, f'fold{fold}.npz')
        if os.path.exists(out):
            continue
        x, y = _compute_fold(root, rows, fold, n_mels, n_frames, mel)
        np.savez(out, x=x, y=y)
        if verbose:
            print(f"  [urbansound8k] fold{fold}: {x.shape} -> {out}")
    return cache_dir


def _load_folds(cache_dir, folds):
    xs, ys = [], []
    for fold in folds:
        d = np.load(os.path.join(cache_dir, f'fold{fold}.npz'))
        xs.append(d['x']); ys.append(d['y'])
    return np.concatenate(xs), np.concatenate(ys)


class UrbanSound8KDataset(Dataset):
    """In-memory log-mel features. Returns ((1, n_mels, n_frames) tensor, label)."""

    def __init__(self, x, y, mean, std, train=False, time_shift=0):
        self.x = ((x - mean) / std).astype(np.float32)
        self.y = y
        self.train = train
        self.time_shift = time_shift

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.train and self.time_shift > 0:
            s = np.random.randint(-self.time_shift, self.time_shift + 1)
            x = np.roll(x, s, axis=-1)
        return torch.from_numpy(np.ascontiguousarray(x)), int(self.y[idx])


def urbansound8k_dataloaders(data_root, batch_size=64, n_mels=64, n_frames=176,
                             test_fold=10, num_workers=4, distributed=False,
                             time_shift=8, **kwargs):
    """(train_loader, test_loader): folds != test_fold train, fold test_fold test."""
    cache_dir = build_cache(data_root, n_mels=n_mels, n_frames=n_frames)
    train_folds = [f for f in range(1, 11) if f != test_fold]
    x_tr, y_tr = _load_folds(cache_dir, train_folds)
    x_te, y_te = _load_folds(cache_dir, [test_fold])
    mean, std = float(x_tr.mean()), float(x_tr.std() + 1e-6)

    train_set = UrbanSound8KDataset(x_tr, y_tr, mean, std, train=True, time_shift=time_shift)
    test_set = UrbanSound8KDataset(x_te, y_te, mean, std, train=False)

    train_sampler = (torch.utils.data.distributed.DistributedSampler(train_set)
                     if distributed else None)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=train_sampler,
        drop_last=True)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
