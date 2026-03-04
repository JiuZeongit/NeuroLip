# utils/data.py
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate


SensorResolution = Tuple[int, int]  # (H, W)


def seed_everything(seed: int) -> None:
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f"Global seed set to {seed}")


def collate_events(batch):
    """Batch events into a single tensor with appended batch index column.

    Each sample event tensor is [N_i, 4] = [x, y, t, p].
    Output events is [sum N_i, 5] = [x, y, t, p, b].
    """
    labels = []
    events_list = []
    for batch_idx, (events, label) in enumerate(batch):
        labels.append(label)
        batch_col = np.full((len(events), 1), batch_idx, dtype=np.float32)
        events_list.append(np.concatenate([events, batch_col], axis=1))

    if len(events_list) == 0:
        events = torch.empty((0, 5), dtype=torch.float32)
    else:
        events = torch.from_numpy(np.concatenate(events_list, axis=0)).float()
    labels = default_collate(labels)
    return events, labels


def event_level_denoise(events: np.ndarray, filter_time_us: int = 10000) -> np.ndarray:
    """4-neighborhood spatiotemporal denoising on raw events.

    Keeps an event if there exists at least one neighbor in 4-connected pixels
    within the temporal window.
    """
    if len(events) == 0:
        return events

    events_out = np.zeros_like(events)
    out_idx = 0

    width = int(events[:, 0].max()) + 1
    height = int(events[:, 1].max()) + 1
    timestamp_memory = np.zeros((width, height), dtype=np.float32) + filter_time_us

    for event in events:
        x = int(event[0])
        y = int(event[1])
        t = event[2]
        timestamp_memory[x, y] = t + filter_time_us

        if (
            (x > 0 and timestamp_memory[x - 1, y] > t)
            or (x < width - 1 and timestamp_memory[x + 1, y] > t)
            or (y > 0 and timestamp_memory[x, y - 1] > t)
            or (y < height - 1 and timestamp_memory[x, y + 1] > t)
        ):
            events_out[out_idx] = event
            out_idx += 1

    return events_out[:out_idx]


def random_spatial_translation(events: np.ndarray, max_shift: int = 20, resolution: SensorResolution = (160, 200)) -> np.ndarray:
    if len(events) == 0:
        return events
    h, w = resolution
    x_shift, y_shift = np.random.randint(-max_shift, max_shift + 1, size=(2,))
    events = events.copy()
    events[:, 0] += x_shift
    events[:, 1] += y_shift
    valid = (
        (events[:, 0] >= 0) & (events[:, 0] < w) &
        (events[:, 1] >= 0) & (events[:, 1] < h)
    )
    return events[valid]


def random_horizontal_flip(events: np.ndarray, resolution: SensorResolution = (160, 200), p: float = 0.5) -> np.ndarray:
    if len(events) == 0:
        return events
    h, w = resolution
    _ = h
    events = events.copy()
    if np.random.rand() < p:
        events[:, 0] = w - 1 - events[:, 0]
    return events


def random_event_sparsification(events: np.ndarray, drop_rate: float = 0.5) -> np.ndarray:
    if len(events) == 0:
        return events
    num_keep = int(len(events) * (1.0 - drop_rate))
    if num_keep <= 0:
        return events[:0]
    keep_idx = np.random.choice(len(events), size=num_keep, replace=False)
    return events[keep_idx]


@dataclass
class AugmentationConfig:
    resolution: SensorResolution = (160, 200)
    translation_max_shift: int = 20
    flip_prob: float = 0.5
    sparsify_apply_prob: float = 0.3
    sparsify_drop_rate_range: Tuple[float, float] = (0.4, 0.7)


class DVSpeakerDataset(Dataset):
    """DVSpeaker dataset loader.

    Directory layout:
      root_dir/
        1/*.npy
        2/*.npy
        ...
        50/*.npy

    Filename format:
      {light}_{degree}_{num}_{A}_{B}.npy
    """

    classes = [str(i) for i in range(50)]
    sensor_size = (200, 160, 2)  # (W, H, P)

    def __init__(
        self,
        root_dir: str,
        train: bool,
        filter_condition: Optional[Dict[str, Sequence]] = None,
        denoise_filter_time_us: int = 10000,
        augmentation: Optional[AugmentationConfig] = None,
    ):
        self.root_dir = root_dir
        self.train = train
        self.filter_condition = filter_condition or {}
        self.denoise_filter_time_us = denoise_filter_time_us
        self.augmentation = augmentation or AugmentationConfig()
        self.samples = self._scan_samples()
        self.targets = [label for _, label in self.samples]

    def _match_filter(self, light: int, degree: int, num: str) -> bool:
        if "Light" in self.filter_condition and light not in self.filter_condition["Light"]:
            return False
        if "Degree" in self.filter_condition and degree not in self.filter_condition["Degree"]:
            return False
        if "Num" in self.filter_condition and num not in self.filter_condition["Num"]:
            return False
        return True

    def _scan_samples(self) -> List[Tuple[str, int]]:
        samples: List[Tuple[str, int]] = []
        for label_name in sorted(os.listdir(self.root_dir), key=lambda x: int(x) if x.isdigit() else x):
            label_dir = os.path.join(self.root_dir, label_name)
            if not os.path.isdir(label_dir):
                continue
            if not str(label_name).isdigit():
                continue
            label = int(label_name)
            for file_name in sorted(os.listdir(label_dir)):
                if not file_name.endswith(".npy"):
                    continue
                parts = file_name.split("_")
                if len(parts) != 5:
                    continue
                try:
                    light = int(parts[0])
                    degree = int(parts[1])
                    num = parts[2]
                    _exp_num = int(parts[3])
                    _times = int(parts[4].split(".")[0])
                except ValueError:
                    continue

                if not self._match_filter(light, degree, num):
                    continue

                samples.append((os.path.join(label_dir, file_name), label - 1))
        return samples

    @staticmethod
    def _structured_to_event_array(orig_events: np.ndarray) -> np.ndarray:
        return np.column_stack([
            orig_events["x"],
            orig_events["y"],
            orig_events["t"],
            orig_events["p"],
        ]).astype(np.float32)

    def _apply_training_augmentation(self, events: np.ndarray) -> np.ndarray:
        cfg = self.augmentation
        events = random_spatial_translation(events, max_shift=cfg.translation_max_shift, resolution=cfg.resolution)
        events = random_horizontal_flip(events, resolution=cfg.resolution, p=cfg.flip_prob)
        if np.random.rand() < cfg.sparsify_apply_prob:
            lo, hi = cfg.sparsify_drop_rate_range
            drop_rate = np.random.uniform(lo, hi)
            events = random_event_sparsification(events, drop_rate=drop_rate)
        return events

    def __getitem__(self, index: int):
        file_path, target = self.samples[index]
        orig_events = np.load(file_path)
        events = self._structured_to_event_array(orig_events)

        # Raw-event denoising first (paper-aligned pipeline)
        events = event_level_denoise(events, filter_time_us=self.denoise_filter_time_us)

        if self.train:
            events = self._apply_training_augmentation(events)

        return events, target

    def __len__(self) -> int:
        return len(self.samples)
