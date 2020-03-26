from typing import List
from tqdm import tqdm

import numpy as np
import scipy.io.wavfile as wav
import torch
import torch.nn as nn
import wavedream as wd


Modules = List[nn.Module]
class Record:
    def __init__(self, layers: Modules, instru: wd.Instrument) -> None:
        self.layers = layers
        self.instru = instru
        self.history: List[int] = []

    def __len__(self) -> int:
        return len(self.history)

    def process(self, layer: int, norm_grad: float) -> int:
        raise NotImplementedError("Process should be implemented to work.")

    def update(self) -> None:
        for i, layer in enumerate(self.layers):
            norm_grad = torch.norm(layer.weight.grad).item()
            self.history.append(self.process(i, norm_grad))

    def save(self, layer_duration: float, path: str, sr: int = 48_000) -> None:
        n_frames = int(np.floor(layer_duration * len(self.history) * sr))
        times = (np.ones(n_frames) * (1.0 / sr)).cumsum()

        played = -1
        last = None
        samples = []
        for t in tqdm(times, desc="Saving"):
            n = int(np.floor(t // layer_duration))
            if n > played:
                played = n

                if last is not None:
                    self.instru.note_off(t, last)

                if played < len(self):
                    last = self.history[played]
                    self.instru.note_on(t, last)

            samples.append(self.instru(t))

        wav.write(path, sr, np.array(samples))