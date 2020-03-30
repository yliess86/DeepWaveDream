"""record.py

The script contains the main Record class responsible for the sonifcation
process of the neural network training. It provides access to the gradient norm
and layer informations to desing custom sonificatopn either for debugging
or artistic purposes.
"""
from typing import List
from tqdm import tqdm

import json
import numpy as np
import scipy.io.wavfile as wav
import torch
import torch.nn as nn
import wavedream as wd


Modules = List[nn.Module]
class Record:
    """Record

    The record class is responsible for the sonifcation process of the neural 
    network training. It provides access to the gradient norm and layer 
    informations to desing custom sonificatopn either for debugging or artistic 
    purposes.
    
    Raises:
        NotImplementedError: process method implementation is left to the user
            to customize the sound mapping to its needs
    
    Attributes:
        layers {List[nn.Module]} -- Chosen layer to record
        instru {wd.Instrument} -- Chosen instrument
        history {List[int]} -- Note records to be played
        cumulate {int} -- batches to cumulate - average (default: {1})
        cumul {List[int]} -- Note to cumulate - will be appended to 
            history when cumulation is done
    """

    def __init__(
        self, 
        layers: Modules, 
        instru: wd.Instrument, 
        cumulate: int = 1
    ) -> None:
        """__init__
        
        Arguments:
            layers {Modules} -- Chosen layer to record
            instru {wd.Instrument} -- Chosen instrument
            cumulate {int} -- batches to cumulate - average (default: {1})
        """
        self.layers = layers
        self.instru = instru
        self.history: List[int] = []

        self.cumulate = cumulate
        self.cumul: List[int] = [0 for i in range(len(self.layers))]

    def __len__(self) -> int:
        """__len__
        
        Returns:
            int -- size of the record history (number of notes to be played)
                num_layers x num_updates
        """
        return len(self.history)

    def process(self, layer: int, norm_grad: float) -> int:
        """[summary]
        
        Arguments:
            layer {int} -- num of current layer
            norm_grad {float} -- norm of the current layer's gradient
        
        Raises:
            NotImplementedError: implementation is left to the user
                to customize the sound mapping to its needs
        
        Returns:
            int -- resulting midi note to be played
        """
        raise NotImplementedError("Process should be implemented to work.")

    def update(self, batch_id: int, batches: int) -> None:
        """update

        Update step to record all notes from all recordred layers.
        """
        for i, layer in enumerate(self.layers):
            norm_grad = torch.norm(layer.weight.grad).item()
            self.cumul[i] += norm_grad

        if (
            (self.cumulate == 1) 
            or (batch_id % (self.cumulate - 1) == 0)
            or (batch_id == (batches - 1))
        ):
            self.history += [
                self.process(i, rec / self.cumulate)
                for layer, rec in enumerate(self.cumul)
            ]
            self.cumul = [0 for i in range(len(self.layers))]


    def save(self, layer_duration: float, path: str, sr: int = 48_000) -> None:
        """save
        
        Arguments:
            layer_duration {float} -- duration of one note
            path {str} -- path to save the resulting wave file 
                (must contain the name and .wav extension)
        
        Keyword Arguments:
            sr {int} -- sample rate of the record to save (default: {48_000})
        """
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

    def checkpoint(self, path: str) -> None:
        """checkpoint
        
        Arguments:
            path {str} -- where to save the checkpoint
        """
        data = {
            "n_layers": len(self.layers),
            "history": self.history 
        }
        with open(path, "w") as fh:
            json.dump(data, fh, indent=4, sort_keys=False)

    @classmethod
    def from_checkpoint(cls, path: str) -> "Record":
        """from checkpoint
        
        Arguments:
            path {str} -- path to the checkpoint

        Returns:
            [Record] -- Fake record for holding checkpoint data
        """
        with open(path, "r") as fh:
            data = json.load(fh.read())
        
        record = cls([None for i in range(data["n_layers"])], None)
        record.history = data["history"]

        return record
