"""record.py

The script contains the main Record class responsible for the sonifcation
process of the neural network training. It provides access to the gradient norm
and layer informations to desing custom sonificatopn either for debugging
or artistic purposes.
"""
from typing import List
from tqdm import tqdm

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
    """

    def __init__(self, layers: Modules, instru: wd.Instrument) -> None:
        """__init__
        
        Arguments:
            layers {Modules} -- Chosen layer to record
            instru {wd.Instrument} -- Chosen instrument
        """
        self.layers = layers
        self.instru = instru
        self.history: List[int] = []

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

    def update(self) -> None:
        """update

        Update step to record all notes from all recordred layers.
        """
        for i, layer in enumerate(self.layers):
            norm_grad = torch.norm(layer.weight.grad).item()
            self.history.append(self.process(i, norm_grad))

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