from deepwavedream.utils import quantize_to_nearest

import deepwavedream as dwd
import numpy as np


def simulate_accuracy(epochs: int) -> np.ndarray:
    X = np.arange(0, epochs) / epochs
    train_acc = 1 - np.exp(1.5 * (-X * 3.0 - 0.1))
    valid_acc = 0.9 - ((X - 0.46) * 1.8) ** 2
    return train_acc, valid_acc


def simulate_grad_norm(epochs: int, layers: int, steps: int) -> np.ndarray:
    grad_norm = np.random.random(size=epochs * layers * steps)
    grad_norm *= (0.7 - 0.4)
    grad_norm += 0.4
    return grad_norm


"""
Custom Recorder extension with custom process function as required by the api.
"""
class Record(dwd.Record):
    def __init__(self, *args, scale: str, key: int, **kwargs) -> None:
        super(Record, self).__init__(*args, **kwargs)
        self.scale = scale
        self.key = key
        self.notes = list(range(len(wd.NOTES)))
        if scale == "minor":
            self.notes = wd.Scale.minor(key)
        elif scale == "major":
            self.notes = wd.Scale.major(key) 

    def process(self, layer: int, norm_grad: float) -> int:
        freq = 110.0 + (norm_grad + 0.5) * 220.0 * (2 ** layer)
        return quantize_to_nearest(freq, self.notes)


if __name__ == "__main__":
    from .base import *
    
    import matplotlib.pyplot as plt
    import wavedream as wd
    
    epochs = 30
    layers = 5
    steps  = 3

    train_acc, valid_acc = simulate_accuracy(epochs)
    grad_norm = simulate_grad_norm(epochs, layers, steps)
    
    base = Bell
    instru = DreamSynth(base, 0.1, 0.1, 0.95, 1.0, 1)
    record = Record(list(range(layers)), instru=instru, scale="minor", key=69)
    record.raw_history = grad_norm
    record.history = [
        record.process(i % layers, raw) 
        for i, raw in enumerate(grad_norm)
    ] 


    wet, feedback, gain = [], [], []
    def callback(self: dwd.Record, note: int) -> None:
        global epochs, record, valid_acc
        global wet, feedback, gain

        epoch = note // (len(record) // epochs)
        v_acc = valid_acc[epoch]
        vinv_acc = 1 - v_acc
        for reverb in self.instru.reverbs:
            reverb.wet = vinv_acc
            reverb.feedback = min(max(vinv_acc, 0.05), 0.95)
            reverb.gain = v_acc * (0.8 - 0.1) + 0.1
        wet.append(self.instru.reverbs[0].wet)
        feedback.append(self.instru.reverbs[0].feedback)
        gain.append(self.instru.reverbs[0].gain)

    record.save(0.1, "out.wav", callback=callback)


    plt.figure()

    plt.subplot(411)
    plt.ylim(0, 1)
    for layer in range(layers):
        plt.plot(record.raw_history[layer::layers], label=f"layer_{layer}")
    plt.legend()

    X = np.arange(len(record) // layers)
    plt.subplot(412)
    for layer in range(layers):
        plt.scatter(
            X, 
            record.history[layer::layers], 
            label=f"layer_{layer}",
            marker="_"
        )
    plt.legend()

    plt.subplot(413)
    plt.ylim(0, 1)
    plt.plot(train_acc, label="train")
    plt.plot(valid_acc, label="valid")
    plt.legend()

    plt.subplot(4, 3, 10)
    plt.ylim(0, 1)
    plt.plot(wet, label="wet")
    plt.legend()

    plt.subplot(4, 3, 11)
    plt.ylim(0, 1)
    plt.plot(feedback, label="feedback")
    plt.legend()

    plt.subplot(4, 3, 12)
    plt.ylim(0, 1)
    plt.plot(gain, label="gain")
    plt.legend()

    plt.tight_layout()
    plt.show()