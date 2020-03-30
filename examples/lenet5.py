"""lenet5.py

The scipt is an example of usage for the deepwavedream library applied to the
training of a LeNet5 convolutional deep learning model on the MNIST dataset
using an SGD optimizer with Pytorch.

python3 -m examples.lenet5 --help 
"""
from deepwavedream.utils import quantize_to_nearest
from tqdm import tqdm
from .base import *

import torch
import torch.nn as nn
import deepwavedream as dwd


class LeNet5(nn.Module):
    def __init__(self) -> None:
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1,  6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120,  84)
        self.fc3 = nn.Linear( 84,  10)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = torch.max_pool2d(torch.relu(self.conv1(X)), 2)
        X = torch.max_pool2d(torch.relu(self.conv2(X)), 2)
        
        X = X.view(X.size(0), -1)
        
        X = torch.relu(self.fc1(X))
        X = torch.relu(self.fc2(X))
        X = torch.relu(self.fc3(X))
        
        return X


"""
Custom Recorder extension with custom process function as required by the api.
"""
class Record(dwd.Record):
    def __init__(self, *args, scale: str, **kwargs) -> None:
        super(Record, self).__init__(*args, **kwargs)
        self.scale = scale
        self.notes = list(range(len(wd.NOTES)))
        if scale == "minor":
            self.notes = wd.Scale.minor(69)
        elif scale == "major":
            self.notes = wd.Scale.major(69) 

    def process(self, layer: int, norm_grad: float) -> int:
        freq = 220.0 + 50.0 * (2 ** layer) + norm_grad * 200.0
        return quantize_to_nearest(freq, self.notes)


if __name__ == "__main__":
    from argparse import ArgumentParser
    from torch.optim import SGD
    from torch.utils.data import DataLoader
    from torchvision.datasets import mnist
    from torchvision.transforms import ToTensor


    parser = ArgumentParser(description="Train and record LeNet5 on MNIST.")
    parser.add_argument("--epochs",         "-e", default=2,      type=int)
    parser.add_argument("--batch_size",     "-b", default=256,    type=int)
    parser.add_argument("--learning_rate",  "-l", default=1e-1,   type=float)
    parser.add_argument("--layer_duration", "-d", default=1e-1,   type=float)
    parser.add_argument("--cumulate",             default=1,      type=int)
    
    parser.add_argument("--scale",          "-s", default="all",  type=str)
    parser.add_argument("--instru",         "-i", default="bell", type=str)
    parser.add_argument("--gain",           "-g", default=0.2,    type=float)
    parser.add_argument("--feedback",       "-f", default=0.95,   type=float)
    parser.add_argument("--wet",            "-w", default=0.9,    type=float)
    
    parser.add_argument("--path",           "-p", required=True,  type=str)
    parser.add_argument("--cuda",           "-c", action="store_true")
    args = parser.parse_args()


    loader = DataLoader(
        mnist.MNIST("./dataset", True, transform=ToTensor(), download=True), 
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )


    if args.cuda:
        model = LeNet5().cuda()
        critertion = nn.CrossEntropyLoss().cuda()
    else:
        model = LeNet5()
        critertion = nn.CrossEntropyLoss()
    optim = SGD(model.parameters(), lr=args.learning_rate)


    """
    Record initialization by defining the instrument to use as well as the 
    layers to be recorded. Layer order is important for the sonification as the
    information is passed to the process function and may be used in some way.
    """
    base   = Bell
    if args.instru == "pad":
        base = Pad

    instru = DreamSynth(base, 0.2, args.gain, args.feedback, args.wet) 
    listen = [model.conv1, model.conv2, model.fc1, model.fc2, model.fc3]
    record = Record(
        listen, 
        scale=args.scale, 
        instru=instru, 
        cumulate=args.cumulate
    )


    for epoch in tqdm(range(args.epochs), desc="Epoch"):
        model.train()
        n = len(loader.dataset)
        acc = 0.0
        pbar = tqdm(loader, desc="Batch")
        for batch_id, (X, l) in enumerate(pbar):
            if args.cuda:
                X, l = X.cuda(), l.cuda()
            
            optim.zero_grad()
            
            Y = model(X.float())
            loss = critertion(Y, l.long())

            loss.backward()
            # Record update call to register current state
            record.update(batch_id, len(loader))
            optim.step()

            L = torch.argmax(Y, axis=-1)
            acc += L.eq(l).sum().cpu().detach().item() / n

            pbar.set_postfix(acc=f"{acc:.2%}")

    """
    Save the current record as a wav file with a define note duration. The 
    duration may be tweeked to analyze with the audio at different speed and
    focus more in depth on certain parts of the training.
    """
    record.save(args.layer_duration, args.path)

    import matplotlib.pyplot as plt

    l = int(len(record.layers))
    n = int(len(record) / l)
    X = range(n)

    plt.figure()
    
    for layer in range(l):
        Y = record.history[layer::l]
        plt.scatter(X, Y)
    
    plt.legend(labels=[f"Layer_{layer}" for layer in range(l)])
    plt.show()