"""lenet5.py

The scipt is an example of usage for the deepwavedream library applied to the
training of a VGG11 convolutional deep learning model on the CIFAR10 dataset
using an SGD optimizer with Pytorch.

python3 -m examples.lenet5 --help 
"""
from deepwavedream.utils import quantize_to_nearest
from tqdm import tqdm
from .base import *

import deepwavedream as dwd
import numpy as np
import torch
import torch.nn as nn


class VGG11(nn.Module):
    def __init__(self) -> None:
        super(VGG11, self).__init__()
        self.conv1 = nn.Conv2d(  3,  64, 3, padding=1)
        self.conv2 = nn.Conv2d( 64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv6 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv7 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv8 = nn.Conv2d(512, 512, 3, padding=1)
        
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512,  10)

        self.drop1 = nn.Dropout() 
        self.drop2 = nn.Dropout() 

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = torch.max_pool2d(torch.relu(self.conv1(X)), 2, stride=2)
        X = torch.max_pool2d(torch.relu(self.conv2(X)), 2, stride=2)
        X = torch.relu(self.conv3(X))
        X = torch.max_pool2d(torch.relu(self.conv4(X)), 2, stride=2)
        X = torch.relu(self.conv5(X))
        X = torch.max_pool2d(torch.relu(self.conv6(X)), 2, stride=2)
        X = torch.relu(self.conv7(X))
        X = torch.max_pool2d(torch.relu(self.conv8(X)), 2, stride=2)
        
        X = X.view(X.size(0), -1)
        
        X = self.drop1(torch.relu(self.fc1(X)))
        X = self.drop2(torch.relu(self.fc2(X)))
        X = torch.relu(self.fc3(X))
        
        return X


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
    from argparse import ArgumentParser
    from torch.optim import SGD
    from torch.utils.data import DataLoader
    from torchvision.datasets import cifar
    from torchvision.transforms import Compose
    from torchvision.transforms import Normalize
    from torchvision.transforms import RandomHorizontalFlip
    from torchvision.transforms import RandomCrop
    from torchvision.transforms import ToTensor


    parser = ArgumentParser(description="Train and record LeNet5 on MNIST.")
    parser.add_argument("--epochs",         "-e", default=2,      type=int)
    parser.add_argument("--batch_size",     "-b", default=256,    type=int)
    parser.add_argument("--learning_rate",  "-l", default=5e-2,   type=float)
    parser.add_argument("--layer_duration", "-d", default=1e-1,   type=float)
    parser.add_argument("--cumulate",             default=1,      type=int)
    parser.add_argument("--n_layers",             default=11,     type=int)
    
    parser.add_argument("--scale",          "-s", default="all",  type=str)
    parser.add_argument("--key",            "-k", default=69,     type=int)

    parser.add_argument("--instru",         "-i", default="bell", type=str)
    parser.add_argument("--gain",           "-g", default=0.2,    type=float)
    parser.add_argument("--feedback",       "-f", default=0.95,   type=float)
    parser.add_argument("--wet",            "-w", default=0.9,    type=float)
    parser.add_argument("--n_reverbs",      "-n", default=1,      type=int)
    
    parser.add_argument("--path",           "-p", required=True,  type=str)
    parser.add_argument("--cuda",           "-c", action="store_true")
    args = parser.parse_args()


    normalize = Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
    transforms = {
        "train": Compose([
            RandomHorizontalFlip(),
            RandomCrop(32, 4),
            ToTensor(),
            normalize,
        ]),
        "valid": Compose([ToTensor(), normalize]),
    }

    loader = {
        "train": DataLoader(
            cifar.CIFAR10(
                "./dataset", True, 
                transform=transforms["train"], download=True
            ), 
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4
        ),
        "valid": DataLoader(
            cifar.CIFAR10(
                "./dataset", False, 
                transform=transforms["valid"], download=True
            ), 
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4
        ),
    }


    if args.cuda:
        model = VGG11().cuda()
        critertion = nn.CrossEntropyLoss().cuda()
    else:
        model = VGG11()
        critertion = nn.CrossEntropyLoss()
    optim = SGD(
        model.parameters(), 
        lr=args.learning_rate, 
        momentum=0.9, 
        weight_decay=5e-4,
    )


    """
    Record initialization by defining the instrument to use as well as the 
    layers to be recorded. Layer order is important for the sonification as the
    information is passed to the process function and may be used in some way.
    """
    base   = Bell
    if args.instru == "pad":
        base = Pad

    instru = DreamSynth(
        base, 0.2, 
        args.gain, args.feedback, args.wet, args.n_reverbs,
    ) 

    n_layers = max(min(args.n_layers, 11), 1)
    listen = [
        model.conv1, 
        model.conv2,
        model.conv3, model.conv4, 
        model.conv5, model.conv6, 
        model.conv7, model.conv8, 
        model.fc1, model.fc2, model.fc3
    ][-n_layers:]
    
    record = Record(
        listen, 
        scale=args.scale, 
        key=args.key,
        instru=instru, 
        cumulate=args.cumulate
    )

    metrics = { "acc": { "train": [], "valid": [] } }
    for epoch in tqdm(range(args.epochs), desc="Epoch"):
        model.train()
        n = len(loader["train"].dataset)
        acc = 0.0
        pbar = tqdm(loader["train"], desc="Train")
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
        metrics["acc"]["train"].append(acc)

        model.eval()
        n = len(loader["valid"].dataset)
        acc = 0.0
        pbar = tqdm(loader["valid"], desc="Valid")
        for batch_id, (X, l) in enumerate(pbar):
            if args.cuda:
                X, l = X.cuda(), l.cuda()
            
            Y = model(X.float())
            L = torch.argmax(Y, axis=-1)
            acc += L.eq(l).sum().cpu().detach().item() / n

            pbar.set_postfix(acc=f"{acc:.2%}")
        metrics["acc"]["valid"].append(acc)

    """
    Save the current record as a wav file with a define note duration. The 
    duration may be tweeked to analyze with the audio at different speed and
    focus more in depth on certain parts of the training.
    """
    record.save(args.layer_duration, args.path)

    import matplotlib.pyplot as plt

    l = int(len(record.layers))
    n = int(len(record) / l)
    X = list(range(n))
    s = n / args.epochs

    plt.figure()
    
    plt.subplot(311)
    plt.ylabel = "Midi Note"
    plt.xlabel = "Batch"
    for x in X:
        if x % s == 0: plt.axvline(x=x, c="gray", alpha=0.3)
    for layer in range(l):
        Y = record.history[layer::l]
        plt.scatter(X, Y, marker="_")
    plt.legend(labels=[f"Layer_{layer}" for layer in range(l)])
    
    plt.subplot(312)
    plt.ylabel = "Gradient Norm"
    plt.xlabel = "Batch"
    for x in X:
        if x % s == 0: plt.axvline(x=x, c="gray", alpha=0.3)
    for layer in range(l):
        Y = record.raw_history[layer::l]
        plt.plot(X, Y)
    plt.legend(labels=[f"Layer_{layer}" for layer in range(l)])

    X = list(range(args.epochs))
    plt.subplot(313)
    plt.ylabel = "Accuracy"
    plt.xlabel = "Epoch"
    for e in range(args.epochs):
        plt.axvline(x=e, c="gray", alpha=0.3)
    plt.plot(X, metrics["acc"]["train"])
    plt.plot(X, metrics["acc"]["valid"])
    plt.legend(labels=["train", "valid"])

    plt.tight_layout()
    plt.show()