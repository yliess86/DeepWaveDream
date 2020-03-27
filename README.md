# DeepWaveDream

DeepWaveDream is an experimental library. It aimes at sonifying the gradient norm evolution of a training deep neural network either for debugging or artistic purposes. The library is based on the experimental [WaveDream](https://github.com/yliess86/WaveDream) virtual synthetizer library and support only [PyTorch](https://pytorch.org/) models.

## Analysis

**Coming soon ...**

## Install

To install the library you first need to install the wavedream library following the instruction on the repository: [WaveDream](https://github.com/yliess86/WaveDream). It consists of installing [PyBind11](https://github.com/pybind/pybind11), [SoundIO](http://libsound.io/) and running a `setup.py` script.

Then the library can be installed using the `setup.py` script as follow (may require sudo):
```bash
$ (sudo) python3 setup.py install
```

## Usage

Example usage are shown in the `examples` folder.
```bash
$ python3 -m examples.lenet5 --help
```

The usage consists of defining wavedream `Instrument`, extending the deepwavedream `Record` class by implementing a custom `process` function and finally by update the record during training and saving the final `wavefile`.

```python
from deepwavedream.utils import quantize_to_nearest

import wavedream as wd
import deepwavedream as dwd


class Bell(wd.Instrument):    
    def __init__(self, volume: float) -> None:
        self.t = wd.Timbre([wd.Formant(wd.Oscillator.Style.SIN, 1.0, 0)])
        self.a = wd.ADSR(0.01, 0.2, 0.0, 0.0)
        wd.Instrument.__init__(self, self.t, self.a, volume)


class Record(dwd.Record):
    def __init__(self, *args, **kwargs) -> None:
        super(Record, self).__init__(*args, **kwargs)

    def process(self, layer: int, norm_grad: float) -> int:
        freq = 110.0 + 100 * 2 ** layer + norm_grad * 200.0
        return quantize_to_nearest(freq)


bell = DreamBell(0.02) 
listen = [model.layer1, model.layer2]
record = Record(listen, instru=bell)

for epoch in epochs:
    for batch in batches:
        # ...
        
        loss.backward()
        record.update()
        optimizer.Step()

        # ...

record.save(0.1, "record.wav")
```