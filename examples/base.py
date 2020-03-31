"""base.py

The example script contains all example instruments used in the example 
scripts.
"""
import wavedream as wd 


"""
A bell like sound desinged using the wavedream api. The initialization must
follow this exact pattern because of the pybind11 wrapping. May be improved
latter.
"""
class Bell(wd.Instrument):    
    def __init__(self, volume: float) -> None:
        self.t = wd.Timbre([wd.Formant(wd.Oscillator.Style.SIN, 1.0, 0)])
        self.a = wd.ADSR(0.01, 0.2, 0.0, 0.0)
        wd.Instrument.__init__(self, self.t, self.a, volume)


class Pad(wd.Instrument):    
    def __init__(self, volume: float) -> None:
        self.t = wd.Timbre([wd.Formant(wd.Oscillator.Style.SIN, 1.0, 0)])
        self.a = wd.ADSR(0.2, 0.2, 0.0, 0.0)
        wd.Instrument.__init__(self, self.t, self.a, volume)


"""
This instrument uses a base instrument and adds a cloudy reverb for ambient
like sounds. If the custom instrument does not extend from the wd.Instrument,
it must reimplement the '__call__', 'note_on' and 'not_off'.
"""
class DreamSynth:
    def __init__(
        self,
        base: wd.Instrument, 
        volume: float, 
        gain: float = 0.2, 
        feedback: float = 0.95,
        wet: float = 0.9,
        n_reverbs: int = 1 
    ) -> None:
        self.bell = base(volume)
        self.reverbs = [
            wd.Reverb(48_000, gain, feedback, wet)
            for i in range(n_reverbs)
        ]

    def __call__(self, t: float) -> float:
        y = self.bell(t)
        for reverb in self.reverbs:
            y = reverb(y)
        return y

    def note_on(self, t: float, note: int) -> None:
        self.bell.note_on(t, note)

    def note_off(self, t: float, note: int) -> None:
        self.bell.note_off(t, note)