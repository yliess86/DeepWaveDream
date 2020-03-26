import wavedream as wd 


class Bell(wd.Instrument):
    def __init__(self, volume: float) -> None:
        self.t = wd.Timbre([wd.Formant(wd.Oscillator.Style.SIN, 1.0, 0)])
        self.a = wd.ADSR(0.01, 0.2, 0.0, 0.0)
        wd.Instrument.__init__(self, self.t, self.a, volume)


class DreamBell:
    def __init__(self, volume: float) -> None:
        self.bell = Bell(volume)
        self.reverb = wd.Reverb(48_000, 0.2, 0.95, 0.9)

    def __call__(self, t: float) -> float:
        return self.reverb(self.bell(t))

    def note_on(self, t: float, note: int) -> None:
        self.bell.note_on(t, note)

    def note_off(self, t: float, note: int) -> None:
        self.bell.note_off(t, note)