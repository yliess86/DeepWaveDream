from typing import List

import numpy as np
import wavedream as wd


NOTES = list(range(len(wd.NOTES)))
def quantize_to_nearest(freq: float, notes: List[int] = NOTES) -> int:
    X = np.array([wd.NOTES[note] for note in notes])
    return (np.abs(X - freq)).argmin()