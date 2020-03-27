"""utils.py

The file contains helper functions to help the user right its own process
functions.
"""
from typing import List

import numpy as np
import wavedream as wd


NOTES = list(range(len(wd.NOTES)))
def quantize_to_nearest(freq: float, notes: List[int] = NOTES) -> int:
    """quantize to neartest note
    
    Arguments:
        freq {float} -- input frequency to quantize
    
    Keyword Arguments:
        notes {List[int]} -- notes available for quantization 
            (default: {NOTES})
    
    Returns:
        int -- quantized frequency to closest note
    """
    X = np.array([wd.NOTES[note] for note in notes])
    return (np.abs(X - freq)).argmin()