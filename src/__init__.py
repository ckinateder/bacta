from enum import Enum
import pandas as pd


class Position(Enum):
    LONG = 1
    SHORT = -1
    NEUTRAL = 0
