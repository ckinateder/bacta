from enum import Enum


class Order(Enum):
    BUY = 1
    SELL = -1


class Position(Enum):
    LONG = 1
    SHORT = -1
    NEUTRAL = 0
