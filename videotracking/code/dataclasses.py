from dataclasses import dataclass

@dataclass
class HOGParameters:
    winSize: tuple
    blockSize: tuple
    blockStride: tuple
    cellSize: tuple
    nbins: int
    winStride: tuple
    padding: tuple
    locations: tuple