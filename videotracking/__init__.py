from videotracking.code.dataclasses import HOGParameters
from videotracking.code.feature_extractor import feature_extraction
import numpy as np

data_path = './videotracking/data/image0001.png'
np.set_printoptions(threshold=np.inf, precision=2, suppress=True, linewidth=np.inf)

hog_parameters = HOGParameters(
    winSize = None, # Especifícase para cada imaxe en feature_extractor.py
    blockSize = (16,16),
    blockStride = (8,8),
    cellSize = (8,8),
    nbins = 11,
    winStride = (8,8),
    padding = (8,8),
    locations = ((10,20),)
)

def start():
    feature_extraction(data_path, hog_parameters)