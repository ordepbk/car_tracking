import cv2
import numpy as np
from videotracking.code.dataclasses import HOGParameters

def img_BGR_readear(data_path):
    return cv2.cvtColor(cv2.imread(data_path), cv2.COLOR_BGR2GRAY)

def hog_extractor(img, hog_parameters: HOGParameters) -> np.ndarray:
    hog = cv2.HOGDescriptor(
        _winSize=hog_parameters.winSize,
        _blockSize=hog_parameters.blockSize,
        _blockStride=hog_parameters.blockStride,
        _cellSize=hog_parameters.cellSize,
        _nbins=hog_parameters.nbins
    )
    return hog.compute(
        img=img,
        winStride=hog_parameters.winStride,
        padding=hog_parameters.padding,
        locations=hog_parameters.locations
    )

def feature_extraction(data_path, hog_parameters: HOGParameters) -> None:
    img = img_BGR_readear(data_path)

    cv2.imshow('Imaxe de proba', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    features = hog_extractor(img, hog_parameters)
    print(np.max(features))