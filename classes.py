from enum import Enum
import numpy as np
import numpy.typing as npt
import pandas as pd


class Label(Enum):
    ANNUALCROP = 0
    FOREST = 1
    HERBACEOUSVEGETATION = 2
    HIGHWAY = 3
    INDUSTRIAL = 4
    PASTURE = 5
    PERMANENTCROP = 6
    RESIDENTIAL = 7
    RIVER = 8
    SEALAKE = 9

    @classmethod
    def label_mappings(cls, encoded=False):
        decoded_mappings = {
            "AnnualCrop": 0,
            "Forest": 1,
            "HerbaceousVegetation": 2,
            "Highway": 3,
            "Industrial": 4,
            "Pasture": 5,
            "PermanentCrop": 6,
            "Residential": 7,
            "River": 8,
            "SeaLake": 9
        }
        if encoded:
            return {value: key for key, value in decoded_mappings.items()}
        return decoded_mappings


class Sample:
    __img_name: str
    __img_array: npt.NDArray[np.float64]
    __img_label: Label

    @property
    def img_name(self):
        return self.__img_name

    @property
    def img_array(self):
        return self.__img_array

    @property
    def img_label(self):
        return self.__img_label

    def __init__(self, img_name: str, img_array: npt.NDArray[np.float64], img_label: Label) -> None:
        self.__img_name = img_name
        self.__img_array = img_array
        self.__img_label = img_label

    @classmethod
    def reading_list(cls, df: pd.DataFrame) -> list:
        return list(map(lambda x: Sample(img_name=x[0], img_array=x[2], img_label=x[1]), df.values.tolist()))

    def __str__(self) -> str:
        return f"Image name: {self.__img_name} | Label: {self.__img_label}"

    def __repr__(self) -> str:
        return self.__str__()
