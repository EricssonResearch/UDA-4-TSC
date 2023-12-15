"""The UCR Time Series Archive for time series classification."""


import os
import csv
from enum import Enum
import datasets
from typing import Dict
from _utils.enumerations import DatasetColumnsEnum


_CITATION = """\
@article{dau2019ucr,
  title={The UCR time series archive},
  author={Dau, Hoang Anh and Bagnall, Anthony and Kamgar, Kaveh and Yeh, Chin-Chia Michael and Zhu, Yan and Gharghabi, Shaghayegh and Ratanamahatana, Chotirat Ann and Keogh, Eamonn},
  journal={IEEE/CAA Journal of Automatica Sinica},
  volume={6},
  number={6},
  pages={1293--1305},
  year={2019},
  publisher={IEEE}
}
"""

_DESCRIPTION = """\
This archive contains 128 univariate time series classification datasets coming from 
various domains such Images, Spectrograph, Medical, Audi, Speech, Traffic, Sensors ... 
"""

_HOMEPAGE = "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/"

_LICENSE = "N/A"

_URLs = {
    "ucr2018": "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCRArchive_2018.zip",
}

num_labels = {
    "ShapesAll": 60,
    "PigAirwayPressure": 52,
    "PigArtPressure": 52,
    "PigCVP": 52,
    "FiftyWords": 50,
    "NonInvasiveFetalECGThorax1": 42,
    "NonInvasiveFetalECGThorax2": 42,
    "Phoneme": 39,
    "PhonemeSpectra": 39,
    "Adiac": 37,
    "GestureMidAirD1": 26,
    "GestureMidAirD2": 26,
    "GestureMidAirD3": 26,
    "Handwriting": 26,
    "ArticularyWordRecognition": 25,
    "WordSynonyms": 25,
    "Crop": 24,
    "CharacterTrajectories": 20,
    "Fungi": 18,
    "SwedishLeaf": 15,
    "Libras": 15,
    "LSST": 14,
    "FaceAll": 14,
    "FacesUCR": 14,
    "CricketX": 12,
    "CricketY": 12,
    "CricketZ": 12,
    "Cricket": 12,
    "EOGHorizontalSignal": 12,
    "EOGVerticalSignal": 12,
    "PLAID": 11,
    "MelbournePedestrian": 10,
    "SpokenArabicDigits": 10,
    "PickupGestureWiimoteZ": 10,
    "ShakeGestureWiimoteZ": 10,
    "Siemens": 10,
    "AllGestureWiimoteX": 10,
    "AllGestureWiimoteY": 10,
    "AllGestureWiimoteZ": 10,
    "PenDigits": 10,
    "MedicalImages": 10,
    "ACSF1": 10,
    "InsectSound": 10,
    "InsectWingbeat": 10,
    "UrbanSound": 10,
    "Tiselac": 9,
    "JapaneseVowels": 9,
    "Mallat": 8,
    "UWaveGestureLibraryAll": 8,
    "UWaveGestureLibraryX": 8,
    "UWaveGestureLibraryY": 8,
    "UWaveGestureLibraryZ": 8,
    "UWaveGestureLibrary": 8,
    "DodgerLoopDay": 7,
    "Lightning7": 7,
    "Plane": 7,
    "InlineSkate": 7,
    "PEMS-SF": 7,
    "Fish": 7,
    "ElectricDevices": 7,
    "SemgHandMovementCh2": 6,
    "SyntheticControl": 6,
    "GesturePebbleZ1": 6,
    "GesturePebbleZ2": 6,
    "Colposcopy": 6,
    "DistalPhalanxTW": 6,
    "MiddlePhalanxTW": 6,
    "OSULeaf": 6,
    "ProximalPhalanxTW": 6,
    "Symbols": 6,
    "ERing": 6,
    "MotionSenseHAR": 6,
    "NATOPS": 6,
    "MosquitoSound": 6,
    "Beef": 5,
    "SemgHandSubjectCh2": 5,
    "EigenWorms": 5,
    "Haptics": 5,
    "Worms": 5,
    "MindReading": 5,
    "MixedShapes": 5,
    "MixedShapesSmallTrain": 5,
    "ECG5000": 5,
    "AbnormalHeartbeat": 5,
    "DuckDuckGeese": 5,
    "DucksAndGeese": 5,
    "EthanolLevel": 4,
    "OliveOil": 4,
    "Rock": 4,
    "TwoPatterns": 4,
    "Car": 4,
    "Trace": 4,
    "EthanolConcentration": 4,
    "AsphaltObstacles": 4,
    "AsphaltObstaclesCoordinates": 4,
    "DiatomSizeReduction": 4,
    "FaceFour": 4,
    "BasicMotions": 4,
    "Epilepsy": 4,
    "RacketSports": 4,
    "HandMovementDirection": 4,
    "CinCECGTorso": 4,
    "Meat": 3,
    "BME": 3,
    "CBF": 3,
    "ChlorineConcentration": 3,
    "SmoothSubspace": 3,
    "UMD": 3,
    "StarLightCurves": 3,
    "AsphaltPavementType": 3,
    "AsphaltPavementTypeCoordinates": 3,
    "ArrowHead": 3,
    "DistalPhalanxOutlineAgeGroup": 3,
    "MiddlePhalanxOutlineAgeGroup": 3,
    "ProximalPhalanxOutlineAgeGroup": 3,
    "CounterMovementJump": 3,
    "EMOPain": 3,
    "InsectEPGRegularTrain": 3,
    "InsectEPGSmallTrain": 3,
    "AtrialFibrillation": 3,
    "StandWalkJump": 3,
    "LargeKitchenAppliances": 3,
    "RefrigerationDevices": 3,
    "ScreenType": 3,
    "SmallKitchenAppliances": 3,
    "FruitFlies": 3,
    "Chinatown": 2,
    "Coffee": 2,
    "Ham": 2,
    "SemgHandGenderCh2": 2,
    "Strawberry": 2,
    "Wine": 2,
    "ShapeletSim": 2,
    "DodgerLoopGame": 2,
    "DodgerLoopWeekend": 2,
    "Earthquakes": 2,
    "ElectricDeviceDetection": 2,
    "FordA": 2,
    "FordB": 2,
    "FreezerRegularTrain": 2,
    "FreezerSmallTrain": 2,
    "ItalyPowerDemand": 2,
    "Lightning2": 2,
    "MoteStrain": 2,
    "SonyAIBORobotSurface1": 2,
    "SonyAIBORobotSurface2": 2,
    "Wafer": 2,
    "AsphaltRegularity": 2,
    "AsphaltRegularityCoordinates": 2,
    "GunPoint": 2,
    "GunPointAgeSpan": 2,
    "GunPointMaleVersusFemale": 2,
    "GunPointOldVersusYoung": 2,
    "ToeSegmentation1": 2,
    "ToeSegmentation2": 2,
    "WormsTwoClass": 2,
    "BeetleFly": 2,
    "BirdChicken": 2,
    "DistalPhalanxOutlineCorrect": 2,
    "HandOutlines": 2,
    "Herring": 2,
    "MiddlePhalanxOutlineCorrect": 2,
    "PhalangesOutlinesCorrect": 2,
    "ProximalPhalanxOutlineCorrect": 2,
    "Yoga": 2,
    "SharePriceIncrease": 2,
    "EyesOpenShut": 2,
    "FaceDetection": 2,
    "FingerMovements": 2,
    "MotorImagery": 2,
    "SelfRegulationSCP1": 2,
    "SelfRegulationSCP2": 2,
    "ECG200": 2,
    "ECGFiveDays": 2,
    "TwoLeadECG": 2,
    "Computers": 2,
    "HouseTwenty": 2,
    "PowerCons": 2,
    "BinaryHeartbeat": 2,
    "CatsDogs": 2,
    "Heartbeat": 2,
    "RightWhaleCalls": 2,
}


class UCR2018DatasetNameEnum:
    ACSF1: str = "ACSF1"
    Adiac: str = "Adiac"
    AllGestureWiimoteX: str = "AllGestureWiimoteX"
    AllGestureWiimoteY: str = "AllGestureWiimoteY"
    AllGestureWiimoteZ: str = "AllGestureWiimoteZ"
    ArrowHead: str = "ArrowHead"
    Beef: str = "Beef"
    BeetleFly: str = "BeetleFly"
    BirdChicken: str = "BirdChicken"
    BME: str = "BME"
    Car: str = "Car"
    CBF: str = "CBF"
    Chinatown: str = "Chinatown"
    ChlorineConcentration: str = "ChlorineConcentration"
    CinCECGTorso: str = "CinCECGTorso"
    Coffee: str = "Coffee"
    Computers: str = "Computers"
    CricketX: str = "CricketX"
    CricketY: str = "CricketY"
    CricketZ: str = "CricketZ"
    Crop: str = "Crop"
    DiatomSizeReduction: str = "DiatomSizeReduction"
    DistalPhalanxOutlineAgeGroup: str = "DistalPhalanxOutlineAgeGroup"
    DistalPhalanxOutlineCorrect: str = "DistalPhalanxOutlineCorrect"
    DistalPhalanxTW: str = "DistalPhalanxTW"
    DodgerLoopDay: str = "DodgerLoopDay"
    DodgerLoopGame: str = "DodgerLoopGame"
    DodgerLoopWeekend: str = "DodgerLoopWeekend"
    Earthquakes: str = "Earthquakes"
    ECG200: str = "ECG200"
    ECG5000: str = "ECG5000"
    ECGFiveDays: str = "ECGFiveDays"
    ElectricDevices: str = "ElectricDevices"
    EOGHorizontalSignal: str = "EOGHorizontalSignal"
    EOGVerticalSignal: str = "EOGVerticalSignal"
    EthanolLevel: str = "EthanolLevel"
    FaceAll: str = "FaceAll"
    FaceFour: str = "FaceFour"
    FacesUCR: str = "FacesUCR"
    FiftyWords: str = "FiftyWords"
    Fish: str = "Fish"
    FordA: str = "FordA"
    FordB: str = "FordB"
    FreezerRegularTrain: str = "FreezerRegularTrain"
    FreezerSmallTrain: str = "FreezerSmallTrain"
    Fungi: str = "Fungi"
    GestureMidAirD1: str = "GestureMidAirD1"
    GestureMidAirD2: str = "GestureMidAirD2"
    GestureMidAirD3: str = "GestureMidAirD3"
    GesturePebbleZ1: str = "GesturePebbleZ1"
    GesturePebbleZ2: str = "GesturePebbleZ2"
    GunPoint: str = "GunPoint"
    GunPointAgeSpan: str = "GunPointAgeSpan"
    GunPointMaleVersusFemale: str = "GunPointMaleVersusFemale"
    GunPointOldVersusYoung: str = "GunPointOldVersusYoung"
    Ham: str = "Ham"
    HandOutlines: str = "HandOutlines"
    Haptics: str = "Haptics"
    Herring: str = "Herring"
    HouseTwenty: str = "HouseTwenty"
    InlineSkate: str = "InlineSkate"
    InsectEPGRegularTrain: str = "InsectEPGRegularTrain"
    InsectEPGSmallTrain: str = "InsectEPGSmallTrain"
    InsectWingbeatSound: str = "InsectWingbeatSound"
    ItalyPowerDemand: str = "ItalyPowerDemand"
    LargeKitchenAppliances: str = "LargeKitchenAppliances"
    Lightning2: str = "Lightning2"
    Lightning7: str = "Lightning7"
    Mallat: str = "Mallat"
    Meat: str = "Meat"
    MedicalImages: str = "MedicalImages"
    MelbournePedestrian: str = "MelbournePedestrian"
    MiddlePhalanxOutlineAgeGroup: str = "MiddlePhalanxOutlineAgeGroup"
    MiddlePhalanxOutlineCorrect: str = "MiddlePhalanxOutlineCorrect"
    MiddlePhalanxTW: str = "MiddlePhalanxTW"
    MixedShapesRegularTrain: str = "MixedShapesRegularTrain"
    MixedShapesSmallTrain: str = "MixedShapesSmallTrain"
    MoteStrain: str = "MoteStrain"
    NonInvasiveFetalECGThorax1: str = "NonInvasiveFetalECGThorax1"
    NonInvasiveFetalECGThorax2: str = "NonInvasiveFetalECGThorax2"
    OliveOil: str = "OliveOil"
    OSULeaf: str = "OSULeaf"
    PhalangesOutlinesCorrect: str = "PhalangesOutlinesCorrect"
    Phoneme: str = "Phoneme"
    PickupGestureWiimoteZ: str = "PickupGestureWiimoteZ"
    PigAirwayPressure: str = "PigAirwayPressure"
    PigArtPressure: str = "PigArtPressure"
    PigCVP: str = "PigCVP"
    PLAID: str = "PLAID"
    Plane: str = "Plane"
    PowerCons: str = "PowerCons"
    ProximalPhalanxOutlineAgeGroup: str = "ProximalPhalanxOutlineAgeGroup"
    ProximalPhalanxOutlineCorrect: str = "ProximalPhalanxOutlineCorrect"
    ProximalPhalanxTW: str = "ProximalPhalanxTW"
    RefrigerationDevices: str = "RefrigerationDevices"
    Rock: str = "Rock"
    ScreenType: str = "ScreenType"
    SemgHandGenderCh2: str = "SemgHandGenderCh2"
    SemgHandMovementCh2: str = "SemgHandMovementCh2"
    SemgHandSubjectCh2: str = "SemgHandSubjectCh2"
    ShakeGestureWiimoteZ: str = "ShakeGestureWiimoteZ"
    ShapeletSim: str = "ShapeletSim"
    ShapesAll: str = "ShapesAll"
    SmallKitchenAppliances: str = "SmallKitchenAppliances"
    SmoothSubspace: str = "SmoothSubspace"
    SonyAIBORobotSurface1: str = "SonyAIBORobotSurface1"
    SonyAIBORobotSurface2: str = "SonyAIBORobotSurface2"
    StarLightCurves: str = "StarLightCurves"
    Strawberry: str = "Strawberry"
    SwedishLeaf: str = "SwedishLeaf"
    Symbols: str = "Symbols"
    SyntheticControl: str = "SyntheticControl"
    ToeSegmentation1: str = "ToeSegmentation1"
    ToeSegmentation2: str = "ToeSegmentation2"
    Trace: str = "Trace"
    TwoLeadECG: str = "TwoLeadECG"
    TwoPatterns: str = "TwoPatterns"
    UMD: str = "UMD"
    UWaveGestureLibraryAll: str = "UWaveGestureLibraryAll"
    UWaveGestureLibraryX: str = "UWaveGestureLibraryX"
    UWaveGestureLibraryY: str = "UWaveGestureLibraryY"
    UWaveGestureLibraryZ: str = "UWaveGestureLibraryZ"
    Wafer: str = "Wafer"
    Wine: str = "Wine"
    WordSynonyms: str = "WordSynonyms"
    Worms: str = "Worms"
    WormsTwoClass: str = "WormsTwoClass"
    Yoga: str = "Yoga"


class UCR2018Config(datasets.BuilderConfig):
    """BuilderConfig for UCR2018."""

    name: UCR2018DatasetNameEnum
    password: str

    def __init__(self, name: UCR2018DatasetNameEnum, password: str, **kwargs):
        """BuilderConfig for UCR2018.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(UCR2018Config, self).__init__(**kwargs)
        self.name = name
        self.password = password

        # password should be changed see the {_HOMEPAGE}


class UCR2018(datasets.GeneratorBasedBuilder):
    """UCR Time Series Archive 2018."""

    BUILDER_CONFIG_CLASS = UCR2018Config

    def _info(self):
        features = datasets.Features(
            {
                DatasetColumnsEnum.labels: datasets.Value("string"),
                DatasetColumnsEnum.mts: [[datasets.Value("double")]],
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        my_urls = _URLs["ucr2018"]
        archive = dl_manager.download(my_urls)

        out_extracted = archive.split("downloads")[0]
        out_extracted = f"{out_extracted}/ucr2018"
        done_out_extracted = f"{out_extracted}/DONE"

        if os.path.exists(done_out_extracted):
            print("Already extracted")
        else:
            from zipfile import ZipFile

            with ZipFile(archive) as zf:
                zf.extractall(pwd=self.config.password.encode("ascii"), path=out_extracted)
            os.mkdir(out_extracted + "/DONE")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": f"{out_extracted}/UCRArchive_2018/{self.config.name}/{self.config.name}_TRAIN.tsv"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": f"{out_extracted}/UCRArchive_2018/{self.config.name}/{self.config.name}_TEST.tsv"
                },
            ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        with open(filepath) as f:
            data = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            for id_, row in enumerate(data):
                yield id_, {
                    DatasetColumnsEnum.mts: [[float(x) for x in row[1:]]],
                    DatasetColumnsEnum.labels: str(row[0]),
                }
