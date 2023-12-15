from enum import Enum


class TLTunerEnum(str, Enum):
    TransportedSource: str = "TransportedSource"
    TransportedTarget: str = "TransportedTarget"
    Reverse: str = "Reverse"
    Reweight_5: str = "Reweight_5"
    Reweight_3: str = "Reweight_3"


class DatasetNameEnum(str, Enum):
    hhar: str = "hhar"
    ucr2018: str = "ucr2018"
    ford: str = "ford"
    har: str = "har"
    sportsActivities: str = "sportsActivities"
    OnHWeq: str = "OnHWeq"
    ultrasoundMuscleContraction: str = "ultrasoundMuscleContraction"
    sleepStage: str = "sleepStage"
    ptbXLecg: str = "ptbXLecg"
    wisdm: str = "wisdm"
    mfd: str = "mfd"
    cwrBearing: str = "cwrBearing"
    miniTimeMatch: str = "miniTimeMatch"


class DomainNameEnum(str, Enum):
    source: str = "source"
    target: str = "target"


class TLClassifierEnum(str, Enum):
    MLPClassifier = "MLPClassifier"
    RandomForest = "RandomForest"
    OTDA = "OTDA"
    SVM = "SVM"
    CoDATS = "CoDATS"
    CoTMix = "CoTMix"
    Inception = "Inception"
    InceptionDANN = "InceptionDANN"
    InceptionCDAN = "InceptionCDAN"
    InceptionMix = "InceptionMix"
    SASA = "SASA"
    VRADA = "VRADA"
    DummyClf = "DummyClf"
    KNNDTW = "KNNDTW"
    Raincoat = "Raincoat"
    InceptionRain = "InceptionRain"
    InceptionSASA = "InceptionSASA"


class PreprocessorEnum(str, Enum):
    ZNormalizer = "ZNormalizer"
    LabelEncoder = "LabelEncoder"
    DatasetSampler = "DatasetSampler"
    GaussianPadding = "GaussianPadding"
    Truncate = "Truncate"
    Interpolate = "Interpolate"
    SubSample = "SubSample"
    RemoveNaN = "RemoveNaN"
    FillNaN = "FillNaN"
    TimeMatchNormalize = "TimeMatchNormalize"
    Resampler = "Resampler"
    ToPyArrow = "ToPyArrow"


class MetricKeysEnum(str, Enum):
    accuracy: str = "accuracy"
    f1_macro: str = "f1_macro"
    f1_micro: str = "f1_micro"
    precision_micro: str = "precision_micro"
    precision_macro: str = "precision_macro"
    recall_micro: str = "recall_micro"
    recall_macro: str = "recall_macro"
    reweight_5_loss: str = "reweight_5_loss"
    reweight_5_target_density_loss: str = "reweight_5_target_density_loss"


class TrainTuneOptionsEnum(str, Enum):
    tune_configs_only: str = "tune_configs_only"
    train_configs_only: str = "train_configs_only"
    tune_overrides_train: str = "tune_overrides_train"  # intersection allowed
    train_overrides_tune: str = "train_overrides_tune"  # intersection allowed
    train_union_tune: str = "train_union_tune"  # intersection not allowed


class SearchMethodNoneEnum(str, Enum):
    none = "None"


class DataSplitEnum(str, Enum):
    train = "train"
    test = "test"


class TorchDeviceEnum(str, Enum):
    cpu = "cpu"
    gpu = "cuda"


class RayRessourcesEnum(str, Enum):
    cpu = "cpu"
    gpu = "gpu"


class EnvVariables(str, Enum):
    DVC_CACHE_DIR = "DVC_CACHE_DIR"


class DatasetColumnsEnum(str, Enum):
    mts = "mts"
    labels = "labels"
    ori_mts = "ori_mts"


class AverageEnum(str, Enum):
    average = "average"


class ModelSelectionEnum(str, Enum):
    ShuffleSplit = "ShuffleSplit"
    StratifiedKFold = "StratifiedKFold"
    KFold = "KFold"
    NoShuffleSplit = "NoShuffleSplit"


class CheckpointLoadEnum(str, Enum):
    best = "best"
    last = "last"


class BestCheckpointEnum(str, Enum):
    maximum = "maximum"
    minimum = "minimum"


class DomainEnumInt(int, Enum):
    source: int = 0
    target: int = 1


class AdditionalColumnEnum(str, Enum):
    domain_y: str = "domain_y"


class InjectConfigEnum(str, Enum):
    backbone: str = "backbone"
    num_labels: str = "num_labels"
    n_input_channels: str = "n_input_channels"
    classifier: str = "classifier"
    input_len: str = "input_len"
