from _preprocessing.Preprocessor import Preprocessor
from _classifiers.base import TLClassifier
from sklearn.model_selection import BaseCrossValidator
import sklearn.model_selection as model_selection
from _tuners.base import TLTuner
from typing import Type, Union
from _utils.enumerations import *


class Hub(object):
    """
    Static class that contains the mapping from str
    to the corresponding classes.
    """

    @staticmethod
    def get_preprocessor_class(
        preprocessor_name: PreprocessorEnum,
    ) -> Type[Preprocessor]:
        if preprocessor_name == PreprocessorEnum.ZNormalizer:
            from _preprocessing.ZNormalizer import ZNormalizer

            return ZNormalizer
        if preprocessor_name == PreprocessorEnum.LabelEncoder:
            from _preprocessing.LabelEncoder import LabelEncoder

            return LabelEncoder
        if preprocessor_name == PreprocessorEnum.DatasetSampler:
            from _preprocessing.DatasetSampler import DatasetSampler

            return DatasetSampler
        if preprocessor_name == PreprocessorEnum.Interpolate:
            from _preprocessing.Interpolate import Interpolate

            return Interpolate
        if preprocessor_name == PreprocessorEnum.GaussianPadding:
            from _preprocessing.GaussianPadding import GaussianPadding

            return GaussianPadding
        if preprocessor_name == PreprocessorEnum.Truncate:
            from _preprocessing.Truncate import Truncate

            return Truncate

        if preprocessor_name == PreprocessorEnum.RemoveNaN:
            from _preprocessing.RemoveNaN import RemoveNaN

            return RemoveNaN

        if preprocessor_name == PreprocessorEnum.FillNaN:
            from _preprocessing.FillNaN import FillNaN

            return FillNaN

        if preprocessor_name == PreprocessorEnum.TimeMatchNormalize:
            from _preprocessing.TimeMatchNormalize import TimeMatchNormalize

            return TimeMatchNormalize

        if preprocessor_name == PreprocessorEnum.SubSample:
            from _preprocessing.SubSample import SubSample

            return SubSample

        if preprocessor_name == PreprocessorEnum.Resampler:
            from _preprocessing.Resampler import Resampler

            return Resampler

        if preprocessor_name == PreprocessorEnum.ToPyArrow:
            from _preprocessing.ToPyArrow import ToPyArrow

            return ToPyArrow

        raise Exception(f"Unsupported {preprocessor_name}.")

    @staticmethod
    def get_tl_classifier_class(
        classifier_name: TLClassifierEnum,
    ) -> Type[TLClassifier]:
        if classifier_name == TLClassifierEnum.MLPClassifier:
            from _classifiers.hf.MLPClassifier import MLPClassifier

            return MLPClassifier
        if classifier_name == TLClassifierEnum.RandomForest:
            from _classifiers.sk.RandomForest import RandomForest

            return RandomForest
        if classifier_name == TLClassifierEnum.OTDA:
            from _classifiers.sk.OTDA import OTDA

            return OTDA
        if classifier_name == TLClassifierEnum.SVM:
            from _classifiers.sk.SVM import SVM

            return SVM

        if classifier_name == TLClassifierEnum.Inception:
            from _classifiers.hf.Inception import Inception

            return Inception

        if classifier_name == TLClassifierEnum.InceptionDANN:
            from _classifiers.hf.dann.InceptionDANN import InceptionDANN

            return InceptionDANN

        if classifier_name == TLClassifierEnum.InceptionMix:
            from _classifiers.hf.cotmix.InceptionMix import InceptionMix

            return InceptionMix

        if classifier_name == TLClassifierEnum.VRADA:
            from _classifiers.hf.dann.VRADA import VRADA

            return VRADA

        if classifier_name == TLClassifierEnum.CoDATS:
            from _classifiers.hf.dann.CoDATS import CoDATS

            return CoDATS

        if classifier_name == TLClassifierEnum.CoTMix:
            from _classifiers.hf.cotmix.CoTMix import CoTMix

            return CoTMix

        if classifier_name == TLClassifierEnum.KNNDTW:
            from _classifiers.sk.KNNDTW import KNNDTW

            return KNNDTW

        if classifier_name == TLClassifierEnum.DummyClf:
            from _classifiers.sk.DummyClf import DummyClf

            return DummyClf

        if classifier_name == TLClassifierEnum.Raincoat:
            from _classifiers.hf.raincoat.Raincoat import Raincoat

            return Raincoat

        if classifier_name == TLClassifierEnum.InceptionRain:
            from _classifiers.hf.raincoat.InceptionRain import InceptionRain

            return InceptionRain

        if classifier_name == TLClassifierEnum.InceptionCDAN:
            from _classifiers.hf.cdan.InceptionCDAN import InceptionCDAN

            return InceptionCDAN

        if classifier_name == TLClassifierEnum.SASA:
            from _classifiers.hf.sasa.SASA import SASA

            return SASA

        if classifier_name == TLClassifierEnum.InceptionSASA:
            from _classifiers.hf.sasa.InceptionSASA import InceptionSASA

            return InceptionSASA

        raise Exception(f"Unsupported {classifier_name}.")

    @staticmethod
    def get_tl_tuner_class(tuner_name: Union[SearchMethodNoneEnum, TLTunerEnum]) -> Type[TLTuner]:
        if tuner_name == SearchMethodNoneEnum.none:
            from _tuners.NoTune import NoTune

            return NoTune

        if tuner_name == TLTunerEnum.TransportedSource:
            from _tuners.TransportedSource import TransportedSource

            return TransportedSource

        if tuner_name == TLTunerEnum.TransportedTarget:
            from _tuners.TransportedTarget import TransportedTarget

            return TransportedTarget

        if tuner_name == TLTunerEnum.Reverse:
            raise NotImplementedError

        if tuner_name == TLTunerEnum.Reweight_5:
            from _tuners.reweight.Reweight_5 import Reweight_5

            return Reweight_5

        if tuner_name == TLTunerEnum.Reweight_3:
            from _tuners.reweight.Reweight_3 import Reweight_3

            return Reweight_3

        raise Exception(f"Unsupported {tuner_name}.")

    @staticmethod
    def get_model_selection(name: ModelSelectionEnum) -> Type[BaseCrossValidator]:
        if name == ModelSelectionEnum.NoShuffleSplit:
            from _classifiers.base import NoShuffleSplit

            return NoShuffleSplit
        return getattr(model_selection, name)
