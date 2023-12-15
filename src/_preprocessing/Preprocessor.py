from pydantic import BaseModel
import abc, numpy as np
from datasets import Dataset
from typing import TypeVar, Dict


TPreprocessor = TypeVar("TPreprocessor", bound="Preprocessor")


class Preprocessor(BaseModel):
    @abc.abstractmethod
    def fit(self, X: Dataset) -> TPreprocessor:
        """
        Fits the current Preprocessor and returns it fitted
        """
        pass

    @abc.abstractmethod
    def transform(self, X: Dataset) -> Dataset:
        """
        Preprocess a Dataset from huggingface and return
        a new one
        """
        pass
