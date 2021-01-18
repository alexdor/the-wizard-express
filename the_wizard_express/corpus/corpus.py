import sys
from abc import ABC, abstractclassmethod
from operator import itemgetter
from typing import List, Optional, Tuple, Union

from datasets import Dataset

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict


# TODO: Find a better name for this
# TODO: Fix the type of this
class DataType(TypedDict):
    question: str
    answer: str
    context: str


RawDataset = Union[List[DataType], Dataset]


class DatasetDict(TypedDict):
    train: Dataset
    test: Dataset
    validation: Dataset


class Corpus(ABC):
    """
    Abstract class for all the corpus
    """

    @property
    def corpus(self):
        if not self._corpus:
            self._build_corpus()
        return self._corpus

    _corpus: Tuple[str] = tuple()
    percent_of_data_to_keep = 1.0

    @abstractclassmethod
    def _build_corpus(self) -> None:
        pass

    @abstractclassmethod
    def save_to_disk(self, file_location: str) -> None:
        pass

    def get_docs_by_index(self, indexes: List[int]) -> Tuple[str]:
        return itemgetter(*indexes)(Corpus.corpus.__get__(self))

    def get_id(self) -> str:
        return f"{self.__class__.__name__}{self.percent_of_data_to_keep}"


class TrainTestDataset(ABC):
    dataset: DatasetDict = None

    def get_train_data(self) -> RawDataset:
        return self.dataset["train"]

    def get_test_data(self) -> RawDataset:
        return self.dataset["test"]

    def get_validation_data(self) -> RawDataset:
        return self.dataset["validation"]
