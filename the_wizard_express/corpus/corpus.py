import sys
from abc import ABC, abstractclassmethod
from errno import ENOENT
from operator import itemgetter
from os import strerror
from os.path import dirname, lexists
from pathlib import Path
from pickle import load
from typing import List, Tuple, Union

from datasets import Dataset
from the_wizard_express.utils.utils import generate_cache_path

from ..config import Config
from ..utils import DatasetDict, pickle_and_save_to_file

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


class Corpus(ABC):
    """
    Abstract class for all the corpus
    """

    _corpus: Tuple[str]
    _corpus_path: str

    def __init__(self) -> None:
        self._corpus_path = generate_cache_path("corpus", self)

    @property
    def corpus(self):
        if not hasattr(self, "_corpus") or len(self._corpus) == 0:
            if lexists(self._corpus_path):
                self._corpus = load(open(self._corpus_path, "rb"))
            else:
                self._build_corpus()
                pickle_and_save_to_file(self._corpus, self._corpus_path)
        return self._corpus

    @abstractclassmethod
    def _build_corpus(self) -> None:
        pass

    def save_to_disk(self, file_location: str) -> None:
        Path(dirname(file_location)).mkdir(parents=True, exist_ok=True)
        with open(file_location, "w+", encoding="utf-8") as f:
            f.writelines([f"{line}\n" for line in self.corpus])

    def get_docs_by_index(self, indexes: List[int]) -> Tuple[str]:
        return itemgetter(*indexes)(Corpus.corpus.__get__(self))

    def get_id(self) -> str:
        return f"{self.__class__.__name__}_{Config.percent_of_data_to_keep}"


class TrainTestDataset(ABC):
    _dataset: DatasetDict
    _dataset_path: str

    def __init__(self) -> None:
        self._dataset_path = generate_cache_path("dataset", self)

    @property
    def dataset(self) -> DatasetDict:
        if not hasattr(self, "_dataset") or not self._dataset:
            try:
                if lexists(self._dataset_path):
                    raise FileNotFoundError
                self._dataset = load(open(self._dataset_path, "rb"))
            except FileNotFoundError:
                self._build_dataset()
                pickle_and_save_to_file(self._dataset, self._dataset_path)
        return self._dataset

    @abstractclassmethod
    def _build_dataset(self) -> None:
        pass

    def get_train_data(self) -> RawDataset:
        return TrainTestDataset.dataset.__get__(self)["train"]

    def get_test_data(self) -> RawDataset:
        return TrainTestDataset.dataset.__get__(self)["test"]

    def get_validation_data(self) -> RawDataset:
        return TrainTestDataset.dataset.__get__(self)["validation"]
