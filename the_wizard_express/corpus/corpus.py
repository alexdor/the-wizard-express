import sys
from abc import ABC, abstractclassmethod
from typing import List, Union

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


class Corpus(ABC):
    """
    Abstract class for all the corpus
    """

    @abstractclassmethod
    def get_corpus(self) -> List[str]:
        pass

    @abstractclassmethod
    def save_to_disk(self, file_location: str):
        pass

    def get_id(self) -> str:
        return self.__class__.__name__


class TrainTestDataset(ABC):
    def get_train_data(self) -> RawDataset:
        return self.dataset["train"]

    def get_test_data(self) -> RawDataset:
        return self.dataset["test"]

    def get_validation_data(self) -> RawDataset:
        return self.dataset["validation"]
