import sys
from abc import ABC, abstractclassmethod
from json import dumps
from operator import itemgetter
from os.path import dirname, lexists
from pathlib import Path
from pickle import load
from typing import List, Tuple, Union

from datasets import Dataset
from numpy import sort, unique
from pyarrow import StringArray, array
from torch.utils.data import Dataset as TorchDataset

from ..config import Config
from ..utils import DatasetDict, generate_cache_path, pickle_and_save_to_file

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict


class DataTypeForWizardExpress(TypedDict):
    question: str
    answer: str
    context: str


RawDataset = Union[List[DataTypeForWizardExpress], Dataset]


class Corpus(ABC):
    """
    Abstract class for all the corpus
    """

    _corpus: Tuple[str]
    _corpus_path: str

    def __init__(self) -> None:
        self._corpus_path = generate_cache_path("corpus", self, skip_vocab_size=True)

    @property
    def corpus(self) -> StringArray:
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

    def save_to_disk(self, file_location: str, as_json=False) -> None:
        Path(dirname(file_location)).mkdir(parents=True, exist_ok=True)
        with open(file_location, "w+", encoding="utf-8") as f:
            if as_json:
                f.write(
                    dumps(
                        [
                            {"id": str(i), "contents": str(content)}
                            for i, content in enumerate(self.corpus)
                        ]
                    )
                )
                return
            f.writelines([f"{line}\n" for line in self.corpus])

    def get_docs_by_index(self, indexes: List[int]) -> Tuple[str]:
        return itemgetter(*indexes)(Corpus.corpus.__get__(self))

    def _transform_datasets_to_corpus(self, dataset, key="context"):
        dataset = dataset.map(
            lambda data: {key: data[key]},
            remove_columns=[
                "question",
                "answer",
            ],
            num_proc=Config.max_proc_to_use,
        )
        # Accessing private property here because huggingface's unique
        # throws an error
        dataset = array(sort(unique(dataset._data.column(key).to_numpy())))
        self._corpus = dataset

    def get_id(self) -> str:
        return f"{self.__class__.__name__}_{Config.percent_of_data_to_keep}"

    def corpus_iterator(self, batch_size=100):
        return (
            self.corpus[i : i + batch_size].to_pylist()
            for i in range(0, len(self.corpus), batch_size)
        )


class ParallelTrainData(TorchDataset):
    def __init__(self, data, retriever, tokenizer):
        self.data = data
        self.retriever = retriever
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        question = self.data[idx]["question"]
        context = self.retriever.retrieve_docs(question, 5)
        item = self.tokenizer(
            question,
            "\n".join(context),
            add_special_tokens=True,
            padding="max_length",
            max_length=70000,
            return_tensors="pt",
        )

        item.data["labels"] = self.tokenizer(
            self.data[idx]["answer"],
            add_special_tokens=True,
            return_tensors="pt",
            padding="max_length",
            max_length=10,
        ).data["input_ids"]

        return item

    def __len__(self):
        return len(self.data)


class TrainTestDataset(ABC):
    _dataset: DatasetDict
    _dataset_path: str

    def __init__(self) -> None:
        self._dataset_path = generate_cache_path("dataset", self, skip_vocab_size=True)

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

    def get_train_ready_data(self, retriever, tokenizer) -> ParallelTrainData:
        return ParallelTrainData(self.get_train_data(), retriever, tokenizer)

    def get_validation_ready_data(self, retriever, tokenizer) -> ParallelTrainData:
        return ParallelTrainData(self.get_validation_data(), retriever, tokenizer)

    def get_dataset(self):
        return TrainTestDataset.dataset.__get__(self)
