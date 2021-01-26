from datasets import concatenate_datasets, load_dataset
from numpy import sort, unique

from ..config import Config
from ..utils import (
    generate_cache_path,
    pickle_and_save_to_file,
    select_part_of_dataset,
)
from .corpus import Corpus, TrainTestDataset


class Squad(Corpus, TrainTestDataset):
    __slots__ = (
        "corpus_path",
        "dataset_path",
        "_corpus",
        "_dataset",
    )
    friendly_name = "squad"

    def __init__(self) -> None:
        self._corpus_path = generate_cache_path("corpus", self)
        self._dataset_path = generate_cache_path("dataset", self)

    def _build_dataset(self):
        dataset = load_dataset(
            "squad",
            cache_dir=Config.cache_dir,
        )

        dataset = dataset.filter(
            lambda row: len(row["context"]) > 0,
            num_proc=Config.max_proc_to_use,
            cache_file_name=f"{self.friendly_name}_filter",
        )
        dataset = select_part_of_dataset(dataset)

        self._dataset = dataset.map(
            lambda data: {
                "question": data["question"],
                "answer": data["answers"]["text"][0],
                "context": data["context"],
            },
            remove_columns=["id", "title", "answers"],
            num_proc=Config.max_proc_to_use,
            cache_file_name=f"{self.friendly_name}_map",
        )
        pickle_and_save_to_file(self._dataset, self._dataset_path)

    def _build_corpus(self) -> None:

        dataset = concatenate_datasets(
            (
                self.dataset["train"],
                self.dataset["validation"],
            )
        )

        dataset = dataset.map(
            lambda data: {"context": data["context"]},
            remove_columns=[
                "question",
                "answer",
            ],
            num_proc=Config.max_proc_to_use,
        )
        dataset = sort(unique(dataset._data.column("context").to_numpy()))
        pickle_and_save_to_file(dataset, self._corpus_path)
        self._corpus = dataset
