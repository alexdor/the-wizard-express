from os.path import dirname, lexists
from pathlib import Path
from pickle import load

from datasets import concatenate_datasets, load_dataset
from datasets.arrow_dataset import Dataset
from numpy import sort, unique
from the_wizard_express.utils import (
    generate_cache_path,
    pickle_and_save_to_file,
)

from ..config import Config
from .corpus import Corpus, TrainTestDataset


class TriviaQA(Corpus, TrainTestDataset):
    __slots__ = ("corpus_path", "percent_of_data_to_keep", "corpus", "dataset")

    def __init__(self, percent_of_data_to_keep: float = 0.1) -> None:
        Corpus.__init__(self)
        self.percent_of_data_to_keep = percent_of_data_to_keep
        self.corpus_path = generate_cache_path("corpus", self)
        dataset = load_dataset(
            "trivia_qa",
            "rc",
            cache_dir=Config.cache_dir,
        )

        dataset = dataset.filter(
            lambda row: len(row["entity_pages"]["wiki_context"]) > 0,
            num_proc=Config.max_proc_to_use,
        )

        def select_part(data: Dataset, percent: float) -> Dataset:
            return data.select(range(round(len(data) * percent)))

        dataset["train"] = select_part(dataset["train"], percent_of_data_to_keep)
        dataset["test"] = select_part(dataset["test"], percent_of_data_to_keep)
        dataset["validation"] = select_part(
            dataset["validation"], percent_of_data_to_keep
        )

        self.dataset = dataset.map(
            lambda data: {
                "question": data["question"],
                "answer": data["answer"]["value"],
                "context": "\n".join(data["entity_pages"]["wiki_context"]),
            },
            remove_columns=[
                "question_id",
                "question_source",
                "entity_pages",
                "search_results",
            ],
            num_proc=Config.max_proc_to_use,
        )

    def _build_corpus(self) -> None:
        if lexists(self.corpus_path):
            self._corpus = load(open(self.corpus_path, "rb"))
            return

        dataset = concatenate_datasets(
            (
                self.dataset["train"],
                self.dataset["test"],
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
        dataset = tuple(sort(unique(dataset._data.column("context").to_numpy())))
        pickle_and_save_to_file(dataset, self.corpus_path)
        self._corpus = dataset

    def save_to_disk(self, file_location: str) -> None:
        Path(dirname(file_location)).mkdir(parents=True, exist_ok=True)
        with open(file_location, "w+", encoding="utf-8") as f:
            f.writelines([f"{line}\n" for line in self.corpus])
