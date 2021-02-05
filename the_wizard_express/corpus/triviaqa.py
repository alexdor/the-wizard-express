from datasets import concatenate_datasets, load_dataset
from numpy import sort, unique

from ..config import Config
from ..utils import generate_cache_path, select_part_of_dataset
from .corpus import Corpus, TrainTestDataset


class TriviaQA(Corpus, TrainTestDataset):
    __slots__ = (
        "corpus_path",
        "dataset_path",
        "percent_of_data_to_keep",
        "_corpus",
        "_dataset",
    )
    friendly_name = "triviaqa"

    def __init__(self) -> None:
        self._corpus_path = generate_cache_path("corpus", self)
        self._dataset_path = generate_cache_path("dataset", self)

    def _build_dataset(self):
        dataset = load_dataset(
            "trivia_qa",
            "rc",
            cache_dir=Config.cache_dir,
        )

        dataset = dataset.filter(
            lambda row: len(row["entity_pages"]["wiki_context"]) > 0,
            num_proc=Config.max_proc_to_use,
        )

        dataset = select_part_of_dataset(dataset)

        self._dataset = dataset.map(
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
        dataset = sort(unique(dataset._data.column("context").to_numpy()))
        self._corpus = dataset
