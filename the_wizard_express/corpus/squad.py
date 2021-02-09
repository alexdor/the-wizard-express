from datasets import concatenate_datasets, load_dataset
from numpy import sort, unique

from ..config import Config
from ..utils import select_part_of_dataset
from . import Corpus, TrainTestDataset


class Squad(Corpus, TrainTestDataset):
    __slots__ = (
        "corpus_path",
        "_dataset_path",
        "_corpus",
        "_dataset",
    )

    friendly_name = "squad"

    def __init__(self) -> None:
        super().__init__()
        TrainTestDataset.__init__(self)

    def _build_dataset(self):
        dataset = load_dataset(
            "squad",
            cache_dir=Config.cache_dir,
        )

        dataset = dataset.filter(
            lambda row: len(row["context"]) > 0,
            num_proc=Config.max_proc_to_use,
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
        )

    def _build_corpus(self) -> None:

        dataset = concatenate_datasets(
            (
                self.dataset["train"],
                self.dataset["validation"],
            )
        )

        self._transform_datasets_to_corpus(dataset)
