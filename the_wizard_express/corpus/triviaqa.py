from datasets import concatenate_datasets, load_dataset

from ..config import Config
from ..utils import select_part_of_dataset
from .corpus import Corpus, TrainTestDataset


class TriviaQA(Corpus, TrainTestDataset):
    __slots__ = (
        "corpus_path",
        "_dataset_path",
        "percent_of_data_to_keep",
        "_corpus",
        "_dataset",
    )

    friendly_name = "triviaqa"

    def __init__(self, return_raw=False) -> None:
        super().__init__()
        TrainTestDataset.__init__(self)
        self.return_raw = return_raw

    def _build_dataset(self):
        dataset = load_dataset(
            "trivia_qa",
            "rc",
            cache_dir=Config.hugging_face_cache_dir,
        )
        dataset = select_part_of_dataset(dataset)
        dataset = dataset.map(
            lambda data: {
                "question": data["question"],
                "answer": data["answer"]["value"],
                "context": data["entity_pages"]["wiki_context"],
            },
            remove_columns=[
                "question_id",
                "question_source",
                "entity_pages",
                "search_results",
            ],
            num_proc=Config.max_proc_to_use,
        )
        if self.return_raw:
            self._dataset = dataset
            return

        self._dataset = dataset.filter(
            lambda row: len(row["entity_pages"]["wiki_context"]) > 0,
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
        self._transform_datasets_to_corpus(dataset)
