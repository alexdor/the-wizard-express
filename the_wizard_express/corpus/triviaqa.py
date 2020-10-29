from datasets import concatenate_datasets, load_dataset

from ..config import Config
from .corpus import Corpus, TrainTestDataset


class TriviaQA(Corpus, TrainTestDataset):
    def __init__(self) -> None:
        self.corpus = None
        dataset = load_dataset("trivia_qa", "rc", cache_dir=Config.cache_dir)

        self.dataset = dataset.filter(
            lambda row: len(row["entity_pages"]["wiki_context"]) > 0
        ).map(
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
                "answer",
            ],
            num_proc=Config.max_proc_to_use,
        )

    def get_corpus(self):
        if self.corpus is None:
            dataset = concatenate_datasets(
                (
                    self.dataset["train"],
                    self.dataset["test"],
                    self.dataset["validation"],
                )
            )

            self.corpus = dataset.map(
                lambda data: {"context": data["context"]},
                remove_columns=[
                    "question",
                    "answer",
                ],
                num_proc=Config.max_proc_to_use,
            )

        return self.corpus["context"]
