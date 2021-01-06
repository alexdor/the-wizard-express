from os.path import dirname
from pathlib import Path
from typing import List

from datasets import concatenate_datasets, load_dataset

from ..config import Config
from .corpus import Corpus, TrainTestDataset


class TriviaQA(Corpus, TrainTestDataset):
    def __init__(self, percent_of_data_to_keep=0.1) -> None:
        self.corpus = None
        dataset = load_dataset(
            "trivia_qa",
            "rc",
            cache_dir=Config.cache_dir,
        )

        dataset = dataset.filter(
            lambda row: len(row["entity_pages"]["wiki_context"]) > 0,
            num_proc=Config.max_proc_to_use,
        )

        def select_part(data, percent):
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

    def get_corpus(self) -> List[str]:
        if self.corpus is None:
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
            ).unique(column="context")
            dataset.sort()
            self.corpus = dataset

        return self.corpus

    def save_to_disk(self, file_location: str):
        corpus = self.get_corpus()
        Path(dirname(file_location)).mkdir(parents=True, exist_ok=True)
        with open(file_location, "w+", encoding="utf-8") as f:
            f.writelines([f"{line}\n" for line in corpus])
