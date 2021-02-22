import sys
from os.path import dirname, join
from pathlib import Path
from pickle import HIGHEST_PROTOCOL, dump

from datasets.arrow_dataset import Dataset
from inflect import engine
from transformers import PreTrainedTokenizerFast

from ..config import Config

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict


class DatasetDict(TypedDict):
    train: Dataset
    test: Dataset
    validation: Dataset


inflect_engine = engine()


def generate_cache_path(name: str, *args, **kwargs):
    return join(
        Config.cache_dir,
        inflect_engine.plural(name),
        "_".join(
            [
                c.get_id()
                for c in args
                if not (c is None or isinstance(c, PreTrainedTokenizerFast))
            ]
        )
        + f"_{inflect_engine.singular_noun(name) or name}"
        + ("" if kwargs.get("skip_vocab_size") else f"_{Config.vocab_size}")
        + kwargs.get("file_ending", ".pickle"),
    )


def pickle_and_save_to_file(item, filepath: str, protocol: int = HIGHEST_PROTOCOL):
    Path(dirname(filepath)).mkdir(parents=True, exist_ok=True)
    dump(item, open(filepath, "wb"), protocol=protocol)


def select_part_of_dataset(dataset: DatasetDict) -> DatasetDict:
    # Return early if we want the full dataset
    if Config.percent_of_data_to_keep == 1:
        return dataset

    def select_part(data: Dataset) -> Dataset:
        return data.select(range(round(len(data) * Config.percent_of_data_to_keep)))

    dataset["train"] = select_part(dataset["train"])
    dataset["test"] = select_part(dataset["test"])
    dataset["validation"] = select_part(dataset["validation"])
    return dataset
