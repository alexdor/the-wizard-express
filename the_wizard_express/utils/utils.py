from os.path import dirname, join
from pathlib import Path
from pickle import HIGHEST_PROTOCOL, dump

from inflect import engine
from the_wizard_express.config import Config

inflect_engine = engine()


def generate_cache_path(name: str, *args, **kwargs):
    return join(
        Config.cache_dir,
        inflect_engine.plural(name),
        "_".join([c.get_id() for c in args])
        + f"_{inflect_engine.singular_noun(name) or name}"
        + ("" if kwargs.get("skip_vocab_size") else f"_{Config.vocab_size}")
        + (kwargs["file_ending"] if kwargs.get("file_ending") else ".json"),
    )


def pickle_and_save_to_file(item, filepath: str, protocol: int = HIGHEST_PROTOCOL):
    Path(dirname(filepath)).mkdir(parents=True, exist_ok=True)
    dump(item, open(filepath, "wb"), protocol=protocol)
