from os.path import join

from inflect import engine

from the_wizard_express.config import Config

inflect_engine = engine()


def generate_cache_path(name: str, *args, **kwargs):
    return join(
        Config.cache_dir,
        inflect_engine.plural(name),
        "_".join([c.get_id() for c in args])
        + f"_{inflect_engine.singular_noun(name) or name}"
        + (".json" if kwargs.get("skip_vocab_size") else f"_{Config.vocab_size}.json"),
    )
