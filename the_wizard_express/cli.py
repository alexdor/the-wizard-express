"""Console script for the_wizard_express."""
import sys
from multiprocessing import cpu_count
from os.path import join

import click

from the_wizard_express.tokenizer.word import WordTokenizer

from .config import Config
from .corpus.triviaqa import TriviaQA


@click.group()
@click.option("--debug/--no-debug", default=False)
@click.option(
    "--max-proc", default=min(cpu_count() - 1, 8), show_default=True, type=int
)
def main(debug, max_proc):
    Config.debug = debug
    Config.proc_to_use = max_proc


@main.command()
def trivia():
    tok = WordTokenizer(TriviaQA())
    t = tok.encode_batch(
        ["This is a batch do you have the ability to encode and decode it"]
    )
    print("encoded", t[0].tokens, t[0].ids)
    print("decoded", tok.tokens_to_sentance(t[0].ids))
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
