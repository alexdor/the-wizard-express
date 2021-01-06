"""Console script for the_wizard_express."""
import sys
from multiprocessing import cpu_count
from os import environ
from os.path import join

import click

from the_wizard_express.retriever import TFIDFRetriever
from the_wizard_express.tokenizer import WordTokenizer

from .config import Config
from .corpus import TriviaQA


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
    retriever = TFIDFRetriever(corpus=TriviaQA(), tokenizer=WordTokenizer(TriviaQA()))
    t = retriever.retrieve_docs(
        "This is a batch do you have the ability to encode and decode it", 5
    )
    print(t)
    return 0


if __name__ == "__main__":
    # Stop hugging face from complaining about forking
    environ["TOKENIZERS_PARALLELISM"] = "true"
    sys.exit(main())  # pragma: no cover
