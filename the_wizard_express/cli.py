"""Console script for the_wizard_express."""
import sys
from multiprocessing import cpu_count

import click

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
    TriviaQA()
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
