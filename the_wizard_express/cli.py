"""Console script for the_wizard_express."""
import sys
from multiprocessing import cpu_count
from os import environ

import click
from the_wizard_express.reader import RealmReader
from the_wizard_express.retriever import TFIDFRetriever
from the_wizard_express.tokenizer import WordTokenizer

from .config import Config
from .corpus import TriviaQA


@click.group()
@click.option("--debug/--no-debug", default=False)
@click.option(
    "--max-proc", default=min(cpu_count() - 1, 15), show_default=True, type=int
)
def main(debug, max_proc):
    Config.debug = debug
    Config.max_proc_to_use = max_proc


@main.command()
def trivia():
    corpus = TriviaQA(percent_of_data_to_keep=1)
    tokenizer = WordTokenizer(corpus)
    retriever = TFIDFRetriever(corpus=corpus, tokenizer=tokenizer)
    train_point = corpus.get_train_data()[20]
    question = train_point["question"]
    docs = retriever.retrieve_docs(question, 3)
    found_documents = train_point["context"] in docs
    if not found_documents:
        docs += tuple([train_point["context"]])
    # for d in docs:
    answer = RealmReader(tokenizer=tokenizer).answer(
        question=question, document=train_point["context"]
    )
    print("\n" * 10)
    print(f"Question: {question}")
    print(f"Expected answer: {train_point['answer']}")
    print(f"Retrived proper document: {found_documents}")
    print("Model's answer:")
    print(answer)
    return 0


if __name__ == "__main__":
    # Stop hugging face from complaining about forking
    environ["TOKENIZERS_PARALLELISM"] = "true"
    sys.exit(main())  # pragma: no cover
