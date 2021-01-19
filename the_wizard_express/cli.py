"""Console script for the_wizard_express."""
import sys
from multiprocessing import cpu_count
from os import environ
from typing import Any, Callable, List, Tuple, Union

import click

from .config import Config
from .corpus import Squad, TriviaQA
from .reader import RealmReader, TinyBertReader
from .retriever import TFIDFRetriever
from .tokenizer import WordTokenizer


@click.group()
@click.option("--debug/--no-debug", default=Config.debug)
@click.option("--max-proc", default=Config.max_proc_to_use, show_default=True, type=int)
@click.option(
    "--data_to_keep",
    default=Config.percent_of_data_to_keep,
    show_default=True,
    type=float,
)
@click.option("--vocab-size", default=Config.vocab_size, show_default=True, type=int)
def main(debug, max_proc, data_to_keep, vocab_size):
    Config.debug = debug
    Config.max_proc_to_use = max_proc
    Config.percent_of_data_to_keep = data_to_keep
    Config.vocab_size = vocab_size


def option_to_type(objects: Union[Tuple[Any, ...], List[Any]]):
    return [(obj.friendly_name, obj) for obj in objects]


retrievers = option_to_type([TFIDFRetriever])
readers = option_to_type((RealmReader, TinyBertReader))
corpuses = option_to_type((TriviaQA, Squad))
tokenizers = option_to_type([WordTokenizer])


def turn_user_selection_to_class(possible_values) -> Callable[[Any, str], Any]:
    return lambda _, selection: next(
        (values[1] for values in possible_values if values[0] == selection), None
    )


@main.command()
@click.option(
    "--retriever",
    type=click.Choice([item[0] for item in retrievers], case_sensitive=False),
    default=retrievers[0][0],
    callback=turn_user_selection_to_class(retrievers),
)
@click.option(
    "--reader",
    type=click.Choice([item[0] for item in readers], case_sensitive=False),
    default=readers[0][0],
    callback=turn_user_selection_to_class(readers),
)
@click.option(
    "--corpus",
    type=click.Choice([item[0] for item in corpuses], case_sensitive=False),
    default=corpuses[0][0],
    callback=turn_user_selection_to_class(corpuses),
)
@click.option(
    "--tokenizer",
    type=click.Choice([item[0] for item in tokenizers], case_sensitive=False),
    default=tokenizers[0][0],
    callback=turn_user_selection_to_class(tokenizers),
)
def eval(retriever, reader, corpus, tokenizer):
    corpus_instance = corpus()
    tokenizer_instance = tokenizer(corpus_instance)
    retriever_instance = retriever(corpus=corpus_instance, tokenizer=tokenizer_instance)
    reader_instance = reader(tokenizer=tokenizer_instance)

    train_point = corpus_instance.get_train_data()[20]
    question = train_point["question"]
    docs = retriever_instance.retrieve_docs(question, 15)
    found_documents = train_point["context"] in docs
    if not found_documents:
        docs += tuple([train_point["context"]])
    answer = reader_instance.answer(question=question, document="\n".join(docs))
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
