from os import environ
from typing import Any, Callable, List, Tuple, Union

import click

from ..corpus import Squad, TriviaQA
from ..reader import BertOnBertReader, SimpleBertReader, TinyBertReader
from ..retriever import PyseriniSimple, TFIDFRetriever
from ..tokenizer import WordTokenizer, WordTokenizerWithoutStopWords
from . import main
from .cli import turn_user_selection_to_class


def option_to_type(objects: Union[Tuple[Any, ...], List[Any]]):
    return [(obj.friendly_name, obj) for obj in objects]


retrievers = option_to_type([TFIDFRetriever])
readers = option_to_type((BertOnBertReader, TinyBertReader, SimpleBertReader))
corpuses = option_to_type((TriviaQA, Squad))
tokenizers = option_to_type((WordTokenizerWithoutStopWords, WordTokenizer))


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
def dev(retriever, reader, corpus, tokenizer):

    # Stop hugging face from complaining about forking
    environ["TOKENIZERS_PARALLELISM"] = "true"

    corpus_instance = corpus()
    tokenizer_instance = tokenizer(corpus_instance)
    retriever_instance = retriever(corpus=corpus_instance, tokenizer=tokenizer_instance)
    reader_instance = reader(tokenizer=tokenizer_instance)

    train_point = corpus_instance.get_train_data()[20]
    question = train_point["question"]
    docs = tuple(retriever_instance.retrieve_docs(question, 5))
    found_documents = train_point["context"] in docs
    if not found_documents:
        docs += tuple([train_point["context"]])
    answer = reader_instance.answer(question=question, documents=docs)
    print("\n" * 10)
    print(f"Question: {question}")
    print(f"Expected answer: {train_point['answer']}")
    print(f"Retrived proper document: {found_documents}")
    print(f"Model's answer: {answer}")
    return 0
