"""Console script for the_wizard_express."""
import sys
from datetime import datetime, timedelta
from json import dumps
from os import environ
from timeit import default_timer as timer
from typing import Any, Callable, List, Tuple, Union

import click
from datasets import load_metric
from tqdm import tqdm

from .config import Config
from .corpus import ParallelTrainData, Squad, TriviaQA
from .language_model import StackedBert
from .qa import TFIDFBertOnBert
from .reader import BertOnBertReader, SimpleBertReader, TinyBertReader
from .retriever import TFIDFRetriever
from .tokenizer import WordTokenizer, WordTokenizerWithoutStopWords


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
readers = option_to_type((BertOnBertReader, TinyBertReader, SimpleBertReader))
corpuses = option_to_type((TriviaQA, Squad))
tokenizers = option_to_type((WordTokenizerWithoutStopWords, WordTokenizer))


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
def dev(retriever, reader, corpus, tokenizer):

    # Stop hugging face from complaining about forking
    environ["TOKENIZERS_PARALLELISM"] = "true"

    corpus_instance = corpus()
    tokenizer_instance = tokenizer(corpus_instance)
    retriever_instance = retriever(corpus=corpus_instance, tokenizer=tokenizer_instance)
    reader_instance = reader(tokenizer=tokenizer_instance)

    train_point = corpus_instance.get_train_data()[20]
    question = train_point["question"]
    docs = retriever_instance.retrieve_docs(question, 5)
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


@main.command()
def evaluate_metrics():
    metrics_to_run = {"rouge": {"rouge_types": "rougeLsum"}}  # "sacrebleu": {}}
    metrics = {
        metric: load_metric(metric, **args, cache_dir=Config.cache_dir)
        for metric, args in metrics_to_run.items()
    }

    corpus = TriviaQA()
    model = TFIDFBertOnBert(corpus=corpus)

    start_time = timer()
    for data in tqdm(corpus.get_validation_data()):
        args = {
            "prediction": model.answer_question(data["question"]),
            "reference": data["answer"],
        }
        for metric in metrics.values():
            metric.add(**args)
    end_time = timer()

    final_metrics = {
        metric: metric_func.compute() for metric, metric_func in metrics.items()
    }
    with open("./results_for_paper/metrics.txt", "w+") as file:
        file.write(
            dumps(
                {
                    "date": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                    "data_it_run_on": "validation",
                    "metrics": final_metrics,
                    "extra_info": {
                        "corpus": corpus.friendly_name,
                        "model": model.friendly_name,
                        "max_vocab_size": Config.vocab_size,
                        "total_time": str(timedelta(seconds=end_time - start_time)),
                    },
                },
                indent=2,
            )
        )

    #
    # TFIDFBertSimple


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
