"""Console script for the_wizard_express."""
import sys
from datetime import datetime, timedelta
from json import dumps
from os import environ
from timeit import default_timer as timer

import click
from datasets import load_metric
from datasets.arrow_dataset import concatenate_datasets
from datasets.utils.py_utils import get_datasets_path
from tqdm import tqdm
from transformers import AutoTokenizer

from ..config import Config
from ..corpus import Squad, SquadV2, TriviaQA
from ..language_model import DistilBertForQA, StackedBert, get_trainer
from ..qa import TFIDFBertOnBert
from ..retriever import PyseriniSimple, TFIDFRetriever
from ..tokenizer import WordTokenizer


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


@main.command()
def train():
    skip_training = False
    squad_v2 = False
    train_start_time = timer()
    distilled_bert = DistilBertForQA(skip_training)
    squad = SquadV2(distilled_bert.tokenizer, v1=not squad_v2)
    dataset = squad.get_dataset()
    dataset["train"] = dataset["train"].map(
        squad.prepare_train_features,
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=Config.max_proc_to_use,
    )
    trainer = get_trainer(distilled_bert.model, dataset["train"])
    if not skip_training:
        trainer.train()
        trainer.save_model(distilled_bert.friendly_name)
    metric_start_time = timer()
    prediction_features = dataset["validation"].map(
        squad.prepare_validation_features,
        batched=True,
        remove_columns=dataset["validation"].column_names,
        num_proc=Config.max_proc_to_use,
    )

    # make a copy of the original validation data because the trainer drops keywords
    validation_features = dataset["validation"].map(
        squad.prepare_validation_features,
        batched=True,
        remove_columns=dataset["validation"].column_names,
        num_proc=Config.max_proc_to_use,
    )

    raw_predictions = trainer.predict(prediction_features)
    final_predictions = squad.postprocess_qa_predictions(
        dataset["validation"],
        validation_features,
        raw_predictions.predictions,
    )

    metric = load_metric(
        "squad_v2" if squad_v2 else "squad",
        cache_dir=Config.hugging_face_cache_dir,
    )
    if squad_v2:
        formatted_predictions = [
            {"id": k, "prediction_text": v, "no_answer_probability": 0.0}
            for k, v in final_predictions.items()
        ]
    else:
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in final_predictions.items()
        ]

    references = [
        {"id": ex["id"], "answers": ex["answers"]} for ex in dataset["validation"]
    ]

    final_metrics = metric.compute(
        predictions=formatted_predictions, references=references
    )
    metric_end_time = timer()
    with open("./results_for_paper/distilled-bert.json", "a+") as file:
        file.write(
            dumps(
                {
                    "date": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                    "metrics": final_metrics,
                    "extra_info": {
                        "corpus": squad.friendly_name,
                        "model": distilled_bert.friendly_name,
                        "max_vocab_size": Config.vocab_size,
                        "total_time": str(
                            timedelta(seconds=metric_end_time - train_start_time)
                        ),
                        "training_time": str(
                            timedelta(seconds=metric_start_time - train_start_time)
                        ),
                        "metric_computation_time": str(
                            timedelta(seconds=metric_end_time - metric_start_time)
                        ),
                    },
                },
                indent=2,
            )
        )


@main.command()
def train_wiki():
    train_start_time = timer()
    distilled_bert = DistilBertForQA()
    squad = SquadV2(distilled_bert.tokenizer, v1=True)
    trivia = TriviaQA(return_raw=True).get_dataset()
    dataset = concatenate_datasets([squad.get_dataset(), trivia.get_dataset()])
    dataset = dataset.map(
        squad.prepare_train_features,
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=Config.max_proc_to_use,
    )
    trainer = get_trainer(
        distilled_bert.model,
        dataset["train"],
        dataset["validation"],
        output_dir="./results_wiki",
        logging_dir="./logs_with_wiki",
    )
    trainer.train()
    trainer.save_model(distilled_bert.friendly_name)
    metric_start_time = timer()
    validation_features = dataset["validation"].map(
        squad.prepare_validation_features,
        batched=True,
        remove_columns=dataset["validation"].column_names,
        num_proc=Config.max_proc_to_use,
    )

    raw_predictions = trainer.predict(validation_features)
    final_predictions = squad.postprocess_qa_predictions(
        dataset["validation"], validation_features, raw_predictions.predictions
    )

    metric = load_metric(
        "squad_v2",
        cache_dir=Config.hugging_face_cache_dir,
    )
    formatted_predictions = [
        {"id": k, "prediction_text": v, "no_answer_probability": 0.0}
        for k, v in final_predictions.items()
    ]

    references = [
        {"id": ex["id"], "answers": ex["answers"]} for ex in dataset["validation"]
    ]

    final_metrics = metric.compute(
        predictions=formatted_predictions, references=references
    )
    metric_end_time = timer()
    with open("./results_for_paper/distilled-bert-with-wiki.json", "a+") as file:
        file.write(
            dumps(
                {
                    "date": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                    "metrics": final_metrics,
                    "extra_info": {
                        "corpus": squad.friendly_name,
                        "model": distilled_bert.friendly_name,
                        "max_vocab_size": Config.vocab_size,
                        "total_time": str(
                            timedelta(seconds=metric_end_time - train_start_time)
                        ),
                        "training_time": str(
                            timedelta(seconds=metric_start_time - train_start_time)
                        ),
                        "metric_computation_time": str(
                            timedelta(seconds=metric_end_time - metric_start_time)
                        ),
                    },
                },
                indent=2,
            )
        )


# @main.command()
# def train2():
#     corpus = TriviaQA()
#     tokenizer = WordTokenizer(corpus=corpus)
#     retriever = TFIDFRetriever(corpus=corpus, tokenizer=tokenizer)
#     reader_tokenizer = AutoTokenizer.from_pretrained(
#         StackedBert.model,
#         use_fast=True,
#         cache_dir=Config.hugging_face_cache_dir,
#     )
#     model = StackedBert(reader_tokenizer)

#     args = {
#         "train_dataset": corpus.get_train_ready_data(retriever, reader_tokenizer),
#         "eval_dataset": corpus.get_eval_ready_data(retriever, reader_tokenizer),
#     }


@main.command()
def evaluate_metrics():
    metrics_to_run = {"rouge": {"rouge_types": "rougeLsum"}}  # "sacrebleu": {}}
    metrics = {
        metric: load_metric(
            metric,
            **args,
            cache_dir=Config.hugging_face_cache_dir,
        )
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


@main.command()
def pyserini():
    corpus = TriviaQA()
    corpus.get_train_data()
    corpus.get_validation_data()
    corpus.corpus
    pyser = PyseriniSimple(corpus=corpus, tokenizer=None)
    pyser.retrieve_docs("Fuck you", 5)
    print(pyser)


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
