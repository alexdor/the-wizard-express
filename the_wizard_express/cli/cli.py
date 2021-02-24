"""Console script for the_wizard_express."""
import random
import sys
from datetime import datetime, timedelta
from json import dumps
from timeit import default_timer as timer
from typing import Optional

import click
from datasets import load_metric
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

from ..config import Config
from ..corpus import Corpus, Squad, SquadV2, TriviaQA
from ..language_model import DistilBertForQA, get_trainer
from ..qa import PyseriniBertOnBert, QAModel, TFIDFBertOnBert


def create_click_args(classes):
    """Helper function to convert a list of classes into click selections"""
    classes = [(obj.friendly_name, obj) for obj in classes]
    return {
        "type": click.Choice([item[0] for item in classes], case_sensitive=False),
        "default": classes[0][0],
        # Get user input and return the corespoding class
        "callback": lambda _, selection: next(
            (values[1] for values in classes if values[0] == selection), None
        ),
    }


models = (PyseriniBertOnBert, TFIDFBertOnBert)

corpuses = (Squad, TriviaQA)


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
    skip_training = True
    squad_v2 = False
    train_start_time = timer()
    distilled_bert = DistilBertForQA()
    squad = TriviaQA(True, distilled_bert.tokenizer, v1=not squad_v2)
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
@click.option("--model", **create_click_args(models))
@click.option("--corpus", **create_click_args(corpuses))
@click.option("--gpu", type=int, default=None)
def run_squad_validation(model: QAModel, corpus: Corpus, gpu: Optional[int]):
    squad_v2 = False
    metric_prep_time = timer()
    corpus_instance = corpus(return_raw=True) if corpus is TriviaQA else corpus()
    model_instance = model(corpus_instance, gpu=gpu)
    validation_data = corpus_instance.get_validation_data()
    metric_start_time = timer()

    if corpus is TriviaQA:
        answers = [
            model_instance.answer_question(ex["question"])
            for ex in tqdm(validation_data)
        ]
        count = sum(
            int(answers[i].lower() == ex["answer"].lower())
            for i, ex in tqdm(enumerate(validation_data))
        )
        final_metrics = {"exact_match_custom": f"{count/float(len(validation_data))}%"}
        print(final_metrics)

        precision, recall, f1, _ = precision_recall_fscore_support(
            validation_data["answer"], answers, average="micro"
        )
        final_metrics["precision"] = precision
        final_metrics["recall"] = recall
        final_metrics[" f1"] = f1
    else:

        formatted_predictions = [
            {
                "id": data["id"],
                "prediction_text": model_instance.answer_question(question),
            }
            for data in tqdm(validation_data)
            for question in (
                data["question"]
                if isinstance(data["question"], list)
                else [data["question"]]
            )
        ]

        metric = load_metric(
            "squad_v2" if squad_v2 else "squad",
            cache_dir=Config.hugging_face_cache_dir,
        )

        references = [
            {
                "id": ex["id"],
                "answers": ex["answers"],
            }
            for ex in validation_data
        ]

        final_metrics = metric.compute(
            predictions=formatted_predictions, references=references
        )
    metric_end_time = timer()
    with open(f"./results_for_paper/{model.friendly_name}.json", "a+") as file:
        file.write(
            dumps(
                {
                    "date": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                    "metrics": final_metrics,
                    "extra_info": {
                        "corpus": corpus_instance.friendly_name,
                        "model": model.friendly_name,
                        "max_vocab_size": Config.vocab_size,
                        "total_time": str(
                            timedelta(seconds=metric_end_time - metric_prep_time)
                        ),
                        "testing_time": str(
                            timedelta(seconds=metric_start_time - metric_start_time)
                        ),
                        "prep_time": str(
                            timedelta(seconds=metric_start_time - metric_prep_time)
                        ),
                    },
                },
                indent=2,
            )
        )


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


@main.command()
@click.option("--model", **create_click_args(models))
@click.option("--corpus", **create_click_args(corpuses))
def random_answers(model: QAModel, corpus: Corpus):
    """This command allows the user to directly pass a question to the model and shows the final prediction."""
    corpus_instance = corpus(return_raw=True) if corpus is TriviaQA else corpus()
    model_instance = model(corpus_instance)
    data = corpus_instance.get_validation_data()
    answers = [
        {
            "question": data[i]["question"],
            "answer": model_instance.answer_question(data[i]["question"]),
            "expected_answer": data[i]["answer"]["text"]
            if corpus is Squad
            else data[i]["answer"],
        }
        for i in random.sample(range(len(data)), 10)
    ]
    with open("./results_for_paper/random.txt", "a+") as file:
        file.write(
            dumps(
                {
                    "answers": answers,
                    "model": model_instance.friendly_name,
                    "corpus": corpus_instance.friendly_name,
                },
                indent=2,
            )
        )


@main.command()
@click.option("--model", **create_click_args(models))
@click.option("--corpus", **create_click_args(corpuses))
@click.argument("question", required=True, type=str)
def answer(model: QAModel, corpus: Corpus, question: str):
    """This command allows the user to directly pass a question to the model and shows the final prediction."""
    corpus_instance = corpus(return_raw=True) if corpus is TriviaQA else corpus()
    model_instance = model(corpus_instance)

    print(model_instance.answer_question(question))


@main.command()
def start_webserver():
    pass


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
