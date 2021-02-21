from os import environ

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import equal
from transformers import Trainer, TrainingArguments


def get_trainer(
    model,
    train_dataset,
    validation_dataset=None,
    validation_examples=None,  # datasets["validation"]
    output_dir="./results",
    logging_dir="./logs",
):
    # 3,4
    environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"

    # def compute_metrics(pred):
    #     # """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    #     # def normalize_text(text):
    #     #     text = "".join(ch for ch in  text.lower() if ch not in exclude)
    #     #     text = " ".join( re.sub(regex, " ", text).split())
    #     #     return text

    #     labels, preds = pred.label_ids, pred.predictions.argmax(-1)

    #     # exact_match = int(equal(labels, preds))
    #     precision, recall, f1, _ = precision_recall_fscore_support(
    #         labels, preds, average="micro"
    #     )
    #     acc = accuracy_score(labels, preds)
    #     return {
    #         "accuracy": acc,
    #         "f1": f1,
    #         "precision": precision,
    #         "recall": recall,
    #         # "exact_match": exact_match,
    #     }

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir=logging_dir,
        learning_rate=2e-5,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
    )

    # trainer = QuestionAnsweringTrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset if training_args.do_train else None,
    #     eval_dataset=validation_dataset,
    #     eval_examples=validation_examples,
    #     tokenizer=tokenizer,
    #     post_process_function=post_processing_function,
    #     compute_metrics=compute_metrics,
    # )

    return trainer
