from datasets import concatenate_datasets, load_dataset, load_metric
from transformers import EvalPrediction

from ..config import Config
from ..utils import postprocess_qa_predictions, select_part_of_dataset
from . import Corpus, TrainTestDataset


def proccess_squad(dataset):
    dataset = dataset.filter(
        lambda row: len(row["context"]) > 0,
        num_proc=Config.max_proc_to_use,
    )
    dataset = select_part_of_dataset(dataset)

    return dataset.map(
        lambda data: {
            "question": data["question"],
            "answer": data["answers"]["text"][:1],
            "context": data["context"],
        },
        remove_columns=["id", "title", "answers"],
        num_proc=Config.max_proc_to_use,
    )


class Squad(Corpus, TrainTestDataset):
    __slots__ = (
        "corpus_path",
        "_dataset_path",
        "_corpus",
        "_dataset",
    )

    friendly_name = "squad"

    def __init__(self) -> None:
        super().__init__()
        TrainTestDataset.__init__(self)

    def _build_dataset(self):
        self._dataset = load_dataset(
            "squad",
            cache_dir=Config.hugging_face_cache_dir,
        )

        # self._dataset = proccess_squad(dataset)

    def _build_corpus(self) -> None:

        dataset = concatenate_datasets(
            (
                self.dataset["train"],
                self.dataset["validation"],
            )
        )

        self._transform_datasets_to_corpus(dataset)


class SquadV2(Corpus, TrainTestDataset):
    __slots__ = (
        "corpus_path",
        "_dataset_path",
        "_corpus",
        "_dataset",
    )

    friendly_name = "squad_new"

    def __init__(
        self, tokenizer, output_dir=None, v1=False, null_score_diff_threshold=0.0
    ) -> None:
        super().__init__()
        TrainTestDataset.__init__(self)
        self.tokenizer = tokenizer
        self.v1 = v1
        self.null_score_diff_threshold = null_score_diff_threshold
        self.friendly_name = "squad" if v1 else "squad-v2"
        self.output_dir = output_dir
        self.metric = load_metric(
            "squad" if v1 else "squad_v2",
            cache_dir=Config.hugging_face_cache_dir,
        )
        if not tokenizer:
            return
        self._pad_on_right = tokenizer.padding_side == "right"
        self._tokenizer_conf = {
            "truncation": "only_second" if self._pad_on_right else "only_first",
            "max_length": Config.max_seq_len,
            "stride": Config.doc_stride,
            "return_overflowing_tokens": True,
            "return_offsets_mapping": True,
            "padding": "max_length",
        }

    def _build_dataset(self):
        dataset = load_dataset(
            "squad_v2" if not self.v1 else "squad",
            cache_dir=Config.hugging_face_cache_dir,
        )
        self._dataset = dataset

    def _build_corpus(self) -> None:
        dataset = concatenate_datasets(
            (
                self.dataset["train"],
                self.dataset["validation"],
            )
        )
        self._transform_datasets_to_corpus(dataset)

    def prepare_train_features(self, examples):

        # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            examples["question" if self._pad_on_right else "context"],
            examples["context" if self._pad_on_right else "question"],
            **self._tokenizer_conf,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
                continue

            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != int(self._pad_on_right):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != int(self._pad_on_right):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (
                offsets[token_start_index][0] <= start_char
                and offsets[token_end_index][1] >= end_char
            ):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
                continue

            # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
            # Note: we could go after the last offset if the answer is the last word (edge case).
            while (
                token_start_index < len(offsets)
                and offsets[token_start_index][0] <= start_char
            ):
                token_start_index += 1
            tokenized_examples["start_positions"].append(token_start_index - 1)
            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    def prepare_validation_features(self, examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            examples["question" if self._pad_on_right else "context"],
            examples["context" if self._pad_on_right else "question"],
            **self._tokenizer_conf,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # We keep the example_id that gave us this feature and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = int(self._pad_on_right)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    def post_processing_function(
        self,
        examples,
        features,
        predictions,
        squad_v2=None,
        n_best_size=20,
        max_answer_length=30,
    ):
        if squad_v2 is None:
            squad_v2 = not self.v1

        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=squad_v2,
            n_best_size=n_best_size,
            max_answer_length=max_answer_length,
            null_score_diff_threshold=self.null_score_diff_threshold,
            output_dir=self.output_dir,
        )

        # Format the result to the format the metric expects.
        if squad_v2:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0}
                for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [
                {"id": k, "prediction_text": v} for k, v in predictions.items()
            ]
        references = [
            {"id": ex["id"], "answers": ex["answers"]}
            for ex in self._dataset["validation"]
        ]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    def compute_metrics(self, p: EvalPrediction):
        return self.metric.compute(predictions=p.predictions, references=p.label_ids)
