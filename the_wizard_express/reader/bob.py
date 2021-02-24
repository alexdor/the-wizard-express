from functools import lru_cache
from typing import Iterator, List

from torch import bool as torch_bool
from torch import equal, split, tensor
from transformers import AutoModelForQuestionAnswering

from ..config import Config
from .reader import Reader


class BertOnBertReader(Reader):
    friendly_name = "bob"
    model_name = "distilbert-base-cased-distilled-squad"

    def _build(self) -> None:
        self.model = (
            AutoModelForQuestionAnswering.from_pretrained(
                self.model_name,
                cache_dir=Config.hugging_face_cache_dir,
            )
            .to(self.device)
            .eval()
        )

        self.max_chunk_len = (
            self.model.config.max_position_embeddings
        )  # Maximum elements that each chunk can have

    def _load_from_file(self) -> None:
        # TODO
        return self._build()

    def answer(self, question: str, documents: Iterator[str]) -> str:
        answers = [
            self._get_partial_answer(question, document) for document in documents
        ]
        if len(answers) == 1:
            return answers[0]
        return self._get_partial_answer(question, ". ".join(answers))

    def _get_partial_answer(self, question: str, document: str):
        model_inputs = self.tokenizer(
            question,
            document,
            add_special_tokens=True,
            return_tensors="pt",
            return_attention_mask=True,
            max_length=Config.max_seq_len,
            stride=Config.doc_stride,
            return_overflowing_tokens=True,
            truncation="only_second",
            padding=True,
        ).to(self.device)
        model_inputs.pop("overflow_to_sample_mapping", None)
        cls_token_id = tensor(self.tokenizer.cls_token_id, device=self.device)
        for input_ids, attention_mask in zip(
            split(model_inputs["input_ids"], 12),
            split(model_inputs["attention_mask"], 12),
        ):
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            # chunk
            answer_start = output.start_logits.argmax(dim=-1)
            answer_end = output.end_logits.argmax(dim=-1) + 1
            answer: List[str] = []

            for i in range(len(output.start_logits)):
                partial_answer = input_ids[i][answer_start[i] : answer_end[i]]
                if len(partial_answer) == 0 or (
                    len(partial_answer) == 1 and equal(partial_answer[0], cls_token_id)
                ):
                    continue

                partial_answer = self.tokenizer.convert_tokens_to_string(
                    self.tokenizer.convert_ids_to_tokens(
                        partial_answer, skip_special_tokens=True
                    )
                )
                if partial_answer[0][:2] == "##":
                    # # Drop whitespace from the previous word
                    # answer = answer[:-1]
                    # # Drop the ## which indicates a subword
                    # partial_answer = partial_answer[2:]
                    if not answer:
                        continue
                    answer[-1] += partial_answer[2:]
                if not answer or answer[-1] != partial_answer:
                    answer += partial_answer
        # Drop whitespace from beginning and end
        return " ".join(self.tokenizer.convert_tokens_to_string(answer)).strip()


class SimpleBertReader(BertOnBertReader):
    friendly_name = "bert"

    def answer(self, question: str, documents: Iterator[str]) -> str:
        return self._get_partial_answer(question, "\n".join(documents))
