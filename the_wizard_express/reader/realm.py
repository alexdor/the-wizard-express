from typing import List

from torch import (
    bool,
    cat,
    masked_select,
    ones,
    split,
    tensor,
    unsqueeze,
    zeros,
)
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from ..config import Config
from . import Reader


class RealmReader(Reader):
    friendly_name = "realm"

    def _build(self) -> None:
        # self.tokenizer = DistilBertTokenizerFast.from_pretrained(
        #     "distilbert-base-uncased-distilled-squad"
        # )
        # self.model = DistilBertModel.from_pretrained(
        #     "distilbert-base-uncased-distilled-squad"
        # )
        # model = "bert-base-uncased"
        model = "distilbert-base-cased-distilled-squad"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            use_fast=True,
            cache_dir=Config.cache_dir,
        )
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            model,
            cache_dir=Config.cache_dir,
        )
        self.max_chunk_len = (
            self.model.config.max_position_embeddings
        )  # Maximum elements that each chunk can have

    def _load_from_file(self) -> None:
        # TODO
        return self._build()

    def answer(self, question: str, documents: List[str]) -> str:
        answers = (
            self._get_partial_answer(question, document) for document in documents
        )
        return self._get_partial_answer(question, ". ".join(answers))

    def _get_partial_answer(self, question: str, document: str):
        model_inputs = self.tokenizer(
            question, document, add_special_tokens=True, return_tensors="pt"
        )

        model_inputs = self._chunkify(model_inputs, question)

        answer: List[str] = []
        for chunk in model_inputs:
            output = self.model(**chunk)

            answer_start = output.start_logits.argmax()
            answer_end = output.end_logits.argmax() + 1

            partial_answer = self._convert_ids_to_string(
                chunk["input_ids"][0][answer_start:answer_end]
            )

            if partial_answer == "[CLS]":
                continue

            if partial_answer.startswith("##"):
                # # Drop whitespace from the previous word
                # answer = answer[:-1]
                # # Drop the ## which indicates a subword
                # partial_answer = partial_answer[2:]
                answer[-1] += partial_answer[2:]
                continue
            if not answer or answer[-1] != partial_answer:
                answer.append(partial_answer)

        # Drop whitespace from begining and end
        return " ".join(answer).strip()

    def _chunkify(self, inputs, question: str):
        """
        Break up a long article into chunks that fit within the max token
        requirement for that Transformer model.

        Calls to BERT / RoBERTa / ALBERT require the following format:
        [CLS] question tokens [SEP] context tokens [SEP].
        """

        # If the inputs fit into one chunk just return them
        if inputs["input_ids"].shape[1] < self.max_chunk_len:
            return [inputs]

        # create question mask
        tokenized_question_len = len(self.tokenizer(question)["input_ids"])
        question_mask = cat(
            (
                ones(tokenized_question_len, dtype=bool),
                zeros(
                    inputs["input_ids"].shape[1] - tokenized_question_len, dtype=bool
                ),
            )
        )

        question_len = masked_select(inputs["input_ids"], question_mask).shape[0]

        # the "-1" accounts for having to add an ending [SEP] token to the end
        chunk_size = self.max_chunk_len - question_len - 1

        # create a dict of dicts; each sub-dict mimics the structure of pre-chunked model input
        chunked_input = []
        for input_key, input_value in inputs.items():
            question_tensor = masked_select(input_value, question_mask)
            corpus_tensor = masked_select(input_value, ~question_mask)
            chunks = split(corpus_tensor, chunk_size)

            if not chunked_input:
                chunked_input = [{}] * len(chunks)
            for i, chunk in enumerate(chunks):

                thing = cat((question_tensor, chunk))
                if i != len(chunks) - 1:
                    if input_key == "input_ids":
                        # add the sep token at the end of the chunk
                        thing = cat((thing, tensor([self.tokenizer.sep_token_id])))
                    else:
                        # mark the sep token as part of the corpus and add some attention
                        thing = cat((thing, tensor([1])))

                chunked_input[i][input_key] = unsqueeze(thing, dim=0)
        return chunked_input

    def _convert_ids_to_string(self, input_ids) -> str:
        if len(input_ids) == 0:
            return ""
        return self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(input_ids)
        )
