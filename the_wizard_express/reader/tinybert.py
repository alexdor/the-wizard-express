from collections import OrderedDict
from typing import Iterator, List

from torch import argmax, cat, masked_select, split, tensor, unsqueeze
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from ..config import Config
from . import Reader


class TinyBertReader(Reader):
    friendly_name = "tiny-bert"
    model_name = "bert-large-uncased"

    def _build(self) -> None:
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            self.model_to_use,
            cache_dir=Config.hugging_face_cache_dir,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_to_use,
            use_fast=True,
            cache_dir=Config.hugging_face_cache_dir,
        )
        self.max_len = self.model.config.max_position_embeddings
        self.chunked = False

    def _tokenize(self, question, text):
        self.inputs = self.tokenizer.encode_plus(
            question, text, add_special_tokens=True, return_tensors="pt"
        )
        self.input_ids = self.inputs["input_ids"].tolist()[0]

        if len(self.input_ids) > self.max_len:
            self.inputs = self._chunkify()
            self.chunked = True

    def _chunkify(self):
        """
        Break up a long article into chunks that fit within the max token
        requirement for that Transformer model.

        Calls to BERT / RoBERTa / ALBERT require the following format:
        [CLS] question tokens [SEP] context tokens [SEP].
        """

        # create question mask based on token_type_ids
        # value is 0 for question tokens, 1 for context tokens
        qmask = self.inputs["token_type_ids"].lt(1)
        qt = masked_select(self.inputs["input_ids"], qmask)
        chunk_size = self.max_len - qt.size()[0] - 1  # the "-1" accounts for
        # having to add an ending [SEP] token to the end

        # create a dict of dicts; each sub-dict mimics the structure of pre-chunked model input
        chunked_input = OrderedDict()
        for k, v in self.inputs.items():
            q = masked_select(v, qmask)
            c = masked_select(v, ~qmask)
            chunks = split(c, chunk_size)

            for i, chunk in enumerate(chunks):
                if i not in chunked_input:
                    chunked_input[i] = {}

                thing = cat((q, chunk))
                if i != len(chunks) - 1:
                    if k == "input_ids":
                        thing = cat((thing, tensor([102])))
                    else:
                        thing = cat((thing, tensor([1])))

                chunked_input[i][k] = unsqueeze(thing, dim=0)
        return chunked_input

    def _get_answer(self):
        if self.chunked:
            answer = ""
            for k, chunk in self.inputs.items():
                res = self.model(**chunk)
                answer_start_scores, answer_end_scores = res
                answer_start_scores, answer_end_scores = (
                    res[answer_start_scores],
                    res[answer_end_scores],
                )

                answer_start = argmax(answer_start_scores)
                answer_end = argmax(answer_end_scores) + 1

                ans = self._convert_ids_to_string(
                    chunk["input_ids"][0][answer_start:answer_end]
                )
                if ans != "[CLS]":
                    answer += ans + " / "
            return answer

        res = self.model(**self.inputs)
        answer_start_scores, answer_end_scores = res
        answer_start_scores, answer_end_scores = (
            res[answer_start_scores],
            res[answer_end_scores],
        )

        answer_start = argmax(
            answer_start_scores
        )  # get the most likely beginning of answer with the argmax of the score
        answer_end = (
            argmax(answer_end_scores) + 1
        )  # get the most likely end of answer with the argmax of the score

        return self._convert_ids_to_string(
            self.inputs["input_ids"][0][answer_start:answer_end]
        )

    def _convert_ids_to_string(self, input_ids):
        return self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(input_ids)
        )

    def answer(self, question: str, documents: Iterator[str]) -> str:
        documents = tuple(documents)
        self._tokenize(question, documents[0])
        return self._get_answer()

    def _load_from_file(self) -> None:
        # Hugging face caches the model for us so we just call build
        return self._build()
