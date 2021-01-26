from collections import namedtuple
from logging import debug
from typing import List

from tensorflow import reshape, shape
from tensorflow.compat import dimension_value
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
from .reader import Reader


def get_shape(tensor, dim=None):
    """Gets the most specific dimension size(s) of the given tensor.

    This is a wrapper around the two ways to get the shape of a tensor: (1)
    t.get_shape() to get the static shape, and (2) tf.shape(t) to get the dynamic
    shape. This function returns the most specific available size for each
    dimension. If the static size is available, that is returned. Otherwise, the
    tensor representing the dynamic size is returned.

    Args:
      tensor: Input tensor.
      dim: Desired dimension. Use None to retrieve the list of all sizes.

    Returns:
      output = Most specific static dimension size(s).
    """
    static_shape = tensor.get_shape()
    dynamic_shape = shape(tensor)
    if dim is not None:
        return dimension_value(static_shape[dim]) or dynamic_shape[dim]

    return [dimension_value(d) or dynamic_shape[i] for i, d in enumerate(static_shape)]


BertInputs = namedtuple("BertInputs", ["token_ids", "mask", "segment_ids"])


def tensor_flatten(t):
    """Collapse all dimensions of a tensor but the last.

    This function also returns a function to unflatten any tensors to recover
    the original shape (aside from the last dimension). This is useful for
    interfacing with functions that are agnostic to all dimensions but the last.
    For example, if we want to apply a linear projection to a batched sequence
    of embeddings:

      t = tf.random_uniform([batch_size, sequence_length, embedding_size])
      w = tf.get_variable("w", [embedding_size, projection_size])
      flat_t, unflatten = flatten(t)
      flat_projected = tf.matmul(flat_t, w)
      projected = unflatten(flat_projected)

    Args:
      t: [dim_1, ..., dim_(n-1), dim_n]

    Returns:
      output: [dim_1 * ... * dim_(n-1), dim_n]
      _unflatten: A function that when called with a flattened tensor returns the
          unflattened version, i.e. reshapes any tensor with shape
          [dim_1 * ... * dim_(n-1), dim_new] to [dim_1, ..., dim_(n-1), dim_new].
    """
    input_shape = get_shape(t)
    if len(input_shape) > 2:
        t = reshape(t, [-1, input_shape[-1]])

    def _unflatten(flat_t):
        if len(input_shape) > 2:
            return reshape(flat_t, input_shape[:-1] + [get_shape(flat_t, -1)])
        return flat_t

    return t, _unflatten


def flatten_bert_inputs(bert_inputs):
    """Flatten all tensors in a BertInput and also return the inverse."""
    flat_token_ids, unflatten = tensor_flatten(bert_inputs.token_ids)
    flat_mask, _ = tensor_flatten(bert_inputs.mask)
    flat_segment_ids, _ = tensor_flatten(bert_inputs.segment_ids)
    flat_bert_inputs = BertInputs(
        token_ids=flat_token_ids, mask=flat_mask, segment_ids=flat_segment_ids
    )
    return flat_bert_inputs, unflatten


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
        document = "\n".join(documents)
        model_inputs = self.tokenizer(
            question, document, add_special_tokens=True, return_tensors="pt"
        )

        model_inputs = self._chunkify(model_inputs, question)

        answer = ""
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
                # Drop whitespace from the previous word
                answer = answer[:-1]
                # Drop the ## which indicates a subword
                partial_answer = partial_answer[2:]

            answer += (
                f"{partial_answer} " if not Config.debug else f"{partial_answer} / "
            )

        # Drop whitespace from begining and end
        return answer.strip()

        # inputs["to"]

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

    # def _read(self, joint_inputs):
    #     # [batch_size * num_candidates, query_seq_len + candidate_seq_len]
    #     flat_joint_inputs, _ = flatten_bert_inputs(joint_inputs)
    #     # [batch_size * num_candidates, num_masks]
    #     flat_mlm_positions, _ = tensor_flatten(
    #         tf.tile(tf.expand_dims(mlm_positions, 1), [1, params["num_candidates"], 1])
    #     )

    #     batch_size, num_masks = tensor_utils.shape(mlm_targets)

    #     # [batch_size * num_candidates, query_seq_len + candidates_seq_len]
    #     flat_joint_bert_outputs = bert_module(
    #         inputs=dict(
    #             input_ids=flat_joint_inputs.token_ids,
    #             input_mask=flat_joint_inputs.mask,
    #             segment_ids=flat_joint_inputs.segment_ids,
    #             mlm_positions=flat_mlm_positions,
    #         ),
    #         signature="mlm",
    #         as_dict=True,
    #     )

    #     # [batch_size, num_candidates]
    #     candidate_score = retrieval_score

    #     # [batch_size, num_candidates]
    #     candidate_log_probs = tf.math.log_softmax(candidate_score)
