from abc import ABC, abstractclassmethod

import os
import tensorflow as tf
import numpy as np
import re
import torch

from torch import nn
from tensorflow.python.saved_model import load_v1_in_v2
from transformers import BertPreTrainedModel, BertModel

from ..corpus.corpus import Corpus


class Retriever(ABC):
    """
    Abstract class for all the retrievers
    """

    __slots__ = ["corpus"]

    def __init__(self, corpus: Corpus) -> None:
        self.corpus = corpus

    @abstractclassmethod
    def retrieve_docs(self, question: str) -> str:
        pass


class TFIDFRetriever(Retriever):
    def retrieve_docs(self, question: str):
        pass


base_path = os.path.abspath("../../../small_ICT/")


# The following 3 classes are copied from HuggingFace codebase, as there are no exports
class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


# Main REALMRetriever model
class REALMRetriever(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=True)

        self.cls = BertOnlyMLMHead(config)

        # projected_emb = tf.layers.dense(output_layer, params["projection_size"])
        # projected_emb = tf.keras.layers.LayerNormalization(axis=-1)(projected_emb)
        # if is_training:
        #     projected_emb = tf.nn.dropout(projected_emb, rate=0.1)

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.projected_emb = nn.Dropout(0.1)

        # No need to init weights, as all are getting imported by REALM
        # self.init_weights()

    # TODO: Forward needs to be tinkered a bit with to use the additional layers
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        next_sentence_label=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        seq_relationship_scores = self.cls(pooled_output)

        next_sentence_loss = None
        if next_sentence_label is not None:
            loss_fct = nn.CrossEntropyLoss()
            next_sentence_loss = loss_fct(
                seq_relationship_scores.view(-1, 2), next_sentence_label.view(-1)
            )

        if not return_dict:
            output = (seq_relationship_scores,) + outputs[2:]
            return (
                ((next_sentence_loss,) + output)
                if next_sentence_loss is not None
                else output
            )

        return next_sentence_loss, seq_relationship_scores
        # return NextSentencePredictorOutput(
        #     loss=next_sentence_loss,
        #     logits=seq_relationship_scores,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )

    @staticmethod
    def load_tf_checkpoints(model, config, tf_checkpoint_path):
        tf_path = os.path.abspath(tf_checkpoint_path)
        print("Converting TensorFlow checkpoint from {}".format(tf_path))
        # Load weights from TF model
        init_vars = load_v1_in_v2.load(tf_checkpoint_path, tags=["train"])
        names = []
        arrays = []
        n_params = 0

        for tf_var in init_vars.variables:
            name = tf_var.name
            print("Loading TF weight {} with shape {}".format(name, tf_var.shape))
            n_params += np.prod(tf_var.shape)
            names.append(name)
            arrays.append(tf_var.numpy())

        for name, array in zip(names, arrays):
            name = re.sub(r"module\/|\:0", "", name).strip()
            name = name.split("/")

            # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
            # which are not required for using pretrained model
            if any(
                n
                in [
                    "adam_v",
                    "adam_m",
                    "AdamWeightDecayOptimizer",
                    "AdamWeightDecayOptimizer_1",
                    "global_step",
                ]
                for n in name
            ):
                print("Skipping {}".format("/".join(name)))
                continue
            pointer = model
            for m_name in name:
                if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                    scope_names = re.split(r"_(\d+)", m_name)
                else:
                    scope_names = [m_name]
                if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                    pointer = getattr(pointer, "weight")
                elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                    pointer = getattr(pointer, "bias")
                elif scope_names[0] == "output_weights":
                    pointer = getattr(pointer, "weight")
                elif scope_names[0] == "squad":
                    pointer = getattr(pointer, "classifier")
                else:
                    try:
                        pointer = getattr(pointer, scope_names[0])
                    except AttributeError:
                        print("Skipping {}".format("/".join(name)))
                        continue
                if len(scope_names) >= 2:
                    num = int(scope_names[1])
                    pointer = pointer[num]
            if m_name[-11:] == "_embeddings":
                pointer = getattr(pointer, "weight")
            elif m_name == "kernel":
                array = np.transpose(array)
            try:
                assert (
                    pointer.shape == array.shape
                ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
            except AssertionError as e:
                e.args += (pointer.shape, array.shape)
                raise
            print("Initialize PyTorch weight {}".format(name))
            pointer.data = torch.from_numpy(array)
        return model
