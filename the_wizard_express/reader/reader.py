from abc import ABC
from transformers import BertForQuestionAnswering

import torch


def getModel(self, pretrainedModelName):

    return BertForQuestionAnswering.from_pretrained(pretrainedModelName)


# Add a bart model? Should this be the QA model or should we implement that ourselves?

# model = BartForQuestionAnswering.from_pretrained('facebook/bart-large', return_dict=True)

#
