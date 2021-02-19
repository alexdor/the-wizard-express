from abc import ABC

from torch import bool as torch_bool
from torch import (
    cat,
    empty,
    equal,
    int32,
    masked_select,
    nn,
    nonzero,
    ones,
    split,
    tensor,
    zeros,
)
from torch.nn.functional import cross_entropy
from transformers import DistilBertForQuestionAnswering


class LanguageModel(ABC):
    """
    Abstract class for all the language models
    """


# class TFIDFBert(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         return F.relu(self.conv2(x))


class StackedBert(nn.Module):
    model = "distilbert-base-cased-distilled-squad"

    def __init__(self, tokenizer):
        super().__init__()
        self.bert = DistilBertForQuestionAnswering.from_pretrained(self.model)
        self.final_bert = DistilBertForQuestionAnswering.from_pretrained(self.model)
        self.max_chunk_len = self.bert.config.max_position_embeddings
        self.tokenizer = tokenizer
        self.sep_token_id = tokenizer.sep_token_id
        self.loss = nn.CrossEntropyLoss()

    #    self.out = nn.Linear(self.bert.config.hidden_size, classes)
    def _get_answer_from_bert(self, inputs, output, device, cls_tensor):
        answer_start = output.start_logits.argmax()
        answer_end = output.end_logits.argmax() + 1
        answer = inputs["input_ids"][0][answer_start:answer_end]
        if not len(answer) or equal(answer, cls_tensor):
            return empty(0, device=device)

        return answer[1:] if answer[0] == self.tokenizer.cls_token_id else answer

    def forward(self, **inputs):
        inputs["labels"] = inputs["labels"].squeeze()
        device = inputs["input_ids"].device
        answers = empty(0, dtype=int32, device=device)
        question_len = nonzero(inputs["input_ids"] == self.sep_token_id)[0][-1]
        separator_tensor = tensor([self.sep_token_id], device=device)
        cls_tensor = tensor([self.tokenizer.cls_token_id], device=device)
        for chunk in self._chunkify(inputs, question_len):
            answer = self._get_answer_from_bert(
                chunk, self.bert(**chunk), device, cls_tensor
            )

            if len(answer) and (not len(answers) or not equal(answers[-1], answer)):
                answers = cat((answers, answer, separator_tensor)).int()
        # final_question = cat(
        #     (
        #         inputs["input_ids"][0][0][:question_len],
        #         answers,
        #     )
        # ).int()
        # final_question = {
        #     "input_ids": final_question,
        #     "attention_mask": ones(
        #         len(final_question), device=device, dtype=torch_bool
        #     ),
        # }
        final_question = self.tokenizer(
            self.tokenizer.decode(inputs["input_ids"][0][0][:question_len]),
            self.tokenizer.decode(answers),
            add_special_tokens=True,
            return_tensors="pt",
        ).to(device)
        answer = self._get_answer_from_bert(
            final_question, self.bert(**final_question), device, cls_tensor
        )
        #
        return (
            cross_entropy(
                cat(
                    (answer, zeros(len(inputs["labels"]) - len(answer), device=device))
                ).unsqueeze(dim=1),
                inputs["labels"],
            ),
            answer,
        )
        # _, output = self.bert(input_ids, attention_mask=attention_mask)
        # out = self.out(output)
        # return answers

    def _chunkify(self, inputs, tokenized_question_len):
        """
        Break up a long article into chunks that fit within the max token
        requirement for that Transformer model.

        Calls to BERT / RoBERTa / ALBERT require the following format:
        [CLS] question tokens [SEP] context tokens [SEP].
        """

        # If the inputs fit into one chunk just return them
        if inputs["input_ids"].shape[-1] < self.max_chunk_len:
            return [inputs]

        device = inputs["input_ids"].device

        # create question mask

        # inputs["input_ids"].split(
        #     [tokenized_question_len, inputs["input_ids"].shape[-1] - tokenized_question_len], dim=2
        # )
        question_mask = cat(
            (
                ones(tokenized_question_len, dtype=torch_bool, device=device),
                zeros(
                    inputs["input_ids"].shape[-1] - tokenized_question_len,
                    dtype=torch_bool,
                    device=device,
                ),
            ),
        )
        question_mask_reverse = ~question_mask

        # question_len = masked_select(inputs["input_ids"], question_mask).shape[0]

        # the "-1" accounts for having to add an ending [SEP] token to the end
        chunk_size = self.max_chunk_len - tokenized_question_len - 1
        tensor_of_one, tensor_sep_token = ones(1, device=device), tensor(
            [self.sep_token_id], device=device
        )
        # create a dict of dicts; each sub-dict mimics the structure of pre-chunked model input
        chunked_input = []
        for input_key, input_value in inputs.items():
            if input_key == "labels":
                continue
            question_tensor = masked_select(input_value, question_mask)
            corpus_tensor = masked_select(input_value, question_mask_reverse)
            chunks = split(corpus_tensor, chunk_size)

            if not chunked_input:
                chunked_input = [{} for _ in range(len(chunks))]

            for i, chunk in enumerate(chunks):
                if not nonzero(chunk).shape[0]:
                    continue
                thing = cat((question_tensor, chunk))
                if i != len(chunks) - 1:
                    if input_key == "input_ids":
                        # add the sep token at the end of the chunk
                        thing = cat((thing, tensor_sep_token))
                    else:
                        # mark the sep token as part of the corpus and add some attention
                        thing = cat((thing, tensor_of_one))
                chunked_input[i][input_key] = thing.unsqueeze(dim=0)
        return [chunk for chunk in chunked_input if chunk]

    # def _
