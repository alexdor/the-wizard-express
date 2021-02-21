from transformers import AutoTokenizer, DistilBertForQuestionAnswering

from ..config import Config

# import re
# import string

# exclude = set(string.punctuation)
# regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)


class DistilBertForQA:
    model_name = "distilbert-base-cased"
    friendly_name = "wizard-bert-uncased-squadv2"

    def __init__(self, load_from_local=False, version=2) -> None:
        self.friendly_name = f"wizard-bert-uncased-squadv{version}"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True,
            cache_dir=Config.hugging_face_cache_dir,
        )
        if load_from_local:
            self.model = DistilBertForQuestionAnswering.from_pretrained(
                f"./{self.friendly_name}",
                cache_dir=Config.hugging_face_cache_dir,
            )

        self.model = DistilBertForQuestionAnswering.from_pretrained(
            self.model_name, cache_dir=Config.hugging_face_cache_dir
        )
