from transformers import AutoTokenizer

from ..config import Config
from ..corpus import Corpus
from . import Tokenizer


class DistilledBertTokenizer(Tokenizer):
    __slots__ = ("tokenizer", "_tokenizer_path")
    friendly_name = "distil_bert_tokenizer"

    def _build(self, corpus: Corpus, path_to_save: str) -> None:
        model = "distilbert-base-cased-distilled-squad"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            use_fast=True,
            cache_dir=Config.hugging_face_cache_dir,
        )

        self.tokenizer.prepare_for_tokenization

    def _load_from_file(self, file: str) -> None:
        return self._build()
