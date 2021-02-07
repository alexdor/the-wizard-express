from abc import ABC, abstractclassmethod
from os.path import lexists
from typing import Dict, List, Optional, Union

from tokenizers import Encoding
from tokenizers import Tokenizer as HuggingFaceTokenizer

from ..corpus.corpus import Corpus
from ..utils import generate_cache_path


class Tokenizer(ABC):
    def get_id(self) -> str:
        return self.__class__.__name__

    def __init__(self, corpus: Corpus) -> None:
        self._tokenizer_path = generate_cache_path(
            "tokenizer", corpus, self, file_ending=".json"
        )

        if lexists(self._tokenizer_path):
            self._load_from_file(self._tokenizer_path)
            return
        print(f"Buidling {self.friendly_name} tokenizer")
        self._build(corpus, self._tokenizer_path)
        self._save_tokenizer()

    @abstractclassmethod
    def _build(self, corpus: Corpus, path_to_save: str) -> None:
        """
        A method that creates a tokenizer from a given corpus
        """

    def _save_tokenizer(self) -> None:
        self.tokenizer.save(self._tokenizer_path)

    def _load_from_file(self, file: str) -> None:
        """
        Method to load an existing tokenizer from disk
        """
        self.tokenizer = HuggingFaceTokenizer.from_file(file)

    def encode(
        self,
        sentences: str,
        text_pair: Optional[Union[str, List[str], List[int]]] = None,
    ) -> Encoding:
        return self.tokenizer.encode(sentences, text_pair)

    def encode_batch(self, sentences: List[str]) -> List[Encoding]:
        return self.tokenizer.encode_batch(sentences)

    def tokens_to_sentence(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    def decode_batch(self, tokens: List[List[int]]) -> str:
        return self.tokenizer.decode_batch(tokens)

    @property
    def vocab(self) -> Dict[str, int]:
        return self.tokenizer.get_vocab()

    def __call__(self, text, text_pair, **kwargs):
        return self.tokenizer(text, text_pair, **kwargs)
