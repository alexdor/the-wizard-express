from abc import ABC, abstractclassmethod
from os.path import lexists
from typing import Dict, List, Optional, Union

from the_wizard_express.utils import generate_cache_path
from tokenizers import Encoding
from tokenizers import Tokenizer as HuggingFaceTokenizer

from ..corpus.corpus import Corpus


class Tokenizer(ABC):
    def get_id(self) -> str:
        return self.__class__.__name__

    def __init__(self, corpus: Corpus) -> None:
        self.tokenizer_path = generate_cache_path("tokenizer", corpus, self)

        if lexists(self.tokenizer_path):
            self._load_from_file(self.tokenizer_path)
            return
        self._build(corpus, self.tokenizer_path)

    @abstractclassmethod
    def _build(self, corpus: Corpus, path_to_save: str) -> None:
        """
        A method that creates a tokenizer from a given corpus
        """

    def _load_from_file(self, file: str) -> None:
        """
        Method to load an existing tokenizer from disk
        """
        self.tokenizer = HuggingFaceTokenizer.from_file(file)

    def encode(
        self,
        sentances: str,
        text_pair: Optional[Union[str, List[str], List[int]]] = None,
    ) -> Encoding:
        return self.tokenizer.encode(sentances, text_pair)

    def encode_batch(self, sentances: List[str]) -> List[Encoding]:
        return self.tokenizer.encode_batch(sentances)

    def tokens_to_sentance(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    def decode_batch(self, tokens: List[List[int]]) -> str:
        return self.tokenizer.decode_batch(tokens)

    def get_vocab(self) -> Dict[str, int]:
        return self.tokenizer.get_vocab()
