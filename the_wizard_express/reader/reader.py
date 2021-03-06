from abc import ABC, abstractclassmethod
from os.path import lexists
from typing import Iterator

from ..config import Config
from ..tokenizer import Tokenizer
from ..utils import generate_cache_path


class Reader(ABC):
    """
    Abstract class for all the readers
    """

    def __init__(self, tokenizer: Tokenizer, device="cpu") -> None:
        self.device = device
        self.tokenizer = tokenizer
        self._reader_path = generate_cache_path(
            "reader",
            tokenizer,
            self,
            skip_vocab_size=True,
        )

        if lexists(self._reader_path):
            self._load_from_file()
            return
        if Config.debug:
            print(f"Building {self.friendly_name} reader")
        self._build()

    @abstractclassmethod
    def answer(self, question: str, documents: Iterator[str]) -> str:
        """
        Main method for the retriever, it's used to get the relavant documents
        for a given question
        """

    @abstractclassmethod
    def _load_from_file(self) -> None:
        """
        A method that loads an already existin retriever from a file
        """

    @abstractclassmethod
    def _build(self) -> None:
        """
        This method is called in order to do any preproccesing needed
        before using the retriever
        """

    def get_id(self) -> str:
        """
        Get the retriever id
        """
        return self.__class__.__name__
