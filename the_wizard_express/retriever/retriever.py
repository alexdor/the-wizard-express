from abc import ABC, abstractclassmethod
from os.path import lexists
from typing import Tuple

from ..corpus import Corpus
from ..tokenizer import Tokenizer
from ..utils import generate_cache_path


class Retriever(ABC):
    """
    Abstract class for all the retrievers
    """

    def __init__(self, corpus: Corpus, tokenizer: Tokenizer) -> None:
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.retriever_path = generate_cache_path("retriever", corpus, tokenizer, self)

        if lexists(self.retriever_path):
            self._load_from_file()
            return
        print(f"Buidling {self.friendly_name} retriever")
        self._build()

    @abstractclassmethod
    def retrieve_docs(self, question: str, number_of_docs: int) -> Tuple[str]:
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
        This method is called in order to do any preprocessing needed
        before using the retriever
        """

    def get_id(self) -> str:
        """
        Get the retriever id
        """
        return self.__class__.__name__
