from abc import ABC, abstractclassmethod

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
