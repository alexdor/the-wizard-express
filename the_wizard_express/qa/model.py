from abc import ABC, abstractclassmethod

from transformers import AutoTokenizer

from ..config import Config
from ..corpus import Corpus
from ..reader import BertOnBertReader, Reader, SimpleBertReader
from ..retriever import Retriever, TFIDFRetriever
from ..tokenizer import Tokenizer, WordTokenizer


class Model(ABC):
    """
    Abstract class for all the final model
    """

    docs_to_retrieve = 5

    def __init__(
        self,
        reader: Reader,
        reader_tokenizer: Tokenizer,
        corpus: Corpus,
        retriever: Retriever,
        retriever_tokenizer: Tokenizer,
    ) -> None:
        self.retriever = retriever(corpus=corpus, tokenizer=retriever_tokenizer(corpus))
        self.reader = reader(tokenizer=reader_tokenizer)

    def answer_question(self, question: str) -> str:
        docs = self.retriever.retrieve_docs(question, self.docs_to_retrieve)
        return self.reader.answer(question=question, documents=docs)


class TFIDFBertOnBert(Model):
    friendly_name = "tfidf-bert-on-bert-model"

    def __init__(self, corpus) -> None:
        args = {
            "retriever": TFIDFRetriever,
            "retriever_tokenizer": WordTokenizer,
            "reader": BertOnBertReader,
            "reader_tokenizer": AutoTokenizer.from_pretrained(
                BertOnBertReader.model_name,
                use_fast=True,
                cache_dir=Config.cache_dir,
            ),
            "corpus": corpus,
        }
        super().__init__(**args)


class TFIDFBertSimple(Model):
    friendly_name = "tfidf-bert-simple-model"

    def __init__(self, corpus) -> None:
        args = {
            "retriever": TFIDFRetriever,
            "retriever_tokenizer": WordTokenizer,
            "reader": SimpleBertReader,
            "reader_tokenizer": AutoTokenizer.from_pretrained(
                SimpleBertReader.model_name,
                use_fast=True,
                cache_dir=Config.cache_dir,
            ),
            "corpus": corpus,
        }
        super().__init__(**args)
