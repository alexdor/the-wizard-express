from abc import ABC
from functools import lru_cache
from json import dumps

from torch.cuda import is_available
from transformers import AutoTokenizer

from ..config import Config
from ..corpus import Corpus
from ..reader import BertOnBertReader, Reader, SimpleBertReader
from ..retriever import PyseriniSimple, Retriever, TFIDFRetriever
from ..tokenizer import Tokenizer, WordTokenizer


class QAModel(ABC):
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
        gpu=None,
    ) -> None:
        self.retriever = retriever(
            corpus=corpus,
            tokenizer=retriever_tokenizer(corpus)
            if retriever_tokenizer is not None
            else None,
        )

        self.reader = reader(
            tokenizer=reader_tokenizer, device=f"cuda:{gpu}" if gpu else "cpu"
        )

    @lru_cache(128)
    def answer_question(self, question: str) -> str:
        docs = self.retriever.retrieve_docs(question, self.docs_to_retrieve)
        return self.reader.answer(question=question, documents=docs)

    # def answer_questions(self, questions: List[str]) -> List[str]:
    #     docs = self.retriever.retrieve_docs(question, self.docs_to_retrieve)
    #     return self.reader.answer(question=question, documents=docs


class TFIDFBertOnBert(QAModel):
    friendly_name = "tfidf-bert-on-bert-model"

    def __init__(self, corpus, gpu) -> None:
        args = {
            "retriever": TFIDFRetriever,
            "retriever_tokenizer": WordTokenizer,
            "reader": BertOnBertReader,
            "reader_tokenizer": AutoTokenizer.from_pretrained(
                BertOnBertReader.model_name,
                use_fast=True,
                cache_dir=Config.hugging_face_cache_dir,
            ),
            "corpus": corpus,
            "gpu": gpu,
        }
        super().__init__(**args)


class PyseriniBertOnBert(QAModel):
    friendly_name = "pyserini-bert-on-bert-model"

    def __init__(self, corpus, gpu) -> None:
        args = {
            "retriever": PyseriniSimple,
            "retriever_tokenizer": None,
            "reader": BertOnBertReader,
            "reader_tokenizer": AutoTokenizer.from_pretrained(
                BertOnBertReader.model_name,
                use_fast=True,
                cache_dir=Config.hugging_face_cache_dir,
            ),
            "corpus": corpus,
            "gpu": gpu,
        }
        super().__init__(**args)


class TFIDFBertSimple(QAModel):
    friendly_name = "tfidf-bert-simple-model"

    def __init__(self, corpus, gpu) -> None:
        args = {
            "retriever": TFIDFRetriever,
            "retriever_tokenizer": WordTokenizer,
            "reader": SimpleBertReader,
            "reader_tokenizer": AutoTokenizer.from_pretrained(
                SimpleBertReader.model_name,
                use_fast=True,
                cache_dir=Config.hugging_face_cache_dir,
            ),
            "corpus": corpus,
            "gpu": gpu,
        }
        super().__init__(**args)
