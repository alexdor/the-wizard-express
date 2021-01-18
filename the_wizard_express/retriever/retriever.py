from abc import ABC, abstractclassmethod
from collections import Counter
from math import log10
from multiprocessing import Pool
from os.path import lexists
from pickle import load
from typing import Counter as CounterType
from typing import Dict, Iterator, Tuple

from numpy import argpartition, empty, ndarray, zeros
from numpy.core.shape_base import hstack
from scipy.sparse import csr_matrix
from the_wizard_express.config import Config
from the_wizard_express.tokenizer.tokenizer import Tokenizer
from the_wizard_express.utils import (
    generate_cache_path,
    pickle_and_save_to_file,
)
from tokenizers import Encoding
from tqdm import tqdm

from ..corpus.corpus import Corpus


class Retriever(ABC):
    """
    Abstract class for all the retrievers
    """

    def __init__(self, corpus: Corpus, tokenizer: Tokenizer) -> None:
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.retriever_path = generate_cache_path(
            "retriever", corpus, tokenizer, self, file_ending=".pickle"
        )

        if lexists(self.retriever_path):
            self._load_from_file()
            return
        self._build()

    @abstractclassmethod
    def retrieve_docs(self, question: str, number_of_docs: int) -> Tuple[str]:
        """
        Main method for the retriever, it's used to get the relavant documents
        for a given question
        """
        pass

    @abstractclassmethod
    def _load_from_file(self) -> None:
        """
        A method that loads an already existin retriever from a file
        """
        pass

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


class TFIDFRetriever(Retriever):
    __slots__ = ("tokenizer", "corpus", "tf_idf", "retriever_path")

    def retrieve_docs(self, question: str, number_of_docs: int) -> Tuple[str]:
        encoded_question = self.tokenizer.encode(question)

        # Drop extra tokens (sep, uknown etc)
        encoded_question = [id for id in encoded_question.ids if id < Config.vocab_size]

        # Get the tf-idf for the encoded tokens and sum it per row
        sum_tf_idf = self.tf_idf[:, encoded_question].sum(axis=1)

        # get the documents with the top sum
        doc_indexes = argpartition(sum_tf_idf, -number_of_docs, axis=0)[
            -number_of_docs:
        ]

        # return the corpuses with the proper index
        return self.corpus.get_docs_by_index(doc_indexes.flatten().tolist()[0])

    def _build(self) -> None:
        vocab = self.tokenizer.get_vocab()

        encoded_corpus = tuple(self.tokenizer.encode_batch(self.corpus.corpus))
        encoded_special_tokens = {vocab[token] for token in Config.special_tokens_list}
        vocab_values = tuple(
            value for value in vocab.values() if value not in encoded_special_tokens
        )
        corpus_length = len(encoded_corpus)
        tf = empty((len(vocab_values), 0))
        chunksize = 300
        idf_counter: CounterType[int] = Counter()

        corpus_iterator = self._get_iterator_for_corpus(
            encoded_corpus, vocab_values, chunksize=chunksize
        )

        # compute tf and df
        with Pool(Config.max_proc_to_use) as pool:
            with tqdm(total=corpus_length) as pbar:
                for (tf_res, idf_res) in pool.imap(
                    func=TFIDFRetriever._create_tf_idf,
                    iterable=corpus_iterator,
                ):
                    pbar.update(chunksize)
                    idf_counter += idf_res
                    tf = hstack((tf, tf_res))

        tf_idf = empty((corpus_length, len(vocab_values)))
        idf = {}
        # compute tf_idf
        for word_id, value in tqdm(enumerate(idf_counter), total=len(idf_counter)):
            idf[word_id] = log10(corpus_length / float(value) + 1) if value != 0 else 0
            for corpus_index in range(len(tf[word_id])):
                tf_idf[corpus_index, word_id] = (
                    tf[word_id, corpus_index] * idf_counter[word_id]
                )

        tf_idf = csr_matrix(tf_idf)
        pickle_and_save_to_file(tf_idf, self.retriever_path)
        self.tf_idf = tf_idf

    def _load_from_file(self) -> None:
        self.tf_idf = load(open(self.retriever_path, "rb"))

    @staticmethod
    def _create_tf_idf(
        params: Tuple[Tuple[Encoding, ...], Tuple[int]]
    ) -> Tuple[ndarray, CounterType[int]]:
        (encoded_corpus, vocab_values) = params
        idf: Dict[int, int] = {}
        tf = zeros((len(vocab_values), len(encoded_corpus)))
        for corpus_index, doc in enumerate(encoded_corpus):
            doc_len = len(doc)
            counter: CounterType[int] = Counter(doc.ids)
            for word_id in vocab_values:
                if word_id in doc.ids:
                    idf[word_id] = idf.get(word_id, 0) + 1
                tf[word_id, corpus_index] = counter[word_id] / float(doc_len)

        return (tf, Counter(idf))

    def _get_iterator_for_corpus(
        self, encoded_corpus: Tuple[Encoding, ...], vocab_values, chunksize=100
    ) -> Iterator[Tuple[Tuple[Encoding, ...], Tuple[int]]]:
        corpus_index = 0
        for corpus_index in range(0, len(encoded_corpus), chunksize):
            yield (
                encoded_corpus[corpus_index : corpus_index + chunksize],
                vocab_values,
            )
